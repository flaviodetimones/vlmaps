"""
interactive_object_nav.py
=========================
Type a natural-language instruction and watch the robot navigate in real time.

Usage (from the repository root)
---------------------------------
    python application/interactive_object_nav.py scene_id=1

Controls
--------
    - A new OpenCV window shows the robot's camera view after every action.
    - A second window shows the semantic top-down map with the target heatmap
      and the robot's current position.
    - Press any key in the window to advance to the next step.
    - Type a new instruction at the prompt to run another navigation.
    - Type 'quit' or 'exit' to stop.
"""

import cv2
import hydra
import json
import numpy as np
import os
from dataclasses import dataclass
from collections import deque
from omegaconf import DictConfig
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from typing import List, Optional, Tuple
from vlmaps.utils.search_state import SearchState

# Milliseconds to wait after each discrete sim step for demo playback.
# Keep this low and throttle heavy redraws separately so the UI stays fluid.
_NAV_STEP_DELAY_MS: int = 1
_MAP_REFRESH_STRIDE: int = 2

from vlmaps.robot.habitat_lang_robot import HabitatLanguageRobot
from vlmaps.utils.llm_utils import (
    OpenVocabTargetResolution,
    parse_object_goal_instruction,
    resolve_open_vocab_target,
)
from vlmaps.utils.mapping_utils import cvt_pose_vec2tf
from vlmaps.utils.matterport3d_categories import mp3dcat, get_categories
from vlmaps.utils.room_map_utils import find_room_goal, load_room_map
from vlmaps.utils.room_provider import room_command_matches
from vlmaps.utils.visualize_utils import pool_3d_label_to_2d, pool_3d_rgb_to_2d


# ── Visualization helpers ─────────────────────────────────────────────────────

# Windows the user has closed — never re-open them
_closed_windows: set = set()
_shown_windows: set = set()   # windows that have been successfully shown at least once
_frozen_detection_bgr = None  # frozen YOLOE frame shown until next search
_frozen_target_cell = None    # last confirmed map target marker

_HEATMAP_MODE_ENV = "VLMAPS_HEATMAP_MODE"
_HEADLESS_EVAL_ENV = "VLMAPS_EVAL_HEADLESS"

_SURROGATE_PITCH_DEG = {
    "counter": [-30.0, -20.0],
    "table": [-25.0, -15.0],
    "desk": [-25.0, -15.0],
    "shelf": [-20.0, -10.0],
    "sofa": [-10.0, 0.0],
    "bed": [-20.0, -10.0],
    "floor": [-35.0, -20.0],
}
_DEFAULT_PITCH_DEG = [0.0]


def is_eval_headless() -> bool:
    return str(os.environ.get(_HEADLESS_EVAL_ENV, "")).strip().lower() in {
        "1", "true", "yes", "on",
    }


def ui_wait(delay_ms: int) -> int:
    """UI wait that becomes a no-op during headless evaluation."""
    if is_eval_headless():
        return -1
    return cv2.waitKey(delay_ms)


def safe_imshow(name: str, img: np.ndarray) -> None:
    """Show img in a named window. If the user closed it, skip silently."""
    if is_eval_headless():
        return
    if name in _closed_windows:
        return
    try:
        if name in _shown_windows:
            # Only check visibility after the window has been created
            prop = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
            if prop < 1:   # -1 = destroyed, 0 = closed by user
                _closed_windows.add(name)
                return
        cv2.imshow(name, img)
        _shown_windows.add(name)
    except Exception:
        _closed_windows.add(name)


def build_rgb_map_2d(robot) -> np.ndarray:
    """Build a top-down RGB map from the loaded VLMap (done once per scene)."""
    return pool_3d_rgb_to_2d(robot.map.grid_rgb, robot.map.grid_pos, robot.map.gs)


def show_obs(robot, label: str = "", yoloe_frame_bgr: np.ndarray = None):
    """Display the first-person camera view (with optional YOLOE overlay)."""
    global _frozen_detection_bgr
    if is_eval_headless():
        return
    if yoloe_frame_bgr is not None:
        frame = yoloe_frame_bgr.copy()
    elif _frozen_detection_bgr is not None:
        frame = _frozen_detection_bgr.copy()
    else:
        obs = robot.sim.get_sensor_observations(0)
        if "color_sensor" in obs:
            frame = cv2.cvtColor(obs["color_sensor"][:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            ui_wait(1)
            return

    if label:
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    safe_imshow("1st person", frame)
    ui_wait(1)


def postprocess_heatmap(
    heatmap: np.ndarray,
    blur_ksize: int = 5,
    blur_sigma: float = 1.0,
    rel_thresh: float = 0.5,
    min_area: int = 3,
    score_mode: str = "mean_log_area",
    keep_ratio: float = 0.2,
):
    """Remove spurious small activations from a 2D heatmap.

    Pipeline:
      1. Light Gaussian blur to suppress high-frequency noise.
      2. Relative threshold (fraction of smoothed max) to binarise.
      3. Connected components on the binary mask.
      4. Score each component: 'mean_log_area' (default) or 'max_sqrt_area'.
      5. Keep components whose score >= keep_ratio * best_score.

    Args:
        heatmap:    float32 (H, W) — raw heatmap from compute_heatmap.
        blur_ksize: Gaussian kernel size (0 = skip blur).
        blur_sigma: Gaussian sigma.
        rel_thresh: Threshold = rel_thresh * smoothed_max.
        min_area:   Minimum component area to consider (px²).
        score_mode: 'mean_log_area' or 'max_sqrt_area'.
        keep_ratio: Components with score >= keep_ratio * best_score are kept.

    Returns:
        cleaned_heatmap: float32, same shape — zeros outside retained components.
        final_mask:      bool array — True where a retained component is.
        kept_components: list of dicts with keys label, area, max_val, mean_val,
                         sum_val, bbox (x,y,w,h), centroid (row,col), score.
                         Sorted by score descending.
    """
    if heatmap.max() < 1e-6:
        return heatmap.copy(), np.zeros(heatmap.shape, dtype=bool), []

    h = heatmap.astype(np.float32)

    # 1. Gaussian blur (blur_ksize must be odd)
    if blur_ksize > 0:
        ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        h_smooth = cv2.GaussianBlur(h, (ksize, ksize), blur_sigma)
    else:
        h_smooth = h.copy()

    # 2. Relative threshold
    thr = rel_thresh * h_smooth.max()
    binary = (h_smooth >= thr).astype(np.uint8)

    # 3. Connected components (8-connectivity)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 4. Score each component (label 0 = background)
    components = []
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        mask_lbl = labels == lbl
        vals = heatmap[mask_lbl]
        max_val = float(vals.max())
        mean_val = float(vals.mean())
        sum_val = float(vals.sum())

        if score_mode == "mean_log_area":
            score = mean_val * np.log1p(area)
        else:  # "max_sqrt_area"
            score = max_val * np.sqrt(area)

        bbox = (
            int(stats[lbl, cv2.CC_STAT_LEFT]),
            int(stats[lbl, cv2.CC_STAT_TOP]),
            int(stats[lbl, cv2.CC_STAT_WIDTH]),
            int(stats[lbl, cv2.CC_STAT_HEIGHT]),
        )
        centroid = (float(centroids[lbl, 1]), float(centroids[lbl, 0]))  # (row, col)

        components.append(dict(
            label=lbl, area=area, max_val=max_val, mean_val=mean_val,
            sum_val=sum_val, bbox=bbox, centroid=centroid, score=score,
        ))

    if not components:
        return heatmap.copy(), np.zeros(heatmap.shape, dtype=bool), []

    # 5. Keep components competitive with the best one
    best_score = max(c["score"] for c in components)
    kept = [c for c in components if c["score"] >= keep_ratio * best_score]

    # Region quality: favour semantically strong regions with real spatial support.
    # Tiny high-density blobs should not beat a large sofa-sized region.
    max_area = max(c["area"] for c in kept) if kept else 1
    max_sum = max(c["sum_val"] for c in kept) if kept else 1.0
    for c in kept:
        norm_area = c["area"] / max_area
        mean_score = c["mean_val"]
        norm_sum = c["sum_val"] / max_sum if max_sum > 1e-6 else 0.0
        c["quality"] = 0.20 * norm_area + 0.35 * mean_score + 0.45 * norm_sum

    kept.sort(key=lambda c: c["quality"], reverse=True)

    # 6. Build outputs
    final_mask = np.zeros(heatmap.shape, dtype=bool)
    for c in kept:
        final_mask |= (labels == c["label"])

    cleaned_heatmap = np.where(final_mask, heatmap, 0.0).astype(np.float32)
    return cleaned_heatmap, final_mask, kept


def get_runtime_heatmap_mode() -> str:
    """Return the heatmap mode requested for runtime evaluation."""
    mode = str(os.environ.get(_HEATMAP_MODE_ENV, "postprocessed")).strip().lower()
    if mode not in {"baseline", "postprocessed"}:
        print(f"  [heatmap] Unknown mode '{mode}' — falling back to 'postprocessed'")
        return "postprocessed"
    return mode


def compute_raw_heatmap(robot, category: str, score_thresh: float = 0.3) -> np.ndarray:
    """Compute the baseline/raw 2D heatmap before postprocessing."""
    from vlmaps.utils.index_utils import find_similar_category_id

    cat_id = find_similar_category_id(category, robot.map.categories)
    scores = robot.map.scores_mat[:, cat_id]
    max_ids = np.argmax(robot.map.scores_mat, axis=1)

    valid = (max_ids == cat_id) & (scores > score_thresh)
    scores_filtered = np.where(valid, scores, 0.0)

    gs = robot.map.gs
    heat_2d = np.zeros((gs, gs), dtype=np.float32)
    for i, pos in enumerate(robot.map.grid_pos):
        row, col, _ = pos
        if scores_filtered[i] > heat_2d[row, col]:
            heat_2d[row, col] = scores_filtered[i]

    mask = heat_2d > 0
    if mask.any():
        dist = distance_transform_edt(~mask)
        heat_2d = np.where(mask, heat_2d, np.clip(1.0 - dist * 0.3, 0, 1).astype(np.float32))
        heat_2d[heat_2d < 0.15] = 0

    return heat_2d


def _extract_heatmap_components(
    heatmap: np.ndarray,
    blur_ksize: int = 5,
    blur_sigma: float = 1.0,
    rel_thresh: float = 0.5,
    min_area: int = 3,
) -> list:
    """Extract connected components from a heatmap without filtering them away."""
    if heatmap.max() < 1e-6:
        return []

    h = heatmap.astype(np.float32)
    if blur_ksize > 0:
        ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        h_smooth = cv2.GaussianBlur(h, (ksize, ksize), blur_sigma)
    else:
        h_smooth = h.copy()

    thr = rel_thresh * h_smooth.max()
    binary = (h_smooth >= thr).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    components = []
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        mask_lbl = labels == lbl
        vals = heatmap[mask_lbl]
        if vals.size == 0:
            continue
        max_val = float(vals.max())
        mean_val = float(vals.mean())
        sum_val = float(vals.sum())
        bbox = (
            int(stats[lbl, cv2.CC_STAT_LEFT]),
            int(stats[lbl, cv2.CC_STAT_TOP]),
            int(stats[lbl, cv2.CC_STAT_WIDTH]),
            int(stats[lbl, cv2.CC_STAT_HEIGHT]),
        )
        centroid = (float(centroids[lbl, 1]), float(centroids[lbl, 0]))
        components.append(
            dict(
                label=lbl,
                area=area,
                max_val=max_val,
                mean_val=mean_val,
                sum_val=sum_val,
                bbox=bbox,
                centroid=centroid,
                score=mean_val * np.log1p(area),
            )
        )

    if not components:
        return []

    max_area = max(c["area"] for c in components)
    max_sum = max(c["sum_val"] for c in components)
    for c in components:
        norm_area = c["area"] / max_area if max_area > 0 else 0.0
        mean_score = c["mean_val"]
        norm_sum = c["sum_val"] / max_sum if max_sum > 1e-6 else 0.0
        c["quality"] = 0.20 * norm_area + 0.35 * mean_score + 0.45 * norm_sum

    components.sort(key=lambda c: c["quality"], reverse=True)
    return components


def compute_heatmap(robot, category: str, score_thresh: float = 0.3):
    """Compute the 2D heatmap using continuous CLIP scores instead of binary argmax.

    Steps:
      1. Get the raw CLIP score for the target category (per-voxel, 0-1).
      2. Keep only voxels where the target is the argmax AND score > threshold.
      3. Project the continuous scores to 2D (max-pool per column).
      4. Apply distance decay from the high-score cells.
      5. Postprocess to suppress spurious small activations.

    Returns:
        heat_2d:         float32 (gs, gs) — cleaned heatmap.
        kept_components: list of dicts from postprocess_heatmap (score-sorted).
    """
    mode = get_runtime_heatmap_mode()
    heat_2d = compute_raw_heatmap(robot, category, score_thresh=score_thresh)

    if mode == "baseline":
        kept_components = _extract_heatmap_components(heat_2d)
        if kept_components:
            print(f"  Heatmap mode: baseline/raw — {len(kept_components)} component(s) "
                  f"(areas: {[c['area'] for c in kept_components[:8]]}, "
                  f"quality: {[round(c['quality'], 3) for c in kept_components[:8]]})")
        else:
            print("  Heatmap mode: baseline/raw — no components retained")
        return heat_2d, kept_components

    heat_2d, _, kept_components = postprocess_heatmap(heat_2d)
    if kept_components:
        print(f"  Heatmap mode: postprocessed — kept {len(kept_components)} component(s) "
              f"(areas: {[c['area'] for c in kept_components]}, "
              f"quality: {[round(c['quality'], 3) for c in kept_components]})")
    else:
        print("  Heatmap mode: postprocessed — no components retained")

    return heat_2d, kept_components


def _room_state_eval_payload(rs) -> dict:
    return {
        "name": rs.name,
        "times_visited": int(rs.times_visited),
        "candidates_tried": int(rs.candidates_tried),
        "candidates_confirmed": int(rs.candidates_confirmed),
        "target_relevance": float(rs.target_relevance),
        "target_found_here": bool(rs.target_found_here),
        "navigable_ratio": float(rs.explored_ratio),
    }


def _search_state_eval_payload(ss: SearchState) -> dict:
    total_tried = int(sum(rs.candidates_tried for rs in ss.rooms.values()))
    total_confirmed = int(sum(rs.candidates_confirmed for rs in ss.rooms.values()))
    return {
        "original_target": getattr(ss, "original_target", ss.target),
        "canonical_target": getattr(ss, "canonical_target", ss.target),
        "effective_target": getattr(ss, "effective_target", ss.target),
        "surrogate_categories": list(getattr(ss, "surrogate_categories", [])),
        "likely_rooms": list(getattr(ss, "likely_rooms", [])),
        "resolution_source": getattr(ss, "resolution_source", "fallback"),
        "target": ss.target,
        "found": bool(ss.found),
        "current_room": ss.current_room,
        "visit_history": list(ss.visit_history),
        "rooms_attempted": list(getattr(ss, "rooms_attempted", [])),
        "furniture_per_room": dict(getattr(ss, "furniture_per_room", {})),
        "found_in_room": getattr(ss, "found_in_room", None),
        "room_transitions": int(max(len(ss.visit_history) - 1, 0)),
        "total_candidates_tried": total_tried,
        "total_candidates_confirmed": total_confirmed,
        "wrong_visits": int(max(total_tried - total_confirmed, 0)),
        "action_log": list(ss.action_log),
        "visited_cells_count": int(len(ss.visited_cells)),
        "found_on_arrival": bool(getattr(ss, "found_on_arrival", False)),
        "found_after_turn_to_face": bool(getattr(ss, "found_after_turn_to_face", False)),
        "found_after_centering": bool(getattr(ss, "found_after_centering", False)),
        "found_after_local_scan": bool(getattr(ss, "found_after_local_scan", False)),
        "found_after_pitch_scan": bool(getattr(ss, "found_after_pitch_scan", False)),
        "found_after_alternative_route": bool(getattr(ss, "found_after_alternative_route", False)),
        "final_confirmation_source": getattr(ss, "final_confirmation_source", None),
        "rooms": {
            name: _room_state_eval_payload(rs)
            for name, rs in ss.rooms.items()
        },
    }


def build_instruction_eval_summary(
    instruction: str,
    categories: list,
    search_states: dict,
    robot,
    room_provider,
) -> dict:
    robot._set_nav_curr_pose()
    final_room = None
    if room_provider and room_provider.is_available():
        final_room = room_provider.get_room_at_cell(
            int(robot.curr_pos_on_map[0]),
            int(robot.curr_pos_on_map[1]),
        )
    return {
        "instruction": instruction,
        "targets": [str(c) for c in categories],
        "final_room": final_room,
        "heatmap_mode": get_runtime_heatmap_mode(),
        "target_summaries": {
            target: _search_state_eval_payload(ss)
            for target, ss in search_states.items()
        },
    }


def emit_instruction_eval_summary(
    instruction: str,
    categories: list,
    search_states: dict,
    robot,
    room_provider,
) -> None:
    payload = build_instruction_eval_summary(
        instruction,
        categories,
        search_states,
        robot,
        room_provider,
    )
    print(f"[eval-summary] {json.dumps(payload, sort_keys=True)}")


def show_map(robot, rgb_map_2d: np.ndarray, heatmap_2d: np.ndarray = None,
             path_cells: list = None, label: str = "",
             zoom_radius: int = 300, output_px: int = 700,
             target_cell: list = None):
    """Display a top-down semantic map with optional zoom centered on the robot.

    Args:
        zoom_radius: Half-side of the crop window in map cells. The displayed
                     region is (2*zoom_radius) × (2*zoom_radius) cells, upscaled
                     to output_px for readability. Use 0 to show the full map.
        output_px:   Target display size in pixels (square).
    """
    if is_eval_headless():
        return
    gs = robot.map.gs
    row = int(robot.curr_pos_on_map[0])
    col = int(robot.curr_pos_on_map[1])

    # ── Crop region ───────────────────────────────────────────────────────────
    if zoom_radius > 0:
        r0 = max(0, row - zoom_radius)
        r1 = min(gs, row + zoom_radius)
        c0 = max(0, col - zoom_radius)
        c1 = min(gs, col + zoom_radius)
    else:
        r0, r1, c0, c1 = 0, gs, 0, gs

    # ── Base: RGB crop ────────────────────────────────────────────────────────
    canvas = rgb_map_2d[r0:r1, c0:c1].astype(np.float32).copy()

    # ── Semantic heatmap overlay ──────────────────────────────────────────────
    if heatmap_2d is not None:
        h_crop = heatmap_2d[r0:r1, c0:c1]
        heatmap_u8 = (np.clip(h_crop, 0, 1) * 255).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heat_rgb = heat_bgr[:, :, ::-1].astype(np.float32)
        canvas = canvas * 0.5 + heat_rgb * 0.5

    canvas_bgr = cv2.cvtColor(np.clip(canvas, 0, 255).astype(np.uint8),
                               cv2.COLOR_RGB2BGR)

    # ── Planned path (remapped to crop coords) ────────────────────────────────
    if path_cells and len(path_cells) > 1:
        pts = np.array(
            [[c[1] - c0, c[0] - r0] for c in path_cells
             if r0 <= c[0] < r1 and c0 <= c[1] < c1],
            dtype=np.int32,
        )
        if len(pts) > 1:
            cv2.polylines(canvas_bgr, [pts], False, (0, 0, 255), 2)

    # ── Robot position ────────────────────────────────────────────────────────
    robot_r = row - r0
    robot_c = col - c0
    cv2.circle(canvas_bgr, (robot_c, robot_r), 5, (0, 255, 0), -1)
    cv2.circle(canvas_bgr, (robot_c, robot_r), 7, (255, 255, 255), 1)

    # ── Frozen / explicit target marker ──────────────────────────────────────
    global _frozen_target_cell
    marker = target_cell if target_cell is not None else _frozen_target_cell
    if marker is not None:
        mr = int(marker[0]) - r0
        mc = int(marker[1]) - c0
        if 0 <= mr < (r1 - r0) and 0 <= mc < (c1 - c0):
            cv2.circle(canvas_bgr, (mc, mr), 7, (0, 215, 255), 2)
            cv2.drawMarker(canvas_bgr, (mc, mr), (0, 215, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

    # ── Label ─────────────────────────────────────────────────────────────────
    if label:
        cv2.putText(canvas_bgr, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas_bgr, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # ── Upscale to output_px ──────────────────────────────────────────────────
    h_crop = r1 - r0
    w_crop = c1 - c0
    scale = output_px / max(h_crop, w_crop, 1)
    if abs(scale - 1.0) > 0.01:
        h_out = max(1, int(h_crop * scale))
        w_out = max(1, int(w_crop * scale))
        canvas_bgr = cv2.resize(canvas_bgr, (w_out, h_out),
                                 interpolation=cv2.INTER_LINEAR)

    safe_imshow("Semantic Map", canvas_bgr)
    ui_wait(1)


# ── Navigation helpers ────────────────────────────────────────────────────────

def _rasterize_segment_cells(start, end) -> list:
    """Rasterize a map segment into integer cells, including endpoints."""
    start_r, start_c = int(round(start[0])), int(round(start[1]))
    end_r, end_c = int(round(end[0])), int(round(end[1]))

    rmin = min(start_r, end_r)
    rmax = max(start_r, end_r)
    cmin = min(start_c, end_c)
    cmax = max(start_c, end_c)

    mask = np.zeros((rmax - rmin + 1, cmax - cmin + 1), dtype=np.uint8)
    cv2.line(mask, (start_c - cmin, start_r - rmin), (end_c - cmin, end_r - rmin), 1, 1)
    rows, cols = np.where(mask > 0)
    if rows.size == 0:
        if (start_r, start_c) == (end_r, end_c):
            return [[start_r, start_c]]
        return [[start_r, start_c], [end_r, end_c]]

    pts = [[int(rmin + rr), int(cmin + cc)] for rr, cc in zip(rows, cols)]
    pts.sort(key=lambda cell: (cell[0] - start_r) ** 2 + (cell[1] - start_c) ** 2)
    return pts


def densify_path_cells(path_cells: list) -> list:
    """Expand sparse path vertices into a cell-by-cell polyline."""
    if not path_cells:
        return []
    if len(path_cells) == 1:
        return [[int(path_cells[0][0]), int(path_cells[0][1])]]

    dense = []
    for i in range(len(path_cells) - 1):
        segment = _rasterize_segment_cells(path_cells[i], path_cells[i + 1])
        if i > 0 and segment:
            segment = segment[1:]
        dense.extend(segment)
    return dense


def normalize_path_cells(path_cells: list) -> list:
    """Convert a path to integer grid cells while preserving vertex structure."""
    if not path_cells:
        return []
    return [[int(round(cell[0])), int(round(cell[1]))] for cell in path_cells]


def _normalize_turn_error(angle_deg: float) -> float:
    """Wrap an angle difference to [-180, 180)."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _closest_path_index(curr_cell, dense_path: list, hint_idx: int, backtrack: int = 12, ahead: int = 80) -> int:
    """Find the closest path index near the current progress hint."""
    if not dense_path:
        return 0
    start = max(0, hint_idx - backtrack)
    end = min(len(dense_path), hint_idx + ahead + 1)
    best_idx = hint_idx if 0 <= hint_idx < len(dense_path) else 0
    best_dist = float("inf")
    for idx in range(start, end):
        cell = dense_path[idx]
        dr = float(cell[0]) - float(curr_cell[0])
        dc = float(cell[1]) - float(curr_cell[1])
        dist_sq = dr * dr + dc * dc
        # Small forward bias avoids oscillating backward on equally-close cells.
        dist_sq += max(0, idx - hint_idx) * 1e-3
        if dist_sq < best_dist:
            best_dist = dist_sq
            best_idx = idx
    return best_idx


def _lookahead_path_index(dense_path: list, start_idx: int, lookahead_cells: float) -> int:
    """Advance along the dense path until the requested arc-length is reached."""
    if not dense_path:
        return 0
    target_idx = start_idx
    traveled = 0.0
    for idx in range(start_idx, len(dense_path) - 1):
        a = dense_path[idx]
        b = dense_path[idx + 1]
        traveled += float(np.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))
        target_idx = idx + 1
        if traveled >= lookahead_cells:
            break
    return target_idx


def _path_cell_distance(a, b) -> float:
    """Euclidean distance between two grid cells."""
    return float(np.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))


def _segment_is_free_on_map(start, end, free_map: np.ndarray) -> bool:
    """Return True if the integer segment lies entirely inside free cells."""
    if free_map is None:
        return True
    h, w = free_map.shape
    for row, col in _rasterize_segment_cells(start, end):
        rr = int(np.clip(row, 0, h - 1))
        cc = int(np.clip(col, 0, w - 1))
        if not bool(free_map[rr, cc]):
            return False
    return True


def _visible_lookahead_index(
    curr_cell,
    dense_path: list,
    start_idx: int,
    preferred_idx: int,
    free_map: np.ndarray,
    min_target_dist_cells: float = 0.0,
) -> int:
    """Pick the farthest lookahead target that is still line-of-sight reachable.

    This stops the follower from aiming around a blind corner and trying to cut
    through the wall with the first forward action.
    """
    if not dense_path:
        return 0
    if free_map is None:
        return preferred_idx

    start_idx = int(np.clip(start_idx, 0, len(dense_path) - 1))
    preferred_idx = int(np.clip(preferred_idx, start_idx, len(dense_path) - 1))

    fallback_idx = None
    fallback_dist = -1.0

    for idx in range(preferred_idx, start_idx, -1):
        if _segment_is_free_on_map(curr_cell, dense_path[idx], free_map):
            cand_dist = _path_cell_distance(curr_cell, dense_path[idx])
            if fallback_idx is None or cand_dist > fallback_dist:
                fallback_idx = idx
                fallback_dist = cand_dist
            if cand_dist >= min_target_dist_cells:
                return idx

    if preferred_idx == start_idx:
        return start_idx
    if fallback_idx is not None:
        return fallback_idx
    return min(len(dense_path) - 1, start_idx + 1)


def face_toward_pos(robot, target_row: float, target_col: float) -> None:
    """Turn robot to face directly toward a specific map (row, col) position.

    Coordinate math (base frame: x=north=-row, y=west=-col, CCW-positive):
      angle = arctan2(-dy_map, -dx_map)  maps (row,col) deltas to base angle.
    Turn sign: robot.turn(+) = turn_right (CW) = decreasing base angle,
      matching convert_goal_to_actions convention (turn_right_angle = curr - target).
    Each step is shown individually for smooth demo visualisation.
    """
    robot._set_nav_curr_pose()
    dx = target_row - robot.curr_pos_on_map[0]   # positive = south = -x_base
    dy = target_col - robot.curr_pos_on_map[1]   # positive = east  = -y_base
    angle = np.arctan2(-dy, -dx) * 180.0 / np.pi  # CCW-positive, 0°=north
    turn = (robot.curr_ang_deg_on_map - angle + 180) % 360 - 180  # +→CW→turn_right
    n_turns = int(abs(turn) / robot.turn_angle)
    action = "turn_right" if turn > 0 else "turn_left"
    for _ in range(n_turns):
        robot.sim.step(action)
        show_obs(robot, "Facing…")
        ui_wait(_NAV_STEP_DELAY_MS)
    robot._set_nav_curr_pose()


def scan_360_and_verify(
    robot,
    cat: str,
    rgb_map_2d: np.ndarray,
    heatmap: np.ndarray,
    path_cells: list,
) -> bool:
    """Rotate 360° with real-time display and asynchronous YOLOE detection.

    - Uses native 5° sim actions for smooth, continuous rotation.
    - YOLOE runs in a background thread — the display never freezes.
    - Shows the latest YOLOE annotation overlaid on the live camera feed.
    - Stops as soon as the object is confirmed.

    Returns True if YOLOE detects the object during the scan.
    """
    global _frozen_detection_bgr
    from vlmaps.utils.yoloe_utils import get_session, runtime_conf_thresh

    session = get_session(cat, conf_thresh=runtime_conf_thresh(0.3))
    if session is None:
        print("  (YOLOE not available — skipping 360° scan)")
        return False

    # Native step is 5° (turn_angle param); full rotation = 72 steps
    step_deg = robot.turn_angle   # 5° per action
    n_steps = int(round(360 / step_deg))
    print(f"  Starting 360° scan ({n_steps} steps × {step_deg}°, async YOLOE)…")

    session.start_bg_thread()
    found = False
    last_ann_bgr = None

    try:
        for step in range(n_steps):
            angle_done = int(step * step_deg)

            # Single native sim action — no batching, so display stays smooth
            robot.sim.step("turn_right")

            obs = robot.sim.get_sensor_observations(0)
            if "color_sensor" in obs:
                frame_rgb = obs["color_sensor"][:, :, :3]

                # Submit to YOLOE (non-blocking, background thread)
                session.push_frame(frame_rgb)

                # Read latest available YOLOE result (non-blocking)
                det_found, ann_rgb, _bbox = session.poll_result()
                if ann_rgb is not None:
                    last_ann_bgr = cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR)
                if det_found:
                    found = True

                # Show frame: annotated if YOLOE has replied, raw otherwise
                if last_ann_bgr is not None:
                    label = f"Scan {angle_done}°: {'FOUND!' if found else cat}"
                    show_obs(robot, label, yoloe_frame_bgr=last_ann_bgr)
                else:
                    show_obs(robot, f"Scan {angle_done}°: {cat}")

            show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                     path_cells=path_cells,
                     label=f"Scan {angle_done}°: {cat}")
            ui_wait(_NAV_STEP_DELAY_MS)

            if found:
                print(f"  YOLOE scan: FOUND '{cat}' at {angle_done}°!")
                if last_ann_bgr is not None:
                    _frozen_detection_bgr = last_ann_bgr.copy()
                break

        # Wait briefly for the last in-flight result
        if not found:
            import time
            time.sleep(0.6)
            det_found, ann_rgb, _bbox = session.poll_result()
            if det_found:
                found = True
                if ann_rgb is not None:
                    last_ann_bgr = cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR)
                    _frozen_detection_bgr = last_ann_bgr.copy()
                    show_obs(robot, f"Scan final: FOUND! {cat}",
                             yoloe_frame_bgr=last_ann_bgr)
                print(f"  YOLOE scan: FOUND '{cat}' (last frame)!")

    finally:
        session.stop_bg_thread()
        # Restore pose tracking after raw sim steps
        robot._set_nav_curr_pose()

    if not found:
        print(f"  YOLOE scan: '{cat}' not found after full 360°.")
    return found


def _get_color_sensor_rotation(robot):
    """Return the current color sensor rotation, or None if unavailable."""
    try:
        agent_state = robot.sim.get_agent(0).get_state()
        sensor_state = getattr(agent_state, "sensor_states", {}).get("color_sensor")
        return getattr(sensor_state, "rotation", None)
    except Exception:
        return None


def _set_color_sensor_rotation(robot, rotation) -> bool:
    """Set the color sensor world rotation while preserving the agent pose."""
    try:
        agent = robot.sim.get_agent(0)
        agent_state = agent.get_state()
        sensor_state = getattr(agent_state, "sensor_states", {}).get("color_sensor")
        if sensor_state is None:
            return False
        sensor_state.rotation = rotation
        agent_state.sensor_states["color_sensor"] = sensor_state
        try:
            agent.set_state(agent_state, infer_sensor_states=False)
        except TypeError:
            agent.set_state(agent_state)
        return True
    except Exception as exc:
        print(f"  [verify] pitch_scan unavailable: {exc}")
        return False


def _set_color_sensor_pitch(robot, base_rotation, pitch_deg: float) -> bool:
    """Apply a camera-local pitch offset relative to *base_rotation*."""
    try:
        import quaternion

        pitch_delta = quaternion.from_rotation_vector(
            [np.deg2rad(float(pitch_deg)), 0.0, 0.0]
        )
        return _set_color_sensor_rotation(robot, base_rotation * pitch_delta)
    except Exception as exc:
        print(f"  [verify] pitch_scan unavailable: {exc}")
        return False


def scan_local_and_verify(
    robot,
    cat: str,
    rgb_map_2d: np.ndarray,
    heatmap: np.ndarray,
    path_cells: list,
    sweep_deg: float = 25.0,
    surrogate_cat: str = "",
) -> bool:
    """Bounded local angular verification — replaces the old 360° scan.

    Only probes the current orientation and two symmetric side views at
    ±*sweep_deg* (default ±25°). Returns True on the first YOLOE confirmation.
    The robot is returned to the original heading whether or not a detection
    is made, so the downstream pose state remains consistent.

    The aim is to avoid long-range false positives that a full 360° rotation
    would pick up from far-away instances of the same category.
    """
    from vlmaps.utils.yoloe_utils import get_session, runtime_conf_thresh

    session = get_session(cat, conf_thresh=runtime_conf_thresh(0.3))
    if session is None:
        print("  (YOLOE not available — skipping local scan)")
        return False
    setattr(robot, "_last_verify_source", None)

    step_deg = float(robot.turn_angle)  # native 5° actions
    if step_deg <= 0:
        print("  (invalid turn_angle — skipping local scan)")
        return False

    n_side_steps = int(round(sweep_deg / step_deg))
    sweep_effective = n_side_steps * step_deg
    print(
        f"  Starting local scan (±{sweep_effective:.0f}°, "
        f"{n_side_steps} steps per side × {step_deg:.0f}°) — NOT a 360° sweep"
    )

    def _check_now(label: str) -> bool:
        obs = robot.sim.get_sensor_observations(0)
        if "color_sensor" not in obs:
            return False
        frame = obs["color_sensor"][:, :, :3]
        det, ann_rgb, _bbox = session.check(frame)
        if ann_rgb is not None:
            ann_bgr = cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR)
            show_obs(robot, label, yoloe_frame_bgr=ann_bgr)
        else:
            show_obs(robot, label)
        show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                 path_cells=path_cells, label=label)
        ui_wait(_NAV_STEP_DELAY_MS)
        return bool(det)

    def _rotate(action: str, steps: int) -> None:
        for _ in range(max(steps, 0)):
            robot.sim.step(action)
            obs = robot.sim.get_sensor_observations(0)
            if "color_sensor" in obs:
                show_obs(robot, f"Local scan turn: {cat}")
            ui_wait(_NAV_STEP_DELAY_MS)

    found = False
    try:
        # 0° (current orientation)
        if _check_now(f"Local scan 0°: {cat}"):
            print(f"  YOLOE local scan: FOUND '{cat}' at 0°")
            setattr(robot, "_last_verify_source", "local_scan")
            return True

        # −sweep_deg
        _rotate("turn_left", n_side_steps)
        if _check_now(f"Local scan -{sweep_effective:.0f}°: {cat}"):
            print(f"  YOLOE local scan: FOUND '{cat}' at -{sweep_effective:.0f}°")
            setattr(robot, "_last_verify_source", "local_scan")
            found = True
        else:
            # Back to 0°
            _rotate("turn_right", n_side_steps)

        if not found:
            # +sweep_deg
            _rotate("turn_right", n_side_steps)
            if _check_now(f"Local scan +{sweep_effective:.0f}°: {cat}"):
                print(f"  YOLOE local scan: FOUND '{cat}' at +{sweep_effective:.0f}°")
                setattr(robot, "_last_verify_source", "local_scan")
                found = True
            # Always return to 0°
            _rotate("turn_left", n_side_steps)
    finally:
        robot._set_nav_curr_pose()

    if not found:
        surrogate_key = (surrogate_cat or "").strip().lower()
        pitch_values = _SURROGATE_PITCH_DEG.get(surrogate_key, _DEFAULT_PITCH_DEG)
        base_sensor_rotation = _get_color_sensor_rotation(robot)
        if base_sensor_rotation is not None:
            print(
                f"  Starting pitch scan for '{cat}' "
                f"(surrogate={surrogate_key or 'none'}, pitches={pitch_values})"
            )
            try:
                for pitch_deg in pitch_values:
                    if not _set_color_sensor_pitch(robot, base_sensor_rotation, pitch_deg):
                        continue
                    if _check_now(f"Pitch scan {pitch_deg:.0f}°: {cat}"):
                        print(f"  [verify] source=pitch_scan pitch={pitch_deg:.1f}")
                        print(f"  YOLOE pitch scan: FOUND '{cat}' at pitch={pitch_deg:.1f}°")
                        setattr(robot, "_last_verify_source", "pitch_scan")
                        found = True
                        break
            finally:
                _set_color_sensor_rotation(robot, base_sensor_rotation)
                robot._set_nav_curr_pose()
        else:
            print("  [verify] pitch_scan skipped: color sensor rotation unavailable")

    if not found:
        print(f"  YOLOE local scan: '{cat}' not found in ±{sweep_effective:.0f}° plus pitch scan.")
    return found


def navigate_to_alternative(
    robot,
    cat: str,
    kept_components: list,
    current_pos,
    rgb_map_2d: np.ndarray,
    heatmap: np.ndarray,
) -> bool:
    """Navigate to the best heatmap component NOT near the current position.

    Picks the highest-scoring component whose centroid is at least 20 cells
    from current_pos, plans a smooth path to its standoff, and faces it.

    Returns True if an alternative was found and navigated to.
    """
    if not kept_components:
        print("  No alternative components in heatmap — giving up.")
        return False

    curr_row, curr_col = float(current_pos[0]), float(current_pos[1])
    MIN_DIST_CELLS = 20  # ~1 m (cell_size=0.05 m)

    # Components are score-sorted descending; pick first that is far enough away
    alt = None
    for c in kept_components:
        dr = c["centroid"][0] - curr_row
        dc = c["centroid"][1] - curr_col
        if np.sqrt(dr * dr + dc * dc) >= MIN_DIST_CELLS:
            alt = c
            break

    if alt is None:
        print("  All heatmap components are near current position — giving up.")
        return False

    alt_centroid = [int(alt["centroid"][0]), int(alt["centroid"][1])]
    print(f"  Alternative: component score={alt['score']:.3f} "
          f"area={alt['area']} centroid={alt_centroid}")

    try:
        robot._set_nav_curr_pose()
        standoff_alt, boundary_alt = robot.map.get_standoff_pos(
            robot.curr_pos_on_map, cat, standoff_m=1.0)
        # Override standoff with the alternative centroid direction if needed
        # (get_standoff_pos picks the nearest VLMap region regardless of centroid)
        print(f"  Alternative standoff: {standoff_alt}  boundary: {boundary_alt}")
        robot.move_to(standoff_alt)
        robot._set_nav_curr_pose()
        face_toward_pos(robot, boundary_alt[0], boundary_alt[1])
        show_obs(robot, f"Alternative arrival: {cat}")
        show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                 label=f"Alternative: {cat}")
        return True
    except Exception as e:
        print(f"  Alternative navigation failed: {e}")
        return False


# ── Navigation robustness helpers ────────────────────────────────────────────

def _agent_xyz(robot) -> np.ndarray:
    """Return the agent's current 3-D world position from the simulator."""
    return np.array(robot.sim.get_agent(0).get_state().position, dtype=np.float64)


def execute_nav_replay(
    robot,
    planned_actions: list,
    cat: str,
    rgb_map_2d: np.ndarray,
    heatmap: np.ndarray,
    path_cells: list,
    display_path_cells: list = None,
    motion_thresh: float = 0.4,
    stuck_threshold: int = 3,
    dist_map: np.ndarray = None,
    goal_reached_tol_cells: float = None,
) -> bool:
    """Follow the planned route using the dense rasterized path as control reference.

    The planner may compact A* output into a short polyline, but executing
    against only those sparse vertices lets the discrete controller cut corners
    and drift away from the intended corridor.  This follower therefore uses
    the dense rasterized path for progress tracking and target selection, while
    still applying lookahead for smoother motion.
    """
    _STALL_STEPS = max(10, stuck_threshold * 3)

    print(f"  [nav] No pre-step shield — using Habitat navmesh collision")

    follow_path = normalize_path_cells(path_cells)
    if not follow_path:
        return True
    dense_path = display_path_cells if display_path_cells is not None else densify_path_cells(follow_path)
    control_path = normalize_path_cells(dense_path)
    if not control_path:
        control_path = follow_path

    goal_cell = control_path[-1]
    free_map = getattr(robot, "_safe_obs_map", None)
    if free_map is None:
        free_map = getattr(robot.map, "obstacles_map", None)
    low_motion_count = 0
    expected_fwd = robot.forward_dist
    _cs = getattr(robot, "cs", 0.05)
    _fwd_cells = max(1, int(round(expected_fwd / _cs)))
    _LOOKAHEAD_OPEN_CELLS = max(6, _fwd_cells * 3)
    _LOOKAHEAD_TIGHT_CELLS = max(3, _fwd_cells * 2)
    _LOOKAHEAD_TIGHT_CL = 5.0
    _GOAL_REACHED_TOL = (
        float(goal_reached_tol_cells)
        if goal_reached_tol_cells is not None
        else float(max(3, _fwd_cells + 1))
    )
    _MAX_FOLLOW_STEPS = max(len(control_path) * 2, len(follow_path) * 12, 120)
    _FORWARD_HEADING_TOL_OPEN = 10.0
    _FORWARD_HEADING_TOL_TIGHT = 18.0
    _OFFTRACK_REATTACH_TH = max(1.5, 0.75 * _fwd_cells)
    _OFFTRACK_DEADBAND_TH = max(1.0, 0.5 * _fwd_cells)
    _MIN_ACTIONABLE_TARGET_DIST = max(1.0, 0.75 * _fwd_cells)
    _STALL_HEADING_EPS = max(2.5, float(robot.turn_angle) * 0.5)
    progress_idx = 0
    progress_floor_idx = 0
    last_progress_idx = -1
    last_cell = None
    last_heading = None
    stall_count = 0
    preview_stall_count = 0
    offtrack_log_counter = 0

    print(f"  [nav] Path follower: {len(follow_path)} polyline waypoint(s), "
          f"{len(control_path)} control cell(s), {len(dense_path)} dense display cell(s), "
          f"lookahead={_LOOKAHEAD_OPEN_CELLS} open / {_LOOKAHEAD_TIGHT_CELLS} tight")

    for i in range(_MAX_FOLLOW_STEPS):
        robot._set_nav_curr_pose()
        _curr_cell = [int(round(robot.curr_pos_on_map[0])), int(round(robot.curr_pos_on_map[1]))]
        _curr_heading = float(robot.curr_ang_deg_on_map)
        progress_idx = _closest_path_index(
            _curr_cell,
            control_path,
            progress_idx,
            backtrack=16,
            ahead=max(120, _LOOKAHEAD_OPEN_CELLS * 8),
        )
        progress_idx = max(progress_idx, progress_floor_idx)
        _heading_changed = (
            last_heading is None
            or abs(_normalize_turn_error(_curr_heading - last_heading)) > _STALL_HEADING_EPS
        )
        if (
            last_cell == tuple(_curr_cell)
            and progress_idx <= last_progress_idx
            and not _heading_changed
        ):
            stall_count += 1
            if stall_count >= _STALL_STEPS:
                print(f"  [nav] No progress for {stall_count} steps at cell {_curr_cell} "
                      f"(progress={progress_idx}) — aborting early")
                return False
        else:
            stall_count = 0
            last_cell = tuple(_curr_cell)
            last_progress_idx = progress_idx
            last_heading = _curr_heading

        _path_anchor = control_path[progress_idx]
        _path_offset = float(
            np.hypot(float(_path_anchor[0]) - float(_curr_cell[0]), float(_path_anchor[1]) - float(_curr_cell[1]))
        )
        _goal_dr = float(goal_cell[0]) - float(_curr_cell[0])
        _goal_dc = float(goal_cell[1]) - float(_curr_cell[1])
        _goal_dist = float(np.hypot(_goal_dr, _goal_dc))
        if progress_idx >= len(control_path) - 1 and _goal_dist <= _GOAL_REACHED_TOL:
            print(f"  [nav] Goal reached on control path "
                  f"(dist={_goal_dist:.1f} cells, progress={progress_idx + 1}/{len(control_path)})")
            return True

        _curr_clearance = float("inf")
        if dist_map is not None:
            _r = int(np.clip(_curr_cell[0], 0, dist_map.shape[0] - 1))
            _c = int(np.clip(_curr_cell[1], 0, dist_map.shape[1] - 1))
            _curr_clearance = float(dist_map[_r, _c])
        _lookahead_cells = (
            _LOOKAHEAD_TIGHT_CELLS if _curr_clearance < _LOOKAHEAD_TIGHT_CL else _LOOKAHEAD_OPEN_CELLS
        )
        if _path_offset > _OFFTRACK_REATTACH_TH:
            _lookahead_cells = max(1, min(_lookahead_cells, _fwd_cells + 1))
            if offtrack_log_counter == 0 or i % 15 == 0:
                print(f"  [nav] Off-path by {_path_offset:.1f} cells at progress={progress_idx} "
                      f"— shortening lookahead to {_lookahead_cells}")
            offtrack_log_counter += 1
        else:
            offtrack_log_counter = 0
        _min_target_dist = (
            0.0
            if (_path_offset > _OFFTRACK_REATTACH_TH or _goal_dist <= _GOAL_REACHED_TOL + _fwd_cells)
            else _MIN_ACTIONABLE_TARGET_DIST
        )
        preferred_idx = _lookahead_path_index(control_path, progress_idx, _lookahead_cells)
        target_idx = _visible_lookahead_index(
            _curr_cell,
            control_path,
            progress_idx,
            preferred_idx,
            free_map,
            min_target_dist_cells=_min_target_dist,
        )
        target_cell = control_path[target_idx]

        _curr_pose = (
            float(robot.curr_pos_on_map[0]),
            float(robot.curr_pos_on_map[1]),
            float(robot.curr_ang_deg_on_map),
        )
        _target_dr = float(target_cell[0]) - _curr_pose[0]
        _target_dc = float(target_cell[1]) - _curr_pose[1]
        _target_dist = float(np.hypot(_target_dr, _target_dc))
        _target_angle = float(np.degrees(np.arctan2(-_target_dc, -_target_dr)))
        _heading_err = _normalize_turn_error(_curr_pose[2] - _target_angle)
        _preview = robot.controller.convert_goal_to_actions(_curr_pose, target_cell)

        if not _preview and target_idx < len(control_path) - 1:
            _search_end = min(len(control_path) - 1, max(preferred_idx, target_idx + _lookahead_cells))
            for cand_idx in range(target_idx + 1, _search_end + 1):
                cand_cell = control_path[cand_idx]
                cand_dist = _path_cell_distance(_curr_cell, cand_cell)
                if cand_dist < _min_target_dist and cand_idx < len(control_path) - 1:
                    continue
                if free_map is not None and not _segment_is_free_on_map(_curr_cell, cand_cell, free_map):
                    continue
                cand_preview = robot.controller.convert_goal_to_actions(_curr_pose, cand_cell)
                if cand_preview:
                    if i == 0 or i % 20 == 0:
                        print(f"  [nav] Advancing target for actionable preview: "
                              f"{target_idx - progress_idx} -> {cand_idx - progress_idx} cells ahead")
                    target_idx = cand_idx
                    target_cell = cand_cell
                    _target_dr = float(target_cell[0]) - _curr_pose[0]
                    _target_dc = float(target_cell[1]) - _curr_pose[1]
                    _target_dist = float(np.hypot(_target_dr, _target_dc))
                    _target_angle = float(np.degrees(np.arctan2(-_target_dc, -_target_dr)))
                    _heading_err = _normalize_turn_error(_curr_pose[2] - _target_angle)
                    _preview = cand_preview
                    break

        if not _preview:
            if target_idx < len(control_path) - 1:
                preview_stall_count += 1
                if preview_stall_count >= _STALL_STEPS:
                    print(f"  [nav] Preview produced no action for {preview_stall_count} steps "
                          f"at progress={progress_idx} target={target_idx} — aborting")
                    return False
                progress_floor_idx = min(max(progress_floor_idx, target_idx), len(control_path) - 1)
                if i == 0 or i % 10 == 0:
                    print(f"  [nav] No-op preview zone at progress={progress_idx}, target={target_idx} "
                          f"— raising progress floor to {progress_floor_idx}")
                continue
            if _goal_dist <= _GOAL_REACHED_TOL + _fwd_cells:
                print(f"  [nav] Goal reached after lookahead convergence "
                      f"(dist={_goal_dist:.1f} cells)")
                return True
            _preview = robot.controller.convert_goal_to_actions(_curr_pose, goal_cell)
            if not _preview:
                return True
        preview_stall_count = 0
        progress_floor_idx = min(progress_floor_idx, progress_idx)

        if preferred_idx != target_idx and (i == 0 or i % 20 == 0):
            print(f"  [nav] Lookahead clipped by local visibility: "
                  f"{preferred_idx - progress_idx} -> {target_idx - progress_idx} cells ahead")

        action = _preview[0]
        _heading_tol = (
            _FORWARD_HEADING_TOL_TIGHT if _curr_clearance < _LOOKAHEAD_TIGHT_CL else _FORWARD_HEADING_TOL_OPEN
        )
        if (
            action in ("turn_left", "turn_right")
            and abs(_heading_err) <= _heading_tol
            and _target_dist >= max(1.0, 0.75 * _fwd_cells)
            and _path_offset <= _OFFTRACK_DEADBAND_TH
            # Do NOT require _preview[1]=="move_forward": when heading error is
            # small, forward motion is always preferable regardless of how many
            # turns the controller would plan after the current one.
        ):
            if i == 0 or i % 20 == 0:
                print(f"  [nav] Heading deadband: |err|={abs(_heading_err):.1f}° "
                      f"<= {_heading_tol:.1f}° — prioritizing forward motion")
            action = "move_forward"

        is_fwd = (action == "move_forward")

        if is_fwd:
            pre_xyz = _agent_xyz(robot)

        robot.sim.step(action)
        robot._set_nav_curr_pose()

        if is_fwd:
            disp = float(np.linalg.norm(_agent_xyz(robot) - pre_xyz))
            if disp < motion_thresh * expected_fwd:
                low_motion_count += 1
                if low_motion_count >= stuck_threshold:
                    print(f"  [nav] Stuck after {i+1}/{_MAX_FOLLOW_STEPS} follower steps "
                          f"(last disp={disp*100:.1f} cm) — triggering recovery")
                    return False
            else:
                low_motion_count = 0

        _step_label = f"[{i+1}] -> {cat}"
        show_obs(robot, _step_label)
        if i % _MAP_REFRESH_STRIDE == 0 or i == _MAX_FOLLOW_STEPS - 1:
            show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                     path_cells=dense_path, label=_step_label)
        ui_wait(_NAV_STEP_DELAY_MS)

    print(f"  [nav] Path follower exceeded safety step budget "
          f"({_MAX_FOLLOW_STEPS}) before reaching goal")
    return False


def nav_recovery_and_replan(
    robot,
    standoff_pos: list,
    cat: str,
    rgb_map_2d: np.ndarray,
    heatmap: np.ndarray,
    recovery_turn_deg: float = 30.0,
) -> bool:
    """Escape a stuck state and replan to standoff_pos from the current pose.

    Executes a small recovery turn, then calls robot.move_to() which both
    plans and executes the new path directly (no reset/replay needed).

    Returns True if recovery navigation succeeded.
    """
    print(f"  [nav] Recovery: turning {recovery_turn_deg}° then replanning…")
    n_turns = max(1, int(round(recovery_turn_deg / robot.turn_angle)))
    for _ in range(n_turns):
        robot.sim.step("turn_right")
    robot._set_nav_curr_pose()

    robot.empty_recorded_actions()
    try:
        robot.move_to(standoff_pos)
        recovery_actions = robot.get_recorded_actions() or []
        n = len(recovery_actions)
        if n == 0:
            print("  [nav] Recovery: already at standoff.")
            return True
        for i, action in enumerate(recovery_actions):
            if action == "stop":
                continue
            robot.sim.step(action)
            robot._set_nav_curr_pose()
            _recovery_label = f"[recovery {i+1}] -> {cat}"
            show_obs(robot, _recovery_label)
            if i % _MAP_REFRESH_STRIDE == 0 or i == n - 1:
                show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                         label=_recovery_label)
            ui_wait(_NAV_STEP_DELAY_MS)
        print("  [nav] Recovery complete.")
        return True
    except Exception as e:
        print(f"  [nav] Recovery replan failed: {e}")
        return False


def fine_visual_center(
    robot,
    session,
    cat: str,
    img_w: int = 640,
    img_fov_h: float = 90.0,
    max_turns: int = 8,
) -> bool:
    """Iterative horizontal centering via YOLOE bounding-box feedback.

    Uses the persistent YOLOE session to get bounding-box centre coordinates
    and turns the robot left/right in single 5° steps until the horizontal
    error falls within tolerance.

    Note on vertical centering: the Habitat discrete action space has no
    pitch control (only turn_left, turn_right, move_forward).  Vertical
    alignment is therefore not achievable here; only horizontal centering
    is performed.

    Args:
        img_w:      Frame width in pixels (default 640 from sim config).
        img_fov_h:  Horizontal field-of-view in degrees (default 90°).
        max_turns:  Maximum number of 5° correction steps in either direction.

    Returns:
        True if the object was detected at least once during centering.
    """
    # Tolerance: half a step's worth of pixels (centred within ±2.5°)
    px_per_step = img_w * (robot.turn_angle / img_fov_h)
    tolerance_px = px_per_step / 2.0
    detected_once = False

    for step_i in range(max_turns):
        obs = robot.sim.get_sensor_observations(0)
        if "color_sensor" not in obs:
            break

        frame = obs["color_sensor"][:, :, :3]
        found, ann_rgb, bbox_center = session.check(frame)

        if ann_rgb is not None:
            show_obs(robot, f"Centering {step_i+1}/{max_turns}: {cat}",
                     yoloe_frame_bgr=cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR))

        if not found or bbox_center is None:
            print(f"  [center] Step {step_i+1}: '{cat}' not detected.")
            break

        detected_once = True
        cx, _cy = bbox_center
        err_x = cx - img_w / 2.0

        if abs(err_x) <= tolerance_px:
            print(f"  [center] Centred (err_x={err_x:+.1f}px ≤ {tolerance_px:.1f})")
            break

        action = "turn_right" if err_x > 0 else "turn_left"
        print(f"  [center] Step {step_i+1}: err_x={err_x:+.1f}px → {action}")
        robot.sim.step(action)
        ui_wait(_NAV_STEP_DELAY_MS)

    robot._set_nav_curr_pose()
    if not detected_once:
        print(f"  [center] '{cat}' not detected — skipping centering.")
    return detected_once


def freeze_found_target(cat: str, ann_frame_rgb, target_cell) -> None:
    """Persist the last positive YOLOE result in both the RGB and map views."""
    global _frozen_detection_bgr, _frozen_target_cell
    if ann_frame_rgb is not None:
        _frozen_detection_bgr = cv2.cvtColor(ann_frame_rgb, cv2.COLOR_RGB2BGR)
    if target_cell is not None:
        _frozen_target_cell = [int(target_cell[0]), int(target_cell[1])]


def _snap_to_nearest_free_cell(cell, free_map: np.ndarray, max_radius: int = 20):
    """Return the nearest free grid cell to *cell* inside *free_map*."""
    if free_map is None:
        return [int(cell[0]), int(cell[1])]

    h, w = free_map.shape[:2]
    r0 = int(np.clip(round(cell[0]), 0, h - 1))
    c0 = int(np.clip(round(cell[1]), 0, w - 1))
    if free_map[r0, c0]:
        return [r0, c0]

    for radius in range(1, max_radius + 1):
        rmin = max(0, r0 - radius)
        rmax = min(h - 1, r0 + radius)
        cmin = max(0, c0 - radius)
        cmax = min(w - 1, c0 + radius)
        best = None
        best_dist = float("inf")
        for rr in range(rmin, rmax + 1):
            for cc in range(cmin, cmax + 1):
                if rr not in (rmin, rmax) and cc not in (cmin, cmax):
                    continue
                if not free_map[rr, cc]:
                    continue
                dist = float((rr - r0) ** 2 + (cc - c0) ** 2)
                if dist < best_dist:
                    best = [rr, cc]
                    best_dist = dist
        if best is not None:
            return best

    return [r0, c0]


def _compute_bfs_cost_map(start_cell, free_map: np.ndarray) -> np.ndarray:
    """Compute 4-neighbour BFS distances from *start_cell* over free cells."""
    h, w = free_map.shape[:2]
    dist = np.full((h, w), -1, dtype=np.int32)
    sr, sc = _snap_to_nearest_free_cell(start_cell, free_map)
    if not free_map[sr, sc]:
        return dist

    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    while q:
        r, c = q.popleft()
        nd = dist[r, c] + 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                continue
            if dist[rr, cc] != -1 or not free_map[rr, cc]:
                continue
            dist[rr, cc] = nd
            q.append((rr, cc))
    return dist


def select_best_room(
    search_state: SearchState,
    robot_pos,
    heatmap_evidence: dict,
    obs_map: np.ndarray,
    kept_components: list,
    *,
    current_room: str = None,
    w_prior: float = 0.30,
    w_evidence: float = 0.25,
    w_unexplored: float = 0.20,
    w_cost: float = 0.15,
    w_penalty: float = 0.10,
):
    """Phase D — choose the next room to search before picking a candidate.

    The current system does not yet maintain per-cell coverage within each room,
    so the "unexplored" term is implemented as a practical proxy derived from
    room visits and failed candidate inspections.
    """
    if search_state is None or not search_state.rooms:
        return None, {}

    if obs_map is None:
        return None, {}

    free_map = obs_map > 0
    room_best_cells = {}
    room_candidate_count = {}
    for comp in kept_components or []:
        room = comp.get("room")
        if room is None:
            continue
        room_candidate_count[room] = room_candidate_count.get(room, 0) + 1
        prev = room_best_cells.get(room)
        if prev is None or float(comp.get("quality", 0.0)) > prev[1]:
            room_best_cells[room] = (list(comp["centroid"]), float(comp.get("quality", 0.0)))

    eligible_rooms = [name for name in search_state.rooms.keys() if room_candidate_count.get(name, 0) > 0]
    if not eligible_rooms:
        eligible_rooms = list(search_state.rooms.keys())

    bfs_dist = _compute_bfs_cost_map(robot_pos, free_map)
    finite_dists = bfs_dist[bfs_dist >= 0]
    max_cost = float(finite_dists.max()) if finite_dists.size > 0 else 1.0
    max_cost = max(max_cost, 1.0)

    room_scores = {}
    for room_name in eligible_rooms:
        rs = search_state.rooms[room_name]
        if room_name in room_best_cells:
            rep_cell = room_best_cells[room_name][0]
        else:
            rep_cell = [rs.centroid[0], rs.centroid[1]]
        rep_cell = _snap_to_nearest_free_cell(rep_cell, free_map)

        cost_cells = float(bfs_dist[rep_cell[0], rep_cell[1]])
        if cost_cells < 0:
            cost_score = 1.0
        else:
            cost_score = min(1.0, cost_cells / max_cost)

        failed_attempts = max(0, rs.candidates_tried - rs.candidates_confirmed)
        penalty_score = min(1.0, failed_attempts / 3.0)

        # Proxy for unexplored-ness until Fase F adds actual coverage tracking.
        exploration_load = min(1.0, 0.35 * rs.times_visited + 0.20 * failed_attempts)
        unexplored_score = 1.0 - exploration_load

        prior_score = float(rs.target_relevance)
        evidence_score = float(heatmap_evidence.get(room_name, 0.0))
        total = (
            w_prior * prior_score
            + w_evidence * evidence_score
            + w_unexplored * unexplored_score
            - w_cost * cost_score
            - w_penalty * penalty_score
        )
        if room_name == current_room:
            total += 0.05

        room_scores[room_name] = {
            "score": total,
            "prior": prior_score,
            "evidence": evidence_score,
            "unexplored": unexplored_score,
            "cost": cost_score,
            "penalty": penalty_score,
            "candidates": room_candidate_count.get(room_name, 0),
            "rep_cell": rep_cell,
        }

    if not room_scores:
        return None, {}

    ranked = sorted(room_scores.items(), key=lambda kv: kv[1]["score"], reverse=True)
    print("  [room-select] Top rooms:")
    for room_name, info in ranked[:5]:
        print(
            f"    {room_name}: score={info['score']:.3f} "
            f"(prior={info['prior']:.2f}, ev={info['evidence']:.2f}, "
            f"unexp={info['unexplored']:.2f}, cost={info['cost']:.2f}, "
            f"pen={info['penalty']:.2f}, cand={info['candidates']})"
        )

    return ranked[0][0], room_scores


def select_best_candidate(
    kept_components: list,
    current_room: str,
    query_priors: dict,
    tried_centroids: set,
    *,
    robot_pos=None,
    enable_room_gate: bool = True,
    prefer_nearest_only: bool = False,
    local_min_quality: float = 0.25,
    switch_margin: float = 0.40,
    same_room_bonus: float = 0.35,
    proximity_weight: float = 0.3,
) -> dict:
    """Task 1 — current-compatible-room-first candidate selection.

    Policy:
      1. If the current room instance belongs to a semantically compatible room
         type (e.g. bathroom.001 → bathroom), and there are untried local
         candidates with acceptable quality, prefer them.
      2. Only allow switching to an external room if the best external candidate
         clearly outscores the best local one (margin > switch_margin).
      3. Falls back to global best if current room is not compatible or has no
         viable local candidates.
      4. Among candidates with similar quality, prefer the nearest one to the
         robot (proximity_weight controls the trade-off).

    Args:
        kept_components:  list of component dicts (quality-sorted, best first).
        current_room:     room instance the robot is currently in.
        query_priors:     {room: normalised_prior} from Phase C.
        tried_centroids:  set of (int_r, int_c) already inspected this query.
        robot_pos:        (row, col) current robot position on map; if provided,
                          nearer candidates are boosted.
        local_min_quality: minimum quality for a local candidate to be accepted.
        switch_margin:    external must beat local by this fraction to trigger switch.
        same_room_bonus:  additive quality bonus applied to local candidates.
        proximity_weight: weight for the proximity bonus (0 = ignore distance).

    Returns:
        The selected component dict.
    """
    from vlmaps.utils.room_priors import canonical_room_type, compatible_room_types

    if not kept_components:
        return None

    # ── Proximity-adjusted ranking ──────────────────────────────────────────
    # Compute effective_quality = quality + proximity_weight * (1 - norm_dist)
    # so that among similar-quality candidates the nearest one wins.
    if robot_pos is not None:
        _rr, _rc = float(robot_pos[0]), float(robot_pos[1])
        _dists = []
        for c in kept_components:
            cr, cc = c["centroid"]
            _dists.append(float(np.sqrt((cr - _rr)**2 + (cc - _rc)**2)))
        for c, d in zip(kept_components, _dists):
            c["_distance_to_robot"] = d

        if proximity_weight > 0:
            _max_dist = max(_dists) if _dists else 1.0
            _max_dist = max(_max_dist, 1.0)  # avoid div-by-zero
            for c, d in zip(kept_components, _dists):
                c["_proximity_bonus"] = proximity_weight * (1.0 - d / _max_dist)
                c["_effective_quality"] = c["quality"] + c["_proximity_bonus"]
            kept_components = sorted(kept_components, key=lambda x: -x["_effective_quality"])
            print(f"  [candidate] Proximity ranking (top 3): "
                  + ", ".join(
                      f"q={c['quality']:.3f}+prox={c.get('_proximity_bonus',0):.3f}"
                      f"→{c.get('_effective_quality', c['quality']):.3f} "
                      f"d={c.get('_distance_to_robot', 0):.0f}"
                      for c in kept_components[:3]
                  ))

    # Separate tried vs untried
    def _is_tried(comp):
        cr, cc = comp["centroid"]
        return (int(cr), int(cc)) in tried_centroids

    untried = [c for c in kept_components if not _is_tried(c)]
    if not untried:
        # All tried — fall back to global best untried (might be empty)
        return kept_components[0]

    if prefer_nearest_only:
        if robot_pos is None:
            return untried[0]
        nearest = sorted(
            untried,
            key=lambda c: (
                c.get("_distance_to_robot", float("inf")),
                -c.get("_effective_quality", c["quality"]),
            ),
        )[0]
        print(f"  [candidate] Direct furniture mode — picking nearest candidate "
              f"(d={nearest.get('_distance_to_robot', float('nan')):.1f}, "
              f"q={nearest.get('_effective_quality', nearest['quality']):.3f})")
        return nearest

    if not enable_room_gate or not current_room or not query_priors:
        return untried[0]

    current_type = canonical_room_type(current_room)
    compat_types = compatible_room_types(query_priors)
    current_compatible = current_type in compat_types

    print(f"  [room-gate] Instance: {current_room}  type: {current_type}")
    print(f"  [room-gate] Compatible types for query: {sorted(compat_types)}")
    print(f"  [room-gate] Current room compatible: {current_compatible}")

    if not current_compatible:
        print(f"  [room-gate] Switch allowed: True (current room not compatible)")
        return untried[0]

    # Separate local (same instance) vs external untried candidates
    local_untried = [c for c in untried if c.get("room") == current_room]
    external_untried = [c for c in untried if c.get("room") != current_room]

    print(f"  [room-gate] Local untried: {len(local_untried)}  external untried: {len(external_untried)}")

    if not local_untried:
        reason = "local room exhausted" if any(c.get("room") == current_room for c in kept_components) else "no local candidates"
        print(f"  [room-gate] Switch allowed: True ({reason})")
        return untried[0]

    best_local = local_untried[0]
    best_ext = external_untried[0] if external_untried else None

    # Apply same-room bonus to local score
    _local_q = best_local.get("_effective_quality", best_local["quality"])
    local_effective = _local_q + same_room_bonus
    _ext_q = best_ext.get("_effective_quality", best_ext["quality"]) if best_ext else 0.0

    print(f"  [room-gate] Best local quality: {_local_q:.3f} "
          f"(+bonus → {local_effective:.3f})")
    if best_ext:
        print(f"  [room-gate] Best external quality: {_ext_q:.3f}")

    # Gate 1: local candidate too weak even with bonus
    if best_local["quality"] < local_min_quality:
        if best_ext and _ext_q > local_effective:
            print(f"  [room-gate] Switch allowed: True (local quality {best_local['quality']:.3f} < threshold {local_min_quality})")
            return untried[0]

    # Gate 2: external must clearly beat boosted local to trigger switch
    if best_ext and _ext_q > local_effective * (1.0 + switch_margin):
        print(f"  [room-gate] Switch allowed: True "
              f"(external {_ext_q:.3f} >> local {local_effective:.3f})")
        return untried[0]

    print(f"  [room-gate] Switch allowed: False "
          f"— staying in compatible room, inspecting local candidate first")
    return best_local


def _compute_path_safety(path_cells: list, obs_map: np.ndarray) -> dict:
    """Compute safety metrics for a planned path.

    Returns a dict with: path_length, min_clearance, mean_clearance,
    safety_penalty (sum of 1/clearance for risky cells), safe_cost.
    """
    dense_path = densify_path_cells(path_cells)
    if not dense_path:
        return {"path_length": 0, "min_clearance": 0.0,
                "mean_clearance": 0.0, "safety_penalty": 0.0, "safe_cost": 0.0}

    dist = distance_transform_edt(obs_map)
    clearances = []
    penalty = 0.0
    _PENALTY_RADIUS = 8.0   # cells within this radius incur cost (0.4 m)
    _LAMBDA = 0.5            # weight of clearance penalty vs path length

    for cell in dense_path:
        r, c = int(cell[0]), int(cell[1])
        if 0 <= r < dist.shape[0] and 0 <= c < dist.shape[1]:
            cl = float(dist[r, c])
        else:
            cl = 0.0
        clearances.append(cl)
        if cl < _PENALTY_RADIUS:
            penalty += (_PENALTY_RADIUS - cl) / _PENALTY_RADIUS

    n = len(dense_path)
    min_cl = float(min(clearances)) if clearances else 0.0
    mean_cl = float(np.mean(clearances)) if clearances else 0.0
    safe_cost = n + _LAMBDA * penalty

    return {
        "path_length": n,
        "min_clearance": min_cl,
        "mean_clearance": mean_cl,
        "safety_penalty": penalty,
        "safe_cost": safe_cost,
    }


def select_safe_goal_from_path(
    path_cells: list,
    heatmap: np.ndarray,
    obs_map: np.ndarray,
    room_provider=None,
    required_room: str = None,
    min_dist_cells: float = 10.0,
    max_dist_cells: float = 24.0,
    clearance_cells: float = 5.0,   # Task 2: raised from 3.0 → 5.0 (25 cm)
) -> tuple:
    """Walk planned path backward to find the best final viewing position.

    The heatmap argmax is used as the estimated object position.  The function
    scans the path from end to start and returns the last cell that is:
      - between min_dist_cells and max_dist_cells from the object centroid, and
      - at least clearance_cells away from the nearest obstacle.

    Grid units: cs=0.05 m → 10 cells ≈ 0.5 m, 24 cells ≈ 1.2 m.

    Args:
        path_cells:      List of [row, col] grid cells from last_planned_path.
        heatmap:         float32 (gs, gs) — current semantic heatmap.
        obs_map:         uint8/bool (gs, gs) — 1=free, 0=obstacle.
        min_dist_cells:  Minimum cells from object centroid (default 0.5 m).
        max_dist_cells:  Maximum cells from object centroid (default 1.2 m).
        clearance_cells: Minimum cells from any obstacle (default 0.15 m).

    Returns:
        best_goal:      [row, col] — selected navigation goal.
        obj_centroid:   [row, col] — heatmap argmax (use as face-toward target).
    """
    obj_row, obj_col = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    obj_centroid = [int(obj_row), int(obj_col)]

    dense_path = densify_path_cells(path_cells)
    if not dense_path:
        return obj_centroid, obj_centroid

    dist_to_obs = distance_transform_edt(obs_map)

    # Task 2: collect ALL valid candidates, then pick the one with best clearance
    # (not just the first one found when walking backward).
    candidates = []
    for cell in dense_path:
        row, col = int(cell[0]), int(cell[1])
        if required_room and room_provider is not None:
            _room_here = room_provider.get_room_at_cell(row, col)
            if _room_here != required_room:
                continue
        dr = row - obj_row
        dc = col - obj_col
        dist = float(np.sqrt(dr * dr + dc * dc))
        if 0 <= row < obs_map.shape[0] and 0 <= col < obs_map.shape[1]:
            clearance = float(dist_to_obs[row, col])
        else:
            clearance = 0.0
        if min_dist_cells <= dist <= max_dist_cells and clearance >= clearance_cells:
            candidates.append(([row, col], clearance))

    if candidates:
        # Pick the candidate with maximum clearance from obstacles
        best = max(candidates, key=lambda x: x[1])
        return best[0], obj_centroid

    # Soft fallback: relax clearance requirement, still pick best clearance
    soft_candidates = []
    for cell in dense_path:
        row, col = int(cell[0]), int(cell[1])
        if required_room and room_provider is not None:
            _room_here = room_provider.get_room_at_cell(row, col)
            if _room_here != required_room:
                continue
        dr = row - obj_row
        dc = col - obj_col
        dist = float(np.sqrt(dr * dr + dc * dc))
        if 0 <= row < obs_map.shape[0] and 0 <= col < obs_map.shape[1]:
            clearance = float(dist_to_obs[row, col])
        else:
            clearance = 0.0
        if min_dist_cells <= dist <= max_dist_cells:
            soft_candidates.append(([row, col], clearance))

    if soft_candidates:
        best = max(soft_candidates, key=lambda x: x[1])
        return best[0], obj_centroid

    # Last resort: path end
    last = dense_path[-1]
    return [int(last[0]), int(last[1])], obj_centroid


def find_safe_candidate_approach_goals(
    component_centroid,
    safe_obs_map: np.ndarray,
    robot_pos=None,
    room_provider=None,
    required_room: str = None,
    min_dist: float = 8.0,
    max_dist: float = 25.0,
    min_clearance: float = 3.0,
    n_angles: int = 16,
    top_k: int = 5,
) -> list:
    """Generate safe approach positions around a heatmap component centroid.

    Samples candidate approach cells in an annulus (min_dist..max_dist cells)
    around *component_centroid*, filters by the dilated safe_obs_map and minimum
    clearance, then ranks by (clearance DESC, proximity to robot ASC).

    This replaces the raw centroid → plan_path_only → select_safe_goal_from_path
    pipeline for the initial goal selection.  Because the candidates are validated
    against the same map the visgraph uses, goal-in-obstacle crashes disappear.

    Parameters
    ----------
    component_centroid : (row, col) of the heatmap component peak.
    safe_obs_map       : uint8 ndarray, 1=free 0=obstacle (dilated, full-map).
    robot_pos          : optional (row, col) — used for tie-breaking by proximity.
    min_dist           : annulus inner radius in cells (default 8 = 0.4 m).
    max_dist           : annulus outer radius in cells (default 25 = 1.25 m).
    min_clearance      : minimum distance-to-obstacle in cells (default 3).
    n_angles           : number of angle samples around the annulus (default 16).
    top_k              : maximum candidates to return.

    Returns
    -------
    List of [row, col] goals, sorted best-first.  Empty list if none found.
    """
    cr, cc = float(component_centroid[0]), float(component_centroid[1])
    h, w   = safe_obs_map.shape

    dist_map = distance_transform_edt(safe_obs_map)

    candidates = []
    # Sample at several radii within the annulus for better coverage
    radii  = np.linspace(min_dist, max_dist, num=max(3, int((max_dist - min_dist) / 4) + 1))
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    for r in radii:
        for ang in angles:
            row = int(round(cr + r * np.sin(ang)))
            col = int(round(cc + r * np.cos(ang)))
            if not (0 <= row < h and 0 <= col < w):
                continue
            if safe_obs_map[row, col] == 0:
                continue
            if required_room and room_provider is not None:
                _room_here = room_provider.get_room_at_cell(row, col)
                if _room_here != required_room:
                    continue
            cl = float(dist_map[row, col])
            if cl < min_clearance:
                continue
            if not _segment_is_free_on_map([row, col], [cr, cc], safe_obs_map):
                continue
            candidates.append((row, col, cl))

    if not candidates:
        return []

    # Deduplicate (keep highest-clearance per grid cell)
    best_by_cell: dict = {}
    for row, col, cl in candidates:
        key = (row, col)
        if key not in best_by_cell or cl > best_by_cell[key]:
            best_by_cell[key] = cl

    # Sort: clearance DESC, then proximity to robot ASC
    def _sort_key(item):
        (row, col), cl = item
        prox = 0.0
        if robot_pos is not None:
            dr = row - robot_pos[0]
            dc = col - robot_pos[1]
            prox = float(dr * dr + dc * dc)
        return (-cl, prox)

    sorted_cells = sorted(best_by_cell.items(), key=_sort_key)
    return [[r, c] for (r, c), _ in sorted_cells[:top_k]]


def find_reachable_room_goal(
    target_room: str,
    room_provider,
    obs_map: np.ndarray,
    top_k: int = 8,
    min_clearance: float = 3.0,
) -> list:
    """Return up to top_k safe navigable cells inside target_room.

    Uses the room provider's region_grid to enumerate all cells that
    belong to the target room, intersects with the free-space obstacle map,
    then ranks by distance-to-obstacle (clearance) and returns the top-K.

    This replaces raw geometric centroid navigation for room commands so
    the planner gets a goal that is actually inside the room AND safe.

    Returns:
        List of [row, col] goals sorted by clearance descending.
        Empty list if the target room is not found or has no navigable cells.
    """
    import re as _re_local

    if room_provider is None or not room_provider.is_available():
        return []

    region_grid = getattr(room_provider, "_region_grid", None)
    regions     = getattr(room_provider, "_regions", [])

    if region_grid is None or not regions:
        return []

    # Find matching region id, preferring the exact resolved room instance.
    resolved_target = room_provider.resolve_room_name(target_room) or target_room
    query   = resolved_target.lower().strip()
    pattern = _re_local.compile(r'\b' + _re_local.escape(query) + r'\b')
    target_rid = None
    for reg in regions:
        label = reg["label"].lower()
        if query == label:
            target_rid = reg["id"]
            break
        if target_rid is None and pattern.search(label):
            target_rid = reg["id"]

    if target_rid is None:
        print(f"  [room-goal] Room '{resolved_target}' not found in region grid")
        return []

    # Navigable cells inside the target room
    room_mask       = (region_grid == target_rid)
    free_mask       = (obs_map > 0)
    navigable_room  = room_mask & free_mask

    if not navigable_room.any():
        print(f"  [room-goal] No navigable cells in room '{resolved_target}'")
        return []

    dist_map = distance_transform_edt(obs_map)
    # Room-interior depth: how far each cell is from the room boundary.
    # This prevents picking goals right at the room edge where the robot
    # might end up classified in the adjacent room.
    room_depth_map = distance_transform_edt(room_mask)
    rows, cols = np.where(navigable_room)
    clearances = dist_map[rows, cols]
    room_depths = room_depth_map[rows, cols]

    # Combined score: obstacle clearance + room depth bonus.
    # Room depth is weighted so goals deep inside the room are preferred
    # over boundary cells, even if boundary cells have slightly higher clearance.
    _ROOM_DEPTH_WEIGHT = 0.5
    _MIN_ROOM_DEPTH = 2.0
    combined = clearances + _ROOM_DEPTH_WEIGHT * room_depths

    sorted_idx = np.argsort(-combined)
    preferred_idx = [
        int(i) for i in sorted_idx
        if float(clearances[i]) >= float(min_clearance) and float(room_depths[i]) >= _MIN_ROOM_DEPTH
    ]
    if preferred_idx:
        ordered_idx = preferred_idx
    else:
        # If the room rasterization is very thin and no cell reaches the target
        # room-depth threshold, fall back to the deepest cells first instead of
        # immediately picking the highest-clearance boundary cell.
        fallback_score = room_depths * 10.0 + clearances
        ordered_idx = list(np.argsort(-fallback_score))
        print(f"  [room-goal] Warning: no room cells reach depth >= {_MIN_ROOM_DEPTH:.1f}; "
              f"falling back to deepest available cells")

    candidates = []
    for i in ordered_idx:
        r, c = int(rows[i]), int(cols[i])
        candidates.append([r, c])
        if len(candidates) >= top_k:
            break

    best_cl = float(clearances[sorted_idx[0]]) if len(sorted_idx) > 0 else 0.0
    best_rd = float(room_depths[sorted_idx[0]]) if len(sorted_idx) > 0 else 0.0
    print(f"  [room-goal] Found {len(candidates)} safe goal(s) in '{resolved_target}' "
          f"(best clearance: {best_cl:.1f} cells, room depth: {best_rd:.1f} cells)")
    return candidates


def navigate_to_room_stage(
    robot,
    room_name: str,
    room_provider,
    rgb_map_2d: np.ndarray,
    obs_map: np.ndarray,
    *,
    label_prefix: str = "Room stage",
) -> Tuple[bool, Optional[str]]:
    """Move to a safe interior goal of *room_name* before candidate inspection."""
    if room_provider is None or not room_provider.is_available():
        return False, None

    safe_goals = find_reachable_room_goal(room_name, room_provider, obs_map)
    if not safe_goals:
        print(f"  [room-stage] No safe interior goals found for '{room_name}'.")
        return False, None

    stage_goal = safe_goals[0]
    _, stage_actions = robot.plan_path_only(stage_goal)
    stage_polyline = normalize_path_cells(getattr(robot, "last_planned_path", None) or [])
    stage_cells = densify_path_cells(stage_polyline)
    stage_steps = len(stage_actions)

    print(f"  [room-stage] Moving to room '{room_name}' via {stage_goal} "
          f"({len(stage_polyline)} waypoint(s), {len(stage_cells)} dense cell(s), "
          f"preview={stage_steps} actions)")
    show_map(robot, rgb_map_2d, path_cells=stage_cells,
             label=f"{label_prefix}: {room_name}")
    ui_wait(300)

    if stage_steps == 0:
        stage_ok = True
    else:
        stage_ok = execute_nav_replay(
            robot,
            stage_actions,
            room_name,
            rgb_map_2d,
            None,
            stage_polyline,
            display_path_cells=stage_cells,
            dist_map=distance_transform_edt(robot.map.obstacles_map),
            goal_reached_tol_cells=1.0,
        )

    robot._set_nav_curr_pose()
    arrived_room = room_provider.get_room_at_cell(
        int(robot.curr_pos_on_map[0]), int(robot.curr_pos_on_map[1])
    )
    print(f"  [room-stage] Actual room after staging: {arrived_room or 'unknown'}")
    return stage_ok, arrived_room


def _room_instance_matches(actual_room, target_room: str) -> bool:
    """Backward-compatible shim for exact/base room-command matching."""
    return room_command_matches(actual_room, target_room)


def _resolve_instruction_room_targets(instruction: str, categories: list, room_provider) -> list:
    """Resolve room-instance aliases after LLM parsing.

    This preserves explicit mentions such as "bedroom 1" even if the parser
    collapses them to the base family "bedroom". For non-room targets, the
    original category is preserved.
    """
    if room_provider is None or not room_provider.is_available():
        return categories

    resolved = []
    explicit_mentions = room_provider.find_room_mentions(instruction)
    mention_cursor = 0

    for cat in categories:
        raw = cat.strip()
        direct_room = room_provider.resolve_room_name(raw)
        if direct_room is not None:
            if direct_room != raw:
                print(f"  [room-parse] Resolved room target '{raw}' -> '{direct_room}'")
            resolved.append(direct_room)
            continue

        upgraded = raw
        for i in range(mention_cursor, len(explicit_mentions)):
            exact_room = explicit_mentions[i]
            if room_command_matches(exact_room, raw):
                upgraded = exact_room
                mention_cursor = i + 1
                if upgraded != raw:
                    print(f"  [room-parse] Preserved explicit room instance '{raw}' -> '{upgraded}'")
                break
        resolved.append(upgraded)
    return resolved


@dataclass(frozen=True)
class ResolvedTargetPlan:
    original_target: str
    canonical_target: str
    effective_target: str
    surrogate_categories: List[str]
    likely_rooms: List[str]
    resolution_source: str
    room_goal: Optional[list] = None


def build_resolved_target_plans(
    categories: list,
    *,
    room_provider,
    room_regions,
    available_categories: list,
    present_categories: list,
) -> List[ResolvedTargetPlan]:
    """Resolve parsed targets into room goals or open-vocabulary object plans."""
    known_rooms = room_provider.list_rooms() if room_provider and room_provider.is_available() else []
    plans: List[ResolvedTargetPlan] = []
    for cat in categories:
        raw = str(cat or "").strip()
        if not raw:
            continue

        room_goal = None
        if room_provider and room_provider.is_available():
            room_goal = room_provider.get_room_centroid(raw)
        if room_goal is None and room_regions:
            room_goal = find_room_goal(raw, room_regions)

        if room_goal is not None:
            plans.append(
                ResolvedTargetPlan(
                    original_target=raw,
                    canonical_target=raw,
                    effective_target=raw,
                    surrogate_categories=[],
                    likely_rooms=[raw],
                    resolution_source="room_command",
                    room_goal=list(room_goal),
                )
            )
            continue

        resolution: OpenVocabTargetResolution = resolve_open_vocab_target(
            raw,
            available_categories,
            known_rooms=known_rooms,
            present_categories=present_categories,
        )
        plans.append(
            ResolvedTargetPlan(
                original_target=resolution.original_target,
                canonical_target=resolution.canonical_target,
                effective_target=resolution.effective_target,
                surrogate_categories=list(resolution.surrogate_categories),
                likely_rooms=list(resolution.likely_rooms),
                resolution_source=resolution.source,
                room_goal=None,
            )
        )
    return plans


def log_resolved_target_plans(plans: List[ResolvedTargetPlan]) -> None:
    """Print the resolved navigation targets in a compact, audit-friendly format."""
    pretty = []
    for plan in plans:
        if plan.room_goal is not None:
            pretty.append(f"{plan.original_target} [room]")
            continue
        if (
            plan.original_target == plan.canonical_target
            and plan.effective_target == plan.canonical_target
            and not plan.surrogate_categories
        ):
            pretty.append(plan.canonical_target)
            continue
        desc = (
            f"{plan.original_target} -> canonical={plan.canonical_target}, "
            f"heatmap={plan.effective_target}, source={plan.resolution_source}"
        )
        if plan.surrogate_categories:
            desc += f", surrogates={plan.surrogate_categories}"
        if plan.likely_rooms:
            desc += f", likely_rooms={plan.likely_rooms}"
        pretty.append(desc)
    print("Targets:")
    for line in pretty:
        print(f"  - {line}")


def find_best_start_pose(robot):
    """
    Scan ~30 evenly-spaced trajectory poses and return the one whose 2-D map
    cell is deepest inside free space (farthest from any obstacle).
    This avoids starting on top of furniture, which would make every
    move_to_object call return 0 actions (already at goal).
    """
    obs_map = robot.map.obstacles_map
    dist_map = distance_transform_edt(obs_map)

    poses = robot.vlmaps_dataloader.base_poses
    n = len(poses)
    step = max(1, n // 30)

    best_idx = 0
    best_dist = -1.0
    gs = obs_map.shape[0]

    for i in range(0, n, step):
        tf = cvt_pose_vec2tf(poses[i])
        robot.set_agent_state(tf)
        robot._set_nav_curr_pose()
        row = int(robot.curr_pos_on_map[0])
        col = int(robot.curr_pos_on_map[1])
        if 0 <= row < gs and 0 <= col < gs and obs_map[row, col]:
            d = float(dist_map[row, col])
            if d > best_dist:
                best_dist = d
                best_idx = i

    print(f"Best start: pose[{best_idx}/{n}]  map=({int(robot.curr_pos_on_map[0])},{int(robot.curr_pos_on_map[1])})  dist_to_obstacle={best_dist:.1f} cells")
    return cvt_pose_vec2tf(poses[best_idx])


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="object_goal_navigation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    global _frozen_detection_bgr, _frozen_target_cell
    # ── Setup ────────────────────────────────────────────────────────────────
    robot = HabitatLanguageRobot(config)
    robot.setup_scene(config.scene_id)
    _dataset_type = str(getattr(config, "dataset_type", "mp3d"))
    _scene_categories = get_categories(_dataset_type)
    robot.map.init_categories(_scene_categories)

    print("\nBuilding top-down RGB map...")
    rgb_map_2d = build_rgb_map_2d(robot)

    # ── Load pre-labelled room map (if available) ─────────────────────────
    scene_dir = robot.vlmaps_data_save_dirs[config.scene_id]
    _room_data = load_room_map(scene_dir)
    if _room_data is not None:
        _room_map, _room_categories, _room_regions = _room_data
        print(f"Room map loaded: {list(_room_regions.keys())}")
    else:
        _room_map, _room_categories, _room_regions = None, [], {}
        print("No room map found. Run build_room_map.py to enable region-aware navigation.")

    print("\nSearching for a good starting position...")
    start_tf = find_best_start_pose(robot)
    robot.set_agent_state(start_tf)
    robot._set_nav_curr_pose()

    show_obs(robot, "Ready")
    show_map(robot, rgb_map_2d, heatmap_2d=None, label="Ready")
    print("Scene:", robot.vlmaps_data_save_dirs[config.scene_id].name)

    _room_provider = getattr(robot, "room_provider", None)
    if _room_provider and _room_provider.is_available():
        print(f"Room provider active. Rooms: {_room_provider.list_rooms()}")
    print(f"Heatmap mode: {get_runtime_heatmap_mode()}")

    # Determine which categories actually have signal in the current scene using
    # the same filter as compute_heatmap: voxel must win argmax AND score > 0.3.
    # Computed once at startup (scores_mat doesn't change during the session).
    _STRUCTURAL = {"void", "wall", "floor", "ceiling"}
    _SCORE_THRESH = 0.3   # must match compute_heatmap's score_thresh
    _MIN_VOXELS = 10      # minimum winning voxels to consider a category present
    def _build_present_categories() -> list:
        from vlmaps.utils.index_utils import find_similar_category_id as _fsid
        cats = getattr(robot.map, "categories", [])
        scores_mat = getattr(robot.map, "scores_mat", None)
        if scores_mat is None:
            return [c for c in cats if c not in _STRUCTURAL]
        max_ids = np.argmax(scores_mat, axis=1)
        present = []
        for i, c in enumerate(cats):
            if c in _STRUCTURAL:
                continue
            mask = (max_ids == i) & (scores_mat[:, i] > _SCORE_THRESH)
            if int(mask.sum()) >= _MIN_VOXELS:
                present.append(c)
        return present

    _present_categories = _build_present_categories()

    # ── Instruction loop ─────────────────────────────────────────────────────
    while True:
        rooms_hint = ""
        if _room_provider and _room_provider.is_available():
            rooms_hint = f"  Rooms     : {_room_provider.list_rooms()}\n"
        print(
            f"\n{'─' * 50}\n"
            f"{rooms_hint}"
            f"  Objects   : {_present_categories}"
        )
        instruction = input("Enter navigation instruction (or 'quit'): ").strip()

        if instruction.lower() in ("quit", "exit", "q"):
            break
        if not instruction:
            continue

        print("Parsing instruction...")
        try:
            categories = parse_object_goal_instruction(instruction)
        except Exception as e:
            print(f"LLM error: {e}")
            continue

        categories = _resolve_instruction_room_targets(
            instruction, categories, _room_provider
        )

        target_plans = build_resolved_target_plans(
            categories,
            room_provider=_room_provider,
            room_regions=_room_regions,
            available_categories=_scene_categories,
            present_categories=_present_categories,
        )
        log_resolved_target_plans(target_plans)

        robot.set_agent_state(start_tf)
        robot._set_nav_curr_pose()
        robot.empty_recorded_actions()
        show_obs(robot, "Start")
        show_map(robot, rgb_map_2d, label="Start")

        # ── Phase B+C: build per-room search state and compute object→room priors ──
        _search_states: dict = {}  # original_target → SearchState
        for _plan in target_plans:
            _ss = SearchState(
                _plan.canonical_target,
                _room_provider,
                robot.map.obstacles_map,
                original_target=_plan.original_target,
                effective_target=_plan.effective_target,
                surrogate_categories=_plan.surrogate_categories,
                likely_rooms=_plan.likely_rooms,
                resolution_source=_plan.resolution_source,
            )
            if _ss.rooms:
                # Phase C: compute priors (LLM + manual table + scene evidence)
                _priors = _ss.compute_priors()
                if _priors:
                    _sorted = sorted(_priors.items(), key=lambda x: x[1], reverse=True)
                    _prior_str = ", ".join(f"{r}={v:.2f}" for r, v in _sorted if v > 0.01)
                    print(f"  Room priors for '{_plan.canonical_target}': {_prior_str}")
            _search_states[_plan.original_target] = _ss

        for _plan in target_plans:
            _raw_cat = _plan.original_target.strip()
            if not _raw_cat:
                continue
            _frozen_detection_bgr = None  # clear previous detection freeze
            _frozen_target_cell = None
            _ss = _search_states.get(_plan.original_target)
            _yoloe_target = _plan.canonical_target
            _heatmap_target = _plan.effective_target
            print(f"\nPlanning path to: {_raw_cat}")
            if _plan.room_goal is None and (
                _plan.original_target != _plan.canonical_target
                or _plan.effective_target != _plan.canonical_target
                or _plan.surrogate_categories
            ):
                print(
                    f"  [open-vocab] original='{_plan.original_target}' "
                    f"canonical='{_plan.canonical_target}' heatmap='{_plan.effective_target}' "
                    f"source={_plan.resolution_source}"
                )
                if _plan.surrogate_categories:
                    print(f"  [open-vocab] validated surrogates: {_plan.surrogate_categories}")
                if _plan.likely_rooms:
                    print(f"  [open-vocab] likely rooms: {_plan.likely_rooms}")

            # ── Check if target is a room name (room-level navigation) ────────
            room_goal = list(_plan.room_goal) if _plan.room_goal is not None else None

            # Track current room in SearchState
            robot._set_nav_curr_pose()
            current_room = None
            if _room_provider and _room_provider.is_available():
                current_room = _room_provider.get_room_at_cell(
                    int(robot.curr_pos_on_map[0]), int(robot.curr_pos_on_map[1])
                )
            if _ss:
                _ss.update_current_room(current_room)

            _room_safe_goals: list = []   # Bug 1: top-K safe goals for retry

            if room_goal is not None:
                # Bug 1: replace raw centroid with safe reachable room goals
                print(f"  Requested room command: {_raw_cat}")
                print(f"  Current room: {current_room or 'unknown'}")
                raw_centroid = list(room_goal)
                print(f"  Raw centroid: {raw_centroid}")
                _rc_r, _rc_c = int(raw_centroid[0]), int(raw_centroid[1])
                _raw_nav = bool(robot.map.obstacles_map[_rc_r, _rc_c]) if (
                    0 <= _rc_r < robot.map.obstacles_map.shape[0] and
                    0 <= _rc_c < robot.map.obstacles_map.shape[1]
                ) else False
                print(f"  Raw centroid navigable: {_raw_nav}")

                # Use the dilated safe map (same map the visgraph was built from)
                # so room goals are guaranteed navigable by the planner.
                _safe_map_for_rooms = getattr(robot, "_safe_obs_map", robot.map.obstacles_map)
                _room_safe_goals = find_reachable_room_goal(
                    _raw_cat, _room_provider, _safe_map_for_rooms
                )
                if _room_safe_goals:
                    goal_pos = _room_safe_goals[0]
                    print(f"  Selected safe room goal: {goal_pos}")
                else:
                    goal_pos = raw_centroid
                    print(f"  No safe interior goals found — falling back to raw centroid")

                heatmap = np.zeros((robot.map.gs, robot.map.gs), dtype=np.float32)
                kept_components = []
                _yoloe_session = None
            else:
                # Object-level: compute heatmap and prepare YOLOE
                print(f"  Current room: {current_room or 'unknown'}")
                print("  Computing semantic heatmap...")
                heatmap, kept_components = compute_heatmap(robot, _heatmap_target)
                _heatmap_ev = {}
                _selected_room = None
                _present_set = {c.lower() for c in _present_categories}
                _direct_query_mode = (
                    len(target_plans) == 1 and _heatmap_target.lower() in _present_set
                    and _heatmap_target.lower() == _yoloe_target.lower()
                )

                # Annotate each component with its room
                if _room_provider and _room_provider.is_available():
                    rooms_count: dict = {}
                    for comp in kept_components:
                        cr, cc = comp["centroid"]
                        comp["room"] = _room_provider.get_room_at_cell(int(cr), int(cc))
                        r = comp["room"] or "unknown"
                        rooms_count[r] = rooms_count.get(r, 0) + 1
                    if rooms_count:
                        print(f"  Candidates by room: {rooms_count}")

                # Bug 2: re-compute room priors with heatmap evidence as
                # dominant signal (direct query — heatmap has signal).
                if _ss and kept_components:
                    from vlmaps.utils.room_priors import (
                        compute_room_priors as _crp,
                        compute_heatmap_room_evidence as _che,
                    )
                    _heatmap_ev = _che(kept_components)
                    print(f"  Query type: direct")
                    if _heatmap_ev:
                        _ev_sorted = sorted(_heatmap_ev.items(), key=lambda x: -x[1])
                        print(f"  Room evidence from heatmap: "
                              f"{ {r: round(v, 3) for r, v in _ev_sorted} }")
                    _known_rooms_list = list(_ss.rooms.keys())
                    _seen_objs = {n: rs.objects_seen for n, rs in _ss.rooms.items()}
                    _new_priors = _crp(
                        _yoloe_target, _known_rooms_list, _seen_objs,
                        heatmap_evidence=_heatmap_ev, query_type="direct",
                    )
                    for _rn, _rv in _new_priors.items():
                        if _rn in _ss.rooms:
                            _ss.rooms[_rn].target_relevance = _rv
                    _np_sorted = sorted(_new_priors.items(), key=lambda x: -x[1])
                    print(f"  Final room priors after evidence fusion: "
                          f"{ ', '.join(f'{r}={v:.2f}' for r, v in _np_sorted[:6] if v > 0.01) }")

                # ── Phase D: choose the room first, then inspect candidates inside it ──
                if _ss and kept_components and _ss.rooms and not _direct_query_mode:
                    robot._set_nav_curr_pose()
                    _robot_rc_room = [robot.curr_pos_on_map[0], robot.curr_pos_on_map[1]]
                    _selected_room, _room_scores = select_best_room(
                        _ss,
                        _robot_rc_room,
                        _heatmap_ev,
                        robot.map.obstacles_map,
                        kept_components,
                        current_room=current_room,
                    )
                    if _selected_room is not None:
                        _room_kept = [c for c in kept_components if c.get("room") == _selected_room]
                        if _room_kept:
                            print(f"  [room-select] Chosen room for '{_yoloe_target}': {_selected_room} "
                                  f"({len(_room_kept)}/{len(kept_components)} candidate(s))")
                            kept_components = _room_kept
                        else:
                            print(f"  [room-select] Chosen room '{_selected_room}' has no direct "
                                  f"candidates — falling back to global ranking")
                elif _direct_query_mode:
                    print(f"  [room-select] Direct furniture query for '{_yoloe_target}' "
                          f"— room selector disabled, nearest candidate policy active")

                if _selected_room and current_room != _selected_room:
                    _safe_map_for_rooms = getattr(robot, "_safe_obs_map", robot.map.obstacles_map)
                    _stage_ok, _stage_room = navigate_to_room_stage(
                        robot,
                        _selected_room,
                        _room_provider,
                        rgb_map_2d,
                        _safe_map_for_rooms,
                    )
                    current_room = _stage_room
                    if _ss:
                        _ss.update_current_room(current_room)
                    if not _stage_ok:
                        print(f"  [room-stage] Warning: staging move toward '{_selected_room}' "
                              f"did not complete cleanly; continuing with candidate approach.")

                show_map(robot, rgb_map_2d, heatmap_2d=heatmap, label=f"Planning: {_yoloe_target}")
                ui_wait(200)

                if not kept_components:
                    print(f"  [skip] No heatmap signal for '{_heatmap_target}' in this scene.")
                    continue

                from vlmaps.utils.yoloe_utils import get_session, shutdown_session, runtime_conf_thresh
                _yoloe_session = get_session(_yoloe_target, conf_thresh=runtime_conf_thresh(0.3))

            if room_goal is not None:
                # goal_pos already set to safe interior goal above (Bug 1 fix)
                obj_centroid = goal_pos
                _, planned_actions = robot.plan_path_only(goal_pos)
                print(f"  Goal room instance: {_room_provider.get_room_at_cell(int(goal_pos[0]), int(goal_pos[1])) if _room_provider and _room_provider.is_available() else 'unknown'}")
            else:
                # ── Task 1: current-compatible-room-first candidate selection ──
                robot._set_nav_curr_pose()
                _tried = getattr(_ss, "_tried_centroids", set()) if _ss else set()
                _query_priors = {r: rs.target_relevance for r, rs in _ss.rooms.items()} if _ss else {}

                _robot_rc_sel = [robot.curr_pos_on_map[0], robot.curr_pos_on_map[1]]
                best_comp = select_best_candidate(
                    kept_components,
                    current_room,
                    _query_priors,
                    _tried,
                    robot_pos=_robot_rc_sel,
                    enable_room_gate=not _direct_query_mode,
                    prefer_nearest_only=_direct_query_mode,
                )
                if best_comp is None:
                    print(f"  [skip] No viable candidate for '{_yoloe_target}'.")
                    continue

                _hc_r, _hc_c = best_comp["centroid"]
                obj_centroid = [int(_hc_r), int(_hc_c)]
                print(f"  Heatmap target: {obj_centroid}  (room: {best_comp.get('room', 'unknown')})")

                # Step 2: generate safe approach goals around the component
                # centroid using the dilated safe map (same as visgraph).
                # This replaces the raw-centroid → path → walk-back chain that
                # broke when the centroid was inside an obstacle cell.
                _safe_map = getattr(robot, "_safe_obs_map", robot.map.obstacles_map)
                robot._set_nav_curr_pose()
                _robot_rc = getattr(robot, "_nav_curr_pos", None)
                _approach_goals = find_safe_candidate_approach_goals(
                    obj_centroid, _safe_map,
                    robot_pos=_robot_rc,
                    room_provider=_room_provider,
                    required_room=best_comp.get("room"),
                    min_dist=8.0, max_dist=25.0, min_clearance=3.0,
                )
                if _approach_goals:
                    goal_pos = _approach_goals[0]
                    print(f"  Approach goal: {goal_pos}  "
                          f"(from {len(_approach_goals)} safe candidates)")
                else:
                    # Fallback: plan to centroid then walk path backward
                    print(f"  [approach] No annulus goals found — falling back to path walk")
                    _initial_path, _ = robot.plan_path_only(obj_centroid)
                    goal_pos, obj_centroid = select_safe_goal_from_path(
                        _initial_path, heatmap, robot.map.obstacles_map,
                        room_provider=_room_provider,
                        required_room=best_comp.get("room"),
                    )
                    print(f"  Fallback path-based goal: {goal_pos}")

                # Step 3: plan to the selected goal
                _, planned_actions = robot.plan_path_only(goal_pos)

            # Capture planned path both as geometric polyline and dense raster.
            path_polyline = normalize_path_cells(getattr(robot, "last_planned_path", None) or [])
            path_cells = densify_path_cells(path_polyline)

            n_actions = len(planned_actions)
            print(f"  Path computed: {len(path_polyline)} polyline waypoint(s), "
                  f"{len(path_cells)} dense path cell(s) "
                  f"(controller preview: {n_actions} actions).")

            # ── Safety metrics + Bug 3 execution gate ────────────────────────
            _MIN_GOAL_CLEARANCE = 3.0   # cells (~15 cm at cs=0.05 m)
            _MIN_PATH_CLEARANCE = 1.0   # cells — zero is definitely in obstacle

            _goal_cl = 0.0
            _safety  = {"path_length": 0, "min_clearance": 0.0,
                        "mean_clearance": 0.0, "safety_penalty": 0.0,
                        "safe_cost": 0.0}

            if path_cells and room_goal is None:
                _dist_map_safety = distance_transform_edt(robot.map.obstacles_map)
                _safety = _compute_path_safety(path_cells, robot.map.obstacles_map)
                if goal_pos and 0 <= int(goal_pos[0]) < robot.map.obstacles_map.shape[0]:
                    _goal_cl = float(_dist_map_safety[int(goal_pos[0]), int(goal_pos[1])])
                print(f"  [safety] length={_safety['path_length']} "
                      f"min_cl={_safety['min_clearance']:.1f} "
                      f"mean_cl={_safety['mean_clearance']:.1f} "
                      f"penalty={_safety['safety_penalty']:.1f} "
                      f"safe_cost={_safety['safe_cost']:.1f} "
                      f"goal_cl={_goal_cl:.1f}")

                # Bug 3: reject unsafe paths before execution
                _goal_unsafe = _goal_cl < _MIN_GOAL_CLEARANCE
                _path_unsafe = _safety["min_clearance"] < _MIN_PATH_CLEARANCE

                if _goal_unsafe or _path_unsafe:
                    _reasons = []
                    if _goal_unsafe:
                        _reasons.append(f"goal_cl={_goal_cl:.1f}<{_MIN_GOAL_CLEARANCE}")
                    if _path_unsafe:
                        _reasons.append(f"min_cl={_safety['min_clearance']:.1f}<{_MIN_PATH_CLEARANCE}")
                    print(f"  [safety] Unsafe path rejected before execution ({', '.join(_reasons)}).")

                    # Mark current centroid as tried; try up to 3 alternatives
                    _bc_r = int(best_comp["centroid"][0])
                    _bc_c = int(best_comp["centroid"][1])
                    _tried_set_now = getattr(_ss, "_tried_centroids", set()) if _ss else set()
                    _tried_set_now.add((_bc_r, _bc_c))
                    if _ss:
                        _ss._tried_centroids = _tried_set_now

                    _alt_untried = [
                        c for c in kept_components
                        if (int(c["centroid"][0]), int(c["centroid"][1])) not in _tried_set_now
                    ]

                    _safe_alt_found = False
                    for _nc in _alt_untried[:3]:
                        _nc_r, _nc_c = _nc["centroid"]
                        _nc_cen = [int(_nc_r), int(_nc_c)]
                        print(f"  [safety] Trying next candidate: {_nc_cen} "
                              f"(room: {_nc.get('room', 'unknown')})")
                        # Use approach-goals for alt candidates too
                        _nc_app = find_safe_candidate_approach_goals(
                            _nc_cen, _safe_map,
                            robot_pos=_robot_rc,
                            room_provider=_room_provider,
                            required_room=_nc.get("room"),
                            min_dist=8.0, max_dist=25.0, min_clearance=3.0,
                        )
                        if _nc_app:
                            _nc_goal = _nc_app[0]
                        else:
                            _nc_init, _ = robot.plan_path_only(_nc_cen)
                            _nc_goal, _nc_cen = select_safe_goal_from_path(
                                _nc_init, heatmap, robot.map.obstacles_map,
                                room_provider=_room_provider,
                                required_room=_nc.get("room"),
                            )
                        _, _nc_acts = robot.plan_path_only(_nc_goal)
                        _nc_polyline = normalize_path_cells(getattr(robot, "last_planned_path", None) or [])
                        _nc_path = densify_path_cells(_nc_polyline)
                        _nc_safety = _compute_path_safety(_nc_path, robot.map.obstacles_map)
                        _nc_gcl = 0.0
                        if _nc_goal and 0 <= int(_nc_goal[0]) < _dist_map_safety.shape[0]:
                            _nc_gcl = float(_dist_map_safety[int(_nc_goal[0]), int(_nc_goal[1])])
                        print(f"  [safety] Alt candidate: goal_cl={_nc_gcl:.1f} "
                              f"min_cl={_nc_safety['min_clearance']:.1f}")

                        if (_nc_gcl >= _MIN_GOAL_CLEARANCE and
                                _nc_safety["min_clearance"] >= _MIN_PATH_CLEARANCE):
                            goal_pos      = _nc_goal
                            obj_centroid  = _nc_cen
                            planned_actions = _nc_acts
                            path_polyline = _nc_polyline
                            path_cells    = _nc_path
                            n_actions     = len(_nc_acts)
                            best_comp     = _nc
                            _goal_cl      = _nc_gcl
                            print(f"  [safety] Safe alternative accepted: {goal_pos}")
                            _safe_alt_found = True
                            break
                        else:
                            _tried_set_now.add((int(_nc_r), int(_nc_c)))
                            if _ss:
                                _ss._tried_centroids = _tried_set_now

                    if not _safe_alt_found:
                        print(f"  [safety] No safe candidate found for '{_yoloe_target}' — skipping.")
                        if _yoloe_session is not None:
                            from vlmaps.utils.yoloe_utils import shutdown_session
                            shutdown_session()
                        continue

            # n_actions == 0 means robot is already at the goal — treat as arrived.
            already_at_goal = (n_actions == 0)


            # Show heatmap + planned path BEFORE executing so the user can see the route
            show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                     path_cells=path_cells, label=f"Path planned: {_yoloe_target}")
            ui_wait(800)

            # Precompute dist_map for the collision shield from the RAW (undilated)
            # obstacle map so doorways are not falsely flagged as low-clearance.
            # The dilated _safe_obs_map is only for planning/goal validation.
            _dist_map_shield = distance_transform_edt(robot.map.obstacles_map)

            if already_at_goal:
                print(f"  Already at goal for '{_yoloe_target}' — proceeding with verification.")
                completed = True
            else:
                print(f"  Executing path follower over {len(path_polyline)} polyline waypoint(s) "
                      f"/ {len(path_cells)} dense cell(s)…")
                # One-shot execution: no recovery or replanning on failure.
                completed = execute_nav_replay(
                    robot, planned_actions, _yoloe_target, rgb_map_2d, heatmap, path_polyline,
                    display_path_cells=path_cells,
                    dist_map=_dist_map_shield,
                    goal_reached_tol_cells=1.0 if room_goal is not None else None,
                )
                if not completed:
                    print(f"  [nav] Path execution stopped early for '{_yoloe_target}' "
                          f"(stuck or shield). Continuing from current position.")

            # ── Room-level navigation: one-shot arrival check (no retry) ────
            if room_goal is not None:
                robot._set_nav_curr_pose()
                _arrived_room = None
                if _room_provider and _room_provider.is_available():
                    _arrived_room = _room_provider.get_room_at_cell(
                        int(robot.curr_pos_on_map[0]), int(robot.curr_pos_on_map[1])
                    )
                print(f"  Actual room after path: {_arrived_room or 'unknown'}")
                _room_ok = _room_instance_matches(_arrived_room, _raw_cat)
                _room_status = f"Arrived: {_raw_cat}" if _room_ok else f"Not reached: {_raw_cat}"
                show_obs(robot, _room_status)
                show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                         path_cells=path_cells, label=_room_status)
                if _room_ok:
                    print(f"  Arrived at room '{_raw_cat}' (actual: {_arrived_room}). Done.")
                else:
                    print(f"  Room navigation FAILED: requested '{_raw_cat}', "
                          f"ended in '{_arrived_room or 'unknown'}'.")

                if _ss:
                    _ss.update_current_room(_arrived_room)
                from vlmaps.utils.habitat_utils import agent_state2tf
                agent_state = robot.sim.get_agent(0).get_state()
                start_tf = agent_state2tf(agent_state)
                continue

            _arrival_label = f"Arrived: {_yoloe_target}" if completed else f"Stopped before goal: {_yoloe_target}"
            show_obs(robot, _arrival_label)
            show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                     path_cells=path_cells, label=_arrival_label)

            # Stage 0: YOLOE check at raw arrival (before any rotation)
            _yoloe_confirmed = False
            _confirmation_source = None
            if _yoloe_session is not None:
                try:
                    obs_data = robot.sim.get_sensor_observations(0)
                    if "color_sensor" in obs_data:
                        frame = obs_data["color_sensor"][:, :, :3]
                        _yoloe_confirmed, _ann_frame, _bbox = _yoloe_session.check(frame)
                        if _ann_frame is not None:
                            ann_bgr = cv2.cvtColor(_ann_frame, cv2.COLOR_RGB2BGR)
                            show_obs(robot, f"YOLOE arrival: {_yoloe_target}", yoloe_frame_bgr=ann_bgr)
                        if _yoloe_confirmed:
                            print(f"  YOLOE stage 0: ✓ Found '{_yoloe_target}' at arrival (no rotation needed)!")
                            print(f"  [verify] source=arrival")
                            _confirmation_source = "arrival"
                            freeze_found_target(_yoloe_target, _ann_frame, obj_centroid)
                            # Fix 4: room-entry gate — reject if robot hasn't crossed doorway
                            _s0_comp_room = best_comp.get("room") if best_comp is not None else None
                            if (_s0_comp_room and _room_provider
                                    and _room_provider.is_available()):
                                robot._set_nav_curr_pose()
                                _s0_actual = _room_provider.get_room_at_cell(
                                    int(robot.curr_pos_on_map[0]),
                                    int(robot.curr_pos_on_map[1]),
                                )
                                if _s0_actual != _s0_comp_room:
                                    _s0_dr = obj_centroid[0] - robot.curr_pos_on_map[0]
                                    _s0_dc = obj_centroid[1] - robot.curr_pos_on_map[1]
                                    _s0_dist = float(np.sqrt(_s0_dr**2 + _s0_dc**2))
                                    if _s0_dist > 20:
                                        print(f"  [room-gate] Object visible but room not yet "
                                              f"entered (robot: {_s0_actual}, "
                                              f"target: {_s0_comp_room}, "
                                              f"dist={_s0_dist:.1f} cells) "
                                              f"\u2192 tentative only")
                                        _yoloe_confirmed = False
                                        _confirmation_source = None
                                    else:
                                        print(f"  [room-gate] Room entry confirmed "
                                              f"(dist={_s0_dist:.1f} \u2264 20 cells) "
                                              f"\u2014 accepting")
                        else:
                            print(f"  YOLOE stage 0: ✗ not visible at arrival.")
                except Exception as e:
                    print(f"  YOLOE stage 0 error: {e}")

            # ── Face the object (skip if already confirmed) ───────────────────
            if not _yoloe_confirmed:
                print(f"  Turning to face '{_yoloe_target}'…")
                if room_goal is None:
                    face_toward_pos(robot, obj_centroid[0], obj_centroid[1])
                else:
                    try:
                        robot.face(_raw_cat)
                        robot._set_nav_curr_pose()
                    except Exception:
                        pass
                show_obs(robot, f"Facing: {_yoloe_target}")
                show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                         path_cells=path_cells, label=f"Facing: {_yoloe_target}")

            # Stage 1: YOLOE check after facing (skip if already confirmed)
            if not _yoloe_confirmed:
                if _yoloe_session is not None:
                    try:
                        obs_data = robot.sim.get_sensor_observations(0)
                        if "color_sensor" in obs_data:
                            frame = obs_data["color_sensor"][:, :, :3]
                            _yoloe_confirmed, _ann_frame, _bbox = _yoloe_session.check(frame)
                            if _ann_frame is not None:
                                ann_bgr = cv2.cvtColor(_ann_frame, cv2.COLOR_RGB2BGR)
                                show_obs(robot, f"YOLOE: {_yoloe_target}", yoloe_frame_bgr=ann_bgr)
                            if _yoloe_confirmed:
                                print(f"  YOLOE: ✓ Found '{_yoloe_target}'! (bbox center: {_bbox})")
                                print(f"  [verify] source=turn_to_face")
                                _confirmation_source = "turn_to_face"
                                freeze_found_target(_yoloe_target, _ann_frame, obj_centroid)
                                # Fix 4: room-entry gate for Stage 1
                                _s1_comp_room = best_comp.get("room") if best_comp is not None else None
                                if (_s1_comp_room and _room_provider
                                        and _room_provider.is_available()):
                                    robot._set_nav_curr_pose()
                                    _s1_actual = _room_provider.get_room_at_cell(
                                        int(robot.curr_pos_on_map[0]),
                                        int(robot.curr_pos_on_map[1]),
                                    )
                                    if _s1_actual != _s1_comp_room:
                                        _s1_dr = obj_centroid[0] - robot.curr_pos_on_map[0]
                                        _s1_dc = obj_centroid[1] - robot.curr_pos_on_map[1]
                                        _s1_dist = float(np.sqrt(_s1_dr**2 + _s1_dc**2))
                                        if _s1_dist > 20:
                                            print(f"  [room-gate] Object visible but room not yet "
                                                  f"entered (robot: {_s1_actual}, "
                                                  f"target: {_s1_comp_room}, "
                                                  f"dist={_s1_dist:.1f} cells) "
                                                  f"\u2192 tentative only")
                                            _yoloe_confirmed = False
                                            _confirmation_source = None
                                        else:
                                            print(f"  [room-gate] Room entry confirmed "
                                                  f"(dist={_s1_dist:.1f} \u2264 20) \u2014 accepting")
                                if _yoloe_confirmed:
                                    fine_visual_center(robot, _yoloe_session, _yoloe_target)
                            else:
                                print(f"  YOLOE: ✗ '{_yoloe_target}' not detected.")
                    except Exception as e:
                        print(f"  YOLOE error: {e}")
                else:
                    print("  (YOLOE not available — skipping visual verification)")

            # Stage 2: local bounded scan (±25°) if not confirmed at arrival
            if not _yoloe_confirmed:
                print(f"  Starting local ±25° scan for '{_yoloe_target}'…")
                _pitch_surrogate = _heatmap_target if _heatmap_target != _yoloe_target else ""
                _yoloe_confirmed = scan_local_and_verify(
                    robot,
                    _yoloe_target,
                    rgb_map_2d,
                    heatmap,
                    path_cells,
                    surrogate_cat=_pitch_surrogate,
                )
                if _yoloe_confirmed:
                    _scan_source = getattr(robot, "_last_verify_source", None) or "local_scan"
                    if _scan_source != "pitch_scan":
                        print(f"  [verify] source=local_scan")
                    _confirmation_source = _scan_source
                    if _yoloe_session is not None:
                        fine_visual_center(robot, _yoloe_session, _yoloe_target)

            # Stage 3: Alternative route if local scan also failed
            if not _yoloe_confirmed:
                print(f"  Local scan failed — searching alternative route…")
                robot._set_nav_curr_pose()
                _navigated_alt = navigate_to_alternative(
                    robot, _yoloe_target, kept_components,
                    robot.curr_pos_on_map, rgb_map_2d, heatmap
                )
                if _navigated_alt and _yoloe_session is not None:
                    try:
                        obs_data = robot.sim.get_sensor_observations(0)
                        if "color_sensor" in obs_data:
                            frame = obs_data["color_sensor"][:, :, :3]
                            _yoloe_confirmed, _ann_frame, _bbox = _yoloe_session.check(frame)
                            if _ann_frame is not None:
                                ann_bgr = cv2.cvtColor(_ann_frame, cv2.COLOR_RGB2BGR)
                                show_obs(robot, f"YOLOE alt: {_yoloe_target}", yoloe_frame_bgr=ann_bgr)
                            if _yoloe_confirmed:
                                print(f"  YOLOE (alternative): ✓ Found '{_yoloe_target}'!")
                                print(f"  [verify] source=alternative_route")
                                _confirmation_source = "alternative_route"
                                freeze_found_target(_yoloe_target, _ann_frame, obj_centroid)
                                fine_visual_center(robot, _yoloe_session, _yoloe_target)
                            else:
                                print(f"  YOLOE (alternative): ✗ '{_yoloe_target}' not found. Giving up.")
                    except Exception as e:
                        print(f"  YOLOE error at alternative: {e}")
                elif not _navigated_alt:
                    print(f"  No viable alternative route for '{_yoloe_target}'.")

            # ── Update SearchState with result ────────────────────────────
            robot._set_nav_curr_pose()
            _end_room = None
            if _room_provider and _room_provider.is_available():
                _end_room = _room_provider.get_room_at_cell(
                    int(robot.curr_pos_on_map[0]), int(robot.curr_pos_on_map[1])
                )
            if _ss:
                _ss.update_current_room(_end_room)
                _ss.record_candidate(_end_room, _yoloe_confirmed,
                                     centroid=obj_centroid if 'obj_centroid' in dir() else None)
                if _yoloe_confirmed:
                    _ss.record_object_seen(_end_room, _yoloe_target)
                    _ss.mark_found(_end_room)
                    if _confirmation_source:
                        _ss.mark_confirmation(_confirmation_source)

            if _yoloe_confirmed:
                show_obs(robot, f"FOUND: {_yoloe_target}")
                show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                         path_cells=path_cells, label=f"FOUND: {_yoloe_target}",
                         target_cell=_frozen_target_cell)

            print(f"  Done. YOLOE confirmed: {_yoloe_confirmed}")

            from vlmaps.utils.habitat_utils import agent_state2tf
            agent_state = robot.sim.get_agent(0).get_state()
            start_tf = agent_state2tf(agent_state)

        # Print search state summary for all targets
        for _cat, _ss in _search_states.items():
            if _ss.rooms:
                print(f"\n{_ss.summary()}")

        emit_instruction_eval_summary(
            instruction,
            [plan.original_target for plan in target_plans],
            _search_states,
            robot,
            _room_provider,
        )
        print("\nInstruction complete.")

    from vlmaps.utils.yoloe_utils import shutdown_session
    shutdown_session()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
