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
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from vlmaps.utils.search_state import SearchState

# Milliseconds to wait after each discrete sim step for smooth demo playback.
# Increase for slower, more visible movements; decrease to speed up.
_NAV_STEP_DELAY_MS: int = 80

from vlmaps.robot.habitat_lang_robot import HabitatLanguageRobot
from vlmaps.utils.llm_utils import parse_object_goal_instruction
from vlmaps.utils.mapping_utils import cvt_pose_vec2tf
from vlmaps.utils.matterport3d_categories import mp3dcat, get_categories
from vlmaps.utils.room_map_utils import find_room_goal, load_room_map
from vlmaps.utils.visualize_utils import pool_3d_label_to_2d, pool_3d_rgb_to_2d


# ── Visualization helpers ─────────────────────────────────────────────────────

# Windows the user has closed — never re-open them
_closed_windows: set = set()
_shown_windows: set = set()   # windows that have been successfully shown at least once
_frozen_detection_bgr = None  # frozen YOLOE frame shown until next search


def safe_imshow(name: str, img: np.ndarray) -> None:
    """Show img in a named window. If the user closed it, skip silently."""
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
    if yoloe_frame_bgr is not None:
        frame = yoloe_frame_bgr.copy()
    elif _frozen_detection_bgr is not None:
        frame = _frozen_detection_bgr.copy()
    else:
        obs = robot.sim.get_sensor_observations(0)
        if "color_sensor" in obs:
            frame = cv2.cvtColor(obs["color_sensor"][:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            cv2.waitKey(1)
            return

    if label:
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    safe_imshow("1st person", frame)
    cv2.waitKey(1)


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

    # Paso B — region quality: 0.3·norm_area + 0.4·mean_score + 0.3·density
    max_area = max(c["area"] for c in kept) if kept else 1
    for c in kept:
        norm_area = c["area"] / max_area
        mean_score = c["mean_val"]
        bx, by, bw, bh = c["bbox"]
        bbox_area = bw * bh
        density = c["area"] / bbox_area if bbox_area > 0 else 0.0
        c["quality"] = 0.3 * norm_area + 0.4 * mean_score + 0.3 * density

    kept.sort(key=lambda c: c["quality"], reverse=True)

    # 6. Build outputs
    final_mask = np.zeros(heatmap.shape, dtype=bool)
    for c in kept:
        final_mask |= (labels == c["label"])

    cleaned_heatmap = np.where(final_mask, heatmap, 0.0).astype(np.float32)
    return cleaned_heatmap, final_mask, kept


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
    from vlmaps.utils.index_utils import find_similar_category_id

    cat_id = find_similar_category_id(category, robot.map.categories)
    scores = robot.map.scores_mat[:, cat_id]               # (N,) raw CLIP score
    max_ids = np.argmax(robot.map.scores_mat, axis=1)       # (N,) winning category

    # Only keep voxels where this category wins AND score is strong enough
    valid = (max_ids == cat_id) & (scores > score_thresh)
    scores_filtered = np.where(valid, scores, 0.0)

    # Project to 2D: max score per (row, col) column
    gs = robot.map.gs
    heat_2d = np.zeros((gs, gs), dtype=np.float32)
    for i, pos in enumerate(robot.map.grid_pos):
        row, col, _ = pos
        if scores_filtered[i] > heat_2d[row, col]:
            heat_2d[row, col] = scores_filtered[i]

    # Tight distance decay — only a ~3-cell (15 cm) fringe around real detections
    mask = heat_2d > 0
    if mask.any():
        dist = distance_transform_edt(~mask)
        heat_2d = np.where(mask, heat_2d, np.clip(1.0 - dist * 0.3, 0, 1).astype(np.float32))
        heat_2d[heat_2d < 0.15] = 0

    # Postprocess: remove spurious small activations
    heat_2d, _, kept_components = postprocess_heatmap(heat_2d)
    if kept_components:
        print(f"  Heatmap postprocess: kept {len(kept_components)} component(s) "
              f"(areas: {[c['area'] for c in kept_components]}, "
              f"quality: {[round(c['quality'], 3) for c in kept_components]})")

    return heat_2d, kept_components


def show_map(robot, rgb_map_2d: np.ndarray, heatmap_2d: np.ndarray = None,
             path_cells: list = None, label: str = "",
             zoom_radius: int = 300, output_px: int = 700):
    """Display a top-down semantic map with optional zoom centered on the robot.

    Args:
        zoom_radius: Half-side of the crop window in map cells. The displayed
                     region is (2*zoom_radius) × (2*zoom_radius) cells, upscaled
                     to output_px for readability. Use 0 to show the full map.
        output_px:   Target display size in pixels (square).
    """
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
    cv2.waitKey(1)


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

    for idx in range(preferred_idx, start_idx, -1):
        if _segment_is_free_on_map(curr_cell, dense_path[idx], free_map):
            return idx

    if preferred_idx == start_idx:
        return start_idx
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
        cv2.waitKey(_NAV_STEP_DELAY_MS)
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
    from vlmaps.utils.yoloe_utils import get_session

    session = get_session(cat, conf_thresh=0.3)
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
            cv2.waitKey(_NAV_STEP_DELAY_MS)

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
    motion_thresh: float = 0.4,
    stuck_threshold: int = 3,
    dist_map: np.ndarray = None,
) -> bool:
    """Follow the planned polyline with lookahead and a footprint-aware shield."""
    # ── Open-space thresholds ─────────────────────────────────────────────
    _STOP_CL      = 1.0   # cells — hard stop on front center
    _SLOW_CL      = 3.0   # cells — low-clearance warning
    _SIDE_STOP_CL = 1.5   # cells — side-contact limit in open space

    # ── Doorway / narrow-passage thresholds ──────────────────────────────
    _DOORWAY_SIDE_TH   = 5.0  # cells — narrow-passage detection threshold
    _DOORWAY_SIDE_EXIT = 6.5  # cells — hysteresis to avoid doorway-mode flicker
    _DOORWAY_FRONT_MIN = 0.5  # cells — minimum front inside doorway mode
    _STOP_CL_DOOR      = 0.5  # cells — relaxed hard stop in doorway

    # ── Robot footprint half-width for lateral footprint sampling ─────────
    _ROBOT_HW = 2  # cells (~0.10 m per side at cs=0.05 m)

    print(f"  [shield] execution thresholds: "
          f"stop={_STOP_CL} slow={_SLOW_CL} side_stop={_SIDE_STOP_CL} | "
          f"doorway: stop={_STOP_CL_DOOR} side_th={_DOORWAY_SIDE_TH}")

    dense_path = densify_path_cells(path_cells)
    if not dense_path:
        return True

    goal_cell = dense_path[-1]
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
    _GOAL_REACHED_TOL = max(3, _fwd_cells + 1)
    _MAX_FOLLOW_STEPS = max(len(planned_actions) * 2, len(dense_path) * 3, 120)
    _doorway_mode = False  # updated before each forward step
    progress_idx = 0

    print(f"  [nav] Path follower: {len(dense_path)} dense path cell(s), "
          f"lookahead={_LOOKAHEAD_OPEN_CELLS} open / {_LOOKAHEAD_TIGHT_CELLS} tight")

    for i in range(_MAX_FOLLOW_STEPS):
        robot._set_nav_curr_pose()
        _curr_cell = [int(round(robot.curr_pos_on_map[0])), int(round(robot.curr_pos_on_map[1]))]
        progress_idx = _closest_path_index(_curr_cell, dense_path, progress_idx)

        _goal_dr = float(goal_cell[0]) - float(_curr_cell[0])
        _goal_dc = float(goal_cell[1]) - float(_curr_cell[1])
        _goal_dist = float(np.hypot(_goal_dr, _goal_dc))
        if progress_idx >= len(dense_path) - 1 and _goal_dist <= _GOAL_REACHED_TOL:
            print(f"  [nav] Goal reached on dense path "
                  f"(dist={_goal_dist:.1f} cells, progress={progress_idx + 1}/{len(dense_path)})")
            return True

        _curr_clearance = float("inf")
        if dist_map is not None:
            _r = int(np.clip(_curr_cell[0], 0, dist_map.shape[0] - 1))
            _c = int(np.clip(_curr_cell[1], 0, dist_map.shape[1] - 1))
            _curr_clearance = float(dist_map[_r, _c])
        _lookahead_cells = (
            _LOOKAHEAD_TIGHT_CELLS if _curr_clearance < _LOOKAHEAD_TIGHT_CL else _LOOKAHEAD_OPEN_CELLS
        )
        preferred_idx = _lookahead_path_index(dense_path, progress_idx, _lookahead_cells)
        target_idx = _visible_lookahead_index(
            _curr_cell,
            dense_path,
            progress_idx,
            preferred_idx,
            free_map,
        )
        target_cell = dense_path[target_idx]

        _curr_pose = (
            float(robot.curr_pos_on_map[0]),
            float(robot.curr_pos_on_map[1]),
            float(robot.curr_ang_deg_on_map),
        )
        _preview = robot.controller.convert_goal_to_actions(_curr_pose, target_cell)

        if not _preview:
            if target_idx < len(dense_path) - 1:
                progress_idx = min(progress_idx + 1, len(dense_path) - 1)
                continue
            if _goal_dist <= _GOAL_REACHED_TOL + _fwd_cells:
                print(f"  [nav] Goal reached after lookahead convergence "
                      f"(dist={_goal_dist:.1f} cells)")
                return True
            _preview = robot.controller.convert_goal_to_actions(_curr_pose, goal_cell)
            if not _preview:
                return True

        if preferred_idx != target_idx and (i == 0 or i % 20 == 0):
            print(f"  [nav] Lookahead clipped by local visibility: "
                  f"{preferred_idx - progress_idx} -> {target_idx - progress_idx} cells ahead")

        action = _preview[0]

        is_fwd = (action == "move_forward")

        # ── Footprint-aware shield — check BEFORE executing forward step ────
        if is_fwd and dist_map is not None:
            _r  = float(robot.curr_pos_on_map[0])
            _c  = float(robot.curr_pos_on_map[1])
            _ang_rad = float(robot.curr_ang_deg_on_map) * np.pi / 180.0
            _h, _w   = dist_map.shape

            # Direction vectors (map: 0°=north=decreasing-row)
            # forward:  dr=-cos, dc=+sin
            # left:     CCW 90° of forward → dr=-sin, dc=-cos   (→ (-dc_fwd, dr_fwd))
            # right:    CW  90° of forward → dr=+sin, dc=+cos   (→ (+dc_fwd, -dr_fwd))
            _dr_fwd   = -np.cos(_ang_rad)
            _dc_fwd   =  np.sin(_ang_rad)
            _dr_left  = -_dc_fwd    # = -sin
            _dc_left  =  _dr_fwd    # = -cos
            _dr_right =  _dc_fwd    # = +sin
            _dc_right = -_dr_fwd    # = +cos

            # ── Predicted next front-center ───────────────────────────
            _nr = int(np.clip(round(_r + _dr_fwd * _fwd_cells), 0, _h - 1))
            _nc = int(np.clip(round(_c + _dc_fwd * _fwd_cells), 0, _w - 1))

            # ── Front-left / front-right at next pose ─────────────────
            _fl_r = int(np.clip(round(_nr + _dr_left  * _ROBOT_HW), 0, _h - 1))
            _fl_c = int(np.clip(round(_nc + _dc_left  * _ROBOT_HW), 0, _w - 1))
            _fr_r = int(np.clip(round(_nr + _dr_right * _ROBOT_HW), 0, _h - 1))
            _fr_c = int(np.clip(round(_nc + _dc_right * _ROBOT_HW), 0, _w - 1))

            _cl_front = float(dist_map[_nr,   _nc  ])
            _cl_left  = float(dist_map[_fl_r, _fl_c])
            _cl_right = float(dist_map[_fr_r, _fr_c])
            _cl_min   = min(_cl_front, _cl_left, _cl_right)

            # ── Side clearances at current pose (perpendicular, 2×HW) ──
            # Used only for doorway detection, not for blocking.
            _sl_r = int(np.clip(round(_r + _dr_left  * _ROBOT_HW * 2), 0, _h - 1))
            _sl_c = int(np.clip(round(_c + _dc_left  * _ROBOT_HW * 2), 0, _w - 1))
            _sr_r = int(np.clip(round(_r + _dr_right * _ROBOT_HW * 2), 0, _h - 1))
            _sr_c = int(np.clip(round(_c + _dc_right * _ROBOT_HW * 2), 0, _w - 1))
            _side_l = float(dist_map[_sl_r, _sl_c])
            _side_r = float(dist_map[_sr_r, _sr_c])

            # ── Doorway detection ─────────────────────────────────────
            # Two signals are useful here:
            # 1) side clearance at current pose (robot already inside a narrow gap)
            # 2) footprint side clearance at the NEXT pose (robot entering a doorway)
            #
            # Using only the current-pose side probes misses doorway entry and the
            # robot gets blocked one step too early by the open-space lateral rule.
            _curr_narrow = (
                min(_side_l, _side_r) < _DOORWAY_SIDE_TH
                and max(_side_l, _side_r) < _DOORWAY_SIDE_EXIT
                and _cl_front > _DOORWAY_FRONT_MIN
            )
            _next_narrow = (
                min(_cl_left, _cl_right) < _DOORWAY_SIDE_TH
                and max(_cl_left, _cl_right) < _DOORWAY_SIDE_EXIT
                and _cl_front > _DOORWAY_FRONT_MIN
            )
            _entering_doorway = (
                min(_cl_left, _cl_right) < _SIDE_STOP_CL
                and max(_cl_left, _cl_right) < (_DOORWAY_SIDE_EXIT + 1.5)
                and _cl_front < _DOORWAY_SIDE_EXIT
                and _cl_front > _DOORWAY_FRONT_MIN
            )
            _was_doorway = _doorway_mode
            if _doorway_mode:
                _still_narrow = (
                    min(_side_l, _side_r) < _DOORWAY_SIDE_EXIT
                    or min(_cl_left, _cl_right) < _DOORWAY_SIDE_EXIT
                    or _cl_front < _DOORWAY_SIDE_EXIT
                )
                _doorway_mode = (
                    _cl_front > _DOORWAY_FRONT_MIN
                    and _still_narrow
                )
            else:
                _doorway_mode = _curr_narrow or _next_narrow or _entering_doorway
            if _doorway_mode != _was_doorway:
                if _doorway_mode:
                    print(f"  [shield] Step {i+1}/{_MAX_FOLLOW_STEPS}: doorway mode ON — "
                          f"curr-sides=({_side_l:.1f}, {_side_r:.1f}) "
                          f"next-sides=({_cl_left:.1f}, {_cl_right:.1f}) "
                          f"front={_cl_front:.1f} — switching to cautious doorway traversal")
                else:
                    print(f"  [shield] Step {i+1}/{_MAX_FOLLOW_STEPS}: doorway mode OFF")

            # ── Footprint log (always in doorway mode, or when close) ─
            if _doorway_mode or _cl_min < _SLOW_CL:
                print(f"  [shield] Step {i+1}/{_MAX_FOLLOW_STEPS}: "
                      f"front clearance={_cl_front:.1f} "
                      f"left clearance={_cl_left:.1f} "
                      f"right clearance={_cl_right:.1f} "
                      f"footprint min clearance={_cl_min:.1f} "
                      f"[doorway mode: {'ON' if _doorway_mode else 'OFF'}]")

            # ── Block / warn ──────────────────────────────────────────
            _stop_thr  = _STOP_CL_DOOR if _doorway_mode else _STOP_CL
            _side_thr  = _DOORWAY_FRONT_MIN if _doorway_mode else _SIDE_STOP_CL

            if _cl_front < _stop_thr:
                print(f"  [shield] Step {i+1}/{_MAX_FOLLOW_STEPS}: front clearance={_cl_front:.1f} "
                      f"< {_stop_thr:.1f} — hard stop")
                return False
            if not _doorway_mode and (_cl_left < _side_thr or _cl_right < _side_thr):
                print(f"  [shield] Step {i+1}/{_MAX_FOLLOW_STEPS}: "
                      f"step blocked by side clearance "
                      f"(left={_cl_left:.1f} right={_cl_right:.1f} < {_side_thr:.1f})")
                return False
            if _doorway_mode and _cl_front < _DOORWAY_FRONT_MIN:
                print(f"  [shield] Step {i+1}/{_MAX_FOLLOW_STEPS}: "
                      f"front clearance={_cl_front:.1f} "
                      f"< {_DOORWAY_FRONT_MIN:.1f} — blocked in doorway mode")
                return False
            if _cl_min < _SLOW_CL:
                print(f"  [shield] Step {i+1}/{_MAX_FOLLOW_STEPS}: "
                      f"continuing in cautious {'doorway' if _doorway_mode else 'close-walls'} mode")

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

        show_obs(robot, f"[{i+1}/{_MAX_FOLLOW_STEPS}] -> {cat}")
        show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                 path_cells=dense_path, label=f"[{i+1}/{_MAX_FOLLOW_STEPS}] -> {cat}")
        cv2.waitKey(_NAV_STEP_DELAY_MS)

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
            show_obs(robot, f"[recovery {i+1}/{n}] -> {cat}")
            show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                     label=f"[recovery {i+1}/{n}] -> {cat}")
            cv2.waitKey(_NAV_STEP_DELAY_MS)
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
        cv2.waitKey(_NAV_STEP_DELAY_MS)

    robot._set_nav_curr_pose()
    if not detected_once:
        print(f"  [center] '{cat}' not detected — skipping centering.")
    return detected_once


def select_best_candidate(
    kept_components: list,
    current_room: str,
    query_priors: dict,
    tried_centroids: set,
    *,
    local_min_quality: float = 0.25,
    switch_margin: float = 0.40,
    same_room_bonus: float = 0.35,
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

    Args:
        kept_components:  list of component dicts (quality-sorted, best first).
        current_room:     room instance the robot is currently in.
        query_priors:     {room: normalised_prior} from Phase C.
        tried_centroids:  set of (int_r, int_c) already inspected this query.
        local_min_quality: minimum quality for a local candidate to be accepted.
        switch_margin:    external must beat local by this fraction to trigger switch.
        same_room_bonus:  additive quality bonus applied to local candidates.

    Returns:
        The selected component dict.
    """
    from vlmaps.utils.room_priors import canonical_room_type, compatible_room_types

    if not kept_components:
        return None

    # Separate tried vs untried
    def _is_tried(comp):
        cr, cc = comp["centroid"]
        return (int(cr), int(cc)) in tried_centroids

    untried = [c for c in kept_components if not _is_tried(c)]
    if not untried:
        # All tried — fall back to global best untried (might be empty)
        return kept_components[0]

    if not current_room or not query_priors:
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
    local_effective = best_local["quality"] + same_room_bonus
    ext_quality = best_ext["quality"] if best_ext else 0.0

    print(f"  [room-gate] Best local quality: {best_local['quality']:.3f} "
          f"(+bonus → {local_effective:.3f})")
    if best_ext:
        print(f"  [room-gate] Best external quality: {ext_quality:.3f}")

    # Gate 1: local candidate too weak even with bonus
    if best_local["quality"] < local_min_quality:
        if best_ext and ext_quality > local_effective:
            print(f"  [room-gate] Switch allowed: True (local quality {best_local['quality']:.3f} < threshold {local_min_quality})")
            return untried[0]

    # Gate 2: external must clearly beat boosted local to trigger switch
    if best_ext and ext_quality > local_effective * (1.0 + switch_margin):
        print(f"  [room-gate] Switch allowed: True "
              f"(external {ext_quality:.3f} >> local {local_effective:.3f})")
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
            cl = float(dist_map[row, col])
            if cl < min_clearance:
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

    # Find matching region id (same word-boundary logic as get_room_centroid)
    query   = target_room.lower().strip()
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
        print(f"  [room-goal] Room '{target_room}' not found in region grid")
        return []

    # Navigable cells inside the target room
    room_mask       = (region_grid == target_rid)
    free_mask       = (obs_map > 0)
    navigable_room  = room_mask & free_mask

    if not navigable_room.any():
        print(f"  [room-goal] No navigable cells in room '{target_room}'")
        return []

    dist_map = distance_transform_edt(obs_map)
    rows, cols = np.where(navigable_room)
    clearances = dist_map[rows, cols]

    # Sort by clearance descending
    sorted_idx = np.argsort(-clearances)

    candidates = []
    for i in sorted_idx:
        cl = float(clearances[i])
        r, c = int(rows[i]), int(cols[i])
        # Always include at least one candidate even if clearance is low
        if cl >= min_clearance or not candidates:
            candidates.append([r, c])
        if len(candidates) >= top_k:
            break

    best_cl = float(clearances[sorted_idx[0]]) if len(sorted_idx) > 0 else 0.0
    print(f"  [room-goal] Found {len(candidates)} safe goal(s) in '{target_room}' "
          f"(best clearance: {best_cl:.1f} cells)")
    return candidates


def _room_instance_matches(actual_room, target_room: str) -> bool:
    """Return True if actual_room satisfies the target_room navigation command.

    Allows canonical-type match so 'kitchen.001' satisfies command 'kitchen'.
    Rejects type mismatch so 'bathroom.001' does NOT satisfy 'dining room'.
    """
    from vlmaps.utils.room_priors import canonical_room_type
    if actual_room is None:
        return False
    a = actual_room.lower().strip()
    t = target_room.lower().strip()
    if a == t:
        return True
    return canonical_room_type(a) == canonical_room_type(t)


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

        print(f"Targets: {categories}")

        robot.set_agent_state(start_tf)
        robot._set_nav_curr_pose()
        robot.empty_recorded_actions()
        show_obs(robot, "Start")
        show_map(robot, rgb_map_2d, label="Start")

        # ── Phase B+C: build per-room search state and compute object→room priors ──
        _search_states: dict = {}  # cat → SearchState
        for cat in categories:
            _c = cat.strip()
            _ss = SearchState(_c, _room_provider, robot.map.obstacles_map)
            if _ss.rooms:
                # Phase C: compute priors (LLM + manual table + scene evidence)
                _priors = _ss.compute_priors()
                if _priors:
                    _sorted = sorted(_priors.items(), key=lambda x: x[1], reverse=True)
                    _prior_str = ", ".join(f"{r}={v:.2f}" for r, v in _sorted if v > 0.01)
                    print(f"  Room priors for '{_c}': {_prior_str}")
            _search_states[_c] = _ss

        for cat in categories:
            cat = cat.strip()
            if not cat:
                continue
            _frozen_detection_bgr = None  # clear previous detection freeze
            _ss = _search_states.get(cat)
            print(f"\nPlanning path to: {cat}")

            # ── Check if target is a room name (room-level navigation) ────────
            room_goal = None
            if _room_provider and _room_provider.is_available():
                room_goal = _room_provider.get_room_centroid(cat)
            if room_goal is None and _room_regions:
                room_goal = find_room_goal(cat, _room_regions)

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
                print(f"  Requested room command: {cat}")
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
                    cat, _room_provider, _safe_map_for_rooms
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
                heatmap, kept_components = compute_heatmap(robot, cat)

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
                        cat, _known_rooms_list, _seen_objs,
                        heatmap_evidence=_heatmap_ev, query_type="direct",
                    )
                    for _rn, _rv in _new_priors.items():
                        if _rn in _ss.rooms:
                            _ss.rooms[_rn].target_relevance = _rv
                    _np_sorted = sorted(_new_priors.items(), key=lambda x: -x[1])
                    print(f"  Final room priors after evidence fusion: "
                          f"{ ', '.join(f'{r}={v:.2f}' for r, v in _np_sorted[:6] if v > 0.01) }")

                show_map(robot, rgb_map_2d, heatmap_2d=heatmap, label=f"Planning: {cat}")
                cv2.waitKey(200)

                if not kept_components:
                    print(f"  [skip] No heatmap signal for '{cat}' in this scene.")
                    continue

                from vlmaps.utils.yoloe_utils import get_session, shutdown_session
                _yoloe_session = get_session(cat, conf_thresh=0.3)

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

                best_comp = select_best_candidate(
                    kept_components,
                    current_room,
                    _query_priors,
                    _tried,
                )
                if best_comp is None:
                    print(f"  [skip] No viable candidate for '{cat}'.")
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
                        _initial_path, heatmap, robot.map.obstacles_map
                    )
                    print(f"  Fallback path-based goal: {goal_pos}")

                # Step 3: plan to the selected goal
                _, planned_actions = robot.plan_path_only(goal_pos)

            # Capture planned path for visualization
            path_cells = densify_path_cells(getattr(robot, "last_planned_path", None) or [])

            n_actions = len(planned_actions)
            print(f"  Path computed: {len(path_cells)} dense path cell(s) "
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
                            min_dist=8.0, max_dist=25.0, min_clearance=3.0,
                        )
                        if _nc_app:
                            _nc_goal = _nc_app[0]
                        else:
                            _nc_init, _ = robot.plan_path_only(_nc_cen)
                            _nc_goal, _nc_cen = select_safe_goal_from_path(
                                _nc_init, heatmap, robot.map.obstacles_map
                            )
                        _, _nc_acts = robot.plan_path_only(_nc_goal)
                        _nc_path = densify_path_cells(getattr(robot, "last_planned_path", None) or [])
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
                        print(f"  [safety] No safe candidate found for '{cat}' — skipping.")
                        if _yoloe_session is not None:
                            from vlmaps.utils.yoloe_utils import shutdown_session
                            shutdown_session()
                        continue

            # n_actions == 0 means robot is already at the goal — treat as arrived.
            already_at_goal = (n_actions == 0)


            # Show heatmap + planned path BEFORE executing so the user can see the route
            show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                     path_cells=path_cells, label=f"Path planned: {cat}")
            cv2.waitKey(800)

            # Precompute dist_map for the collision shield from the RAW (undilated)
            # obstacle map so doorways are not falsely flagged as low-clearance.
            # The dilated _safe_obs_map is only for planning/goal validation.
            _dist_map_shield = distance_transform_edt(robot.map.obstacles_map)

            if already_at_goal:
                print(f"  Already at goal for '{cat}' — proceeding with verification.")
                completed = True
            else:
                print(f"  Executing path follower over {len(path_cells)} dense cell(s)…")
                # One-shot execution: no recovery or replanning on failure.
                completed = execute_nav_replay(
                    robot, planned_actions, cat, rgb_map_2d, heatmap, path_cells,
                    dist_map=_dist_map_shield,
                )
                if not completed:
                    print(f"  [nav] Path execution stopped early for '{cat}' "
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
                _room_ok = _room_instance_matches(_arrived_room, cat)
                _room_status = f"Arrived: {cat}" if _room_ok else f"Not reached: {cat}"
                show_obs(robot, _room_status)
                show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                         path_cells=path_cells, label=_room_status)
                if _room_ok:
                    print(f"  Arrived at room '{cat}' (actual: {_arrived_room}). Done.")
                else:
                    print(f"  Room navigation FAILED: requested '{cat}', "
                          f"ended in '{_arrived_room or 'unknown'}'.")

                if _ss:
                    _ss.update_current_room(_arrived_room)
                from vlmaps.utils.habitat_utils import agent_state2tf
                agent_state = robot.sim.get_agent(0).get_state()
                start_tf = agent_state2tf(agent_state)
                continue

            _arrival_label = f"Arrived: {cat}" if completed else f"Stopped before goal: {cat}"
            show_obs(robot, _arrival_label)
            show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                     path_cells=path_cells, label=_arrival_label)

            # Stage 0: YOLOE check at raw arrival (before any rotation)
            _yoloe_confirmed = False
            if _yoloe_session is not None:
                try:
                    obs_data = robot.sim.get_sensor_observations(0)
                    if "color_sensor" in obs_data:
                        frame = obs_data["color_sensor"][:, :, :3]
                        _yoloe_confirmed, _ann_frame, _bbox = _yoloe_session.check(frame)
                        if _ann_frame is not None:
                            ann_bgr = cv2.cvtColor(_ann_frame, cv2.COLOR_RGB2BGR)
                            show_obs(robot, f"YOLOE arrival: {cat}", yoloe_frame_bgr=ann_bgr)
                        if _yoloe_confirmed:
                            print(f"  YOLOE stage 0: ✓ Found '{cat}' at arrival (no rotation needed)!")
                            if _ann_frame is not None:
                                _frozen_detection_bgr = cv2.cvtColor(_ann_frame, cv2.COLOR_RGB2BGR)
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
                print(f"  Turning to face '{cat}'…")
                if room_goal is None:
                    face_toward_pos(robot, obj_centroid[0], obj_centroid[1])
                else:
                    try:
                        robot.face(cat)
                        robot._set_nav_curr_pose()
                    except Exception:
                        pass
                show_obs(robot, f"Facing: {cat}")
                show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                         path_cells=path_cells, label=f"Facing: {cat}")

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
                                show_obs(robot, f"YOLOE: {cat}", yoloe_frame_bgr=ann_bgr)
                            if _yoloe_confirmed:
                                print(f"  YOLOE: ✓ Found '{cat}'! (bbox center: {_bbox})")
                                if _ann_frame is not None:
                                    _frozen_detection_bgr = cv2.cvtColor(_ann_frame, cv2.COLOR_RGB2BGR)
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
                                        else:
                                            print(f"  [room-gate] Room entry confirmed "
                                                  f"(dist={_s1_dist:.1f} \u2264 20) \u2014 accepting")
                                if _yoloe_confirmed:
                                    fine_visual_center(robot, _yoloe_session, cat)
                            else:
                                print(f"  YOLOE: ✗ '{cat}' not detected.")
                    except Exception as e:
                        print(f"  YOLOE error: {e}")
                else:
                    print("  (YOLOE not available — skipping visual verification)")

            # Stage 2: 360° real-time scan if not confirmed at arrival
            if not _yoloe_confirmed:
                print(f"  Starting 360° real-time scan for '{cat}'…")
                _yoloe_confirmed = scan_360_and_verify(
                    robot, cat, rgb_map_2d, heatmap, path_cells
                )
                if _yoloe_confirmed and _yoloe_session is not None:
                    fine_visual_center(robot, _yoloe_session, cat)

            # Stage 3: Alternative route if 360° scan also failed
            if not _yoloe_confirmed:
                print(f"  360° scan failed — searching alternative route…")
                robot._set_nav_curr_pose()
                _navigated_alt = navigate_to_alternative(
                    robot, cat, kept_components,
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
                                show_obs(robot, f"YOLOE alt: {cat}", yoloe_frame_bgr=ann_bgr)
                            if _yoloe_confirmed:
                                print(f"  YOLOE (alternative): ✓ Found '{cat}'!")
                                if _ann_frame is not None:
                                    _frozen_detection_bgr = cv2.cvtColor(_ann_frame, cv2.COLOR_RGB2BGR)
                                fine_visual_center(robot, _yoloe_session, cat)
                            else:
                                print(f"  YOLOE (alternative): ✗ '{cat}' not found. Giving up.")
                    except Exception as e:
                        print(f"  YOLOE error at alternative: {e}")
                elif not _navigated_alt:
                    print(f"  No viable alternative route for '{cat}'.")

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
                    _ss.record_object_seen(_end_room, cat)
                    _ss.mark_found(_end_room)

            print(f"  Done. YOLOE confirmed: {_yoloe_confirmed}")

            from vlmaps.utils.habitat_utils import agent_state2tf
            agent_state = robot.sim.get_agent(0).get_state()
            start_tf = agent_state2tf(agent_state)

        # Print search state summary for all targets
        for _cat, _ss in _search_states.items():
            if _ss.rooms:
                print(f"\n{_ss.summary()}")

        print("\nInstruction complete.")

    from vlmaps.utils.yoloe_utils import shutdown_session
    shutdown_session()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
