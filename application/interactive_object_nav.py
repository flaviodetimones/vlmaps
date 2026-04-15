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
) -> bool:
    """Replay a pre-planned action list with stuck detection.

    Monitors actual displacement after each move_forward action.  If
    real displacement is below motion_thresh × forward_dist for
    stuck_threshold consecutive steps, returns False immediately so the
    caller can trigger recovery.  Turn actions are never counted as stuck.

    Args:
        motion_thresh:   Fraction of forward_dist considered "barely moved".
        stuck_threshold: Consecutive low-motion steps before declaring stuck.

    Returns:
        True  — replay completed without stuck event.
        False — stuck detected; caller should recover and re-plan.
    """
    low_motion_count = 0
    n = len(planned_actions)
    expected_fwd = robot.forward_dist  # metres per move_forward action

    for i, action in enumerate(planned_actions):
        if action == "stop":
            continue

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
                    print(f"  [nav] Stuck after {i+1}/{n} actions "
                          f"(last disp={disp*100:.1f} cm) — triggering recovery")
                    return False
            else:
                low_motion_count = 0

        show_obs(robot, f"[{i+1}/{n}] -> {cat}")
        show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                 path_cells=path_cells, label=f"[{i+1}/{n}] -> {cat}")
        cv2.waitKey(_NAV_STEP_DELAY_MS)

    return True


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


def select_safe_goal_from_path(
    path_cells: list,
    heatmap: np.ndarray,
    obs_map: np.ndarray,
    min_dist_cells: float = 10.0,
    max_dist_cells: float = 24.0,
    clearance_cells: float = 3.0,
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

    if not path_cells:
        return obj_centroid, obj_centroid

    dist_to_obs = distance_transform_edt(obs_map)

    for cell in reversed(path_cells):
        row, col = int(cell[0]), int(cell[1])
        dr = row - obj_row
        dc = col - obj_col
        dist = float(np.sqrt(dr * dr + dc * dc))
        if 0 <= row < obs_map.shape[0] and 0 <= col < obs_map.shape[1]:
            clearance = float(dist_to_obs[row, col])
        else:
            clearance = 0.0
        if min_dist_cells <= dist <= max_dist_cells and clearance >= clearance_cells:
            return [row, col], obj_centroid

    # Fallback: use path end
    last = path_cells[-1]
    return [int(last[0]), int(last[1])], obj_centroid


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

    # Pre-compute which object categories actually have cells in the current map.
    # labeled_map_cropped[i] is the boolean mask for category i after load_categories().
    _STRUCTURAL = {"void", "wall", "floor", "ceiling"}
    _MIN_CELLS = 15  # ignore categories with fewer active cells (noise)
    def _get_present_categories() -> list:
        labeled = getattr(robot.map, "labeled_map_cropped", None)
        cats = getattr(robot.map, "categories", [])
        if labeled is None or not cats:
            return [c for c in cats if c not in _STRUCTURAL]
        present = []
        for i, c in enumerate(cats):
            if c in _STRUCTURAL:
                continue
            try:
                if int(labeled[i].sum()) >= _MIN_CELLS:
                    present.append(c)
            except Exception:
                pass
        return present

    # ── Instruction loop ─────────────────────────────────────────────────────
    while True:
        _present = _get_present_categories()
        rooms_hint = ""
        if _room_provider and _room_provider.is_available():
            rooms_hint = f"  Rooms     : {_room_provider.list_rooms()}\n"
        print(
            f"\n{'─' * 50}\n"
            f"{rooms_hint}"
            f"  Objects   : {_present}"
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

        for cat in categories:
            cat = cat.strip()
            if not cat:
                continue
            _frozen_detection_bgr = None  # clear previous detection freeze
            print(f"\nPlanning path to: {cat}")

            # ── Check if target is a room name (room-level navigation) ────────
            room_goal = None
            if _room_provider and _room_provider.is_available():
                room_goal = _room_provider.get_room_centroid(cat)
            if room_goal is None and _room_regions:
                room_goal = find_room_goal(cat, _room_regions)

            if room_goal is not None:
                # Room-level: navigate directly to centroid, no heatmap/YOLOE
                robot._set_nav_curr_pose()
                current_room = None
                if _room_provider and _room_provider.is_available():
                    current_room = _room_provider.get_room_at_cell(
                        int(robot.curr_pos_on_map[0]), int(robot.curr_pos_on_map[1])
                    )
                print(f"  Current room: {current_room or 'unknown'}")
                print(f"  Room match '{cat}': navigating to centroid {room_goal}")
                goal_pos = list(room_goal)
                _, planned_actions = robot.plan_path_only(goal_pos)
                heatmap = np.zeros((robot.map.gs, robot.map.gs), dtype=np.float32)
                kept_components = []
                _yoloe_session = None
            else:
                # Object-level: compute heatmap and prepare YOLOE
                print("  Computing semantic heatmap...")
                heatmap, kept_components = compute_heatmap(robot, cat)

                # Annotate each component with its room
                if _room_provider and _room_provider.is_available():
                    robot._set_nav_curr_pose()
                    current_room = _room_provider.get_room_at_cell(
                        int(robot.curr_pos_on_map[0]), int(robot.curr_pos_on_map[1])
                    )
                    print(f"  Current room: {current_room or 'unknown'}")
                    rooms_count: dict = {}
                    for comp in kept_components:
                        cr, cc = comp["centroid"]
                        comp["room"] = _room_provider.get_room_at_cell(int(cr), int(cc))
                        r = comp["room"] or "unknown"
                        rooms_count[r] = rooms_count.get(r, 0) + 1
                    if rooms_count:
                        print(f"  Candidates by room: {rooms_count}")

                show_map(robot, rgb_map_2d, heatmap_2d=heatmap, label=f"Planning: {cat}")
                cv2.waitKey(200)

                from vlmaps.utils.yoloe_utils import get_session, shutdown_session
                _yoloe_session = get_session(cat, conf_thresh=0.3)

            if room_goal is not None:
                goal_pos = list(room_goal)
                obj_centroid = goal_pos
                _, planned_actions = robot.plan_path_only(goal_pos)
            else:
                # Step 1: plan (without executing) to a close-approach standoff to get the path
                robot._set_nav_curr_pose()
                standoff_pos, _ = robot.map.get_standoff_pos(
                    robot.curr_pos_on_map, cat, standoff_m=0.25)
                print(f"  Planning initial path to standoff: {standoff_pos}")
                _initial_path, _ = robot.plan_path_only(standoff_pos)

                # Step 2: walk path backward to pick best viewpoint + object centroid
                goal_pos, obj_centroid = select_safe_goal_from_path(
                    _initial_path, heatmap, robot.map.obstacles_map
                )
                print(f"  Path-based goal: {goal_pos}  object centroid: {obj_centroid}")

                # Step 3: plan (without executing) to the selected goal
                _, planned_actions = robot.plan_path_only(goal_pos)

            # Capture planned path for visualization
            path_cells = getattr(robot, "last_planned_path", None) or []

            n_actions = len(planned_actions)
            print(f"  Path computed: {n_actions} actions.")

            # n_actions == 0 means robot is already at the goal — treat as arrived,
            # not as failure.  A truly unreachable goal produces a non-empty path that
            # ends before reaching the destination (handled by execute_nav_replay).
            already_at_goal = (n_actions == 0)

            if not already_at_goal:
                # Sanity-check: reject paths whose length is excessively longer than
                # the straight-line distance (detour ratio).  Wild visgraph paths that
                # leave the house and loop back produce ratios >> 5.
                _MAX_DETOUR_RATIO = 4.0
                robot._set_nav_curr_pose()
                cr, cc = robot.curr_pos_on_map
                gr, gc = goal_pos[0], goal_pos[1]
                straight_dist = float(np.sqrt((cr - gr) ** 2 + (cc - gc) ** 2))
                if straight_dist > 1.0 and n_actions > _MAX_DETOUR_RATIO * straight_dist:
                    print(
                        f"  [warn] Path too long ({n_actions} actions vs "
                        f"{straight_dist:.0f}-cell straight line, ratio "
                        f"{n_actions / straight_dist:.1f}x > {_MAX_DETOUR_RATIO}x) — skipping."
                    )
                    continue

                # Narrow-passage check: if any cell on the path has less than
                # _MIN_PASSAGE_CELLS clearance to an obstacle, the robot will likely
                # collide.  Try one alternative standoff (further away, different
                # approach angle).  If that also produces a narrow path, give up.
                _MIN_PASSAGE_CELLS = 4  # 4 × 0.05 m = 20 cm minimum passage width
                _obs_map_nav = robot.map.obstacles_map
                _dist_nav = distance_transform_edt(_obs_map_nav)
                _min_clearance = min(
                    (float(_dist_nav[int(c[0]), int(c[1])])
                     if 0 <= int(c[0]) < _dist_nav.shape[0]
                        and 0 <= int(c[1]) < _dist_nav.shape[1]
                     else 0.0)
                    for c in path_cells
                ) if path_cells else _MIN_PASSAGE_CELLS

                if _min_clearance < _MIN_PASSAGE_CELLS and room_goal is None:
                    print(
                        f"  [narrow] Min clearance on path: {_min_clearance:.1f} cells "
                        f"(< {_MIN_PASSAGE_CELLS}) — trying alternative standoff."
                    )
                    # Single retry: use a wider standoff from the opposite direction
                    robot._set_nav_curr_pose()
                    _alt_standoff, _ = robot.map.get_standoff_pos(
                        robot.curr_pos_on_map, cat, standoff_m=1.5)
                    _alt_path, _ = robot.plan_path_only(_alt_standoff)
                    _alt_path_cells = getattr(robot, "last_planned_path", None) or []
                    _alt_n = len(_alt_path)

                    if _alt_n == 0:
                        print(f"  [narrow] Alternative path also unreachable — skipping '{cat}'.")
                        continue

                    _alt_min_clearance = min(
                        (float(_dist_nav[int(c[0]), int(c[1])])
                         if 0 <= int(c[0]) < _dist_nav.shape[0]
                            and 0 <= int(c[1]) < _dist_nav.shape[1]
                         else 0.0)
                        for c in _alt_path_cells
                    ) if _alt_path_cells else 0.0

                    if _alt_min_clearance < _MIN_PASSAGE_CELLS:
                        print(
                            f"  [narrow] Alternative path also too narrow "
                            f"(clearance {_alt_min_clearance:.1f}) — skipping '{cat}'."
                        )
                        continue

                    # Alternative is viable — use it
                    print(
                        f"  [narrow] Alternative path accepted "
                        f"(clearance {_alt_min_clearance:.1f} cells, {_alt_n} actions)."
                    )
                    goal_pos, obj_centroid = select_safe_goal_from_path(
                        _alt_path, heatmap, robot.map.obstacles_map
                    )
                    _, planned_actions = robot.plan_path_only(goal_pos)
                    path_cells = getattr(robot, "last_planned_path", None) or []
                    n_actions = len(planned_actions)
                    already_at_goal = (n_actions == 0)

            # Show heatmap + planned path BEFORE executing so the user can see the route
            show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                     path_cells=path_cells, label=f"Path planned: {cat}")
            cv2.waitKey(800)

            if already_at_goal:
                print(f"  Already at goal for '{cat}' — proceeding with verification.")
            else:
                print(f"  Executing path ({n_actions} actions)…")
                # Execute step by step — no teleport, robot moves from its current position
                completed = execute_nav_replay(
                    robot, planned_actions, cat, rgb_map_2d, heatmap, path_cells
                )
                if not completed:
                    nav_recovery_and_replan(
                        robot, goal_pos, cat, rgb_map_2d, heatmap
                    )

            show_obs(robot, f"Arrived: {cat}")
            show_map(robot, rgb_map_2d, heatmap_2d=heatmap,
                     path_cells=path_cells, label=f"Arrived: {cat}")

            # ── Room-level navigation: just arrive, no verification needed ────
            if room_goal is not None:
                print(f"  Arrived at room '{cat}'. Done.")
                from vlmaps.utils.habitat_utils import agent_state2tf
                agent_state = robot.sim.get_agent(0).get_state()
                start_tf = agent_state2tf(agent_state)
                robot._set_nav_curr_pose()
                continue

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

            print(f"  Done. YOLOE confirmed: {_yoloe_confirmed}")

            from vlmaps.utils.habitat_utils import agent_state2tf
            agent_state = robot.sim.get_agent(0).get_state()
            start_tf = agent_state2tf(agent_state)
            robot._set_nav_curr_pose()

        print("\nInstruction complete.")

    from vlmaps.utils.yoloe_utils import shutdown_session
    shutdown_session()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
