"""
Generate and save the top-down map images for a scene:
  - obstacle_map.png      : binary (white=obstacle, black=free)
  - topdown_rgb.png       : RGB top-down view from the VLMap point cloud
  - topdown_labeled.png   : RGB blended with obstacle overlay (for LabelMe)

The labeled image is what gets opened in LabelMe for room annotation.

Usage:
    python application/generate_obstacle_map_png.py data_paths=docker scene_id=0
    python application/generate_obstacle_map_png.py data_paths=hssd dataset_type=hssd scene_id=0 \
        +scene_dataset_config_file=/workspace/data/hssd-hab/hssd-hab.scene_dataset_config.json
"""

import hydra
import numpy as np
import cv2
from omegaconf import DictConfig
from pathlib import Path


def _build_navmesh_obstacle_map(sim, map_obj, poses_path: Path) -> None:
    """Replace the voxel-height obstacle map with one derived from Habitat's navmesh.

    For HSSD scenes the depth-based height approach fails because almost every
    floor cell also has obstacle voxels above it (robot navigated near walls and
    furniture). The navmesh correctly encodes where the robot can stand regardless
    of observation coverage. Updates map_obj in-place.
    """
    from vlmaps.utils.mapping_utils import cvt_pose_vec2tf

    gs = map_obj.gs
    cs = map_obj.cs

    # Crop bounds: every grid cell that has at least one mapped voxel
    any_mapped = np.any(map_obj.occupied_ids >= 0, axis=2)
    x_idx, y_idx = np.where(any_mapped)
    if len(x_idx) == 0:
        print("[navmesh map] No mapped voxels found; skipping.")
        return
    rmin, rmax = int(np.min(x_idx)), int(np.max(x_idx))
    cmin, cmax = int(np.min(y_idx)), int(np.max(y_idx))
    print(f"[navmesh map] Mapped area rows [{rmin},{rmax}], cols [{cmin},{cmax}]")

    # Coordinate conversion: grid (row,col) → Habitat world at floor level
    #   grid  → relative mobile-base (x_fwd, y_left, 0)
    #   relative mobile-base → Habitat relative: inv(base_transform) @ p
    #   Habitat relative → Habitat absolute: init_hab_tf @ p
    poses = np.loadtxt(poses_path)
    init_hab_tf = cvt_pose_vec2tf(poses[0])         # 4×4
    base_rot_inv = map_obj.base_transform[:3, :3].T  # orthogonal ⟹ inv = T

    rows_arr = np.arange(rmin, rmax + 1)
    cols_arr = np.arange(cmin, cmax + 1)
    R, C = np.meshgrid(rows_arr, cols_arr, indexing="ij")
    x_fwd  = (gs / 2 - R) * cs
    y_left = (gs / 2 - C) * cs

    pts_base = np.stack([x_fwd.ravel(), y_left.ravel(), np.zeros(R.size)], axis=1)
    pts_hab_rel = (base_rot_inv @ pts_base.T).T
    pts_hab_abs = (init_hab_tf[:3, :3] @ pts_hab_rel.T).T + init_hab_tf[:3, 3]

    print(f"[navmesh map] Querying navmesh for {len(pts_hab_abs):,} cells ...")
    navigable = np.array([sim.pathfinder.is_navigable(p) for p in pts_hab_abs], dtype=bool)
    print(f"[navmesh map] Navigable: {navigable.sum():,} / {len(navigable):,}")

    obstacle_map = np.zeros((gs, gs), dtype=np.uint8)
    obstacle_map[rmin:rmax + 1, cmin:cmax + 1] = navigable.reshape(R.shape).astype(np.uint8)

    # If the navmesh did not include static objects (older habitat_sim), mark
    # rigid-object footprints by projecting their world-space AABBs onto the grid.
    n_nav_before = int(navigable.sum())
    _mark_rigid_object_footprints(sim, obstacle_map, gs, cs, init_hab_tf, map_obj.base_transform)
    n_nav_after = int(np.sum(obstacle_map[rmin:rmax + 1, cmin:cmax + 1]))
    if n_nav_before != n_nav_after:
        print(f"[navmesh map] After object footprints: {n_nav_after:,} navigable cells "
              f"(removed {n_nav_before - n_nav_after:,})")

    map_obj.obstacles_map = obstacle_map
    map_obj.rmin = rmin
    map_obj.rmax = rmax
    map_obj.cmin = cmin
    map_obj.cmax = cmax
    map_obj.obstacles_cropped = obstacle_map[rmin:rmax + 1, cmin:cmax + 1]


def _mark_rigid_object_footprints(sim, obstacle_map: np.ndarray, gs: int, cs: float,
                                   init_hab_tf: np.ndarray, base_transform: np.ndarray) -> None:
    """Mark rigid-object footprints as non-navigable on obstacle_map (in-place).

    Projects each object's world-space AABB onto the 2D grid and sets those
    cells to 0 (obstacle). Used as a fallback when the navmesh was computed
    without include_static_objects support.
    """
    try:
        rom = sim.get_rigid_object_manager()
        handles = rom.get_object_handles()
    except Exception as e:
        print(f"[navmesh map] Rigid object manager unavailable ({e}); skipping footprints.")
        return

    if not handles:
        print("[navmesh map] No rigid objects found in scene.")
        return

    base_rot = base_transform[:3, :3]
    init_rot_inv = np.linalg.inv(init_hab_tf[:3, :3])
    init_pos = init_hab_tf[:3, 3]

    def hab_to_grid(p_hab: np.ndarray):
        """Habitat world point → (row, col) grid indices."""
        p_rel = p_hab - init_pos
        p_init = init_rot_inv @ p_rel
        p_base = base_rot @ p_init
        row = int(gs / 2 - p_base[0] / cs)
        col = int(gs / 2 - p_base[1] / cs)
        return row, col

    marked = 0
    for handle in handles:
        try:
            obj = rom.get_object_by_handle(handle)
            if obj is None:
                continue
            # Local AABB
            local_bb = obj.aabb
            min_l = np.array([local_bb.min[0], local_bb.min[1], local_bb.min[2]])
            max_l = np.array([local_bb.max[0], local_bb.max[1], local_bb.max[2]])
            # World transform as numpy 4×4
            T = np.array(obj.transformation)
            if T.shape != (4, 4):
                T = T.reshape(4, 4)
            # 8 corners of local AABB → world space
            lx, ly, lz = zip(*[(min_l[0], min_l[1], min_l[2]),
                                (max_l[0], min_l[1], min_l[2]),
                                (min_l[0], max_l[1], min_l[2]),
                                (max_l[0], max_l[1], min_l[2]),
                                (min_l[0], min_l[1], max_l[2]),
                                (max_l[0], min_l[1], max_l[2]),
                                (min_l[0], max_l[1], max_l[2]),
                                (max_l[0], max_l[1], max_l[2])])
            corners_l = np.array([lx, ly, lz])  # (3, 8)
            corners_w = (T[:3, :3] @ corners_l).T + T[:3, 3]  # (8, 3)
            # Grid bounding rect
            grid_pts = [hab_to_grid(c) for c in corners_w]
            r_min = max(0, min(p[0] for p in grid_pts))
            r_max = min(gs - 1, max(p[0] for p in grid_pts))
            c_min = max(0, min(p[1] for p in grid_pts))
            c_max = min(gs - 1, max(p[1] for p in grid_pts))
            if r_min <= r_max and c_min <= c_max:
                obstacle_map[r_min:r_max + 1, c_min:c_max + 1] = 0
                marked += 1
        except Exception:
            continue

    print(f"[navmesh map] Marked footprints for {marked}/{len(handles)} rigid objects.")


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="object_goal_navigation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    from vlmaps.robot.habitat_lang_robot import HabitatLanguageRobot
    from vlmaps.utils.matterport3d_categories import get_categories
    from vlmaps.utils.visualize_utils import pool_3d_rgb_to_2d

    robot = HabitatLanguageRobot(config)
    robot.setup_scene(config.scene_id)
    _dataset_type = str(getattr(config, "dataset_type", "mp3d"))
    robot.map.init_categories(get_categories(_dataset_type))

    scene_dir = Path(robot.vlmaps_data_save_dirs[config.scene_id])

    # For HSSD: the voxel-height obstacle map is unreliable (robot navigates near
    # walls so almost every floor cell also has obstacle voxels). Replace it with
    # a navmesh-based map, which correctly reflects navigable space.
    if _dataset_type == "hssd":
        _build_navmesh_obstacle_map(robot.sim, robot.map, scene_dir / "poses.txt")

    # 1. Binary obstacle map
    obs = robot.map.obstacles_cropped.astype(np.uint8) * 255
    obs_path = scene_dir / "obstacle_map.png"
    cv2.imwrite(str(obs_path), obs)
    print(f"Saved: {obs_path}  ({obs.shape[1]}x{obs.shape[0]} px)")

    # 2. RGB top-down map from point cloud
    rgb_map = pool_3d_rgb_to_2d(robot.map.grid_rgb, robot.map.grid_pos, robot.map.gs)
    # pool_3d_rgb_to_2d returns RGB float [0,1] or uint8 — normalise to uint8
    if rgb_map.dtype != np.uint8:
        rgb_map = (np.clip(rgb_map, 0, 1) * 255).astype(np.uint8)
    rgb_bgr = cv2.cvtColor(rgb_map, cv2.COLOR_RGB2BGR)
    rgb_path = scene_dir / "topdown_rgb.png"
    cv2.imwrite(str(rgb_path), rgb_bgr)
    print(f"Saved: {rgb_path}  ({rgb_bgr.shape[1]}x{rgb_bgr.shape[0]} px)")

    # 3. Labeled composite for LabelMe.
    # Highlight navigable floor in red, keep furniture/layout visible, and
    # darken non-navigable/outside areas so room boundaries are easier to draw.
    # Crop RGB to match obstacle map dimensions
    rmin, rmax = robot.map.rmin, robot.map.rmax
    cmin, cmax = robot.map.cmin, robot.map.cmax
    rgb_crop = rgb_bgr[rmin:rmax, cmin:cmax]
    if rgb_crop.shape[:2] != obs.shape[:2]:
        rgb_crop = cv2.resize(rgb_crop, (obs.shape[1], obs.shape[0]),
                              interpolation=cv2.INTER_LINEAR)

    # obstacle_map.png stores navigable cells as white (255) and blocked cells
    # as black (0). For annotation, use a semi-transparent red tint over the
    # navigable area instead of flooding the whole scene, so furniture remains
    # readable. Darken non-navigable cells and add a crisp boundary outline.
    nav_mask = obs > 128
    blocked_mask = ~nav_mask

    base = cv2.convertScaleAbs(rgb_crop, alpha=1.05, beta=8)
    labeled = (base * 0.78).astype(np.uint8)

    nav_overlay = np.zeros_like(base)
    nav_overlay[:, :] = (20, 25, 185)  # BGR: soft red, not pure saturated red
    labeled[nav_mask] = cv2.addWeighted(
        base[nav_mask], 0.50, nav_overlay[nav_mask], 0.50, 0
    )

    labeled[blocked_mask] = (base[blocked_mask] * 0.18).astype(np.uint8)

    boundary = cv2.morphologyEx(nav_mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    labeled[boundary] = (235, 235, 235)

    labeled_path = scene_dir / "topdown_labeled.png"
    cv2.imwrite(str(labeled_path), labeled)
    print(f"Saved: {labeled_path}  ({labeled.shape[1]}x{labeled.shape[0]} px)")
    print(f"\nUse topdown_labeled.png in LabelMe for room annotation.")

    robot.sim.close()


if __name__ == "__main__":
    main()
