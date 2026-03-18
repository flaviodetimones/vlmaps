"""
build_room_map.py
=================
Pre-label the semantic regions of a scene using VLMap heatmaps (Step A)
and optionally refine labels with GPT-4o Vision (Step B).

Usage (inside container)
------------------------
    cd /workspace/third_party/vlmaps

    # Build with VLMap segmentation only:
    python application/build_room_map.py data_paths=docker scene_id=0

    # Build + GPT-4o Vision refinement:
    python application/build_room_map.py data_paths=docker scene_id=0 refine=true

Output
------
    <vlmaps_data_dir>/<scene>/room_map/room_map.npy
    <vlmaps_data_dir>/<scene>/room_map/regions.json
    <vlmaps_data_dir>/<scene>/room_map/room_map_viz.png
"""

import os

import cv2
import hydra
from omegaconf import DictConfig
from pathlib import Path

from vlmaps.robot.habitat_lang_robot import HabitatLanguageRobot
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.room_map_utils import (
    ROOM_CATEGORIES,
    apply_corrections,
    build_room_segmentation,
    compute_room_masks,
    refine_with_gpt4v,
    render_room_map,
    save_room_map,
)
from vlmaps.utils.visualize_utils import pool_3d_rgb_to_2d


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="object_goal_navigation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    refine = getattr(config, "refine", False)

    print("\n── Loading scene ────────────────────────────────────────────────")
    robot = HabitatLanguageRobot(config)
    robot.setup_scene(config.scene_id)
    robot.map.init_categories(mp3dcat.copy())

    scene_dir = robot.vlmaps_data_save_dirs[config.scene_id]
    print(f"Scene: {scene_dir.name}")

    # ── Step A: compute per-room masks ────────────────────────────────────
    print("\n── Step A: Computing room masks from VLMap ──────────────────────")
    masks = compute_room_masks(robot, ROOM_CATEGORIES)

    print("\n── Building room segmentation ───────────────────────────────────")
    room_map, categories, regions = build_room_segmentation(masks, min_area=30)

    detected = [c for c in categories if c in regions]
    print(f"  Detected rooms: {detected}")

    # ── Render ────────────────────────────────────────────────────────────
    rgb_bg = pool_3d_rgb_to_2d(robot.map.grid_rgb, robot.map.grid_pos, robot.map.gs)
    gs = robot.map.gs
    scale = max(1, 600 // gs)
    viz = render_room_map(room_map, categories, regions, rgb_bg=rgb_bg, scale=scale)

    # ── Step B: GPT-4o Vision refinement (optional) ───────────────────────
    if refine:
        openai_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            print("\n  [warn] OPENAI_API_KEY not set — skipping GPT-4o refinement.")
        else:
            print("\n── Step B: GPT-4o Vision refinement ─────────────────────────────")
            result = refine_with_gpt4v(viz, categories, regions, openai_key)
            print(f"  Confidence : {result.get('confidence', '?')}")
            print(f"  Notes      : {result.get('notes', '')}")
            corrections = result.get("corrections", [])
            if corrections:
                print(f"  Applying {len(corrections)} correction(s)...")
                room_map, regions = apply_corrections(
                    room_map, regions, categories, corrections
                )
                # Re-render after corrections
                viz = render_room_map(
                    room_map, categories, regions, rgb_bg=rgb_bg, scale=scale
                )
            else:
                print("  No corrections needed.")

    # ── Save ──────────────────────────────────────────────────────────────
    print("\n── Saving ───────────────────────────────────────────────────────")
    save_room_map(scene_dir, room_map, categories, regions, viz_img=viz)

    # Show result
    cv2.imshow("Room Map", viz)
    print("\nPress any key in the window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nDone.")


if __name__ == "__main__":
    main()
