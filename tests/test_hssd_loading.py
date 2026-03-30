"""
Phase 3.2 — HSSD Compatibility Gate
====================================
Minimal test: verify HSSD scenes load on Habitat-Sim 0.3.1.
Run BEFORE any integration work. If this fails, STOP.

Usage (inside Docker):
    python tests/test_hssd_loading.py \
        --scene_dataset_config /workspace/data/hssd-hab/hssd-hab.scene_dataset_config.json \
        --scene_id 102344280
"""
import argparse
import os
import sys


def test_hssd_basic(scene_dataset_config: str, scene_id: str) -> bool:
    import habitat_sim
    import numpy as np
    from pathlib import Path

    print(f"\n=== HSSD Compatibility Gate ===")
    print(f"scene_dataset_config : {scene_dataset_config}")
    print(f"scene_id             : {scene_id}\n")

    # 1. Simulator configuration
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file = scene_dataset_config
    cfg.scene_id = scene_id
    cfg.enable_physics = True

    # 2. Sensors (RGB + Depth)
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [480, 640]
    rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [480, 640]
    depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }

    # 3. Initialize simulator
    print("Loading simulator...")
    sim_config = habitat_sim.Configuration(cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_config)
    print("✓ Simulator initialized")

    # 4. Test rendering
    obs = sim.get_sensor_observations()
    assert "rgb" in obs, "RGB sensor missing from observations"
    assert "depth" in obs, "Depth sensor missing from observations"
    print(f"✓ RGB shape : {obs['rgb'].shape}")
    print(f"✓ Depth shape: {obs['depth'].shape}")

    # 5. Test semantic region annotations via semantic_config.json (HSSD format)
    import json as _json
    from pathlib import Path as _Path
    semantic_cfg_path = (
        _Path(scene_dataset_config).parent / "semantics" / "scenes"
        / f"{scene_id}.semantic_config.json"
    )
    if semantic_cfg_path.exists():
        with open(semantic_cfg_path) as _f:
            _sdata = _json.load(_f)
        regions = _sdata.get("region_annotations", [])
        print(f"✓ Semantic config found: {len(regions)} region(s)")
        for r in regions:
            print(f"    '{r.get('name')}' | {r.get('label')} "
                  f"| poly_pts={len(r.get('poly_loop', []))}")
        assert len(regions) > 0, "semantic_config.json has no region_annotations"
    else:
        print(f"  [info] No semantic_config.json for scene {scene_id} — "
              f"room reasoning will use LabelMe instead.")

    # 6. Test navmesh
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = 0.1
    navmesh_settings.agent_height = 1.5
    success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    assert success, "NavMesh computation failed"
    assert sim.pathfinder.is_loaded, "PathFinder reports navmesh not loaded"
    print("✓ NavMesh computed and loaded")

    # 7. Test physics / object placement
    obj_mgr = sim.get_object_template_manager()
    print(f"✓ Object template manager accessible")

    # 8. Save a test RGB frame for visual inspection
    out_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)
    frame_path = os.path.join(out_dir, "hssd_test_frame.jpg")
    try:
        from PIL import Image
        rgb = np.array(obs["rgb"])[:, :, :3]
        Image.fromarray(rgb).save(frame_path)
        print(f"✓ Test frame saved: {frame_path}")
    except ImportError:
        import cv2
        rgb_bgr = np.array(obs["rgb"])[:, :, [2, 1, 0]]
        cv2.imwrite(frame_path, rgb_bgr)
        print(f"✓ Test frame saved: {frame_path}")

    sim.close()
    print("\n=== ALL HSSD COMPATIBILITY TESTS PASSED ===\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HSSD compatibility gate test")
    parser.add_argument(
        "--scene_dataset_config",
        default="/workspace/data/hssd-hab/hssd-hab.scene_dataset_config.json",
        help="Path to hssd-hab.scene_dataset_config.json",
    )
    parser.add_argument(
        "--scene_id",
        default="102344280",
        help="HSSD scene ID (bare string, no path)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.scene_dataset_config):
        print(f"ERROR: scene_dataset_config not found: {args.scene_dataset_config}")
        print("\nDownload HSSD inside Docker with:")
        print("  python -m habitat_sim.utils.datasets_download \\")
        print("      --uids hssd-hab \\")
        print("      --data-path /workspace/data/")
        sys.exit(1)

    success = test_hssd_basic(args.scene_dataset_config, args.scene_id)
    sys.exit(0 if success else 1)
