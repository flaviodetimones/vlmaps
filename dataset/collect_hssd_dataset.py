"""
Phase 5 — Interactive HSSD Data Collection
============================================
Manual keyboard-driven navigation through an HSSD scene,
saving RGB + depth frames + poses in the same format expected by VLMapBuilder:

    <output_dir>/
        rgb/      000000.png  000001.png  ...
        depth/    000000.npy  000001.npy  ...
        poses.txt             (N x 7: x y z qx qy qz qw)

Controls:
    w   — move forward
    a   — turn left
    d   — turn right
    s   — save current frame
    q   — quit and write poses.txt

Usage (inside Docker):
    cd /workspace/third_party/vlmaps
    python dataset/collect_hssd_dataset.py \\
        --scene_dataset_config /workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json \\
        --scene_id 102344280

After collection, build the VLMap with:
    python application/create_map.py data_paths=hssd scene_id=0
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import habitat_sim
import numpy as np


SENSOR_HEIGHT = 1.5  # metres (same as interactive_object_nav.py)
WIDTH  = 1080
HEIGHT = 720


def make_cfg(scene_dataset_config: str, scene_id: str) -> habitat_sim.Configuration:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = scene_dataset_config
    sim_cfg.scene_id = scene_id
    sim_cfg.enable_physics = False

    color_spec = habitat_sim.CameraSensorSpec()
    color_spec.uuid = "color_sensor"
    color_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_spec.resolution = [HEIGHT, WIDTH]
    color_spec.position = [0.0, SENSOR_HEIGHT, 0.0]
    color_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [HEIGHT, WIDTH]
    depth_spec.position = [0.0, SENSOR_HEIGHT, 0.0]
    depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [color_spec, depth_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.1)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def save_frame(obs, state, rgb_dir: Path, depth_dir: Path,
               frame_id: int, poses_list: list) -> None:
    rgb = obs["color_sensor"][:, :, :3]
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(rgb_dir / f"{frame_id:06d}.png"), rgb_bgr)

    depth = obs["depth_sensor"]
    np.save(str(depth_dir / f"{frame_id:06d}.npy"), depth)

    pos = state.position
    q = state.rotation
    poses_list.append([pos[0], pos[1], pos[2], q.x, q.y, q.z, q.w])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dataset_config",
                        default="/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json")
    parser.add_argument("--scene_id", default="102344280")
    parser.add_argument("--output_dir",
                        default="/workspace/data/vlmaps_dataset_hssd/102344280_1")
    args = parser.parse_args()

    if not os.path.exists(args.scene_dataset_config):
        print(f"ERROR: not found: {args.scene_dataset_config}")
        sys.exit(1)

    out = Path(args.output_dir)
    rgb_dir   = out / "rgb"
    depth_dir = out / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scene  : {args.scene_id}")
    print(f"Output : {out}")
    print()
    print("Controls:")
    print("  w — move forward")
    print("  a — turn left")
    print("  d — turn right")
    print("  s — save current frame")
    print("  q — quit and save poses.txt")
    print()

    cfg = make_cfg(args.scene_dataset_config, args.scene_id)
    sim = habitat_sim.Simulator(cfg)
    agent = sim.initialize_agent(0)

    state = habitat_sim.AgentState()
    state.position = sim.pathfinder.get_random_navigable_point()
    agent.set_state(state)

    poses_list = []
    frame_id = 0

    while True:
        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"][:, :, :3]
        display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # HUD
        cv2.putText(display, f"Frames saved: {frame_id}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "w=fwd  a=left  d=right  s=save  q=quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("HSSD Collection", display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("w"):
            agent.act("move_forward")
        elif key == ord("a"):
            agent.act("turn_left")
        elif key == ord("d"):
            agent.act("turn_right")
        elif key == ord("s"):
            obs = sim.get_sensor_observations()
            save_frame(obs, agent.get_state(), rgb_dir, depth_dir, frame_id, poses_list)
            print(f"Saved frame {frame_id:06d}")
            frame_id += 1
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

    if poses_list:
        poses = np.array(poses_list, dtype=np.float64)
        np.savetxt(str(out / "poses.txt"), poses)
        print(f"\n✓ Saved {len(poses_list)} frames to {out}")
        print(f"  poses.txt written.")
    else:
        print("No frames saved.")

    sim.close()


if __name__ == "__main__":
    main()
