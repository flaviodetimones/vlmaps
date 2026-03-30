"""
Phase 5 — Automated HSSD Data Collection
==========================================
Drives the agent through an HSSD scene via random pathfinder navigation,
saving RGB + depth frames + poses in the same format expected by VLMapBuilder:

    <output_dir>/
        rgb/      000000.png  000001.png  ...
        depth/    000000.npy  000001.npy  ...
        poses.txt             (N x 7: x y z qx qy qz qw)

Only frames that differ by at least MIN_DIST metres OR MIN_ROT degrees from the
previous saved frame are kept — same filtering strategy as the MP3D collection.

Usage (inside Docker):
    cd /workspace/third_party/vlmaps
    python dataset/collect_hssd_dataset.py \\
        --scene_dataset_config /workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json \\
        --scene_id 102344280 \\
        --output_dir /workspace/data/vlmaps_dataset/102344280_1 \\
        --n_frames 2000

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
from scipy.spatial.transform import Rotation as R


# ── Tuneable parameters ───────────────────────────────────────────────────────
MIN_DIST = 0.10      # metres between saved frames
MIN_ROT  = 5.0       # degrees between saved frames
SENSOR_HEIGHT = 1.5  # metres (same as interactive_object_nav.py)
WIDTH    = 1080
HEIGHT   = 720
# ─────────────────────────────────────────────────────────────────────────────


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


def quat_xyzw(q) -> np.ndarray:
    return np.array([q.x, q.y, q.z, q.w], dtype=np.float64)


def angle_between_quats(q1: np.ndarray, q2: np.ndarray) -> float:
    """Return the rotation angle in degrees between two unit quaternions (xyzw)."""
    dot = abs(float(np.dot(q1, q2)))
    dot = min(1.0, dot)
    return np.degrees(2.0 * np.arccos(dot))


def save_frame(obs, state, rgb_dir: Path, depth_dir: Path,
               frame_id: int, poses_list: list) -> None:
    rgb = obs["color_sensor"][:, :, :3]
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(rgb_dir / f"{frame_id:06d}.png"), rgb_bgr)

    depth = obs["depth_sensor"]
    np.save(str(depth_dir / f"{frame_id:06d}.npy"), depth)

    pos = state.position
    quat = quat_xyzw(state.rotation)
    poses_list.append([pos[0], pos[1], pos[2],
                       quat[0], quat[1], quat[2], quat[3]])


def navigate_to(sim, agent, target: np.ndarray, poses_list: list,
                rgb_dir: Path, depth_dir: Path, frame_id: int,
                last_pos: np.ndarray, last_quat: np.ndarray,
                n_frames_target: int) -> tuple:
    """
    Follow the shortest path from current position to target, saving frames.
    Returns (frame_id, last_pos, last_quat).
    """
    follower = habitat_sim.GreedyGeodesicFollower(
        sim.pathfinder, agent,
        goal_radius=0.5,
        stop_key=None,
        forward_key="move_forward",
        left_key="turn_left",
        right_key="turn_right",
    )

    max_steps = 2000
    for _ in range(max_steps):
        if frame_id >= n_frames_target:
            break
        try:
            action = follower.next_action_along(target)
        except Exception:
            break
        if action is None:
            break

        obs = sim.step(action)
        state = agent.get_state()
        pos = np.array(state.position)
        quat = quat_xyzw(state.rotation)

        dist = float(np.linalg.norm(pos - last_pos))
        rot = angle_between_quats(quat, last_quat)

        if dist >= MIN_DIST or rot >= MIN_ROT:
            save_frame(obs, state, rgb_dir, depth_dir, frame_id, poses_list)
            frame_id += 1
            last_pos = pos
            last_quat = quat

            if frame_id % 100 == 0:
                print(f"  {frame_id}/{n_frames_target} frames saved...")

    return frame_id, last_pos, last_quat


def main():
    parser = argparse.ArgumentParser(description="Automated HSSD dataset collection")
    parser.add_argument("--scene_dataset_config",
                        default="/workspace/data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json")
    parser.add_argument("--scene_id", default="102344280")
    parser.add_argument("--output_dir",
                        default="/workspace/data/vlmaps_dataset_hssd/102344280_1")
    parser.add_argument("--n_frames", type=int, default=2000,
                        help="Target number of frames to collect")
    args = parser.parse_args()

    if not os.path.exists(args.scene_dataset_config):
        print(f"ERROR: scene_dataset_config not found: {args.scene_dataset_config}")
        sys.exit(1)

    out = Path(args.output_dir)
    rgb_dir   = out / "rgb"
    depth_dir = out / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scene   : {args.scene_id}")
    print(f"Output  : {out}")
    print(f"Target  : {args.n_frames} frames")
    print(f"Filters : MIN_DIST={MIN_DIST}m  MIN_ROT={MIN_ROT}°")
    print()

    cfg = make_cfg(args.scene_dataset_config, args.scene_id)
    sim = habitat_sim.Simulator(cfg)
    agent = sim.initialize_agent(0)

    # Recompute navmesh
    nav_settings = habitat_sim.NavMeshSettings()
    nav_settings.set_defaults()
    nav_settings.agent_radius = 0.1
    nav_settings.agent_height = 1.5
    sim.recompute_navmesh(sim.pathfinder, nav_settings)
    assert sim.pathfinder.is_loaded, "NavMesh not loaded — aborting"
    print("✓ NavMesh loaded")

    # Random starting position
    start_pos = sim.pathfinder.get_random_navigable_point()
    state = habitat_sim.AgentState()
    state.position = start_pos
    agent.set_state(state)

    poses_list = []
    frame_id = 0
    last_pos  = np.array(start_pos)
    last_quat = quat_xyzw(agent.get_state().rotation)

    # Save initial frame
    obs = sim.get_sensor_observations()
    save_frame(obs, agent.get_state(), rgb_dir, depth_dir, frame_id, poses_list)
    frame_id += 1
    print(f"✓ Initial frame saved. Starting navigation...")

    # Random walk: repeatedly pick a new random navigable point and follow path
    while frame_id < args.n_frames:
        target = sim.pathfinder.get_random_navigable_point()
        frame_id, last_pos, last_quat = navigate_to(
            sim, agent, target, poses_list,
            rgb_dir, depth_dir, frame_id, last_pos, last_quat,
            args.n_frames,
        )

    # Save poses
    poses = np.array(poses_list, dtype=np.float64)
    np.savetxt(str(out / "poses.txt"), poses)
    print(f"\n✓ Collection complete.")
    print(f"  Frames saved : {len(poses_list)}")
    print(f"  poses.txt    : {out / 'poses.txt'}")

    sim.close()


if __name__ == "__main__":
    main()
