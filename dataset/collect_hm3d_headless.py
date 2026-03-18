"""
Headless automated dataset collection for HM3D scenes.

Navigates each scene using ShortestPathFollower between random waypoints,
saving RGB + depth + semantic observations in VLMaps format:
    <scene>_N/
        rgb/NNNNNN.png
        depth/NNNNNN.npy
        semantic/NNNNNN.npy
        poses.txt          # (N, 7): px py pz qx qy qz qw

Usage (inside container):
    cd /workspace/third_party/vlmaps
    python dataset/collect_hm3d_headless.py
    # single scene:
    python dataset/collect_hm3d_headless.py scene_names='[00800-TEEsavR23oF]'
"""

import os
from collections import defaultdict
from pathlib import Path

import habitat_sim
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from vlmaps.utils.habitat_utils import get_obj2cls_dict, save_obs, save_states

_VALID_ACTIONS = {"move_forward", "turn_left", "turn_right"}


def _make_cfg(scene_path: str, cfg: DictConfig) -> habitat_sim.Configuration:
    """
    Minimal make_cfg for dataset collection.
    Avoids the back_color_sensor in the original make_cfg which has an
    orientation type bug with habitat-sim 0.3.1 (expects magnum.Vector3).
    """
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False

    h, w = cfg.data_cfg.resolution.h, cfg.data_cfg.resolution.w
    pos = [0.0, cfg.data_cfg.camera_height, 0.0]
    sensors = []

    if cfg.data_cfg.rgb:
        s = habitat_sim.CameraSensorSpec()
        s.uuid = "color_sensor"
        s.sensor_type = habitat_sim.SensorType.COLOR
        s.resolution = [h, w]
        s.position = pos
        sensors.append(s)

    if cfg.data_cfg.depth:
        s = habitat_sim.CameraSensorSpec()
        s.uuid = "depth_sensor"
        s.sensor_type = habitat_sim.SensorType.DEPTH
        s.resolution = [h, w]
        s.position = pos
        sensors.append(s)

    if cfg.data_cfg.semantic:
        s = habitat_sim.CameraSensorSpec()
        s.uuid = "semantic_sensor"
        s.sensor_type = habitat_sim.SensorType.SEMANTIC
        s.resolution = [h, w]
        s.position = pos
        sensors.append(s)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors
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

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def _make_sim_settings(scene_path: str, cfg: DictConfig) -> dict:
    """Settings dict used by save_obs to know which sensors are active."""
    return {
        "scene": scene_path,
        "color_sensor": cfg.data_cfg.rgb,
        "depth_sensor": cfg.data_cfg.depth,
        "semantic_sensor": cfg.data_cfg.semantic,
    }


def collect_scene(scene_name: str, save_dir: Path, config: DictConfig) -> int:
    scene_short = scene_name.split("-")[1]  # 00800-TEEsavR23oF -> TEEsavR23oF
    scene_path = (
        Path(config.data_paths.habitat_scene_dir)
        / scene_name
        / f"{scene_short}.basis.glb"
    )

    if not scene_path.exists():
        print(f"  [skip] scene file not found: {scene_path}")
        return 0

    save_dir.mkdir(parents=True, exist_ok=True)
    sim_settings = _make_sim_settings(str(scene_path), config)

    sim = habitat_sim.Simulator(_make_cfg(scene_path=str(scene_path), cfg=config))
    # Use defaultdict so missing IDs (e.g. when semantic SSD fails to load)
    # don't raise KeyError in cvt_obj_id_2_cls_id
    obj2cls = defaultdict(lambda: (0, "unknown"), get_obj2cls_dict(sim))
    obj2cls[0] = (0, "void")
    agent = sim.initialize_agent(0)

    # GreedyGeodesicFollower (habitat-sim 0.3.x replaces ShortestPathFollower)
    follower = habitat_sim.nav.GreedyGeodesicFollower(
        sim.pathfinder, agent, goal_radius=0.5
    )

    # Sample random navigable waypoints
    np.random.seed(42)
    waypoints = [
        sim.pathfinder.get_random_navigable_point()
        for _ in range(config.num_waypoints)
    ]

    frame_id = 0
    agent_states = []

    # Place agent at first waypoint
    init_state = habitat_sim.AgentState()
    init_state.position = waypoints[0]
    agent.set_state(init_state)

    pbar = tqdm(range(len(waypoints) - 1), desc=scene_name, leave=False)
    for i in pbar:
        goal = waypoints[i + 1]
        follower.reset()

        for _ in range(config.max_steps_per_leg):
            action = follower.next_action_along(goal)

            # None means goal reached
            if action is None or action not in _VALID_ACTIONS:
                break

            sim.step(action)

            if frame_id % config.sample_every == 0:
                obs = sim.get_sensor_observations(0)
                save_obs(save_dir, sim_settings, obs, frame_id // config.sample_every, obj2cls)
                agent_states.append(agent.get_state())

            frame_id += 1

        pbar.set_postfix(frames=len(agent_states))

    save_states(save_dir, agent_states)
    sim.close()
    return len(agent_states)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="collect_hm3d.yaml",
)
def main(config: DictConfig) -> None:
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    dataset_dir = Path(config.data_paths.vlmaps_data_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for scene_name in config.scene_names:
        # Find next available index (don't overwrite existing data)
        idx = 1
        while (dataset_dir / f"{scene_name}_{idx}").exists():
            idx += 1
        save_dir = dataset_dir / f"{scene_name}_{idx}"

        print(f"\nScene: {scene_name}  ->  {save_dir}")
        n_frames = collect_scene(scene_name, save_dir, config)
        print(f"  Saved {n_frames} frames")


if __name__ == "__main__":
    main()
