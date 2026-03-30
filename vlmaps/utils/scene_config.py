"""
scene_config.py
===============
SceneConfig dataclass — centralises all dataset-specific parameters so the
rest of the pipeline can stay scene-agnostic.

Usage
-----
    from vlmaps.utils.scene_config import SceneConfig, from_hydra_config

    scene_cfg = from_hydra_config(hydra_cfg)
    robot.setup_scene_from_config(scene_cfg)
"""

from dataclasses import dataclass, field
from typing import Optional
from omegaconf import DictConfig


@dataclass
class SceneConfig:
    # "mp3d", "hm3d", or "hssd"
    dataset_type: str = "mp3d"

    # Integer index into vlmaps_data_save_dirs
    scene_id: int = 0

    # Human-readable scene name, e.g. "mJXqzFtmKg4" or "102344280"
    scene_name: str = ""

    # Full path to the .glb file (MP3D/HM3D) OR bare scene ID string (HSSD)
    scene_path: str = ""

    # Path to hssd-hab.scene_dataset_config.json — None for MP3D/HM3D
    scene_dataset_config_file: Optional[str] = None

    # Directory containing collected RGB-D data and the built VLMap
    vlmap_data_dir: str = ""

    # "labelme" (manual LabelMe annotation) or "semantic_scene_api" (automatic)
    room_labels_source: str = "labelme"

    # Path to placements.json for custom object placement — empty = disabled
    placements_json: str = ""

    # Category vocabulary: "mp3d" (42 cats) or "hssd" (common furniture)
    category_mode: str = "mp3d"


def from_hydra_config(cfg: DictConfig) -> SceneConfig:
    """Build a SceneConfig from a Hydra DictConfig (object_goal_navigation_cfg)."""
    dataset_type = str(getattr(cfg, "dataset_type", "mp3d"))
    scene_dataset_config_file = getattr(cfg, "scene_dataset_config_file", None)
    if scene_dataset_config_file:
        scene_dataset_config_file = str(scene_dataset_config_file)

    category_mode = dataset_type if dataset_type in ("mp3d", "hssd", "hm3d") else "mp3d"

    return SceneConfig(
        dataset_type=dataset_type,
        scene_id=int(cfg.scene_id),
        scene_dataset_config_file=scene_dataset_config_file,
        category_mode=category_mode,
    )
