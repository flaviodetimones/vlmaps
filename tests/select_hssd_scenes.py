"""
Phase 4 — HSSD Scene Selection
================================
Iterates over available HSSD scenes, loads each one, and reports:
  - Number of levels
  - Number of regions and their names
  - Approximate navigable area (from navmesh)
  - Saves one RGB frame per scene for visual review

Usage (inside Docker, after HSSD download):
    python tests/select_hssd_scenes.py \
        --scene_dataset_config /workspace/data/hssd-hab/hssd-hab.scene_dataset_config.json \
        --output_dir /workspace/data/hssd_scene_survey \
        [--max_scenes 20]

Output: CSV summary + one frame per scene in output_dir/frames/
"""
import argparse
import csv
import json
import os
import sys
import glob
from pathlib import Path


def survey_scene(sim, scene_id: str, dataset_dir: "Path") -> dict:
    """Return a dict with scene metadata extracted from an open simulator."""
    import json
    result = {
        "scene_id": scene_id,
        "levels": 1,   # HSSD scenes are single-level by design
        "regions": 0,
        "region_names": [],
        "navmesh_area_m2": 0.0,
        "error": None,
    }
    try:
        # Read region annotations directly from semantic_config.json
        sem_cfg = dataset_dir / "semantics" / "scenes" / f"{scene_id}.semantic_config.json"
        if sem_cfg.exists():
            with open(sem_cfg) as f:
                data = json.load(f)
            for ann in data.get("region_annotations", []):
                result["regions"] += 1
                result["region_names"].append(ann.get("name", "?"))

        # Recompute navmesh and estimate navigable area
        import habitat_sim
        nav = habitat_sim.NavMeshSettings()
        nav.set_defaults()
        nav.agent_radius = 0.1
        nav.agent_height = 1.5
        sim.recompute_navmesh(sim.pathfinder, nav)
        if sim.pathfinder.is_loaded:
            bounds = sim.pathfinder.get_bounds()
            dx = bounds[1][0] - bounds[0][0]
            dz = bounds[1][2] - bounds[0][2]
            result["navmesh_area_m2"] = round(dx * dz, 1)
    except Exception as e:
        result["error"] = str(e)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_dataset_config",
        default="/workspace/data/hssd-hab/hssd-hab.scene_dataset_config.json",
    )
    parser.add_argument("--output_dir", default="/workspace/data/hssd_scene_survey")
    parser.add_argument("--max_scenes", type=int, default=30,
                        help="Max scenes to survey (set to 0 for all)")
    args = parser.parse_args()

    if not os.path.exists(args.scene_dataset_config):
        print(f"ERROR: not found: {args.scene_dataset_config}")
        sys.exit(1)

    # Find scene instance files to get scene IDs
    dataset_dir = Path(args.scene_dataset_config).parent
    scene_files = sorted(glob.glob(str(dataset_dir / "scenes" / "*.scene_instance.json")))
    if not scene_files:
        # Try alternative structure
        scene_files = sorted(glob.glob(str(dataset_dir / "**" / "*.scene_instance.json"),
                                        recursive=True))

    if not scene_files:
        print("ERROR: no .scene_instance.json files found under", dataset_dir)
        sys.exit(1)

    scene_ids = [Path(f).stem.replace(".scene_instance", "") for f in scene_files]
    if args.max_scenes > 0:
        scene_ids = scene_ids[: args.max_scenes]

    print(f"Found {len(scene_ids)} scene(s) to survey (max={args.max_scenes})\n")

    frames_dir = Path(args.output_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.output_dir) / "scene_survey.csv"

    import habitat_sim
    import numpy as np

    rows = []
    for i, scene_id in enumerate(scene_ids):
        print(f"[{i+1}/{len(scene_ids)}] {scene_id} ...", end=" ", flush=True)

        try:
            cfg = habitat_sim.SimulatorConfiguration()
            cfg.scene_dataset_config_file = args.scene_dataset_config
            cfg.scene_id = scene_id
            cfg.enable_physics = False  # faster for survey

            rgb_spec = habitat_sim.CameraSensorSpec()
            rgb_spec.uuid = "rgb"
            rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
            rgb_spec.resolution = [240, 320]
            rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

            agent_cfg = habitat_sim.agent.AgentConfiguration()
            agent_cfg.sensor_specifications = [rgb_spec]

            sim_cfg = habitat_sim.Configuration(cfg, [agent_cfg])
            sim = habitat_sim.Simulator(sim_cfg)

            meta = survey_scene(sim, scene_id, dataset_dir)

            # Save one RGB frame
            obs = sim.get_sensor_observations()
            frame_path = frames_dir / f"{scene_id}.jpg"
            rgb = np.array(obs["rgb"])[:, :, :3]
            try:
                from PIL import Image
                Image.fromarray(rgb).save(str(frame_path))
            except ImportError:
                import cv2
                cv2.imwrite(str(frame_path), rgb[:, :, [2, 1, 0]])

            sim.close()

            flag = ""
            if meta["regions"] >= 5 and meta["navmesh_area_m2"] >= 100:
                flag = "★ CANDIDATE"
            print(f"levels={meta['levels']} regions={meta['regions']} "
                  f"area={meta['navmesh_area_m2']}m² {flag}")

        except Exception as e:
            meta = {"scene_id": scene_id, "levels": "?", "regions": "?",
                    "region_names": [], "navmesh_area_m2": 0, "error": str(e)}
            print(f"ERROR: {e}")

        meta["region_names_str"] = ", ".join(meta.get("region_names", []))
        rows.append(meta)

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["scene_id", "levels", "regions", "navmesh_area_m2",
                           "region_names_str", "error"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

    print(f"\nSurvey written to: {csv_path}")
    print(f"Frames saved to  : {frames_dir}/")

    # Print top candidates
    candidates = [r for r in rows
                  if isinstance(r.get("regions"), int)
                  and r["regions"] >= 5
                  and r.get("navmesh_area_m2", 0) >= 100
                  and not r.get("error")]
    candidates.sort(key=lambda r: -r["regions"])
    print(f"\n=== Top single-level candidates (≥5 regions) ===")
    for r in candidates[:10]:
        print(f"  {r['scene_id']:20s}  levels={r['levels']}  "
              f"regions={r['regions']}  area={r['navmesh_area_m2']}m²")


if __name__ == "__main__":
    main()
