"""
Convert a LabelMe JSON annotation file to VLMaps room_map format.

Usage:
    python application/labelme_to_room_map.py \
        --json  /workspace/data/vlmaps_dataset/<scene>/room_map/room_labels.json \
        --scene /workspace/data/vlmaps_dataset/<scene>

The script reads the LabelMe polygon annotations, rasterises each room polygon
onto a grid that matches the obstacle map, runs connected-component analysis to
build room regions, and saves the result with save_room_map().
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _load_labelme(json_path: str):
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data


def _rasterise(shapes, img_h: int, img_w: int):
    """Rasterise LabelMe polygon shapes → room_map (H×W int array, 0=unknown)."""
    labels = []
    for s in shapes:
        lbl = s["label"].strip()
        if lbl not in labels:
            labels.append(lbl)

    room_map = np.zeros((img_h, img_w), dtype=np.int32)
    max_scores = np.zeros((img_h, img_w), dtype=np.float32)

    for s in shapes:
        lbl = s["label"].strip()
        cat_idx = labels.index(lbl) + 1  # 0 = unlabelled
        pts = np.array(s["points"], dtype=np.int32).reshape((-1, 1, 2))
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        room_map[mask == 1] = cat_idx
        max_scores[mask == 1] = 1.0

    return room_map, labels, max_scores


def _extract_regions(room_map: np.ndarray, labels, min_region_size: int = 50):
    """Build the lightweight room-region structure expected by VLMaps."""
    regions = {}
    for idx, label in enumerate(labels):
        binary = (room_map == idx).astype(np.uint8)
        n_labels, comps, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        cat_regions = []
        for comp_id in range(1, n_labels):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area < min_region_size:
                continue
            cy = int(round(centroids[comp_id][1]))
            cx = int(round(centroids[comp_id][0]))
            cat_regions.append(
                {
                    "centroid": [cy, cx],
                    "area": area,
                    "quality": float(area) / float(room_map.shape[0] * room_map.shape[1]),
                }
            )
        cat_regions.sort(key=lambda r: r["quality"], reverse=True)
        if cat_regions:
            regions[label] = cat_regions
    return regions


def convert(json_path: str, scene_dir: str, min_region_size: int = 50, preview: bool = True):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from vlmaps.utils.room_map_utils import (
        save_room_map,
        render_room_map,
    )

    data = _load_labelme(json_path)
    img_h = data["imageHeight"]
    img_w = data["imageWidth"]
    shapes = [s for s in data["shapes"] if s["shape_type"] == "polygon"]

    if not shapes:
        print("ERROR: No polygon shapes found in the LabelMe JSON.")
        sys.exit(1)

    room_map_raw, labels, _max_scores = _rasterise(shapes, img_h, img_w)

    # Shift indices so 0 = first category (room_map_utils convention)
    room_map = (room_map_raw - 1).astype(np.int32)
    room_map[room_map_raw == 0] = -1  # keep unlabelled as -1 for safety

    regions = _extract_regions(room_map, labels, min_region_size=min_region_size)

    # Visualisation overlay. Keep -1 as "unlabelled" in the saved map; only
    # the renderer ignores it naturally because no category index matches -1.
    viz = render_room_map(room_map, labels, regions)

    save_room_map(scene_dir, room_map, labels, regions, viz_img=viz)

    print(f"\nRoom map saved to: {Path(scene_dir) / 'room_map'}/")
    print(f"  Categories ({len(labels)}): {labels}")
    n_regions = sum(len(v) for v in regions.values())
    print(f"  Regions    ({n_regions}):")
    for label, infos in regions.items():
        for idx, info in enumerate(infos, start=1):
            print(
                f"    [{label}#{idx}] {label:25s}  "
                f"centroid={info['centroid']}  area={info['area']}"
            )

    if preview:
        scale = max(1, min(4, 800 // max(img_h, img_w)))
        preview_img = cv2.resize(viz, (img_w * scale, img_h * scale),
                                 interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Room map preview (press any key to close)", preview_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="LabelMe JSON → VLMaps room_map")
    parser.add_argument("--json",   required=True,
                        help="Path to the LabelMe .json file")
    parser.add_argument("--scene",  required=True,
                        help="Scene dataset directory (contains obstacle_map.png)")
    parser.add_argument("--min-region-size", type=int, default=50,
                        help="Minimum pixel area for a room region (default: 50)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Do not open the OpenCV room-map preview window")
    args = parser.parse_args()

    if not Path(args.json).exists():
        print(f"ERROR: JSON not found: {args.json}")
        sys.exit(1)
    if not Path(args.scene).is_dir():
        print(f"ERROR: Scene dir not found: {args.scene}")
        sys.exit(1)

    convert(args.json, args.scene, args.min_region_size, preview=not args.no_preview)


if __name__ == "__main__":
    main()
