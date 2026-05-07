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
from collections import deque
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


def _build_voronoi_room_map(
    room_map: np.ndarray,
    *,
    scene_dir,
    max_distance_cells: int = 50,
    domain_threshold: int = 20,
    domain=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand LabelMe labels through the visible indoor domain, not through walls.

    ``room_map`` remains the authoritative navigable mask. The Voronoi map is a
    derived ownership layer for furniture/object centroids that usually fall on
    non-navigable cells outside the manually painted floor.
    """
    known = room_map >= 0
    if domain is None:
        domain = _load_voronoi_domain(scene_dir, room_map.shape, room_map, domain_threshold)
    voronoi = np.full_like(room_map, -1, dtype=np.int32)
    distance_cells = np.full(room_map.shape, np.inf, dtype=np.float32)
    if not np.any(known):
        return voronoi, distance_cells

    # Multi-source breadth-first expansion. Unlike Euclidean EDT, this respects
    # the domain barrier, so labels do not jump across black wall/outside pixels.
    q = deque()
    src_rows, src_cols = np.where(known)
    for row, col in zip(src_rows.tolist(), src_cols.tolist()):
        voronoi[row, col] = int(room_map[row, col])
        distance_cells[row, col] = 0.0
        q.append((int(row), int(col)))

    h, w = room_map.shape[:2]
    max_d = int(max(0, max_distance_cells))
    while q:
        row, col = q.popleft()
        next_d = int(distance_cells[row, col]) + 1
        if next_d > max_d:
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if nr < 0 or nc < 0 or nr >= h or nc >= w:
                continue
            if voronoi[nr, nc] >= 0:
                continue
            if not bool(domain[nr, nc]):
                continue
            voronoi[nr, nc] = voronoi[row, col]
            distance_cells[nr, nc] = float(next_d)
            q.append((nr, nc))
    return voronoi.astype(np.int32), distance_cells.astype(np.float32)


def _load_voronoi_domain(scene_dir, shape, room_map: np.ndarray, threshold: int = 20) -> np.ndarray:
    """Return pixels where Voronoi may propagate.

    We use topdown_labeled/topdown_rgb as a wall/outside barrier: black pixels
    are walls or exterior; textured furniture and labelled floor are allowed.
    This lets ownership reach furniture but prevents crossing walls. Only
    visible connected components that touch a LabelMe room are kept, so isolated
    exterior artefacts cannot receive ownership.
    """
    scene_dir = Path(scene_dir)
    h, w = int(shape[0]), int(shape[1])
    domain = np.zeros((h, w), dtype=bool)
    for name in ("topdown_labeled.png", "topdown_rgb.png", "obstacle_map.png"):
        path = scene_dir / name
        if not path.exists():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3:
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            domain |= gray > int(threshold)
        else:
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
            # obstacle_map is useful as a fallback for navigable floor.
            domain |= img > 0
    domain |= room_map >= 0
    domain = _keep_domain_components_touching_rooms(domain, room_map)
    return domain


def _keep_domain_components_touching_rooms(domain: np.ndarray, room_map: np.ndarray) -> np.ndarray:
    if not np.any(domain):
        return room_map >= 0
    labelled = room_map >= 0
    n_labels, comps, stats, _centroids = cv2.connectedComponentsWithStats(
        domain.astype(np.uint8),
        connectivity=4,
    )
    kept = np.zeros_like(domain, dtype=bool)
    for comp_id in range(1, n_labels):
        comp = comps == comp_id
        if np.any(comp & labelled):
            kept |= comp
    kept |= labelled
    return kept


def _render_domain(domain: np.ndarray, room_map: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*domain.shape[:2], 3), dtype=np.uint8)
    canvas[domain] = (55, 55, 55)
    canvas[room_map >= 0] = (255, 255, 255)
    return canvas


def _render_voronoi_map(voronoi_map: np.ndarray, room_map: np.ndarray, labels, domain=None) -> np.ndarray:
    """Render a compact preview of navigable rooms plus Voronoi ownership."""
    h, w = voronoi_map.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if domain is not None:
        canvas[domain] = (18, 18, 18)
    rng = np.random.default_rng(7)
    colors = rng.integers(70, 230, size=(max(1, len(labels)), 3), dtype=np.uint8)
    for idx, _label in enumerate(labels):
        expanded = voronoi_map == idx
        navigable = room_map == idx
        color = colors[idx].astype(np.float32)
        canvas[expanded] = np.clip(color * 0.45, 0, 255).astype(np.uint8)
        canvas[navigable] = color.astype(np.uint8)

    boundary = ((room_map >= 0).astype(np.uint8) * 255)
    contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (255, 255, 255), 1)
    return canvas


def convert(
    json_path: str,
    scene_dir: str,
    min_region_size: int = 50,
    preview: bool = True,
    voronoi_max_distance_cells: int = 50,
    voronoi_domain_threshold: int = 20,
):
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
    voronoi_domain = _load_voronoi_domain(
        scene_dir,
        room_map.shape,
        room_map,
        threshold=voronoi_domain_threshold,
    )
    voronoi_map, voronoi_dist = _build_voronoi_room_map(
        room_map,
        scene_dir=scene_dir,
        max_distance_cells=voronoi_max_distance_cells,
        domain_threshold=voronoi_domain_threshold,
        domain=voronoi_domain,
    )
    room_map_dir = Path(scene_dir) / "room_map"
    np.save(room_map_dir / "room_voronoi.npy", voronoi_map)
    np.save(room_map_dir / "room_voronoi_distance.npy", voronoi_dist)
    cv2.imwrite(
        str(room_map_dir / "room_voronoi_domain.png"),
        _render_domain(voronoi_domain, room_map),
    )
    cv2.imwrite(
        str(room_map_dir / "room_voronoi_viz.png"),
        _render_voronoi_map(voronoi_map, room_map, labels, domain=voronoi_domain),
    )

    print(f"\nRoom map saved to: {Path(scene_dir) / 'room_map'}/")
    print(f"  Categories ({len(labels)}): {labels}")
    print(
        "  Voronoi    : "
        f"{int((voronoi_map >= 0).sum())} assigned cells "
        f"(max distance {voronoi_max_distance_cells} cells, "
        f"domain {int(voronoi_domain.sum())} cells)"
    )
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
    parser.add_argument("--voronoi-max-distance-cells", type=int, default=50,
                        help="Maximum expansion distance for furniture ownership (default: 50 cells)")
    parser.add_argument("--voronoi-domain-threshold", type=int, default=20,
                        help="Minimum grayscale value considered indoor/visible domain (default: 20)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Do not open the OpenCV room-map preview window")
    args = parser.parse_args()

    if not Path(args.json).exists():
        print(f"ERROR: JSON not found: {args.json}")
        sys.exit(1)
    if not Path(args.scene).is_dir():
        print(f"ERROR: Scene dir not found: {args.scene}")
        sys.exit(1)

    convert(
        args.json,
        args.scene,
        args.min_region_size,
        preview=not args.no_preview,
        voronoi_max_distance_cells=args.voronoi_max_distance_cells,
        voronoi_domain_threshold=args.voronoi_domain_threshold,
    )


if __name__ == "__main__":
    main()
