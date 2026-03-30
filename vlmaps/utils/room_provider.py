"""
room_provider.py
================
Abstract interface for room/region information, with two concrete implementations:

- LabelMeRoomProvider        : loads room_map/ saved by labelme_to_room_map.py.
                                Works for both MP3D and HSSD (manual annotation).
- SemanticSceneRoomProvider  : reads region annotations directly from the
                                per-scene semantic_config.json files in the
                                HSSD dataset (semantics/scenes/<id>.semantic_config.json).
                                Free annotations — no manual labeling needed.

Both expose the same three methods so the rest of the pipeline is agnostic.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Abstract base ────────────────────────────────────────────────────────────

class RoomProvider(ABC):
    @abstractmethod
    def get_room_at_cell(self, row: int, col: int) -> Optional[str]:
        """Return the room name for a map grid cell, or None if unknown."""

    @abstractmethod
    def get_room_centroid(self, room_name: str) -> Optional[Tuple[float, float]]:
        """Return (row, col) centroid of the best matching room, or None."""

    @abstractmethod
    def list_rooms(self) -> List[str]:
        """Return a list of all labeled room names."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if room information could be loaded."""


# ── LabelMe provider ─────────────────────────────────────────────────────────

class LabelMeRoomProvider(RoomProvider):
    """
    Loads the room_map/ directory produced by labelme_to_room_map.py.
    Works for any scene where manual LabelMe annotation has been done.
    """

    def __init__(self, scene_dir: str):
        from vlmaps.utils.room_map_utils import load_room_map
        result = load_room_map(scene_dir)
        if result is None:
            self._available = False
            self._room_map = None
            self._categories = []
            self._regions = {}
        else:
            self._available = True
            self._room_map, self._categories, self._regions = result

    def is_available(self) -> bool:
        return self._available

    def get_room_at_cell(self, row: int, col: int) -> Optional[str]:
        if not self._available or self._room_map is None:
            return None
        if row < 0 or col < 0 or row >= self._room_map.shape[0] or col >= self._room_map.shape[1]:
            return None
        label_id = int(self._room_map[row, col])
        if label_id <= 0:
            return None
        # regions keys are str or int depending on save format
        region = self._regions.get(label_id) or self._regions.get(str(label_id))
        if region is None:
            return None
        return region.get("label") or region.get("category")

    def get_room_centroid(self, room_name: str) -> Optional[Tuple[float, float]]:
        if not self._available:
            return None
        from vlmaps.utils.room_map_utils import find_room_goal
        curr_pos = (0, 0)  # dummy — find_room_goal picks closest
        result = find_room_goal(room_name, self._regions, curr_pos)
        return result  # (row, col) or None

    def list_rooms(self) -> List[str]:
        if not self._available:
            return []
        labels = []
        for region in self._regions.values():
            lbl = region.get("label") or region.get("category")
            if lbl and lbl not in labels:
                labels.append(lbl)
        return labels


# ── Semantic scene provider (HSSD) ───────────────────────────────────────────

def _point_in_polygon(px: float, pz: float, poly_xz: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test. poly_xz: (N, 2) array of (x, z) vertices."""
    n = len(poly_xz)
    inside = False
    j = n - 1
    for i in range(n):
        xi, zi = poly_xz[i]
        xj, zj = poly_xz[j]
        if ((zi > pz) != (zj > pz)) and (px < (xj - xi) * (pz - zi) / (zj - zi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


class SemanticSceneRoomProvider(RoomProvider):
    """
    Reads HSSD region annotations from the per-scene semantic_config.json.

    HSSD stores room/region info as 2D polygon floor plans (not via
    sim.semantic_scene.levels which is empty for HSSD). Each region has:
      - name   : human-readable room name (e.g. "kitchen")
      - label  : category string (e.g. "kitchen/cooking area")
      - poly_loop : list of [x, 0, z] world-coord vertices
      - floor_height, extrusion_height

    Call build(semantic_config_path, rmin, cmin, cs, gs) after loading the scene.
    The semantic_config_path is typically:
      <hssd_root>/semantics/scenes/<scene_id>.semantic_config.json
    """

    def __init__(self):
        self._available = False
        self._regions: List[Dict] = []
        self._region_grid: Optional[np.ndarray] = None   # (H, W) int, 0 = unlabeled

    # ------------------------------------------------------------------
    @staticmethod
    def find_config_path(scene_dataset_config: str, scene_id: str) -> Optional[str]:
        """
        Derive the semantic_config.json path from the dataset config path and scene_id.
        Looks in <dataset_root>/semantics/scenes/<scene_id>.semantic_config.json
        """
        root = Path(scene_dataset_config).parent
        p = root / "semantics" / "scenes" / f"{scene_id}.semantic_config.json"
        return str(p) if p.exists() else None

    # ------------------------------------------------------------------
    def build(self, semantic_config_path: str,
              rmin: float, cmin: float, cs: float, gs: int) -> None:
        """
        Load region polygons from the HSSD semantic_config.json and rasterise
        them onto the VLMap grid using point-in-polygon.

        Parameters
        ----------
        semantic_config_path : path to <scene_id>.semantic_config.json
        rmin  : top-left Z offset of the VLMap grid (world metres)
        cmin  : top-left X offset of the VLMap grid (world metres)
        cs    : cell size in metres (typically 0.05)
        gs    : grid size in cells (typically 1000)
        """
        import json

        p = Path(semantic_config_path)
        if not p.exists():
            print(f"[SemanticSceneRoomProvider] config not found: {p}")
            return

        with open(p) as f:
            data = json.load(f)

        annotations = data.get("region_annotations", [])
        if not annotations:
            print("[SemanticSceneRoomProvider] no region_annotations in config")
            return

        H = W = gs
        self._region_grid = np.zeros((H, W), dtype=np.int32)
        self._regions = []

        for region_id, ann in enumerate(annotations, start=1):
            name = ann.get("name", f"region_{region_id}")
            label = ann.get("label", name)
            poly_raw = ann.get("poly_loop", [])  # [[x, 0, z], ...]

            if len(poly_raw) < 3:
                continue

            # Extract (x, z) pairs — y is always 0 in HSSD poly_loops
            poly_xz = np.array([[v[0], v[2]] for v in poly_raw], dtype=np.float32)

            # Bounding box in world coords → grid coords for fast pre-filter
            x_min, z_min = poly_xz.min(axis=0)
            x_max, z_max = poly_xz.max(axis=0)

            c_lo = max(0, int((x_min - cmin) / cs))
            c_hi = min(W - 1, int((x_max - cmin) / cs) + 1)
            r_lo = max(0, int((z_min - rmin) / cs))
            r_hi = min(H - 1, int((z_max - rmin) / cs) + 1)

            # Rasterise: test each grid cell centre against the polygon
            for r in range(r_lo, r_hi + 1):
                for c in range(c_lo, c_hi + 1):
                    wx = cmin + c * cs + cs / 2.0
                    wz = rmin + r * cs + cs / 2.0
                    if _point_in_polygon(wx, wz, poly_xz):
                        # Later regions overwrite earlier ones (fine for non-overlapping rooms)
                        self._region_grid[r, c] = region_id

            # Centroid = average of polygon vertices
            cx_world = float(poly_xz[:, 0].mean())
            cz_world = float(poly_xz[:, 1].mean())
            centroid_r = int((cz_world - rmin) / cs)
            centroid_c = int((cx_world - cmin) / cs)

            self._regions.append({
                "id": region_id,
                "name": name,
                "category": label,
                "label": name,          # use short name as label for matching
                "centroid": [centroid_r, centroid_c],
                "poly_xz": poly_xz,
                "floor_height": ann.get("floor_height", 0.0),
            })

        self._available = bool(self._regions)
        n_labeled = int((self._region_grid > 0).sum())
        print(f"[SemanticSceneRoomProvider] loaded {len(self._regions)} regions, "
              f"{n_labeled} grid cells labeled")

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        return self._available

    def get_room_at_cell(self, row: int, col: int) -> Optional[str]:
        if not self._available or self._region_grid is None:
            return None
        H, W = self._region_grid.shape
        if row < 0 or col < 0 or row >= H or col >= W:
            return None
        rid = int(self._region_grid[row, col])
        if rid <= 0:
            return None
        for reg in self._regions:
            if reg["id"] == rid:
                return reg["label"]
        return None

    def get_room_centroid(self, room_name: str) -> Optional[Tuple[float, float]]:
        if not self._available:
            return None
        query = room_name.lower().strip()
        best = None
        for reg in self._regions:
            if query in reg["label"].lower() or query in reg["category"].lower():
                best = reg
                break
        if best is None:
            return None
        r, c = best["centroid"]
        return (float(r), float(c))

    def list_rooms(self) -> List[str]:
        return [r["label"] for r in self._regions]
