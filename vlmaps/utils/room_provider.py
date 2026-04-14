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
    def build(self, semantic_config_path: str, gs: int,
              hab_to_grid=None) -> None:
        """
        Load region polygons from the HSSD semantic_config.json and rasterise
        them onto the VLMap grid using point-in-polygon.

        Parameters
        ----------
        semantic_config_path : path to <scene_id>.semantic_config.json
        gs    : grid size in cells (typically 1000)
        hab_to_grid : callable (x_hab, z_hab) -> (row, col) in grid space.
                      Converts Habitat world coordinates to VLMap grid indices.
        """
        import json

        p = Path(semantic_config_path)
        if not p.exists():
            print(f"[SemanticSceneRoomProvider] config not found: {p}")
            return

        if hab_to_grid is None:
            print("[SemanticSceneRoomProvider] hab_to_grid transform required")
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

            # Convert polygon from Habitat world (x, z) to VLMap grid (row, col)
            poly_grid = np.array(
                [hab_to_grid(v[0], v[2]) for v in poly_raw], dtype=np.float32
            )

            # Bounding box in grid coords for fast pre-filter
            r_lo = max(0, int(poly_grid[:, 0].min()))
            r_hi = min(H - 1, int(poly_grid[:, 0].max()) + 1)
            c_lo = max(0, int(poly_grid[:, 1].min()))
            c_hi = min(W - 1, int(poly_grid[:, 1].max()) + 1)

            # Rasterise: point-in-polygon in grid space
            for r in range(r_lo, r_hi + 1):
                for c in range(c_lo, c_hi + 1):
                    if _point_in_polygon(float(r), float(c), poly_grid):
                        self._region_grid[r, c] = region_id

            # Centroid in grid space
            centroid_r = int(poly_grid[:, 0].mean())
            centroid_c = int(poly_grid[:, 1].mean())

            self._regions.append({
                "id": region_id,
                "name": name,
                "category": label,
                "label": name,
                "centroid": [centroid_r, centroid_c],
                "poly_xz": np.array([[v[0], v[2]] for v in poly_raw], dtype=np.float32),
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
