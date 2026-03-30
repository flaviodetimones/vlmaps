"""
room_provider.py
================
Abstract interface for room/region information, with two concrete implementations:

- LabelMeRoomProvider  : loads room_map/ saved by labelme_to_room_map.py.
                          Works for both MP3D and HSSD (manual annotation).
- SemanticSceneRoomProvider : reads region annotations directly from
                          Habitat-Sim's semantic_scene API (HSSD only — free,
                          no manual labeling needed).

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

class SemanticSceneRoomProvider(RoomProvider):
    """
    Reads region annotations directly from Habitat-Sim's semantic_scene API.
    Available on HSSD scenes without any manual labeling.

    Call build(sim, rmin, cmin, cs, gs) after the simulator is initialized.
    """

    def __init__(self):
        self._available = False
        self._regions: List[Dict] = []       # [{name, centroid_rc, aabb, ...}, ...]
        self._region_grid: Optional[np.ndarray] = None   # (H, W) int

    def build(self, sim, rmin: float, cmin: float, cs: float, gs: int) -> None:
        """
        Extract region information from sim.semantic_scene and rasterise
        them onto the VLMap grid.

        Parameters
        ----------
        sim   : habitat_sim.Simulator
        rmin  : top-left row offset of the cropped VLMap grid (world units)
        cmin  : top-left col offset of the cropped VLMap grid (world units)
        cs    : cell size in metres (typically 0.05)
        gs    : grid size in cells (typically 1000)
        """
        try:
            scene = sim.semantic_scene
        except Exception:
            return

        self._regions = []
        H = W = gs
        self._region_grid = np.zeros((H, W), dtype=np.int32)

        region_id = 1
        for level in scene.levels:
            for region in level.regions:
                try:
                    cat_name = region.category.name() if region.category else "unknown"
                except Exception:
                    cat_name = "unknown"

                center = region.aabb.center   # magnum Vector3: (x, y, z), y-up
                half = region.aabb.half_extents

                # World X/Z → row/col (habitat uses Y-up, navigation is XZ plane)
                cx_world = float(center[0])
                cz_world = float(center[2])
                hx = float(half[0])
                hz = float(half[2])

                # Rasterise AABB onto grid
                r_lo = int((cz_world - hz - rmin) / cs)
                r_hi = int((cz_world + hz - rmin) / cs)
                c_lo = int((cx_world - hx - cmin) / cs)
                c_hi = int((cx_world + hx - cmin) / cs)

                r_lo = max(0, r_lo); r_hi = min(H - 1, r_hi)
                c_lo = max(0, c_lo); c_hi = min(W - 1, c_hi)

                centroid_r = int((cz_world - rmin) / cs)
                centroid_c = int((cx_world - cmin) / cs)

                self._regions.append({
                    "id": region_id,
                    "category": cat_name,
                    "label": f"{cat_name}_{region_id}",
                    "centroid": [centroid_r, centroid_c],
                    "aabb_r": (r_lo, r_hi),
                    "aabb_c": (c_lo, c_hi),
                })

                if r_lo <= r_hi and c_lo <= c_hi:
                    self._region_grid[r_lo:r_hi+1, c_lo:c_hi+1] = region_id

                region_id += 1

        self._available = bool(self._regions)

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
        room_name_lower = room_name.lower().replace(" ", "_")
        for reg in self._regions:
            if room_name_lower in reg["label"].lower() or room_name_lower in reg["category"].lower():
                r, c = reg["centroid"]
                return (float(r), float(c))
        return None

    def list_rooms(self) -> List[str]:
        return [r["label"] for r in self._regions]
