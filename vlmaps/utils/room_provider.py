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
import re


_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _normalize_room_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s*\.\s*", ".", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_room_instance(text: str) -> Tuple[str, Optional[int]]:
    """Split a room query into canonical base name + optional explicit instance.

    Examples:
        "bathroom"      -> ("bathroom", None)
        "bathroom.001"  -> ("bathroom", 1)
        "bathroom 1"    -> ("bathroom", 1)
        "bedroom one"   -> ("bedroom", 1)
    """
    q = _normalize_room_text(text)
    m = re.fullmatch(r"(.+?)\.(\d+)$", q)
    if m:
        return m.group(1).strip(), int(m.group(2))
    m = re.fullmatch(r"(.+?)\s+(\d+)$", q)
    if m:
        return m.group(1).strip(), int(m.group(2))
    m = re.fullmatch(r"(.+?)\s+([a-z]+)$", q)
    if m and m.group(2) in _NUMBER_WORDS:
        return m.group(1).strip(), _NUMBER_WORDS[m.group(2)]
    return q, None


def _room_aliases(room_name: str) -> set:
    aliases = {_normalize_room_text(room_name)}
    base, idx = _parse_room_instance(room_name)
    if idx is not None:
        aliases.add(f"{base} {idx}")
        aliases.add(f"{base}.{idx}")
        aliases.add(f"{base}.{idx:03d}")
        for word, value in _NUMBER_WORDS.items():
            if value == idx:
                aliases.add(f"{base} {word}")
                break
    return aliases


def room_command_matches(actual_room: Optional[str], target_room: str) -> bool:
    """Return True if actual_room satisfies a room navigation command.

    Base commands like "bathroom" accept any bathroom instance.
    Explicit-instance commands like "bathroom.001" or "bathroom 1" require
    that exact instance.
    """
    if actual_room is None:
        return False
    a_base, a_idx = _parse_room_instance(actual_room)
    t_base, t_idx = _parse_room_instance(target_room)
    if a_base != t_base:
        return False
    if t_idx is not None:
        return a_idx == t_idx
    return True


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

    def resolve_room_name(self, room_query: str) -> Optional[str]:
        """Resolve a user/LLM room query to an exact room label if possible.

        Supports exact instance labels and friendly aliases such as:
        - "bathroom.001" -> "bathroom.001"
        - "bathroom 1"   -> "bathroom.001"
        - "bathroom one" -> "bathroom.001"
        - "bathroom"     -> "bathroom" (preferred if a base room exists)
        """
        if not self.is_available():
            return None
        rooms = self.list_rooms()
        if not rooms:
            return None

        query = _normalize_room_text(room_query)
        q_base, q_idx = _parse_room_instance(query)

        # First, exact alias match against every known room instance.
        for room in rooms:
            if query in _room_aliases(room):
                return room

        family = [room for room in rooms if _parse_room_instance(room)[0] == q_base]
        if not family:
            return None

        # Explicit instance required: look for the corresponding numbered variant.
        if q_idx is not None:
            for room in family:
                _base, _idx = _parse_room_instance(room)
                if _idx == q_idx:
                    return room
            return None

        # Base room preferred if it exists explicitly.
        for room in family:
            if _normalize_room_text(room) == q_base:
                return room

        # Fallback to the first family member if the scene only has numbered
        # instances and the user asked for the base family.
        def _family_sort_key(room: str):
            _, idx = _parse_room_instance(room)
            return (0 if idx is None else 1, idx if idx is not None else -1, _normalize_room_text(room))

        return sorted(family, key=_family_sort_key)[0]

    def find_room_mentions(self, text: str) -> List[str]:
        """Return exact room labels explicitly mentioned in free text, in order."""
        if not self.is_available():
            return []
        haystack = _normalize_room_text(text)
        matches = []
        for room in self.list_rooms():
            for alias in sorted(_room_aliases(room), key=len, reverse=True):
                pattern = re.compile(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])")
                m = pattern.search(haystack)
                if m:
                    matches.append((m.start(), -len(alias), m.end(), room))
                    break
        matches.sort()
        ordered = []
        seen = set()
        occupied = []
        for start, _, end, room in matches:
            if any(not (end <= occ_start or start >= occ_end) for occ_start, occ_end in occupied):
                continue
            if room not in seen:
                ordered.append(room)
                seen.add(room)
                occupied.append((start, end))
        return ordered


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
        resolved = self.resolve_room_name(room_name) or room_name
        curr_pos = (0, 0)  # dummy — find_room_goal picks closest
        result = find_room_goal(resolved, self._regions, curr_pos)
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
        query = _normalize_room_text(self.resolve_room_name(room_name) or room_name)
        # Require query to match a whole word in the label/category to avoid
        # "bed" matching "bedroom", "bath" matching "bathroom", etc.
        pattern = re.compile(r'\b' + re.escape(query) + r'\b')
        best = None
        for reg in self._regions:
            label = _normalize_room_text(reg["label"])
            category = _normalize_room_text(reg["category"])
            # Exact match wins immediately
            if query == label or query == category:
                best = reg
                break
            # Word-boundary match — keep first hit, keep looking for exact
            if best is None and (pattern.search(label) or pattern.search(category)):
                best = reg
        if best is None:
            return None
        r, c = best["centroid"]
        return (float(r), float(c))

    def list_rooms(self) -> List[str]:
        return [r["label"] for r in self._regions]
