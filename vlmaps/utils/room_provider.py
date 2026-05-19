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

import cv2
import numpy as np
import re
import unicodedata


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


_ROOM_BASE_ALIASES = {
    "cocina": "kitchen",
    "salon": "living room",
    "sala": "living room",
    "sala de estar": "living room",
    "comedor": "dining room",
    "dormitorio": "bedroom",
    "habitacion": "bedroom",
    "cuarto": "bedroom",
    "bano": "bathroom",
    "aseo": "bathroom",
    "lavanderia": "laundry room",
    "oficina": "office",
    "entrada": "entryway",
    "recibidor": "entryway",
    "trastero": "storage room",
    "almacen": "storage room",
    "despensa": "storage room",
    "pasillo": "hallway",
    "armario": "closet",
}


def _normalize_room_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s*\.\s*", ".", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _canonical_room_base_alias(base: str) -> str:
    return _ROOM_BASE_ALIASES.get(_normalize_room_text(base), _normalize_room_text(base))


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
        return _canonical_room_base_alias(m.group(1).strip()), int(m.group(2))
    m = re.fullmatch(r"(.+?)\s+(\d+)$", q)
    if m:
        return _canonical_room_base_alias(m.group(1).strip()), int(m.group(2))
    m = re.fullmatch(r"(.+?)\s+([a-z]+)$", q)
    if m and m.group(2) in _NUMBER_WORDS:
        return _canonical_room_base_alias(m.group(1).strip()), _NUMBER_WORDS[m.group(2)]
    return _canonical_room_base_alias(q), None


def _room_aliases(room_name: str) -> set:
    aliases = {_normalize_room_text(room_name)}
    base, idx = _parse_room_instance(room_name)
    aliases.add(base)
    for alias, canonical in _ROOM_BASE_ALIASES.items():
        if canonical == base:
            aliases.add(alias)
            if idx is not None:
                aliases.add(f"{alias} {idx}")
                aliases.add(f"{alias}.{idx}")
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

    def get_nearest_room_at_cell(self, row: int, col: int) -> Optional[str]:
        """Return expanded room ownership for occupied/unlabelled cells.

        Implementations that do not have a separate ownership layer fall back
        to the navigable/manual room query.
        """
        return self.get_room_at_cell(row, col)

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

    def __init__(self, scene_dir: str, full_shape=None, offset=(0, 0)):
        from vlmaps.utils.room_map_utils import load_room_map
        result = load_room_map(scene_dir)
        if result is None:
            self._available = False
            self._room_map = None
            self._voronoi_map = None
            self._room_map_dir = Path(scene_dir) / "room_map"
            self._categories = []
            self._regions = []
            self._regions_by_label = {}
            self._region_grid = None
        else:
            self._available = True
            self._room_map, self._categories, loaded_regions = result
            _room_dir = Path(scene_dir) / "room_map"
            self._room_map_dir = _room_dir
            _voronoi_file = _room_dir / "room_voronoi.npy"
            self._voronoi_map = np.load(_voronoi_file) if _voronoi_file.exists() else None
            self._regions_by_label = loaded_regions or {}
            self._region_grid = np.zeros_like(self._room_map, dtype=np.int32)
            self._regions = []

            next_region_id = 1
            for cat_idx, label in enumerate(self._categories):
                binary = (self._room_map == cat_idx).astype(np.uint8)
                n_labels, comps, stats, centroids = cv2.connectedComponentsWithStats(
                    binary, connectivity=8
                )
                declared = list(self._regions_by_label.get(label, []))
                for comp_id in range(1, n_labels):
                    area = int(stats[comp_id, cv2.CC_STAT_AREA])
                    cy = int(round(centroids[comp_id][1]))
                    cx = int(round(centroids[comp_id][0]))
                    meta = declared.pop(0) if declared else {}
                    self._region_grid[comps == comp_id] = next_region_id
                    self._regions.append(
                        {
                            "id": next_region_id,
                            "label": label,
                            "category": label,
                            "centroid": meta.get("centroid", [cy, cx]),
                            "area": int(meta.get("area", area)),
                            "quality": float(meta.get("quality", area)),
                        }
                    )
                    next_region_id += 1

            if full_shape is not None and tuple(self._region_grid.shape) != tuple(full_shape):
                full_h, full_w = int(full_shape[0]), int(full_shape[1])
                off_r, off_c = int(offset[0]), int(offset[1])
                h, w = self._region_grid.shape
                expanded_grid = np.zeros((full_h, full_w), dtype=self._region_grid.dtype)
                expanded_map = np.full((full_h, full_w), -1, dtype=self._room_map.dtype)
                expanded_voronoi = None
                if self._voronoi_map is not None:
                    expanded_voronoi = np.full((full_h, full_w), -1, dtype=self._voronoi_map.dtype)

                r0 = max(0, off_r)
                c0 = max(0, off_c)
                r1 = min(full_h, off_r + h)
                c1 = min(full_w, off_c + w)
                src_r0 = max(0, -off_r)
                src_c0 = max(0, -off_c)
                src_r1 = src_r0 + max(0, r1 - r0)
                src_c1 = src_c0 + max(0, c1 - c0)

                if r1 > r0 and c1 > c0:
                    expanded_grid[r0:r1, c0:c1] = self._region_grid[src_r0:src_r1, src_c0:src_c1]
                    expanded_map[r0:r1, c0:c1] = self._room_map[src_r0:src_r1, src_c0:src_c1]
                    if expanded_voronoi is not None:
                        expanded_voronoi[r0:r1, c0:c1] = self._voronoi_map[src_r0:src_r1, src_c0:src_c1]
                    for region in self._regions:
                        cr, cc = region["centroid"]
                        region["centroid"] = [float(cr) + off_r, float(cc) + off_c]

                self._region_grid = expanded_grid
                self._room_map = expanded_map
                self._voronoi_map = expanded_voronoi

    def is_available(self) -> bool:
        return self._available

    def get_room_at_cell(self, row: int, col: int) -> Optional[str]:
        if not self._available or self._region_grid is None:
            return None
        if row < 0 or col < 0 or row >= self._region_grid.shape[0] or col >= self._region_grid.shape[1]:
            return None
        region_id = int(self._region_grid[row, col])
        if region_id <= 0:
            return None
        for region in self._regions:
            if int(region["id"]) == region_id:
                return region.get("label") or region.get("category")
        return None

    def get_nearest_room_at_cell(self, row: int, col: int) -> Optional[str]:
        if not self._available:
            return None
        if self._voronoi_map is None:
            return self.get_room_at_cell(row, col)
        if row < 0 or col < 0 or row >= self._voronoi_map.shape[0] or col >= self._voronoi_map.shape[1]:
            return None
        cat_idx = int(self._voronoi_map[row, col])
        if cat_idx < 0 or cat_idx >= len(self._categories):
            return None
        return self._categories[cat_idx]

    def get_room_centroid(self, room_name: str) -> Optional[Tuple[float, float]]:
        if not self._available:
            return None
        resolved = self.resolve_room_name(room_name) or room_name
        query = _normalize_room_text(resolved)
        for region in self._regions:
            label = _normalize_room_text(region.get("label") or region.get("category") or "")
            if query == label:
                r, c = region["centroid"]
                return (float(r), float(c))
        return None

    def list_rooms(self) -> List[str]:
        if not self._available:
            return []
        return list(self._regions_by_label.keys())

    def has_voronoi(self) -> bool:
        return self._available and self._voronoi_map is not None

    def room_map_shape(self) -> Optional[Tuple[int, int]]:
        if self._room_map is None:
            return None
        return tuple(self._room_map.shape[:2])

    def voronoi_shape(self) -> Optional[Tuple[int, int]]:
        if self._voronoi_map is None:
            return None
        return tuple(self._voronoi_map.shape[:2])

    def room_map_dir(self) -> Path:
        return self._room_map_dir


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
