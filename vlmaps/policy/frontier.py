"""
frontier.py
===========
Phase F — frontier picking *within* a room.

`explore_room(room)` needs a target cell that is (a) navigable, (b) inside
the requested room, and (c) far from any cell the robot has already
visited. This module supplies that pick without touching the existing
inline navigation in interactive_object_nav.py.

The visited set is supplied by the caller (typically the executor records
the robot pose into a small deque). When no visited cells are known, the
function falls back to the cell of the room with maximum clearance — i.e.
the most useful "interior" point to head toward.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


def _room_mask(room_name: str, room_provider) -> Optional[np.ndarray]:
    """Return a boolean mask of cells belonging to *room_name* on the VLMap grid.

    Works with both LabelMeRoomProvider and SemanticSceneRoomProvider by
    reading the private `_region_grid` and matching the region label.
    Returns None if the provider does not expose a grid (older providers).
    """
    if room_provider is None or not room_provider.is_available():
        return None
    grid = getattr(room_provider, "_region_grid", None)
    if grid is None:
        return None
    regions = getattr(room_provider, "_regions", None) or []
    target_ids = []
    for reg in regions:
        label = reg.get("label") or reg.get("category") or reg.get("name")
        if label and label.lower() == room_name.lower():
            target_ids.append(int(reg["id"]))
    if not target_ids:
        return None
    mask = np.zeros_like(grid, dtype=bool)
    for rid in target_ids:
        mask |= (grid == rid)
    return mask


def find_frontier_in_room(
    room_name: str,
    room_provider,
    obstacles_map: np.ndarray,
    visited_cells: Optional[Iterable[Tuple[int, int]]] = None,
    *,
    visited_radius_cells: int = 6,
    min_clearance_cells: float = 2.0,
) -> Optional[Tuple[int, int]]:
    """Pick a navigable cell inside *room_name* that has not been visited yet.

    Parameters
    ----------
    room_name           : room label (matched case-insensitively)
    room_provider       : RoomProvider instance with a `_region_grid`
    obstacles_map       : VLMap free-space map (>0 = navigable)
    visited_cells       : iterable of (row, col) cells the robot has already
                          stood on this episode
    visited_radius_cells: cells within this radius of any visited cell are
                          considered already covered
    min_clearance_cells : reject cells whose distance to the nearest obstacle
                          is below this threshold (avoid wall hugging)

    Returns
    -------
    (row, col) of the frontier cell, or None if the room is fully covered /
    unavailable.
    """
    if obstacles_map is None:
        return None

    room_mask = _room_mask(room_name, room_provider)
    if room_mask is None or not room_mask.any():
        return None

    free_mask = (obstacles_map > 0)
    candidates = room_mask & free_mask
    if not candidates.any():
        return None

    clearance = distance_transform_edt(free_mask)
    candidates &= (clearance >= min_clearance_cells)
    if not candidates.any():
        # Fall back without the clearance filter — wall-adjacent is better
        # than no-go in tight rooms.
        candidates = room_mask & free_mask

    # Build a "distance from visited" field so we can pick the most uncovered
    # cell. If no visited cells, pick the cell with maximum clearance.
    if visited_cells:
        visited_mask = np.zeros_like(free_mask, dtype=bool)
        H, W = visited_mask.shape
        for r, c in visited_cells:
            ri, ci = int(r), int(c)
            if 0 <= ri < H and 0 <= ci < W:
                visited_mask[ri, ci] = True
        if visited_mask.any():
            # distance_transform_edt computes distance to nearest *zero* cell,
            # so invert: distance from any visited cell.
            dist_from_visited = distance_transform_edt(~visited_mask)
            score = dist_from_visited.copy()
            score[~candidates] = -1.0

            # Reject anything within visited_radius of a visited cell.
            score[dist_from_visited < visited_radius_cells] = -1.0
            if (score > 0).any():
                idx = int(np.argmax(score))
                r, c = divmod(idx, score.shape[1])
                return (int(r), int(c))
        # No visited yet, or every candidate already covered — fall through.

    score = clearance.copy()
    score[~candidates] = -1.0
    if not (score > 0).any():
        return None
    idx = int(np.argmax(score))
    r, c = divmod(idx, score.shape[1])
    return (int(r), int(c))
