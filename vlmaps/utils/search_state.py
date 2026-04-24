"""
search_state.py
===============
Per-room and per-search state tracking for room-aware object navigation.

RoomState  — one per room, tracks exploration progress and object evidence.
SearchState — one per search instruction, aggregates all RoomStates and
             records the global search history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RoomState:
    """Mutable state for a single room during a search episode."""

    name: str
    centroid: Tuple[float, float]          # (row, col) in VLMap grid

    # Geometry (computed once at init from region_grid + obstacles_map)
    total_cells: int = 0                   # room polygon cells in region_grid
    free_cells: int = 0                    # room ∩ free-space (navigable area)

    # Exploration
    times_visited: int = 0                 # how many times the robot entered
    objects_seen: List[str] = field(default_factory=list)  # objects found here
    candidates_tried: int = 0             # heatmap components inspected here
    candidates_confirmed: int = 0         # YOLOE-confirmed detections here

    # Phase C: prior plausibility that the target is in this room (0-1, normalised)
    target_relevance: float = 0.0

    # Search result
    target_found_here: bool = False

    # ── Derived properties ──────────────────────────────────────────────
    @property
    def explored_ratio(self) -> float:
        """Fraction of room area that is navigable (static, from map build)."""
        if self.total_cells == 0:
            return 0.0
        return self.free_cells / self.total_cells

    def summary(self) -> str:
        """One-line human-readable summary for LLM context or logging."""
        return (
            f"{self.name}: prior={self.target_relevance:.2f}, "
            f"visited={self.times_visited}, "
            f"candidates={self.candidates_tried}/{self.candidates_confirmed}, "
            f"objects={self.objects_seen}, "
            f"navigable={self.explored_ratio:.0%}, "
            f"found={self.target_found_here}"
        )


class SearchState:
    """Tracks the full state of a single object-search episode.

    Constructed once per navigation instruction.  Updated as the robot
    navigates, inspects candidates, and confirms/rejects with YOLOE.
    """

    def __init__(
        self,
        target: str,
        room_provider,
        obstacles_map: np.ndarray,
    ):
        self.target: str = target
        self.rooms: Dict[str, RoomState] = {}
        self.visit_history: List[str] = []   # ordered list of room visits
        self.current_room: Optional[str] = None
        self.found: bool = False
        self._tried_centroids: set = set()   # Phase D gate: (int_r, int_c) tried
        # Phase F: ordered log of high-level actions emitted this episode.
        # Each entry is a dict {"action": <str>, "outcome": <str|None>}; the
        # heuristic and (later) LLM strategist append here so that the next
        # decision and the post-hoc metrics can read recent activity.
        self.action_log: List[Dict[str, Optional[str]]] = []
        self.visited_cells: List[Tuple[int, int]] = []  # robot cells stepped on

        # Verification-stage telemetry (per-episode). Records at which stage
        # YOLOE confirmed the target, if any. Five boolean flags + one string
        # source. Additive — legacy consumers ignore unseen keys.
        self.found_on_arrival: bool = False
        self.found_after_turn_to_face: bool = False
        self.found_after_centering: bool = False
        self.found_after_local_scan: bool = False
        self.found_after_alternative_route: bool = False
        self.final_confirmation_source: Optional[str] = None

        self._build_rooms(room_provider, obstacles_map)

    # ------------------------------------------------------------------
    def mark_confirmation(self, source: str) -> None:
        """Mark that YOLOE confirmed the target at stage *source*.

        Valid values: arrival, turn_to_face, centering, local_scan,
        alternative_route. Only the first confirmation sets the final source;
        further calls are ignored so the *final_confirmation_source* is the
        stage where the episode actually succeeded.
        """
        if source == "arrival":
            self.found_on_arrival = True
        elif source == "turn_to_face":
            self.found_after_turn_to_face = True
        elif source == "centering":
            self.found_after_centering = True
        elif source == "local_scan":
            self.found_after_local_scan = True
        elif source == "alternative_route":
            self.found_after_alternative_route = True
        else:
            return
        if self.final_confirmation_source is None:
            self.final_confirmation_source = source

    # ------------------------------------------------------------------
    def _build_rooms(self, room_provider, obstacles_map: np.ndarray) -> None:
        """Initialize one RoomState per room from the provider + obstacle map."""
        if room_provider is None or not room_provider.is_available():
            return

        region_grid = getattr(room_provider, "_region_grid", None)
        regions = getattr(room_provider, "_regions", [])
        if region_grid is None or not regions:
            return

        free_mask = (obstacles_map > 0) if obstacles_map is not None else None

        for reg in regions:
            rid = reg["id"]
            name = reg["label"]
            cr, cc = reg["centroid"]

            room_mask = (region_grid == rid)
            total = int(room_mask.sum())
            free = int((room_mask & free_mask).sum()) if free_mask is not None else total

            self.rooms[name] = RoomState(
                name=name,
                centroid=(float(cr), float(cc)),
                total_cells=total,
                free_cells=free,
            )

    # ------------------------------------------------------------------
    def update_current_room(self, room_name: Optional[str]) -> None:
        """Record that the robot is now in *room_name*."""
        if room_name is None:
            return
        prev = self.current_room
        self.current_room = room_name
        if room_name != prev and room_name in self.rooms:
            self.rooms[room_name].times_visited += 1
            self.visit_history.append(room_name)

    def record_candidate(self, room_name: Optional[str], confirmed: bool,
                         centroid: tuple = None) -> None:
        """Record that a heatmap candidate in *room_name* was inspected."""
        if room_name and room_name in self.rooms:
            rs = self.rooms[room_name]
            rs.candidates_tried += 1
            if confirmed:
                rs.candidates_confirmed += 1
        if centroid is not None:
            self._tried_centroids.add((int(centroid[0]), int(centroid[1])))

    def record_object_seen(self, room_name: Optional[str], obj: str) -> None:
        """Record that *obj* was visually confirmed in *room_name*."""
        if room_name and room_name in self.rooms:
            rs = self.rooms[room_name]
            if obj not in rs.objects_seen:
                rs.objects_seen.append(obj)

    # ------------------------------------------------------------------
    # Phase F additions — high-level action log + visited cell trail.
    def record_action(self, action_str: str, outcome: Optional[str] = None) -> None:
        """Append a high-level Action (already stringified) to the episode log."""
        self.action_log.append({"action": action_str, "outcome": outcome})

    def record_visited_cell(self, row: int, col: int) -> None:
        """Track that the robot occupied (row, col) — used by explore_room."""
        self.visited_cells.append((int(row), int(col)))

    def recent_actions(self, n: int = 5) -> List[Dict[str, Optional[str]]]:
        """Return the last *n* action records (most recent last)."""
        if n <= 0:
            return []
        return list(self.action_log[-n:])

    # ------------------------------------------------------------------
    def mark_found(self, room_name: Optional[str]) -> None:
        """Mark the target as found in *room_name*."""
        self.found = True
        if room_name and room_name in self.rooms:
            self.rooms[room_name].target_found_here = True

    # ------------------------------------------------------------------
    def compute_priors(self) -> Dict[str, float]:
        """Phase C — compute and store object→room priors for this target.

        Fuses three sources: LLM, manual table, scene evidence.
        Stores the result in each RoomState.target_relevance and returns
        the full prior dict for logging / downstream use.
        """
        from vlmaps.utils.room_priors import compute_room_priors

        known_rooms = list(self.rooms.keys())
        objects_seen_by_room = {
            name: rs.objects_seen for name, rs in self.rooms.items()
        }
        priors = compute_room_priors(
            self.target, known_rooms, objects_seen_by_room
        )
        for room, score in priors.items():
            if room in self.rooms:
                self.rooms[room].target_relevance = score
        return priors

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Multi-line summary of the entire search state, with a room table."""
        lines = [f"Search target: '{self.target}'  found={self.found}"]
        lines.append(f"Visit history: {' → '.join(self.visit_history) or '(none)'}")

        headers = ("room", "prior", "visited", "cand", "conf", "objects",
                   "nav", "found")
        rows = []
        for rs in self.rooms.values():
            objs = ",".join(rs.objects_seen) if rs.objects_seen else "-"
            rows.append((
                rs.name,
                f"{rs.target_relevance:.2f}",
                str(rs.times_visited),
                str(rs.candidates_tried),
                str(rs.candidates_confirmed),
                objs,
                f"{rs.explored_ratio:.0%}",
                "yes" if rs.target_found_here else "no",
            ))

        widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
                  for i, h in enumerate(headers)]
        fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
        lines.append(fmt.format(*headers))
        lines.append("  " + "  ".join("-" * w for w in widths))
        for r in rows:
            lines.append(fmt.format(*r))
        return "\n".join(lines)

    def room_summaries_for_llm(self) -> str:
        """Compact room table suitable for LLM context window."""
        lines = []
        for rs in self.rooms.values():
            lines.append(
                f"- {rs.name} | visited {rs.times_visited}x | "
                f"candidates {rs.candidates_tried} tried, "
                f"{rs.candidates_confirmed} confirmed | "
                f"navigable {rs.explored_ratio:.0%}"
            )
        return "\n".join(lines)
