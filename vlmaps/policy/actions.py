"""
actions.py
==========
Phase F — typed high-level action vocabulary.

The room-aware policy expresses every decision as one of five actions:

    go_to_room(room)             -- navigate to a safe interior goal of *room*
    explore_room(room)           -- visit an unexplored frontier inside *room*
    inspect_candidate(room,      -- approach a heatmap component and verify
                      cand)         it with YOLOE
    verify_target()              -- 360 scan + fine visual centering at the
                                    current pose
    done(reason)                 -- end the episode (target found / give up)

Each action is a small immutable record. Producers (heuristic policy, future
LLM strategist) emit Actions; an executor maps them onto existing pipeline
calls. JSON helpers are provided so an LLM can emit/consume actions through
the OpenAI tool-call interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional, Tuple


# ── Action enum ───────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    GO_TO_ROOM = "go_to_room"
    EXPLORE_ROOM = "explore_room"
    INSPECT_CANDIDATE = "inspect_candidate"
    VERIFY_TARGET = "verify_target"
    DONE = "done"


# ── Action record ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Action:
    """A single high-level decision.

    Fields are union-typed by ActionType — only the fields relevant to the
    chosen type need to be set; the rest stay at default. Validation in
    __post_init__ rejects malformed combinations early so the executor can
    trust them.
    """

    type: ActionType
    room: Optional[str] = None
    candidate_centroid: Optional[Tuple[int, int]] = None
    reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = self.type
        if t in (ActionType.GO_TO_ROOM, ActionType.EXPLORE_ROOM):
            if not self.room:
                raise ValueError(f"{t.value} requires a non-empty 'room'")
        elif t is ActionType.INSPECT_CANDIDATE:
            if not self.room:
                raise ValueError("inspect_candidate requires 'room'")
            if self.candidate_centroid is None:
                raise ValueError("inspect_candidate requires 'candidate_centroid'")
            r, c = self.candidate_centroid
            if not (isinstance(r, int) and isinstance(c, int)):
                raise ValueError("candidate_centroid must be (int row, int col)")
        elif t is ActionType.VERIFY_TARGET:
            pass
        elif t is ActionType.DONE:
            pass
        else:  # pragma: no cover — Enum prevents this
            raise ValueError(f"Unknown ActionType {t!r}")

    # ── Pretty + log forms ────────────────────────────────────────────────
    def short(self) -> str:
        if self.type is ActionType.GO_TO_ROOM:
            return f"go_to_room({self.room})"
        if self.type is ActionType.EXPLORE_ROOM:
            return f"explore_room({self.room})"
        if self.type is ActionType.INSPECT_CANDIDATE:
            cr, cc = self.candidate_centroid
            return f"inspect_candidate({self.room}, ({cr},{cc}))"
        if self.type is ActionType.VERIFY_TARGET:
            return "verify_target()"
        return f"done({self.reason or ''})"


# ── JSON exchange (for the Phase G LLM tool-call) ─────────────────────────────

def action_to_json(a: Action) -> str:
    """Serialise an Action to a single-line JSON string."""
    d: Dict[str, Any] = {"type": a.type.value}
    if a.room is not None:
        d["room"] = a.room
    if a.candidate_centroid is not None:
        d["candidate_centroid"] = [int(a.candidate_centroid[0]),
                                   int(a.candidate_centroid[1])]
    if a.reason is not None:
        d["reason"] = a.reason
    if a.extra:
        d["extra"] = a.extra
    return json.dumps(d, ensure_ascii=False)


def parse_action_json(payload: str) -> Action:
    """Parse the JSON form back into an Action.

    Raises ValueError on missing/invalid fields so the caller can fall back
    to the heuristic policy when the LLM produces garbage.
    """
    try:
        d = json.loads(payload) if isinstance(payload, str) else dict(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"action payload is not valid JSON: {exc}") from exc

    if not isinstance(d, dict) or "type" not in d:
        raise ValueError("action payload must be an object with a 'type' key")

    try:
        t = ActionType(d["type"])
    except ValueError as exc:
        raise ValueError(f"unknown action type: {d['type']!r}") from exc

    cc = d.get("candidate_centroid")
    if cc is not None:
        if not (isinstance(cc, (list, tuple)) and len(cc) == 2):
            raise ValueError("candidate_centroid must be a 2-element list")
        cc = (int(cc[0]), int(cc[1]))

    extra = d.get("extra") or {}
    if not isinstance(extra, dict):
        raise ValueError("extra must be an object if present")

    return Action(
        type=t,
        room=d.get("room"),
        candidate_centroid=cc,
        reason=d.get("reason"),
        extra=extra,
    )


# Compact prompt fragment describing the schema. Phase G can paste this into
# the system prompt so the LLM knows exactly what JSON shape to return.
ACTION_SCHEMA_PROMPT = """\
Reply with a single JSON object describing the next action. Allowed shapes:

  {"type": "go_to_room",        "room": "<room name>"}
  {"type": "explore_room",      "room": "<room name>"}
  {"type": "inspect_candidate", "room": "<room name>",
                                "candidate_centroid": [<row>, <col>]}
  {"type": "verify_target"}
  {"type": "done",              "reason": "<short string>"}

Pick exactly one action. Do not wrap the JSON in markdown fences.
"""


# Backwards-friendly conversion from a (string, dict) heuristic emit pair.
def action_from_dict(payload: Dict[str, Any]) -> Action:
    """Build an Action from an in-memory dict — convenience for unit tests."""
    return parse_action_json(json.dumps(payload))
