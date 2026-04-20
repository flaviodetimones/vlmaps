"""
Phase F smoke tests — typed action vocabulary, JSON round-trip, frontier
picking, and the additive SearchState action log.

Run inside the docker container:

    cd /workspace/third_party/vlmaps
    python -m pytest tests/test_phase_f_actions.py -q

These tests do not require Habitat or any heavy dependency — they only
exercise the policy package and the SearchState additions.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vlmaps.policy import (
    Action,
    ActionType,
    action_to_json,
    parse_action_json,
    find_frontier_in_room,
)
from vlmaps.utils.search_state import SearchState


# ── Action validation ─────────────────────────────────────────────────────────

def test_action_requires_room_for_go_to_room():
    with pytest.raises(ValueError):
        Action(type=ActionType.GO_TO_ROOM)


def test_inspect_candidate_requires_centroid():
    with pytest.raises(ValueError):
        Action(type=ActionType.INSPECT_CANDIDATE, room="kitchen")


def test_done_action_is_minimal():
    a = Action(type=ActionType.DONE, reason="target found")
    assert a.short().startswith("done(")


# ── JSON round-trip ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "action",
    [
        Action(type=ActionType.GO_TO_ROOM, room="kitchen"),
        Action(type=ActionType.EXPLORE_ROOM, room="bathroom"),
        Action(type=ActionType.INSPECT_CANDIDATE,
               room="kitchen", candidate_centroid=(123, 456)),
        Action(type=ActionType.VERIFY_TARGET),
        Action(type=ActionType.DONE, reason="give up"),
    ],
)
def test_json_roundtrip(action):
    payload = action_to_json(action)
    # Must be a single-line JSON string the LLM can emit verbatim.
    assert "\n" not in payload
    parsed = parse_action_json(payload)
    assert parsed == action


def test_parse_rejects_unknown_type():
    with pytest.raises(ValueError):
        parse_action_json(json.dumps({"type": "teleport", "room": "kitchen"}))


def test_parse_rejects_malformed_centroid():
    bad = json.dumps({"type": "inspect_candidate", "room": "kitchen",
                      "candidate_centroid": [42]})
    with pytest.raises(ValueError):
        parse_action_json(bad)


# ── SearchState action log ────────────────────────────────────────────────────

class _FakeProvider:
    """Minimal stand-in — search_state only needs is_available()."""
    def is_available(self):
        return False


def test_search_state_action_log_round_trip():
    ss = SearchState("sink", _FakeProvider(), obstacles_map=None)
    ss.record_action("go_to_room(kitchen)", outcome="arrived")
    ss.record_action("inspect_candidate(kitchen, (10, 12))", outcome="rejected")
    ss.record_action("verify_target()", outcome="confirmed")

    assert len(ss.action_log) == 3
    last2 = ss.recent_actions(2)
    assert last2[0]["action"].startswith("inspect_candidate")
    assert last2[-1]["outcome"] == "confirmed"


def test_visited_cell_trail():
    ss = SearchState("sink", _FakeProvider(), obstacles_map=None)
    ss.record_visited_cell(10, 20)
    ss.record_visited_cell(11, 22)
    assert ss.visited_cells == [(10, 20), (11, 22)]


# ── Frontier picker ───────────────────────────────────────────────────────────

class _GridProvider:
    """Synthetic provider that mimics SemanticSceneRoomProvider's interface."""

    def __init__(self, region_grid, regions):
        self._region_grid = region_grid
        self._regions = regions

    def is_available(self):
        return True


def _make_two_room_map(size=40):
    """Left half = kitchen, right half = bathroom, all navigable."""
    grid = np.zeros((size, size), dtype=np.int32)
    grid[:, : size // 2] = 1
    grid[:, size // 2:] = 2
    obs = np.ones((size, size), dtype=np.uint8)  # all free
    # Carve a single obstacle column down the middle to give clearance gradient.
    obs[:, size // 2] = 0
    regions = [
        {"id": 1, "label": "kitchen"},
        {"id": 2, "label": "bathroom"},
    ]
    return _GridProvider(grid, regions), obs


def test_frontier_returns_cell_inside_room():
    provider, obs = _make_two_room_map()
    cell = find_frontier_in_room("kitchen", provider, obs)
    assert cell is not None
    r, c = cell
    assert provider._region_grid[r, c] == 1


def test_frontier_avoids_visited_neighborhood():
    provider, obs = _make_two_room_map(size=60)
    visited = [(30, 5), (30, 6), (30, 7)]
    cell = find_frontier_in_room(
        "kitchen", provider, obs, visited_cells=visited,
        visited_radius_cells=4,
    )
    assert cell is not None
    r, c = cell
    # Must be at least visited_radius cells away from every visited cell.
    for vr, vc in visited:
        assert max(abs(vr - r), abs(vc - c)) >= 4


def test_frontier_returns_none_for_unknown_room():
    provider, obs = _make_two_room_map()
    assert find_frontier_in_room("garage", provider, obs) is None
