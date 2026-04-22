"""
Phase G — surrogate furniture (level-3 enabling) and LLM retry/stats.

These tests cover the additive Phase G hardening on top of the existing
strategic policy:

  1. ``_select_search_proxies`` swaps small-object targets for an
     available surrogate furniture category and preserves the original
     label for downstream YOLOE verification.
  2. ``LlmStats`` accumulates calls / invalid / retries / fallbacks.
  3. ``choose_next_action`` retries the LLM with feedback when the first
     attempt is invalid and reports success on the retry.
  4. Retry exhaustion falls back to the heuristic and increments the
     fallback counter.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vlmaps.policy import ActionType, LlmStats
from vlmaps.policy import strategic_policy as sp
from vlmaps.utils.object_priors import (
    get_surrogates,
    has_surrogates,
    pick_surrogate,
)


# ---------------------------------------------------------------------------
# Test doubles (mirrors the ones used in test_phase_g_strategic_policy.py).
# ---------------------------------------------------------------------------


class _FakeSearchState:
    def __init__(self, rooms, recent=None):
        self.rooms = {
            room: SimpleNamespace(target_relevance=0.0) for room in rooms
        }
        self._recent = list(recent or [])
        self.visited_cells = []

    def recent_actions(self, n: int = 5):
        return list(self._recent[-n:])

    def room_summaries_for_llm(self) -> str:
        return "\n".join(f"- {room}" for room in self.rooms.keys())


class _FakeMap:
    def __init__(self, categories):
        self.categories = list(categories)


class _FakeRobot:
    def __init__(self, categories):
        self.map = _FakeMap(categories)


class _FakeCtx:
    def __init__(
        self,
        *,
        target,
        search_state,
        current_room="hallway",
        last_candidate_centroid=None,
        map_categories=None,
    ):
        self.target = target
        self.search_state = search_state
        self.current_room = current_room
        self.last_candidate_centroid = last_candidate_centroid
        self.room_provider = None
        self.robot = _FakeRobot(map_categories or [])
        self.rgb_map_2d = None
        self.heatmap = None
        self.kept_components = []


def _snapshot(**kwargs):
    defaults = dict(
        target="sink",
        current_room="hallway",
        direct_query_mode=False,
        heatmap_evidence={},
        room_scores={},
        selected_room=None,
        best_component=None,
        candidate_pool=[],
        frontier_rooms={},
        query_priors={},
    )
    defaults.update(kwargs)
    return sp.StrategySnapshot(**defaults)


# ---------------------------------------------------------------------------
# 1. Object → surrogate furniture mapping.
# ---------------------------------------------------------------------------


def test_object_priors_known_objects_have_surrogates():
    for obj in ("bottle", "cup", "laptop", "backpack", "book"):
        assert has_surrogates(obj), obj
        assert get_surrogates(obj), obj


def test_object_priors_unknown_object_has_no_surrogates():
    assert not has_surrogates("dragon")
    assert get_surrogates("dragon") == []


def test_pick_surrogate_returns_first_available():
    chosen, considered = pick_surrogate("bottle", ["chair", "table", "counter"])
    # The mapping for "bottle" lists "counter" before "table"; both available
    # → "counter" must be picked.
    assert chosen == "counter"
    assert "counter" in considered
    assert considered[0] == "counter"


def test_pick_surrogate_skips_missing_categories():
    # Only "table" is available out of bottle's surrogates → table is picked
    # even though it's not the first choice in the curated list.
    chosen, considered = pick_surrogate("bottle", ["chair", "table"])
    assert chosen == "table"
    assert considered[0] == "counter"  # original ranking preserved in considered


def test_pick_surrogate_returns_empty_when_nothing_matches():
    chosen, considered = pick_surrogate("bottle", ["chair", "sofa"])
    assert chosen == ""
    assert considered  # list is non-empty (curated mapping was found)


# ---------------------------------------------------------------------------
# 2. _select_search_proxies behaviour.
# ---------------------------------------------------------------------------


def test_select_proxies_passthrough_for_furniture_target():
    ss = _FakeSearchState(["kitchen"])
    ctx = _FakeCtx(target="table", search_state=ss, map_categories=["table", "chair"])
    original, effective, considered = sp._select_search_proxies(
        ctx, ["table"], ["table"]
    )
    assert original == "table"
    assert effective == "table"
    assert considered == []


def test_select_proxies_swaps_small_object_for_surrogate():
    ss = _FakeSearchState(["kitchen"])
    ctx = _FakeCtx(
        target="bottle", search_state=ss,
        map_categories=["counter", "table", "chair"],
    )
    original, effective, considered = sp._select_search_proxies(
        ctx, ["bottle"], ["counter", "table"]
    )
    assert original == "bottle"
    assert effective == "counter"  # first surrogate available
    assert "counter" in considered


def test_select_proxies_keeps_original_when_no_surrogate_matches():
    ss = _FakeSearchState(["kitchen"])
    ctx = _FakeCtx(
        target="bottle", search_state=ss,
        map_categories=["sofa", "bed"],
    )
    original, effective, considered = sp._select_search_proxies(
        ctx, ["bottle"], ["sofa"]
    )
    assert original == "bottle"
    assert effective == "bottle"
    assert considered  # surrogate list was looked up but none matched


def test_select_proxies_handles_unknown_object_without_surrogates():
    ss = _FakeSearchState(["kitchen"])
    ctx = _FakeCtx(
        target="dragon", search_state=ss,
        map_categories=["table"],
    )
    original, effective, considered = sp._select_search_proxies(
        ctx, ["dragon"], ["table"]
    )
    assert original == "dragon"
    assert effective == "dragon"
    assert considered == []


# ---------------------------------------------------------------------------
# 3. LLM retry path through choose_next_action.
# ---------------------------------------------------------------------------


def _patch_snapshot(monkeypatch, snap):
    monkeypatch.setattr(sp, "prepare_strategy_snapshot", lambda *a, **kw: snap)


def test_llm_retry_succeeds_on_second_attempt(monkeypatch):
    ss = _FakeSearchState(["hallway", "bathroom"])
    ctx = _FakeCtx(target="sink", search_state=ss, current_room="hallway")
    snap = _snapshot(
        selected_room="bathroom",
        best_component={"room": "bathroom", "centroid": [42, 55]},
        candidate_pool=[{"room": "bathroom", "centroid": [42, 55], "quality": 1.0}],
    )
    _patch_snapshot(monkeypatch, snap)

    payloads = iter([
        # First attempt: room not in known_rooms → validator rejects.
        '{"type":"go_to_room","room":"garage"}',
        # Retry: corrected room.
        '{"type":"go_to_room","room":"bathroom"}',
    ])

    def fake_query(ctx_, snap_, heur, model="gpt-4o-mini", feedback=None):
        return next(payloads)

    monkeypatch.setattr(sp, "_query_llm_action_text", fake_query)
    stats = LlmStats()

    action, _ = sp.choose_next_action(
        ctx, ["sink"], ["sink"], policy_mode="hybrid", stats=stats
    )

    assert action.type is ActionType.GO_TO_ROOM
    assert action.room == "bathroom"
    assert stats.calls == 2
    assert stats.invalid == 1
    assert stats.retries == 1
    assert stats.retries_succeeded == 1
    assert stats.fallbacks == 0


def test_llm_retry_exhausted_falls_back_to_heuristic(monkeypatch):
    ss = _FakeSearchState(["hallway", "bathroom"])
    ctx = _FakeCtx(target="sink", search_state=ss, current_room="hallway")
    snap = _snapshot(
        selected_room="bathroom",
        best_component={"room": "bathroom", "centroid": [42, 55]},
        candidate_pool=[{"room": "bathroom", "centroid": [42, 55], "quality": 1.0}],
    )
    _patch_snapshot(monkeypatch, snap)

    monkeypatch.setattr(
        sp, "_query_llm_action_text",
        lambda *a, **kw: '{"type":"go_to_room","room":"garage"}',
    )
    stats = LlmStats()
    action, _ = sp.choose_next_action(
        ctx, ["sink"], ["sink"], policy_mode="hybrid", stats=stats, max_retries=1,
    )

    # Heuristic: not in selected_room yet → GO_TO_ROOM bathroom.
    assert action.type is ActionType.GO_TO_ROOM
    assert action.room == "bathroom"
    assert stats.calls == 2
    assert stats.invalid == 2
    assert stats.retries == 1
    assert stats.retries_succeeded == 0
    assert stats.fallbacks == 1


def test_llm_unavailable_does_not_retry(monkeypatch):
    ss = _FakeSearchState(["hallway", "bathroom"])
    ctx = _FakeCtx(target="sink", search_state=ss, current_room="hallway")
    snap = _snapshot(
        selected_room="bathroom",
        best_component={"room": "bathroom", "centroid": [42, 55]},
        candidate_pool=[{"room": "bathroom", "centroid": [42, 55], "quality": 1.0}],
    )
    _patch_snapshot(monkeypatch, snap)

    monkeypatch.setattr(sp, "_query_llm_action_text", lambda *a, **kw: None)
    stats = LlmStats()
    sp.choose_next_action(
        ctx, ["sink"], ["sink"], policy_mode="hybrid", stats=stats,
    )
    assert stats.calls == 0
    assert stats.retries == 0
    assert stats.fallbacks == 1


def test_llm_stats_as_dict_keys_are_eval_summary_friendly():
    stats = LlmStats(calls=3, invalid=1, retries=1, retries_succeeded=1, fallbacks=0)
    d = stats.as_dict()
    assert set(d.keys()) == {
        "llm_calls", "llm_invalid", "llm_retries",
        "llm_retries_succeeded", "llm_fallbacks",
    }
    assert d["llm_calls"] == 3
