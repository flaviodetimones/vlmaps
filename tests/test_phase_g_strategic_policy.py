from __future__ import annotations

from types import SimpleNamespace

from vlmaps.policy import ActionType
from vlmaps.policy.strategic_policy import StrategySnapshot, choose_next_action


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


class _FakeCtx:
    def __init__(self, *, search_state, current_room="hallway", last_candidate_centroid=None):
        self.target = "sink"
        self.search_state = search_state
        self.current_room = current_room
        self.last_candidate_centroid = last_candidate_centroid
        self.room_provider = None
        self.robot = SimpleNamespace()
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
    return StrategySnapshot(**defaults)


def test_choose_next_action_returns_verify_after_failed_inspection():
    ss = _FakeSearchState(
        ["hallway", "bathroom"],
        recent=[{"action": "inspect_candidate(bathroom, (10,12))", "outcome": "inspect_no_detection"}],
    )
    ctx = _FakeCtx(search_state=ss, last_candidate_centroid=(10, 12))
    action, snapshot = choose_next_action(ctx, ["sink"], ["sink"], policy_mode="heuristic")
    assert action.type is ActionType.VERIFY_TARGET
    assert snapshot is None


def test_heuristic_prefers_go_to_room_before_inspecting(monkeypatch):
    ss = _FakeSearchState(["hallway", "bathroom"])
    ctx = _FakeCtx(search_state=ss, current_room="hallway")
    snap = _snapshot(
        selected_room="bathroom",
        best_component={"room": "bathroom", "centroid": [42, 55]},
        candidate_pool=[{"room": "bathroom", "centroid": [42, 55], "quality": 1.0}],
    )
    monkeypatch.setattr(
        "vlmaps.policy.strategic_policy.prepare_strategy_snapshot",
        lambda *_args, **_kwargs: snap,
    )
    action, _ = choose_next_action(ctx, ["sink"], ["sink"], policy_mode="heuristic")
    assert action.type is ActionType.GO_TO_ROOM
    assert action.room == "bathroom"


def test_heuristic_direct_query_picks_inspect_candidate(monkeypatch):
    ss = _FakeSearchState(["living room"])
    ctx = _FakeCtx(search_state=ss, current_room="living room")
    snap = _snapshot(
        current_room="living room",
        direct_query_mode=True,
        best_component={"room": "living room", "centroid": [8, 9]},
        candidate_pool=[{"room": "living room", "centroid": [8, 9], "quality": 2.0}],
    )
    monkeypatch.setattr(
        "vlmaps.policy.strategic_policy.prepare_strategy_snapshot",
        lambda *_args, **_kwargs: snap,
    )
    action, _ = choose_next_action(ctx, ["sofa"], ["sofa"], policy_mode="heuristic")
    assert action.type is ActionType.INSPECT_CANDIDATE
    assert action.room == "living room"
    assert action.candidate_centroid == (8, 9)


def test_heuristic_uses_explore_room_when_no_candidate(monkeypatch):
    ss = _FakeSearchState(["bathroom"])
    ctx = _FakeCtx(search_state=ss, current_room="bathroom")
    snap = _snapshot(
        current_room="bathroom",
        selected_room="bathroom",
        best_component=None,
        frontier_rooms={"bathroom": (20, 30)},
    )
    monkeypatch.setattr(
        "vlmaps.policy.strategic_policy.prepare_strategy_snapshot",
        lambda *_args, **_kwargs: snap,
    )
    action, _ = choose_next_action(ctx, ["sink"], ["sink"], policy_mode="heuristic")
    assert action.type is ActionType.EXPLORE_ROOM
    assert action.room == "bathroom"


def test_llm_action_overrides_heuristic_when_valid(monkeypatch):
    ss = _FakeSearchState(["hallway", "bathroom"])
    ctx = _FakeCtx(search_state=ss, current_room="hallway")
    snap = _snapshot(
        selected_room="bathroom",
        best_component={"room": "bathroom", "centroid": [42, 55]},
        candidate_pool=[{"room": "bathroom", "centroid": [42, 55], "quality": 1.0}],
    )
    monkeypatch.setattr(
        "vlmaps.policy.strategic_policy.prepare_strategy_snapshot",
        lambda *_args, **_kwargs: snap,
    )
    monkeypatch.setattr(
        "vlmaps.policy.strategic_policy._query_llm_action_text",
        lambda *_args, **_kwargs: '{"type":"inspect_candidate","room":"bathroom","candidate_centroid":[42,55]}',
    )
    action, _ = choose_next_action(ctx, ["sink"], ["sink"], policy_mode="hybrid")
    assert action.type is ActionType.INSPECT_CANDIDATE
    assert action.candidate_centroid == (42, 55)


def test_invalid_llm_action_falls_back_to_heuristic(monkeypatch):
    ss = _FakeSearchState(["hallway", "bathroom"])
    ctx = _FakeCtx(search_state=ss, current_room="hallway")
    snap = _snapshot(
        selected_room="bathroom",
        best_component={"room": "bathroom", "centroid": [42, 55]},
        candidate_pool=[{"room": "bathroom", "centroid": [42, 55], "quality": 1.0}],
    )
    monkeypatch.setattr(
        "vlmaps.policy.strategic_policy.prepare_strategy_snapshot",
        lambda *_args, **_kwargs: snap,
    )
    monkeypatch.setattr(
        "vlmaps.policy.strategic_policy._query_llm_action_text",
        lambda *_args, **_kwargs: '{"type":"go_to_room","room":"garage"}',
    )
    action, _ = choose_next_action(ctx, ["sink"], ["sink"], policy_mode="hybrid")
    assert action.type is ActionType.GO_TO_ROOM
    assert action.room == "bathroom"
