from __future__ import annotations

from types import SimpleNamespace

from application import interactive_object_nav as base_nav
from vlmaps.policy import strategic_policy as sp
from vlmaps.utils.llm_utils import (
    canonicalize_open_vocab_target,
    resolve_open_vocab_target,
)


def test_canonicalize_open_vocab_target_aliases():
    assert canonicalize_open_vocab_target("computer") == "laptop"
    assert canonicalize_open_vocab_target("books") == "book"
    assert canonicalize_open_vocab_target("Tea Pot") == "teapot"


def test_resolve_open_vocab_target_falls_back_to_manual_surrogates(monkeypatch):
    monkeypatch.setattr(
        "vlmaps.utils.llm_utils._query_open_vocab_target_resolution",
        lambda *args, **kwargs: None,
    )

    resolution = resolve_open_vocab_target(
        "computer",
        ["desk", "table", "chair", "bed"],
        known_rooms=["office", "bedroom", "living room"],
        present_categories=["desk", "chair"],
    )

    assert resolution.original_target == "computer"
    assert resolution.canonical_target == "laptop"
    assert resolution.effective_target == "desk"
    assert resolution.source == "fallback"
    assert resolution.surrogate_categories[:2] == ["desk", "table"]
    assert "office" in resolution.likely_rooms


def test_resolve_open_vocab_target_accepts_llm_resolution(monkeypatch):
    monkeypatch.setattr(
        "vlmaps.utils.llm_utils._query_open_vocab_target_resolution",
        lambda *args, **kwargs: {
            "canonical_target": "bottle",
            "surrogate_categories": ["counter", "table", "lamp"],
            "likely_rooms": ["kitchen", "bathroom"],
        },
    )

    resolution = resolve_open_vocab_target(
        "water bottle",
        ["counter", "table", "shelf"],
        known_rooms=["kitchen", "bathroom", "bedroom"],
        present_categories=["counter"],
    )

    assert resolution.canonical_target == "bottle"
    assert resolution.effective_target == "counter"
    assert resolution.source == "llm"
    assert resolution.surrogate_categories == ["counter", "table"]
    assert resolution.likely_rooms == ["kitchen", "bathroom"]


def test_build_resolved_target_plans_marks_room_commands(monkeypatch):
    monkeypatch.setattr(
        "vlmaps.utils.llm_utils._query_open_vocab_target_resolution",
        lambda *args, **kwargs: None,
    )

    room_provider = SimpleNamespace(
        is_available=lambda: True,
        list_rooms=lambda: ["kitchen", "bedroom"],
        get_room_centroid=lambda name: [10, 20] if name == "kitchen" else None,
    )
    plans = base_nav.build_resolved_target_plans(
        ["kitchen", "computer"],
        room_provider=room_provider,
        room_regions={},
        available_categories=["desk", "table", "chair", "bed"],
        present_categories=["desk"],
    )

    assert len(plans) == 2
    assert plans[0].room_goal == [10, 20]
    assert plans[0].resolution_source == "room_command"
    assert plans[1].canonical_target == "laptop"
    assert plans[1].effective_target == "desk"


def test_strategy_proxy_selection_uses_context_resolution():
    ctx = SimpleNamespace(
        target="laptop",
        original_target="computer",
        effective_target="desk",
        surrogate_categories=["desk", "table"],
        likely_rooms=["office"],
        search_state=None,
        robot=SimpleNamespace(map=SimpleNamespace(categories=["desk", "table"])),
    )

    original, effective, considered = sp._select_search_proxies(
        ctx,
        ["laptop"],
        ["desk"],
    )

    assert original == "laptop"
    assert effective == "desk"
    assert considered == ["desk", "table"]
