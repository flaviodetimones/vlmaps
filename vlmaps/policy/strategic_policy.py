"""
strategic_policy.py
===================
Phase G — strategic room-aware policy over structured state.

This layer sits above the typed Action vocabulary from Phase F and decides the
*next* action to execute, not the low-level trajectory. It works in three
stages:

1. Build a compact structured snapshot of the current search episode
   (rooms, priors, evidence, recent actions, candidate shortlist, frontiers).
2. Optionally ask an LLM for the next Action JSON.
3. Validate the proposed action and fall back to a deterministic heuristic
   when the LLM is unavailable, returns garbage, or proposes something unsafe.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vlmaps.policy.actions import ACTION_SCHEMA_PROMPT, Action, ActionType, parse_action_json
from vlmaps.policy.frontier import find_frontier_in_room
from vlmaps.utils.object_priors import has_surrogates, pick_surrogate


def _base_nav():
    from application import interactive_object_nav as base_nav

    return base_nav


@dataclass
class StrategySnapshot:
    target: str
    current_room: Optional[str]
    direct_query_mode: bool
    heatmap_evidence: Dict[str, float]
    room_scores: Dict[str, Dict[str, Any]]
    selected_room: Optional[str]
    best_component: Optional[Dict[str, Any]]
    candidate_pool: List[Dict[str, Any]]
    frontier_rooms: Dict[str, Tuple[int, int]]
    query_priors: Dict[str, float]
    # Phase G — level-3 object support. When the requested target is not in
    # the VLMap furniture vocabulary, the snapshot is built around a surrogate
    # furniture category (``effective_target``) but the original object label
    # is preserved for the YOLOE verification step at arrival.
    original_target: Optional[str] = None
    effective_target: Optional[str] = None
    surrogates_considered: List[str] = field(default_factory=list)


@dataclass
class LlmStats:
    """Counters accumulated across a target search for [eval-summary]."""
    calls: int = 0
    invalid: int = 0
    retries: int = 0
    retries_succeeded: int = 0
    fallbacks: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "llm_calls": self.calls,
            "llm_invalid": self.invalid,
            "llm_retries": self.retries,
            "llm_retries_succeeded": self.retries_succeeded,
            "llm_fallbacks": self.fallbacks,
        }


def _annotate_candidate_rooms(room_provider, kept_components: list) -> None:
    if room_provider is None or not room_provider.is_available():
        return
    rooms_count = {}
    for comp in kept_components:
        cr, cc = comp["centroid"]
        comp["room"] = room_provider.get_room_at_cell(int(cr), int(cc))
        name = comp["room"] or "unknown"
        rooms_count[name] = rooms_count.get(name, 0) + 1
    if rooms_count:
        print(f"  Candidates by room: {rooms_count}")


def _fuse_heatmap_evidence(cat: str, search_state, kept_components: list) -> dict:
    from vlmaps.utils.room_priors import (
        compute_heatmap_room_evidence,
        compute_room_priors,
    )

    if search_state is None or not kept_components:
        return {}
    heatmap_ev = compute_heatmap_room_evidence(kept_components)
    if heatmap_ev:
        ev_sorted = sorted(heatmap_ev.items(), key=lambda x: -x[1])
        print(
            f"  Room evidence from heatmap: "
            f"{ {r: round(v, 3) for r, v in ev_sorted} }"
        )

    known_rooms = list(search_state.rooms.keys())
    seen_objs = {n: rs.objects_seen for n, rs in search_state.rooms.items()}
    new_priors = compute_room_priors(
        cat,
        known_rooms,
        seen_objs,
        heatmap_evidence=heatmap_ev,
        query_type="direct",
    )
    for room_name, score in new_priors.items():
        if room_name in search_state.rooms:
            search_state.rooms[room_name].target_relevance = score
    np_sorted = sorted(new_priors.items(), key=lambda x: -x[1])
    print(
        "  Final room priors after evidence fusion: "
        + ", ".join(f"{r}={v:.2f}" for r, v in np_sorted[:6] if v > 0.01)
    )
    return heatmap_ev


def _candidate_shortlist(candidate_pool: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    shortlist = []
    for comp in candidate_pool[:limit]:
        centroid = (int(comp["centroid"][0]), int(comp["centroid"][1]))
        shortlist.append(
            {
                "room": comp.get("room"),
                "centroid": centroid,
                "quality": float(comp.get("_effective_quality", comp.get("quality", 0.0))),
                "distance": float(comp.get("_distance_to_robot", 0.0)),
            }
        )
    return shortlist


def _frontier_suggestions(ctx, selected_room: Optional[str], room_scores: dict) -> Dict[str, Tuple[int, int]]:
    suggestions: Dict[str, Tuple[int, int]] = {}
    rooms = []
    if selected_room:
        rooms.append(selected_room)
    ranked_rooms = sorted(room_scores.items(), key=lambda kv: kv[1]["score"], reverse=True)
    for room_name, _info in ranked_rooms[:3]:
        if room_name not in rooms:
            rooms.append(room_name)
    if ctx.current_room and ctx.current_room not in rooms:
        rooms.append(ctx.current_room)

    for room_name in rooms:
        frontier = find_frontier_in_room(
            room_name,
            ctx.room_provider,
            ctx.robot.map.obstacles_map,
            visited_cells=(ctx.search_state.visited_cells if ctx.search_state else None),
        )
        if frontier is not None:
            suggestions[room_name] = (int(frontier[0]), int(frontier[1]))
    return suggestions


def _select_search_proxies(ctx, categories: list, present_categories: list) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Pick the heatmap-side label for ``ctx.target``.

    For furniture targets (already in the VLMap vocabulary) the original
    target is used unchanged. For level-3 small objects (bottle, cup,
    laptop, ...) that the VLMap does not represent, swap the heatmap
    target for the first available *surrogate furniture* category — the
    YOLOE verification at arrival keeps using the original object label,
    so the policy still earns the level-3 detection.

    Returns ``(original, effective, considered)``:
      - ``original``: original ctx.target (preserved for YOLOE).
      - ``effective``: surrogate to use for the heatmap, or original if no
        substitution applies. ``None`` only if ctx.target itself is empty.
      - ``considered``: surrogate list looked up (empty if not applicable).
    """
    target = (ctx.target or "").strip()
    if not target:
        return None, None, []

    map_categories = []
    try:
        map_categories = list(getattr(ctx.robot.map, "categories", []) or [])
    except Exception:
        map_categories = []
    available = set(c.strip().lower() for c in (map_categories + list(present_categories)) if c)

    target_lower = target.lower()
    if target_lower in available:
        return target, target, []

    if not has_surrogates(target):
        return target, target, []

    chosen, considered = pick_surrogate(target, list(available))
    if not chosen:
        print(
            f"  [strategy] Target '{target}' not in VLMap and no surrogate from "
            f"{considered} is available — falling back to raw heatmap query"
        )
        return target, target, considered

    print(
        f"  [strategy] Target '{target}' not in VLMap categories; using "
        f"surrogate furniture '{chosen}' for heatmap (considered={considered})"
    )
    return target, chosen, considered


def prepare_strategy_snapshot(ctx, categories: list, present_categories: list) -> StrategySnapshot:
    base = _base_nav()
    original_target, effective_target, surrogates_considered = _select_search_proxies(
        ctx, categories, present_categories
    )
    cat = effective_target or ctx.target
    # Use the surrogate (if any) only for the heatmap-side reasoning; the
    # original target stays in ctx.target so downstream YOLOE keeps verifying
    # the actual object the user asked for.
    surrogate_swap_active = bool(
        original_target and effective_target and original_target != effective_target
    )
    search_state = ctx.search_state
    room_provider = ctx.room_provider

    print(f"  Current room: {ctx.current_room or 'unknown'}")
    print("  Computing semantic heatmap...")
    heatmap, kept_components = base.compute_heatmap(ctx.robot, cat)
    ctx.heatmap = heatmap
    ctx.kept_components = kept_components

    # Direct furniture query is decided over the *original* target; if we
    # surrogate-swapped (level-3 object), this is by definition not a direct
    # furniture nearest-first lookup.
    _direct_check_label = (original_target or cat).lower()
    direct_query_mode = (
        not surrogate_swap_active
        and len(categories) == 1
        and _direct_check_label in {c.lower() for c in present_categories}
    )
    if direct_query_mode:
        print("  Query type: direct furniture — nearest-first policy remains preferred")
    else:
        print("  Query type: strategic room-aware")

    _annotate_candidate_rooms(room_provider, kept_components)

    heatmap_ev = {}
    selected_room = None
    room_scores = {}
    candidate_pool = kept_components
    if search_state and kept_components:
        heatmap_ev = _fuse_heatmap_evidence(cat, search_state, kept_components)

    if search_state and kept_components and search_state.rooms and not direct_query_mode:
        robot_rc = [ctx.robot.curr_pos_on_map[0], ctx.robot.curr_pos_on_map[1]]
        selected_room, room_scores = base.select_best_room(
            search_state,
            robot_rc,
            heatmap_ev,
            ctx.robot.map.obstacles_map,
            kept_components,
            current_room=ctx.current_room,
        )
        if selected_room is not None:
            room_kept = [c for c in kept_components if c.get("room") == selected_room]
            if room_kept:
                print(
                    f"  [strategy] Chosen room for '{cat}': {selected_room} "
                    f"({len(room_kept)}/{len(kept_components)} candidate(s))"
                )
                candidate_pool = room_kept
            else:
                print(
                    f"  [strategy] Chosen room '{selected_room}' has no direct "
                    f"candidates — falling back to global ranking"
                )
    elif direct_query_mode:
        print(
            f"  [strategy] Direct furniture query for '{cat}' "
            f"— room selector disabled"
        )

    base.show_map(
        ctx.robot,
        ctx.rgb_map_2d,
        heatmap_2d=heatmap,
        label=f"Strategic planning: {cat}",
    )
    base.cv2.waitKey(150)

    query_priors = {r: rs.target_relevance for r, rs in search_state.rooms.items()} if search_state else {}
    tried = getattr(search_state, "_tried_centroids", set()) if search_state else set()
    robot_rc_sel = [ctx.robot.curr_pos_on_map[0], ctx.robot.curr_pos_on_map[1]]
    best_component = base.select_best_candidate(
        candidate_pool,
        ctx.current_room,
        query_priors,
        tried,
        robot_pos=robot_rc_sel,
        enable_room_gate=not direct_query_mode,
        prefer_nearest_only=direct_query_mode,
    ) if candidate_pool else None

    frontier_rooms = _frontier_suggestions(ctx, selected_room, room_scores)
    return StrategySnapshot(
        target=cat,
        current_room=ctx.current_room,
        direct_query_mode=direct_query_mode,
        heatmap_evidence=heatmap_ev,
        room_scores=room_scores,
        selected_room=selected_room,
        best_component=best_component,
        candidate_pool=candidate_pool,
        frontier_rooms=frontier_rooms,
        query_priors=query_priors,
        original_target=original_target or ctx.target,
        effective_target=cat,
        surrogates_considered=surrogates_considered,
    )


def _recent_actions_text(search_state) -> str:
    if search_state is None:
        return "- (none)"
    recent = search_state.recent_actions(5)
    if not recent:
        return "- (none)"
    return "\n".join(
        f"- {entry.get('action')} -> {entry.get('outcome') or 'pending'}"
        for entry in recent
    )


def _room_scores_text(snapshot: StrategySnapshot) -> str:
    if not snapshot.room_scores:
        return "- (no room scores)"
    ranked = sorted(snapshot.room_scores.items(), key=lambda kv: kv[1]["score"], reverse=True)
    lines = []
    for room_name, info in ranked[:5]:
        lines.append(
            f"- {room_name}: score={info['score']:.3f}, prior={info['prior']:.2f}, "
            f"ev={info['evidence']:.2f}, unexp={info['unexplored']:.2f}, "
            f"cost={info['cost']:.2f}, pen={info['penalty']:.2f}, cand={info['candidates']}"
        )
    return "\n".join(lines)


def _candidate_text(snapshot: StrategySnapshot) -> str:
    shortlist = _candidate_shortlist(snapshot.candidate_pool)
    if not shortlist:
        return "- (no candidates)"
    lines = []
    for item in shortlist:
        lines.append(
            f"- room={item['room'] or 'unknown'} | centroid={list(item['centroid'])} | "
            f"quality={item['quality']:.3f} | dist={item['distance']:.1f}"
        )
    return "\n".join(lines)


def _frontier_text(snapshot: StrategySnapshot) -> str:
    if not snapshot.frontier_rooms:
        return "- (no frontier suggestions)"
    return "\n".join(
        f"- {room}: frontier={list(cell)}"
        for room, cell in snapshot.frontier_rooms.items()
    )


def _build_llm_messages(ctx, snapshot: StrategySnapshot, heuristic_action: Action) -> List[Dict[str, str]]:
    search_state = ctx.search_state
    room_summary = search_state.room_summaries_for_llm() if search_state is not None else "- (no room state)"
    current_room = snapshot.current_room or "unknown"
    allowed_rooms = sorted(search_state.rooms.keys()) if search_state and search_state.rooms else []
    allowed_centroids = [
        list(item["centroid"]) for item in _candidate_shortlist(snapshot.candidate_pool, limit=8)
    ]
    system_prompt = (
        "You are the strategic high-level policy of a room-aware mobile robot.\n"
        "Choose only the NEXT high-level action. Do not describe trajectories.\n"
        "The low-level navigation and visual verification are handled elsewhere.\n"
        "Prefer go_to_room/explore_room/inspect_candidate/verify_target/done as appropriate.\n"
        "If the last action was inspect_candidate and it did not detect the target, "
        "verify_target is often the right immediate follow-up.\n"
        "Use only known room names and only candidate centroids from the shortlist.\n\n"
        + ACTION_SCHEMA_PROMPT
    )
    user_prompt = (
        f"Task: find {snapshot.target}\n"
        f"Current room: {current_room}\n"
        f"Direct furniture query: {'yes' if snapshot.direct_query_mode else 'no'}\n"
        f"Suggested room by heuristic: {snapshot.selected_room or 'none'}\n"
        f"Heuristic fallback action: {heuristic_action.short()}\n\n"
        f"Known rooms:\n{room_summary}\n\n"
        f"Recent actions:\n{_recent_actions_text(search_state)}\n\n"
        f"Room ranking:\n{_room_scores_text(snapshot)}\n\n"
        f"Candidate shortlist:\n{_candidate_text(snapshot)}\n\n"
        f"Frontier suggestions:\n{_frontier_text(snapshot)}\n\n"
        f"Allowed rooms: {allowed_rooms}\n"
        f"Allowed centroids for inspect_candidate: {allowed_centroids}\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _query_llm_action_text(
    ctx,
    snapshot: StrategySnapshot,
    heuristic_action: Action,
    model: str = "gpt-4o-mini",
    feedback: Optional[str] = None,
) -> Optional[str]:
    openai_key = os.environ.get("OPENAI_KEY")
    if not openai_key:
        return None

    import openai

    messages = _build_llm_messages(ctx, snapshot, heuristic_action)
    if feedback:
        # Phase G — retry path. Append the previous attempt's parser/validator
        # error so the LLM gets concrete feedback rather than a blind retry.
        messages.append({
            "role": "user",
            "content": (
                "Your previous action was rejected.\n"
                f"Reason: {feedback}\n"
                "Emit ONE corrected action JSON object now, using only the "
                "allowed rooms and centroids listed above. Do not repeat the "
                "previous mistake. Respond with the JSON only."
            ),
        })
    client = openai.OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=200,
        temperature=0.0,
    )
    text = response.choices[0].message.content.strip()
    return text


def _validate_llm_action(action: Action, snapshot: StrategySnapshot, ctx) -> bool:
    search_state = ctx.search_state
    known_rooms = set(search_state.rooms.keys()) if search_state and search_state.rooms else set()
    allowed_centroids = {
        (int(comp["centroid"][0]), int(comp["centroid"][1]))
        for comp in snapshot.candidate_pool
    }

    if action.type in (ActionType.GO_TO_ROOM, ActionType.EXPLORE_ROOM):
        return bool(action.room and action.room in known_rooms)

    if action.type is ActionType.INSPECT_CANDIDATE:
        if not action.room or action.room not in known_rooms:
            return False
        if action.candidate_centroid not in allowed_centroids:
            return False
        return True

    if action.type is ActionType.VERIFY_TARGET:
        return ctx.last_candidate_centroid is not None

    if action.type is ActionType.DONE:
        return True

    return False


def _heuristic_next_action(ctx, snapshot: StrategySnapshot) -> Action:
    recent = ctx.search_state.recent_actions(1) if ctx.search_state is not None else []
    if (
        recent
        and recent[-1].get("action", "").startswith("inspect_candidate")
        and recent[-1].get("outcome") == "inspect_no_detection"
        and ctx.last_candidate_centroid is not None
    ):
        return Action(type=ActionType.VERIFY_TARGET)

    best_comp = snapshot.best_component
    if snapshot.selected_room and snapshot.current_room != snapshot.selected_room and not snapshot.direct_query_mode:
        return Action(type=ActionType.GO_TO_ROOM, room=snapshot.selected_room)

    if best_comp is not None:
        comp_room = best_comp.get("room") or snapshot.selected_room or snapshot.current_room or "unknown"
        centroid = (int(best_comp["centroid"][0]), int(best_comp["centroid"][1]))
        return Action(
            type=ActionType.INSPECT_CANDIDATE,
            room=comp_room,
            candidate_centroid=centroid,
        )

    if snapshot.selected_room and snapshot.selected_room in snapshot.frontier_rooms:
        return Action(type=ActionType.EXPLORE_ROOM, room=snapshot.selected_room)

    if snapshot.current_room and snapshot.current_room in snapshot.frontier_rooms:
        return Action(type=ActionType.EXPLORE_ROOM, room=snapshot.current_room)

    return Action(type=ActionType.DONE, reason="no viable candidate")


def _try_llm_once(
    ctx,
    snapshot: StrategySnapshot,
    heuristic_action: Action,
    *,
    llm_model: str,
    feedback: Optional[str],
    stats: Optional[LlmStats],
) -> Tuple[Optional[Action], Optional[str]]:
    """Run one LLM attempt. Returns (validated_action, error_text).

    On success: ``(action, None)``. On unavailable / parse / validation
    failure: ``(None, reason)`` so the caller can decide whether to retry.
    """
    try:
        payload = _query_llm_action_text(
            ctx, snapshot, heuristic_action, model=llm_model, feedback=feedback
        )
    except Exception as exc:
        return None, f"llm_error: {exc}"

    if not payload:
        return None, "llm_unavailable"

    if stats is not None:
        stats.calls += 1
    try:
        llm_action = parse_action_json(payload)
    except Exception as exc:
        if stats is not None:
            stats.invalid += 1
        return None, f"parse_error: {exc}"

    if not _validate_llm_action(llm_action, snapshot, ctx):
        if stats is not None:
            stats.invalid += 1
        return None, f"validator_rejected: {llm_action.short()}"

    return llm_action, None


def choose_next_action(
    ctx,
    categories: list,
    present_categories: list,
    *,
    policy_mode: str = "hybrid",
    llm_model: str = "gpt-4o-mini",
    stats: Optional[LlmStats] = None,
    max_retries: int = 1,
) -> Tuple[Action, Optional[StrategySnapshot]]:
    """Choose the next high-level Action for the executor.

    policy_mode:
      - "heuristic": deterministic fallback only
      - "llm": require the LLM, fall back only on malformed output/errors
      - "hybrid": prefer LLM, deterministic fallback if unavailable/invalid

    When ``stats`` is provided, LLM call counters (calls, invalid, retries,
    fallbacks) are accumulated for emission in the per-query [eval-summary].
    ``max_retries`` bounds the number of feedback-driven LLM retries (1 by
    default — first try plus one corrective retry).
    """
    policy_mode = (policy_mode or "hybrid").strip().lower()

    recent = ctx.search_state.recent_actions(1) if ctx.search_state is not None else []
    if (
        recent
        and recent[-1].get("action", "").startswith("inspect_candidate")
        and recent[-1].get("outcome") == "inspect_no_detection"
        and ctx.last_candidate_centroid is not None
    ):
        action = Action(type=ActionType.VERIFY_TARGET)
        print(f"  [strategy] Pending follow-up action: {action.short()}")
        return action, None

    snapshot = prepare_strategy_snapshot(ctx, categories, present_categories)
    heuristic_action = _heuristic_next_action(ctx, snapshot)

    if policy_mode == "heuristic" or snapshot.direct_query_mode:
        print(f"  [strategy] Heuristic action: {heuristic_action.short()}")
        return heuristic_action, snapshot

    feedback: Optional[str] = None
    last_reason: Optional[str] = None
    for attempt in range(max_retries + 1):
        action, reason = _try_llm_once(
            ctx, snapshot, heuristic_action,
            llm_model=llm_model, feedback=feedback, stats=stats,
        )
        if action is not None:
            if attempt > 0 and stats is not None:
                stats.retries_succeeded += 1
            tag = "LLM action" if attempt == 0 else f"LLM action (retry {attempt})"
            print(f"  [strategy] {tag}: {action.short()}")
            return action, snapshot

        last_reason = reason
        if reason == "llm_unavailable":
            print("  [strategy] LLM unavailable — falling back to heuristic")
            break
        if attempt >= max_retries:
            print(f"  [strategy] LLM attempt {attempt} rejected ({reason}) — exhausted retries")
            break
        if stats is not None:
            stats.retries += 1
        feedback = reason
        print(f"  [strategy] LLM attempt {attempt} rejected ({reason}) — retrying with feedback")

    if stats is not None:
        stats.fallbacks += 1
    if last_reason and last_reason != "llm_unavailable":
        print(f"  [strategy] LLM final reason: {last_reason}")
    print(f"  [strategy] Heuristic fallback: {heuristic_action.short()}")
    return heuristic_action, snapshot
