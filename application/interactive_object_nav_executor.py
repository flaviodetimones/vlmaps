"""
interactive_object_nav_executor.py
==================================
Alternate interactive navigator that routes decisions through typed Actions.

This keeps `interactive_object_nav.py` as the stable baseline while exercising
the new Phase F executor path:

    decision -> Action -> execute_action(...)

The low-level navigation, approach planning and YOLOE verification are still
the same proven primitives from the baseline script.
"""

from __future__ import annotations

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig

from application import interactive_object_nav as base
from vlmaps.policy import Action, ActionType, ExecutorContext, execute_action, close_executor_context, sync_pose_state
from vlmaps.robot.habitat_lang_robot import HabitatLanguageRobot
from vlmaps.utils.llm_utils import parse_object_goal_instruction
from vlmaps.utils.matterport3d_categories import get_categories
from vlmaps.utils.room_map_utils import find_room_goal, load_room_map
from vlmaps.utils.search_state import SearchState


def _build_present_categories(robot) -> list:
    structural = {"void", "wall", "floor", "ceiling"}
    score_thresh = 0.3
    min_voxels = 10
    cats = getattr(robot.map, "categories", [])
    scores_mat = getattr(robot.map, "scores_mat", None)
    if scores_mat is None:
        return [c for c in cats if c not in structural]
    max_ids = np.argmax(scores_mat, axis=1)
    present = []
    for i, c in enumerate(cats):
        if c in structural:
            continue
        mask = (max_ids == i) & (scores_mat[:, i] > score_thresh)
        if int(mask.sum()) >= min_voxels:
            present.append(c)
    return present


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


def _fuse_heatmap_evidence(cat: str, search_state: SearchState, kept_components: list) -> dict:
    from vlmaps.utils.room_priors import (
        compute_heatmap_room_evidence,
        compute_room_priors,
    )

    if search_state is None or not kept_components:
        return {}
    heatmap_ev = compute_heatmap_room_evidence(kept_components)
    print("  Query type: direct")
    if heatmap_ev:
        ev_sorted = sorted(heatmap_ev.items(), key=lambda x: -x[1])
        print(f"  Room evidence from heatmap: "
              f"{ {r: round(v, 3) for r, v in ev_sorted} }")

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


def _plan_actions_for_object(
    ctx: ExecutorContext,
    categories: list,
    present_categories: list,
) -> list:
    cat = ctx.target
    search_state = ctx.search_state
    room_provider = ctx.room_provider

    print(f"  Current room: {ctx.current_room or 'unknown'}")
    print("  Computing semantic heatmap...")
    heatmap, kept_components = base.compute_heatmap(ctx.robot, cat)
    ctx.heatmap = heatmap
    ctx.kept_components = kept_components

    direct_query_mode = len(categories) == 1 and cat.lower() in {c.lower() for c in present_categories}

    _annotate_candidate_rooms(room_provider, kept_components)

    heatmap_ev = {}
    selected_room = None
    if search_state and kept_components:
        heatmap_ev = _fuse_heatmap_evidence(cat, search_state, kept_components)

    if search_state and kept_components and search_state.rooms and not direct_query_mode:
        robot_rc = [ctx.robot.curr_pos_on_map[0], ctx.robot.curr_pos_on_map[1]]
        selected_room, _room_scores = base.select_best_room(
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
                    f"  [executor-policy] Chosen room for '{cat}': {selected_room} "
                    f"({len(room_kept)}/{len(kept_components)} candidate(s))"
                )
                kept_components = room_kept
                ctx.kept_components = room_kept
            else:
                print(
                    f"  [executor-policy] Chosen room '{selected_room}' has no direct "
                    f"candidates — falling back to global ranking"
                )
    elif direct_query_mode:
        print(
            f"  [executor-policy] Direct furniture query for '{cat}' "
            f"— room selector disabled, nearest candidate policy active"
        )

    base.show_map(ctx.robot, ctx.rgb_map_2d, heatmap_2d=heatmap, label=f"Executor planning: {cat}")
    cv2.waitKey(200)

    if not kept_components:
        print(f"  [skip] No heatmap signal for '{cat}' in this scene.")
        return []

    tried = getattr(search_state, "_tried_centroids", set()) if search_state else set()
    query_priors = {r: rs.target_relevance for r, rs in search_state.rooms.items()} if search_state else {}
    robot_rc_sel = [ctx.robot.curr_pos_on_map[0], ctx.robot.curr_pos_on_map[1]]
    best_comp = base.select_best_candidate(
        kept_components,
        ctx.current_room,
        query_priors,
        tried,
        robot_pos=robot_rc_sel,
        enable_room_gate=not direct_query_mode,
        prefer_nearest_only=direct_query_mode,
    )
    if best_comp is None:
        print(f"  [skip] No viable candidate for '{cat}'.")
        return []

    ctx.last_candidate = best_comp
    comp_room = best_comp.get("room") or selected_room or ctx.current_room or "unknown"
    centroid = (int(best_comp["centroid"][0]), int(best_comp["centroid"][1]))
    print(f"  [executor-policy] Best component: {centroid} (room: {comp_room})")

    actions = []
    if selected_room and ctx.current_room != selected_room:
        actions.append(Action(type=ActionType.GO_TO_ROOM, room=selected_room))
    actions.append(
        Action(
            type=ActionType.INSPECT_CANDIDATE,
            room=comp_room,
            candidate_centroid=centroid,
        )
    )
    actions.append(Action(type=ActionType.VERIFY_TARGET))
    return actions


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="object_goal_navigation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    robot = HabitatLanguageRobot(config)
    robot.setup_scene(config.scene_id)
    dataset_type = str(getattr(config, "dataset_type", "mp3d"))
    scene_categories = get_categories(dataset_type)
    robot.map.init_categories(scene_categories)

    print("\nBuilding top-down RGB map...")
    rgb_map_2d = base.build_rgb_map_2d(robot)

    scene_dir = robot.vlmaps_data_save_dirs[config.scene_id]
    room_data = load_room_map(scene_dir)
    if room_data is not None:
        room_map, room_categories, room_regions = room_data
        print(f"Room map loaded: {list(room_regions.keys())}")
    else:
        room_map, room_categories, room_regions = None, [], {}
        print("No room map found. Run build_room_map.py to enable region-aware navigation.")

    print("\nSearching for a good starting position...")
    start_tf = base.find_best_start_pose(robot)
    robot.set_agent_state(start_tf)
    robot._set_nav_curr_pose()

    base.show_obs(robot, "Ready")
    base.show_map(robot, rgb_map_2d, heatmap_2d=None, label="Ready")
    print("Scene:", robot.vlmaps_data_save_dirs[config.scene_id].name)

    room_provider = getattr(robot, "room_provider", None)
    if room_provider and room_provider.is_available():
        print(f"Room provider active. Rooms: {room_provider.list_rooms()}")

    present_categories = _build_present_categories(robot)

    while True:
        rooms_hint = ""
        if room_provider and room_provider.is_available():
            rooms_hint = f"  Rooms     : {room_provider.list_rooms()}\n"
        print(
            f"\n{'─' * 50}\n"
            f"{rooms_hint}"
            f"  Objects   : {present_categories}"
        )
        instruction = input("Enter navigation instruction (or 'quit'): ").strip()
        if instruction.lower() in ("quit", "exit", "q"):
            break
        if not instruction:
            continue

        print("Parsing instruction...")
        try:
            categories = parse_object_goal_instruction(instruction)
        except Exception as exc:
            print(f"LLM error: {exc}")
            continue

        categories = base._resolve_instruction_room_targets(
            instruction, categories, room_provider
        )

        print(f"Targets: {categories}")

        robot.set_agent_state(start_tf)
        robot._set_nav_curr_pose()
        robot.empty_recorded_actions()
        base.show_obs(robot, "Start")
        base.show_map(robot, rgb_map_2d, label="Start")

        search_states = {}
        for cat in categories:
            target = cat.strip()
            ss = SearchState(target, room_provider, robot.map.obstacles_map)
            if ss.rooms:
                priors = ss.compute_priors()
                if priors:
                    sorted_priors = sorted(priors.items(), key=lambda x: x[1], reverse=True)
                    prior_str = ", ".join(f"{r}={v:.2f}" for r, v in sorted_priors if v > 0.01)
                    print(f"  Room priors for '{target}': {prior_str}")
            search_states[target] = ss

        for cat in categories:
            target = cat.strip()
            if not target:
                continue

            base._frozen_detection_bgr = None
            base._frozen_target_cell = None

            ctx = ExecutorContext(
                robot=robot,
                rgb_map_2d=rgb_map_2d,
                target=target,
                room_provider=room_provider,
                search_state=search_states.get(target),
            )
            sync_pose_state(ctx)

            print(f"\nPlanning actions for: {target}")

            room_goal = None
            if room_provider and room_provider.is_available():
                room_goal = room_provider.get_room_centroid(target)
            if room_goal is None and room_regions:
                room_goal = find_room_goal(target, room_regions)

            if room_goal is not None:
                print(f"  [executor-policy] Room command detected for '{target}'")
                actions = [
                    Action(type=ActionType.GO_TO_ROOM, room=target),
                    Action(type=ActionType.DONE, reason="room command handled"),
                ]
            else:
                actions = _plan_actions_for_object(ctx, categories, present_categories)

            if not actions:
                close_executor_context(ctx)
                continue

            print("  [executor-policy] Planned actions:")
            for idx, action in enumerate(actions, start=1):
                print(f"    {idx}. {action.short()}")

            found = False
            for action in actions:
                result = execute_action(ctx, action)
                if result.found:
                    found = True
                    execute_action(
                        ctx,
                        Action(type=ActionType.DONE, reason="target found"),
                    )
                    break
                if result.done:
                    break
                if action.type is ActionType.GO_TO_ROOM and room_goal is not None and not result.success:
                    print(f"  [executor] Room navigation failed for '{target}'.")
                    break

            if not found and room_goal is None and ctx.last_candidate_centroid is None:
                execute_action(
                    ctx,
                    Action(type=ActionType.DONE, reason="no viable candidate"),
                )

            from vlmaps.utils.habitat_utils import agent_state2tf

            agent_state = robot.sim.get_agent(0).get_state()
            start_tf = agent_state2tf(agent_state)
            close_executor_context(ctx)

        for target, ss in search_states.items():
            if ss.rooms:
                print(f"\n{ss.summary()}")

        print("\nInstruction complete.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
