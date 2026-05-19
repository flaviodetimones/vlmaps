"""
executor.py
===========
Phase F/G bridge: execute typed Actions on top of the existing interactive
navigation primitives without rewriting the full baseline navigator.

This module intentionally reuses the working low-level helpers from
`application.interactive_object_nav`:
  - room staging / room-goal navigation
  - candidate approach planning
  - path replay follower
  - arrival YOLOE checks
  - 360 verification

The goal is to change orchestration, not the underlying robot primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vlmaps.policy.actions import Action, ActionType
from vlmaps.policy.frontier import find_frontier_in_room


def _base_nav():
    """Lazy import to avoid circular imports at module load time."""
    from application import interactive_object_nav as base_nav

    return base_nav


@dataclass
class ExecutorContext:
    """Mutable execution context shared across a target-search episode."""

    robot: Any
    rgb_map_2d: np.ndarray
    target: str
    original_target: Optional[str] = None
    effective_target: Optional[str] = None
    surrogate_categories: List[str] = field(default_factory=list)
    likely_rooms: List[str] = field(default_factory=list)
    resolution_source: str = "fallback"
    room_provider: Any = None
    search_state: Any = None
    heatmap: Optional[np.ndarray] = None
    kept_components: List[Dict[str, Any]] = field(default_factory=list)
    current_room: Optional[str] = None
    yoloe_session: Any = None
    last_candidate: Optional[Dict[str, Any]] = None
    last_candidate_centroid: Optional[Tuple[int, int]] = None
    last_path_polyline: List[List[int]] = field(default_factory=list)
    last_path_cells: List[List[int]] = field(default_factory=list)
    last_goal_pos: Optional[List[int]] = None


@dataclass
class ActionResult:
    success: bool
    found: bool = False
    done: bool = False
    actual_room: Optional[str] = None
    message: str = ""


def _ensure_yoloe_session(ctx: ExecutorContext):
    if ctx.yoloe_session is None:
        from vlmaps.utils.yoloe_utils import get_session, runtime_conf_thresh

        ctx.yoloe_session = get_session(ctx.target, conf_thresh=runtime_conf_thresh(0.65))
    return ctx.yoloe_session


def close_executor_context(ctx: ExecutorContext) -> None:
    """Release the cached YOLOE session if one was created."""
    if ctx.yoloe_session is None:
        return
    from vlmaps.utils.yoloe_utils import shutdown_session

    shutdown_session()
    ctx.yoloe_session = None


def sync_pose_state(ctx: ExecutorContext) -> Optional[str]:
    """Refresh robot pose, current room and visited-cell trail."""
    ctx.robot._set_nav_curr_pose()
    current_room = None
    if ctx.room_provider and ctx.room_provider.is_available():
        current_room = ctx.room_provider.get_room_at_cell(
            int(ctx.robot.curr_pos_on_map[0]),
            int(ctx.robot.curr_pos_on_map[1]),
        )
    ctx.current_room = current_room
    if ctx.search_state is not None:
        ctx.search_state.update_current_room(current_room)
        ctx.search_state.record_visited_cell(
            int(ctx.robot.curr_pos_on_map[0]),
            int(ctx.robot.curr_pos_on_map[1]),
        )
    return current_room


def _record_action(ctx: ExecutorContext, action: Action, outcome: str) -> None:
    if ctx.search_state is not None:
        ctx.search_state.record_action(action.short(), outcome=outcome)


def _lookup_component(ctx: ExecutorContext, centroid: Tuple[int, int]) -> Dict[str, Any]:
    cr, cc = int(centroid[0]), int(centroid[1])
    for comp in ctx.kept_components:
        ccr, ccc = int(comp["centroid"][0]), int(comp["centroid"][1])
        if ccr == cr and ccc == cc:
            return comp
    return {
        "centroid": [cr, cc],
        "room": ctx.current_room,
        "quality": 0.0,
        "score": 0.0,
        "area": 0,
    }


def _apply_room_gate(
    ctx: ExecutorContext,
    component: Optional[Dict[str, Any]],
    obj_centroid: Tuple[int, int],
    confirmed: bool,
) -> bool:
    """Reject detections seen from the wrong room if the target is still far away."""
    if not confirmed or component is None:
        return confirmed

    comp_room = component.get("room")
    if not comp_room or not ctx.room_provider or not ctx.room_provider.is_available():
        return confirmed

    actual_room = sync_pose_state(ctx)
    if actual_room == comp_room:
        return confirmed

    dr = float(obj_centroid[0]) - float(ctx.robot.curr_pos_on_map[0])
    dc = float(obj_centroid[1]) - float(ctx.robot.curr_pos_on_map[1])
    dist = float(np.hypot(dr, dc))
    if dist > 20.0:
        print(
            f"  [room-gate] Object visible but room not yet entered "
            f"(robot: {actual_room}, target: {comp_room}, dist={dist:.1f} cells) "
            f"→ tentative only"
        )
        return False

    print(f"  [room-gate] Room entry confirmed (dist={dist:.1f} <= 20 cells) — accepting")
    return confirmed


def _check_current_view(
    ctx: ExecutorContext,
    component: Optional[Dict[str, Any]],
    obj_centroid: Tuple[int, int],
    *,
    label: str,
) -> bool:
    """Run a blocking YOLOE check on the current RGB frame."""
    base = _base_nav()
    session = _ensure_yoloe_session(ctx)
    if session is None:
        print("  (YOLOE not available — skipping visual verification)")
        return False

    try:
        obs = ctx.robot.sim.get_sensor_observations(0)
        if "color_sensor" not in obs:
            return False
        frame = obs["color_sensor"][:, :, :3]
        found, ann_rgb, bbox = session.check(frame)
        if ann_rgb is not None:
            ann_bgr = base.cv2.cvtColor(ann_rgb, base.cv2.COLOR_RGB2BGR)
            base.show_obs(ctx.robot, label, yoloe_frame_bgr=ann_bgr)
        if not found:
            print(f"  YOLOE {label}: ✗ '{ctx.target}' not detected.")
            return False
        found = _apply_room_gate(ctx, component, obj_centroid, found)
        if found:
            print(f"  YOLOE {label}: ✓ Found '{ctx.target}'! (bbox center: {bbox})")
            base.freeze_found_target(ctx.target, ann_rgb, obj_centroid)
        return found
    except Exception as exc:
        print(f"  YOLOE {label} error: {exc}")
        return False


def _mark_candidate_result(
    ctx: ExecutorContext,
    confirmed: bool,
    source: Optional[str] = None,
) -> Optional[str]:
    end_room = sync_pose_state(ctx)
    if ctx.search_state is not None and ctx.last_candidate_centroid is not None:
        ctx.search_state.record_candidate(
            end_room,
            confirmed,
            centroid=ctx.last_candidate_centroid,
        )
        if confirmed:
            ctx.search_state.record_object_seen(end_room, ctx.target)
            ctx.search_state.mark_found(end_room)
            if source and hasattr(ctx.search_state, "mark_confirmation"):
                ctx.search_state.mark_confirmation(source)
    return end_room


def _execute_go_to_room(ctx: ExecutorContext, action: Action) -> ActionResult:
    base = _base_nav()
    safe_map = getattr(ctx.robot, "_safe_obs_map", ctx.robot.map.obstacles_map)
    stage_ok, actual_room = base.navigate_to_room_stage(
        ctx.robot,
        action.room,
        ctx.room_provider,
        ctx.rgb_map_2d,
        safe_map,
        label_prefix="Executor room stage",
    )
    room_ok = base._room_instance_matches(actual_room, action.room)
    outcome = "arrived" if room_ok else "not_reached"
    _record_action(ctx, action, outcome)
    return ActionResult(
        success=bool(stage_ok and room_ok),
        actual_room=actual_room,
        message=outcome,
    )


def _execute_explore_room(ctx: ExecutorContext, action: Action) -> ActionResult:
    base = _base_nav()
    frontier = find_frontier_in_room(
        action.room,
        ctx.room_provider,
        ctx.robot.map.obstacles_map,
        visited_cells=(ctx.search_state.visited_cells if ctx.search_state else None),
    )
    if frontier is None:
        _record_action(ctx, action, "no_frontier")
        return ActionResult(False, message="no_frontier")

    print(f"  [executor] explore_room frontier for '{action.room}': {frontier}")
    _, planned_actions = ctx.robot.plan_path_only(list(frontier))
    path_polyline = base.normalize_path_cells(getattr(ctx.robot, "last_planned_path", None) or [])
    path_cells = base.densify_path_cells(path_polyline)
    completed = True
    if planned_actions:
        completed = base.execute_nav_replay(
            ctx.robot,
            planned_actions,
            action.room,
            ctx.rgb_map_2d,
            ctx.heatmap,
            path_polyline,
            display_path_cells=path_cells,
            dist_map=base.distance_transform_edt(ctx.robot.map.obstacles_map),
            goal_reached_tol_cells=1.0,
        )
    actual_room = sync_pose_state(ctx)
    room_ok = base._room_instance_matches(actual_room, action.room)
    outcome = "frontier_reached" if completed and room_ok else "frontier_failed"
    _record_action(ctx, action, outcome)
    return ActionResult(
        success=bool(completed and room_ok),
        actual_room=actual_room,
        message=outcome,
    )


def _plan_candidate_approach(ctx: ExecutorContext, component: Dict[str, Any]):
    base = _base_nav()
    obj_centroid = [
        int(component["centroid"][0]),
        int(component["centroid"][1]),
    ]
    safe_map = getattr(ctx.robot, "_safe_obs_map", ctx.robot.map.obstacles_map)
    ctx.robot._set_nav_curr_pose()
    robot_pos = getattr(ctx.robot, "_nav_curr_pos", None)
    approach_goals = base.find_safe_candidate_approach_goals(
        obj_centroid,
        safe_map,
        robot_pos=robot_pos,
        room_provider=ctx.room_provider,
        required_room=component.get("room"),
        min_dist=8.0,
        max_dist=25.0,
        min_clearance=3.0,
    )
    if approach_goals:
        goal_pos = approach_goals[0]
        print(f"  [executor] Approach goal: {goal_pos} ({len(approach_goals)} safe candidates)")
    else:
        print("  [executor] No annulus goals found — falling back to path walk")
        initial_path, _ = ctx.robot.plan_path_only(obj_centroid)
        goal_pos, obj_centroid = base.select_safe_goal_from_path(
            initial_path,
            ctx.heatmap,
            ctx.robot.map.obstacles_map,
            room_provider=ctx.room_provider,
            required_room=component.get("room"),
        )
    _, planned_actions = ctx.robot.plan_path_only(goal_pos)
    path_polyline = base.normalize_path_cells(getattr(ctx.robot, "last_planned_path", None) or [])
    path_cells = base.densify_path_cells(path_polyline)
    return obj_centroid, goal_pos, planned_actions, path_polyline, path_cells


def _execute_inspect_candidate(ctx: ExecutorContext, action: Action) -> ActionResult:
    base = _base_nav()
    component = _lookup_component(ctx, action.candidate_centroid)
    component["room"] = component.get("room") or action.room
    obj_centroid, goal_pos, planned_actions, path_polyline, path_cells = _plan_candidate_approach(ctx, component)

    ctx.last_candidate = component
    ctx.last_candidate_centroid = (int(obj_centroid[0]), int(obj_centroid[1]))
    ctx.last_goal_pos = [int(goal_pos[0]), int(goal_pos[1])]
    ctx.last_path_polyline = path_polyline
    ctx.last_path_cells = path_cells

    n_actions = len(planned_actions)
    print(
        f"  [executor] Candidate path: {len(path_polyline)} waypoint(s), "
        f"{len(path_cells)} dense cell(s), preview={n_actions} actions"
    )
    base.show_map(
        ctx.robot,
        ctx.rgb_map_2d,
        heatmap_2d=ctx.heatmap,
        path_cells=path_cells,
        label=f"Executor plan: {ctx.target}",
    )
    base.cv2.waitKey(400)

    completed = True
    if n_actions > 0:
        completed = base.execute_nav_replay(
            ctx.robot,
            planned_actions,
            ctx.target,
            ctx.rgb_map_2d,
            ctx.heatmap,
            path_polyline,
            display_path_cells=path_cells,
            dist_map=base.distance_transform_edt(ctx.robot.map.obstacles_map),
        )
        if not completed:
            print(f"  [executor] Path execution stopped early for '{ctx.target}'.")

    arrival_label = f"Executor arrival: {ctx.target}" if completed else f"Executor stop: {ctx.target}"
    base.show_obs(ctx.robot, arrival_label)
    base.show_map(
        ctx.robot,
        ctx.rgb_map_2d,
        heatmap_2d=ctx.heatmap,
        path_cells=path_cells,
        label=arrival_label,
    )

    found_source: Optional[str] = None
    found = _check_current_view(
        ctx,
        component,
        tuple(ctx.last_candidate_centroid),
        label=f"YOLOE arrival: {ctx.target}",
    )
    if found:
        print(f"  [verify] source=arrival")
        found_source = "arrival"
    else:
        print(f"  [executor] Turning to face '{ctx.target}'…")
        base.face_toward_pos(ctx.robot, obj_centroid[0], obj_centroid[1])
        base.show_obs(ctx.robot, f"Executor facing: {ctx.target}")
        base.show_map(
            ctx.robot,
            ctx.rgb_map_2d,
            heatmap_2d=ctx.heatmap,
            path_cells=path_cells,
            label=f"Executor facing: {ctx.target}",
        )
        found = _check_current_view(
            ctx,
            component,
            tuple(ctx.last_candidate_centroid),
            label=f"YOLOE facing: {ctx.target}",
        )
        if found:
            print(f"  [verify] source=turn_to_face")
            found_source = "turn_to_face"

    if found and _ensure_yoloe_session(ctx) is not None:
        base.fine_visual_center(ctx.robot, ctx.yoloe_session, ctx.target)
        end_room = _mark_candidate_result(ctx, True, source=found_source)
        base.show_obs(ctx.robot, f"FOUND: {ctx.target}")
        base.show_map(
            ctx.robot,
            ctx.rgb_map_2d,
            heatmap_2d=ctx.heatmap,
            path_cells=path_cells,
            label=f"FOUND: {ctx.target}",
            target_cell=list(ctx.last_candidate_centroid),
        )
        _record_action(ctx, action, "found")
        return ActionResult(True, found=True, actual_room=end_room, message="found")

    actual_room = sync_pose_state(ctx)
    _record_action(ctx, action, "inspect_no_detection")
    return ActionResult(completed, found=False, actual_room=actual_room, message="inspect_no_detection")


def _execute_verify_target(ctx: ExecutorContext, action: Action) -> ActionResult:
    base = _base_nav()
    if ctx.last_candidate_centroid is None:
        _record_action(ctx, action, "no_candidate")
        return ActionResult(False, message="no_candidate")

    print(f"  [executor] Starting local ±25° scan for '{ctx.target}'…")
    surrogate_cat = ctx.effective_target if ctx.effective_target != ctx.target else ""
    confirmed = base.scan_local_and_verify(
        ctx.robot,
        ctx.target,
        ctx.rgb_map_2d,
        ctx.heatmap,
        ctx.last_path_cells,
        surrogate_cat=surrogate_cat or "",
    )
    if confirmed:
        verify_source = getattr(ctx.robot, "_last_verify_source", None) or "local_scan"
        if verify_source != "pitch_scan":
            print(f"  [verify] source=local_scan")
        if _ensure_yoloe_session(ctx) is not None:
            base.fine_visual_center(ctx.robot, ctx.yoloe_session, ctx.target)
            base.freeze_found_target(ctx.target, None, ctx.last_candidate_centroid)

    end_room = _mark_candidate_result(
        ctx, confirmed, source=(verify_source if confirmed else None)
    )
    outcome = "found" if confirmed else "not_found"
    _record_action(ctx, action, outcome)

    if confirmed:
        base.show_obs(ctx.robot, f"FOUND: {ctx.target}")
        base.show_map(
            ctx.robot,
            ctx.rgb_map_2d,
            heatmap_2d=ctx.heatmap,
            path_cells=ctx.last_path_cells,
            label=f"FOUND: {ctx.target}",
            target_cell=list(ctx.last_candidate_centroid),
        )
    return ActionResult(True, found=confirmed, actual_room=end_room, message=outcome)


def _execute_done(ctx: ExecutorContext, action: Action) -> ActionResult:
    _record_action(ctx, action, action.reason or "done")
    return ActionResult(True, done=True, actual_room=sync_pose_state(ctx), message=action.reason or "done")


def execute_action(ctx: ExecutorContext, action: Action) -> ActionResult:
    """Dispatch one typed Action onto the existing runtime primitives."""
    if action.type is ActionType.GO_TO_ROOM:
        return _execute_go_to_room(ctx, action)
    if action.type is ActionType.EXPLORE_ROOM:
        return _execute_explore_room(ctx, action)
    if action.type is ActionType.INSPECT_CANDIDATE:
        return _execute_inspect_candidate(ctx, action)
    if action.type is ActionType.VERIFY_TARGET:
        return _execute_verify_target(ctx, action)
    if action.type is ActionType.DONE:
        return _execute_done(ctx, action)
    raise ValueError(f"Unsupported action type: {action.type}")
