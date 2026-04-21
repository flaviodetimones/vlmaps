"""
vlmaps.policy
=============
Phase F — high-level action vocabulary for room-aware object search.

This package introduces an explicit action layer that the rest of the
pipeline (interactive_object_nav.py, future LLM strategist in Phase G)
can produce and consume as discrete decisions instead of inline calls.

Nothing in this package replaces existing pipeline logic: it provides
typed action records, a JSON schema for LLM exchange, and helpers
(frontier picking) that the executor can dispatch to existing functions.
"""

from vlmaps.policy.actions import (
    ActionType,
    Action,
    parse_action_json,
    action_to_json,
    ACTION_SCHEMA_PROMPT,
)
from vlmaps.policy.frontier import find_frontier_in_room
from vlmaps.policy.executor import (
    ExecutorContext,
    ActionResult,
    execute_action,
    sync_pose_state,
    close_executor_context,
)

__all__ = [
    "ActionType",
    "Action",
    "parse_action_json",
    "action_to_json",
    "ACTION_SCHEMA_PROMPT",
    "find_frontier_in_room",
    "ExecutorContext",
    "ActionResult",
    "execute_action",
    "sync_pose_state",
    "close_executor_context",
]
