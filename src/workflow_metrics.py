"""State-based metrics with distinct response, attempted-call, and dispatch costs.

Supplied checkpoint/injection history is never counted as a model event.
"""

from __future__ import annotations

from typing import Any

from .execution_environment import completion_condition_matches, workflow_stage
from .tasks import Task

METRICS_VERSION = "workflow-metrics-v5"
LEGACY_METRICS_VERSION = "workflow-metrics-v2"
LEGACY_AGENT_POLICY = "legacy-unversioned"
MODEL_ERRORS = {"model_output_error", "invalid_action", "invalid_arguments"}
# Tool rejections for an identifier that no earlier result issued (e.g. a
# mistyped offer_id). Choosing an existing but wrong resource is instead a
# selection error, recorded as saved_result_violates_request.
IDENTIFIER_ERRORS = {"unknown_request", "context_not_found", "offer_not_found", "detail_not_found",
                     "unknown_option", "draft_not_found"}


def _executed(event: dict[str, Any]) -> bool:
    if "tool_executed" in event:
        return event["tool_executed"] is True
    # A malformed request is not a dispatch. Older traces have tool results but
    # no explicit dispatch flag, so only those results provide execution evidence.
    return "tool_result" in event


def _fatal_event(event: dict[str, Any], trace: dict[str, Any]) -> bool:
    return (event.get("error") in {"infrastructure_error", "internal_error"}
            or bool(trace.get("runtime_error"))
            and event.get("error") == trace.get("runtime_error"))


def _behavior_error(event: dict[str, Any]) -> bool:
    return bool(event.get("tool_error") or event.get("semantic_error")
                or event.get("model_error") or event.get("error") in MODEL_ERRORS)


def _error_kind(event: dict[str, Any]) -> str | None:
    """One label per erroneous response: model output, tool rejection, or semantic."""
    if event.get("model_error"):
        return f"model_output:{event['model_error']}"
    if event.get("error") in MODEL_ERRORS:
        return f"model_output:{event['error']}"
    if event.get("tool_error"):
        return f"tool_error:{event['tool_error']}"
    if event.get("semantic_error"):
        return f"semantic:{event['semantic_error']}"
    return None


def _same_call(event: dict[str, Any], expected: dict[str, Any] | None) -> bool:
    action = event.get("action")
    return (expected is not None and isinstance(action, dict) and action.get("type") == "tool_call"
            and action.get("name") == expected.get("tool")
            and (action.get("arguments") or {}) == (expected.get("arguments") or {}))


def _stable_false_turn(events: list[dict[str, Any]], name: str, final_absent: bool) -> int | None:
    """First turn of the final absent suffix, not just a temporary repair."""
    if not final_absent:
        return None
    observed = [event for event in events if isinstance(event.get(name), bool)]
    suffix = []
    for event in reversed(observed):
        if event[name]:
            break
        suffix.append(event["step"])
    return min(suffix, default=None)


def score_workflow_trace(task: Task, trace: dict[str, Any]) -> dict[str, Any]:
    condition = trace["condition"]
    recovery = condition in {"error_restart", "recovery"}
    events = trace.get("events", [])
    response_events = [event for event in events
                       if event.get("response_received", not _fatal_event(event, trace))]
    tool_events = [event for event in events if _executed(event)]
    invalid = [event for event in response_events if event.get("error") in {"invalid_action", "invalid_arguments"}]
    model_errors = [event for event in response_events
                    if event.get("model_error") or event.get("error") in MODEL_ERRORS]
    tool_errors = [event for event in response_events if event.get("tool_error")]
    semantic = [event for event in response_events if event.get("semantic_error")]
    identifier_errors = [event for event in tool_errors if event.get("tool_error") in IDENTIFIER_ERRORS]
    errors = [event for event in response_events if _behavior_error(event)]
    tool_error_actions = [event for event in tool_events if _behavior_error(event)]
    first_error_event = min(errors, key=lambda event: event["step"], default=None)
    first_error = first_error_event["step"] if first_error_event else None
    executed_indexes = {id(event): index for index, event in enumerate(tool_events, 1)}
    first_tool_index = (first_error_event.get("tool_call_index", executed_indexes.get(id(first_error_event)))
                        if first_error_event and _executed(first_error_event) else None)
    completed = [event["step"] for event in response_events if event.get("success_after_action") is True]
    state_match = completion_condition_matches(trace.get("final_state", {}), task.success_condition)
    infrastructure_count = int(trace.get("infrastructure_error") is not None)
    internal_count = int(trace.get("internal_error") is not None)
    unclassified_runtime_count = int(trace.get("runtime_error") is not None
                                     and not infrastructure_count and not internal_count)
    valid = not (infrastructure_count or internal_count or unclassified_runtime_count)
    success = valid and state_match
    paired = task.schema_version == "recovery-v2"
    normal_calls = task.normal_remaining_calls if paired else task.horizon - (
        task.checkpoint_step if condition == "normal_restart" else 0)
    oracle_calls = task.recovery_min_calls if recovery else normal_calls
    wrong_results = [item["result"]["saved_result"] for item in trace.get("injected_actions", [])
                     if "saved_result" in item.get("result", {})]
    saved = trace.get("final_state", {}).get("completed_requests", {})
    error_resolved = (all(saved.get(wrong["request_id"]) != wrong for wrong in wrong_results)
                      if recovery and wrong_results else None)
    # Next action: the first model response at the shared starting state.
    if recovery:
        oracle = task.recovery_actions or []
    elif condition == "normal_restart" and not paired:
        oracle = task.gold_actions[task.checkpoint_step:]
    else:
        oracle = task.gold_actions
    first_response = response_events[0] if response_events else None
    next_action_correct = _same_call(first_response, oracle[0] if oracle else None) if first_response else (
        False if valid else None)
    next_action_valid = (bool(first_response and _executed(first_response) and not _behavior_error(first_response))
                         if valid else None)
    first_error_action = (first_error_event or {}).get("action")
    first_error_tool = first_error_action.get("name") if isinstance(first_error_action, dict) else None
    wrong_saves = [event for event in response_events if event.get("semantic_error") == "saved_result_violates_request"]
    natural_error = bool(errors) if condition == "free" and valid else None
    resolution_steps = [event["step"] for event in response_events
                        if event.get("injected_error_present_after") is False]
    attempts = sum(event.get("attempted_tool_call_count", int(
        isinstance(event.get("action"), dict) and event["action"].get("type") == "tool_call"))
        for event in events)
    response_count, executed_count = len(response_events), len(tool_events)
    return {
        "task_id": task.task_id, "base_id": task.base_id, "task_type": task.task_type,
        "task_schema_version": task.schema_version, "protocol_version": trace.get("protocol_version", task.schema_version),
        "metrics_version": trace.get("metrics_version", LEGACY_METRICS_VERSION),
        "agent_policy_version": trace.get("agent_policy_version", LEGACY_AGENT_POLICY),
        "pair_id": task.base_id if paired else None, "fixture_sha256": trace.get("fixture_sha256"),
        "checkpoint_state_hash": trace.get("checkpoint_state_hash"),
        "resume_workflow_step": trace.get("resume_workflow_step"),
        "horizon": task.horizon, "condition": condition,
        "normal_remaining_calls": normal_calls, "recovery_min_calls": task.recovery_min_calls,
        "strict_success": success, "final_state_match": state_match,
        "valid_for_behavior_metrics": valid,
        "infrastructure_error_count": infrastructure_count, "internal_error_count": internal_count,
        "unclassified_runtime_error_count": unclassified_runtime_count,
        "termination_reason": trace.get("termination_reason"),
        "budget_exhausted": trace.get("termination_reason") == "budget_exhausted",
        "natural_error": natural_error,
        "natural_error_recovered": (natural_error and success) if natural_error is not None else None,
        "wrong_save_count": len(wrong_saves),
        "identifier_error_count": len(identifier_errors),
        "identifier_error_episode": bool(identifier_errors) if valid else None,
        "wrong_save_episode": bool(wrong_saves) if valid else None,
        "wrong_save_recovered": (bool(wrong_saves) and success) if valid else None,
        "error_count": len(errors), "model_output_error_count": len(model_errors),
        "output_truncated_count": sum(event.get("model_error") == "output_truncated" for event in response_events),
        "multiple_tool_call_response_count": sum(event.get("model_error") == "multiple_tool_calls" for event in response_events),
        "error_turn_rate": len(errors) / response_count if response_count else None,
        "tool_error_action_count": len(tool_error_actions),
        "tool_action_error_rate": len(tool_error_actions) / executed_count if executed_count else None,
        "first_error_step": first_error, "first_error_turn": first_error,
        "first_error_tool_call_index": first_tool_index,
        "first_error_kind": _error_kind(first_error_event) if first_error_event else None,
        "first_error_tool": first_error_tool,
        # Absolute 1..5 workflow stage of the first erroneous tool call, comparable across horizons.
        "first_error_workflow_stage": workflow_stage(task.domain, first_error_tool),
        "resume_workflow_step": trace.get("resume_workflow_step"),
        "next_action_correct": next_action_correct,
        "next_action_valid": next_action_valid,
        "recovery_success": success if recovery else None,
        "recovery_error_resolved": error_resolved,
        "recovery_resolution_step": min(resolution_steps, default=None) if recovery else None,
        "recovery_first_resolution_turn": min(resolution_steps, default=None) if recovery else None,
        "recovery_stable_resolution_turn": _stable_false_turn(
            response_events, "injected_error_present_after", error_resolved is True) if recovery else None,
        "response_turn_count": response_count, "attempted_tool_call_count": attempts,
        "executed_tool_call_count": executed_count, "tool_call_count": executed_count,
        "extra_calls": max(0, executed_count - normal_calls) if recovery else 0,
        "extra_calls_over_normal": max(0, executed_count - normal_calls),
        "extra_calls_over_oracle": max(0, executed_count - oracle_calls),
        "wrong_tool_count": sum(event.get("tool_error") == "unknown_tool" for event in tool_errors),
        "wrong_argument_count": sum(event.get("tool_error") != "unknown_tool" for event in tool_errors) + len(invalid),
        "tool_error_count": len(tool_errors), "semantic_error_count": len(semantic),
        "recovery_error_count": len(errors) if recovery else None,
        "recovery_model_output_error_count": len(model_errors) if recovery else None,
        "recovery_tool_error_count": len(tool_errors) if recovery else None,
        "recovery_invalid_action_count": len(invalid) if recovery else None,
        "recovery_semantic_error_count": len(semantic) if recovery else None,
        "recovery_runtime_error_count": int(not valid) if recovery else None,
        "recovery_infrastructure_error_count": infrastructure_count if recovery else None,
        "recovery_internal_error_count": internal_count if recovery else None,
        "recovery_first_error_step": first_error if recovery else None,
        "recovery_first_error_turn": first_error if recovery else None,
        "recovery_first_error_tool_call_index": first_tool_index if recovery else None,
        "recovery_completion_step": min(completed, default=None) if recovery else None,
        "elapsed_ms": trace.get("elapsed_ms", 0.0),
        "prompt_tokens": trace.get("prompt_tokens", 0), "completion_tokens": trace.get("completion_tokens", 0),
        "max_ram_mb": trace.get("max_ram_mb"), "peak_server_ram_mb": trace.get("peak_server_ram_mb"),
        "peak_gpu_memory_mb": trace.get("peak_gpu_memory_mb"),
    }
