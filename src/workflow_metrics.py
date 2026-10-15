"""State-based workflow metrics; optional valid lookups are not path errors."""

from __future__ import annotations

from typing import Any

from .execution_environment import completion_condition_matches
from .tasks import Task


def score_workflow_trace(task: Task, trace: dict[str, Any]) -> dict[str, Any]:
    condition = trace["condition"]
    recovery = condition in {"error_restart", "recovery"}
    events = trace.get("events", [])
    tool_events = [event for event in events if (event.get("action") or {}).get("type") == "tool_call"]
    invalid = [event for event in events if event.get("error") in {"invalid_action", "invalid_arguments"}]
    tool_errors = [event for event in events if event.get("tool_error")]
    semantic = [event for event in events if event.get("semantic_error")]
    errors = [event for event in events if event.get("tool_error") or event.get("semantic_error")
              or event.get("error") in {"invalid_action", "invalid_arguments"}]
    first_error = min((event["step"] for event in errors), default=None)
    completed = [event["step"] for event in events if event.get("success_after_action") is True]
    state_match = completion_condition_matches(trace.get("final_state", {}), task.success_condition)
    valid = trace.get("runtime_error") is None
    success = valid and state_match
    paired = task.schema_version == "recovery-v2"
    normal_calls = task.normal_remaining_calls if paired else task.horizon - (
        task.checkpoint_step if condition == "normal_restart" else 0)
    oracle_calls = task.recovery_min_calls if recovery else normal_calls
    injected_ids = [item["result"]["saved_result"]["request_id"]
                    for item in trace.get("injected_actions", [])]
    saved = trace.get("final_state", {}).get("completed_requests", {})
    resolution_steps = [event["step"] for event in events if event.get("injected_error_present_after") is False]
    return {
        "task_id": task.task_id, "base_id": task.base_id, "task_type": task.task_type,
        "task_schema_version": task.schema_version, "protocol_version": trace.get("protocol_version", task.schema_version),
        "pair_id": task.base_id if paired else None, "fixture_sha256": trace.get("fixture_sha256"),
        "checkpoint_state_hash": trace.get("checkpoint_state_hash"),
        "resume_workflow_step": trace.get("resume_workflow_step"),
        "horizon": task.horizon, "condition": condition,
        "normal_remaining_calls": normal_calls, "recovery_min_calls": task.recovery_min_calls,
        "strict_success": success, "final_state_match": state_match,
        "valid_for_behavior_metrics": valid,
        "natural_error": bool(errors) if condition == "free" and valid else None,
        "first_error_step": first_error,
        "next_action_correct": None,
        "recovery_success": success if recovery else None,
        "recovery_error_resolved": all(key not in saved for key in injected_ids) if recovery and injected_ids else None,
        "recovery_resolution_step": min(resolution_steps, default=None) if recovery else None,
        "tool_call_count": len(tool_events),
        "extra_calls": max(0, len(tool_events) - normal_calls) if recovery else 0,
        "extra_calls_over_normal": max(0, len(tool_events) - normal_calls),
        "extra_calls_over_oracle": max(0, len(tool_events) - oracle_calls),
        "wrong_tool_count": sum(event.get("tool_error") == "unknown_tool" for event in tool_errors),
        "wrong_argument_count": sum(event.get("tool_error") != "unknown_tool" for event in tool_errors) + len(invalid),
        "tool_error_count": len(tool_errors), "semantic_error_count": len(semantic),
        "recovery_error_count": len(errors) if recovery else None,
        "recovery_tool_error_count": len(tool_errors) if recovery else None,
        "recovery_invalid_action_count": len(invalid) if recovery else None,
        "recovery_semantic_error_count": len(semantic) if recovery else None,
        "recovery_runtime_error_count": int(not valid) if recovery else None,
        "recovery_first_error_step": first_error if recovery else None,
        "recovery_completion_step": min(completed, default=None) if recovery else None,
        "elapsed_ms": trace.get("elapsed_ms", 0.0),
        "prompt_tokens": trace.get("prompt_tokens", 0), "completion_tokens": trace.get("completion_tokens", 0),
        "max_ram_mb": trace.get("max_ram_mb"), "peak_server_ram_mb": trace.get("peak_server_ram_mb"),
    }
