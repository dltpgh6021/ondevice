"""Metrics that separate natural errors from recovery performance."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .tasks import Task


def _tool_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in trace.get("events", [])
        if (event.get("action") or {}).get("type") == "tool_call"
    ]


def _same_action(event: dict[str, Any], expected: dict[str, Any]) -> bool:
    action = event.get("action") or {}
    return (
        action.get("name") == expected.get("tool")
        and (action.get("arguments") or {}) == (expected.get("arguments") or {})
    )


def _final_state_success(state: dict[str, Any], condition: dict[str, Any]) -> bool:
    if "completed_request_equals" in condition:
        from .execution_environment import completion_condition_matches

        return completion_condition_matches(state, condition)
    if "cart_contains" in condition:
        return condition["cart_contains"] in state.get("cart", [])
    if "cart_equals" in condition:
        return list(condition["cart_equals"]) == state.get("cart", [])
    if "calendar_contains" in condition:
        required = condition["calendar_contains"]
        return any(
            all(event.get(key) == value for key, value in required.items())
            for event in state.get("calendar", [])
        )
    if "last_result_field_equals" in condition:
        spec = condition["last_result_field_equals"]
        return (state.get("last_result") or {}).get(spec["field"]) == spec["value"]
    raise ValueError(f"unsupported success condition: {condition}")


def score_trace(task: Task, trace: dict[str, Any]) -> dict[str, Any]:
    if task.schema_version in {"execution-v2", "recovery-v3"}:
        from .workflow_metrics import score_workflow_trace

        return score_workflow_trace(task, trace)
    condition = trace["condition"]
    observed = _tool_events(trace)
    expected = task.gold_actions
    if condition == "normal_restart":
        expected = task.gold_actions[task.checkpoint_step :]
    elif condition == "recovery":
        expected = []

    first_error_step: int | None = None
    wrong_tool_count = 0
    wrong_argument_count = 0
    if condition in {"free", "normal_restart"}:
        for index, expected_action in enumerate(expected):
            if index >= len(observed):
                first_error_step = first_error_step or (index + 1)
                break
            if not _same_action(observed[index], expected_action):
                first_error_step = first_error_step or observed[index]["step"]
                action = observed[index].get("action") or {}
                if action.get("name") != expected_action.get("tool"):
                    wrong_tool_count += 1
                else:
                    wrong_argument_count += 1
        if len(observed) > len(expected):
            first_error_step = first_error_step or observed[len(expected)]["step"]
    tool_error_count = sum(
        1 for event in trace.get("events", []) if event.get("tool_error") or event.get("error")
    )
    recovery_tool_error_count = sum(
        1 for event in trace.get("events", []) if event.get("tool_error")
    )
    recovery_invalid_action_count = sum(
        1
        for event in trace.get("events", [])
        if event.get("error") in {"invalid_action", "invalid_arguments"}
    )
    recovery_runtime_error_count = int(trace.get("runtime_error") is not None)
    recovery_error_count = (
        recovery_tool_error_count
        + recovery_invalid_action_count
        + recovery_runtime_error_count
        if condition == "recovery"
        else None
    )
    recovery_error_steps = [
        event.get("step")
        for event in trace.get("events", [])
        if event.get("tool_error")
        or event.get("error") in {"invalid_action", "invalid_arguments"}
        or (
            trace.get("runtime_error") is not None
            and event.get("error") == trace.get("runtime_error")
        )
    ]
    recovery_first_error_step = min(recovery_error_steps) if recovery_error_steps else None
    recovery_completion_steps = [
        event.get("step")
        for event in trace.get("events", [])
        if event.get("success_after_action") is True
    ]
    recovery_completion_step = min(recovery_completion_steps) if recovery_completion_steps else None
    success = (
        trace.get("runtime_error") is None
        and _final_state_success(trace.get("final_state", {}), task.success_condition)
    )
    remaining_gold_calls = max(0, task.horizon - task.checkpoint_step)
    extra_calls = max(0, len(observed) - remaining_gold_calls)
    next_action_correct: bool | None = None
    if condition == "normal_restart":
        next_action_correct = bool(observed) and _same_action(observed[0], expected[0])

    result: dict[str, Any] = {
        "task_id": task.task_id,
        "base_id": task.base_id,
        "task_type": task.task_type,
        "horizon": task.horizon,
        "condition": condition,
        "strict_success": success,
        "final_state_match": success,
        "natural_error": condition == "free" and first_error_step is not None,
        "first_error_step": first_error_step,
        "next_action_correct": next_action_correct,
        "recovery_success": success if condition == "recovery" else None,
        "tool_call_count": len(observed),
        "extra_calls": extra_calls if condition == "recovery" else 0,
        "wrong_tool_count": wrong_tool_count,
        "wrong_argument_count": wrong_argument_count,
        "tool_error_count": tool_error_count,
        "recovery_error_count": recovery_error_count,
        "recovery_tool_error_count": recovery_tool_error_count if condition == "recovery" else None,
        "recovery_invalid_action_count": recovery_invalid_action_count if condition == "recovery" else None,
        "recovery_runtime_error_count": recovery_runtime_error_count if condition == "recovery" else None,
        "recovery_first_error_step": recovery_first_error_step if condition == "recovery" else None,
        "recovery_completion_step": recovery_completion_step if condition == "recovery" else None,
        "elapsed_ms": trace.get("elapsed_ms", 0.0),
        "prompt_tokens": trace.get("prompt_tokens", 0),
        "completion_tokens": trace.get("completion_tokens", 0),
        "max_ram_mb": trace.get("max_ram_mb"),
        "peak_server_ram_mb": trace.get("peak_server_ram_mb"),
    }
    return result


def aggregate(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        key = (result.get("model_label", "unknown"), result.get("protocol_version", "legacy-v1"),
               result.get("metrics_version", "legacy-unversioned"),
               result.get("agent_policy_version", "legacy-unversioned"),
               result["condition"], result["horizon"], result.get("error_kind") or "")
        groups[key].append(result)

    summaries: list[dict[str, Any]] = []
    for (model_label, protocol_version, metrics_version, agent_policy_version, condition, horizon, error_kind), rows in sorted(groups.items()):
        count = len(rows)
        valid_rows = [row for row in rows if row.get("valid_for_behavior_metrics", True)]
        is_recovery = condition in {"recovery", "error_restart", "signaled_error_restart"}

        def mean(name: str) -> float | None:
            source = [row for row in valid_rows if row.get("strict_success")] if name.startswith("extra_calls") else valid_rows
            values = [row[name] for row in source if isinstance(row.get(name), (int, float))
                      and not isinstance(row[name], bool)]
            return round(sum(values) / len(values), 4) if values else None

        def rate(name: str) -> float | None:
            values = [row[name] for row in valid_rows if isinstance(row.get(name), bool)]
            return round(sum(values) / len(values), 4) if values else None

        def pooled(numerator: str, denominator: str) -> float | None:
            available = [row for row in valid_rows
                         if isinstance(row.get(numerator), (int, float)) and not isinstance(row[numerator], bool)
                         and isinstance(row.get(denominator), (int, float)) and not isinstance(row[denominator], bool)]
            total = sum(row[denominator] for row in available)
            return round(sum(row[numerator] for row in available) / total, 4) if total else None

        def pooled_rate(tokens: str, milliseconds: str) -> float | None:
            """Tokens per second over all responses: sum(tokens) / sum(seconds)."""
            available = [row for row in valid_rows
                         if isinstance(row.get(tokens), (int, float)) and isinstance(row.get(milliseconds), (int, float))]
            total_ms = sum(row[milliseconds] for row in available)
            return round(sum(row[tokens] for row in available) / (total_ms / 1000), 3) if total_ms > 0 else None

        error_episodes = [row["error_count"] > 0 for row in valid_rows
                          if isinstance(row.get("error_count"), (int, float))]

        def conditional_rate(condition_field: str, outcome_field: str) -> float | None:
            subset = [row for row in valid_rows if row.get(condition_field) is True]
            return round(sum(bool(row.get(outcome_field)) for row in subset) / len(subset), 4) if subset else None

        stage_counts: dict[str, int] = defaultdict(int)
        for row in valid_rows:
            if row.get("first_error_turn") is not None:
                stage_counts[str(row.get("first_error_workflow_stage") or "non_workflow")] += 1
        extra_all = [row["extra_calls_over_normal"] for row in valid_rows
                     if isinstance(row.get("extra_calls_over_normal"), int)]

        summaries.append(
            {
                "model_label": model_label,
                "protocol_version": protocol_version,
                "metrics_version": metrics_version,
                "agent_policy_version": agent_policy_version,
                "condition": condition,
                "horizon": horizon,
                "error_kind": error_kind or None,
                "n": count,
                "n_valid": len(valid_rows),
                "n_runtime_errors": count - len(valid_rows),
                "n_infrastructure_errors": sum(row.get("infrastructure_error_count", 0) for row in rows),
                "n_internal_errors": sum(row.get("internal_error_count", 0) for row in rows),
                "n_unclassified_runtime_errors": sum(row.get("unclassified_runtime_error_count", 0) for row in rows),
                "n_budget_exhausted": sum(row.get("budget_exhausted") is True for row in valid_rows),
                "success_rate": rate("strict_success"),
                "behavior_error_episode_rate": round(sum(error_episodes) / len(error_episodes), 4) if error_episodes else None,
                "mean_error_count": mean("error_count"),
                "mean_model_output_error_count": mean("model_output_error_count"),
                "mean_response_turn_count": mean("response_turn_count"),
                "mean_attempted_tool_call_count": mean("attempted_tool_call_count"),
                "mean_executed_tool_call_count": mean("executed_tool_call_count"),
                "mean_error_turn_rate": mean("error_turn_rate"),
                "pooled_error_turn_rate": pooled("error_count", "response_turn_count"),
                "mean_tool_action_error_rate": mean("tool_action_error_rate"),
                "pooled_tool_action_error_rate": pooled("tool_error_action_count", "executed_tool_call_count"),
                "natural_error_rate": rate("natural_error") if condition == "free" else None,
                # Among free runs with a natural error, the share that still ended correct.
                "natural_error_recovery_rate": conditional_rate("natural_error", "strict_success")
                if condition == "free" else None,
                # Same mistake as the injected recovery error, made by the model itself.
                "wrong_save_episode_rate": rate("wrong_save_episode"),
                "wrong_save_recovery_rate": conditional_rate("wrong_save_episode", "strict_success"),
                # Nonexistent identifiers (copy errors), separate from wrong but existing choices above.
                "identifier_error_episode_rate": rate("identifier_error_episode"),
                "mean_identifier_error_count": mean("identifier_error_count"),
                "first_error_stage_counts": dict(sorted(stage_counts.items())),
                "next_action_accuracy": rate("next_action_correct"),
                "next_action_valid_rate": rate("next_action_valid"),
                "n_successful": sum(row.get("strict_success") is True for row in valid_rows),
                # Survivorship-free companion to the success-only extra-call means.
                "mean_extra_calls_over_normal_all_runs": round(sum(extra_all) / len(extra_all), 4) if extra_all else None,
                "recovery_rate": rate("recovery_success")
                if is_recovery
                else None,
                "mean_extra_calls": mean("extra_calls")
                if is_recovery
                else None,
                "mean_recovery_error_count": mean("recovery_error_count")
                if is_recovery
                else None,
                "mean_recovery_tool_error_count": mean("recovery_tool_error_count")
                if is_recovery
                else None,
                "mean_recovery_invalid_action_count": mean("recovery_invalid_action_count")
                if is_recovery
                else None,
                "mean_recovery_model_output_error_count": mean("recovery_model_output_error_count")
                if is_recovery
                else None,
                "mean_recovery_first_error_step": mean("recovery_first_error_step")
                if is_recovery
                else None,
                "mean_recovery_completion_step": mean("recovery_completion_step")
                if is_recovery
                else None,
                # All valid runs: a run that fails early also ends early, so this
                # mixes speed with behavior. Prefer the normalized fields below.
                "mean_elapsed_ms": mean("elapsed_ms"),
                "mean_elapsed_ms_successful": (lambda values: round(sum(values) / len(values), 4) if values else None)(
                    [row["elapsed_ms"] for row in valid_rows if row.get("strict_success") is True
                     and isinstance(row.get("elapsed_ms"), (int, float))]),
                "mean_ms_per_response_turn": mean("ms_per_response_turn"),
                "pooled_prompt_tokens_per_second": pooled_rate("prompt_eval_tokens", "prompt_eval_ms"),
                "pooled_generation_tokens_per_second": pooled_rate("generated_tokens", "generation_ms"),
                "mean_extra_calls_over_normal": mean("extra_calls_over_normal"),
                "mean_extra_calls_over_oracle": mean("extra_calls_over_oracle"),
                "recovery_error_resolved_rate": rate("recovery_error_resolved") if is_recovery else None,
                "mean_recovery_resolution_step": mean("recovery_resolution_step") if is_recovery else None,
                "mean_recovery_first_resolution_turn": mean("recovery_first_resolution_turn") if is_recovery else None,
                "mean_recovery_stable_resolution_turn": mean("recovery_stable_resolution_turn") if is_recovery else None,
                "mean_recovery_semantic_error_count": mean("recovery_semantic_error_count") if is_recovery else None,
                "mean_prompt_tokens": mean("prompt_tokens"),
                "mean_completion_tokens": mean("completion_tokens"),
                "mean_max_ram_mb": mean("max_ram_mb"),
                "mean_peak_server_ram_mb": mean("peak_server_ram_mb"),
                "mean_peak_gpu_memory_mb": mean("peak_gpu_memory_mb"),
            }
        )
    return summaries

