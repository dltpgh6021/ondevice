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
        "elapsed_ms": trace.get("elapsed_ms", 0.0),
        "prompt_tokens": trace.get("prompt_tokens", 0),
        "completion_tokens": trace.get("completion_tokens", 0),
        "max_ram_mb": trace.get("max_ram_mb"),
    }
    return result


def aggregate(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        key = (result.get("model_label", "unknown"), result["condition"], result["horizon"])
        groups[key].append(result)

    summaries: list[dict[str, Any]] = []
    for (model_label, condition, horizon), rows in sorted(groups.items()):
        count = len(rows)

        def mean(name: str) -> float | None:
            values = [row[name] for row in rows if isinstance(row.get(name), (int, float))]
            return round(sum(values) / len(values), 4) if values else None

        def rate(name: str) -> float | None:
            values = [row[name] for row in rows if isinstance(row.get(name), bool)]
            return round(sum(values) / len(values), 4) if values else None

        summaries.append(
            {
                "model_label": model_label,
                "condition": condition,
                "horizon": horizon,
                "n": count,
                "success_rate": rate("strict_success"),
                "natural_error_rate": rate("natural_error") if condition == "free" else None,
                "next_action_accuracy": rate("next_action_correct")
                if condition == "normal_restart"
                else None,
                "recovery_rate": rate("recovery_success")
                if condition == "recovery"
                else None,
                "mean_extra_calls": mean("extra_calls")
                if condition == "recovery"
                else None,
                "mean_elapsed_ms": mean("elapsed_ms"),
                "mean_prompt_tokens": mean("prompt_tokens"),
                "mean_completion_tokens": mean("completion_tokens"),
                "mean_max_ram_mb": mean("max_ram_mb"),
            }
        )
    return summaries

