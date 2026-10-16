"""A transparent Agent loop shared by every model condition."""

from __future__ import annotations

import copy
import json
import time
from typing import Any

from .client import InfrastructureError, ModelOutputError
from .environment import MockEnvironment
from .tasks import Task

AGENT_POLICY_VERSION = "single-action-feedback-v2"
METRICS_VERSION = "workflow-metrics-v6"
ERROR_CONDITIONS = {"recovery", "error_restart", "signaled_error_restart"}
_TIMING_FIELDS = ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms")


def _model_text(output: Any) -> str:
    """What the model actually produced, as plain text for the conversation."""
    if isinstance(output, dict):
        parts = []
        content = output.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        calls = output.get("tool_calls")
        if calls:
            parts.append(json.dumps(calls, ensure_ascii=False))
        if parts:
            return "\n".join(parts)
        if output.get("type") == "tool_call":
            return json.dumps({"name": output.get("name"), "arguments": output.get("arguments")}, ensure_ascii=False)
        if "error" in output:
            return f"[unusable response: {output['error']}]"
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False)


def _add_timings(totals: dict[str, float], payload: Any) -> None:
    """Accumulate llama.cpp per-response timings (prompt eval and generation)."""
    timings = payload.get("timings") if isinstance(payload, dict) else None
    if not isinstance(timings, dict):
        return
    totals["responses_with_timings"] += 1
    for field in _TIMING_FIELDS:
        value = timings.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            totals[field] += value


def _output_feedback(messages: list[dict[str, Any]], output: Any, kind: str) -> None:
    """Fixed format feedback, never a task answer or a corrected tool argument."""
    messages.append({"role": "assistant", "content": _model_text(output)})
    messages.append({"role": "user", "content": (
        f"Your last response could not be used ({kind}). Return exactly one tool call "
        "with a JSON object for its arguments, or a final answer if the task is complete. "
        "No action from that response was executed."
    )})


def _assistant_tool_message(action: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": action["name"],
                    "arguments": json.dumps(action.get("arguments", {}), ensure_ascii=False),
                },
            }
        ],
    }


def run_agent(
    task: Task,
    client: Any,
    environment: MockEnvironment,
    *,
    condition: str,
    max_calls: int = 10,
) -> dict[str, Any]:
    """Run one task and return a fully inspectable trace.

    Invalid model output consumes one response turn and receives fixed format
    feedback. No task-specific correction is supplied. Infrastructure failures
    stop the run and are excluded from behavioral rates, not silently retried.
    """

    messages, _ = environment.prepare_condition(task, condition)
    initial_state = environment.snapshot()
    initial_messages = copy.deepcopy(messages)
    reference_actions = task.gold_actions
    if condition == "normal_restart":
        reference_actions = task.gold_actions[task.checkpoint_step :]
    elif condition in ERROR_CONDITIONS:
        reference_actions = task.recovery_actions or []

    trace: dict[str, Any] = {
        "task_id": task.task_id,
        "base_id": task.base_id,
        "task_type": task.task_type,
        "condition": condition,
        "horizon": task.horizon,
        "checkpoint_step": task.checkpoint_step,
        "initial_state": initial_state,
        "initial_state_hash": environment.state_hash(),
        "initial_messages": initial_messages,
        "reference_tools": [action["tool"] for action in reference_actions],
        "events": [],
        "runtime_error": None,
        "infrastructure_error": None,
        "internal_error": None,
        "termination_reason": "budget_exhausted",
        "metrics_version": METRICS_VERSION,
        "agent_policy_version": AGENT_POLICY_VERSION,
        "finished": False,
        "final_answer": "",
        "schema_version": task.schema_version,
        "dataset_language": task.language,
        "dataset_version": task.dataset_version,
        "pair_id": task.base_id if task.task_type == "recovery" else None,
        "normal_remaining_calls": task.normal_remaining_calls,
        "recovery_min_calls": task.recovery_min_calls if condition in ERROR_CONDITIONS else None,
    }
    trace.update(copy.deepcopy(getattr(environment, "preparation_metadata", {})))

    start_time = time.perf_counter()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    timing_totals: dict[str, float] = {field: 0.0 for field in _TIMING_FIELDS}
    timing_totals["responses_with_timings"] = 0
    executed_calls = 0
    for step in range(max_calls):
        event: dict[str, Any] = {"step": step + 1}
        call_start = time.perf_counter()
        try:
            response = client.chat(messages, environment.tool_definitions())
            action = response.get("action") or {}
            usage = response.get("usage") or {}
            total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            _add_timings(timing_totals, response.get("raw_response"))
        except ModelOutputError as exc:
            total_prompt_tokens += int(exc.usage.get("prompt_tokens", 0) or 0)
            total_completion_tokens += int(exc.usage.get("completion_tokens", 0) or 0)
            _add_timings(timing_totals, exc.payload)
            event.update({"error": "model_output_error", "model_error": exc.kind,
                          "error_detail": str(exc), "raw_response": exc.payload,
                          "usage": exc.usage, "finish_reason": exc.finish_reason,
                          "attempted_tool_call_count": exc.attempted_tool_call_count,
                          "response_received": True,
                          "tool_executed": False,
                          "latency_ms": round((time.perf_counter() - call_start) * 1000, 3)})
            trace["events"].append(event)
            choices = exc.payload.get("choices") or []
            output = choices[0].get("message") if choices else {"error": exc.kind}
            _output_feedback(messages, output, exc.kind)
            continue
        except Exception as exc:  # keep a failed run available for analysis
            trace["runtime_error"] = f"{type(exc).__name__}: {exc}"
            category = "infrastructure_error" if isinstance(exc, InfrastructureError) else "internal_error"
            trace[category] = trace["runtime_error"]
            trace["termination_reason"] = category
            event["error"] = trace["runtime_error"]
            event["error_category"] = category
            event["failure_kind"] = getattr(exc, "kind", type(exc).__name__)
            event["tool_executed"] = False
            event["response_received"] = False
            event["latency_ms"] = round((time.perf_counter() - call_start) * 1000, 3)
            trace["events"].append(event)
            break

        event["action"] = action
        event["response_received"] = True
        event["usage"] = usage
        if "raw_response" in response:
            event["raw_response"] = response["raw_response"]
        event["finish_reason"] = response.get("finish_reason")
        event["tool_executed"] = False
        event["attempted_tool_call_count"] = response.get("attempted_tool_call_count", int(
            isinstance(action, dict) and action.get("type") == "tool_call"))
        event["latency_ms"] = round((time.perf_counter() - call_start) * 1000, 3)
        action_type = action.get("type") if isinstance(action, dict) else None
        if action_type == "finish":
            answer = str(action.get("answer", ""))
            messages.append({"role": "assistant", "content": answer})
            event["finish"] = True
            trace["finished"] = True
            trace["termination_reason"] = "finished"
            trace["final_answer"] = answer
            if task.schema_version in {"execution-v2", "recovery-v3"} and not environment.check_success(task.success_condition):
                event["semantic_error"] = "premature_finish"
            trace["events"].append(event)
            break

        if action_type != "tool_call" or not isinstance(action.get("name"), str) or not action["name"]:
            event["error"] = "invalid_action"
            trace["events"].append(event)
            _output_feedback(messages, action, "invalid_action")
            continue

        call_id = f"call_{step + 1}"
        name = str(action["name"])
        arguments = action.get("arguments", {})
        if not isinstance(arguments, dict):
            event["error"] = "invalid_arguments"
            trace["events"].append(event)
            _output_feedback(messages, action, "invalid_arguments")
            continue

        messages.append(_assistant_tool_message(action, call_id))
        executed_calls += 1
        event["tool_executed"] = True
        event["tool_call_index"] = executed_calls
        try:
            result = environment.execute(name, arguments)
        except Exception as exc:
            trace["runtime_error"] = f"{type(exc).__name__}: {exc}"
            trace["internal_error"] = trace["runtime_error"]
            trace["termination_reason"] = "internal_error"
            event.update({"error": trace["runtime_error"], "error_category": "internal_error"})
            trace["events"].append(event)
            break
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
        event["tool_result"] = result
        event["state_hash_after"] = environment.state_hash()
        if (task.schema_version == "recovery-v3" and condition in ERROR_CONDITIONS
                and hasattr(environment, "injected_error_present")):
            event["injected_error_present_after"] = environment.injected_error_present()
        try:
            event["success_after_action"] = environment.check_success(task.success_condition)
        except Exception:
            event["success_after_action"] = False
        if not result.get("ok", False):
            event["tool_error"] = result.get("error", "tool_error")
        elif hasattr(environment, "assess_action_result"):
            semantic_error = environment.assess_action_result(result)
            if semantic_error:
                event["semantic_error"] = semantic_error
        trace["events"].append(event)

    trace["elapsed_ms"] = round((time.perf_counter() - start_time) * 1000, 3)
    trace["prompt_tokens"] = total_prompt_tokens
    trace["completion_tokens"] = total_completion_tokens
    trace["server_timings"] = {key: (round(value, 3) if isinstance(value, float) else value)
                               for key, value in timing_totals.items()}
    trace["final_state"] = environment.snapshot()
    trace["final_state_hash"] = environment.state_hash()
    trace["messages"] = messages
    return trace

