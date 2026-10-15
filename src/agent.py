"""A transparent Agent loop shared by every model condition."""

from __future__ import annotations

import copy
import json
import time
from typing import Any

from .environment import MockEnvironment
from .tasks import Task


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

    The loop has no hidden retry or correction policy. The model receives the
    tool result and must decide how to recover itself.
    """

    messages, _ = environment.prepare_condition(task, condition)
    initial_state = environment.snapshot()
    initial_messages = copy.deepcopy(messages)
    reference_actions = task.gold_actions
    if condition == "normal_restart":
        reference_actions = task.gold_actions[task.checkpoint_step :]
    elif condition in {"recovery", "error_restart"}:
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
        "finished": False,
        "final_answer": "",
        "schema_version": task.schema_version,
        "dataset_language": task.language,
        "dataset_version": task.dataset_version,
        "pair_id": task.base_id if task.task_type == "recovery" else None,
        "normal_remaining_calls": task.normal_remaining_calls,
        "recovery_min_calls": task.recovery_min_calls,
    }
    trace.update(copy.deepcopy(getattr(environment, "preparation_metadata", {})))

    start_time = time.perf_counter()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for step in range(max_calls):
        event: dict[str, Any] = {"step": step + 1}
        call_start = time.perf_counter()
        try:
            response = client.chat(messages, environment.tool_definitions())
            action = response.get("action") or {}
            usage = response.get("usage") or {}
            total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        except Exception as exc:  # keep a failed run available for analysis
            trace["runtime_error"] = f"{type(exc).__name__}: {exc}"
            event["error"] = trace["runtime_error"]
            event["latency_ms"] = round((time.perf_counter() - call_start) * 1000, 3)
            trace["events"].append(event)
            break

        event["action"] = action
        event["latency_ms"] = round((time.perf_counter() - call_start) * 1000, 3)
        action_type = action.get("type")
        if action_type == "finish":
            answer = str(action.get("answer", ""))
            messages.append({"role": "assistant", "content": answer})
            event["finish"] = True
            trace["finished"] = True
            trace["final_answer"] = answer
            if task.schema_version in {"execution-v2", "recovery-v2"} and not environment.check_success(task.success_condition):
                event["semantic_error"] = "premature_finish"
            trace["events"].append(event)
            break

        if action_type != "tool_call" or not action.get("name"):
            event["error"] = "invalid_action"
            trace["events"].append(event)
            continue

        call_id = f"call_{step + 1}"
        name = str(action["name"])
        arguments = action.get("arguments") or {}
        if not isinstance(arguments, dict):
            event["error"] = "invalid_arguments"
            trace["events"].append(event)
            continue

        messages.append(_assistant_tool_message(action, call_id))
        result = environment.execute(name, arguments)
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
        if task.schema_version == "recovery-v2" and condition in {"error_restart", "recovery"}:
            injected_ids = [item["result"]["saved_result"]["request_id"] for item in trace.get("injected_actions", [])]
            event["injected_error_present_after"] = any(key in environment.state["completed_requests"] for key in injected_ids)
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
    trace["final_state"] = environment.snapshot()
    trace["final_state_hash"] = environment.state_hash()
    trace["messages"] = messages
    return trace

