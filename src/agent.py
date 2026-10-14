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
    elif condition == "recovery":
        reference_actions = []

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
    }

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
        if not result.get("ok", False):
            event["tool_error"] = result.get("error", "tool_error")
        trace["events"].append(event)

    trace["elapsed_ms"] = round((time.perf_counter() - start_time) * 1000, 3)
    trace["prompt_tokens"] = total_prompt_tokens
    trace["completion_tokens"] = total_completion_tokens
    trace["final_state"] = environment.snapshot()
    trace["final_state_hash"] = environment.state_hash()
    trace["messages"] = messages
    return trace

