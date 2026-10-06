from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .llm_client import LLMClient
from .mock_tools import MockToolEnvironment, ToolError


TOOL_SPECS = [
    {
        "name": "lookup_customer",
        "description": "Look up a customer by email.",
        "arguments": {"email": "string"},
    },
    {
        "name": "get_order",
        "description": "Retrieve one order by its order ID.",
        "arguments": {"order_id": "string"},
    },
    {
        "name": "approve_refund",
        "description": "Approve a refund for an eligible order by order ID. This must happen before refund_order.",
        "arguments": {"order_id": "string"},
    },
    {
        "name": "refund_order",
        "description": "Execute an already-approved refund by order ID. Call only after approve_refund succeeds.",
        "arguments": {"order_id": "string"},
    },
]


@dataclass
class AgentRunner:
    client: LLMClient
    max_steps: int = 6

    def run(self, task: dict[str, Any]) -> dict[str, Any]:
        environment = MockToolEnvironment.from_task(task)
        history: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        started = time.perf_counter()
        finished_reason = "max_steps"

        for step in range(1, min(self.max_steps, task["max_steps"]) + 1):
            record: dict[str, Any] = {"step": step, "state_before": environment.snapshot()}
            try:
                action = self.client.next_action(
                    task["instruction"],
                    history,
                    TOOL_SPECS,
                )
                record["action"] = action
                if action["action"] == "finish":
                    record["success_at_finish"] = environment.is_success(task["expected_final"])
                    if record["success_at_finish"]:
                        finished_reason = "model_finish"
                        history.append({"step": step, "action": action})
                        records.append(record)
                        break
                    record["error_type"] = "PrematureFinish:task_not_complete"
                    history.append(
                        {
                            "step": step,
                            "action": action,
                            "error": record["error_type"],
                            "recovery_required": True,
                        }
                    )
                    record["state_after"] = environment.snapshot()
                    records.append(record)
                    continue
                result = environment.call(action["tool"], action.get("arguments", {}))
                record["tool_result"] = result
                record["error_type"] = None
                history.append({"step": step, "action": action, "tool_result": result})
            except (ToolError, ValueError, KeyError, TypeError) as exc:
                record["error_type"] = type(exc).__name__ + ":" + str(exc)
                history.append(
                    {
                        "step": step,
                        "action": record.get("action"),
                        "error": record["error_type"],
                        "recovery_required": True,
                    }
                )
            record["state_after"] = environment.snapshot()
            records.append(record)

        success = environment.is_success(task["expected_final"])
        if success:
            finished_reason = "success"
        return {
            "task_id": task["task_id"],
            "success": success,
            "finished_reason": finished_reason,
            "steps_executed": len(records),
            "elapsed_seconds": time.perf_counter() - started,
            "trajectory": records,
            "final_state": environment.snapshot(),
        }

