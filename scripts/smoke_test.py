from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import run_agent
from src.client import ScriptedClient
from src.environment import environment_for_task
from src.metrics import score_trace
from src.tasks import load_tasks


def tool(name: str, arguments: dict) -> dict:
    return {"type": "tool_call", "name": name, "arguments": arguments}


def finish() -> dict:
    return {"type": "finish", "answer": "done"}


def main() -> int:
    task_path = ROOT / "tasks" / "tasks.json"
    tasks = load_tasks(task_path)
    execution_task = next(task for task in tasks if task.task_id == "shopping_001_h3")
    recovery_task = next(task for task in tasks if task.task_id == "recovery_shopping_001_h3")
    scripted_actions = {
        "free": [tool(action["tool"], action.get("arguments", {}))
                 for action in execution_task.gold_actions] + [finish()],
        "normal_restart": [tool(action["tool"], action.get("arguments", {}))
                           for action in execution_task.gold_actions[execution_task.checkpoint_step:]] + [finish()],
        "recovery": [
            tool("inspect_state", {}),
            tool("remove_from_cart", {"product_id": "keyboard_99"}),
            tool("add_to_cart", {"product_id": "keyboard_02"}),
            finish(),
        ],
    }
    task_by_condition = {
        "free": execution_task,
        "normal_restart": execution_task,
        "recovery": recovery_task,
    }

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    all_scores = []
    for condition, actions in scripted_actions.items():
        environment = environment_for_task(task_by_condition[condition])
        trace = run_agent(
            task_by_condition[condition],
            ScriptedClient(actions),
            environment,
            condition=condition,
            max_calls=10,
        )
        score = score_trace(task_by_condition[condition], trace)
        all_scores.append(score)
        print(json.dumps(score, ensure_ascii=False))
        (output_dir / f"smoke_{condition}.json").write_text(
            json.dumps({"trace": trace, "score": score}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if not all(score["strict_success"] for score in all_scores):
        print("SMOKE TEST FAILED", file=sys.stderr)
        return 1
    execution_count = 0
    for task in tasks:
        if task.task_type != "execution":
            continue
        actions = [tool(action["tool"], action.get("arguments", {})) for action in task.gold_actions] + [finish()]
        trace = run_agent(task, ScriptedClient(actions), environment_for_task(task),
                          condition="free", max_calls=task.horizon + 1)
        score = score_trace(task, trace)
        if not score["strict_success"] or score["natural_error"] or score["tool_call_count"] != task.horizon:
            print(f"SMOKE TEST FAILED: {task.task_id}", file=sys.stderr)
            return 1
        execution_count += 1
    print(f"Execution scripted replay: {execution_count}/{execution_count} passed (not an LLM quality test)")
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

