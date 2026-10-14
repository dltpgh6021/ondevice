from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.client import LlamaCppClient
from src.experiment import collect_metadata, run_batch
from src.metrics import aggregate
from src.tasks import load_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one condition against a llama.cpp server")
    parser.add_argument("--tasks", default=str(ROOT / "tasks" / "tasks.json"))
    parser.add_argument("--condition", choices=["free", "normal_restart", "recovery"], required=True)
    parser.add_argument("--task-type", choices=["all", "execution", "recovery"], default=None,
                        help="Optional task-type filter; defaults to execution for free and recovery for recovery")
    parser.add_argument("--model-label", required=True, help="Label recorded in results, e.g. FP16, Q8, Q4")
    parser.add_argument("--api-model", default="local", help="Model name accepted by llama-server")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-calls", type=int, default=None, help="Optional fixed cap; otherwise required calls plus recovery budget")
    parser.add_argument("--recovery-budget", type=int, default=3, help="Extra tool calls allowed for recovery (default: 3)")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = load_tasks(args.tasks)
    task_type = args.task_type
    if task_type is None:
        task_type = "recovery" if args.condition == "recovery" else "execution"
    if task_type != "all":
        tasks = [task for task in tasks if task.task_type == task_type]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = Path(args.output or ROOT / "results" / f"{args.model_label}_{args.condition}_{timestamp}.jsonl")
    metadata = collect_metadata(
        model_label=args.model_label,
        model_path=args.model_path,
        base_url=args.base_url,
        seed=args.seed,
        tasks_path=args.tasks,
    )
    client = LlamaCppClient(args.base_url, args.api_model, seed=args.seed)
    records = run_batch(
        tasks,
        client,
        model_label=args.model_label,
        condition=args.condition,
        output_path=output,
        metadata=metadata,
        max_calls=args.max_calls,
        recovery_budget=args.recovery_budget,
    )
    scores = [record["score"] for record in records]
    print(json.dumps({"output": str(output), "summary": aggregate(scores)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

