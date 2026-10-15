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
from src.dataset_validation import DatasetValidationError, load_validated_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one condition against a llama.cpp server")
    parser.add_argument("--tasks", default=str(ROOT / "tasks" / "tasks.json"))
    parser.add_argument("--condition", choices=["free", "normal_restart", "recovery", "error_restart"], required=True)
    parser.add_argument("--task-type", choices=["all", "execution", "recovery"], default=None,
                        help="Defaults to execution for free; recovery for normal_restart/error_restart. recovery is an alias for error_restart")
    parser.add_argument("--model-label", required=True, help="Label recorded in results, e.g. FP16, Q8, Q4")
    parser.add_argument("--api-model", default="local", help="Model name accepted by llama-server")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--server-pid",
        type=int,
        default=None,
        help="Optional llama/llama-server PID for peak RSS measurement",
    )
    parser.add_argument(
        "--ram-sample-interval-ms",
        type=int,
        default=100,
        help="Server RAM sampling interval in milliseconds (default: 100)",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-calls", type=int, default=None, help="Fixed model response-turn cap, including finish/invalid responses; otherwise normal horizon plus shared extra turns")
    parser.add_argument("--recovery-budget", type=int, default=3, help="Shared extra response turns above normal horizon, identical in both recovery branches (default: 3)")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.condition == "recovery":
        args.condition = "error_restart"
    if args.ram_sample_interval_ms <= 0:
        raise SystemExit("--ram-sample-interval-ms must be greater than zero")
    if args.output and Path(args.output).resolve() == Path(args.tasks).resolve():
        raise SystemExit("--output must not overwrite the input dataset")
    try:
        tasks, validation = load_validated_tasks(args.tasks)
    except DatasetValidationError as exc:
        print(json.dumps(exc.report, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    task_type = args.task_type
    if task_type is None:
        task_type = "execution" if args.condition == "free" else "recovery"
    if task_type != "all":
        tasks = [task for task in tasks if task.task_type == task_type]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("No tasks selected.")
    if args.recovery_budget < 0 or (args.max_calls is not None and args.max_calls < 1):
        raise SystemExit("Call budgets must be positive (extra budget may be zero).")
    if any((args.condition == "free" and task.task_type != "execution") or
           (args.condition == "error_restart" and task.schema_version != "recovery-v2") for task in tasks):
        raise SystemExit("free requires execution tasks; error_restart requires recovery-v2 tasks.")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = Path(args.output or ROOT / "results" / f"{args.model_label}_{args.condition}_{timestamp}.jsonl")
    metadata = collect_metadata(
        model_label=args.model_label,
        model_path=args.model_path,
        base_url=args.base_url,
        seed=args.seed,
        tasks_path=args.tasks,
        server_pid=args.server_pid,
        server_ram_sample_interval_ms=args.ram_sample_interval_ms,
    )
    metadata.update({"condition": args.condition, "task_type": task_type,
                     "max_calls": args.max_calls, "shared_extra_action_turns": args.recovery_budget,
                     "dataset_language": validation["language"],
                     "dataset_version": validation["dataset_version"],
                     "dataset_sha256": validation["dataset_sha256"],
                     "tasks_sha256": validation["tasks_file_sha256"],
                     "dataset_validation": {"valid": True, "validator_version": validation["validator_version"]}})
    client = LlamaCppClient(args.base_url, args.api_model, seed=args.seed)
    metadata["inference_settings"] = {"temperature": client.temperature, "seed": client.seed,
                                       "max_tokens": client.max_tokens, "timeout_seconds": client.timeout_seconds}
    records = run_batch(
        tasks,
        client,
        model_label=args.model_label,
        condition=args.condition,
        output_path=output,
        metadata=metadata,
        max_calls=args.max_calls,
        recovery_budget=args.recovery_budget,
        server_pid=args.server_pid,
        server_ram_sample_interval_ms=args.ram_sample_interval_ms,
    )
    scores = [record["score"] for record in records]
    print(json.dumps({"output": str(output), "summary": aggregate(scores)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

