from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.client import InfrastructureError, LlamaCppClient
from src.experiment import collect_metadata, run_batch
from src.metrics import aggregate
from src.dataset_validation import DatasetValidationError, load_validated_tasks
from src.run_preparation import PreparationError, prepare_run
from src.tasks import Task


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
    parser.add_argument("--gpu-index", type=int, default=None,
                        help="Optional NVIDIA GPU index; samples device memory.used via nvidia-smi during each task")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512, help="Per-response token limit, identical across all conditions")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--warmup-requests", type=int, default=2, help="Held-out warm-up requests before measured tasks; must be at least 1")
    parser.add_argument("--expected-context-size", type=int, default=8192, help="Required per-slot context from /props; one server slot is required")
    parser.add_argument("--settings-lock", default=str(ROOT / "results" / "experiment_settings.json"),
                        help="Create once, then require the same model-independent settings for every precision and condition")
    parser.add_argument("--max-calls", type=int, default=None, help="Fixed model response-turn cap, including finish/invalid responses; otherwise normal horizon plus shared extra turns")
    parser.add_argument("--recovery-budget", type=int, default=3, help="Shared extra response turns above normal horizon, identical in both recovery branches (default: 3)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--horizon", type=int, choices=[1, 3, 5], default=None,
                        help="Select the required remaining horizon before applying --limit")
    parser.add_argument("--task-id", default=None,
                        help="Select one exact task ID; must agree with the selected task type and horizon")
    return parser.parse_args()


def select_tasks(tasks: list[Task], *, task_type: str, horizon: int | None = None,
                 task_id: str | None = None, limit: int | None = None) -> list[Task]:
    """Filter an already validated full dataset, preserving its original order."""
    if task_type not in {"all", "execution", "recovery"}:
        raise SystemExit("Invalid --task-type")
    if horizon is not None and (type(horizon) is not int or horizon not in {1, 3, 5}):
        raise SystemExit("--horizon must be 1, 3 or 5")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise SystemExit("--limit must be a positive integer")
    selected = list(tasks)
    if task_id is not None:
        selected = [task for task in selected if task.task_id == task_id]
        if not selected:
            raise SystemExit(f"Unknown --task-id: {task_id}")
    if task_type != "all":
        selected = [task for task in selected if task.task_type == task_type]
    if horizon is not None:
        selected = [task for task in selected if task.horizon == horizon]
    if not selected:
        raise SystemExit("No tasks match --task-type, --horizon and --task-id. "
                         "Execution and recovery IDs are different; no automatic ID conversion is performed.")
    return selected[:limit] if limit is not None else selected


def main() -> int:
    args = parse_args()
    if args.condition == "recovery":
        args.condition = "error_restart"
    if args.ram_sample_interval_ms <= 0:
        raise SystemExit("--ram-sample-interval-ms must be greater than zero")
    if (not math.isfinite(args.temperature) or args.temperature < 0 or args.max_tokens < 1
            or not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0
            or not 0 <= args.seed < 2 ** 32 - 1 or args.warmup_requests < 1
            or args.expected_context_size < 1 or (args.limit is not None and args.limit < 1)):
        raise SystemExit("Invalid inference settings: finite temperature >= 0, max-tokens/context/timeout/limit > 0, "
                         "warmup >= 1, and seed 0..4294967294 are required.")
    if args.output and Path(args.output).resolve() == Path(args.tasks).resolve():
        raise SystemExit("--output must not overwrite the input dataset")
    if Path(args.settings_lock).resolve() == Path(args.tasks).resolve():
        raise SystemExit("--settings-lock must not be the input dataset")
    try:
        tasks, validation = load_validated_tasks(args.tasks)
    except DatasetValidationError as exc:
        print(json.dumps(exc.report, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    task_type = args.task_type
    if task_type is None:
        task_type = "execution" if args.condition == "free" else "recovery"
    tasks = select_tasks(tasks, task_type=task_type, horizon=args.horizon, task_id=args.task_id, limit=args.limit)
    if args.recovery_budget < 0 or (args.max_calls is not None and args.max_calls < 1):
        raise SystemExit("Call budgets must be positive (extra budget may be zero).")
    if any((args.condition == "free" and task.task_type != "execution") or
           (args.condition == "error_restart" and task.schema_version != "recovery-v2") for task in tasks):
        raise SystemExit("free requires execution tasks; error_restart requires recovery-v2 tasks.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(args.output or ROOT / "results" / f"{args.model_label}_{args.condition}_{timestamp}.jsonl")
    report_path = output.with_suffix(".preflight.json")
    lock_path = Path(args.settings_lock)
    if (output.resolve() == lock_path.resolve() or report_path.resolve() == lock_path.resolve()
            or report_path.resolve() == Path(args.tasks).resolve()):
        raise SystemExit("Task output, preflight report, settings lock and input dataset must be different files.")
    if output.exists() or report_path.exists():
        raise SystemExit("Output/preflight report already exists. Choose a new --output; existing files are never overwritten.")
    if args.model_path and not Path(args.model_path).is_file():
        raise SystemExit("--model-path must point to an existing local model file")
    metadata = collect_metadata(
        model_label=args.model_label,
        model_path=args.model_path,
        base_url=args.base_url,
        seed=args.seed,
        tasks_path=args.tasks,
        server_pid=args.server_pid,
        server_ram_sample_interval_ms=args.ram_sample_interval_ms,
    )
    if args.gpu_index is not None and args.gpu_index < 0:
        raise SystemExit("--gpu-index must be a nonnegative integer")
    metadata.update({"condition": args.condition, "task_type": task_type, "gpu_index": args.gpu_index,
                     "max_calls": args.max_calls, "shared_extra_action_turns": args.recovery_budget,
                     "dataset_language": validation["language"],
                     "dataset_version": validation["dataset_version"],
                     "dataset_sha256": validation["dataset_sha256"],
                     "tasks_sha256": validation["tasks_file_sha256"],
                     "task_selection": {"task_type": task_type, "horizon": args.horizon,
                                        "task_id": args.task_id, "limit": args.limit,
                                        "selected_count": len(tasks),
                                        "selected_task_ids": [task.task_id for task in tasks]},
                     "dataset_validation": {"valid": True, "validator_version": validation["validator_version"]}})
    client = LlamaCppClient(args.base_url, args.api_model, seed=args.seed, temperature=args.temperature,
                            max_tokens=args.max_tokens, timeout_seconds=args.timeout_seconds)
    metadata["api_model"] = args.api_model
    print(f"Checking server/settings lock, then warming up with {args.warmup_requests} held-out requests...", file=sys.stderr, flush=True)
    try:
        metadata.update(prepare_run(
            client, expected_context_size=args.expected_context_size, warmup_requests=args.warmup_requests,
            settings_lock_path=lock_path, report_path=report_path,
            hardware={field: metadata[field] for field in ("hostname", "platform", "processor", "machine")},
        ))
    except (PreparationError, InfrastructureError, OSError) as exc:
        print(json.dumps({"ready": False, "error": str(exc), "preflight_report": str(report_path),
                          "task_results_created": False}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
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
        gpu_index=args.gpu_index,
    )
    scores = [record["score"] for record in records]
    print(json.dumps({"output": str(output), "selected_count": len(tasks),
                      "selected_task_ids": [task.task_id for task in tasks],
                      "summary": aggregate(scores)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

