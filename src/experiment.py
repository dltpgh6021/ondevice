"""Batch execution and reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from .agent import run_agent
from .environment import MockEnvironment
from .metrics import score_trace
from .tasks import Task, stable_hash


def sha256_file(path: str | Path) -> str | None:
    if not path:
        return None
    source = Path(path)
    if not source.exists() or not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_ram_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 3)
    except Exception:
        return None


def collect_metadata(
    *,
    model_label: str,
    model_path: str | None,
    base_url: str | None,
    seed: int,
    tasks_path: str | Path,
) -> dict[str, Any]:
    return {
        "model_label": model_label,
        "model_path": model_path,
        "model_sha256": sha256_file(model_path) if model_path else None,
        "base_url": base_url,
        "seed": seed,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "llama_cpp_version": None,
        "tasks_sha256": sha256_file(tasks_path),
        "created_at_unix": time.time(),
    }


def run_batch(
    tasks: list[Task],
    client: Any,
    *,
    model_label: str,
    condition: str,
    output_path: str | Path,
    metadata: dict[str, Any],
    max_calls: int | None,
    recovery_budget: int = 3,
) -> list[dict[str, Any]]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            environment = MockEnvironment()
            required_calls = task.horizon
            if condition != "free":
                required_calls = max(0, task.horizon - task.checkpoint_step)
            allowed_calls = max_calls if max_calls is not None else required_calls + recovery_budget
            trace = run_agent(
                task,
                client,
                environment,
                condition=condition,
                max_calls=allowed_calls,
            )
            trace["max_calls"] = allowed_calls
            trace["max_ram_mb"] = current_ram_mb()
            score = score_trace(task, trace)
            score["model_label"] = model_label
            record = {
                "metadata": metadata,
                "task": {
                    "id": task.task_id,
                    "base_id": task.base_id,
                    "domain": task.domain,
                    "task_type": task.task_type,
                    "horizon": task.horizon,
                    "checkpoint_step": task.checkpoint_step,
                    "query": task.query,
                    "recovery_query": task.recovery_query,
                    "error_description": task.error_description,
                    "error_injection": task.error_injection,
                },
                "trace": trace,
                "score": score,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
    return records


def load_records(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]

