"""Batch execution and reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .agent import run_agent, METRICS_VERSION, AGENT_POLICY_VERSION
from .environment import environment_for_task
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


class ProcessRamMonitor:
    """Sample a separate process RSS while one task is running."""

    def __init__(self, pid: int | None, interval_seconds: float = 0.1) -> None:
        self.pid = pid
        self.interval_seconds = max(0.01, interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_mb: float | None = None

    def _sample_once(self) -> None:
        if self.pid is None:
            return
        try:
            import psutil  # type: ignore

            process = psutil.Process(self.pid)
            rss_mb = process.memory_info().rss / (1024 * 1024)
            self._peak_mb = max(self._peak_mb or 0.0, rss_mb)
        except Exception:
            # The server may have exited or the PID may be stale. Keep the
            # result null instead of failing the whole experiment.
            return

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        if self.pid is None:
            return
        self._sample_once()
        self._thread = threading.Thread(target=self._run, name="server-ram-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        if self._thread is None:
            return None
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 5))
        self._sample_once()
        return round(self._peak_mb, 3) if self._peak_mb is not None else None


def collect_metadata(
    *,
    model_label: str,
    model_path: str | None,
    base_url: str | None,
    seed: int,
    tasks_path: str | Path,
    server_pid: int | None = None,
    server_ram_sample_interval_ms: int = 100,
) -> dict[str, Any]:
    return {
        "model_label": model_label,
        "model_path": model_path,
        "model_sha256": sha256_file(model_path) if model_path else None,
        "base_url": base_url,
        "server_pid": server_pid,
        "server_ram_sample_interval_ms": server_ram_sample_interval_ms,
        "seed": seed,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
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
    server_pid: int | None = None,
    server_ram_sample_interval_ms: int = 100,
) -> list[dict[str, Any]]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    metadata = {**metadata, "metrics_version": METRICS_VERSION,
                "agent_policy_version": AGENT_POLICY_VERSION}
    with output.open("x", encoding="utf-8") as handle:
        for task in tasks:
            environment = environment_for_task(task)
            required_calls = task.horizon
            if task.schema_version == "recovery-v2":
                required_calls = task.normal_remaining_calls
            elif condition != "free":
                required_calls = max(0, task.horizon - task.checkpoint_step)
            allowed_calls = max_calls if max_calls is not None else required_calls + recovery_budget
            server_ram_monitor = ProcessRamMonitor(
                server_pid,
                interval_seconds=server_ram_sample_interval_ms / 1000,
            )
            server_ram_monitor.start()
            try:
                trace = run_agent(
                    task,
                    client,
                    environment,
                    condition=condition,
                    max_calls=allowed_calls,
                )
            finally:
                trace_server_peak_ram_mb = server_ram_monitor.stop()
            trace["max_calls"] = allowed_calls
            trace["budget_basis"] = "normal_remaining_calls + shared extra action turns"
            trace["max_ram_mb"] = current_ram_mb()
            trace["peak_server_ram_mb"] = trace_server_peak_ram_mb
            score = score_trace(task, trace)
            score["model_label"] = model_label
            record = {
                "metadata": metadata,
                "task": {
                    "id": task.task_id,
                    "base_id": task.base_id,
                    "domain": task.domain,
                    "task_type": task.task_type,
                    "schema_version": task.schema_version,
                    "language": task.language,
                    "dataset_version": task.dataset_version,
                    "initial_actions": task.initial_actions,
                    "horizon": task.horizon,
                    "checkpoint_step": task.checkpoint_step,
                    "query": task.query,
                    "recovery_query": task.recovery_query,
                    "error_description": task.error_description,
                    "error_injection": task.error_injection,
                    "success_condition": task.success_condition,
                    "normal_remaining_calls": task.normal_remaining_calls,
                    "recovery_min_calls": task.recovery_min_calls,
                    "recovery_actions": task.recovery_actions,
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

