"""Task loading, validation, and stable JSON helpers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Task:
    task_id: str
    base_id: str
    domain: str
    task_type: str
    horizon: int
    query: str
    recovery_query: str | None
    error_description: str | None
    gold_actions: list[dict[str, Any]]
    checkpoint_step: int
    error_injection: dict[str, Any] | None
    success_condition: dict[str, Any]
    schema_version: str = "legacy-v1"
    initial_actions: list[dict[str, Any]] | None = None
    execution_spec: dict[str, Any] | None = None
    recovery_actions: list[dict[str, Any]] | None = None
    normal_remaining_calls: int | None = None
    recovery_min_calls: int | None = None
    language: str = "unspecified"
    dataset_version: str = "legacy"
    # recovery-v3: every error family at the shared checkpoint. The flat
    # error_injection / recovery_actions / recovery_min_calls fields hold the
    # currently selected family (error_kind); see with_error_kind().
    error_kind: str | None = None
    error_variants: dict[str, Any] | None = None

    def with_error_kind(self, kind: str) -> "Task":
        """The same fixture resolved to one error family."""
        if not self.error_variants:
            if kind != (self.error_kind or kind):
                raise ValueError(f"{self.task_id}: no error variant {kind}")
            return self
        if kind not in self.error_variants:
            raise ValueError(f"{self.task_id}: unknown error kind {kind}; available {sorted(self.error_variants)}")
        variant = self.error_variants[kind]
        return replace(self, error_kind=kind, error_injection=variant["error_injection"],
                       recovery_actions=list(variant["recovery_actions"]),
                       recovery_min_calls=int(variant["recovery_min_calls"]))

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Task":
        variants = data.get("error_variants")
        kind = None
        selected: dict[str, Any] = {}
        if isinstance(variants, dict) and variants:
            from .dataset_spec import DEFAULT_ERROR_KIND

            kind = DEFAULT_ERROR_KIND if DEFAULT_ERROR_KIND in variants else sorted(variants)[0]
            selected = variants[kind]
        task = cls(
            task_id=str(data["id"]),
            base_id=str(data.get("base_id", data["id"])),
            domain=str(data.get("domain", "generic")),
            task_type=str(data.get("task_type", "execution")),
            horizon=int(data["horizon"]),
            query=str(data["query"]),
            recovery_query=str(data["recovery_query"]) if data.get("recovery_query") else None,
            error_description=str(data["error_description"]) if data.get("error_description") else None,
            gold_actions=list(data["gold_actions"]),
            checkpoint_step=int(data.get("checkpoint_step", 0)),
            error_injection=selected.get("error_injection", data.get("error_injection")),
            success_condition=dict(data["success_condition"]),
            schema_version=str(data.get("schema_version", "legacy-v1")),
            initial_actions=list(data.get("initial_actions", [])),
            execution_spec=data.get("execution_spec"),
            recovery_actions=list(selected.get("recovery_actions", data.get("recovery_actions", []))),
            normal_remaining_calls=int(data["normal_remaining_calls"]) if "normal_remaining_calls" in data else None,
            recovery_min_calls=(int(selected["recovery_min_calls"]) if "recovery_min_calls" in selected
                                else int(data["recovery_min_calls"]) if "recovery_min_calls" in data else None),
            language=str(data.get("language", "unspecified")),
            dataset_version=str(data.get("dataset_version", "legacy")),
            error_kind=kind,
            error_variants=variants if isinstance(variants, dict) and variants else None,
        )
        validate_task(task)
        for other in sorted(task.error_variants or {}):
            validate_task(task.with_error_kind(other))
        return task


def validate_task(task: Task) -> None:
    if not task.task_id:
        raise ValueError("task id must not be empty")
    if task.task_type not in {"execution", "recovery"}:
        raise ValueError(f"{task.task_id}: task_type must be execution or recovery")
    if task.horizon != len(task.gold_actions):
        raise ValueError(
            f"{task.task_id}: horizon={task.horizon} but "
            f"gold_actions={len(task.gold_actions)}"
        )
    if not 0 <= task.checkpoint_step <= task.horizon:
        raise ValueError(f"{task.task_id}: invalid checkpoint_step")
    if not task.success_condition:
        raise ValueError(f"{task.task_id}: success_condition is required")
    if task.schema_version == "execution-v2":
        if task.task_type != "execution" or not task.execution_spec:
            raise ValueError(f"{task.task_id}: execution-v2 requires an execution fixture")
        if task.error_injection or task.recovery_query:
            raise ValueError(f"{task.task_id}: execution task must not inject an error")
        # initial_actions are user-message seeds: none at h5, at least one otherwise.
        if bool(task.initial_actions) != (task.horizon < 5):
            raise ValueError(f"{task.task_id}: only shorter horizons may seed provided context")
    if task.task_type == "recovery" and not task.error_injection:
        raise ValueError(f"{task.task_id}: recovery task requires error_injection")
    if task.schema_version == "recovery-v3":
        if task.task_type != "recovery" or not task.execution_spec or task.recovery_query:
            raise ValueError(f"{task.task_id}: recovery-v3 requires a fixture and the original query only")
        # h5 has no checkpoint history, so recovery fixtures exist only for h1 and h3.
        if task.horizon not in {1, 3} or not task.initial_actions or task.checkpoint_step != 0:
            raise ValueError(f"{task.task_id}: invalid recovery checkpoint")
        if task.normal_remaining_calls != task.horizon:
            raise ValueError(f"{task.task_id}: normal remaining calls must equal the horizon")
        if not task.error_variants or not task.error_kind:
            raise ValueError(f"{task.task_id}: recovery-v3 requires error_variants")
        if (task.error_injection or {}).get("error_kind") != task.error_kind:
            raise ValueError(f"{task.task_id}: error variant label differs from its injection")
        if not isinstance(task.recovery_min_calls, int) or not 1 <= task.recovery_min_calls <= task.horizon:
            raise ValueError(f"{task.task_id}: repair minimum must be between 1 and the horizon")
        if len(task.recovery_actions or []) != task.recovery_min_calls:
            raise ValueError(f"{task.task_id}: recovery oracle length differs from recovery_min_calls")
    for index, action in enumerate(task.gold_actions):
        if not isinstance(action, dict) or not action.get("tool"):
            raise ValueError(f"{task.task_id}: invalid gold action at {index}")
        if "arguments" in action and not isinstance(action["arguments"], dict):
            raise ValueError(f"{task.task_id}: arguments must be an object at {index}")


def load_tasks(path: str | Path) -> list[Task]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("task file must contain a JSON list")
    tasks = [Task.from_mapping(item) for item in data]
    validate_task_set(tasks)
    return tasks


def validate_task_set(
    tasks: list[Task],
    *,
    expected_total: int | None = None,
    expected_counts: dict[str, int] | None = None,
    required_horizons: set[int] | None = None,
    expected_per_type_horizon: int | None = None,
    require_pairs: bool = False,
    recovery_horizons: set[int] | None = None,
) -> dict[str, Any]:
    """Validate collection-level integrity and return count diagnostics."""
    errors: list[str] = []
    ids = [task.task_id for task in tasks]
    duplicate_ids = sorted(task_id for task_id, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate task ids: {', '.join(duplicate_ids)}")

    if expected_total is not None and len(tasks) != expected_total:
        errors.append(f"expected {expected_total} tasks, found {len(tasks)}")

    by_type = Counter(task.task_type for task in tasks)
    if expected_counts:
        for task_type, expected in expected_counts.items():
            found = by_type.get(task_type, 0)
            if found != expected:
                errors.append(f"expected {expected} {task_type} tasks, found {found}")

    by_horizon = Counter(task.horizon for task in tasks)
    if required_horizons:
        missing_horizons = sorted(required_horizons - set(by_horizon))
        if missing_horizons:
            errors.append(f"missing horizons: {', '.join(map(str, missing_horizons))}")

    by_type_horizon = Counter((task.task_type, task.horizon) for task in tasks)
    if expected_per_type_horizon is not None and required_horizons:
        for task_type in ("execution", "recovery"):
            for horizon in sorted(required_horizons):
                found = by_type_horizon.get((task_type, horizon), 0)
                expected = expected_per_type_horizon
                if task_type == "recovery" and recovery_horizons is not None and horizon not in recovery_horizons:
                    expected = 0
                if found != expected:
                    errors.append(
                        f"expected {expected} {task_type} tasks at horizon {horizon}, found {found}"
                    )

    if require_pairs:
        execution_by_id = {
            task.task_id: task for task in tasks if task.task_type == "execution"
        }
        recovery_tasks = [task for task in tasks if task.task_type == "recovery"]
        for recovery in recovery_tasks:
            execution = execution_by_id.get(recovery.base_id)
            if execution is None:
                errors.append(
                    f"recovery task {recovery.task_id} has no execution pair for base_id={recovery.base_id}"
                )
                continue
            if recovery.horizon != execution.horizon:
                errors.append(
                    f"pair horizon mismatch: {execution.task_id}={execution.horizon}, "
                    f"{recovery.task_id}={recovery.horizon}"
                )
            if recovery.domain != execution.domain:
                errors.append(
                    f"pair domain mismatch: {execution.task_id}={execution.domain}, "
                    f"{recovery.task_id}={recovery.domain}"
                )
        recovery_base_ids = {task.base_id for task in recovery_tasks}
        for execution in execution_by_id.values():
            if recovery_horizons is not None and execution.horizon not in recovery_horizons:
                continue
            if execution.task_id not in recovery_base_ids:
                errors.append(f"execution task {execution.task_id} has no recovery pair")

    if errors:
        raise ValueError("task set validation failed:\n- " + "\n- ".join(errors))

    return {
        "total": len(tasks),
        "by_task_type": dict(sorted(by_type.items())),
        "by_horizon": {str(key): value for key, value in sorted(by_horizon.items())},
        "by_task_type_horizon": {
            f"{task_type}/h{horizon}": count
            for (task_type, horizon), count in sorted(by_type_horizon.items())
        },
        "duplicate_ids": duplicate_ids,
        "paired": require_pairs,
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

