"""Task loading, validation, and stable JSON helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Task":
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
            error_injection=data.get("error_injection"),
            success_condition=dict(data["success_condition"]),
        )
        validate_task(task)
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
    if task.task_type == "recovery" and not task.error_injection:
        raise ValueError(f"{task.task_id}: recovery task requires error_injection")
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
    return [Task.from_mapping(item) for item in data]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

