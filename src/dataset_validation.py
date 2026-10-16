"""One offline gate for syntax, schema, language and task semantics.

Bounded trajectory probes do not prove every possible strategy. ASCII checks
do not prove English prose correctness. Canonical source matching protects
the reviewed wording from accidental edits to generated JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

from .dataset_spec import (DATASET_VERSION, ERROR_KINDS, EXECUTION_HORIZONS, EXECUTION_TASKS, RECOVERY_CONDITIONS,
                           RECOVERY_HORIZONS, RECOVERY_TASKS, BASE_SCENARIOS, TOTAL_TASKS)
from .environment import environment_for_task
from .execution_environment import WORKFLOWS, resource_id
from .execution_validation import validate_execution_tasks
from .recovery_validation import validate_recovery_tasks
from .tasks import Task, canonical_json, stable_hash, validate_task_set

VALIDATOR_VERSION = "dataset-validator-v4"


class DatasetValidationError(ValueError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.report = {"valid": False, "validator_version": VALIDATOR_VERSION,
                       "stage": stage, "errors": [message]}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _type(value: Any, expected: type, path: str) -> None:
    _require(type(value) is expected, f"{path}: expected {expected.__name__}, got {type(value).__name__}")


def _text(value: Any, path: str) -> None:
    _type(value, str, path)
    _require(bool(value.strip()), f"{path}: must not be empty")


def _fields(value: Any, required: set[str], optional: set[str], path: str) -> None:
    _type(value, dict, path)
    _require(not (required - value.keys()), f"{path}: missing fields {sorted(required - value.keys())}")
    _require(not (value.keys() - required - optional), f"{path}: unsupported fields {sorted(value.keys() - required - optional)}")


def _numeric(value: Any, path: str) -> None:
    _require(type(value) is int or (type(value) is float and math.isfinite(value)),
             f"{path}: expected a finite number, not a boolean")


def _action(value: Any, path: str) -> None:
    _fields(value, {"tool", "arguments"}, set(), path)
    _text(value["tool"], path + ".tool")
    _type(value["arguments"], dict, path + ".arguments")
    for key, argument in value["arguments"].items():
        _text(key, path + ".arguments key")
        _text(argument, path + ".arguments." + key)


def _completion(value: Any, path: str) -> None:
    _fields(value, {"request_id", "domain", "candidate_id", "option_id", "result"}, set(), path)
    for key in ("request_id", "domain", "candidate_id", "option_id"):
        _text(value[key], path + "." + key)
    _type(value["result"], dict, path + ".result")
    _require(bool(value["result"]), path + ".result: must not be empty")
    for key, item in value["result"].items():
        _text(key, path + ".result key")
        if type(item) is str:
            _text(item, path + ".result." + key)
        else:
            _numeric(item, path + ".result." + key)


def _schema(rows: Any) -> None:
    _type(rows, list, "dataset")
    common = {"id", "base_id", "domain", "task_type", "schema_version", "language", "dataset_version",
              "horizon", "query", "initial_actions", "gold_actions", "checkpoint_step", "success_condition", "execution_spec"}
    for index, row in enumerate(rows):
        path = f"tasks[{index}]"
        _type(row, dict, path)
        is_recovery = row.get("task_type") == "recovery"
        recovery = {"error_variants", "normal_remaining_calls"} if is_recovery else set()
        _fields(row, common | recovery, {"error_description", "recovery_query"}, path)
        for key in ("id", "base_id", "domain", "task_type", "schema_version", "language", "dataset_version", "query"):
            _text(row[key], path + "." + key)
        _require(row["task_type"] in {"execution", "recovery"}, path + ": unsupported task_type")
        _require(row["schema_version"] == ("recovery-v3" if is_recovery else "execution-v2"), path + ": unsupported schema_version")
        _require(row["domain"] in WORKFLOWS, path + ": unsupported domain")
        _require(row["language"] == "en" and row["dataset_version"] == DATASET_VERSION, path + ": expected English dataset " + DATASET_VERSION)
        _type(row["horizon"], int, path + ".horizon")
        _type(row["checkpoint_step"], int, path + ".checkpoint_step")
        _require(row["horizon"] in (set(RECOVERY_HORIZONS) if is_recovery else set(EXECUTION_HORIZONS)),
                 path + (": recovery horizon must be 1 or 3" if is_recovery else ": horizon must be 1, 3 or 5"))
        _require(0 <= row["checkpoint_step"] < row["horizon"], path + ": checkpoint must leave unfinished work")
        for key in ("initial_actions", "gold_actions"):
            _type(row[key], list, path + "." + key)
            for number, action in enumerate(row[key]):
                _action(action, f"{path}.{key}[{number}]")
        for key in ("error_description", "recovery_query"):
            if key in row and row[key] is not None:
                _text(row[key], path + "." + key)
        _require(not row.get("recovery_query"), path + ": use the unchanged original question, not corrective instructions")
        _fields(row["success_condition"], {"completed_request_equals"}, set(), path + ".success_condition")
        _completion(row["success_condition"]["completed_request_equals"], path + ".success_condition.completed_request_equals")
        if is_recovery:
            _type(row["normal_remaining_calls"], int, path + ".normal_remaining_calls")
            variants = row["error_variants"]
            _type(variants, dict, path + ".error_variants")
            _require(tuple(variants) == ERROR_KINDS, path + f": error_variants must be exactly {list(ERROR_KINDS)} in order")
            for kind, variant in variants.items():
                where = f"{path}.error_variants.{kind}"
                _fields(variant, {"error_injection", "recovery_actions", "recovery_min_calls", "error_description"}, set(), where)
                _type(variant["recovery_min_calls"], int, where + ".recovery_min_calls")
                _text(variant["error_description"], where + ".error_description")
                _type(variant["recovery_actions"], list, where + ".recovery_actions")
                for number, action in enumerate(variant["recovery_actions"]):
                    _action(action, f"{where}.recovery_actions[{number}]")
                injection = variant["error_injection"]
                _fields(injection, {"type", "error_kind", "actions"}, set(), where + ".error_injection")
                _require(injection["type"] == "wrong_saved_result" and injection["error_kind"] == kind,
                         where + ": unsupported or mislabeled error injection")
                _type(injection["actions"], list, where + ".error_injection.actions")
                _require(bool(injection["actions"]), where + ": error injection needs actions")
                for number, action in enumerate(injection["actions"]):
                    _action(action, f"{where}.error_injection.actions[{number}]")
        spec = row["execution_spec"]
        _fields(spec, {"domain", "request_id", "request", "candidates"}, set(), path + ".execution_spec")
        _require(spec["domain"] == row["domain"], path + ": fixture/task domain mismatch")
        _text(spec["request_id"], path + ".execution_spec.request_id")
        request = spec["request"]
        _fields(request, {"description", "constraints", "ranking", "ranking_note", "required_option_kind", "result_fields"}, set(), path + ".request")
        for key in ("description", "ranking_note", "required_option_kind"):
            _text(request[key], path + ".request." + key)
        _type(request["constraints"], dict, path + ".request.constraints")
        _require(bool(request["constraints"]), path + ": constraints must not be empty")
        for key, value in request["constraints"].items():
            _require(key.startswith(("min_", "max_")), path + ": constraint must begin with min_ or max_")
            _numeric(value, path + ".request.constraints." + key)
        _type(request["ranking"], list, path + ".request.ranking")
        _require(bool(request["ranking"]), path + ": ranking must not be empty")
        for rule in request["ranking"]:
            _fields(rule, {"field", "direction"}, set(), path + ".request.ranking")
            _text(rule["field"], path + ".request.ranking.field")
            _require(rule["direction"] in {"min", "max"}, path + ": invalid ranking direction")
        _type(request["result_fields"], dict, path + ".request.result_fields")
        _require(bool(request["result_fields"]), path + ": result_fields must not be empty")
        for key, value in request["result_fields"].items():
            if type(value) is str:
                _text(value, path + ".request.result_fields." + key)
            else:
                _numeric(value, path + ".request.result_fields." + key)
        if "date" in request["result_fields"]:
            value = request["result_fields"]["date"]
            _text(value, path + ".request.result_fields.date")
            _require(date.fromisoformat(value).isoformat() == value, path + ": use an ISO YYYY-MM-DD date")
        _type(spec["candidates"], list, path + ".candidates")
        for candidate in spec["candidates"]:
            _fields(candidate, {"candidate_id", "summary", "options"}, set(), path + ".candidate")
            _text(candidate["candidate_id"], path + ".candidate.candidate_id")
            _type(candidate["summary"], dict, path + ".candidate.summary")
            _text(candidate["summary"].get("label"), path + ".candidate.summary.label")
            for key, value in candidate["summary"].items():
                if key != "label":
                    _numeric(value, path + ".candidate.summary." + key)
            _type(candidate["options"], list, path + ".candidate.options")
            for option in candidate["options"]:
                _fields(option, {"option_id", "kind", "available"}, set(), path + ".option")
                _text(option["option_id"], path + ".option.option_id")
                _text(option["kind"], path + ".option.kind")
                _type(option["available"], bool, path + ".option.available")
        if "initial_drafts" in spec:
            _type(spec["initial_drafts"], dict, path + ".initial_drafts")
            for key, value in spec["initial_drafts"].items():
                _text(key, path + ".initial_drafts key")
                _completion(value, path + ".initial_drafts." + key)


def _ascii(value: Any, path: str) -> int:
    if isinstance(value, str):
        _require(value.isascii(), f"{path}: non-ASCII text found; supplied model data must use English ASCII text")
        return 1
    if isinstance(value, list):
        return sum(_ascii(item, f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(value, dict):
        return sum(_ascii(key, path + " key") + _ascii(item, path + "." + str(key)) for key, item in value.items())
    return 0


def _information(tasks: list[Task]) -> None:
    requests: dict[str, str] = {}
    fixtures: dict[str, str] = {}
    for task in tasks:
        spec = task.execution_spec
        _require(task.query.count(spec["request_id"]) == 1, f"{task.task_id}: question must contain its request_id exactly once")
        forbidden = set()
        for candidate in spec["candidates"]:
            candidate_id = candidate["candidate_id"]
            forbidden.update((candidate_id, resource_id(spec, "offer", candidate_id), resource_id(spec, "detail", candidate_id)))
            for option in candidate["options"]:
                forbidden.update((option["option_id"], resource_id(spec, "draft", candidate_id, option["option_id"])))
        forbidden.add(resource_id(spec, "context"))
        forbidden.update(spec.get("initial_drafts", {}))
        forbidden.update(value["request_id"] for value in spec.get("initial_drafts", {}).values())
        _require(not any(identifier in task.query for identifier in forbidden), f"{task.task_id}: original question leaks a candidate/option/workflow answer identifier")
        if task.task_type == "execution":
            for mapping, key, label in ((requests, spec["request_id"], "request_id"), (fixtures, stable_hash(spec), "fixture")):
                _require(key not in mapping or mapping[key] == task.base_id,
                         f"{task.task_id}: duplicated {label} across base scenarios {mapping.get(key)} and {task.base_id}")
                mapping[key] = task.base_id


def _model_payloads(tasks: list[Task]) -> dict[str, int]:
    strings = 0
    continuations = 0
    for task in tasks:
        conditions = ("free",) if task.task_type == "execution" else RECOVERY_CONDITIONS
        if task.task_type == "execution":
            # This optional diagnostic intentionally replays extra normal work.
            diagnostic = environment_for_task(task)
            diagnostic.prepare_condition(task, "normal_restart")
            _require(not diagnostic.check_success(task.success_condition), f"{task.task_id}: diagnostic starts complete")
            for action in task.gold_actions[task.checkpoint_step:]:
                _require(diagnostic.execute(action["tool"], action["arguments"])["ok"], f"{task.task_id}: invalid diagnostic action")
            _require(diagnostic.check_success(task.success_condition), f"{task.task_id}: diagnostic cannot complete")
        for condition in conditions:
            kinds = ERROR_KINDS if condition not in {"free", "normal_restart"} else (None,)
            for kind in kinds:
                resolved = task.with_error_kind(kind) if kind else task
                environment = environment_for_task(resolved)
                messages, _ = environment.prepare_condition(resolved, condition)
                strings += _ascii(environment.tool_definitions(), task.task_id + ".tools")
                strings += _ascii(messages, task.task_id + ".messages")
                actions = resolved.gold_actions if kind is None else resolved.recovery_actions
                for action in actions:
                    strings += _ascii(environment.execute(action["tool"], action["arguments"]), task.task_id + ".tool_result")
                continuations += 1
    return {"continuations_checked": continuations, "ascii_string_fields_checked": strings}


def _condition_separation(tasks: list[Task]) -> dict[str, Any]:
    """free and normal_restart hold the same information but must differ in delivery.

    Recovery fixtures exist only where earlier results are supplied (h1, h3), so
    no matched pair may present identical model inputs.
    """
    by_id = {task.task_id: task for task in tasks}
    identical: dict[str, int] = {"1": 0, "3": 0}
    for recovery in tasks:
        if recovery.task_type != "recovery":
            continue
        task = by_id[recovery.base_id]
        free_env, normal_env = environment_for_task(task), environment_for_task(recovery)
        free_messages, _ = free_env.prepare_condition(task, "free")
        normal_messages, _ = normal_env.prepare_condition(recovery, "normal_restart")
        same = canonical_json([free_messages, free_env.tool_definitions()]) == canonical_json(
            [normal_messages, normal_env.tool_definitions()])
        _require(not same, f"{task.task_id}: free and normal_restart present identical model inputs")
        _require(free_env.snapshot() == normal_env.snapshot(),
                 f"{task.task_id}: free and normal_restart start with different information")
        identical[str(task.horizon)] += int(same)
    return {"identical_free_normal_inputs_by_horizon": identical,
            "note": "matched free and normal_restart share the checkpoint state and differ only in delivery; h5 has no recovery fixture"}


def validate_dataset_records(rows: Any) -> tuple[list[Task], dict[str, Any]]:
    """Validate a full generated dataset without filesystem/network/model access."""
    stage = "schema"
    try:
        _schema(rows)
        tasks = [Task.from_mapping(row) for row in rows]
        collection = validate_task_set(tasks, expected_total=TOTAL_TASKS,
                                       expected_counts={"execution": EXECUTION_TASKS, "recovery": RECOVERY_TASKS},
                                       required_horizons=set(EXECUTION_HORIZONS), expected_per_type_horizon=BASE_SCENARIOS,
                                       require_pairs=True, recovery_horizons=set(RECOVERY_HORIZONS))
        stage = "language_and_information"
        _ascii(rows, "dataset")
        _information(tasks)
        stage = "execution_semantics"
        execution = validate_execution_tasks(tasks)
        stage = "recovery_semantics"
        recovery = validate_recovery_tasks(tasks)
        stage = "model_payloads"
        payloads = _model_payloads(tasks)
        separation = _condition_separation(tasks)
        stage = "source_integrity"
        from scripts.build_execution_tasks import build_execution_tasks
        from scripts.build_recovery_tasks import build_recovery_tasks

        expected_execution = build_execution_tasks()
        expected = expected_execution + build_recovery_tasks(expected_execution)
        expected_by_id = {row["id"]: row for row in expected}
        for row in rows:
            _require(canonical_json(row) == canonical_json(expected_by_id.get(row["id"])), f"{row['id']}: differs from the reviewed scenario definitions; edit generators and regenerate instead of editing tasks.json")
        _require(canonical_json(rows) == canonical_json(expected), "Task ordering differs from the canonical generated dataset.")
        return tasks, {"valid": True, "validator_version": VALIDATOR_VERSION, "language": "en", "dataset_version": DATASET_VERSION,
                       "dataset_sha256": stable_hash(rows), **collection, "errors": [],
                       "source_integrity": {"matches_generators": True}, "model_input_audit": payloads,
                       "condition_separation": separation,
                       "execution_semantics": execution, "recovery_semantics": recovery,
                       "limitations": ["Bounded reference trajectory probes, not exhaustive search over all actions.",
                                       "ASCII checks and canonical wording do not prove natural-language correctness; human review is still required."]}
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        raise DatasetValidationError(stage, str(exc)) from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _no_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def load_validated_tasks(path: str | Path) -> tuple[list[Task], dict[str, Any]]:
    source = Path(path)
    try:
        raw = source.read_bytes()
        rows = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object, parse_constant=_no_constant)
    except (OSError, UnicodeError, ValueError) as exc:
        raise DatasetValidationError("json", str(exc)) from exc
    tasks, report = validate_dataset_records(rows)
    report["tasks_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return tasks, report


def validate_dataset(path: str | Path) -> dict[str, Any]:
    return load_validated_tasks(path)[1]
