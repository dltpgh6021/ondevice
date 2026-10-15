"""Check execution-v2 semantics and specific prerequisite bypass attempts.

This enumerates reference subsequences, not arbitrary model strategies.
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any

from .execution_environment import ExecutionEnvironment, WORKFLOWS, resource_id
from .tasks import Task


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, description: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{description} must be numeric")
    return float(value)


def _infer_target(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the public constraints and ranking without target markers."""
    request, candidates = spec["request"], spec["candidates"]
    constraints, ranking = request.get("constraints", {}), request.get("ranking", [])
    _require(isinstance(constraints, dict), "constraints must be an object")
    _require(isinstance(ranking, list) and bool(ranking), "ranking must be nonempty")
    _require(len(candidates) >= 2, "at least two candidates are required")
    _require(len({c["candidate_id"] for c in candidates}) == len(candidates), "duplicate candidate IDs")
    eligible = []
    for candidate in candidates:
        matches = True
        for rule, limit in constraints.items():
            _require(rule.startswith(("min_", "max_")), f"unsupported constraint: {rule}")
            value = _number(candidate["summary"].get(rule[4:]), f"candidate constraint {rule}")
            bound = _number(limit, f"constraint {rule}")
            matches = matches and (value >= bound if rule.startswith("min_") else value <= bound)
        if matches:
            eligible.append(candidate)
        options = candidate["options"]
        _require(len({o["option_id"] for o in options}) == len(options), "duplicate option IDs")
        _require(all(isinstance(o.get("available"), bool) for o in options), "availability must be boolean")
    _require(bool(eligible), "no candidate satisfies the constraints")

    def rank(candidate: dict[str, Any]) -> tuple[float, ...]:
        values = []
        for rule in ranking:
            _require(rule.get("direction") in {"min", "max"}, "ranking direction must be min/max")
            value = _number(candidate["summary"].get(rule["field"]), "ranking value")
            values.append(value if rule["direction"] == "min" else -value)
        return tuple(values)

    ranked = sorted(eligible, key=rank)
    _require(len(ranked) == 1 or rank(ranked[0]) != rank(ranked[1]), "ranking does not resolve a unique winner")
    selected = ranked[0]
    options = [o for o in selected["options"] if o["available"] and o["kind"] == request["required_option_kind"]]
    _require(len(options) == 1, "winner needs exactly one available requested option")
    return selected, options[0]


def _actions(spec: dict[str, Any], candidate: dict[str, Any], option: dict[str, Any]) -> list[dict[str, Any]]:
    arguments = (
        {"request_id": spec["request_id"]},
        {"context_id": resource_id(spec, "context")},
        {"offer_id": resource_id(spec, "offer", candidate["candidate_id"])},
        {"detail_id": resource_id(spec, "detail", candidate["candidate_id"]), "option_id": option["option_id"]},
        {"draft_id": resource_id(spec, "draft", candidate["candidate_id"], option["option_id"])},
    )
    return [{"tool": name, "arguments": args}
            for name, args in zip(WORKFLOWS[spec["domain"]]["names"], arguments)]


def _environment(task: Task) -> ExecutionEnvironment:
    environment = ExecutionEnvironment(task.execution_spec)
    environment.prepare_condition(task, "free")
    return environment


def validate_execution_tasks(tasks: list[Task]) -> dict[str, Any]:
    """Validate 20 matched scenarios with 1/3/5 remaining workflow calls."""
    execution = [t for t in tasks if t.task_type == "execution"]
    _require(len(execution) == 60, f"expected 60 execution tasks, found {len(execution)}")
    _require(all(t.schema_version == "execution-v2" for t in execution), "execution tasks must all use execution-v2")
    _require(len({t.task_id for t in execution}) == 60, "duplicate execution task IDs")
    by_base: dict[str, list[Task]] = defaultdict(list)
    for task in execution:
        by_base[task.base_id].append(task)
    _require(len(by_base) == 20, f"expected 20 base scenarios, found {len(by_base)}")
    domains = Counter(group[0].domain for group in by_base.values())
    _require(domains == Counter({d: 5 for d in WORKFLOWS}), "expected five base scenarios per domain")
    checks: Counter[str] = Counter()
    for base_id, variants in by_base.items():
        _require(Counter(t.horizon for t in variants) == Counter({1: 1, 3: 1, 5: 1}), f"{base_id}: missing matched horizons")
        reference = variants[0]
        full_chain = list(reference.initial_actions or []) + reference.gold_actions
        for task in variants:
            _require((task.query, task.execution_spec, task.success_condition, task.domain) ==
                     (reference.query, reference.execution_spec, reference.success_condition, reference.domain),
                     f"{base_id}: horizon variants differ in question, fixture, domain or goal")
            _require(list(task.initial_actions or []) + task.gold_actions == full_chain,
                     f"{base_id}: horizon variants do not partition the same chain")
            _require(not task.error_injection and not task.recovery_query, f"{task.task_id}: execution injects an error")
        checks["matched_scenarios"] += 1
        spec = reference.execution_spec
        _require(spec["domain"] == reference.domain, f"{base_id}: fixture domain differs")
        candidate, option = _infer_target(spec)
        _require(full_chain == _actions(spec, candidate, option), f"{base_id}: gold differs from inferred target")
        completion = {
            "request_id": spec["request_id"], "domain": spec["domain"],
            "candidate_id": candidate["candidate_id"], "option_id": option["option_id"],
            "result": {**spec["request"]["result_fields"], **candidate["summary"], "option_kind": option["kind"]},
        }
        _require(reference.success_condition == {"completed_request_equals": completion},
                 f"{base_id}: state grading differs from request rules")
        checks["unique_target_inference"] += 1
        for task in variants:
            try:
                env = _environment(task)
                _require(not env.check_success(task.success_condition), "task starts complete")
                for index, action in enumerate(task.gold_actions):
                    result = env.execute(action["tool"], action.get("arguments", {}))
                    _require(result.get("ok") is True, f"gold step {index + 1} failed: {result}")
                    _require(env.check_success(task.success_condition) == (index == task.horizon - 1),
                             "gold completes before its last step or never completes")
                saved = copy.deepcopy(env.state["completed_requests"])
                _require(env.execute("inspect_state", {}).get("ok") is True, "inspection failed")
                _require(env.check_success(task.success_condition) and saved == env.state["completed_requests"],
                         "inspection changes durable completion")
                checks["gold_replays"] += 1
                checks["durable_inspections"] += 1
                for mask in range((1 << task.horizon) - 1):
                    env = _environment(task)
                    for index, action in enumerate(task.gold_actions):
                        if mask & (1 << index):
                            env.execute(action["tool"], action.get("arguments", {}))
                            _require(not env.check_success(task.success_condition), f"subsequence mask {mask} succeeds")
                    _require(not env.check_success(task.success_condition), f"subsequence mask {mask} succeeds")
                    checks["proper_subsequences"] += 1
                for index in range(task.horizon - 1):
                    reordered = copy.deepcopy(task.gold_actions)
                    reordered[index], reordered[index + 1] = reordered[index + 1], reordered[index]
                    env = _environment(task)
                    for action in reordered:
                        env.execute(action["tool"], action.get("arguments", {}))
                        _require(not env.check_success(task.success_condition), f"swap {index + 1} bypasses prerequisite")
                    checks["adjacent_swaps"] += 1
                legacy = {
                    "add_to_cart": {"product_id": candidate["candidate_id"]},
                    "remove_from_cart": {"product_id": candidate["candidate_id"]},
                    "replace_cart_item": {"wrong_product_id": "wrong", "product_id": candidate["candidate_id"]},
                    "create_event": {"title": "event", "date": "2026-10-20", "time": "10:00"},
                    "delete_event": {"title": "event", "date": "2026-10-20", "time": "10:00"},
                    "replace_event": {"old_title": "wrong", "old_date": "2026-10-20", "old_time": "09:00",
                                      "title": "event", "date": "2026-10-20", "time": "10:00"},
                }
                for name, arguments in legacy.items():
                    env = _environment(task)
                    before = env.snapshot()
                    _require(env.execute(name, arguments).get("ok") is False and env.snapshot() == before,
                             f"legacy mutation {name} accepted or changed state")
                    checks["legacy_mutation_rejections"] += 1
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{task.task_id}: {exc}") from exc
        for alternative_candidate in spec["candidates"]:
            for alternative in alternative_candidate["options"]:
                if alternative_candidate["candidate_id"] == candidate["candidate_id"] and alternative["option_id"] == option["option_id"]:
                    continue
                env = ExecutionEnvironment(spec)
                for index, action in enumerate(_actions(spec, alternative_candidate, alternative)):
                    result = env.execute(action["tool"], action["arguments"])
                    if index == 3 and not alternative["available"]:
                        _require(result.get("ok") is False, f"{base_id}: unavailable option accepted")
                        break
                    _require(result.get("ok") is True, f"{base_id}: wrong-choice probe not executable: {result}")
                _require(not env.check_success(reference.success_condition), f"{base_id}: wrong choice passes grading")
                checks["wrong_choice_rejections"] += 1
    return {
        "valid": True, "execution_tasks": len(execution), "base_scenarios": len(by_base),
        "by_horizon": {str(h): n for h, n in sorted(Counter(t.horizon for t in execution).items())},
        "by_domain": dict(sorted(Counter(t.domain for t in execution).items())),
        "checks": dict(sorted(checks.items())),
        "validation_scope": "reference subsequences, prerequisite swaps, legacy mutations, and wrong choices",
        "limitations": [
            "Not exhaustive search over arbitrary tool arguments or action sequences.",
            "Horizon is remaining calls in a controlled five-resource workflow with a supplied correct prefix.",
            "Recovery tasks are outside this execution-v2 validation scope.",
        ],
    }
