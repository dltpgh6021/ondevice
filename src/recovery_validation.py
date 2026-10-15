"""Validate matched recovery fixtures with one observable wrong saved result.

The injected error is the natural free-execution mistake: a candidate that
violates the request constraints is confirmed for the same request. This checks
supplied checkpoint artifacts and reference repair subsequences. It does not
claim exhaustive search over arbitrary tool arguments or policies.
"""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from typing import Any

from .agent import run_agent
from .client import ScriptedClient
from .environment import environment_for_task
from .execution_environment import ExecutionEnvironment, WORKFLOWS, candidate_is_eligible
from .execution_validation import _actions, _infer_target
from .tasks import Task, stable_hash
from .metrics import score_trace


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _start(task: Task, condition: str) -> tuple[ExecutionEnvironment, list[dict[str, Any]]]:
    environment = environment_for_task(task)
    _require(isinstance(environment, ExecutionEnvironment), "recovery-v2 must use ExecutionEnvironment")
    messages, _ = environment.prepare_condition(task, condition)
    return environment, messages


def _replay(environment: ExecutionEnvironment, actions: list[dict[str, Any]]) -> None:
    for index, action in enumerate(actions):
        result = environment.execute(action["tool"], action.get("arguments", {}))
        _require(result.get("ok") is True, f"oracle action {index + 1} failed: {result}")


def _model_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"type": "tool_call", "name": action["tool"], "arguments": copy.deepcopy(action.get("arguments", {}))}
        for action in actions
    ] + [{"type": "finish", "answer": "completed"}]


def _expected_plan(spec: dict[str, Any], horizon: int) -> dict[str, list[dict[str, Any]]]:
    """Independent derivation from public rules, not from the generator."""
    candidate, option = _infer_target(spec)
    chain = _actions(spec, candidate, option)
    ineligible = [item for item in spec["candidates"] if not candidate_is_eligible(item, spec)]
    _require(len(ineligible) == 1, "exactly one constraint-violating candidate is required")
    required = spec["request"]["required_option_kind"]
    wrong_options = [item for item in ineligible[0]["options"] if item["available"] and item["kind"] == required]
    _require(len(wrong_options) == 1, "violating candidate needs one available requested option")
    wrong_chain = _actions(spec, ineligible[0], wrong_options[0])
    return {
        "prefix": chain[:5 - horizon], "normal": chain[5 - horizon:],
        "injected": wrong_chain[2:] if horizon == 1 else wrong_chain[5 - horizon:],
        "repair": chain[4:] if horizon == 1 else chain[2:],
    }


def validate_recovery_tasks(tasks: list[Task]) -> dict[str, Any]:
    """Check 60 recovery fixtures and their 60 corresponding execution tasks."""
    execution = {task.task_id: task for task in tasks if task.task_type == "execution"}
    recovery = [task for task in tasks if task.task_type == "recovery"]
    _require(len(recovery) == 60, f"expected 60 recovery tasks, found {len(recovery)}")
    _require(len({task.task_id for task in recovery}) == 60, "duplicate recovery task IDs")
    _require(all(task.schema_version == "recovery-v2" for task in recovery), "recovery tasks must all use recovery-v2")
    _require(Counter(task.horizon for task in recovery) == Counter({1: 20, 3: 20, 5: 20}),
             "recovery tasks must have twenty fixtures per nominal horizon")
    _require(Counter(task.domain for task in recovery) == Counter({domain: 15 for domain in WORKFLOWS}),
             "recovery tasks must have fifteen fixtures per domain")
    _require(Counter(task.base_id for task in recovery) == Counter(execution.keys()),
             "each execution task must have exactly one corresponding recovery fixture")
    by_scenario: dict[str, list[Task]] = defaultdict(list)
    checks: Counter[str] = Counter()

    for task in recovery:
        try:
            pair = execution[task.base_id]
            by_scenario[pair.base_id].append(task)
            _require(pair.schema_version == "execution-v2", "execution pair must use execution-v2")
            _require((task.query, task.domain, task.horizon, task.gold_actions, task.success_condition) ==
                     (pair.query, pair.domain, pair.horizon, pair.gold_actions, pair.success_condition),
                     "paired fixture changes original query, domain, horizon, normal oracle, or goal")
            _require(task.execution_spec == pair.execution_spec, "fixture changes request or candidates")
            _require(not task.recovery_query, "recovery must retain the original query without corrective instructions")
            plan = _expected_plan(task.execution_spec, task.horizon)
            _require(task.initial_actions == plan["prefix"], "checkpoint prefix is not the correct workflow prefix")
            _require(task.gold_actions == plan["normal"], "normal oracle differs from the inferred target")
            _require(task.normal_remaining_calls == task.horizon, "normal remaining calls must equal nominal horizon")
            _require(task.error_injection == {"type": "wrong_saved_result", "actions": plan["injected"]},
                     "injection must continue with the constraint-violating candidate for the same request")
            _require(task.recovery_actions == plan["repair"], "repair oracle is not the minimal re-confirmation path")
            _require(task.recovery_min_calls == len(task.recovery_actions) <= task.horizon,
                     "repair minimum must equal the oracle and not exceed the normal horizon")
            checks["matched_execution_pairs"] += 1

            normal, normal_messages = _start(task, "normal_restart")
            error, error_messages = _start(task, "error_restart")
            alias, alias_messages = _start(task, "recovery")
            _require(normal.tool_definitions() == error.tool_definitions() == alias.tool_definitions(),
                     "paired conditions expose different tool schemas")
            _require(error.snapshot() == alias.snapshot() and error_messages == alias_messages,
                     "recovery alias differs from error_restart")
            _require(error.preparation_metadata == alias.preparation_metadata, "recovery alias changes preparation metadata")
            normal_meta, error_meta = normal.preparation_metadata, error.preparation_metadata
            _require(normal_meta["checkpoint_state"] == error_meta["checkpoint_state"] == normal.snapshot(),
                     "conditions do not branch from the same checkpoint state")
            _require(normal_meta["checkpoint_state_hash"] == error_meta["checkpoint_state_hash"] == stable_hash(normal.snapshot()),
                     "pre-branch checkpoint hashes differ or are incorrect")
            _require(normal_meta["resume_workflow_step"] == error_meta["resume_workflow_step"] == 5 - task.horizon,
                     "resume step does not match the provided correct prefix")
            _require(normal_meta["injected_actions"] == [], "normal condition contains an injected action")
            injected = error_meta["injected_actions"]
            _require([item["action"] for item in injected] == plan["injected"] and
                     all(item["result"].get("ok") is True for item in injected),
                     "injected actions were not executed successfully")
            target = task.success_condition["completed_request_equals"]
            wrong = injected[-1]["result"].get("saved_result")
            _require(isinstance(wrong, dict) and wrong.get("request_id") == target["request_id"],
                     "injected history must save a result for the same request")
            _require(wrong["candidate_id"] != target["candidate_id"] and error.assess_action_result(injected[-1]["result"])
                     == "saved_result_violates_request", "injected saved result must violate the request")
            _require(normal.state["completed_requests"] == {} and error.state["completed_requests"] == {target["request_id"]: wrong},
                     "injection does not create exactly the intended wrong saved record")
            _require(error.injected_error_present() and not normal.injected_error_present(),
                     "injected error presence is not observable")
            _require(error_messages[:len(normal_messages)] == normal_messages and
                     len(normal_messages) == 2 + 2 * len(task.initial_actions or []) and
                     len(error_messages) == len(normal_messages) + 2 * len(injected),
                     "paired histories differ beyond the observable injected calls")
            for offset, item in enumerate(injected):
                call_message, result_message = error_messages[len(normal_messages) + 2 * offset:][:2]
                _require(call_message["role"] == "assistant" and len(call_message.get("tool_calls", [])) == 1,
                         "injected assistant tool call is missing")
                call = call_message["tool_calls"][0]
                _require(call["function"]["name"] == item["action"]["tool"] and
                         json.loads(call["function"]["arguments"]) == item["action"]["arguments"],
                         "visible injected call differs from the executed action")
                _require(result_message["role"] == "tool" and result_message["tool_call_id"] == call["id"] and
                         json.loads(result_message["content"]) == item["result"],
                         "injected result is absent or differs from the actual tool response")
            _require(not normal.check_success(task.success_condition) and not error.check_success(task.success_condition),
                     "task begins complete")
            repeated, repeated_messages = _start(task, "error_restart")
            _require(repeated.snapshot() == error.snapshot() and repeated_messages == error_messages and
                     repeated.preparation_metadata == error_meta, "fresh error preparation is not deterministic")
            checks["common_checkpoint_and_visible_injection"] += 1
            checks["deterministic_preparations"] += 1

            _replay(normal, task.gold_actions)
            _require(normal.check_success(task.success_condition), "normal oracle does not complete the request")
            checks["normal_oracle_replays"] += 1
            _replay(error, task.recovery_actions)
            _require(error.check_success(task.success_condition) and not error.injected_error_present(),
                     "repair oracle does not replace the wrong saved result")
            before = error.snapshot()
            _require(error.execute("inspect_state", {}).get("ok") is True and error.snapshot() == before and
                     error.check_success(task.success_condition), "read-only inspection changes completion")
            error.execute(WORKFLOWS[task.domain]["names"][4], plan["injected"][-1]["arguments"])
            _require(error.injected_error_present() and not error.check_success(task.success_condition),
                     "re-confirming the wrong draft is not detected")
            checks["repair_oracle_replays"] += 1
            checks["wrong_reconfirmation_rejections"] += 1

            error, _ = _start(task, "error_restart")
            _replay(error, [{"tool": "discard_saved_result", "arguments": {"request_id": target["request_id"]}}]
                    + task.recovery_actions)
            _require(error.check_success(task.success_condition), "discard-then-confirm repair is rejected")
            error, _ = _start(task, "error_restart")
            _replay(error, task.gold_actions)
            _require(error.check_success(task.success_condition), "following the normal workflow does not repair")
            checks["alternative_repairs"] += 1

            for mask in range((1 << len(task.recovery_actions)) - 1):
                probe, _ = _start(task, "error_restart")
                for index, action in enumerate(task.recovery_actions):
                    if mask & (1 << index):
                        probe.execute(action["tool"], action.get("arguments", {}))
                        _require(not probe.check_success(task.success_condition), f"proper repair subsequence {mask} succeeds")
                _require(not probe.check_success(task.success_condition), f"proper repair subsequence {mask} succeeds")
                checks["proper_repair_subsequences"] += 1

            for condition, oracle, expected_messages in (
                ("normal_restart", task.gold_actions, normal_messages),
                ("error_restart", task.recovery_actions, error_messages),
            ):
                runner_environment = environment_for_task(task)
                trace = run_agent(task, ScriptedClient(_model_actions(oracle)), runner_environment,
                                  condition=condition, max_calls=task.horizon + 3)
                _require(trace["finished"] and trace["runtime_error"] is None and
                         runner_environment.check_success(task.success_condition), "scripted model cannot complete the condition")
                model_events = [event for event in trace["events"]
                                if (event.get("action") or {}).get("type") == "tool_call"]
                _require(len(model_events) == len(oracle) and len(trace["events"]) == len(oracle) + 1,
                         "model event count includes provided or injected history, or drops model calls")
                _require(trace["initial_messages"] == expected_messages, "agent does not retain the validated starting history")
                _require(all(not event.get("error") and not event.get("tool_error") for event in trace["events"]),
                         "oracle execution is counted as an agent error")
                scored = score_trace(task, trace)
                _require(scored["strict_success"] and scored["tool_call_count"] == len(oracle)
                         and scored["next_action_correct"] is True, "oracle scoring differs from state")
                if condition == "error_restart":
                    _require(scored["recovery_error_count"] == 0 and scored["extra_calls_over_normal"] == 0 and
                             scored["extra_calls_over_oracle"] == 0 and scored["recovery_error_resolved"] is True,
                             "injected errors or oracle costs are scored incorrectly")
                checks["scripted_agent_continuations"] += 1
        except (KeyError, TypeError, ValueError, AttributeError, StopIteration, IndexError) as exc:
            raise ValueError(f"{task.task_id}: {exc}") from exc

    _require(len(by_scenario) == 20, "expected twenty matched base scenarios")
    for base_id, variants in by_scenario.items():
        _require(Counter(task.horizon for task in variants) == Counter({1: 1, 3: 1, 5: 1}),
                 f"{base_id}: recovery horizons are not matched")
        wrong = {task.error_injection["actions"][-1]["arguments"]["draft_id"] for task in variants}
        _require(len(wrong) == 1 and all(task.execution_spec == variants[0].execution_spec for task in variants),
                 f"{base_id}: injected wrong result changes across horizons")
        checks["common_error_across_horizons"] += 1

    return {
        "valid": True,
        "recovery_tasks": len(recovery),
        "base_scenarios": len(by_scenario),
        "paired_conditions_per_model": len(recovery) * 2,
        "by_horizon": {str(h): n for h, n in sorted(Counter(task.horizon for task in recovery).items())},
        "normal_remaining_calls": {str(h): h for h in (1, 3, 5)},
        "recovery_min_calls": {str(h): sorted({task.recovery_min_calls for task in recovery if task.horizon == h})
                               for h in (1, 3, 5)},
        "checks": dict(sorted(checks.items())),
        "validation_scope": "matched checkpoints, visible wrong continuations, exact saved-state grading, reference repair subsequences, and scripted continuations",
        "limitations": [
            "Not exhaustive search over arbitrary arguments, action sequences, or model policies.",
            "The error is one supplied wrong continuation that saves the constraint-violating candidate for the same request; conclusions do not cover every agent failure type.",
            "Nominal horizons 1/3/5 describe normal remaining work; minimum repair calls are 1/3/3.",
            "The injected continuation is supplied history rather than a naturally generated past session.",
            "Twenty base scenarios, rather than sixty independent problems, form the matched experimental units.",
        ],
    }
