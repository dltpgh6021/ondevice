"""Validate matched recovery fixtures with one observable extra saved result.

This checks supplied checkpoint artifacts and reference repair subsequences.
It does not claim exhaustive search over arbitrary tool arguments or policies.
"""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from typing import Any

from .agent import run_agent
from .client import ScriptedClient
from .environment import environment_for_task
from .execution_environment import ExecutionEnvironment, WORKFLOWS
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
            _require((task.query, task.domain, task.horizon, task.initial_actions, task.gold_actions, task.success_condition) ==
                     (pair.query, pair.domain, pair.horizon, pair.initial_actions, pair.gold_actions, pair.success_condition),
                     "paired fixture changes original query, domain, horizon, correct prefix, oracle, or goal")
            _require(not task.recovery_query, "recovery must retain the original query without corrective instructions")
            _require(task.normal_remaining_calls == task.horizon, "normal remaining calls must equal nominal horizon")
            _require(task.recovery_min_calls == task.horizon + 1, "repair minimum must include the additional cleanup call")
            _require(len(task.gold_actions) == task.normal_remaining_calls, "normal oracle length differs from metadata")
            _require(len(task.recovery_actions) == task.recovery_min_calls, "repair oracle length differs from metadata")
            spec = copy.deepcopy(task.execution_spec)
            initial_drafts = spec.pop("initial_drafts", None)
            _require(spec == pair.execution_spec, "fixture changes request or candidates beyond the shared seeded artifact")
            _require(isinstance(initial_drafts, dict) and len(initial_drafts) == 1, "exactly one stale draft is required")
            stale_draft_id, stale = next(iter(initial_drafts.items()))
            target = task.success_condition["completed_request_equals"]
            stale_request_id = stale.get("request_id")
            _require(isinstance(stale_request_id, str) and stale_request_id.startswith("request_"),
                     "stale draft needs a distinct opaque request ID")
            _require(stale_request_id != target["request_id"], "wrong saved result must be for a different request")
            _require(stale.get("domain") == task.domain, "stale artifact changes domain")
            _require(stale.get("candidate_id") != target["candidate_id"] and stale.get("option_id") != target["option_id"],
                     "stale artifact must contain a wrong candidate and option")
            stale_candidate = next((candidate for candidate in spec["candidates"]
                                    if candidate["candidate_id"] == stale.get("candidate_id")), None)
            _require(stale_candidate is not None, "stale artifact candidate does not exist")
            _require(any(option["option_id"] == stale.get("option_id") and option.get("available") is True
                         for option in stale_candidate["options"]), "stale artifact option does not exist or is unavailable")
            injection = {"tool": WORKFLOWS[task.domain]["names"][4], "arguments": {"draft_id": stale_draft_id}}
            _require(task.error_injection == {"type": "extra_saved_record", "action": injection},
                     "injection must be an actual confirmation of the seeded stale draft")
            cleanup = {"tool": "discard_saved_result", "arguments": {"request_id": stale_request_id}}
            _require(task.recovery_actions == [cleanup] + task.gold_actions,
                     "repair oracle must clean the stale result and then follow the normal oracle")
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
            _require(normal_meta["resume_workflow_step"] == error_meta["resume_workflow_step"] == len(task.initial_actions or []),
                     "resume step does not match the provided correct prefix")
            _require(normal_meta["injected_actions"] == [], "normal condition contains an injected action")
            _require(len(error_meta["injected_actions"]) == 1, "error condition needs exactly one injected action")
            injected = error_meta["injected_actions"][0]
            _require(injected["action"] == injection and injected["result"].get("ok") is True,
                     "injected action was not executed successfully")
            _require(normal.state["drafts"].get(stale_draft_id) == stale and error.state["drafts"].get(stale_draft_id) == stale,
                     "stale artifact is not present in both pre-existing resource sets")
            _require(normal.state["completed_requests"] == {} and error.state["completed_requests"] == {stale_request_id: stale},
                     "injection does not create exactly the intended extra saved record")
            expected_after = normal.snapshot()
            expected_after["completed_requests"][stale_request_id] = copy.deepcopy(stale)
            _require(error.snapshot() == expected_after, "injection changes resources beyond the intended saved record")
            _require(error_messages[:-2] == normal_messages and len(normal_messages) == 2 + 2 * len(task.initial_actions or []),
                     "paired histories differ beyond the observable injected call")
            injected_call, injected_result = error_messages[-2:]
            _require(injected_call["role"] == "assistant" and len(injected_call.get("tool_calls", [])) == 1,
                     "injected assistant tool call is missing")
            call = injected_call["tool_calls"][0]
            _require(call["function"]["name"] == injection["tool"] and
                     json.loads(call["function"]["arguments"]) == injection["arguments"],
                     "visible injected call differs from the executed action")
            _require(injected_result["role"] == "tool" and injected_result["tool_call_id"] == call["id"] and
                     json.loads(injected_result["content"]) == injected["result"],
                     "injected result is absent or differs from the actual tool response")
            _require(injected["result"].get("saved_result") == stale, "tool result does not expose the incorrect saved record")
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
            _replay(error, task.gold_actions)
            _require(not error.check_success(task.success_condition), "correct target alone hides the extra wrong saved result")
            _require(set(error.state["completed_requests"]) == {target["request_id"], stale_request_id},
                     "target confirmation unexpectedly removes the wrong request")
            _replay(error, [cleanup])
            _require(error.check_success(task.success_condition), "cleanup after normal work does not succeed")
            checks["cleanup_last_replays"] += 1
            error, _ = _start(task, "error_restart")
            _replay(error, task.recovery_actions)
            _require(error.check_success(task.success_condition), "repair oracle does not complete the request")
            before = error.snapshot()
            _require(error.execute("inspect_state", {}).get("ok") is True and error.snapshot() == before and
                     error.check_success(task.success_condition), "read-only inspection changes completion")
            error.execute("discard_saved_result", {"request_id": "unknown_request_not_in_fixture"})
            _require(error.snapshot() == before and error.check_success(task.success_condition),
                     "discarding an unknown request removes the requested result")
            checks["repair_oracle_replays"] += 1
            checks["durable_inspections_and_target_preservation"] += 1

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
                _require(scored["strict_success"] and scored["tool_call_count"] == len(oracle), "oracle scoring differs from state")
                if condition == "error_restart":
                    _require(scored["recovery_error_count"] == 0 and scored["extra_calls_over_normal"] == 1 and
                             scored["extra_calls_over_oracle"] == 0 and scored["recovery_error_resolved"] is True,
                             "injected errors or oracle costs are scored incorrectly")
                checks["scripted_agent_continuations"] += 1
        except (KeyError, TypeError, ValueError, AttributeError, StopIteration) as exc:
            raise ValueError(f"{task.task_id}: {exc}") from exc

    _require(len(by_scenario) == 20, "expected twenty matched base scenarios")
    for base_id, variants in by_scenario.items():
        _require(Counter(task.horizon for task in variants) == Counter({1: 1, 3: 1, 5: 1}),
                 f"{base_id}: recovery horizons are not matched")
        reference = variants[0]
        _require(all(task.error_injection == reference.error_injection and task.execution_spec == reference.execution_spec
                     for task in variants), f"{base_id}: error mechanism or seeded artifact changes across horizons")
        checks["common_error_across_horizons"] += 1

    return {
        "valid": True,
        "recovery_tasks": len(recovery),
        "base_scenarios": len(by_scenario),
        "paired_conditions_per_model": len(recovery) * 2,
        "by_horizon": {str(h): n for h, n in sorted(Counter(task.horizon for task in recovery).items())},
        "normal_remaining_calls": {str(h): h for h in (1, 3, 5)},
        "recovery_min_calls": {str(h): h + 1 for h in (1, 3, 5)},
        "checks": dict(sorted(checks.items())),
        "validation_scope": "matched checkpoints, visible side effects, exact saved-state grading, reference repair subsequences, and scripted continuations",
        "limitations": [
            "Not exhaustive search over arbitrary arguments, action sequences, or model policies.",
            "The error is one supplied extra saved record for a wrong request; conclusions do not cover every agent failure type.",
            "Nominal horizons 1/3/5 describe normal remaining work; minimum repair calls are 2/4/6.",
            "Checkpoint draft artifacts are supplied fixtures rather than a complete naturally generated past session.",
            "Error conditions contain one additional visible assistant/tool pair and the wrong saved state.",
            "Twenty base scenarios, rather than sixty independent problems, form the matched experimental units.",
        ],
    }
