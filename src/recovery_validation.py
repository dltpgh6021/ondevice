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
from .dataset_spec import BASE_SCENARIOS, ERROR_KINDS, RECOVERY_HORIZONS, RECOVERY_TASKS, SCENARIOS_PER_DOMAIN
from .environment import environment_for_task
from .execution_environment import ExecutionEnvironment, SIGNALED_ERROR_MESSAGE, WORKFLOWS, candidate_is_eligible
from .execution_validation import _actions, _infer_target
from .metrics import score_trace
from .tasks import Task, stable_hash


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _start(task: Task, condition: str) -> tuple[ExecutionEnvironment, list[dict[str, Any]]]:
    environment = environment_for_task(task)
    _require(isinstance(environment, ExecutionEnvironment), "recovery-v3 must use ExecutionEnvironment")
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


def _model_view(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    """Model-visible input with call-ID labels removed (chat templates may omit them)."""
    view = copy.deepcopy(messages)
    for message in view:
        for call in message.get("tool_calls") or []:
            call["id"] = "call"
        if "tool_call_id" in message:
            message["tool_call_id"] = "call"
    return json.dumps([view, tools], sort_keys=True, ensure_ascii=False)


def _expected_plan(spec: dict[str, Any], horizon: int, seeds: list[dict[str, Any]], kind: str) -> dict[str, list[dict[str, Any]]]:
    """Independent derivation from public rules, not from the generator.

    The checkpoint history is the matched free task's provided context.
    violating_candidate repair costs the normal remaining work; wrong_option at
    h3 reuses the correct detail already in the history (prepare -> confirm).
    """
    _require(horizon in RECOVERY_HORIZONS, "recovery fixtures exist only for horizons 1 and 3")
    _require(kind in ERROR_KINDS, f"unknown error kind {kind}")
    candidate, option = _infer_target(spec)
    chain = _actions(spec, candidate, option)
    required = spec["request"]["required_option_kind"]
    if kind == "violating_candidate":
        ineligible = [item for item in spec["candidates"] if not candidate_is_eligible(item, spec)]
        _require(len(ineligible) == 1, "exactly one constraint-violating candidate is required")
        wrong_candidate = ineligible[0]
        wrong_options = [item for item in wrong_candidate["options"] if item["available"] and item["kind"] == required]
        _require(len(wrong_options) == 1, "violating candidate needs one available requested option")
    else:
        wrong_candidate = candidate
        wrong_options = [item for item in candidate["options"] if item["available"] and item["kind"] != required]
        _require(len(wrong_options) == 1, "correct candidate needs exactly one other available option kind")
    wrong_chain = _actions(spec, wrong_candidate, wrong_options[0])
    if horizon == 1:
        _require(wrong_chain[3] in seeds, f"h1 checkpoint must already hold the {kind} draft")
        injected, repair = wrong_chain[4:], chain[4:]
    else:
        injected = wrong_chain[5 - horizon:]
        repair = chain[5 - horizon:] if kind == "violating_candidate" else chain[3:]
    return {"prefix": seeds, "normal": chain[5 - horizon:], "injected": injected, "repair": repair}


def validate_recovery_tasks(tasks: list[Task]) -> dict[str, Any]:
    """Check the recovery fixtures, both error kinds, and their h1/h3 execution pairs."""
    execution = {task.task_id: task for task in tasks if task.task_type == "execution"}
    recovery = [task for task in tasks if task.task_type == "recovery"]
    _require(len(recovery) == RECOVERY_TASKS, f"expected {RECOVERY_TASKS} recovery tasks, found {len(recovery)}")
    _require(len({task.task_id for task in recovery}) == RECOVERY_TASKS, "duplicate recovery task IDs")
    _require(all(task.schema_version == "recovery-v3" for task in recovery), "recovery tasks must all use recovery-v3")
    _require(Counter(task.horizon for task in recovery) == Counter({h: BASE_SCENARIOS for h in RECOVERY_HORIZONS}),
             f"recovery tasks must have {BASE_SCENARIOS} fixtures at each of h1 and h3, and none at h5")
    _require(Counter(task.domain for task in recovery) ==
             Counter({domain: SCENARIOS_PER_DOMAIN * len(RECOVERY_HORIZONS) for domain in WORKFLOWS}),
             "recovery tasks must be balanced across domains")
    _require(Counter(task.base_id for task in recovery) ==
             Counter(task_id for task_id, task in execution.items() if task.horizon in RECOVERY_HORIZONS),
             "each h1/h3 execution task must have exactly one corresponding recovery fixture")
    _require(all(tuple(task.error_variants or {}) == ERROR_KINDS for task in recovery),
             f"every recovery fixture must carry the error kinds {list(ERROR_KINDS)}")
    by_scenario: dict[str, list[Task]] = defaultdict(list)
    checks: Counter[str] = Counter()
    min_calls: dict[tuple[str, int], set[int]] = defaultdict(set)
    views: dict[str, str] = {}
    for task in execution.values():
        free_environment = environment_for_task(task)
        free_messages, _ = free_environment.prepare_condition(task, "free")
        views[_model_view(free_messages, free_environment.tool_definitions())] = f"{task.task_id}/free"

    def remember(view: str, name: str) -> None:
        _require(view not in views, f"{name} input duplicates {views.get(view)}")
        views[view] = name

    for fixture in recovery:
        try:
            pair = execution[fixture.base_id]
            by_scenario[pair.base_id].append(fixture)
            _require(pair.schema_version == "execution-v2", "execution pair must use execution-v2")
            _require((fixture.query, fixture.domain, fixture.horizon, fixture.gold_actions, fixture.success_condition) ==
                     (pair.query, pair.domain, pair.horizon, pair.gold_actions, pair.success_condition),
                     "paired fixture changes original query, domain, horizon, normal oracle, or goal")
            _require(fixture.execution_spec == pair.execution_spec, "fixture changes request or candidates")
            _require(not fixture.recovery_query, "recovery must retain the original query without corrective instructions")
            _require(fixture.initial_actions == list(pair.initial_actions or []),
                     "checkpoint history differs from the matched free task's provided context")
            _require(fixture.normal_remaining_calls == fixture.horizon, "normal remaining calls must equal nominal horizon")
            target = fixture.success_condition["completed_request_equals"]

            normal, normal_messages = _start(fixture, "normal_restart")
            normal_meta = normal.preparation_metadata
            free_environment = environment_for_task(pair)
            free_environment.prepare_condition(pair, "free")
            _require(free_environment.snapshot() == normal_meta["checkpoint_state"] == normal.snapshot(),
                     "free and normal_restart must hold the same information at the checkpoint")
            _require(normal_meta["injected_actions"] == [] and normal_meta["error_signal"] is None
                     and normal_meta["error_kind"] is None, "normal condition contains an injected error")
            _require(normal_meta["resume_workflow_step"] == 5 - fixture.horizon,
                     "resume step does not match the provided correct prefix")
            _require(len(normal_messages) == 2 + 2 * len(fixture.initial_actions or []),
                     "normal history differs from the provided correct prefix")
            _require(not normal.check_success(fixture.success_condition), "task begins complete")
            remember(_model_view(normal_messages, normal.tool_definitions()), f"{fixture.task_id}/normal_restart")
            checks["free_normal_information_parity"] += 1
            _replay(normal, fixture.gold_actions)
            _require(normal.check_success(fixture.success_condition), "normal oracle does not complete the request")
            checks["normal_oracle_replays"] += 1
            trace = run_agent(fixture, ScriptedClient(_model_actions(fixture.gold_actions)), environment_for_task(fixture),
                              condition="normal_restart", max_calls=fixture.horizon + 3)
            scored = score_trace(fixture, trace)
            _require(trace["finished"] and scored["strict_success"] and scored["next_action_correct"] is True
                     and trace["initial_messages"] == normal_messages, "scripted normal continuation fails")
            checks["scripted_agent_continuations"] += 1

            for kind in ERROR_KINDS:
                task = fixture.with_error_kind(kind)
                plan = _expected_plan(task.execution_spec, task.horizon, list(pair.initial_actions or []), kind)
                _require(task.gold_actions == plan["normal"], "normal oracle differs from the inferred target")
                _require(task.error_injection == {"type": "wrong_saved_result", "error_kind": kind, "actions": plan["injected"]},
                         f"{kind}: injected continuation differs from the independent derivation")
                _require(task.recovery_actions == plan["repair"], f"{kind}: repair oracle is not the minimal path")
                _require(task.recovery_min_calls == len(task.recovery_actions), f"{kind}: repair minimum differs from its oracle")
                if kind == "violating_candidate":
                    _require(task.recovery_min_calls == task.normal_remaining_calls,
                             "violating_candidate repair must cost the normal remaining work")
                min_calls[(kind, task.horizon)].add(task.recovery_min_calls)
                checks["matched_execution_pairs"] += 1

                error, error_messages = _start(task, "error_restart")
                signaled, signaled_messages = _start(task, "signaled_error_restart")
                if kind == ERROR_KINDS[0]:
                    alias, alias_messages = _start(task, "recovery")
                    _require(error.snapshot() == alias.snapshot() and error_messages == alias_messages
                             and error.preparation_metadata == alias.preparation_metadata,
                             "recovery alias differs from error_restart")
                _require(normal.tool_definitions() == error.tool_definitions() == signaled.tool_definitions(),
                         "paired conditions expose different tool schemas")
                error_meta = error.preparation_metadata
                _require(error_meta["checkpoint_state"] == normal_meta["checkpoint_state"]
                         and error_meta["checkpoint_state_hash"] == normal_meta["checkpoint_state_hash"]
                         == stable_hash(normal_meta["checkpoint_state"]),
                         "conditions do not branch from the same checkpoint state")
                _require(error_meta["error_kind"] == kind and error_meta["error_signal"] is None,
                         "error metadata is mislabeled")
                _require(signaled.snapshot() == error.snapshot() and signaled_messages[:-1] == error_messages
                         and signaled_messages[-1] == {"role": "user", "content": SIGNALED_ERROR_MESSAGE}
                         and signaled.preparation_metadata["error_signal"] == SIGNALED_ERROR_MESSAGE,
                         "signaled_error_restart must equal error_restart plus exactly the fixed audit notice")
                _require(not any(identifier in SIGNALED_ERROR_MESSAGE for identifier in (target["candidate_id"], target["option_id"])),
                         "the audit notice must not reveal the answer")
                checks["signaled_error_branches"] += 1
                injected = error_meta["injected_actions"]
                _require([item["action"] for item in injected] == plan["injected"]
                         and all(item["result"].get("ok") is True for item in injected),
                         "injected actions were not executed successfully")
                wrong = injected[-1]["result"].get("saved_result")
                _require(isinstance(wrong, dict) and wrong.get("request_id") == target["request_id"],
                         "injected history must save a result for the same request")
                if kind == "violating_candidate":
                    _require(wrong["candidate_id"] != target["candidate_id"], "violating_candidate must save another candidate")
                else:
                    _require(wrong["candidate_id"] == target["candidate_id"] and wrong["option_id"] != target["option_id"],
                             "wrong_option must keep the candidate and change only the option")
                _require(error.assess_action_result(injected[-1]["result"]) == "saved_result_violates_request",
                         "injected saved result must violate the request")
                _require(error.state["completed_requests"] == {target["request_id"]: wrong},
                         "injection does not create exactly the intended wrong saved record")
                _require(error.injected_error_present() and not normal.injected_error_present(),
                         "injected error presence is not observable")
                _require(error_messages[:len(normal_messages)] == normal_messages
                         and len(error_messages) == len(normal_messages) + 2 * len(injected),
                         "paired histories differ beyond the observable injected calls")
                for offset, item in enumerate(injected):
                    call_message, result_message = error_messages[len(normal_messages) + 2 * offset:][:2]
                    _require(call_message["role"] == "assistant" and len(call_message.get("tool_calls", [])) == 1,
                             "injected assistant tool call is missing")
                    call = call_message["tool_calls"][0]
                    _require(call["function"]["name"] == item["action"]["tool"]
                             and json.loads(call["function"]["arguments"]) == item["action"]["arguments"],
                             "visible injected call differs from the executed action")
                    _require(result_message["role"] == "tool" and result_message["tool_call_id"] == call["id"]
                             and json.loads(result_message["content"]) == item["result"],
                             "injected result is absent or differs from the actual tool response")
                _require(not error.check_success(task.success_condition), "error branch begins complete")
                remember(_model_view(error_messages, error.tool_definitions()), f"{task.task_id}/error_restart/{kind}")
                remember(_model_view(signaled_messages, signaled.tool_definitions()),
                         f"{task.task_id}/signaled_error_restart/{kind}")
                checks["distinct_model_inputs"] += 1
                repeated, repeated_messages = _start(task, "error_restart")
                _require(repeated.snapshot() == error.snapshot() and repeated_messages == error_messages
                         and repeated.preparation_metadata == error_meta, "fresh error preparation is not deterministic")
                checks["common_checkpoint_and_visible_injection"] += 1
                checks["deterministic_preparations"] += 1

                _replay(error, task.recovery_actions)
                _require(error.check_success(task.success_condition) and not error.injected_error_present(),
                         "repair oracle does not replace the wrong saved result")
                before = error.snapshot()
                _require(error.execute("inspect_state", {}).get("ok") is True and error.snapshot() == before
                         and error.check_success(task.success_condition), "read-only inspection changes completion")
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

                for condition, expected_messages in (("error_restart", error_messages),
                                                     ("signaled_error_restart", signaled_messages)):
                    runner_environment = environment_for_task(task)
                    trace = run_agent(task, ScriptedClient(_model_actions(task.recovery_actions)), runner_environment,
                                      condition=condition, max_calls=task.horizon + 3)
                    _require(trace["finished"] and trace["runtime_error"] is None
                             and runner_environment.check_success(task.success_condition),
                             "scripted model cannot complete the condition")
                    model_events = [event for event in trace["events"]
                                    if (event.get("action") or {}).get("type") == "tool_call"]
                    _require(len(model_events) == len(task.recovery_actions)
                             and len(trace["events"]) == len(task.recovery_actions) + 1,
                             "model event count includes provided or injected history, or drops model calls")
                    _require(trace["initial_messages"] == expected_messages, "agent does not retain the validated starting history")
                    _require(all(not event.get("error") and not event.get("tool_error") for event in trace["events"]),
                             "oracle execution is counted as an agent error")
                    scored = score_trace(task, trace)
                    _require(scored["strict_success"] and scored["tool_call_count"] == len(task.recovery_actions)
                             and scored["next_action_correct"] is True and scored["error_kind"] == kind
                             and scored["recovery_error_count"] == 0 and scored["extra_calls_over_oracle"] == 0
                             and scored["recovery_error_resolved"] is True,
                             "injected errors or oracle costs are scored incorrectly")
                    checks["scripted_agent_continuations"] += 1
        except (KeyError, TypeError, ValueError, AttributeError, StopIteration, IndexError) as exc:
            raise ValueError(f"{fixture.task_id}: {exc}") from exc

    _require(len(by_scenario) == BASE_SCENARIOS, f"expected {BASE_SCENARIOS} matched base scenarios")
    for base_id, variants in by_scenario.items():
        _require(Counter(task.horizon for task in variants) == Counter({h: 1 for h in RECOVERY_HORIZONS}),
                 f"{base_id}: recovery horizons are not matched")
        _require(all(task.execution_spec == variants[0].execution_spec for task in variants),
                 f"{base_id}: fixture changes across horizons")
        for kind in ERROR_KINDS:
            wrong = {task.with_error_kind(kind).error_injection["actions"][-1]["arguments"]["draft_id"] for task in variants}
            _require(len(wrong) == 1, f"{base_id}: injected {kind} result changes across horizons")
            checks["common_error_across_horizons"] += 1

    return {
        "valid": True,
        "recovery_tasks": len(recovery),
        "base_scenarios": len(by_scenario),
        "error_kinds": list(ERROR_KINDS),
        "paired_conditions_per_model": len(recovery) * (1 + 2 * len(ERROR_KINDS)),
        "by_horizon": {str(h): n for h, n in sorted(Counter(task.horizon for task in recovery).items())},
        "normal_remaining_calls": {str(h): h for h in RECOVERY_HORIZONS},
        "recovery_min_calls": {f"{kind}/h{h}": sorted(values) for (kind, h), values in sorted(min_calls.items())},
        "checks": dict(sorted(checks.items())),
        "validation_scope": "matched checkpoints, both error kinds, visible wrong continuations, exact saved-state grading, reference repair subsequences, and scripted continuations",
        "limitations": [
            "Not exhaustive search over arbitrary arguments, action sequences, or model policies.",
            "Two error kinds of one mistake family (a wrong result saved for the requested ID); conclusions do not cover every agent failure type.",
            "Recovery covers h1 and h3 only; h5 has no checkpoint history and is analyzed with free execution.",
            "wrong_option at h3 needs 2 repair calls while violating_candidate needs 3; report recovery per error kind.",
            "The injected continuation is supplied history rather than a naturally generated past session.",
            f"{BASE_SCENARIOS} base scenarios, rather than {RECOVERY_TASKS} independent problems, form the matched experimental units.",
            "signaled_error_restart adds one fixed audit notice; it is a control for the silent error, not a different error type.",
        ],
    }
