"""Offline regression tests for behavior-error denominators and paired recovery."""
from __future__ import annotations

import copy
import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import run_agent
from src.client import InfrastructureError, LlamaCppClient, ModelOutputError, parse_chat_response
from src.environment import environment_for_task
from src.metrics import aggregate, score_trace
from src.recovery_comparison import compare_recovery_records
from src.tasks import load_tasks
from scripts.analyze import validate_analysis_configurations


def payload(arguments="{}", *, calls=1, finish_reason="tool_calls"):
    return {"choices": [{"message": {"role": "assistant", "tool_calls": [
        {"id": f"call_{i}", "type": "function", "function": {
            "name": "inspect_state", "arguments": arguments,
        }} for i in range(calls)
    ]}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 17, "completion_tokens": 9}}


def action(entry):
    return {"type": "tool_call", "name": entry["tool"],
            "arguments": copy.deepcopy(entry.get("arguments", {}))}


def finish():
    return {"type": "finish", "answer": "done"}


def parse_error(body):
    try:
        parse_chat_response(body)
    except ModelOutputError as exc:
        return exc
    raise AssertionError("Expected a model-output error")


class SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.histories = []

    def chat(self, messages, tools):
        self.histories.append(copy.deepcopy(messages))
        item = self.responses.pop(0) if self.responses else finish()
        if isinstance(item, Exception):
            raise item
        return {"action": item, "usage": {}}


class ParserAndTransportTests(unittest.TestCase):
    def response(self, raw):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = raw
        return response

    def test_valid_tool_call_parses(self):
        result = parse_chat_response(payload('{"request_id":"sample"}'))
        self.assertEqual(result["arguments"], {"request_id": "sample"})
        self.assertEqual(result["name"], "inspect_state")

    def test_malformed_and_nonobject_arguments_are_model_errors(self):
        for arguments in ("{bad", "[]", "0", [], 0, False):
            with self.subTest(arguments=arguments):
                body = payload(arguments)
                with self.assertRaises(ModelOutputError) as caught:
                    parse_chat_response(body)
                self.assertEqual(caught.exception.kind, "invalid_arguments")
                self.assertEqual(caught.exception.payload, body)
                self.assertEqual(caught.exception.usage, body["usage"])

    def test_multiple_calls_do_not_silently_select_the_first(self):
        error = parse_error(payload(calls=2))
        self.assertEqual(error.kind, "multiple_tool_calls")
        self.assertEqual(error.attempted_tool_call_count, 2)

    def test_length_finish_reason_is_model_truncation(self):
        error = parse_error(payload(finish_reason="length"))
        self.assertEqual(error.kind, "output_truncated")
        self.assertEqual(error.finish_reason, "length")

    def test_missing_choices_are_infrastructure_protocol_errors(self):
        for body in ({}, {"choices": []}, {"choices": [None]}):
            with self.subTest(body=body):
                with self.assertRaises(InfrastructureError):
                    parse_chat_response(body)

    def test_legacy_bad_arguments_are_model_errors(self):
        body = {"choices": [{"message": {"function_call": {
            "name": "inspect_state", "arguments": "[]",
        }}, "finish_reason": "function_call"}]}
        self.assertEqual(parse_error(body).kind, "invalid_arguments")

    def test_http_client_preserves_model_payload_and_usage(self):
        body = payload("{bad")
        with patch("src.client.urllib.request.urlopen", return_value=self.response(json.dumps(body).encode())):
            with self.assertRaises(ModelOutputError) as caught:
                LlamaCppClient("http://127.0.0.1:8080", "test").chat([], [])
        self.assertEqual(caught.exception.payload, body)
        self.assertEqual(caught.exception.usage["completion_tokens"], 9)

    def test_timeout_is_infrastructure_failure(self):
        with patch("src.client.urllib.request.urlopen", side_effect=TimeoutError("slow")):
            with self.assertRaises(InfrastructureError):
                LlamaCppClient("http://127.0.0.1:8080", "test").chat([], [])

    def test_network_error_is_infrastructure_failure(self):
        with patch("src.client.urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
            with self.assertRaises(InfrastructureError):
                LlamaCppClient("http://127.0.0.1:8080", "test").chat([], [])

    def test_http_error_is_infrastructure_failure(self):
        error = urllib.error.HTTPError("http://localhost", 503, "unavailable", {}, io.BytesIO(b"busy"))
        with patch("src.client.urllib.request.urlopen", side_effect=error):
            with self.assertRaises(InfrastructureError):
                LlamaCppClient("http://127.0.0.1:8080", "test").chat([], [])

    def test_invalid_response_json_is_infrastructure_failure(self):
        with patch("src.client.urllib.request.urlopen", return_value=self.response(b"not JSON")):
            with self.assertRaises(InfrastructureError):
                LlamaCppClient("http://127.0.0.1:8080", "test").chat([], [])

    def test_malformed_usage_is_a_typed_protocol_error_even_with_bad_model_arguments(self):
        for usage in ("bad", True, {"prompt_tokens": "bad"},
                      {"completion_tokens": -1}, {"prompt_tokens": True}):
            with self.subTest(usage=usage):
                body = payload("{bad")
                body["usage"] = usage
                with patch("src.client.urllib.request.urlopen", return_value=self.response(json.dumps(body).encode())):
                    with self.assertRaises(InfrastructureError):
                        LlamaCppClient("http://127.0.0.1:8080", "test").chat([], [])


class BehaviorMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tasks = load_tasks(ROOT / "tasks" / "tasks.json")
        cls.execution = next(t for t in tasks if t.task_id == "shopping_001_h1")
        cls.recovery = next(t for t in tasks if t.task_id == "recovery_shopping_001_h1")

    def run_case(self, task, actions, *, condition="free", max_calls=5):
        client = SequenceClient(actions)
        trace = run_agent(task, client, environment_for_task(task),
                          condition=condition, max_calls=max_calls)
        return trace, score_trace(task, trace), client

    def test_parse_error_is_counted_and_can_recover(self):
        body = payload("{bad")
        trace, score, client = self.run_case(self.execution, [
            parse_error(body), action(self.execution.gold_actions[0]), finish(),
        ])
        self.assertIsNone(trace["runtime_error"])
        self.assertTrue(score["valid_for_behavior_metrics"])
        self.assertTrue(score["strict_success"])
        self.assertTrue(score["natural_error"])
        self.assertEqual(score["model_output_error_count"], 1)
        self.assertEqual(score["error_count"], 1)
        self.assertEqual(score["first_error_turn"], 1)
        self.assertIsNone(score["first_error_tool_call_index"])
        self.assertEqual(trace["events"][0]["raw_response"], body)
        self.assertEqual(trace["prompt_tokens"], 17)
        self.assertEqual(trace["completion_tokens"], 9)
        self.assertGreater(len(client.histories[1]), len(client.histories[0]))

    def test_model_only_failure_is_valid_behavior_failure(self):
        trace, score, _ = self.run_case(self.execution, [parse_error(payload("[]"))], max_calls=1)
        self.assertTrue(score["valid_for_behavior_metrics"])
        self.assertFalse(score["strict_success"])
        self.assertEqual(score["response_turn_count"], 1)
        self.assertEqual(score["error_turn_rate"], 1.0)
        self.assertTrue(score["budget_exhausted"])
        self.assertEqual(trace["termination_reason"], "budget_exhausted")

    def test_multiple_calls_execute_none(self):
        trace, score, _ = self.run_case(self.execution, [parse_error(payload(calls=2))], max_calls=1)
        self.assertEqual(score["tool_call_count"], 0)
        self.assertEqual(score["attempted_tool_call_count"], 2)
        self.assertEqual(score["model_output_error_count"], 1)
        self.assertFalse(trace["final_state"]["completed_requests"])

    def test_falsy_nonobject_arguments_are_not_empty_objects(self):
        for arguments in ([], 0, False):
            with self.subTest(arguments=arguments):
                _, score, _ = self.run_case(self.execution, [{
                    "type": "tool_call", "name": "inspect_state", "arguments": arguments,
                }], max_calls=1)
                self.assertEqual(score["tool_call_count"], 0)
                self.assertEqual(score["error_count"], 1)

    def test_timeout_is_excluded_not_model_error(self):
        client = LlamaCppClient("http://127.0.0.1:8080", "test")
        with patch("src.client.urllib.request.urlopen", side_effect=TimeoutError("slow")):
            trace = run_agent(self.execution, client, environment_for_task(self.execution),
                              condition="free", max_calls=3)
        score = score_trace(self.execution, trace)
        self.assertFalse(score["valid_for_behavior_metrics"])
        self.assertIsNotNone(trace["infrastructure_error"])
        self.assertEqual(trace["termination_reason"], "infrastructure_error")
        self.assertEqual(score["model_output_error_count"], 0)
        summary = aggregate([{**score, "model_label": "test"}])[0]
        self.assertEqual(summary["n_valid"], 0)
        self.assertIsNone(summary["success_rate"])

    def test_unknown_exception_is_invalid_internal_error(self):
        trace, score, _ = self.run_case(self.execution, [AssertionError("implementation defect")])
        self.assertFalse(score["valid_for_behavior_metrics"])
        self.assertEqual(trace["termination_reason"], "internal_error")
        self.assertEqual(score["model_output_error_count"], 0)

    def test_malformed_usage_cannot_escape_agent_or_lose_trace(self):
        body = payload("{bad")
        body["usage"] = {"prompt_tokens": "bad"}
        response = ParserAndTransportTests().response(json.dumps(body).encode())
        client = LlamaCppClient("http://127.0.0.1:8080", "test")
        with patch("src.client.urllib.request.urlopen", return_value=response):
            trace = run_agent(self.execution, client, environment_for_task(self.execution),
                              condition="free", max_calls=3)
        self.assertEqual(trace["termination_reason"], "infrastructure_error")
        self.assertFalse(score_trace(self.execution, trace)["valid_for_behavior_metrics"])

    def test_valid_extra_lookup_is_not_a_natural_error(self):
        _, score, _ = self.run_case(self.execution, [
            {"type": "tool_call", "name": "inspect_state", "arguments": {}},
            action(self.execution.gold_actions[0]), finish(),
        ])
        self.assertTrue(score["strict_success"])
        self.assertFalse(score["natural_error"])
        self.assertEqual(score["error_count"], 0)
        self.assertEqual(score["tool_call_count"], 2)
        self.assertEqual(score["extra_calls_over_oracle"], 1)

    def test_premature_finish_is_a_behavior_error(self):
        trace, score, _ = self.run_case(self.execution, [finish()])
        self.assertTrue(score["valid_for_behavior_metrics"])
        self.assertFalse(score["strict_success"])
        self.assertEqual(score["semantic_error_count"], 1)
        self.assertEqual(score["error_count"], 1)
        self.assertIsNone(score["first_error_tool_call_index"])
        self.assertEqual(trace["events"][0]["semantic_error"], "premature_finish")

    def test_budget_exhaustion_can_have_zero_action_errors(self):
        lookup = {"type": "tool_call", "name": "inspect_state", "arguments": {}}
        trace, score, _ = self.run_case(self.execution, [lookup, lookup], max_calls=2)
        self.assertFalse(score["strict_success"])
        self.assertEqual(score["error_count"], 0)
        self.assertEqual(score["response_turn_count"], 2)
        self.assertTrue(score["budget_exhausted"])
        self.assertEqual(trace["termination_reason"], "budget_exhausted")

    def test_first_error_tool_index_is_one_based_among_executed_calls(self):
        _, score, _ = self.run_case(self.execution, [
            {"type": "tool_call", "name": "inspect_state", "arguments": {}},
            {"type": "tool_call", "name": "missing_tool", "arguments": {}}, finish(),
        ])
        self.assertEqual(score["first_error_turn"], 2)
        self.assertEqual(score["first_error_tool_call_index"], 2)

    def test_injected_history_is_not_a_new_agent_error(self):
        actions = [action(entry) for entry in self.recovery.recovery_actions] + [finish()]
        trace, score, _ = self.run_case(self.recovery, actions, condition="error_restart")
        self.assertTrue(trace["injected_actions"])
        self.assertTrue(score["recovery_success"])
        self.assertEqual(score["error_count"], 0)
        self.assertEqual(score["recovery_error_count"], 0)
        self.assertEqual(score["tool_call_count"], self.recovery.recovery_min_calls)
        self.assertEqual(score["extra_calls_over_normal"], 1)
        self.assertEqual(score["extra_calls_over_oracle"], 0)

    def test_creating_target_without_cleanup_is_not_recovery(self):
        trace, score, _ = self.run_case(self.recovery,
            [action(entry) for entry in self.recovery.gold_actions] + [finish()],
            condition="error_restart")
        self.assertFalse(score["strict_success"])
        self.assertFalse(score["recovery_error_resolved"])
        self.assertEqual(len(trace["final_state"]["completed_requests"]), 2)

    def test_reintroducing_bad_saved_record_invalidates_final_recovery(self):
        actions = [action(entry) for entry in self.recovery.recovery_actions]
        actions += [action(self.recovery.error_injection["action"]), finish()]
        _, score, _ = self.run_case(self.recovery, actions, condition="error_restart")
        self.assertFalse(score["recovery_success"])
        self.assertFalse(score["recovery_error_resolved"])
        self.assertGreaterEqual(score["semantic_error_count"], 1)
        self.assertIsNone(score["recovery_stable_resolution_turn"])

    def test_trace_records_metric_and_shared_policy_versions(self):
        trace, score, _ = self.run_case(self.execution,
                                      [action(self.execution.gold_actions[0]), finish()])
        self.assertEqual(trace["metrics_version"], "workflow-metrics-v3")
        self.assertEqual(trace["agent_policy_version"], "single-action-feedback-v1")
        self.assertEqual(score["metrics_version"], trace["metrics_version"])

    def test_aggregation_keeps_metric_versions_separate(self):
        _, score, _ = self.run_case(self.execution, [finish()])
        rows = [{**score, "model_label": "same-model", "metrics_version": version}
                for version in ("workflow-metrics-v2", "workflow-metrics-v3")]
        summaries = aggregate(rows)
        self.assertEqual(len(summaries), 2)
        self.assertEqual({row["metrics_version"] for row in summaries},
                         {"workflow-metrics-v2", "workflow-metrics-v3"})

    def test_mean_episode_rate_and_pooled_turn_rate_are_different(self):
        _, score, _ = self.run_case(self.execution, [finish()])
        first = {**score, "model_label": "test", "error_count": 1,
                 "response_turn_count": 1, "error_turn_rate": 1.0}
        second = {**score, "model_label": "test", "error_count": 0,
                  "response_turn_count": 9, "error_turn_rate": 0.0}
        invalid = {**score, "model_label": "test", "error_count": 100,
                   "response_turn_count": 100, "error_turn_rate": 1.0,
                   "valid_for_behavior_metrics": False}
        summary = aggregate([first, second, invalid])[0]
        self.assertEqual(summary["n_valid"], 2)
        self.assertEqual(summary["mean_error_turn_rate"], 0.5)
        self.assertEqual(summary["pooled_error_turn_rate"], 0.1)

    def recovery_record(self, condition, *, version="workflow-metrics-v3"):
        oracle = self.recovery.recovery_actions if condition == "error_restart" else self.recovery.gold_actions
        trace, score, _ = self.run_case(self.recovery,
            [action(entry) for entry in oracle] + [finish()], condition=condition)
        trace.update(max_calls=4, metrics_version=version)
        score["metrics_version"] = version
        return {"metadata": {"model_label": "test", "model_sha256": "test-model-hash",
                "hostname": "same-machine", "seed": 42, "max_calls": None,
                "metrics_version": version, "agent_policy_version": trace["agent_policy_version"]},
            "task": {"id": self.recovery.task_id, "schema_version": self.recovery.schema_version,
                     "horizon": self.recovery.horizon,
                     "success_condition": self.recovery.success_condition},
            "trace": trace, "score": score}

    def test_identical_versions_form_one_valid_recovery_pair(self):
        comparison = compare_recovery_records([
            self.recovery_record("normal_restart"), self.recovery_record("error_restart"),
        ])
        self.assertEqual(comparison["diagnostics"]["valid_pairs"], 1)
        self.assertEqual(comparison["summaries"][0]["recovery_penalty"], 0.0)

    def test_old_and_new_versions_never_form_a_recovery_pair(self):
        comparison = compare_recovery_records([
            self.recovery_record("normal_restart", version="workflow-metrics-v2"),
            self.recovery_record("error_restart"),
        ])
        self.assertEqual(comparison["diagnostics"]["total_matched_pairs"], 0)
        self.assertFalse(comparison["summaries"])

    def test_infrastructure_failure_excludes_the_matched_pair(self):
        normal = self.recovery_record("normal_restart")
        error = self.recovery_record("error_restart")
        error["trace"].update(runtime_error="HTTP timeout", infrastructure_error="HTTP timeout")
        error["score"]["valid_for_behavior_metrics"] = False
        compared = compare_recovery_records([normal, error])
        self.assertEqual(compared["diagnostics"]["total_matched_pairs"], 1)
        self.assertEqual(compared["diagnostics"]["excluded_invalid_pairs"], 1)
        self.assertFalse(compared["summaries"])

    def test_model_output_error_does_not_exclude_the_matched_pair(self):
        normal = self.recovery_record("normal_restart")
        error = self.recovery_record("error_restart")
        actions = [parse_error(payload("[]"))] + [action(a) for a in self.recovery.recovery_actions] + [finish()]
        trace, score, _ = self.run_case(self.recovery, actions, condition="error_restart", max_calls=4)
        trace["max_calls"] = 4
        error.update(trace=trace, score=score)
        compared = compare_recovery_records([normal, error])
        self.assertEqual(compared["diagnostics"]["valid_pairs"], 1)
        self.assertEqual(score["recovery_model_output_error_count"], 1)
        self.assertTrue(score["recovery_success"])

    def test_nondict_action_is_a_model_error_not_a_scorer_crash(self):
        _, score, _ = self.run_case(self.execution, [["invalid", "action"]], max_calls=1)
        self.assertTrue(score["valid_for_behavior_metrics"])
        self.assertEqual(score["error_count"], 1)
        self.assertEqual(score["tool_call_count"], 0)

    def test_analysis_rejects_same_label_condition_across_task_protocols(self):
        record = {"metadata": {"model_label": "same-model"},
                  "trace": {"condition": "free", "protocol_version": "execution-v2"}, "score": {}}
        legacy = copy.deepcopy(record)
        legacy["trace"]["protocol_version"] = "legacy-v1"
        with self.assertRaisesRegex(ValueError, "Mixed task protocols"):
            validate_analysis_configurations([record, legacy])

    def test_internally_conflicting_version_tags_cannot_form_a_pair(self):
        normal = self.recovery_record("normal_restart")
        error = self.recovery_record("error_restart")
        error["trace"]["metrics_version"] = "workflow-metrics-v2"
        comparison = compare_recovery_records([normal, error])
        self.assertEqual(comparison["diagnostics"]["total_matched_pairs"], 0)
        self.assertEqual(comparison["diagnostics"]["conflicting_version_records"], 1)


if __name__ == "__main__":
    unittest.main()
