"""Offline regression tests for fixed settings and held-out warm-up preparation."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.client import InfrastructureError, LlamaCppClient, ModelOutputError
from src.run_preparation import PreparationError, prepare_run


def server_properties() -> dict[str, Any]:
    return {
        "build_info": "b9876-test-build",
        "chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}",
        "chat_template_caps": {"supports_tool_calls": True},
        "total_slots": 1,
        "default_generation_settings": {
            "n_ctx": 8192,
            "params": {"temperature": 0.8, "top_k": 40, "seed": 4294967295},
        },
        "model_path": "/models/Qwen3-1.7B-f16.gguf",
        "model_alias": "self-FP16",
        "model_ftype": "F16",
        "is_sleeping": False,
    }


class FakePreparationClient:
    def __init__(self, *, props: dict[str, Any] | None = None,
                 outcomes: list[Any] | None = None, **settings: Any) -> None:
        self.real = LlamaCppClient("http://127.0.0.1:18080", "local", **settings)
        self.timeout_seconds = self.real.timeout_seconds
        self.props = copy.deepcopy(props if props is not None else server_properties())
        self.outcomes = list(outcomes or [])
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
        self.props_calls = 0

    def request_settings(self) -> dict[str, Any]:
        return self.real.request_settings()

    def get_server_properties(self) -> dict[str, Any]:
        self.props_calls += 1
        return copy.deepcopy(self.props)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        outcome = self.outcomes.pop(0) if self.outcomes else {
            "action": {"type": "finish", "answer": "An intentionally unsolved warm-up."},
            "usage": {"prompt_tokens": 100, "completion_tokens": 7},
        }
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class ClientSettingsTests(unittest.TestCase):
    def test_fixed_request_settings_disable_thinking_caching_and_parallelism(self) -> None:
        settings = LlamaCppClient("http://localhost:18080", "local").request_settings()
        self.assertEqual(settings["temperature"], 0.0)
        self.assertEqual(settings["seed"], 42)
        self.assertEqual(settings["max_tokens"], 512)
        self.assertEqual(settings["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(settings["reasoning_effort"], "none")
        self.assertIs(settings["stream"], False)
        self.assertEqual(settings["n"], 1)
        self.assertIs(settings["parallel_tool_calls"], False)
        self.assertIs(settings["cache_prompt"], False)
        self.assertEqual(settings["samplers"], ["temperature"])
        self.assertEqual(settings["dynatemp_range"], 0)
        self.assertEqual(settings["mirostat"], 0)
        self.assertNotIn("messages", settings)
        self.assertNotIn("model", settings)
        self.assertNotIn("tools", settings)

    def test_request_settings_are_a_defensive_copy(self) -> None:
        client = LlamaCppClient("http://localhost:18080", "local")
        settings = client.request_settings()
        settings["chat_template_kwargs"]["enable_thinking"] = True
        settings["samplers"].append("top_k")
        self.assertEqual(client.request_settings()["samplers"], ["temperature"])
        self.assertIs(client.request_settings()["chat_template_kwargs"]["enable_thinking"], False)

    def test_invalid_temperatures_are_rejected(self) -> None:
        for value in (-1.0, float("nan"), float("inf"), -float("inf"), True, "0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LlamaCppClient("http://localhost", "local", temperature=value)

    def test_invalid_timeout_values_are_rejected(self) -> None:
        for value in (0, -1, float("nan"), float("inf"), True, "120"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LlamaCppClient("http://localhost", "local", timeout_seconds=value)

    def test_invalid_seed_values_are_rejected(self) -> None:
        for value in (-1, 4294967295, True, 42.5, "42"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LlamaCppClient("http://localhost", "local", seed=value)

    def test_seed_boundary_and_positive_numeric_settings_are_accepted(self) -> None:
        for seed in (0, 4294967294):
            client = LlamaCppClient("http://localhost", "local", seed=seed,
                                    temperature=0.25, max_tokens=1024, timeout_seconds=60)
            self.assertEqual(client.request_settings()["seed"], seed)
            self.assertEqual(client.request_settings()["temperature"], 0.25)

    def test_invalid_max_tokens_are_rejected(self) -> None:
        for value in (0, -1, True, 1.5, "512"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LlamaCppClient("http://localhost", "local", max_tokens=value)

    def test_chat_payload_uses_recorded_settings_exactly(self) -> None:
        client = LlamaCppClient("http://localhost:18080/", "self-Q8", temperature=0.0,
                               seed=17, max_tokens=768, timeout_seconds=31)
        messages = [{"role": "user", "content": "Hello"}]
        tools: list[dict[str, Any]] = []
        payload = {"choices": [{"message": {"content": "Done"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 2, "completion_tokens": 1}}
        with patch("src.client.urllib.request.urlopen") as opener:
            opener.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()
            client.chat(messages, tools)
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:18080/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        body = json.loads(request.data)
        self.assertEqual(body.pop("model"), "self-Q8")
        self.assertEqual(body.pop("messages"), messages)
        self.assertEqual(body.pop("tools"), tools)
        self.assertEqual(body, client.request_settings())
        self.assertEqual(opener.call_args.kwargs["timeout"], 31)

    def test_server_properties_use_get_at_server_root(self) -> None:
        client = LlamaCppClient("http://localhost:18080/", "local", timeout_seconds=19)
        properties = server_properties()
        with patch("src.client.urllib.request.urlopen") as opener:
            opener.return_value.__enter__.return_value.read.return_value = json.dumps(properties).encode()
            actual = client.get_server_properties()
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:18080/props")
        self.assertEqual(request.get_method(), "GET")
        self.assertIsNone(request.data)
        self.assertEqual(opener.call_args.kwargs["timeout"], 19)
        self.assertEqual(actual, properties)

    def test_invalid_server_properties_json_is_an_infrastructure_error(self) -> None:
        client = LlamaCppClient("http://localhost:18080", "local")
        with patch("src.client.urllib.request.urlopen") as opener:
            opener.return_value.__enter__.return_value.read.return_value = b"not JSON"
            with self.assertRaises(InfrastructureError):
                client.get_server_properties()

    def test_nonfinite_and_overflow_server_numbers_are_typed_infrastructure_errors(self) -> None:
        client = LlamaCppClient("http://localhost:18080", "local")
        for value in ("NaN", "Infinity", "-Infinity", "1e309", "-1e309"):
            with self.subTest(value=value), patch("src.client.urllib.request.urlopen") as opener:
                opener.return_value.__enter__.return_value.read.return_value = ('{"value":' + value + '}').encode()
                with self.assertRaises(InfrastructureError) as caught:
                    client.get_server_properties()
                self.assertEqual(caught.exception.kind, "response_protocol")


class PreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch_root = ROOT / "results" / "warmup-tests"
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(prefix="ondevice-preflight-", dir=scratch_root)
        self.scratch_path = Path(self.scratch.name)
        self.assertTrue(self.scratch_path.resolve().is_relative_to(scratch_root.resolve()))
        self.addCleanup(self.scratch.cleanup)
        self.lock = self.scratch_path / "settings.json"
        self.report = self.scratch_path / "preflight.json"
        self.report_number = 0
        self.hardware = {"hostname": "reference-pc", "platform": "test-os", "machine": "x86_64",
                         "processor": "test-cpu", "python_version": "3.11-test"}

    def prepare(self, client: FakePreparationClient | None = None, **overrides: Any) -> dict[str, Any]:
        self.report_number += 1
        self.report = self.scratch_path / f"preflight_{self.report_number}.json"
        kwargs = {"expected_context_size": 8192, "warmup_requests": 2,
                  "settings_lock_path": self.lock, "report_path": self.report,
                  "hardware": self.hardware}
        kwargs.update(overrides)
        return prepare_run(client or FakePreparationClient(), **kwargs)

    def assert_failed_before_inference(self, client: FakePreparationClient, **kwargs: Any) -> None:
        with self.assertRaises(PreparationError):
            self.prepare(client, **kwargs)
        self.assertEqual(client.calls, [])
        self.assertTrue(self.report.is_file(), "Failed preflight must save a diagnostic sidecar.")

    def test_success_records_settings_server_identity_and_warmup_metadata(self) -> None:
        client = FakePreparationClient()
        result = self.prepare(client)
        self.assertEqual(result["preparation_protocol"], "fixed-inference-warmup-v1")
        self.assertEqual(result["llama_cpp_version"], "b9876-test-build")
        self.assertEqual(result["inference_settings"]["max_tokens"], 512)
        self.assertEqual(result["inference_settings"]["timeout_seconds"], 120)
        self.assertIsInstance(result["server_configuration"], dict)
        self.assertIsInstance(result["warmup_settings"], dict)
        self.assertIn("warmup", result)
        self.assertEqual(Path(result["preflight_report_path"]), self.report)
        self.assertEqual(len(result["settings_lock_sha256"]), 64)
        self.assertTrue(self.lock.is_file())
        self.assertTrue(self.report.is_file())
        self.assertEqual(client.props_calls, 1)
        self.assertEqual(len(client.calls), 2)

    def test_warmups_use_fresh_held_out_english_histories(self) -> None:
        client = FakePreparationClient()
        self.prepare(client)
        first_messages, first_tools = client.calls[0]
        second_messages, second_tools = client.calls[1]
        self.assertEqual(first_messages, second_messages)
        self.assertEqual(first_tools, second_tools)
        self.assertTrue(first_tools)
        self.assertFalse(any(message.get("role") == "tool" for message in second_messages))
        encoded = json.dumps(client.calls, ensure_ascii=False)
        self.assertTrue(encoded.isascii())
        for leaked in ("shopping_001", "recovery_", "Request ID:", "keyboard_02", "gold_actions"):
            self.assertNotIn(leaked, encoded)

    def test_weak_model_dummy_answer_is_not_a_preflight_failure(self) -> None:
        client = FakePreparationClient(outcomes=[
            {"action": {"type": "finish", "answer": "I cannot solve this."}, "usage": {}},
            {"action": {"type": "finish", "answer": "Wrong answer."}, "usage": {}},
        ])
        result = self.prepare(client)
        self.assertEqual(len(client.calls), 2)
        self.assertIn("warmup", result)

    def test_model_output_errors_are_recorded_without_selective_exclusion(self) -> None:
        payload = {"choices": [{"message": {"content": "{"}, "finish_reason": "length"}],
                   "usage": {"prompt_tokens": 101, "completion_tokens": 512}}
        client = FakePreparationClient(outcomes=[ModelOutputError("truncated", kind="output_truncated", payload=payload)])
        result = self.prepare(client)
        self.assertEqual(len(client.calls), 2)
        self.assertIn("output_truncated", json.dumps(result["warmup"]))

    def test_infrastructure_failure_aborts_and_records_diagnostic(self) -> None:
        client = FakePreparationClient(outcomes=[InfrastructureError("server stopped", kind="connection_error")])
        with self.assertRaises((PreparationError, InfrastructureError)):
            self.prepare(client)
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(self.report.is_file())
        self.assertIn("server stopped", self.report.read_text(encoding="utf-8"))

    def test_missing_essential_server_properties_abort_before_warmup(self) -> None:
        for key in ("build_info", "chat_template", "total_slots", "default_generation_settings"):
            with self.subTest(key=key):
                props = server_properties()
                props.pop(key)
                self.assert_failed_before_inference(FakePreparationClient(props=props))

    def test_empty_build_or_template_is_rejected(self) -> None:
        for key in ("build_info", "chat_template"):
            with self.subTest(key=key):
                props = server_properties()
                props[key] = ""
                self.assert_failed_before_inference(FakePreparationClient(props=props))

    def test_context_mismatch_is_rejected(self) -> None:
        props = server_properties()
        props["default_generation_settings"]["n_ctx"] = 4096
        self.assert_failed_before_inference(FakePreparationClient(props=props))

    def test_boolean_or_string_context_is_rejected(self) -> None:
        for value in (True, "8192"):
            with self.subTest(value=value):
                props = server_properties()
                props["default_generation_settings"]["n_ctx"] = value
                self.assert_failed_before_inference(FakePreparationClient(props=props))

    def test_multiple_or_boolean_slots_are_rejected(self) -> None:
        for value in (2, True, "1"):
            with self.subTest(value=value):
                props = server_properties()
                props["total_slots"] = value
                self.assert_failed_before_inference(FakePreparationClient(props=props))

    def test_missing_generation_params_are_rejected(self) -> None:
        props = server_properties()
        props["default_generation_settings"].pop("params")
        self.assert_failed_before_inference(FakePreparationClient(props=props))

    def test_existing_lock_reuses_identical_configuration_without_rewriting(self) -> None:
        first = self.prepare()
        original = self.lock.read_bytes()
        second = self.prepare()
        self.assertEqual(self.lock.read_bytes(), original)
        self.assertEqual(first["settings_lock_sha256"], second["settings_lock_sha256"])

    def test_quantization_model_identity_changes_do_not_break_shared_lock(self) -> None:
        self.prepare()
        original = self.lock.read_bytes()
        props = server_properties()
        props.update({"model_path": "/models/Qwen3-1.7B-Q4_K_M.gguf", "model_alias": "self-Q4",
                      "model_ftype": "Q4_K_M", "is_sleeping": True})
        self.prepare(FakePreparationClient(props=props))
        self.assertEqual(self.lock.read_bytes(), original)

    def test_changed_request_setting_is_rejected_without_overwriting_lock(self) -> None:
        self.prepare()
        original = self.lock.read_bytes()
        self.assert_failed_before_inference(FakePreparationClient(max_tokens=1024))
        self.assertEqual(self.lock.read_bytes(), original)

    def test_changed_hardware_is_rejected_without_overwriting_lock(self) -> None:
        self.prepare()
        original = self.lock.read_bytes()
        hardware = {**self.hardware, "hostname": "another-pc"}
        self.assert_failed_before_inference(FakePreparationClient(), hardware=hardware)
        self.assertEqual(self.lock.read_bytes(), original)

    def test_changed_warmup_count_is_rejected_before_inference(self) -> None:
        self.prepare()
        self.assert_failed_before_inference(FakePreparationClient(), warmup_requests=3)

    def test_changed_server_build_or_template_is_rejected(self) -> None:
        self.prepare()
        for key in ("build_info", "chat_template"):
            with self.subTest(key=key):
                props = server_properties()
                props[key] += " changed"
                self.assert_failed_before_inference(FakePreparationClient(props=props))

    def test_corrupted_lock_is_rejected_and_preserved(self) -> None:
        self.lock.write_text("{broken", encoding="utf-8")
        self.assert_failed_before_inference(FakePreparationClient())
        self.assertEqual(self.lock.read_text(encoding="utf-8"), "{broken")

    def test_existing_report_is_not_overwritten_or_followed_by_inference(self) -> None:
        report = self.scratch_path / "existing.json"
        report.write_text("retained report", encoding="utf-8")
        client = FakePreparationClient()
        with self.assertRaises(PreparationError):
            self.prepare(client, report_path=report)
        self.assertEqual(report.read_text(encoding="utf-8"), "retained report")
        self.assertEqual(client.calls, [])
        self.assertEqual(client.props_calls, 0)

    def test_report_cannot_overwrite_shared_lock(self) -> None:
        client = FakePreparationClient()
        with self.assertRaises(PreparationError):
            self.prepare(client, report_path=self.lock)
        self.assertFalse(self.lock.exists())
        self.assertEqual(client.calls, [])
        self.assertEqual(client.props_calls, 0)

    def test_tampered_lock_configuration_is_rejected(self) -> None:
        self.prepare()
        value = json.loads(self.lock.read_text(encoding="utf-8"))
        value["configuration"]["unexpected_tampering"] = True
        self.lock.write_text(json.dumps(value), encoding="utf-8")
        original = self.lock.read_bytes()
        self.assert_failed_before_inference(FakePreparationClient())
        self.assertEqual(self.lock.read_bytes(), original)

    def test_warmup_usage_is_not_part_of_measured_task_rows(self) -> None:
        from src.experiment import run_batch
        from src.tasks import load_tasks

        task = next(task for task in load_tasks(ROOT / "tasks" / "tasks.json")
                    if task.task_type == "execution" and task.horizon == 1)
        call = task.gold_actions[0]
        client = FakePreparationClient(outcomes=[
            {"action": {"type": "finish", "answer": "warmup"},
             "usage": {"prompt_tokens": 1000, "completion_tokens": 100}},
            {"action": {"type": "finish", "answer": "warmup"},
             "usage": {"prompt_tokens": 2000, "completion_tokens": 200}},
            {"action": {"type": "tool_call", "name": call["tool"], "arguments": call["arguments"]},
             "usage": {"prompt_tokens": 3, "completion_tokens": 4}},
            {"action": {"type": "finish", "answer": "done"},
             "usage": {"prompt_tokens": 5, "completion_tokens": 6}},
        ])
        output = self.scratch_path / "measured.jsonl"
        clock = [0.0]
        original_chat = client.chat

        def clocked_chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            response = original_chat(messages, tools)
            clock[0] += 1000.0 if len(client.calls) <= 2 else 0.01
            return response

        with patch.object(client, "chat", side_effect=clocked_chat), \
                patch("time.perf_counter", side_effect=lambda: clock[0]):
            metadata = self.prepare(client)
            records = run_batch([task], client, model_label="offline-FP16", condition="free",
                                output_path=output, metadata=metadata, max_calls=2)
        trace = records[0]["trace"]
        self.assertEqual(trace["prompt_tokens"], 8)
        self.assertEqual(trace["completion_tokens"], 10)
        self.assertEqual(trace["elapsed_ms"], 20.0)
        self.assertEqual(len(trace["events"]), 2)
        self.assertTrue(records[0]["score"]["strict_success"])
        self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)
        self.assertEqual(len(client.calls), 4)

    def test_batch_never_overwrites_existing_task_results(self) -> None:
        from src.experiment import run_batch

        output = self.scratch_path / "existing.jsonl"
        output.write_text("previous research results\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            run_batch([], FakePreparationClient(), model_label="offline", condition="free",
                      output_path=output, metadata={}, max_calls=2)
        self.assertEqual(output.read_text(encoding="utf-8"), "previous research results\n")


class PreparationCliTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch_root = ROOT / "results" / "warmup-tests"
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(prefix="ondevice-cli-preflight-", dir=scratch_root)
        self.scratch_path = Path(self.scratch.name)
        self.assertTrue(self.scratch_path.resolve().is_relative_to(scratch_root.resolve()))
        self.addCleanup(self.scratch.cleanup)

    def cli_arguments(self) -> list[str]:
        return ["run_experiment.py", "--condition", "free", "--model-label", "OFFLINE-NOT-QWEN",
                "--limit", "1", "--output", str(self.scratch_path / "measured.jsonl"),
                "--settings-lock", str(self.scratch_path / "settings.json")]

    def test_parser_exposes_fixed_settings_and_default_lock(self) -> None:
        from scripts import run_experiment

        with patch.object(sys, "argv", ["run_experiment.py", "--condition", "free", "--model-label", "self-FP16"]):
            args = run_experiment.parse_args()
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.max_tokens, 512)
        self.assertEqual(args.timeout_seconds, 120.0)
        self.assertEqual(args.warmup_requests, 2)
        self.assertEqual(args.expected_context_size, 8192)
        self.assertEqual(Path(args.settings_lock).name, "experiment_settings.json")

    def test_invalid_new_cli_values_fail_before_creating_client(self) -> None:
        from scripts import run_experiment

        options = [("--temperature", "nan"), ("--temperature", "inf"), ("--temperature", "-1"),
                   ("--timeout-seconds", "nan"), ("--timeout-seconds", "0"),
                   ("--max-tokens", "0"), ("--seed", "4294967295"),
                   ("--warmup-requests", "0"), ("--warmup-requests", "-1"),
                   ("--expected-context-size", "0")]
        for flag, value in options:
            with self.subTest(flag=flag, value=value):
                argv = ["run_experiment.py", "--condition", "free", "--model-label", "offline", flag, value]
                with patch.object(sys, "argv", argv), patch.object(run_experiment, "LlamaCppClient") as client, \
                        redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                    with self.assertRaises(SystemExit):
                        run_experiment.main()
                client.assert_not_called()

    def test_full_cli_prepares_then_measures_without_warmup_rows(self) -> None:
        from scripts import run_experiment
        from src.tasks import load_tasks

        task = load_tasks(ROOT / "tasks" / "tasks.json")[0]
        client = FakePreparationClient(outcomes=[
            {"action": {"type": "finish", "answer": "warmup"}, "usage": {"prompt_tokens": 1000}},
            {"action": {"type": "finish", "answer": "warmup"}, "usage": {"prompt_tokens": 1000}},
            *[{"action": {"type": "tool_call", "name": call["tool"], "arguments": call["arguments"]},
               "usage": {"prompt_tokens": 3, "completion_tokens": 4}} for call in task.gold_actions],
            {"action": {"type": "finish", "answer": "done"}, "usage": {"prompt_tokens": 5}},
        ])
        with patch.object(sys, "argv", self.cli_arguments()), \
                patch.object(run_experiment, "LlamaCppClient", return_value=client), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(run_experiment.main(), 0)
        rows = (self.scratch_path / "measured.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 1)
        record = json.loads(rows[0])
        self.assertTrue(record["score"]["strict_success"])
        self.assertEqual(record["trace"]["prompt_tokens"], len(task.gold_actions) * 3 + 5)
        self.assertEqual(len(record["metadata"]["warmup"]), 2)
        self.assertEqual(record["metadata"]["llama_cpp_version"], "b9876-test-build")
        self.assertEqual(json.loads((self.scratch_path / "measured.preflight.json").read_text(encoding="utf-8"))["status"], "ready")

    def test_cli_warmup_infrastructure_failure_saves_only_failed_preflight(self) -> None:
        from scripts import run_experiment

        client = FakePreparationClient(outcomes=[InfrastructureError("warmup server disconnected")])
        with patch.object(sys, "argv", self.cli_arguments()), \
                patch.object(run_experiment, "LlamaCppClient", return_value=client), \
                patch.object(run_experiment, "run_batch") as batch, \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(run_experiment.main(), 1)
        batch.assert_not_called()
        self.assertFalse((self.scratch_path / "measured.jsonl").exists())
        self.assertFalse((self.scratch_path / "settings.json").exists())
        self.assertEqual(json.loads((self.scratch_path / "measured.preflight.json").read_text(encoding="utf-8"))["status"], "failed")


if __name__ == "__main__":
    unittest.main()
