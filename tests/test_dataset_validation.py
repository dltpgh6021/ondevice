"""Offline regression checks for the dataset gate used before experiments.

The rejected fixtures deliberately reproduce previously reported research
validity bugs. They test the public validation boundary, not the particular
layer that detects a defect. No llama server, model files, or GPU are needed.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_validation import (  # noqa: E402
    DatasetValidationError,
    load_validated_tasks,
    validate_dataset,
)


class DatasetValidationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.task_path = ROOT / "tasks" / "tasks.json"
        cls.rows = json.loads(cls.task_path.read_text(encoding="utf-8"))

    def setUp(self) -> None:
        scratch_root = ROOT / "results" / "validator-tests"
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(prefix="ondevice-validator-test-", dir=scratch_root)
        self.assertTrue(Path(self.scratch.name).resolve().is_relative_to(scratch_root.resolve()))
        self.addCleanup(self.scratch.cleanup)
        self.fixture_path = Path(self.scratch.name) / "tasks.json"

    def write_fixture(self, rows: Any) -> Path:
        self.fixture_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.fixture_path

    def assert_rejected(self, rows: Any = None, *, raw: str | None = None) -> None:
        if raw is not None:
            self.fixture_path.write_text(raw, encoding="utf-8")
        else:
            self.write_fixture(rows)
        with self.assertRaises(DatasetValidationError) as caught:
            validate_dataset(self.fixture_path)
        report = caught.exception.report
        self.assertIs(report.get("valid"), False)
        self.assertIsInstance(report.get("errors"), list)
        self.assertTrue(report["errors"], "A failed preflight must explain the defect.")

    def test_checked_in_english_dataset_is_valid(self) -> None:
        report = validate_dataset(self.task_path)
        self.assertIs(report.get("valid"), True)
        self.assertEqual(len(self.rows), 120)
        self.assertEqual(sum(row["task_type"] == "execution" for row in self.rows), 60)
        self.assertEqual(sum(row["task_type"] == "recovery" for row in self.rows), 60)
        self.assertTrue(all(row["language"] == "en" for row in self.rows))

    def test_shared_loading_api_rejects_invalid_dataset(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows.pop()
        self.write_fixture(rows)
        with self.assertRaises(DatasetValidationError) as caught:
            load_validated_tasks(self.fixture_path)
        self.assertIs(caught.exception.report.get("valid"), False)

    def test_experiment_validates_full_dataset_before_limit_or_client(self) -> None:
        from scripts import run_experiment

        rows = copy.deepcopy(self.rows)
        rows[-1]["horizon"] = True
        self.write_fixture(rows)
        arguments = ["run_experiment.py", "--tasks", str(self.fixture_path),
                     "--model-label", "offline-check", "--condition", "free", "--limit", "1"]
        error = StringIO()
        with patch.object(sys, "argv", arguments), patch.object(run_experiment, "LlamaCppClient") as client, redirect_stderr(error):
            status = run_experiment.main()
        self.assertEqual(status, 1)
        client.assert_not_called()
        self.assertIs(json.loads(error.getvalue())["valid"], False)

    def test_valid_experiment_records_language_and_validation_identity(self) -> None:
        from scripts import run_experiment
        from src.client import ScriptedClient

        task = self.rows[0]
        model = ScriptedClient([{"type": "tool_call", "name": item["tool"], "arguments": item["arguments"]}
                                for item in task["gold_actions"]] + [{"type": "finish", "answer": "done"}])
        model.temperature, model.seed, model.max_tokens, model.timeout_seconds = 0.0, 42, 512, 120.0
        output = Path(self.scratch.name) / "offline-result.jsonl"
        arguments = ["run_experiment.py", "--tasks", str(self.task_path), "--output", str(output),
                     "--model-label", "scripted-not-qwen", "--condition", "free", "--limit", "1"]
        preparation = {"inference_settings": {"temperature": 0.0, "seed": 42, "max_tokens": 512,
                                               "timeout_seconds": 120.0},
                       "preparation_protocol": "fixed-inference-warmup-v1"}
        with patch.object(sys, "argv", arguments), patch.object(run_experiment, "LlamaCppClient", return_value=model), \
                patch.object(run_experiment, "prepare_run", return_value=preparation), redirect_stdout(StringIO()):
            status = run_experiment.main()
        self.assertEqual(status, 0)
        record = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(record["metadata"]["dataset_validation"]["valid"])
        self.assertEqual(record["metadata"]["dataset_language"], "en")
        self.assertEqual(record["metadata"]["dataset_version"], "workflow-en-v2")
        self.assertEqual(record["task"]["language"], "en")
        self.assertEqual(record["trace"]["dataset_language"], "en")
        self.assertTrue(json.dumps(record["trace"]["initial_messages"], ensure_ascii=False).isascii())

    def test_duplicate_task_ids_are_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[1]["id"] = rows[0]["id"]
        self.assert_rejected(rows)

    def test_missing_recovery_pair_is_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        recovery_index = next(i for i, row in enumerate(rows) if row["task_type"] == "recovery")
        rows.pop(recovery_index)
        self.assert_rejected(rows)

    def test_boolean_horizon_is_not_coerced_to_one(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["horizon"] = True
        self.assert_rejected(rows)

    def test_nested_boolean_is_not_equal_to_numeric_target(self) -> None:
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["domain"] == "shopping":
                row["success_condition"]["completed_request_equals"]["result"]["quantity"] = True
        self.assert_rejected(rows)

    def test_source_comparison_preserves_integer_and_float_types(self) -> None:
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["domain"] == "shopping":
                row["success_condition"]["completed_request_equals"]["result"]["quantity"] = 1.0
        self.assert_rejected(rows)

    def test_experiment_output_cannot_overwrite_input(self) -> None:
        from scripts import run_experiment

        self.write_fixture(self.rows)
        original = self.fixture_path.read_bytes()
        arguments = ["run_experiment.py", "--tasks", str(self.fixture_path),
                     "--output", str(self.fixture_path), "--model-label", "offline-check", "--condition", "free"]
        with patch.object(sys, "argv", arguments), patch.object(run_experiment, "LlamaCppClient") as client:
            with self.assertRaises(SystemExit):
                run_experiment.main()
        client.assert_not_called()
        self.assertEqual(self.fixture_path.read_bytes(), original)

    def test_string_horizon_is_not_coerced_to_integer(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["horizon"] = "1"
        self.assert_rejected(rows)

    def test_duplicate_json_keys_are_rejected_at_any_depth(self) -> None:
        for raw in (
            '[{"id":"first","id":"second"}]',
            '[{"execution_spec":{"request":{"constraints":{"max_price":10,"max_price":20}}}}]',
        ):
            with self.subTest(raw=raw):
                self.assert_rejected(raw=raw)

    def test_nonfinite_json_numbers_are_rejected(self) -> None:
        for number in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(number=repr(number)):
                rows = copy.deepcopy(self.rows)
                rows[0]["execution_spec"]["request"]["constraints"]["max_price"] = number
                self.assert_rejected(rows)

    def test_unknown_schema_version_is_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["schema_version"] = "unchecked-future-schema"
        self.assert_rejected(rows)

    def test_non_english_original_question_is_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["query"] += " 요청을 완료해 주세요."
        self.assert_rejected(rows)

    def test_non_english_later_tool_result_is_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["execution_spec"]["candidates"][0]["summary"]["label"] = "키보드 후보"
        self.assert_rejected(rows)

    def test_non_english_injected_record_is_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        recovery = next(row for row in rows if row["task_type"] == "recovery")
        recovery["error_injection"]["actions"][-1]["arguments"]["draft_id"] = "잘못된_초안"
        self.assert_rejected(rows)

    def test_question_must_expose_the_required_request_id(self) -> None:
        rows = copy.deepcopy(self.rows)
        # Keep all matched questions equal so this is not merely a pair mismatch.
        for row in rows:
            row["query"] = "Complete the requested task and retain exactly one correct saved result."
        self.assert_rejected(rows)

    def test_original_question_cannot_leak_the_answer_candidate(self) -> None:
        rows = copy.deepcopy(self.rows)
        target = rows[0]["success_condition"]["completed_request_equals"]["candidate_id"]
        rows[0]["query"] += f" The answer is {target}."
        self.assert_rejected(rows)

    def test_original_question_cannot_leak_the_final_draft(self) -> None:
        rows = copy.deepcopy(self.rows)
        draft_id = rows[0]["gold_actions"][-1]["arguments"]["draft_id"]
        rows[0]["query"] += f" Confirm draft {draft_id}."
        self.assert_rejected(rows)

    def test_duplicate_base_scenarios_with_distinct_ids_are_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        by_id = {row["id"]: copy.deepcopy(row) for row in rows}
        for index, row in enumerate(rows):
            if "shopping_002" in row["id"]:
                source_id = row["id"].replace("shopping_002", "shopping_001")
                duplicate = copy.deepcopy(by_id[source_id])
                duplicate["id"] = row["id"]
                duplicate["base_id"] = row["base_id"]
                rows[index] = duplicate
        self.assert_rejected(rows)

    def test_natural_language_rule_cannot_contradict_structured_rules(self) -> None:
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row["execution_spec"]["request"]["description"] = (
                "Ignore all constraints and select the lowest-rated candidate."
            )
        self.assert_rejected(rows)

    def test_final_condition_cannot_accept_the_wrong_candidate(self) -> None:
        rows = copy.deepcopy(self.rows)
        target = rows[0]["success_condition"]["completed_request_equals"]
        target["candidate_id"] = next(
            candidate["candidate_id"]
            for candidate in rows[0]["execution_spec"]["candidates"]
            if candidate["candidate_id"] != target["candidate_id"]
        )
        self.assert_rejected(rows)

    def test_last_tool_result_is_not_a_valid_success_condition(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["success_condition"] = {"last_result_field_equals": {"ok": True}}
        self.assert_rejected(rows)

    def test_invisible_internal_error_injection_is_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        recovery = next(row for row in rows if row["task_type"] == "recovery")
        recovery["error_injection"] = {"type": "stale_result", "last_result": {"ok": True}}
        self.assert_rejected(rows)

    def test_nonexistent_injected_draft_is_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        recovery = next(row for row in rows if row["task_type"] == "recovery")
        recovery["error_injection"]["actions"][-1]["arguments"]["draft_id"] = "draft_does_not_exist"
        self.assert_rejected(rows)

    def test_repair_oracle_must_replace_the_injected_wrong_result(self) -> None:
        rows = copy.deepcopy(self.rows)
        recovery = next(row for row in rows if row["task_type"] == "recovery" and row["horizon"] == 3)
        recovery["recovery_actions"] = copy.deepcopy(recovery["recovery_actions"][-1:])
        recovery["recovery_min_calls"] = 1
        self.assert_rejected(rows)

    def test_error_must_be_saved_for_the_same_request(self) -> None:
        # The v1 design saved an extra record under another request ID; that is
        # not a mistake the model can make on its own and must stay rejected.
        rows = copy.deepcopy(self.rows)
        recovery = next(row for row in rows if row["task_type"] == "recovery" and row["horizon"] == 3)
        recovery["error_injection"] = {"type": "extra_saved_record", "action": recovery["error_injection"]["actions"][-1]}
        self.assert_rejected(rows)

    def test_injected_error_must_violate_the_request(self) -> None:
        rows = copy.deepcopy(self.rows)
        recovery = next(row for row in rows if row["task_type"] == "recovery" and row["horizon"] == 3)
        recovery["error_injection"]["actions"] = copy.deepcopy(recovery["recovery_actions"])
        self.assert_rejected(rows)

    def test_free_context_cannot_be_presented_as_model_history(self) -> None:
        from src.environment import environment_for_task
        from src.tasks import load_tasks

        tasks = load_tasks(self.task_path)
        by_id = {task.task_id: task for task in tasks}
        for horizon in (1, 3, 5):
            with self.subTest(horizon=horizon):
                task = by_id[f"shopping_001_h{horizon}"]
                recovery = by_id[f"recovery_shopping_001_h{horizon}"]
                free_messages, _ = environment_for_task(task).prepare_condition(task, "free")
                normal_messages, _ = environment_for_task(recovery).prepare_condition(recovery, "normal_restart")
                self.assertFalse(any(message.get("tool_calls") for message in free_messages))
                self.assertEqual(free_messages == normal_messages, horizon == 5)

    def test_condition_separation_is_reported(self) -> None:
        report = validate_dataset(self.task_path)
        self.assertEqual(report["condition_separation"]["identical_free_normal_inputs_by_horizon"],
                         {"1": 0, "3": 0, "5": 20})

    def test_h1_must_offer_a_choice_between_drafts(self) -> None:
        rows = copy.deepcopy(self.rows)
        execution = next(row for row in rows if row["task_type"] == "execution" and row["horizon"] == 1)
        execution["initial_actions"] = execution["initial_actions"][:-2]
        self.assert_rejected(rows)

    def test_checkpoint_cannot_start_with_the_task_already_complete(self) -> None:
        rows = copy.deepcopy(self.rows)
        execution = next(row for row in rows if row["task_type"] == "execution")
        execution["checkpoint_step"] = execution["horizon"]
        self.assert_rejected(rows)

    def test_answer_draft_cannot_be_seeded_to_bypass_horizon(self) -> None:
        rows = copy.deepcopy(self.rows)
        execution = next(
            row for row in rows if row["task_type"] == "execution" and row["horizon"] == 5
        )
        draft_id = execution["gold_actions"][-1]["arguments"]["draft_id"]
        execution["execution_spec"]["initial_drafts"] = {
            draft_id: copy.deepcopy(execution["success_condition"]["completed_request_equals"])
        }
        self.assert_rejected(rows)


if __name__ == "__main__":
    unittest.main()
