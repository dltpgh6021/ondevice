"""Offline selection checks; no live server, model weights or GPU are needed."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_experiment
from src.client import ScriptedClient
from src.dataset_spec import BASE_SCENARIOS, DATASET_VERSION, EXECUTION_TASKS
from src.tasks import load_tasks


class TaskSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.task_path = ROOT / "tasks" / "tasks.json"
        cls.tasks = load_tasks(cls.task_path)
        cls.validation = {
            "language": "en", "dataset_version": DATASET_VERSION,
            "dataset_sha256": "offline-dataset-hash",
            "tasks_file_sha256": "offline-file-hash",
            "validator_version": "dataset-validator-v1",
        }

    def setUp(self) -> None:
        scratch_root = ROOT / "results" / "selection-tests"
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(prefix="ondevice-selection-", dir=scratch_root)
        self.assertTrue(Path(self.scratch.name).resolve().is_relative_to(scratch_root.resolve()))
        self.addCleanup(self.scratch.cleanup)

    def test_default_type_selection_preserves_original_order(self) -> None:
        actual = run_experiment.select_tasks(self.tasks, task_type="execution")
        self.assertEqual(actual, [task for task in self.tasks if task.task_type == "execution"])
        self.assertEqual(len(actual), EXECUTION_TASKS)

    def test_all_type_without_filters_preserves_all_120(self) -> None:
        self.assertEqual(run_experiment.select_tasks(self.tasks, task_type="all"), self.tasks)

    def test_horizon_selects_one_task_per_scenario_of_each_requested_type(self) -> None:
        for task_type, horizons in (("execution", (1, 3, 5)), ("recovery", (1, 3))):
            for horizon in horizons:
                with self.subTest(task_type=task_type, horizon=horizon):
                    selected = run_experiment.select_tasks(self.tasks, task_type=task_type, horizon=horizon)
                    self.assertEqual(len(selected), BASE_SCENARIOS)
                    self.assertTrue(all(task.task_type == task_type and task.horizon == horizon for task in selected))

    def test_recovery_has_no_h5_fixture(self) -> None:
        with self.assertRaises(SystemExit):
            run_experiment.select_tasks(self.tasks, task_type="recovery", horizon=5)

    def test_all_type_intersects_horizon_without_reordering(self) -> None:
        selected = run_experiment.select_tasks(self.tasks, task_type="all", horizon=3)
        self.assertEqual(selected, [task for task in self.tasks if task.horizon == 3])
        self.assertEqual(len(selected), 2 * BASE_SCENARIOS)

    def test_limit_is_applied_after_horizon_and_type(self) -> None:
        selected = run_experiment.select_tasks(self.tasks, task_type="recovery", horizon=3, limit=1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].task_id, "recovery_shopping_001_h3")

    def test_task_id_selects_exact_existing_id(self) -> None:
        selected = run_experiment.select_tasks(self.tasks, task_type="execution", task_id="shopping_004_h3")
        self.assertEqual([task.task_id for task in selected], ["shopping_004_h3"])

    def test_task_id_can_intersect_horizon_and_limit(self) -> None:
        selected = run_experiment.select_tasks(self.tasks, task_type="recovery", horizon=3,
                                               task_id="recovery_travel_003_h3", limit=1)
        self.assertEqual([task.task_id for task in selected], ["recovery_travel_003_h3"])

    def test_unknown_id_and_prefixes_are_not_automatically_corrected(self) -> None:
        for task_id in ("shopping_001", "shopping_001_h30", "SHOPPING_001_H3", "missing_task"):
            with self.subTest(task_id=task_id), self.assertRaises(SystemExit):
                run_experiment.select_tasks(self.tasks, task_type="all", task_id=task_id)

    def test_exact_id_must_match_selected_type(self) -> None:
        for task_type, task_id in (("execution", "recovery_shopping_001_h3"),
                                   ("recovery", "shopping_001_h3")):
            with self.subTest(task_type=task_type), self.assertRaises(SystemExit):
                run_experiment.select_tasks(self.tasks, task_type=task_type, task_id=task_id)

    def test_exact_id_must_match_selected_horizon(self) -> None:
        with self.assertRaises(SystemExit):
            run_experiment.select_tasks(self.tasks, task_type="execution", horizon=5,
                                        task_id="shopping_001_h3")

    def test_selection_does_not_mutate_input_tasks(self) -> None:
        before = copy.deepcopy(self.tasks)
        run_experiment.select_tasks(self.tasks, task_type="recovery", horizon=3, limit=1)
        self.assertEqual(self.tasks, before)

    def test_cli_rejects_unsupported_horizon_before_model_client(self) -> None:
        arguments = ["run_experiment.py", "--model-label", "offline", "--condition", "free", "--horizon", "2"]
        with patch.object(sys, "argv", arguments), patch.object(run_experiment, "LlamaCppClient") as client, \
                redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            run_experiment.main()
        client.assert_not_called()

    def test_actual_full_dataset_is_validated_before_any_selection(self) -> None:
        rows = json.loads(self.task_path.read_text(encoding="utf-8"))
        rows[-1]["horizon"] = True
        fixture = Path(self.scratch.name) / "invalid-tasks.json"
        fixture.write_text(json.dumps(rows), encoding="utf-8")
        output = Path(self.scratch.name) / "must-not-exist.jsonl"
        arguments = ["run_experiment.py", "--tasks", str(fixture), "--model-label", "offline",
                     "--condition", "free", "--horizon", "3", "--task-id", "shopping_001_h3",
                     "--limit", "1", "--output", str(output)]
        errors = StringIO()
        with patch.object(sys, "argv", arguments), patch.object(run_experiment, "select_tasks") as selector, \
                patch.object(run_experiment, "LlamaCppClient") as client, redirect_stderr(errors):
            status = run_experiment.main()
        self.assertEqual(status, 1)
        selector.assert_not_called()
        client.assert_not_called()
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".preflight.json").exists())
        self.assertIs(json.loads(errors.getvalue())["valid"], False)

    def test_bad_exact_selection_stops_before_client_and_output(self) -> None:
        cases = (("free", "shopping_999_h3", "3"),
                 ("free", "recovery_shopping_001_h3", "3"),
                 ("normal_restart", "shopping_001_h3", "3"),
                 ("error_restart", "recovery_shopping_001_h3", "5"))
        for index, (condition, task_id, horizon) in enumerate(cases):
            with self.subTest(condition=condition, task_id=task_id, horizon=horizon):
                output = Path(self.scratch.name) / f"rejected-{index}.jsonl"
                arguments = ["run_experiment.py", "--model-label", "offline", "--condition", condition,
                             "--task-id", task_id, "--horizon", horizon, "--output", str(output)]
                with patch.object(sys, "argv", arguments), \
                        patch.object(run_experiment, "load_validated_tasks", return_value=(self.tasks, self.validation)), \
                        patch.object(run_experiment, "LlamaCppClient") as client, \
                        patch.object(run_experiment, "prepare_run") as preparation, self.assertRaises(SystemExit):
                    run_experiment.main()
                client.assert_not_called()
                preparation.assert_not_called()
                self.assertFalse(output.exists())
                self.assertFalse(output.with_suffix(".preflight.json").exists())

    def _run_scripted_selection(self, *, condition: str, horizon: int, task_id: str | None = None) -> tuple[dict, dict]:
        task_type = "execution" if condition == "free" else "recovery"
        selected = next(task for task in self.tasks if task.task_type == task_type and task.horizon == horizon
                        and (task_id is None or task.task_id == task_id))
        actions = selected.recovery_actions if condition in {"error_restart", "recovery"} else selected.gold_actions
        # selected resolves to the default error kind; run just that one so one record is written.
        scripted = ScriptedClient([{"type": "tool_call", "name": item["tool"], "arguments": item["arguments"]}
                                   for item in actions] + [{"type": "finish", "answer": "done"}])
        output = Path(self.scratch.name) / f"{condition}-h{horizon}.jsonl"
        arguments = ["run_experiment.py", "--tasks", str(self.task_path), "--model-label", "scripted-not-qwen",
                     "--condition", condition, "--horizon", str(horizon), "--limit", "1", "--output", str(output)]
        if task_id:
            arguments.extend(["--task-id", task_id])
        if condition in {"error_restart", "recovery"}:
            arguments.extend(["--error-kind", selected.error_kind])
        stdout = StringIO()
        with patch.object(sys, "argv", arguments), \
                patch.object(run_experiment, "load_validated_tasks", return_value=(self.tasks, self.validation)), \
                patch.object(run_experiment, "LlamaCppClient", return_value=scripted), \
                patch.object(run_experiment, "prepare_run", return_value={"preparation_protocol": "offline-mock"}), \
                redirect_stdout(stdout), redirect_stderr(StringIO()):
            status = run_experiment.main()
        self.assertEqual(status, 0)
        lines = output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        summary = json.loads(stdout.getvalue())
        self.assertEqual(record["task"]["id"], selected.task_id)
        self.assertTrue(record["score"]["strict_success"])
        self.assertEqual(record["metadata"]["task_selection"], {
            "task_type": task_type, "horizon": horizon, "task_id": task_id, "limit": 1,
            "selected_count": 1, "selected_task_ids": [selected.task_id],
        })
        self.assertEqual(summary["selected_count"], 1)
        self.assertEqual(summary["selected_task_ids"], [selected.task_id])
        return record, summary

    def test_h3_h5_cli_runs_actual_selected_tasks_for_all_three_conditions(self) -> None:
        for horizon, conditions in ((3, ("free", "normal_restart", "error_restart")), (5, ("free",))):
            for condition in conditions:
                with self.subTest(condition=condition, horizon=horizon):
                    record, _ = self._run_scripted_selection(condition=condition, horizon=horizon)
                    self.assertEqual(record["task"]["horizon"], horizon)
                    self.assertEqual(record["task"]["task_type"], "execution" if condition == "free" else "recovery")
                    expected_calls = (record["task"]["recovery_min_calls"] if condition == "error_restart"
                                      else horizon)
                    self.assertEqual(record["score"]["executed_tool_call_count"], expected_calls)

    def test_error_condition_runs_every_error_kind_by_default(self) -> None:
        fixture = next(task for task in self.tasks if task.task_type == "recovery" and task.horizon == 3)
        responses = []
        for kind in fixture.error_variants:
            resolved = fixture.with_error_kind(kind)
            responses += [{"type": "tool_call", "name": item["tool"], "arguments": item["arguments"]}
                          for item in resolved.recovery_actions] + [{"type": "finish", "answer": "done"}]
        output = Path(self.scratch.name) / "all-kinds.jsonl"
        arguments = ["run_experiment.py", "--tasks", str(self.task_path), "--model-label", "scripted-not-qwen",
                     "--condition", "error_restart", "--task-id", fixture.task_id, "--output", str(output)]
        with patch.object(sys, "argv", arguments), \
                patch.object(run_experiment, "load_validated_tasks", return_value=(self.tasks, self.validation)), \
                patch.object(run_experiment, "LlamaCppClient", return_value=ScriptedClient(responses)), \
                patch.object(run_experiment, "prepare_run", return_value={"preparation_protocol": "offline-mock"}), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(run_experiment.main(), 0)
        records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([record["score"]["error_kind"] for record in records], list(fixture.error_variants))
        self.assertTrue(all(record["score"]["strict_success"] for record in records))
        self.assertEqual([record["score"]["recovery_min_calls"] for record in records], [3, 2])

    def test_exact_nonfirst_task_id_is_used_by_real_offline_run(self) -> None:
        record, _ = self._run_scripted_selection(condition="free", horizon=3, task_id="place_005_h3")
        self.assertEqual(record["task"]["id"], "place_005_h3")

    def test_recovery_alias_preserves_selection_and_records_canonical_condition(self) -> None:
        record, _ = self._run_scripted_selection(condition="recovery", horizon=3,
                                                task_id="recovery_travel_003_h3")
        self.assertEqual(record["score"]["condition"], "error_restart")
        self.assertEqual(record["metadata"]["condition"], "error_restart")


if __name__ == "__main__":
    unittest.main()
