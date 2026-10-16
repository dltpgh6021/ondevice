"""Offline checks for paired precision comparisons and GPU memory sampling."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import GpuMemoryMonitor
from src.precision_comparison import (
    bootstrap_ci, compare_precisions, default_reference_label, holm_adjust, mcnemar_exact_p, sign_flip_p,
)


def record(label, condition, task_id, success, *, horizon=3, valid=True, base_id=None, hostname=None,
           error_kind=None):
    return {"metadata": {"model_label": label, "hostname": hostname},
            "task": {"id": task_id, "base_id": base_id or task_id},
            "trace": {"condition": condition, "protocol_version": "test-protocol", "error_kind": error_kind},
            "score": {"strict_success": success, "horizon": horizon, "condition": condition,
                      "valid_for_behavior_metrics": valid, "task_id": task_id, "error_kind": error_kind}}


class PrecisionComparisonTests(unittest.TestCase):
    def test_mcnemar_exact_values(self):
        self.assertEqual(mcnemar_exact_p(0, 0), 1.0)
        self.assertEqual(mcnemar_exact_p(5, 0), 0.0625)
        self.assertEqual(mcnemar_exact_p(3, 3), 1.0)

    def test_bootstrap_is_deterministic_and_bounded(self):
        values = [0, 0, -1, -1, 0, 1, -1, 0]
        first, second = bootstrap_ci(values), bootstrap_ci(values)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], sum(values) / len(values))
        self.assertGreaterEqual(first[1], sum(values) / len(values))
        self.assertIsNone(bootstrap_ci([]))

    def test_success_difference_is_paired_by_task(self):
        records = [record("self-FP16", "free", f"t{i}", True) for i in range(4)]
        records += [record("self-Q4", "free", f"t{i}", i < 2) for i in range(4)]
        records.append(record("self-Q4", "free", "unpaired", True))
        row = compare_precisions(records, "self-FP16")["success_differences"][0]
        self.assertEqual(row["n_paired"], 4)
        self.assertEqual(row["success_difference"], -0.5)
        self.assertEqual((row["only_reference_succeeded"], row["only_model_succeeded"]), (2, 0))

    def test_invalid_runs_are_excluded_from_both_sides(self):
        records = [record("self-FP16", "free", "t1", True), record("self-FP16", "free", "t2", True),
                   record("self-Q4", "free", "t1", False), record("self-Q4", "free", "t2", False, valid=False)]
        row = compare_precisions(records, "self-FP16")["success_differences"][0]
        self.assertEqual(row["n_paired"], 1)

    def test_recovery_penalty_interaction(self):
        records = []
        for task in ("r1", "r2"):
            records += [record("self-FP16", "normal_restart", task, True), record("self-FP16", "error_restart", task, True),
                        record("self-Q4", "normal_restart", task, True), record("self-Q4", "error_restart", task, task == "r1")]
        row = compare_precisions(records, "self-FP16")["recovery_penalty_differences"][0]
        self.assertEqual(row["reference_recovery_penalty"], 0.0)
        self.assertEqual(row["model_recovery_penalty"], 0.5)
        self.assertEqual(row["penalty_difference"], 0.5)

    def test_duplicate_runs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare_precisions([record("self-FP16", "free", "t1", True)] * 2, "self-FP16")

    def test_default_reference_requires_one_fp16_label(self):
        self.assertEqual(default_reference_label(["self-FP16", "self-Q4"]), "self-FP16")
        self.assertIsNone(default_reference_label(["self-FP16", "reference-FP16"]))
        self.assertIsNone(default_reference_label(["self-Q8", "self-Q4"]))


    def test_sign_flip_matches_mcnemar_for_one_sided_binary_differences(self):
        self.assertEqual(sign_flip_p([-1] * 5), mcnemar_exact_p(5, 0))
        self.assertEqual(sign_flip_p([0, 0, 0]), 1.0)
        self.assertEqual(sign_flip_p([1, -1]), 1.0)

    def test_holm_adjustment_is_monotone_and_capped(self):
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(holm_adjust([0.5, 0.9]), [1.0, 1.0])

    def test_precision_by_horizon_interaction_pairs_base_scenarios(self):
        records = []
        for index in range(6):
            base = f"shopping_{index:03d}"
            for horizon in (1, 5):
                task_id = f"{base}_h{horizon}"
                records.append(record("self-FP16", "free", task_id, True, horizon=horizon, base_id=base))
                # The quantized model only fails at the long horizon.
                records.append(record("self-Q4", "free", task_id, horizon == 1, horizon=horizon, base_id=base))
        result = compare_precisions(records, "self-FP16")
        row = next(item for item in result["horizon_interactions"]
                   if item["short_horizon"] == 1 and item["long_horizon"] == 5)
        self.assertEqual(row["n_scenarios"], 6)
        self.assertEqual((row["gap_short"], row["gap_long"], row["interaction"]), (0.0, -1.0, -1.0))
        self.assertEqual(row["sign_flip_p"], 0.03125)
        self.assertIn("sign_flip_holm_p", row)
        self.assertIn("mcnemar_holm_p", result["success_differences"][0])

    def test_recovery_horizons_pair_by_scenario_across_fixture_ids(self):
        records = []
        for base in ("travel_001", "travel_002"):
            for horizon in (1, 3):
                task_id = f"recovery_{base}_h{horizon}"
                for label in ("self-FP16", "self-Q4"):
                    records.append(record(label, "normal_restart", task_id, True, horizon=horizon,
                                          base_id=f"{base}_h{horizon}"))
        rows = compare_precisions(records, "self-FP16")["horizon_interactions"]
        self.assertEqual(rows[0]["n_scenarios"], 2)

    def test_mixed_machines_are_rejected_unless_allowed(self):
        records = [record("self-FP16", "free", "t1", True, hostname="laptop"),
                   record("self-Q4", "free", "t1", False, hostname="gpu-server")]
        with self.assertRaisesRegex(ValueError, "different machines"):
            compare_precisions(records, "self-FP16")
        allowed = compare_precisions(records, "self-FP16", allow_mixed_configuration=True)
        self.assertEqual(allowed["success_differences"][0]["n_paired"], 1)

    def test_error_kinds_are_compared_separately(self):
        records = []
        for label, wrong_option_success in (("self-FP16", True), ("self-Q4", False)):
            records.append(record(label, "normal_restart", "r1", True))
            records.append(record(label, "error_restart", "r1", True, error_kind="violating_candidate"))
            records.append(record(label, "error_restart", "r1", wrong_option_success, error_kind="wrong_option"))
        result = compare_precisions(records, "self-FP16")
        cells = {row["condition"]: row["success_difference"] for row in result["success_differences"]}
        self.assertEqual(cells["error_restart:violating_candidate"], 0.0)
        self.assertEqual(cells["error_restart:wrong_option"], -1.0)
        penalties = {row["error_condition"]: row["penalty_difference"] for row in result["recovery_penalty_differences"]}
        self.assertEqual(penalties, {"error_restart:violating_candidate": 0.0, "error_restart:wrong_option": 1.0})


class GpuMemoryMonitorTests(unittest.TestCase):
    def test_disabled_without_gpu_index(self):
        monitor = GpuMemoryMonitor(None)
        monitor.start()
        self.assertIsNone(monitor.stop())

    def test_peak_is_recorded_from_nvidia_smi(self):
        readings = iter(["1200\n", "1850\n", "1500\n"] + ["1400\n"] * 100)

        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], 0, stdout=next(readings), stderr="")

        monitor = GpuMemoryMonitor(0)
        with patch("src.experiment.subprocess.run", side_effect=fake_run):
            for _ in range(3):
                monitor.sample_once()
        self.assertEqual(monitor._peak_mb, 1850.0)

    def test_missing_nvidia_smi_leaves_null(self):
        monitor = GpuMemoryMonitor(0)
        with patch("src.experiment.subprocess.run", side_effect=FileNotFoundError("nvidia-smi")):
            self.assertIsNone(monitor.sample_once())
        self.assertIsNone(monitor._peak_mb)


if __name__ == "__main__":
    unittest.main()
