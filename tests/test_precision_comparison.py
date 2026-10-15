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
    bootstrap_ci, compare_precisions, default_reference_label, mcnemar_exact_p,
)


def record(label, condition, task_id, success, *, horizon=3, valid=True):
    return {"metadata": {"model_label": label},
            "task": {"id": task_id},
            "trace": {"condition": condition, "protocol_version": "test-protocol"},
            "score": {"strict_success": success, "horizon": horizon, "condition": condition,
                      "valid_for_behavior_metrics": valid, "task_id": task_id}}


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
