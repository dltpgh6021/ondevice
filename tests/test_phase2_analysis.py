"""Integration test for paired estimates, including the length contrast."""
import unittest
from unittest.mock import patch

from scripts.analyze_phase2 import analyze
from scripts.phase2 import score
from src.agent import run_agent
from src.client import ScriptedClient
from src.phase2 import CONDITIONS, HORIZONS, PROTOCOL, SIZES, PurchaseChain, trial_design


class Phase2AnalysisTests(unittest.TestCase):
    def test_known_paired_length_and_recovery_effect(self):
        series = {}
        with patch.dict(SIZES, main=3):
            for model in ("FP16", "Q8", "Q4"):
                series[model] = []
                for i, h, condition, rotation in trial_design("main"):
                    env = PurchaseChain("main", i, h, condition, rotation)
                    env.prepare_condition(env.task(), condition)
                    actions = []
                    for line in range(0 if condition == "error_restart" else 1, h + 1):
                        args = {"line": line, "offer": env.best(line)}
                        actions.append({"type": "tool_call", "name": "save_offer", "arguments": args})
                        env.execute("save_offer", args)
                    # Lose one Q4 h5 case, in both free and error but not normal.
                    if model == "Q4" and i == 0 and h == 5 and condition != "normal_restart":
                        actions.pop()
                    trace = run_agent(env.task(), ScriptedClient(actions), env, condition=condition, max_calls=h+5)
                    series[model].append({
                        "rotation": rotation, "base": env.base, "h": h, "split": "main", "condition": condition,
                        "precision": model, "model_sha256": model, "source_sha256":"FP16",
                        "settings_lock_sha256": "lock", "protocol_hash":"test", "protocol": PROTOCOL,
                        "trace": trace, **score(env, trace)})
            with patch("scripts.analyze_phase2.read_runs", side_effect=series.__getitem__):
                report = analyze(list(series))
                contrasts = {e["metric"]: e for e in report["paired_effects"]
                             if e.get("contrast") == "h5_minus_h1" and e["precision_vs_fp16"] == "Q4"}
                self.assertAlmostEqual(contrasts["free_success_difference_pp"]["mean_pp"], -100/3)
                self.assertAlmostEqual(contrasts["additional_error_loss_difference_pp"]["mean_pp"], 100/3)
                self.assertEqual(len(report["cells"]), 27)
                series["FP16"][0]["trace"]["initial_state"] = {"1": "unpaired"}
                with self.assertRaisesRegex(ValueError, "starting input or state"):
                    analyze(list(series))