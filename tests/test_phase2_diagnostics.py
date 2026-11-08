"""Keep diagnostic interventions paired and separate from the primary experiment."""
import copy
import json
import unittest
from collections import Counter

from scripts.diagnose_phase2 import (
    LAUNCH_FLAGS, REVIEW, IsolatedChoice, RestartProbe, build_probes,
    launch_settings, probe_score, summarize,
)
from scripts.phase2 import fingerprint, score
from src.agent import run_agent
from src.client import ScriptedClient
from src.phase2 import PROTOCOL, PurchaseChain, trial_design
from src.tasks import stable_hash


def fake_baseline():
    rows = []
    code_hash = fingerprint()
    for i, h, condition, rotation in trial_design("dev"):
        env = PurchaseChain("dev", i, h, condition, rotation)
        env.prepare_condition(env.task(), condition)
        actions = []
        for line in range(0 if condition == "error_restart" else 1, h + 1):
            offer = env.best(line)
            if i == 0 and h == 5 and condition == "free" and line == 1:
                offer = next(o["offer"] for o in env.menu(1)["offers"] if o["offer"] != offer)
            args = {"line": line, "offer": offer}
            actions.append({"type": "tool_call", "name": "save_offer", "arguments": args})
            env.execute("save_offer", args)
        trace = run_agent(env.task(), ScriptedClient(actions), env, condition=condition, max_calls=h+5)
        rows.append({"protocol": PROTOCOL, "protocol_hash": code_hash, "precision": "FP16", "split": "dev",
                     "source_sha256": "fp16", "model_sha256": "fp16", "settings_lock_sha256": "lock",
                     "base": env.base, "h": h, "condition": condition, "rotation": rotation,
                     "trace": trace, **score(env, trace)})
    return rows


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = fake_baseline()
        cls.probes = build_probes(cls.baseline)

    def test_fixed_design_includes_wrong_choices_and_every_scenario(self):
        self.assertEqual(len(self.probes), 132)
        self.assertEqual(Counter(m["mode"] for _, m in self.probes),
                         {"isolated_choice": 60, "history": 24, "state_only": 24, "history_review": 24})
        self.assertEqual(len({m["probe_id"] for _, m in self.probes}), 132)
        self.assertTrue(all(n == 11 for n in Counter(e.base for e, _ in self.probes).values()))
        isolated = [(e, m) for e, m in self.probes if m["mode"] == "isolated_choice"]
        self.assertEqual(sum(m["baseline_first_save_correct"] for _, m in isolated), 59)

    def test_isolated_menu_is_from_actual_wrong_parent_path(self):
        env, plan = next((e, m) for e, m in self.probes if m["probe_id"] == "dev-000/isolated/line2")
        original = next(r for r in self.baseline if r["base"] == "dev-000" and r["h"] == 5 and r["condition"] == "free")
        after_first = next(e["tool_result"] for e in original["trace"]["events"] if e.get("tool_result", {}).get("updated_line") == 1)
        actual = copy.deepcopy(next(m for m in after_first["lines"] if m["line"] == 2))
        actual.pop("saved_offer")
        self.assertEqual(env.menu(2), actual)
        self.assertEqual(stable_hash(actual), plan["menu_sha256"])
        oracle = PurchaseChain("dev", 0, 5, "free")
        oracle.execute("save_offer", {"line": 1, "offer": oracle.best(1)})
        self.assertNotEqual(env.menu(2), oracle.menu(2))
        messages, _ = env.prepare_condition(env.task(), "free")
        self.assertEqual(env.snapshot(), {})
        self.assertEqual(env.view()["required_lines"], [2])
        self.assertEqual(len(env.view()["lines"]), 1)
        self.assertNotIn("baseline_first_save_correct", json.dumps(messages))

    def test_restart_interventions_preserve_state_and_blind_pairing(self):
        for i in range(12):
            for condition in ("normal_restart", "error_restart"):
                original = PurchaseChain("dev", i, 1, condition)
                original_messages, _ = original.prepare_condition(original.task(), condition)
                for mode in ("history", "state_only", "history_review"):
                    env = RestartProbe(i, condition, mode)
                    messages, _ = env.prepare_condition(env.task(), condition)
                    self.assertEqual(env.snapshot(), original.snapshot())
                    self.assertEqual(env.view(), original.view())
                    self.assertEqual(env.tool_definitions(), original.tool_definitions())
                    if mode == "history":
                        self.assertEqual(messages, original_messages)
                    elif mode == "history_review":
                        self.assertEqual(messages[:-1], original_messages)
                        self.assertEqual(messages[-1], {"role": "user", "content": REVIEW})
                    else:
                        self.assertEqual([m["role"] for m in messages], ["system", "user"])
                        self.assertEqual(messages[-1]["content"], env.query + "\nCurrent order: " + json.dumps(original.view()))

    def test_isolated_repair_and_verbal_only_finish_are_distinct(self):
        env = IsolatedChoice(0, PurchaseChain("dev", 0, 1, "free").menu(1))
        gold = env.best(1)
        wrong = next(o["offer"] for o in env.menu(1)["offers"] if o["offer"] != gold)
        actions = [{"type": "tool_call", "name": "save_offer", "arguments": {"line": 1, "offer": offer}}
                   for offer in (wrong, gold)]
        trace = run_agent(env.task(), ScriptedClient(actions), env, condition="free", max_calls=6)
        result = probe_score(env, trace)
        self.assertTrue(result["success"])
        self.assertFalse(result["first_save_correct"])
        verbal = run_agent(env.task(), ScriptedClient([{"type": "finish", "answer": gold}]), env, condition="free")
        result = probe_score(env, verbal)
        self.assertFalse(result["success"])
        self.assertFalse(result["first_save_present"])

    def test_repair_without_bundle_completion_and_normal_damage_are_measured(self):
        env = RestartProbe(0, "error_restart", "state_only")
        action = {"type": "tool_call", "name": "save_offer", "arguments": {"line": 0, "offer": env.best(0)}}
        trace = run_agent(env.task(), ScriptedClient([action]), env, condition="error_restart")
        result = probe_score(env, trace)
        self.assertTrue(result["injected_error_resolved"])
        self.assertFalse(result["success"])
        normal = RestartProbe(0, "normal_restart", "history_review")
        wrong = next(o["offer"] for o in normal.menu(0)["offers"] if o["offer"] != normal.best(0))
        action["arguments"]["offer"] = wrong
        trace = run_agent(normal.task(), ScriptedClient([action]), normal, condition="normal_restart")
        result = probe_score(normal, trace)
        self.assertTrue(result["accessory_ever_damaged"])
        self.assertFalse(result["accessory_correct_at_end"])

    def test_baseline_must_be_complete_and_untampered(self):
        with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
            build_probes(self.baseline[:-1])
        changed = copy.deepcopy(self.baseline)
        row = next(r for r in changed if r["h"] == 5 and r["condition"] == "free")
        event = next(e for e in row["trace"]["events"] if e.get("tool_executed"))
        event["tool_result"]["ok"] = False
        with self.assertRaisesRegex(ValueError, "replay mismatch"):
            build_probes(changed)

    def test_launch_settings_include_physical_batch_and_cache_precision(self):
        argv = ["llama-server"]
        for flag in LAUNCH_FLAGS:
            argv += [flag, "128" if flag == "--ubatch-size" else "value"]
        changed = list(argv)
        changed[changed.index("--ubatch-size") + 1] = "512"
        self.assertNotEqual(launch_settings(argv), launch_settings(changed))
        with self.assertRaises(ValueError):
            launch_settings(argv + ["--ubatch-size", "128"])

    def test_summary_does_not_count_infrastructure_as_model_failure(self):
        common = {"mode": "isolated_choice", "condition": "free", "format_rejections": 0,
                  "trace": {"termination_reason": "finished"}, "first_save_present": True,
                  "baseline_first_save_correct": False}
        rows = [{**common, "success": True, "first_save_correct": True},
                {**common, "success": None, "first_save_correct": None}]
        cell = summarize(rows)["cells"][0]
        self.assertEqual((cell["n"], cell["n_valid"], cell["successes"]), (2, 1, 1))
        self.assertEqual(cell["changed_wrong_to_right"], 1)
