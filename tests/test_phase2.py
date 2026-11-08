import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.phase2 import PROTOCOL_SOURCES, fingerprint, model_family, pilot_gate, score
from src.agent import run_agent
from src.client import ModelOutputError, ScriptedClient, parse_chat_response
from src.phase2 import CONDITIONS, HORIZONS, SIZES, PROTOCOL, PurchaseChain, trial_design


class Phase2Tests(unittest.TestCase):
    def test_fingerprint_ignores_checkout_newlines_but_detects_code_changes(self):
        scratch_root = Path(__file__).resolve().parents[1] / "results" / "phase2-tests"
        scratch_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=scratch_root) as directory:
            root = Path(directory)
            for name in PROTOCOL_SOURCES:
                path = root / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"setting = 1\n")
            original = fingerprint(root)
            for name in PROTOCOL_SOURCES:
                (root / name).write_bytes(b"setting = 1\r\n")
            self.assertEqual(fingerprint(root), original)
            (root / "src/agent.py").write_bytes(b"setting = 2\r\n")
            self.assertNotEqual(fingerprint(root), original)

    def test_every_fixture_replay_and_position(self):
        for split, size in SIZES.items():
            for h in HORIZONS:
                counts = Counter()
                for i in range(size):
                    for condition in CONDITIONS:
                        env = PurchaseChain(split, i, h, condition)
                        initial, _ = env.prepare_condition(env.task(), condition)
                        if condition == "normal_restart":
                            clean = initial
                        elif condition == "error_restart":
                            self.assertEqual(clean[:2], initial[:2])
                            self.assertEqual([m["role"] for m in clean],[m["role"] for m in initial])
                            self.assertNotEqual(clean[2:], initial[2:])
                        actions = []
                        for line in range(0 if condition=="error_restart" else 1,h+1):
                            menu=env.menu(line)
                            eligible=[o for o in menu["offers"] if o["price"] <= menu["budget"]]
                            self.assertTrue(1 <= len(eligible) <= 3)
                            target=max(eligible,key=lambda o:o["rating"])
                            if condition=="free":
                                counts[line,menu["offers"].index(target)] += 1
                            args={"line":line,"offer":target["offer"]}
                            actions.append({"type":"tool_call","name":"save_offer","arguments":args})
                            self.assertFalse(env.check_success())
                            self.assertTrue(env.execute("save_offer",args)["ok"])
                        trace=run_agent(env.task(),ScriptedClient(actions),env,condition=condition,max_calls=h+5,
                                        replay_rejected_output=False)
                        result=score(env,trace)
                        self.assertTrue(result["success"])
                        self.assertFalse(result["error_turns"])
                        self.assertEqual(result["executed_calls"],h+int(condition=="error_restart"))
                self.assertTrue(all(counts[line,slot]==size//3 for line in range(1,h+1) for slot in range(3)))

    def test_replacing_parent_invalidates_child_and_stale_offer(self):
        e=PurchaseChain("dev",0,3,"free")
        a,b=e.menu(1)["offers"][:2]
        e.execute("save_offer",{"line":1,"offer":a["offer"]})
        child=e.best(2)
        e.execute("save_offer",{"line":2,"offer":child})
        e.execute("save_offer",{"line":1,"offer":b["offer"]})
        self.assertNotIn(2,e.saved)
        self.assertFalse(e.execute("save_offer",{"line":2,"offer":child})["ok"])

    def test_progress_exposes_missing_lines_without_grading_selections(self):
        for condition in CONDITIONS:
            env = PurchaseChain("dev", 0, 3, condition)
            env.prepare_condition(env.task(), condition)
            view = env.view()
            self.assertEqual(view["required_lines"], [1, 2, 3] if condition == "free" else [0, 1, 2, 3])
            self.assertEqual(view["unsaved_lines"], [1, 2, 3])
            self.assertEqual([menu["line"] for menu in view["lines"]], [1] if condition == "free" else [0, 1])
            for menu in view["lines"]:
                self.assertEqual(set(menu), {"line", "budget", "offers", "saved_offer"})
                self.assertEqual(menu["offers"], env.menu(menu["line"])["offers"])
                expected = next((o for o in menu["offers"] if o["offer"] == env.saved.get(menu["line"])), None)
                self.assertEqual(menu["saved_offer"], expected)
            # Even wrong choices remove a line from unsaved_lines. Empty means
            # selected, not successful, and the action is still allowed.
            for line in range(1, 4):
                wrong = next(o["offer"] for o in env.menu(line)["offers"] if o["offer"] != env.best(line))
                self.assertTrue(env.execute("save_offer", {"line": line, "offer": wrong})["ok"])
            self.assertEqual(env.view()["unsaved_lines"], [])
            self.assertFalse(env.check_success())

    def test_progress_restores_unsaved_children_after_parent_replacement(self):
        env = PurchaseChain("dev", 0, 3, "free")
        for line in range(1, 4):
            env.execute("save_offer", {"line": line, "offer": env.best(line)})
        wrong = next(o["offer"] for o in env.menu(1)["offers"] if o["offer"] != env.saved[1])
        view = env.execute("save_offer", {"line": 1, "offer": wrong})
        self.assertEqual(view["unsaved_lines"], [2, 3])
        self.assertEqual([menu["line"] for menu in view["lines"]], [1, 2])
        self.assertIsNone(view["lines"][1]["saved_offer"])

    def test_correct_state_does_not_automatically_end_or_prevent_rewrites(self):
        env = PurchaseChain("dev", 0, 1, "free")
        gold = env.best(1)
        wrong = next(o["offer"] for o in env.menu(1)["offers"] if o["offer"] != gold)
        actions = [{"type": "tool_call", "name": "save_offer", "arguments": {"line": 1, "offer": offer}}
                   for offer in (gold, wrong)]
        trace = run_agent(env.task(), ScriptedClient(actions), env, condition="free", max_calls=6)
        result = score(env, trace)
        self.assertEqual(result["executed_calls"], 2)
        self.assertEqual(result["first_goal_turn"], 1)
        self.assertTrue(result["goal_lost_at_end"])
        self.assertFalse(result["success"])

    def test_plain_call_scope(self):
        def payload(text):
            return {"choices":[{"message":{"content":text}}]}
        data={"name":"inspect_state","arguments":{}}
        with self.assertRaises(ModelOutputError):
            parse_chat_response(payload(json.dumps(data)))
        self.assertEqual(parse_chat_response(payload(json.dumps(data)),accept_plain_call=True)["name"],"inspect_state")
        with self.assertRaises(ModelOutputError):
            parse_chat_response(payload(json.dumps({**data,"arguments":[]})),accept_plain_call=True)

    def test_rating_only_shortcut_cannot_clear_pilot(self):
        for split, size in SIZES.items():
            for h in HORIZONS:
                successes = Counter()
                for i in range(size):
                    for rank in range(3):
                        env = PurchaseChain(split, i, h, "free")
                        for line in range(1, h + 1):
                            offers = sorted(env.menu(line)["offers"], key=lambda o: o["rating"])
                            env.execute("save_offer", {"line": line, "offer": offers[rank]["offer"]})
                        successes[rank] += env.check_success()
                self.assertLess(max(successes.values()), size * 2 / 3)

    def test_accessory_rewrite_and_resolution_are_separate(self):
        for condition in ("normal_restart", "error_restart"):
            env = PurchaseChain("dev", 3, 1, condition)
            repair = {"type": "tool_call", "name": "save_offer", "arguments": {"line": 0, "offer": env.best(0)}}
            trace = run_agent(env.task(), ScriptedClient([repair]), env, condition=condition)
            result = score(env, trace)
            self.assertEqual(result["accessory_write_count"], 1)
            self.assertEqual(result["first_repair_turn"], 1 if condition == "error_restart" else None)
            self.assertFalse(result["success"])
            self.assertEqual(result["injected_error_resolved"], True if condition == "error_restart" else None)

    def test_manifest_rejects_mislabelled_precision(self):
        source = {"quantization": "F16", "sha256": "source"}
        q4 = {"quantization": "Q4_K_M", "sha256": "smaller"}
        manifest = {"source_model": source, "variants": [source, q4]}
        self.assertEqual(model_family(manifest, "Q4", "smaller"), "source")
        with self.assertRaises(ValueError):
            model_family(manifest, "FP16", "smaller")

    def test_gate_requires_all_cells_and_valid_baseline(self):
        rows=[{"base":f"dev-{i:03d}","h":h,"condition":c,"precision":"FP16","split":"dev","success":True,"rotation":k,"trace":{"events":[]},
               "source_sha256":"f","protocol":PROTOCOL,"model_sha256":"f","settings_lock_sha256":"s","protocol_hash":"p"}
              for i,h,c,k in trial_design("dev")]
        for row in rows:
            env = PurchaseChain("dev", int(row["base"].split("-")[1]), row["h"], row["condition"], row["rotation"])
            messages, _ = env.prepare_condition(env.task(), row["condition"])
            row["trace"].update(initial_messages=messages, initial_state=env.snapshot())
        pilot_gate(rows)
        no_recovery = [{**r, "success": r["condition"] != "error_restart"} for r in rows]
        with self.assertRaises(ValueError):
            pilot_gate(no_recovery)
        with self.assertRaises(ValueError):
            pilot_gate(rows[:-1])
        for r in rows:
            if r["condition"]=="free" and r["h"]==5 and r["base"] < "dev-005":
                r["success"]=False
        with self.assertRaises(ValueError):
            pilot_gate(rows)
