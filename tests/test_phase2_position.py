"""Position changes must preserve the problem, including the injected mistake."""
import unittest
from collections import Counter

from scripts.phase2 import position_summary
from src.agent import run_agent
from src.client import ScriptedClient
from src.phase2 import CONDITIONS, PurchaseChain, trial_design


class PositionAuditTests(unittest.TestCase):
    def test_rotation_preserves_goal_saved_error_ids_and_other_menus(self):
        for i in range(12):
            for c in CONDITIONS:
                snapshots, winners, slots, menus, histories = [], [], [], [], []
                for k in range(3):
                    env = PurchaseChain("dev", i, 1, c, k)
                    messages, _ = env.prepare_condition(env.task(), c)
                    snapshots.append(env.snapshot())
                    winners.append(env.best(env.audit_line))
                    canonical = env.view()
                    audit_menu = next(m for m in canonical["lines"] if m["line"] == env.audit_line)
                    slots.append([o["offer"] for o in audit_menu["offers"]].index(winners[-1]))
                    audit_menu["offers"].sort(key=lambda o:o["offer"])
                    menus.append(canonical)
                    histories.append([m for m in messages if m["role"] == "assistant"])
                self.assertEqual(len(set(winners)), 1)
                self.assertEqual(set(slots), {0,1,2})
                self.assertEqual(menus, [menus[0]]*3)
                self.assertEqual(histories, [histories[0]]*3)
                self.assertEqual(snapshots, [snapshots[0]]*3)

    def test_first_slot_policy_flips_but_content_policy_does_not(self):
        for policy in ("first", "oracle", "finish"):
            rows = []
            for i in range(12):
                for k in range(3):
                    env = PurchaseChain("dev",i,1,"free",k)
                    offer = env.menu(1)["offers"][0]["offer"] if policy=="first" else env.best(1)
                    actions = [] if policy=="finish" else [{"type":"tool_call","name":"save_offer",
                                                            "arguments":{"line":1,"offer":offer}}]
                    trace = run_agent(env.task(), ScriptedClient(actions), env, condition="free")
                    rows.append({"split":"dev","base":env.base,"h":1,"condition":"free","rotation":k,
                                 "success":env.check_success(),"trace":trace})
            result = position_summary(rows)["free"]
            self.assertEqual(result["scenario_triples"], 12)
            self.assertEqual(result["final_success_flips_across_orders"], 12 if policy=="first" else 0)
            self.assertEqual(result["first_slot_saved_all_orders"], 12 if policy=="first" else 0)
            self.assertEqual(result["same_first_saved_offer_all_orders"], 12 if policy=="oracle" else 0)
            self.assertEqual(result["by_gold_slot"][2]["no_save"], 12 if policy=="finish" else 0)

    def test_following_slot_counter_excludes_last_slot_and_missing_saves(self):
        rows = []
        for k in range(3):
            env = PurchaseChain("dev", 0, 1, "error_restart", k)
            env.prepare_condition(env.task(), "error_restart")
            ids = [o["offer"] for o in env.menu(0)["offers"]]
            selected = ids[(ids.index(env.saved[0]) + 1) % 3]
            actions = [{"type": "tool_call", "name": "save_offer", "arguments": {"line": 0, "offer": selected}}]
            trace = run_agent(env.task(), ScriptedClient(actions), env, condition="error_restart")
            rows.append({"split":"dev","base":env.base,"h":1,"condition":"error_restart",
                         "rotation":k,"success":env.check_success(),"trace":trace})
        results = position_summary(rows)["error_restart"]
        self.assertEqual(results["first_accessory_save_in_next_slot"], 2)
        self.assertEqual(results["accessory_saves_with_a_following_slot"], 2)

    def test_workload_and_audit_subset_fixed_in_advance(self):
        for split,base_n,total in (("dev",12,180), ("main",60,612)):
            schedule = list(trial_design(split))
            self.assertEqual(len(schedule), total)
            self.assertEqual(len(set(schedule)), total)
            self.assertEqual(sum(k==0 for _,_,_,k in schedule),base_n*9)
            self.assertTrue(all(h==1 for _,h,_,k in schedule if k))
            first_orders = {}
            for i,h,c,k in schedule:
                if h==1 and c=="free" and (split=="dev" or i%5==0):
                    first_orders.setdefault(i, k)
            self.assertEqual(Counter(first_orders.values()), {0:4,1:4,2:4})