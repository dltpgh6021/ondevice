import json
import unittest

from scripts.check_phase2_current_menu import CONDITIONS, MODE, CurrentMenu, compare, score_rows
from src.agent import run_agent
from src.client import ScriptedClient
from src.recovery_diagnostic import ValidationProbe


class CurrentMenuTests(unittest.TestCase):
    def test_focus_contains_only_visible_facts_and_preserves_validator(self):
        for condition in CONDITIONS:
            old = ValidationProbe(4, condition, MODE)
            new = CurrentMenu(4, condition, MODE)
            om, _ = old.prepare_condition(old.task(), condition)
            nm, _ = new.prepare_condition(new.task(), condition)
            self.assertEqual(old.query, new.query)
            self.assertEqual(om[-2:], nm[-2:])
            self.assertEqual(old.tool_definitions(), new.tool_definitions())
            self.assertEqual(new.view()['current_bundle_menu'], old.menu(1) | {'saved_offer': None})
            for line in [1, 2, 1, 2, 3]:
                available = old.menu(line)['offers']
                offer = next(o['offer'] for o in available if o['offer'] != old.saved.get(line))
                for env in (old, new):
                    env.execute('save_offer', {'line': line, 'offer': offer})
                a, b = old.view(), new.view()
                focus = b.pop('current_bundle_menu')
                visible = b.pop('other_visible_lines')
                if focus is not None:
                    self.assertEqual(focus['line'], min(i for i in a['unsaved_lines'] if i > 0))
                    visible.append(focus)
                self.assertEqual(sorted(visible, key=lambda m: m['line']), a.pop('lines'))
                self.assertEqual(a, b)
                self.assertEqual(old.validation_report(), new.validation_report())
            self.assertIsNone(new.view()['current_bundle_menu'])

    def test_all_pairs_and_replay(self):
        rows = []
        for i in range(12):
            for condition in CONDITIONS:
                env = CurrentMenu(i, condition, MODE)
                env.prepare_condition(env.task(), condition)
                calls = []
                for line in range(4):
                    if env.saved.get(line) != env.best(line):
                        args = {'line': line, 'offer': env.best(line)}
                        calls.append({'type': 'tool_call', 'name': 'save_offer', 'arguments': args})
                        env.execute('save_offer', args)
                trace = run_agent(env.task(), ScriptedClient(calls), env, condition=condition,
                                  max_calls=8, replay_rejected_output=False)
                rows.append({'base': env.base, 'condition': condition, 'trace': trace})
        rows = json.loads(json.dumps(rows))
        scores = score_rows(rows, CurrentMenu)
        result = compare(scores, scores)
        self.assertTrue(result['h3_order_thresholds_met'])
        for c in CONDITIONS:
            self.assertEqual(result['conditions'][c]['after']['first_save_correct_lines'], 36)
            self.assertEqual(result['conditions'][c]['after']['line0_correct'], 12)
            self.assertEqual(set(result['conditions'][c]['delta'].values()), {0})
        with self.assertRaises(ValueError):
            score_rows(rows[:-1] + [rows[0]], CurrentMenu)
