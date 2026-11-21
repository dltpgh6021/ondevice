"""Regression checks for paired first saves and isolated validation."""
import unittest

from scripts.compare_phase2_granite_choices import IsolatedMenu, evaluate, outcome, summary
from src.agent import run_agent
from src.client import ScriptedClient
from src.recovery_diagnostic import ValidationProbe


class MatchedChoiceTests(unittest.TestCase):
    def setUp(self):
        self.menu = {'line': 2, 'budget': 50, 'offers': [
            {'offer': 'too-much', 'price': 51, 'rating': 5},
            {'offer': 'best', 'price': 50, 'rating': 4},
            {'offer': 'lower', 'price': 40, 'rating': 3}]}
        self.env = IsolatedMenu(0, self.menu)
        self.pair = {'base': self.env.base, 'menu': self.menu, 'h3_turn': 2, 'h3_offer': 'best'}

    def trace(self, offers):
        calls = [{'type': 'tool_call', 'name': 'save_offer', 'arguments': {'line': 2, 'offer': offer}}
                 for offer in offers]
        return run_agent(self.env.task(), ScriptedClient(calls), self.env, condition='free',
                         max_calls=8, replay_rejected_output=False)

    def test_first_and_final_are_distinct_and_a_missing_save_stays_in_denominator(self):
        row = {'base': self.env.base, 'line': 2, 'trace': self.trace(['too-much', 'best'])}
        result = summary([self.pair], [row])
        self.assertEqual(result['outcomes']['isolated_first'], {'over_budget': 1})
        self.assertEqual(result['outcomes']['isolated_final'], {'correct': 1})
        self.assertEqual(result['changes']['lost'], ['dev-000/line2'])
        self.assertEqual(evaluate(self.pair, self.trace([]))['isolated_first'], 'no_save')
        self.assertEqual(outcome(self.menu, 'lower'), 'lower_rating')

    def test_input_order_and_validator_follow_the_frozen_menu(self):
        self.env.prepare_condition(self.env.task(), 'free')
        self.assertEqual(self.env.view()['lines'][0]['offers'], self.menu['offers'])
        self.assertEqual(self.env.tool_definitions(),
                         ValidationProbe(0, 'normal_restart', 'validation_supplied').tool_definitions())
        self.assertEqual(self.env.execute('validate_order', {})['unselected_lines'], [2])
        self.env.execute('save_offer', {'line': 2, 'offer': 'lower'})
        self.assertEqual(self.env.execute('validate_order', {})['violations'],
                         [{'line': 2, 'rule': 'not_highest_rating_within_budget'}])
