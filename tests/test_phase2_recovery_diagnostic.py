"""Check diagnostic separation, truthful validation and replay without running an LLM."""

import copy
import unittest

from scripts import diagnose_phase2_recovery as runner
from src.agent import run_agent
from src.client import ScriptedClient
from src.recovery_diagnostic import SingleLineProbe,ValidationProbe,input_plan,score_probe,trial_plan
from src.tasks import stable_hash


def oracle_trace(env):
    env.prepare_condition(env.task(),env.condition)
    actions=[]
    if env.mode=='judge_saved':
        actions=[{'type':'tool_call','name':'submit_assessment','arguments':{
            'is_valid':env.saved[0]==env.best(0),'best_offer':env.best(0)}}]
    elif env.mode=='apply_exact':
        if env.saved[0]!=env.target:
            actions=[{'type':'tool_call','name':'save_offer','arguments':{'line':0,'offer':env.target}}]
    else:
        if env.mode=='validation_optional':
            actions.append({'type':'tool_call','name':'validate_order','arguments':{}})
        for line in range(4):
            if env.saved.get(line)!=env.best(line):
                a={'type':'tool_call','name':'save_offer','arguments':{'line':line,'offer':env.best(line)}}
                actions.append(a);env.execute(a['name'],a['arguments'])
    trace=run_agent(env.task(),ScriptedClient(actions),env,condition=env.condition,
                    max_calls=3 if isinstance(env,SingleLineProbe) else 8,replay_rejected_output=False)
    return {'protocol':runner.PROTOCOL,'stage':'basic' if isinstance(env,SingleLineProbe) else 'validation',
            'base':env.base,'mode':env.mode,'condition':env.condition,**score_probe(env,trace),'trace':trace}


class RecoveryDiagnosticTests(unittest.TestCase):
    def test_plan_has_normal_controls_and_no_main_scenarios(self):
        for stage in ('basic','validation'):
            plan=input_plan(stage)
            self.assertEqual(len(plan),12)
            self.assertEqual({p['base'] for p in plan},{'dev-000','dev-001','dev-002'})
            self.assertEqual(len({(p['base'],p['mode'],p['condition']) for p in plan}),12)

    def test_validator_reports_rule_violation_without_replacement_answer(self):
        env=ValidationProbe(0,'error_restart','validation_optional')
        env.prepare_condition(env.task(),env.condition)
        before=env.state_hash();report=env.execute('validate_order',{})
        self.assertEqual(env.state_hash(),before)
        self.assertEqual(report['violations'],[{'line':0,'rule':'budget_exceeded','selected_price':100,'budget':88}])
        self.assertNotIn(env.best(0),str(report))
        self.assertEqual(report['unselected_lines'],[1,2,3])
        normal=ValidationProbe(0,'normal_restart','validation_optional')
        normal.prepare_condition(normal.task(),normal.condition)
        self.assertEqual(normal.validation_report()['violations'],[])
        self.assertFalse(normal.check_success())

    def test_supplied_report_preserves_query_tools_and_state(self):
        optional=ValidationProbe(1,'error_restart','validation_optional')
        supplied=ValidationProbe(1,'error_restart','validation_supplied')
        om,_=optional.prepare_condition(optional.task(),optional.condition)
        sm,_=supplied.prepare_condition(supplied.task(),supplied.condition)
        self.assertEqual(optional.query,supplied.query)
        self.assertEqual(optional.tool_definitions(),supplied.tool_definitions())
        self.assertEqual(optional.state_hash(),supplied.state_hash())
        self.assertEqual(sm[:len(om)],om)
        self.assertEqual(sm[-1]['name'],'validate_order')

    def test_assessment_is_scored_without_changing_or_correcting_order(self):
        env=SingleLineProbe(0,'error_restart','judge_saved')
        env.prepare_condition(env.task(),env.condition);saved=copy.deepcopy(env.saved)
        bad=env.execute('submit_assessment',{'is_valid':True,'best_offer':env.best(0)})
        self.assertTrue(bad['ok'])
        self.assertNotIn('correct',bad)
        self.assertFalse(env.check_success())
        env.execute('submit_assessment',{'is_valid':False,'best_offer':env.best(0)})
        self.assertTrue(env.check_success())
        self.assertEqual(saved,env.saved)
        self.assertFalse(env.execute('submit_assessment',{'is_valid':'false','best_offer':env.best(0)})['ok'])

    def test_exact_edit_is_an_answer_given_control(self):
        env=SingleLineProbe(2,'error_restart','apply_exact')
        env.prepare_condition(env.task(),env.condition)
        self.assertIn(env.target,env.query)
        self.assertFalse(env.check_success())
        self.assertFalse(env.execute('save_offer',{'line':1,'offer':env.target})['ok'])
        env.execute('save_offer',{'line':0,'offer':env.target})
        self.assertTrue(env.check_success())

    def test_all_twenty_four_oracle_trials_fit_budgets_and_replay(self):
        for stage in ('basic','validation'):
            rows=[oracle_trace(e) for e in trial_plan(stage)]
            runner.validate_rows(rows,stage)
            self.assertTrue(all(r['success'] for r in rows))
            self.assertTrue(all(r['finished'] for r in rows))
            self.assertTrue(all(r['scope']!='full_order_h3' for r in rows) if stage=='basic'
                            else all(r['scope']=='full_order_h3' for r in rows))

    def test_harness_report_does_not_count_as_model_validation(self):
        env=ValidationProbe(0,'error_restart','validation_supplied')
        trace=run_agent(env.task(),ScriptedClient([]),env,condition=env.condition,max_calls=8,
                        replay_rejected_output=False)
        scores=score_probe(env,trace)
        self.assertEqual(scores['preloaded_validation_calls'],1)
        self.assertEqual(scores['model_validation_calls'],0)
        self.assertFalse(scores['success'])
        self.assertTrue(scores['premature_finish'])

    def test_dev12_replays_all_four_cells_and_matches_gains_by_scenario(self):
        stage = 'validation_dev12'
        rows = []
        for env in trial_plan(stage):
            # A scripted contrast: supplied completes all lines; optional stops immediately.
            if env.mode == 'validation_supplied':
                trace = oracle_trace(env)['trace']
            else:
                trace = run_agent(env.task(), ScriptedClient([]), env, condition=env.condition,
                                  max_calls=8, replay_rejected_output=False)
            rows.append(dict(protocol=runner.PROTOCOL, stage=stage, base=env.base,
                             mode=env.mode, condition=env.condition, trace=trace,
                             **runner.trial_scores(env, trace, stage)))
        runner.validate_rows(rows, stage)
        summary = runner.summarize(rows[::-1], stage)
        self.assertEqual([cell['n_valid'] for cell in summary['cells']], [12]*4)
        self.assertEqual([cell['bundle_correct_lines'] for cell in summary['cells']], [0,0,36,36])
        expected = [f'dev-{i:03d}' for i in range(12)]
        for item in summary['paired']:
            preserved = item['condition']=='normal_restart' and item['metric']=='accessory_correct_at_end'
            self.assertEqual(item['pairs'], 12)
            self.assertEqual(item['gained_with_report'], [] if preserved else expected)
            self.assertEqual(item['lost_with_report'], [])
        with self.assertRaises(ValueError):
            runner.validate_rows(rows[:-1] + [rows[0]], stage)

    def test_summary_excludes_infrastructure_and_replay_rejects_tampering(self):
        rows=[oracle_trace(e) for e in trial_plan('validation')]
        bad=copy.deepcopy(rows);bad[0]['success']=None
        summary=runner.summarize(bad,'validation')
        self.assertEqual(sum(c['n_valid'] for c in summary['cells']),11)
        with self.assertRaises(ValueError):runner.validate_rows(bad,'validation')
        with self.assertRaises(ValueError):runner.validate_rows(rows+[rows[0]],'validation')
        bad=copy.deepcopy(rows);bad[0]['trace']['events'][0]['tool_result']['ok']=False
        with self.assertRaises(ValueError):runner.validate_rows(bad,'validation')
        self.assertEqual(stable_hash(rows),stable_hash(copy.deepcopy(rows)))


if __name__=='__main__':unittest.main()
