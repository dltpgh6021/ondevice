"""Development-only probes separating judgement, exact editing and validation exposure."""

import copy
import json
from dataclasses import replace

from .phase2 import PurchaseChain, TOOLS
from .tasks import stable_hash

PROTOCOL = 'recovery-components-v1'
STAGES = {'basic': ('judge_saved', 'apply_exact'),
          'validation': ('validation_optional', 'validation_supplied')}
ASSESS_TOOL = {'type':'function','function':{
    'name':'submit_assessment',
    'description':'Record your assessment of line 0. This does not change the stored order.',
    'parameters':{'type':'object','properties':{
        'is_valid':{'type':'boolean','description':'Whether the saved offer satisfies the full selection rule.'},
        'best_offer':{'type':'string','description':'The ID of the highest-rated offer within budget.'}},
        'required':['is_valid','best_offer'],'additionalProperties':False}}}
VALIDATE_TOOL = {'type':'function','function':{
    'name':'validate_order',
    'description':('Check saved selections against their budgets and the highest-rating rule. '
                   'Returns violations and line numbers, without choosing replacement offers. '
                   'Unselected lines are listed separately; no violations does not mean the order is complete.'),
    'parameters':{'type':'object','properties':{},'additionalProperties':False}}}


class SingleLineProbe(PurchaseChain):
    """Isolate the exact line-0 facts; this is not a horizon-effect experiment."""

    def __init__(self, scenario, condition, mode):
        if mode not in STAGES['basic']:
            raise ValueError('Unknown basic probe')
        super().__init__('dev', scenario, 1, condition)
        self.mode = mode
        self.assessment = None
        self.target = self.best(0)
        self.query = (
            'Evaluate the saved selection for line 0 only. The selection rule is: choose the highest-rated '
            'offer whose price is at or below the budget. Use submit_assessment to state whether the saved '
            'offer satisfies this rule, and give the best offer ID even if the saved offer is already valid. '
            'Do not edit the order. Then give a short final answer.'
            if mode == 'judge_saved' else
            f'For this editing check only, ensure line 0 stores offer {self.target}. '
            'If that exact offer is already saved, keep it. Otherwise call save_offer to replace the saved '
            'offer. Do not work on any other line. Then give a short final answer.'
        )

    def view(self):
        menu = self.menu(0)
        menu['saved_offer'] = next((dict(o) for o in menu['offers'] if o['offer']==self.saved.get(0)), None)
        return {'required_lines':[0], 'saved':dict(self.saved), 'lines':[menu]}

    def snapshot(self):
        return {'saved':copy.deepcopy(self.saved), 'assessment':copy.deepcopy(self.assessment)}

    def state_hash(self):
        return stable_hash(self.snapshot())

    def prepare_condition(self, task, condition):
        self.assessment = None
        menu = self.menu(0)
        selected = (self.best(0) if condition=='normal_restart' else
                    next(o['offer'] for o in menu['offers'] if o['price']>menu['budget']))
        self.saved = {0:selected}
        return [{'role':'system','content':'Use one tool call per response or a final answer. Use English.'},
                {'role':'user','content':self.query+'\nCurrent order: '+json.dumps(self.view())}], None

    def tool_definitions(self):
        if self.mode=='judge_saved':
            return [copy.deepcopy(ASSESS_TOOL)]
        tools = copy.deepcopy(TOOLS)
        tools[0]['function']['description'] = (
            'Store one available offer on line 0. Saving again replaces the previous selection. '
            'An ok result means the update was applied.')
        tools[1]['function']['description'] = 'Read the current line-0 selection and available offers.'
        return tools

    def execute(self, name, args):
        if self.mode=='judge_saved':
            if (name!='submit_assessment' or set(args)!={'is_valid','best_offer'}
                    or type(args['is_valid']) is not bool
                    or args['best_offer'] not in [o['offer'] for o in self.menu(0)['offers']]):
                return {'ok':False,'error':'Use submit_assessment with a boolean is_valid and an available best_offer ID.'}
            self.assessment = copy.deepcopy(args)
            return {'ok':True,'assessment_recorded':True}
        if name=='inspect_order' and not args:
            return {'ok':True, **self.view()}
        if (name!='save_offer' or set(args)!={'line','offer'} or type(args['line']) is not int
                or args['line']!=0 or args['offer'] not in [o['offer'] for o in self.menu(0)['offers']]):
            return {'ok':False,'error':'Only line 0 and its available offer IDs are supported.'}
        self.saved[0] = args['offer']
        return {'ok':True,'updated_line':0,**self.view()}

    def check_success(self, ignored=None):
        if self.mode=='apply_exact':
            return self.saved.get(0)==self.target
        return bool(self.assessment is not None
                    and self.assessment['is_valid']==(self.saved.get(0)==self.best(0))
                    and self.assessment['best_offer']==self.best(0))

    def assess_action_result(self, result):
        if result.get('assessment_recorded'):
            return None if self.check_success() else 'wrong_assessment'
        return super().assess_action_result(result)

    def task(self):
        return replace(super().task(), task_id=f'{self.base}/{self.mode}/{self.condition}',
                       dataset_version=PROTOCOL, normal_remaining_calls=None, recovery_min_calls=None)


class ValidationProbe(PurchaseChain):
    def __init__(self, scenario, condition, mode):
        if mode not in STAGES['validation']:
            raise ValueError('Unknown validation probe')
        super().__init__('dev', scenario, 3, condition)
        self.mode = mode

    def tool_definitions(self):
        return copy.deepcopy(TOOLS)+[copy.deepcopy(VALIDATE_TOOL)]

    def validation_report(self):
        violations = []
        for line, saved in sorted(self.saved.items()):
            menu = self.menu(line)
            offer = next((o for o in menu['offers'] if o['offer']==saved), None)
            if offer is None:
                violations.append({'line':line,'rule':'offer_unavailable'})
            elif offer['price']>menu['budget']:
                violations.append({'line':line,'rule':'budget_exceeded',
                                   'selected_price':offer['price'],'budget':menu['budget']})
            elif saved!=self.best(line):
                violations.append({'line':line,'rule':'not_highest_rating_within_budget'})
        return {'ok':True,'check_scope':'saved_selections_only', 'checked_lines':sorted(self.saved),
                'unselected_lines':self.view()['unsaved_lines'],'violations':violations}

    def execute(self, name, args):
        if name=='validate_order':
            return self.validation_report() if not args else {'ok':False,'error':'validate_order takes no arguments.'}
        return super().execute(name,args)

    def prepare_condition(self, task, condition):
        messages,_ = super().prepare_condition(task,condition)
        if self.mode=='validation_supplied':
            # A diagnostic intervention: the harness supplies one validation result.
            # It is not a model-selected call and is not charged to generated turns.
            messages += [{'role':'assistant','tool_calls':[{'id':'validation_checkpoint','type':'function',
                          'function':{'name':'validate_order','arguments':'{}'}}]},
                         {'role':'tool','tool_call_id':'validation_checkpoint','name':'validate_order',
                          'content':json.dumps(self.validation_report())}]
        return messages,None

    def task(self):
        return replace(super().task(), task_id=f'{self.base}/{self.mode}/{self.condition}',dataset_version=PROTOCOL)


def trial_plan(stage):
    if stage not in STAGES:
        raise ValueError('Unknown stage')
    kind = SingleLineProbe if stage=='basic' else ValidationProbe
    for scenario in (0,1,2):
        # Rotate mode/condition order; every scenario still gets all four cells.
        modes = STAGES[stage][scenario%2:]+STAGES[stage][:scenario%2]
        for j,mode in enumerate(modes):
            conditions = ('normal_restart','error_restart') if (scenario+j)%2==0 else ('error_restart','normal_restart')
            for condition in conditions:
                yield kind(scenario,condition,mode)


def input_plan(stage):
    result = []
    for env in trial_plan(stage):
        messages,_ = env.prepare_condition(env.task(),env.condition)
        result.append({'base':env.base,'mode':env.mode,'condition':env.condition,
                       'messages_sha256':stable_hash(messages),'state_sha256':env.state_hash(),
                       'tools_sha256':stable_hash(env.tool_definitions())})
    return result


def score_probe(env, trace):
    valid = not (trace['infrastructure_error'] or trace['internal_error'] or trace['runtime_error'])
    writes = [e for e in trace['events'] if e.get('tool_result',{}).get('updated_line')==0]
    checks = [e for e in trace['events'] if e.get('tool_executed') and e['action']['name']=='validate_order']
    result = {'success':env.check_success() if valid else None,
              'scope':'line0_assessment' if env.mode=='judge_saved' else 'line0_edit' if env.mode=='apply_exact' else 'full_order_h3',
              'accessory_write_count':len(writes),
              'accessory_correct_at_end':env.saved.get(0)==env.best(0) if valid else None,
              'model_validation_calls':len(checks),
              'first_model_validation_turn':checks[0]['step'] if checks else None,
              'preloaded_validation_calls':int(env.mode=='validation_supplied'),
              'first_correct_accessory_write_turn':next((e['step'] for e in writes if not e.get('semantic_error')),None),
              'new_action_error_count':sum(bool(e.get('semantic_error') or e.get('tool_error')) for e in trace['events']),
              'format_rejections':sum(e.get('error')=='model_output_error' for e in trace['events']),
              'finished':trace['finished'],'termination_reason':trace['termination_reason'],
              'premature_finish':bool(trace['finished'] and not env.check_success()) if valid else None}
    if env.mode=='judge_saved':
        assessment = env.assessment
        result.update(assessment=copy.deepcopy(assessment),
                      validity_correct=(bool(assessment and assessment['is_valid']==(env.saved[0]==env.best(0))) if valid else None),
                      best_offer_correct=(bool(assessment and assessment['best_offer']==env.best(0)) if valid else None))
    return result
