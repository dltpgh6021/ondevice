"""Granite-only development diagnostics: judgement/editing first, validation exposure second."""

import argparse
import copy
import json
import sys
from importlib.metadata import version
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts import check_phase2_granite as granite
from src.agent import run_agent
from src.client import ModelOutputError
from src.experiment import sha256_file
from src.granite_client import GraniteClient
from src.recovery_diagnostic import PROTOCOL,STAGES,input_plan,score_probe,trial_plan
from src.run_preparation import _WARMUP_MESSAGES,_WARMUP_TOOLS
from src.tasks import stable_hash


def source_hash():
    names=('scripts/diagnose_phase2_recovery.py','src/recovery_diagnostic.py')
    return stable_hash({n:(ROOT/n).read_text(encoding='utf-8') for n in names})


def load_reference(directory):
    plan,_=granite.load_plan(directory)
    metadata=json.loads((directory/'granite/metadata.json').read_text(encoding='utf-8'))
    rows=granite.read_runs(directory/'granite/trials.jsonl')
    granite.validate_rows(rows)
    for key in ('model_id','revision','hardware','input_plan','runner_hash','primary_protocol_hash','manifest_hash'):
        if metadata[key]!=plan[key]:
            raise ValueError(f'Granite reference differs: {key}')
    if metadata['loaded_dtype']!='float16' or metadata['generation_config']['do_sample'] is not False:
        raise ValueError('Use the completed Granite FP16 greedy screening run')
    return plan,metadata


def validate_rows(rows,stage):
    expected={(p['base'],p['mode'],p['condition']):p for p in input_plan(stage)}
    if len(rows)!=12 or {(r['base'],r['mode'],r['condition']) for r in rows}!=set(expected):
        raise ValueError('Need twelve unique trials in this stage')
    environments={(e.base,e.mode,e.condition):e for e in trial_plan(stage)}
    for row in rows:
        if row.get('protocol')!=PROTOCOL or row.get('stage')!=stage or type(row.get('success')) is not bool:
            raise ValueError('Wrong protocol or incomplete/infrastructure trial')
        key=row['base'],row['mode'],row['condition'];p=expected[key];trace=row['trace'];env=environments[key]
        env.prepare_condition(env.task(),env.condition)
        if (stable_hash(trace['initial_messages'])!=p['messages_sha256']
                or trace['initial_state_hash']!=p['state_sha256']
                or stable_hash(trace['initial_state'])!=p['state_sha256']):
            raise ValueError('Diagnostic starting input differs')
        for event in trace['events']:
            if event.get('tool_executed'):
                a=event['action'];result=env.execute(a['name'],a['arguments'])
                if (stable_hash(result)!=stable_hash(event['tool_result'])
                        or env.state_hash()!=event['state_hash_after']
                        or env.check_success()!=event['success_after_action']):
                    raise ValueError('Diagnostic tool/state replay differs')
        if env.state_hash()!=trace['final_state_hash'] or env.state_hash()!=stable_hash(trace['final_state']):
            raise ValueError('Diagnostic final state differs')
        if any(stable_hash(row[k])!=stable_hash(v) for k,v in score_probe(env,trace).items()):
            raise ValueError('Diagnostic metrics differ')


def summarize(rows,stage):
    cells=[]
    for mode in STAGES[stage]:
        for condition in ('normal_restart','error_restart'):
            group=[r for r in rows if r['mode']==mode and r['condition']==condition]
            valid=[r for r in group if r['success'] is not None]
            cell={'mode':mode,'condition':condition,'n':len(group),'n_valid':len(valid),
                  'successes':sum(r['success'] for r in valid),
                  'accessory_correct_at_end':sum(r['accessory_correct_at_end'] for r in valid),
                  'accessory_write_episodes':sum(r['accessory_write_count']>0 for r in valid),
                  'model_validation_episodes':sum(r['model_validation_calls']>0 for r in valid),
                  'preloaded_validation_episodes':sum(r['preloaded_validation_calls']>0 for r in valid),
                  'format_rejections':sum(r['format_rejections'] for r in valid),
                  'new_action_error_count':sum(r['new_action_error_count'] for r in valid),
                  'budget_exhausted':sum(r['termination_reason']=='budget_exhausted' for r in valid)}
            if mode=='judge_saved':
                cell.update(validity_correct=sum(r['validity_correct'] for r in valid),
                            best_offer_correct=sum(r['best_offer_correct'] for r in valid))
            cells.append(cell)
    return {'protocol':PROTOCOL,'stage':stage,'cells':cells,
            'interpretation':('Three development scenarios. Basic probes isolate line 0 and are not h3 order success. '
                              'Supplied validation is a harness-provided report, not a model-selected call. '
                              'No quantization or efficiency conclusion follows from these diagnostics.')}


def run(args):
    reference=args.reference_run.resolve()
    plan,source_metadata=load_reference(reference)
    print(json.dumps({'protocol':PROTOCOL,'stage':args.stage,'trials':12,
                      'modes':STAGES[args.stage],'scenarios':[0,1,2],
                      'model_id':granite.MODEL_ID,'max_generated_turns':3 if args.stage=='basic' else 8},indent=2),flush=True)
    if args.dry_run:return
    if args.output_dir.exists():raise ValueError('Use a new output directory; keep previous or partial results')
    manifest=json.loads((reference/'model_manifest.json').read_text(encoding='utf-8'))
    model_dir=Path(plan['model_dir'])
    print('Checking cached model files. No model download is required.',flush=True)
    if any(sha256_file(model_dir/n)!=h for n,h in manifest['files'].items()):
        raise ValueError('Granite model hashes differ')
    args.output_dir.mkdir(parents=True)
    snapshot=args.output_dir/'reference';snapshot.mkdir()
    for origin,name in (('plan.json','plan.json'),('model_manifest.json','model_manifest.json'),
                        ('granite/metadata.json','metadata.json'),('granite/trials.jsonl','trials.jsonl')):
        (snapshot/name).write_bytes((reference/origin).read_bytes())
    metadata={'protocol':PROTOCOL,'stage':args.stage,'source_hash':source_hash(),
              'primary_protocol_hash':granite.fingerprint(),'granite_runner_hash':granite.runner_hash(),
              'reference_run':str(reference),'model_id':granite.MODEL_ID,'revision':granite.REVISION,
              'hardware':granite.hardware(),'loaded_dtype':'float16','max_tokens':512,
              'max_generated_turns':3 if args.stage=='basic' else 8,'seed':42,
              'input_plan':input_plan(args.stage),
              'reference_hashes':{p.name:sha256_file(p) for p in snapshot.iterdir()},'warmup':[]}
    print('Loading the existing Granite checkpoint as FP16...',flush=True)
    client=GraniteClient(model_dir,max_tokens=512)
    metadata['generation_config']=client.generation.to_dict()
    metadata['package_versions']={n:version(n) for n in ('torch','transformers','accelerate','huggingface-hub','safetensors')}
    metadata['reference_package_versions']=source_metadata['package_versions']
    for _ in range(2):
        try:
            response=client.chat(copy.deepcopy(_WARMUP_MESSAGES),copy.deepcopy(_WARMUP_TOOLS))
            metadata['warmup'].append({'finish_reason':response['finish_reason']})
        except ModelOutputError as exc:metadata['warmup'].append({'model_output_error':exc.kind})
    template=client.tokenizer.get_chat_template()
    (args.output_dir/'chat_template.jinja').write_text(template,encoding='utf-8')
    metadata['chat_template_sha256']=stable_hash(template)
    granite.write_json(args.output_dir/'metadata.json',metadata)
    rows=[]
    with (args.output_dir/'trials.jsonl').open('x',encoding='utf-8') as stream:
        for i,env in enumerate(trial_plan(args.stage),1):
            trace=run_agent(env.task(),client,env,condition=env.condition,
                            max_calls=metadata['max_generated_turns'],replay_rejected_output=False)
            row={'protocol':PROTOCOL,'stage':args.stage,'base':env.base,'mode':env.mode,
                 'condition':env.condition,**score_probe(env,trace),'trace':trace}
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            print(f"[{i}/12] {env.base} {env.mode} {env.condition} success={row['success']}",flush=True)
            if row['success'] is None:raise SystemExit('Infrastructure/internal error; partial log retained. Retry in a new directory.')
            rows.append(row)
    validate_rows(rows,args.stage)
    result=summarize(rows,args.stage)
    granite.write_json(args.output_dir/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)
    print('COMPLETED: 12 valid diagnostic trials.',flush=True)


def report(args):
    metadata=json.loads((args.run_dir/'metadata.json').read_text(encoding='utf-8'))
    if (metadata['protocol']!=PROTOCOL or metadata['source_hash']!=source_hash()
            or metadata['primary_protocol_hash']!=granite.fingerprint()
            or metadata['granite_runner_hash']!=granite.runner_hash()):
        raise ValueError('Use the matching diagnostic code for this result')
    if any(sha256_file(args.run_dir/'reference'/n)!=h for n,h in metadata['reference_hashes'].items()):
        raise ValueError('Reference snapshot changed')
    if (metadata['input_plan']!=input_plan(metadata['stage'])
            or stable_hash((args.run_dir/'chat_template.jinja').read_text(encoding='utf-8'))!=metadata['chat_template_sha256']):
        raise ValueError('Input plan or template changed')
    rows=granite.read_runs(args.run_dir/'trials.jsonl')
    validate_rows(rows,metadata['stage'])
    result=summarize(rows,metadata['stage'])
    saved=json.loads((args.run_dir/'summary.json').read_text(encoding='utf-8'))
    if result!=saved:raise ValueError('Saved summary differs from replay')
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    run_parser=commands.add_parser('run')
    run_parser.add_argument('--stage',choices=tuple(STAGES),required=True)
    run_parser.add_argument('--reference-run',type=Path,required=True)
    run_parser.add_argument('--output-dir',type=Path,required=True)
    run_parser.add_argument('--dry-run',action='store_true')
    report_parser=commands.add_parser('report')
    report_parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    (run if args.command=='run' else report)(args)


if __name__=='__main__':main()
