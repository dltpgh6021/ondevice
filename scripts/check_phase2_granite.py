"""Prepare, run and compare six Granite FP16 trials against the completed Qwen HF check."""

import argparse
import copy
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_phase2_backends import (
    MODEL_ID as QWEN_ID, REVISION as QWEN_REVISION, code_hash as qwen_code_hash,
    hardware, input_plan, summarize, trials, write_json,
)
from scripts.diagnose_phase2 import probe_score
from scripts.phase2 import fingerprint, read_runs
from src.agent import run_agent
from src.client import ModelOutputError
from src.experiment import sha256_file
from src.granite_client import GraniteClient, validate_granite_config
from src.run_preparation import _WARMUP_MESSAGES, _WARMUP_TOOLS
from src.tasks import stable_hash

PROTOCOL = 'phase2-granite-screen-v1'
MODEL_ID = 'ibm-granite/granite-4.0-micro'
REVISION = '56111ae135df9c53a78c99028e7bc24035a9e979'
MODEL_FILES = ('config.json', 'generation_config.json', 'tokenizer_config.json', 'tokenizer.json',
               'vocab.json', 'merges.txt', 'special_tokens_map.json', 'chat_template.jinja',
               'model.safetensors.index.json')


def runner_hash():
    names = ('scripts/check_phase2_granite.py', 'src/granite_client.py')
    return stable_hash({name: (ROOT/name).read_text(encoding='utf-8') for name in names})


def validate_rows(rows):
    expected = {(p['base'], p['condition']): p for p in input_plan()}
    if (len(rows) != 6 or {(r['base'], r['condition']) for r in rows} != set(expected)
            or any(type(r.get('success')) is not bool for r in rows)):
        raise ValueError('Need six complete, unique and valid trials')
    for row in rows:
        trace = row['trace']
        p = expected[row['base'], row['condition']]
        if (stable_hash(trace['initial_messages']) != p['messages_sha256']
                or stable_hash(trace['initial_state']) != p['state_sha256']
                or trace['initial_state_hash'] != p['state_sha256']
                or trace['horizon'] != 3):
            raise ValueError('Starting messages/state differ from the fixed six tasks')
        env = next(e for e in trials() if (e.base, e.condition) == (row['base'], row['condition']))
        env.prepare_condition(env.task(), env.condition)
        for event in trace['events']:
            if event.get('tool_executed'):
                action = event['action']
                result = env.execute(action['name'], action['arguments'])
                if (stable_hash(result) != stable_hash(event['tool_result'])
                        or env.state_hash() != event['state_hash_after']
                        or env.check_success() != event['success_after_action']):
                    raise ValueError('Tool result or state does not replay')
        if env.state_hash() != trace['final_state_hash'] or env.state_hash() != stable_hash(trace['final_state']):
            raise ValueError('Final state does not replay')
        if any(stable_hash(row[k]) != stable_hash(v) for k,v in probe_score(env, trace).items()):
            raise ValueError('Recorded metrics differ from replay')


def load_reference(directory):
    metadata = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    if (metadata.get('backend') != 'hf' or metadata.get('model_id') != QWEN_ID
            or metadata.get('revision') != QWEN_REVISION or metadata.get('loaded_dtype') != 'float16'
            or metadata.get('runner_hash') != qwen_code_hash()
            or metadata.get('input_plan') != input_plan()):
        raise ValueError('Use the official Qwen HF results from the completed backend check')
    check = json.loads((directory.parent/'check.json').read_text(encoding='utf-8'))
    if check['primary_protocol_hash'] != fingerprint() or check['hardware'] != metadata['hardware']:
        raise ValueError('Qwen protocol or hardware metadata differs')
    generation = metadata['generation_config']
    inference = metadata['inference_settings']
    if (generation['do_sample'] is not False or generation['num_beams'] != 1
            or generation['max_new_tokens'] != 512 or generation['repetition_penalty'] != 1
            or inference['temperature'] != 0 or inference['seed'] != 42
            or inference['max_tokens'] != 512):
        raise ValueError('Restore the reference greedy, seed-42, 512-token settings')
    rows = read_runs(directory/'trials.jsonl')
    validate_rows(rows)
    return metadata, rows


def weight_names(model_dir):
    index = json.loads((model_dir/'model.safetensors.index.json').read_text(encoding='utf-8'))
    names = sorted(set(index['weight_map'].values()))
    if not names or any(Path(n).name != n or not n.endswith('.safetensors') for n in names):
        raise ValueError('Unexpected weight shard path in the model index')
    return names


def prepare(args):
    reference = args.reference_dir.resolve()
    metadata, _ = load_reference(reference)
    if metadata['hardware'] != hardware():
        raise ValueError('Use the same PC as the Qwen reference; rerun that reference on a new PC')
    print(json.dumps({'model_id': MODEL_ID, 'revision': REVISION, 'trials': 6, 'horizon': 3,
                      'max_response_turns': 8, 'max_tokens_per_response': 512,
                      'reference_dir': str(reference)}, indent=2), flush=True)
    if args.dry_run:
        return
    if args.output_dir.exists():
        raise ValueError('Use a new output directory; keep partial results for diagnosis')
    from huggingface_hub import snapshot_download
    print('Downloading the pinned Granite checkpoint (about 6.81 GB weights)...', flush=True)
    model_dir = Path(snapshot_download(MODEL_ID, revision=REVISION, local_dir=args.model_dir,
                                     allow_patterns=[*MODEL_FILES, '*.safetensors'], max_workers=2)).resolve()
    validate_granite_config(json.loads((model_dir/'config.json').read_text(encoding='utf-8')))
    files = [*MODEL_FILES, *weight_names(model_dir)]
    print('Hashing model files...', flush=True)
    hashes = {name: sha256_file(model_dir/name) for name in files}
    manifest = {'model_id': MODEL_ID, 'revision': REVISION, 'source_dtype': 'bfloat16',
                'execution_dtype': 'float16', 'files': hashes,
                'weight_bytes': sum((model_dir/n).stat().st_size for n in weight_names(model_dir))}
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'model_manifest.json', manifest)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    plan = {'protocol': PROTOCOL, 'model_id': MODEL_ID, 'revision': REVISION, 'model_dir': str(model_dir),
            'hardware': hardware(), 'input_plan': input_plan(), 'primary_protocol_hash': fingerprint(),
            'reference_runner_hash': qwen_code_hash(), 'runner_hash': runner_hash(), 'project_commit': commit,
            'reference_dir': str(reference), 'reference_hashes': {
                name: sha256_file(reference/name) for name in ('metadata.json', 'trials.jsonl')},
            'inference_settings': metadata['inference_settings'], 'manifest_hash': sha256_file(args.output_dir/'model_manifest.json')}
    write_json(args.output_dir/'plan.json', plan)
    print('PREPARED: six paired Granite trials. No GPU inference has run yet.', flush=True)


def load_plan(directory):
    plan = json.loads((directory/'plan.json').read_text(encoding='utf-8'))
    if (plan['protocol'] != PROTOCOL or plan['model_id'] != MODEL_ID or plan['revision'] != REVISION
            or plan['hardware'] != hardware() or plan['runner_hash'] != runner_hash()
            or plan['reference_runner_hash'] != qwen_code_hash()
            or plan['primary_protocol_hash'] != fingerprint() or plan['input_plan'] != input_plan()):
        raise ValueError('Code, hardware or task plan changed; prepare a new run')
    if sha256_file(directory/'model_manifest.json') != plan['manifest_hash']:
        raise ValueError('Model manifest changed')
    reference = Path(plan['reference_dir'])
    if any(sha256_file(reference/n) != h for n,h in plan['reference_hashes'].items()):
        raise ValueError('Qwen reference files changed')
    metadata, rows = load_reference(reference)
    if metadata['inference_settings'] != plan['inference_settings']:
        raise ValueError('Reference inference settings changed')
    return plan, rows


def run(args):
    plan, _ = load_plan(args.run_dir)
    output = args.run_dir/'granite'
    if output.exists():
        raise ValueError('Granite results already exist; prepare a new run for a retry')
    model_dir = Path(plan['model_dir'])
    manifest = json.loads((args.run_dir/'model_manifest.json').read_text(encoding='utf-8'))
    print('Verifying Granite model hashes...', flush=True)
    if any(sha256_file(model_dir/n) != h for n,h in manifest['files'].items()):
        raise ValueError('Granite model files changed')
    output.mkdir()
    print('Loading Granite as FP16 on cuda:0. The previous llama-server must be stopped.', flush=True)
    client = GraniteClient(model_dir, max_tokens=512)
    metadata = {'protocol': PROTOCOL, 'model_id': MODEL_ID, 'revision': REVISION,
                'loaded_dtype': 'float16', 'attention_implementation': 'sdpa', 'hardware': hardware(),
                'runner_hash': runner_hash(), 'primary_protocol_hash': fingerprint(),
                'reference_runner_hash': qwen_code_hash(), 'input_plan': input_plan(),
                'inference_settings': plan['inference_settings'], 'manifest_hash': plan['manifest_hash'],
                'generation_config': client.generation.to_dict(), 'warmup': [],
                'tokenizer': {'bos_token_id': client.tokenizer.bos_token_id,
                              'eos_token_id': client.tokenizer.eos_token_id,
                              'pad_token_id': client.tokenizer.pad_token_id},
                'package_versions': {n: version(n) for n in ('torch', 'transformers', 'accelerate', 'huggingface-hub', 'safetensors')}}
    for _ in range(2):
        try:
            response = client.chat(copy.deepcopy(_WARMUP_MESSAGES), copy.deepcopy(_WARMUP_TOOLS))
            metadata['warmup'].append({'finish_reason': response['finish_reason']})
        except ModelOutputError as exc:
            metadata['warmup'].append({'model_output_error': exc.kind})
    template = client.tokenizer.get_chat_template()
    (output/'chat_template.jinja').write_text(template, encoding='utf-8')
    metadata['chat_template_sha256'] = stable_hash(template)
    write_json(output/'metadata.json', metadata)
    rows = []
    with (output/'trials.jsonl').open('x', encoding='utf-8') as stream:
        for i, env in enumerate(trials(), 1):
            trace = run_agent(env.task(), client, env, condition=env.condition, max_calls=8,
                              replay_rejected_output=False)
            row = {'protocol': PROTOCOL, 'backend': 'granite', 'base': env.base, 'condition': env.condition,
                   **probe_score(env, trace), 'trace': trace}
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
            stream.flush()
            print(f"[{i}/6] {env.base} {env.condition} success={row['success']}", flush=True)
            if row['success'] is None:
                raise SystemExit('Infrastructure/internal error; partial log retained. Use a new run after fixing.')
            rows.append(row)
    validate_rows(rows)
    write_json(output/'summary.json', summarize(rows))
    print('COMPLETED: six valid trials. Success=False is a model outcome, not an execution error.', flush=True)


def compare(args):
    plan, qwen = load_plan(args.run_dir)
    granite = read_runs(args.run_dir/'granite/trials.jsonl')
    validate_rows(granite)
    metadata = json.loads((args.run_dir/'granite/metadata.json').read_text(encoding='utf-8'))
    for k in ('protocol', 'model_id', 'revision', 'runner_hash', 'hardware', 'input_plan',
              'primary_protocol_hash', 'reference_runner_hash', 'inference_settings', 'manifest_hash'):
        if metadata[k] != plan[k]:
            raise ValueError(f'Granite metadata differs: {k}')
    if metadata['loaded_dtype'] != 'float16' or any(r.get('backend') != 'granite' for r in granite):
        raise ValueError('Unexpected model precision or backend')
    qwen_by_key = {(r['base'],r['condition']):r for r in qwen}
    pairs = []
    for g in granite:
        q = qwen_by_key[g['base'],g['condition']]
        pairs.append({'base':g['base'], 'condition':g['condition'],
                      'qwen_success':q['success'], 'granite_success':g['success'],
                      'qwen_accessory_correct':q['accessory_correct_at_end'],
                      'granite_accessory_correct':g['accessory_correct_at_end'],
                      'qwen_accessory_writes':q['accessory_write_count'],
                      'granite_accessory_writes':g['accessory_write_count']})
    result = {'protocol':PROTOCOL, 'summaries':{'qwen_hf':summarize(qwen), 'granite_hf':summarize(granite)},
              'pairs':pairs, 'interpretation':'Three development scenarios, two paired states. Candidate screening only; native chat templates differ.'}
    write_json(args.run_dir/'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('--reference-dir', type=Path, required=True, help='Completed backend_check directory / hf')
    prep.add_argument('--output-dir', type=Path, required=True)
    prep.add_argument('--model-dir', type=Path, default=ROOT/'models/granite-4.0-micro-hf')
    prep.add_argument('--dry-run', action='store_true')
    for name in ('run', 'compare'):
        sub = subs.add_parser(name)
        sub.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    {'prepare':prepare, 'run':run, 'compare':compare}[args.command](args)


if __name__ == '__main__':
    main()
