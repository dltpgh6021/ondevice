"""Final paired h3 display check on the twelve Granite development orders."""
import argparse
import contextlib
import io
import json
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import diagnose_phase2_recovery as recovery
from src.agent import run_agent
from src.client import ModelOutputError
from src.experiment import sha256_file
from src.granite_client import GraniteClient
from src.recovery_diagnostic import ValidationProbe
from src.run_preparation import _WARMUP_MESSAGES, _WARMUP_TOOLS
from src.tasks import stable_hash

PROTOCOL = 'granite-h3-current-menu-v1'
MODE = 'validation_supplied'
CONDITIONS = ('normal_restart', 'error_restart')
METRICS = ('order_correct', 'line0_correct', 'bundle_all_correct', 'bundle_correct_lines',
           'first_save_correct_lines', 'lines_with_save')


class CurrentMenu(ValidationProbe):
    def view(self):
        state = super().view()
        visible = state.pop('lines')
        current = next((item for item in visible if item['line'] > 0 and item['saved_offer'] is None), None)
        return {'current_bundle_menu': current, **state,
                'other_visible_lines': [item for item in visible if item is not current]}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def replay(row, environment):
    trace = row['trace']
    messages, _ = environment.prepare_condition(environment.task(), environment.condition)
    if (trace['condition'] != environment.condition or trace['base_id'] != environment.base
            or trace['initial_messages'] != messages or trace['initial_state_hash'] != environment.state_hash()
            or stable_hash(trace['initial_state']) != environment.state_hash()):
        raise ValueError('Starting checkpoint or messages differ')
    if trace['runtime_error'] or trace['internal_error'] or trace['infrastructure_error']:
        raise ValueError('Incomplete run; keep the partial output and console log')
    first = {}
    for event in trace['events']:
        if event.get('tool_executed'):
            action = event['action']
            result = environment.execute(action['name'], action['arguments'])
            if (stable_hash(result) != stable_hash(event['tool_result'])
                    or environment.state_hash() != event['state_hash_after']
                    or environment.check_success() != event['success_after_action']):
                raise ValueError('Tool replay differs')
            line = result.get('updated_line')
            if line in (1, 2, 3):
                first.setdefault(line, environment.saved[line] == environment.best(line))
    if (environment.state_hash() != trace['final_state_hash']
            or stable_hash(trace['final_state']) != environment.state_hash()):
        raise ValueError('Final state differs')
    scores = recovery.trial_scores(environment, trace, 'validation_dev12')
    return {'order_correct': scores['success'], 'line0_correct': scores['accessory_correct_at_end'],
            'bundle_all_correct': scores['bundle_all_correct'], 'bundle_correct_lines': scores['bundle_correct_lines'],
            'first_save_correct_lines': sum(first.values()), 'lines_with_save': len(first)}


def score_rows(rows, kind):
    by_key = {}
    for row in rows:
        base, condition = row['base'], row['condition']
        key = base, condition
        if key in by_key:
            raise ValueError(f'Duplicate trial: {key}')
        by_key[key] = replay(row, kind(int(base.removeprefix('dev-')), condition, MODE))
    expected = {(f'dev-{i:03d}', c) for i in range(12) for c in CONDITIONS}
    if set(by_key) != expected:
        raise ValueError('Expected twelve matched scenarios for each condition')
    return by_key


def totals(scores, condition):
    return {metric: sum(v[metric] for (_, c), v in scores.items() if c == condition) for metric in METRICS}


def compare(before, after):
    pairs = [{'base': base, 'condition': c, 'before': before[base, c], 'after': after[base, c]}
             for base, c in sorted(before)]
    cells = {}
    for condition in CONDITIONS:
        old, new = totals(before, condition), totals(after, condition)
        cells[condition] = {'before': old, 'after': new,
                            'delta': {m: new[m] - old[m] for m in METRICS}}
    gate = (cells['normal_restart']['after']['order_correct'] >= 8
            and cells['error_restart']['after']['order_correct'] >= 4)
    return {'protocol': PROTOCOL, 'pairs': pairs, 'conditions': cells, 'h3_order_thresholds_met': gate}


def read_source(directory):
    meta = read_json(directory/'metadata.json')
    if (meta['stage'] != 'validation_dev12' or meta['model_id'] != recovery.granite.MODEL_ID
            or meta['revision'] != recovery.granite.REVISION or meta['loaded_dtype'] != 'float16'
            or meta['max_generated_turns'] != 8 or meta['max_tokens'] != 512 or meta['seed'] != 42):
        raise ValueError('Use the completed 48-trial Granite FP16 h3 validation run')
    with contextlib.redirect_stdout(io.StringIO()):
        recovery.report(argparse.Namespace(run_dir=directory))
    rows = [r for r in recovery.granite.read_runs(directory/'trials.jsonl') if r['mode'] == MODE]
    return meta, score_rows(rows, ValidationProbe)


def run(source, out, meta, before, record):
    if out.exists():
        raise ValueError('Choose a new output directory')
    if recovery.granite.hardware() != meta['hardware']:
        raise ValueError('Use the original WSL host')
    if any(version(k) != v for k, v in meta['package_versions'].items()):
        raise ValueError('Package versions differ from the source experiment')
    manifest = read_json(source/'reference/model_manifest.json')
    model_dir = Path(read_json(source/'reference/plan.json')['model_dir'])
    print('Checking cached Granite files and loading FP16...', flush=True)
    if any(sha256_file(model_dir/name) != digest for name, digest in manifest['files'].items()):
        raise ValueError('Model hash mismatch')
    client = GraniteClient(model_dir, max_tokens=meta['max_tokens'])
    if (client.generation.to_dict() != meta['generation_config']
            or stable_hash(client.tokenizer.get_chat_template()) != meta['chat_template_sha256']):
        raise ValueError('Generation settings or template differ')
    out.mkdir(parents=True)
    recovery.granite.write_json(out/'metadata.json', record)
    for _ in range(2):
        try:
            client.chat(_WARMUP_MESSAGES, _WARMUP_TOOLS)
        except ModelOutputError:
            continue
    with (out/'trials.jsonl').open('x', encoding='utf-8') as stream:
        # Preserve the condition order of the source's 24 supplied-report trials.
        for i, (base, condition) in enumerate(before, 1):
            env = CurrentMenu(int(base.removeprefix('dev-')), condition, MODE)
            trace = run_agent(env.task(), client, env, condition=condition, max_calls=8,
                              replay_rejected_output=False)
            row = {'base': base, 'condition': condition, 'trace': trace}
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            scores = replay(row, env)
            print(f"[{i}/24] {base} {condition} order={scores['order_correct']} "
                  f"line0={scores['line0_correct']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run', 'check'))
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    meta, before = read_source(args.source)
    for c in CONDITIONS:
        print(f"Baseline {c}: {totals(before, c)}", flush=True)
    if args.command == 'plan':
        print('Plan: 24 new h3 trials; 12 normal, 12 error.')
        return
    if args.out is None:
        parser.error('--out is required for run and check')
    record = {k: meta[k] for k in ('model_id', 'revision', 'loaded_dtype', 'hardware', 'package_versions',
                                    'generation_config', 'chat_template_sha256', 'max_generated_turns', 'seed')}
    record.update(protocol=PROTOCOL, source_metadata_sha256=sha256_file(args.source/'metadata.json'),
                  source_trials_sha256=sha256_file(args.source/'trials.jsonl'),
                  script_hash=stable_hash(Path(__file__).read_text(encoding='utf-8')))
    if args.command == 'run':
        run(args.source, args.out, meta, before, record)
    if read_json(args.out/'metadata.json') != record:
        raise ValueError('Result metadata differs from this source and code')
    after = score_rows(recovery.granite.read_runs(args.out/'trials.jsonl'), CurrentMenu)
    result = compare(before, after)
    if args.command == 'run':
        recovery.granite.write_json(args.out/'comparison.json', result)
    elif result != read_json(args.out/'comparison.json'):
        raise ValueError('Stored comparison differs from replay')
    print(json.dumps(result['conditions'], indent=2))
    print('H3 order thresholds met:', result['h3_order_thresholds_met'])
    print('Verified 24 paired trials.')


if __name__ == '__main__':
    main()
