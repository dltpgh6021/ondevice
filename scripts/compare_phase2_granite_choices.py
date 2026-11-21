"""Pair Granite's choices in h3 with choices on the exact same menus in isolation."""

import argparse
import contextlib
import io
import json
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import diagnose_phase2_recovery as recovery
from scripts.diagnose_phase2 import IsolatedChoice
from src.agent import run_agent
from src.client import ModelOutputError
from src.experiment import sha256_file
from src.granite_client import GraniteClient
from src.recovery_diagnostic import ValidationProbe, VALIDATE_TOOL
from src.run_preparation import _WARMUP_MESSAGES, _WARMUP_TOOLS
from src.tasks import stable_hash

PROTOCOL = 'granite-h3-matched-choice-v1'
CONDITION = 'normal_restart'
MODE = 'validation_supplied'


class IsolatedMenu(IsolatedChoice):
    """Keep the original validator and save/inspect schemas available on a frozen menu."""

    validation_report = ValidationProbe.validation_report

    def tool_definitions(self):
        return [*super().tool_definitions(), VALIDATE_TOOL]

    def execute(self, name, args):
        if name == 'validate_order':
            if args:
                return {'ok': False, 'error': 'validate_order takes no arguments.'}
            return self.validation_report()
        return super().execute(name, args)


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def gather_pairs(rows):
    pairs = []
    for row in sorted(rows, key=lambda item: item['base']):
        if (row['condition'], row['mode']) != (CONDITION, MODE):
            continue
        trace = row['trace']
        # Track the last menu actually shown before each first save. A validation
        # result has no menus; a save/inspect result replaces the visible order.
        views = [json.loads(m['content']) for m in trace['initial_messages'] if m['role'] == 'tool']
        visible = next(v['lines'] for v in reversed(views) if 'lines' in v)
        seen = set()
        for event in trace['events']:
            result = event.get('tool_result', {})
            line = result.get('updated_line')
            if line in (1, 2, 3) and line not in seen:
                item = next(m for m in visible if m['line'] == line)
                menu = {k: item[k] for k in ('line', 'budget', 'offers')}
                pairs.append({'base': row['base'], 'menu': menu, 'h3_turn': event['step'],
                              'h3_offer': event['action']['arguments']['offer']})
                seen.add(line)
            if 'lines' in result:
                visible = result['lines']
        if seen != {1, 2, 3}:
            raise ValueError(f"Missing observed menu in {row['base']}; inspect source trace")
    keys = {(p['base'], p['menu']['line']) for p in pairs}
    if len(pairs) != 36 or keys != {(f'dev-{i:03d}', line) for i in range(12) for line in (1, 2, 3)}:
        raise ValueError('Expected 36 matched menus from twelve different h3 orders')
    return pairs


def outcome(menu, offer):
    if offer is None:
        return 'no_save'
    selected = next(o for o in menu['offers'] if o['offer'] == offer)
    affordable = [o for o in menu['offers'] if o['price'] <= menu['budget']]
    if selected['price'] > menu['budget']:
        return 'over_budget'
    if selected['rating'] < max(o['rating'] for o in affordable):
        return 'lower_rating'
    return 'correct'


def evaluate(pair, trace):
    menu = pair['menu']
    env = IsolatedMenu(int(pair['base'][-3:]), menu)
    messages, _ = env.prepare_condition(env.task(), 'free')
    if trace['runtime_error'] or trace['internal_error'] or trace['infrastructure_error']:
        raise ValueError('Incomplete trial; keep partial traces and inspect console.log')
    if trace['initial_messages'] != messages or trace['initial_state_hash'] != env.state_hash():
        raise ValueError('Isolated input differs from the paired menu')
    first = None
    for event in trace['events']:
        if event.get('tool_executed'):
            action = event['action']
            result = env.execute(action['name'], action['arguments'])
            if stable_hash(result) != stable_hash(event['tool_result']) or env.state_hash() != event['state_hash_after']:
                raise ValueError('Tool observation or state differs on replay')
            if first is None and result.get('updated_line') == env.line:
                first = env.saved[env.line]
    if env.state_hash() != trace['final_state_hash'] or env.state_hash() != stable_hash(trace['final_state']):
        raise ValueError('Final state differs on replay')
    return {'base': pair['base'], 'line': env.line, 'menu_sha256': stable_hash(menu),
            'h3_first': outcome(menu, pair['h3_offer']), 'isolated_first': outcome(menu, first),
            'isolated_final': outcome(menu, env.saved.get(env.line))}


def summary(pairs, rows):
    if len(rows) != len(pairs):
        raise ValueError('Incomplete comparison')
    decisions = []
    for pair, row in zip(pairs, rows):
        if (row['base'], row['line']) != (pair['base'], pair['menu']['line']):
            raise ValueError('Mismatched scenario/line')
        decisions.append(evaluate(pair, row['trace']))
    changes = {'both_correct': [], 'gained': [], 'lost': [], 'both_failed': []}
    for d in decisions:
        before, after = d['h3_first'] == 'correct', d['isolated_first'] == 'correct'
        label = {(True, True): 'both_correct', (False, True): 'gained',
                 (True, False): 'lost', (False, False): 'both_failed'}[before, after]
        changes[label].append(f"{d['base']}/line{d['line']}")
    totals = {field: dict(Counter(d[field] for d in decisions))
              for field in ('h3_first', 'isolated_first', 'isolated_final')}
    return {'protocol': PROTOCOL, 'n_menus': len(rows), 'outcomes': totals, 'changes': changes,
            'decisions': decisions}


def source_data(directory):
    metadata = load_json(directory/'metadata.json')
    if (metadata['stage'] != 'validation_dev12' or metadata['model_id'] != recovery.granite.MODEL_ID
            or metadata['revision'] != recovery.granite.REVISION or metadata['loaded_dtype'] != 'float16'):
        raise ValueError('Use the completed Granite FP16 validation_dev12 run directory')
    with contextlib.redirect_stdout(io.StringIO()):
        recovery.report(argparse.Namespace(run_dir=directory))
    return metadata, gather_pairs(recovery.granite.read_runs(directory/'trials.jsonl'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run', 'check'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    meta, pairs = source_data(args.source)
    baseline = Counter(outcome(p['menu'], p['h3_offer']) for p in pairs)
    print(f"Menus: {len(pairs)}. Original h3 first choices: {dict(baseline)}", flush=True)
    if args.command == 'plan':
        return
    if not args.out:
        parser.error('run and check need --out')
    out = args.out
    provenance = {'protocol': PROTOCOL, 'source_trials_sha256': sha256_file(args.source/'trials.jsonl'),
                  'source_metadata_sha256': sha256_file(args.source/'metadata.json'),
                  'pairs_sha256': stable_hash(pairs),
                  'script_sha256': stable_hash(Path(__file__).read_text(encoding='utf-8')),
                  'isolated_env_sha256': stable_hash((ROOT/'scripts/diagnose_phase2.py').read_text(encoding='utf-8'))}
    if args.command == 'run':
        if out.exists():
            raise ValueError('Use a new output directory')
        if recovery.granite.hardware() != meta['hardware']:
            raise ValueError('Use the same WSL PC as the source experiment')
        installed = {name: version(name) for name in meta['package_versions']}
        if installed != meta['package_versions']:
            raise ValueError('Restore package versions from source metadata')
        manifest = load_json(args.source/'reference/model_manifest.json')
        model_dir = Path(load_json(args.source/'reference/plan.json')['model_dir'])
        print('Checking cached model and loading Granite FP16...', flush=True)
        if any(sha256_file(model_dir/name) != digest for name, digest in manifest['files'].items()):
            raise ValueError('Model files differ from the reference manifest')
        client = GraniteClient(model_dir, max_tokens=meta['max_tokens'])
        if (client.generation.to_dict() != meta['generation_config'] or
                stable_hash(client.tokenizer.get_chat_template()) != meta['chat_template_sha256']):
            raise ValueError('Reference generation settings or chat template differ')
        out.mkdir(parents=True)
        recovery.granite.write_json(out/'metadata.json', {**provenance, 'hardware': meta['hardware'],
            'model_id': meta['model_id'], 'revision': meta['revision'], 'loaded_dtype': 'float16',
            'package_versions': installed, 'generation_config': meta['generation_config'],
            'max_generated_turns': meta['max_generated_turns']})
        recovery.granite.write_json(out/'menus.json', pairs)
        for _ in range(2):
            try:
                client.chat(_WARMUP_MESSAGES, _WARMUP_TOOLS)
            except ModelOutputError:
                pass  # Warm-up is excluded from the comparison.
        with (out/'trials.jsonl').open('x', encoding='utf-8') as stream:
            for i, pair in enumerate(pairs, 1):
                env = IsolatedMenu(int(pair['base'][-3:]), pair['menu'])
                trace = run_agent(env.task(), client, env, condition='free',
                                  max_calls=meta['max_generated_turns'], replay_rejected_output=False)
                row = {'base': pair['base'], 'line': env.line, 'trace': trace}
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                scored = evaluate(pair, trace)
                print(f"[{i}/36] {env.base} line {env.line}: {scored['isolated_first']}", flush=True)
    recorded = load_json(out/'metadata.json')
    if any(recorded.get(k) != v for k, v in provenance.items()) or load_json(out/'menus.json') != pairs:
        raise ValueError('Source or comparison inputs differ')
    result = summary(pairs, recovery.granite.read_runs(out/'trials.jsonl'))
    if args.command == 'run':
        recovery.granite.write_json(out/'summary.json', result)
    elif result != load_json(out/'summary.json'):
        raise ValueError('Stored summary differs from replay')
    print(json.dumps(result['outcomes'], indent=2))
    print('Verified 36 paired menus.', flush=True)


if __name__ == '__main__':
    main()
