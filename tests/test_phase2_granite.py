"""Validate Granite screening inputs, recovery grading and native client settings without weights."""

import argparse
import contextlib
import copy
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import check_phase2_granite as screen
from src.agent import run_agent
from src.client import InfrastructureError, ScriptedClient
from src.granite_client import GraniteClient, validate_granite_config


class GraniteTests(unittest.TestCase):
    def make_reference(self, directory):
        directory.mkdir()
        rows = []
        for env in screen.trials():
            trace = run_agent(env.task(), ScriptedClient([]), env, condition=env.condition,
                              max_calls=8, replay_rejected_output=False)
            rows.append({'base':env.base, 'condition':env.condition, 'backend':'hf',
                         **screen.probe_score(env, trace), 'trace':trace})
        metadata = {'backend':'hf', 'model_id':screen.QWEN_ID, 'revision':screen.QWEN_REVISION,
                    'loaded_dtype':'float16', 'runner_hash':screen.qwen_code_hash(),
                    'hardware':screen.hardware(), 'input_plan':screen.input_plan(),
                    'inference_settings':{'temperature':0, 'seed':42, 'max_tokens':512},
                    'generation_config':{'do_sample':False, 'num_beams':1,
                                         'max_new_tokens':512, 'repetition_penalty':1}}
        (directory/'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
        (directory/'trials.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        (directory.parent/'check.json').write_text(json.dumps({
            'primary_protocol_hash':screen.fingerprint(), 'hardware':screen.hardware()}), encoding='utf-8')
        return rows

    def test_rejects_other_granite_architectures(self):
        good = {'model_type':'granitemoehybrid', 'layer_types':['attention'], 'num_local_experts':0}
        validate_granite_config(good)
        for change in ({'layer_types':['mamba']}, {'num_local_experts':8}, {'model_type':'qwen3'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_granite_config({**good, **change})

    def test_native_eos_and_fp16_loading_are_not_qwen_settings(self):
        tokenizer = types.SimpleNamespace(eos_token_id=100257, eos_token='<|end_of_text|>', pad_token_id=100256)
        model = types.SimpleNamespace(config=types.SimpleNamespace(eos_token_id=100257),
                                      parameters=lambda:[types.SimpleNamespace(dtype='float16')])
        model.eval = lambda:model
        loaders = {}
        def load_model(*args, **kwargs):
            loaders.update(kwargs)
            return model
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True),
            float16='float16', manual_seed=lambda _:None,
            backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True))))
        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:tokenizer),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=load_model),
            GenerationConfig=lambda **kwargs:types.SimpleNamespace(**kwargs))
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)
            (p/'config.json').write_text(json.dumps({'model_type':'granitemoehybrid',
                'layer_types':['attention'], 'num_local_experts':0}), encoding='utf-8')
            with patch.dict('sys.modules', {'torch':fake_torch, 'transformers':fake_transformers}):
                client=GraniteClient(p)
                self.assertEqual(client.stop_ids,[100257])
                self.assertFalse(client.generation.do_sample)
                self.assertEqual(client.generation.max_new_tokens,512)
                self.assertEqual(loaders['torch_dtype'],'float16')
                tokenizer.eos_token_id=151645
                with self.assertRaises(InfrastructureError):
                    GraniteClient(p)

    def test_reference_preserves_zero_recovery_in_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'hf'
            self.make_reference(directory)
            _, rows=screen.load_reference(directory)
            self.assertEqual(len(rows),6)
            summary=screen.summarize(rows)
            self.assertEqual(summary[1]['n'],3)
            self.assertEqual(summary[1]['successes'],0)
            self.assertEqual(summary[1]['accessory_write_episodes'],0)

    def test_duplicate_infra_and_changed_start_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows=self.make_reference(Path(temporary)/'hf')
            bad_cases=[rows+[rows[0]]]
            wrong=copy.deepcopy(rows); wrong[0]['success']=None; bad_cases.append(wrong)
            wrong=copy.deepcopy(rows); wrong[0]['trace']['initial_messages'][1]['content']+='changed'; bad_cases.append(wrong)
            for bad in bad_cases:
                with self.assertRaises(ValueError): screen.validate_rows(bad)

    def test_replay_catches_altered_result_and_false_success(self):
        rows=[]
        for env in screen.trials():
            env.prepare_condition(env.task(),env.condition)
            actions=[]
            for line in range(0 if env.condition=='error_restart' else 1,4):
                action={'type':'tool_call','name':'save_offer','arguments':{'line':line,'offer':env.best(line)}}
                actions.append(action); env.execute(action['name'],action['arguments'])
            trace=run_agent(env.task(),ScriptedClient(actions),env,condition=env.condition,max_calls=8,
                            replay_rejected_output=False)
            rows.append({'base':env.base,'condition':env.condition,**screen.probe_score(env,trace),'trace':trace})
        screen.validate_rows(rows)
        self.assertTrue(all(r['success'] for r in rows))
        bad=copy.deepcopy(rows); bad[0]['trace']['events'][0]['tool_result']['ok']=False
        with self.assertRaises(ValueError): screen.validate_rows(bad)
        bad=copy.deepcopy(rows); bad[0]['success']=False
        with self.assertRaises(ValueError): screen.validate_rows(bad)

    def test_prepare_dry_run_needs_no_gpu_or_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary); self.make_reference(p/'hf')
            args=argparse.Namespace(reference_dir=p/'hf',output_dir=p/'screen',dry_run=True)
            with contextlib.redirect_stdout(io.StringIO()): screen.prepare(args)
            self.assertFalse(args.output_dir.exists())

    def test_compare_checks_plan_reference_and_candidate_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary); rows=self.make_reference(p/'hf'); run_dir=p/'screen'; run_dir.mkdir()
            reference=p/'hf'
            manifest={'files':{}}
            screen.write_json(run_dir/'model_manifest.json',manifest)
            plan={'protocol':screen.PROTOCOL,'model_id':screen.MODEL_ID,'revision':screen.REVISION,
                  'hardware':screen.hardware(),'runner_hash':screen.runner_hash(),
                  'reference_runner_hash':screen.qwen_code_hash(),'primary_protocol_hash':screen.fingerprint(),
                  'input_plan':screen.input_plan(),'reference_dir':str(reference),
                  'reference_hashes':{n:screen.sha256_file(reference/n) for n in ('metadata.json','trials.jsonl')},
                  'inference_settings':{'temperature':0,'seed':42,'max_tokens':512},
                  'manifest_hash':screen.sha256_file(run_dir/'model_manifest.json')}
            screen.write_json(run_dir/'plan.json',plan)
            destination=run_dir/'granite'; destination.mkdir()
            screen.write_json(destination/'metadata.json',{**plan,'loaded_dtype':'float16'})
            for r in rows:r['backend']='granite'
            (destination/'trials.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):screen.compare(argparse.Namespace(run_dir=run_dir))
            report=json.loads((run_dir/'comparison.json').read_text(encoding='utf-8'))
            self.assertEqual(len(report['pairs']),6)
            self.assertEqual(report['summaries']['granite_hf'][1]['successes'],0)
            (reference/'trials.jsonl').write_text('changed',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'reference files changed'):screen.load_plan(run_dir)


if __name__=='__main__':
    unittest.main()
