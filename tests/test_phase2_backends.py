"""Check raw Qwen parsing, matched inputs and backend result comparisons."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_phase2_backends import (
    PROTOCOL, code_hash, compare, hardware, input_plan, summarize, trials,
)
from scripts.diagnose_phase2 import probe_score
from scripts.phase2 import fingerprint
from src.agent import run_agent
from src.client import ModelOutputError, ScriptedClient
from src.tasks import stable_hash
from src.transformers_client import qwen_response


class BackendTests(unittest.TestCase):
    def test_official_tool_tags_are_executed_and_raw_text_is_retained(self):
        text = '<tool_call>\n{"name":"save_offer","arguments":{"line":0,"offer":"abc"}}\n</tool_call>'
        response = qwen_response(text, finish_reason="stop", usage={"completion_tokens": 20})
        self.assertEqual(response["action"], {"type": "tool_call", "name": "save_offer",
                                              "arguments": {"line": 0, "offer": "abc"}})
        self.assertEqual(response["raw_response"]["generated_text"], text)

    def test_plain_final_and_plain_json_use_the_common_parser(self):
        self.assertEqual(qwen_response("Done.", finish_reason="stop", usage={})["action"]["type"], "finish")
        text = '{"name":"inspect_order","arguments":{}}'
        self.assertEqual(qwen_response(text, finish_reason="stop", usage={})["action"]["name"], "inspect_order")

    def test_malformed_and_multiple_calls_are_rejected(self):
        valid = '<tool_call>{"name":"inspect_order","arguments":{}}</tool_call>'
        for text in (valid + valid, '<tool_call>{broken}</tool_call>',
                     '<tool_call>{"name":"inspect_order","arguments":{}}',
                     '<think>Unfinished', valid + '<tool_call>'):
            with self.subTest(text=text), self.assertRaises(ModelOutputError):
                qwen_response(text, finish_reason="stop", usage={})

    def test_truncated_output_never_executes_a_complete_looking_call(self):
        for text in ('<tool_call>{"name":"inspect_order","arguments":{}}</tool_call>', '<think>unfinished'):
            with self.assertRaises(ModelOutputError) as caught:
                qwen_response(text, finish_reason="length", usage={})
            self.assertEqual(caught.exception.kind, "output_truncated")

    def test_closed_reasoning_is_retained_in_raw_response(self):
        text = '<think>review</think>\n<tool_call>{"name":"inspect_order","arguments":{}}</tool_call>'
        response = qwen_response(text, finish_reason="stop", usage={})
        self.assertEqual(response["action"]["name"], "inspect_order")
        self.assertEqual(response["raw_response"]["generated_text"], text)

    def test_six_inputs_are_paired_and_repairable_within_budget(self):
        plan = input_plan()
        self.assertEqual(len(plan), 6)
        self.assertEqual({p["base"] for p in plan}, {"dev-000", "dev-001", "dev-002"})
        for normal, error in zip(plan[::2], plan[1::2]):
            self.assertEqual(normal["tools_sha256"], error["tools_sha256"])
            self.assertNotEqual(normal["state_sha256"], error["state_sha256"])
        for env in trials():
            env.prepare_condition(env.task(), env.condition)
            actions = []
            for line in range(0 if env.condition == "error_restart" else 1, 4):
                args = {"line": line, "offer": env.best(line)}
                actions.append({"type": "tool_call", "name": "save_offer", "arguments": args})
                env.execute("save_offer", args)
            trace = run_agent(env.task(), ScriptedClient(actions), env, condition=env.condition,
                              max_calls=8, replay_rejected_output=False)
            self.assertTrue(probe_score(env, trace)["success"])
            self.assertTrue(trace["finished"])

    def make_comparison(self, directory):
        plan = input_plan()
        check = {"protocol": PROTOCOL, "primary_protocol_hash": fingerprint(), "runner_hash": code_hash(),
                 "hardware": hardware(), "input_plan": plan}
        (directory/"check.json").write_text(json.dumps(check), encoding="utf-8")
        for backend in ("gguf", "hf"):
            destination = directory/backend
            destination.mkdir()
            metadata = {**check, "loaded_dtype": "float16"}
            (destination/"metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            rows = []
            for env in trials():
                trace = run_agent(env.task(), ScriptedClient([]), env, condition=env.condition,
                                  max_calls=8, replay_rejected_output=False)
                trace["events"][0]["raw_response"] = {"input_audit": {"rendered_prompt": "same"}}
                rows.append({"backend": backend, "base": env.base, "condition": env.condition,
                             **probe_score(env, trace), "trace": trace})
            (destination/"trials.jsonl").write_text(
                "".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")

    def test_comparison_checks_starting_inputs_and_counts_zero_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_comparison(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                compare(argparse.Namespace(check_dir=directory))
            report = json.loads((directory/"comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["pairs"]), 6)
            self.assertTrue(all(p["initial_rendered_prompts_equal"] for p in report["pairs"]))
            self.assertTrue(all(c["successes"] == 0 for c in report["summaries"]["hf"]))

    def test_comparison_rejects_input_tampering_or_duplicate_trials(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_comparison(directory)
            path = directory/"hf/trials.jsonl"
            original = path.read_text(encoding="utf-8")
            rows = [json.loads(line) for line in original.splitlines()]
            rows[0]["trace"]["initial_messages"][1]["content"] += "changed"
            path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different starting"):
                compare(argparse.Namespace(check_dir=directory))
            path.write_text(original + original.splitlines()[0]+"\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing, duplicate"):
                compare(argparse.Namespace(check_dir=directory))


if __name__ == "__main__":
    unittest.main()
