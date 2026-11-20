"""Prepare, run and compare six paired tasks on GGUF and official Qwen FP16."""

import argparse
import copy
import json
import platform
import shlex
import sys
import urllib.request
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_phase2 import launch_settings, probe_score
from scripts.phase2 import fingerprint, read_runs, require_complete
from src.agent import run_agent
from src.client import InfrastructureError, LlamaCppClient, ModelOutputError
from src.experiment import sha256_file
from src.phase2 import PurchaseChain, TOOLS
from src.run_preparation import prepare_run, _WARMUP_MESSAGES, _WARMUP_TOOLS
from src.tasks import stable_hash
from src.transformers_client import TransformersClient

PROTOCOL = "phase2-official-backend-check-v1"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
GGUF_SHA256 = "336228a2810f17724ecd6e9e2e40e8381eee81e886316fee311ade2899c31030"
MODEL_FILES = ("config.json", "generation_config.json", "tokenizer_config.json",
               "tokenizer.json", "vocab.json", "merges.txt", "model.safetensors.index.json")


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def code_hash():
    names = ("scripts/compare_phase2_backends.py", "src/transformers_client.py")
    return stable_hash({name: (ROOT / name).read_text(encoding="utf-8") for name in names})


def trials():
    for scenario in (0, 1, 2):
        for condition in ("normal_restart", "error_restart"):
            yield PurchaseChain("dev", scenario, 3, condition)


def input_plan():
    result = []
    for env in trials():
        messages, _ = env.prepare_condition(env.task(), env.condition)
        result.append({"base": env.base, "condition": env.condition, "horizon": env.h,
                       "messages_sha256": stable_hash(messages), "state_sha256": env.state_hash(),
                       "tools_sha256": stable_hash(env.tool_definitions())})
    return result


def hardware():
    return {"hostname": platform.node(), "platform": platform.platform(),
            "processor": platform.processor(), "machine": platform.machine()}


def load_check(directory):
    check = json.loads((directory / "check.json").read_text(encoding="utf-8"))
    if (check["protocol"] != PROTOCOL or check["primary_protocol_hash"] != fingerprint()
            or check["runner_hash"] != code_hash() or check["input_plan"] != input_plan()):
        raise ValueError("Prepared code or task inputs changed; prepare a new check directory")
    if check["hardware"] != hardware():
        raise ValueError("Use both backends on the prepared machine")
    return check


def prepare(args):
    baseline = read_runs(args.baseline_log)
    require_complete(baseline, "dev", "FP16")
    if baseline[0]["model_sha256"] != GGUF_SHA256 or baseline[0]["protocol_hash"] != fingerprint():
        raise ValueError("Use the completed 4B FP16 pilot from the current protocol")
    print(json.dumps({"protocol": PROTOCOL, "trials_per_backend": 6, "total": 12,
                      "scenarios": [0, 1, 2], "horizon": 3, "input_plan": input_plan()}, indent=2))
    if args.dry_run:
        return
    if args.output_dir.exists():
        raise ValueError("Choose a new output directory")
    from huggingface_hub import snapshot_download
    args.output_dir.mkdir(parents=True)
    model_dir = Path(snapshot_download(MODEL_ID, revision=REVISION, local_dir=args.hf_dir,
                                     allow_patterns=list(MODEL_FILES), max_workers=2)).resolve()
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    template = tokenizer_config["chat_template"]
    (args.output_dir / "official_template.jinja").write_text(template, encoding="utf-8")
    check = {"protocol": PROTOCOL, "model_id": MODEL_ID, "revision": REVISION,
             "hf_dir": str(model_dir), "baseline_log": str(args.baseline_log.resolve()),
             "baseline_sha256": sha256_file(args.baseline_log),
             "primary_protocol_hash": fingerprint(), "runner_hash": code_hash(),
             "hardware": hardware(), "input_plan": input_plan(),
             "official_template_sha256": stable_hash(template),
             "metadata_files": {name: sha256_file(model_dir/name) for name in MODEL_FILES}}
    write_json(args.output_dir / "check.json", check)
    print("Prepared six inputs. Official weights download when the hf stage starts.", flush=True)


class AuditedLlamaClient(LlamaCppClient):
    def chat(self, messages, tools):
        body = {"messages": messages, "tools": tools, "add_generation_prompt": True, **self.request_settings()}
        request = urllib.request.Request(self.base_url + "/apply-template",
                                         data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        try:
            rendered = self._request_json(request).get("prompt")
            audit = {"rendered_prompt": rendered, "source": "server_apply_template"}
        except (InfrastructureError, ModelOutputError) as exc:
            audit = {"rendered_prompt": None, "render_error": str(exc)}
        try:
            response = super().chat(messages, tools)
        except ModelOutputError as exc:
            exc.payload["input_audit"] = audit
            raise
        response["raw_response"]["input_audit"] = audit
        return response


def run(args):
    check = load_check(args.check_dir)
    output = args.check_dir / args.backend
    if output.exists():
        raise ValueError("This backend already has a result directory; use a newly prepared check for retries")
    baseline_path = Path(check["baseline_log"])
    if sha256_file(baseline_path) != check["baseline_sha256"]:
        raise ValueError("The baseline log changed")
    lock_path = baseline_path.parent / "settings.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    baseline = read_runs(baseline_path)
    if (stable_hash(lock["configuration"]) != lock["configuration_sha256"]
            or lock["configuration_sha256"] != baseline[0]["settings_lock_sha256"]):
        raise ValueError("The baseline settings lock changed")
    inference = lock["configuration"]["inference_settings"]
    if (inference["temperature"], inference["seed"], inference["cache_prompt"]) != (0, 42, False):
        raise ValueError("This comparison implements the pilot's greedy seed-42 settings with prompt caching off")
    metadata = {"protocol": PROTOCOL, "backend": args.backend, "hardware": hardware(),
                "input_plan": input_plan(), "runner_hash": code_hash(),
                "inference_settings": inference, "loaded_dtype": "float16"}
    if args.backend == "gguf":
        if not args.model_path or not args.server_pid:
            raise ValueError("The gguf stage requires --model-path and the current --server-pid")
        if sha256_file(args.model_path) != GGUF_SHA256:
            raise ValueError("Use the exact F16 GGUF from the pilot")
        client = AuditedLlamaClient(args.base_url, "local", max_tokens=inference["max_tokens"],
                                    timeout_seconds=inference["timeout_seconds"], accept_plain_call=True)
        props = client.get_server_properties()
        served = Path(props.get("model_path") or "")
        if not served.is_file() or (not served.samefile(args.model_path) and sha256_file(served) != GGUF_SHA256):
            raise ValueError("The server loaded a different model")
        import psutil
        argv = psutil.Process(args.server_pid).cmdline()
        expected = shlex.split((baseline_path.parent / "server_command.txt").read_text(encoding="utf-8"))
        if launch_settings(argv) != launch_settings(expected):
            raise ValueError("Restore the pilot's server launch settings")
        from urllib.parse import urlparse
        address, options = urlparse(args.base_url), launch_settings(argv)
        if (address.hostname, address.port or 80) != (options["--host"], int(options["--port"])):
            raise ValueError("The server PID and endpoint differ")
        output.mkdir()
        setup = prepare_run(client, expected_context_size=8192, warmup_requests=2, settings_lock_path=lock_path,
                            report_path=output/"preflight.json", hardware=hardware())
        metadata.update(model_sha256=GGUF_SHA256, server_argv=argv, warmup=setup["warmup"],
                        settings_lock_sha256=setup["settings_lock_sha256"], tokenizer={
                            name: props.get(name) for name in ("bos_token", "eos_token", "chat_template_caps")})
        template = props["chat_template"]
    else:
        from huggingface_hub import snapshot_download
        print("Downloading/verifying the official checkpoint...", flush=True)
        model_dir = Path(snapshot_download(MODEL_ID, revision=REVISION, local_dir=check["hf_dir"],
                                         allow_patterns=[*MODEL_FILES, "*.safetensors"], max_workers=2))
        for name, digest in check["metadata_files"].items():
            if sha256_file(model_dir/name) != digest:
                raise ValueError("Official tokenizer/config metadata changed")
        files = sorted(model_dir.glob("*.safetensors"))
        if not files:
            raise ValueError("Official weight shards are missing")
        print("Hashing the official weight shards...", flush=True)
        weights = {p.name: sha256_file(p) for p in files}
        print("Loading the official checkpoint as FP16 on cuda:0...", flush=True)
        client = TransformersClient(model_dir, max_tokens=inference["max_tokens"])
        output.mkdir()
        template = client.tokenizer.get_chat_template()
        if stable_hash(template) != check["official_template_sha256"]:
            raise ValueError("The official tokenizer selected a different chat template")
        metadata.update(model_id=MODEL_ID, revision=REVISION, weights=weights,
                        attention_implementation="sdpa", generation_config=client.generation.to_dict(),
                        tokenizer={"bos_token": client.tokenizer.bos_token, "bos_token_id": client.tokenizer.bos_token_id,
                                   "eos_token": client.tokenizer.eos_token, "eos_token_id": client.tokenizer.eos_token_id},
                        package_versions={name: version(name) for name in
                                          ("torch", "transformers", "accelerate", "huggingface-hub", "safetensors")},
                        warmup=[])
        for _ in range(2):
            try:
                response = client.chat(copy.deepcopy(_WARMUP_MESSAGES), copy.deepcopy(_WARMUP_TOOLS))
                metadata["warmup"].append({"finish_reason": response["finish_reason"]})
            except ModelOutputError as exc:
                metadata["warmup"].append({"model_output_error": exc.kind})
    (output/"chat_template.jinja").write_text(template, encoding="utf-8")
    metadata["chat_template_sha256"] = stable_hash(template)
    write_json(output/"metadata.json", metadata)
    rows = []
    with (output/"trials.jsonl").open("x", encoding="utf-8") as stream:
        for index, env in enumerate(trials(), 1):
            trace = run_agent(env.task(), client, env, condition=env.condition, max_calls=8,
                              replay_rejected_output=False)
            row = {"protocol": PROTOCOL, "backend": args.backend, "base": env.base, "condition": env.condition,
                   **probe_score(env, trace), "trace": trace}
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")
            stream.flush()
            print(f"[{index}/6] {env.base} {env.condition} success={row['success']}", flush=True)
            if row["success"] is None:
                raise SystemExit("Infrastructure/internal error; partial trace retained")
            rows.append(row)
    write_json(output/"summary.json", summarize(rows))


def summarize(rows):
    result = []
    for condition in ("normal_restart", "error_restart"):
        group = [r for r in rows if r["condition"] == condition]
        result.append({
            "condition": condition, "n": len(group), "successes": sum(r["success"] for r in group),
            "accessory_correct_at_end": sum(r["accessory_correct_at_end"] for r in group),
            "accessory_write_episodes": sum(r["accessory_write_count"] > 0 for r in group),
            "first_bundle_saves_correct": sum(
                save["correct"] for r in group for save in r["bundle_first_saves"].values()),
            "format_rejections": sum(r["format_rejections"] for r in group),
        })
    return result


def compare(args):
    check = load_check(args.check_dir)
    stages = {}
    for backend in ("gguf", "hf"):
        metadata = json.loads((args.check_dir/backend/"metadata.json").read_text(encoding="utf-8"))
        if (metadata["protocol"] != PROTOCOL or metadata["runner_hash"] != check["runner_hash"]
                or metadata["hardware"] != check["hardware"] or metadata["input_plan"] != check["input_plan"]
                or metadata["loaded_dtype"] != "float16"):
            raise ValueError("Backend metadata differs from the prepared comparison")
        rows = read_runs(args.check_dir/backend/"trials.jsonl")
        if (len(rows) != 6 or {(r["base"],r["condition"]) for r in rows}
                != {(p["base"],p["condition"]) for p in check["input_plan"]}
                or any(r["success"] is None or r["backend"] != backend for r in rows)):
            raise ValueError("Missing, duplicate or invalid backend trials")
        stages[backend] = {(r["base"],r["condition"]):r for r in rows}
    pairs = []
    for key, gguf in stages["gguf"].items():
        hf = stages["hf"][key]
        if stable_hash(gguf["trace"]["initial_messages"]) != stable_hash(hf["trace"]["initial_messages"]):
            raise ValueError("The backends received different starting messages")
        if gguf["trace"]["initial_state_hash"] != hf["trace"]["initial_state_hash"]:
            raise ValueError("The backends started from different states")
        expected = next(p for p in check["input_plan"] if (p["base"],p["condition"]) == key)
        if (stable_hash(gguf["trace"]["initial_messages"]) != expected["messages_sha256"]
                or gguf["trace"]["initial_state_hash"] != expected["state_sha256"]):
            raise ValueError("Starting inputs differ from the prepared task")
        prompts = [r["trace"]["events"][0].get("raw_response",{}).get("input_audit",{}).get("rendered_prompt")
                   for r in (gguf, hf)]
        pairs.append({"base": key[0], "condition": key[1], "gguf_success": gguf["success"],
                      "hf_success": hf["success"], "gguf_accessory_correct": gguf["accessory_correct_at_end"],
                      "hf_accessory_correct": hf["accessory_correct_at_end"],
                      "initial_rendered_prompts_equal": prompts[0] == prompts[1] if all(p is not None for p in prompts) else None})
    result = {"protocol": PROTOCOL, "summaries": {b:summarize(list(group.values())) for b,group in stages.items()},
              "pairs": pairs, "interpretation": "Six development episodes per path. Runtime, conversion and parsing can all contribute."}
    write_json(args.check_dir/"comparison.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--baseline-log", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--hf-dir", type=Path, default=ROOT/"models/qwen3-4b-instruct-2507-hf")
    prep.add_argument("--dry-run", action="store_true")
    execution = sub.add_parser("run")
    execution.add_argument("--check-dir", type=Path, required=True)
    execution.add_argument("--backend", choices=("gguf", "hf"), required=True)
    execution.add_argument("--model-path", type=Path)
    execution.add_argument("--server-pid", type=int)
    execution.add_argument("--base-url", default="http://127.0.0.1:18080")
    analysis = sub.add_parser("compare")
    analysis.add_argument("--check-dir", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
