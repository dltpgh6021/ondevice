"""Compare restart instructions on the same FP16 development cases."""

import argparse
import json
import platform
import shlex
import sys
from itertools import permutations
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_phase2 import launch_settings, probe_score
from scripts.phase2 import fingerprint, model_family, read_runs, require_complete
from src.agent import run_agent
from src.client import LlamaCppClient
from src.experiment import sha256_file
from src.phase2 import PurchaseChain, SIZES
from src.run_preparation import prepare_run
from src.tasks import stable_hash


PROTOCOL = "phase2-restart-prompt-check-v1"
PROMPTS = ("original", "next_offer", "review_all")
CONDITIONS = ("normal_restart", "error_restart")
NEXT_OFFER = (
    "When you save a bundle line, the tool result shows the next bundle line's "
    "budget and offers if another bundle line remains."
)


def make_trial(scenario, horizon, condition, prompt):
    env = PurchaseChain("dev", scenario, horizon, condition)
    if prompt == "next_offer":
        anchor = f"Complete them in order, including line {horizon}."
        if env.query.count(anchor) != 1:
            raise ValueError("Original prompt changed; review the insertion point")
        env.query = env.query.replace(anchor, f"{anchor} {NEXT_OFFER}")
    elif prompt == "review_all":
        env.query = (
            f"Complete the order with one saved offer for each required line: {list(range(horizon + 1))}. "
            "On every required line, keep or select the highest-rated offer whose price is at or below that line's "
            "budget. Check saved selections against the same rule. Keep a correct saved selection and call "
            "save_offer to replace a selection that does not satisfy the rule. Use save_offer to select an offer "
            "on a line without one. "
            f"Complete the bundle lines {list(range(1, horizon + 1))} in order. "
            "Line 0 is an independent accessory. You may update it before or after the bundle lines. "
            f"{NEXT_OFFER} "
            "Changing an earlier bundle selection clears dependent later selections, which must be made again. "
            "An unsaved line still needs a selection. Before finishing, check that the saved offer on every "
            "required line satisfies its selection rule. Use the tool to apply any changes, then finish with a "
            "short final answer when all required selections are correct."
        )
    return env, {"base": env.base, "horizon": horizon, "condition": condition, "prompt": prompt}


def trial_plan(horizon):
    orders = tuple(permutations(PROMPTS))
    for scenario in range(SIZES["dev"]):
        for offset, prompt in enumerate(orders[scenario % len(orders)]):
            reverse = (scenario // len(orders) + scenario + offset) % 2
            condition_order = CONDITIONS[::-1] if reverse else CONDITIONS
            for condition in condition_order:
                yield make_trial(scenario, horizon, condition, prompt)


def experiment_summary(rows, horizon):
    cells = []
    paired = []
    for condition in CONDITIONS:
        by_prompt = {
            prompt: {r["base"]: r for r in rows if r["condition"] == condition and r["prompt"] == prompt}
            for prompt in PROMPTS
        }
        for prompt, group in by_prompt.items():
            outcomes = list(group.values())
            cells.append({
                "condition": condition, "prompt": prompt, "n": len(outcomes),
                "successes": sum(r["success"] for r in outcomes),
                "bundle_all_reached": sum(len(r["bundle_first_saves"]) == horizon for r in outcomes),
                "direct_next_bundle_saves": sum(r["direct_next_bundle_saves"] for r in outcomes),
                "possible_bundle_transitions": (horizon - 1) * len(outcomes),
                "first_bundle_saves_correct": sum(
                    call["correct"] for r in outcomes for call in r["bundle_first_saves"].values()
                ),
                "inspects": sum(r["inspects"] for r in outcomes),
                "accessory_correct_at_end": sum(r["accessory_correct_at_end"] for r in outcomes),
                "accessory_write_episodes": sum(r["accessory_write_count"] > 0 for r in outcomes),
                "accessory_ever_damaged_episodes": (
                    sum(r["accessory_ever_damaged"] for r in outcomes) if condition == "normal_restart" else None
                ),
                "format_rejections": sum(r["format_rejections"] for r in outcomes),
                "budget_exhausted": sum(r["trace"]["termination_reason"] == "budget_exhausted" for r in outcomes),
            })
        control = by_prompt["original"]
        for prompt in PROMPTS[1:]:
            for metric in ("success", "accessory_correct_at_end"):
                pairs = [(control[base][metric], row[metric]) for base, row in by_prompt[prompt].items()]
                paired.append({
                    "condition": condition, "prompt": prompt, "metric": metric, "n_pairs": len(pairs),
                    "failed_to_passed": sum(not a and b for a, b in pairs),
                    "passed_to_failed": sum(a and not b for a, b in pairs),
                })
    return {"protocol": PROTOCOL, "horizon": horizon, "cells": cells, "paired_vs_original": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, choices=(3, 5), default=3)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    baseline = read_runs(args.baseline_log)
    require_complete(baseline, "dev", "FP16")
    primary_hash = fingerprint()
    if baseline[0]["protocol_hash"] != primary_hash:
        parser.error("Pilot protocol differs from the current code")
    plan = list(trial_plan(args.horizon))
    print(json.dumps({"protocol": PROTOCOL, "horizon": args.horizon, "trials": len(plan),
                      "prompts": PROMPTS, "conditions": CONDITIONS}, indent=2), flush=True)
    if args.dry_run:
        return
    if not args.model_path or not args.server_pid or not args.output_dir or args.output_dir.exists():
        parser.error("Provide the FP16 model, live server PID and a new output directory")

    print("Verifying the FP16 model and server settings...", flush=True)
    digest = sha256_file(args.model_path)
    manifest = json.loads((args.model_path.parent / "manifest.json").read_text(encoding="utf-8"))
    source_hash = model_family(manifest, "FP16", digest)
    if digest != baseline[0]["model_sha256"] or source_hash != baseline[0]["source_sha256"]:
        parser.error("Use the exact model from the baseline pilot")
    lock_path = args.baseline_log.parent / "settings.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if stable_hash(lock["configuration"]) != lock["configuration_sha256"] or (
        lock["configuration_sha256"] != baseline[0]["settings_lock_sha256"]
    ):
        parser.error("Baseline settings lock differs from the pilot log")
    inference = lock["configuration"]["inference_settings"]
    client = LlamaCppClient(args.base_url, "local", temperature=inference["temperature"], seed=inference["seed"],
                            max_tokens=inference["max_tokens"], timeout_seconds=inference["timeout_seconds"],
                            cache_prompt=inference["cache_prompt"], accept_plain_call=True)
    served = Path(client.get_server_properties().get("model_path") or "")
    if not served.is_file() or (not served.samefile(args.model_path) and sha256_file(served) != digest):
        parser.error("The server is not serving the supplied FP16 model")
    import psutil
    argv = psutil.Process(args.server_pid).cmdline()
    old_argv = shlex.split((args.baseline_log.parent / "server_command.txt").read_text(encoding="utf-8"))
    server_options = launch_settings(argv)
    address = urlparse(args.base_url)
    if server_options != launch_settings(old_argv) or (
        address.hostname, address.port or 80
    ) != (server_options["--host"], int(server_options["--port"])):
        parser.error("Restore the baseline server options and use its current PID and endpoint")

    output = args.output_dir
    output.mkdir(parents=True)
    setup = prepare_run(client, expected_context_size=8192, warmup_requests=2, settings_lock_path=lock_path,
                        report_path=output / "preflight.json",
                        hardware={"hostname": platform.node(), "platform": platform.platform(),
                                  "processor": platform.processor(), "machine": platform.machine()})
    metadata = {
        "protocol": PROTOCOL, "primary_protocol_hash": primary_hash, "runner_sha256": sha256_file(Path(__file__)),
        "baseline_log_sha256": sha256_file(args.baseline_log), "model_sha256": digest, "server_argv": argv,
        "settings_lock_sha256": setup["settings_lock_sha256"], "response_cap": args.horizon + 5,
        "trial_plan": [info for _, info in plan],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "settings.json").write_text(lock_path.read_text(encoding="utf-8"), encoding="utf-8")
    rows = []
    with (output / "trials.jsonl").open("x", encoding="utf-8") as stream:
        for index, (env, info) in enumerate(plan, 1):
            trace = run_agent(env.task(), client, env, condition=env.condition, max_calls=env.h + 5,
                              replay_rejected_output=False)
            result = probe_score(env, trace)
            turn_by_line = {line: action["turn"] for line, action in result["bundle_first_saves"].items()}
            transitions = sum(
                line in turn_by_line and turn_by_line.get(line + 1) == turn_by_line[line] + 1
                for line in range(1, env.h)
            )
            row = {"protocol": PROTOCOL, **info, **result, "direct_next_bundle_saves": transitions,
                   "inspects": sum(e.get("tool_executed", False) and
                                   e.get("action", {}).get("name") == "inspect_order"
                                   for e in trace["events"]), "trace": trace}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{index}/{len(plan)}] {env.base} {env.condition} {info['prompt']} success={row['success']}",
                  flush=True)
            if row["success"] is None:
                raise SystemExit("Infrastructure/internal error; partial trace retained")
            rows.append(row)
    summary = experiment_summary(rows, args.horizon)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
