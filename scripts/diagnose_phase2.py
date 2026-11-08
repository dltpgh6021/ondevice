"""Run development-only choice and recovery probes against a complete FP16 pilot."""
import argparse
import copy
import json
import platform
import shlex
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.phase2 import fingerprint, model_family, read_runs, require_complete, score
from src.agent import run_agent
from src.client import LlamaCppClient
from src.experiment import sha256_file
from src.phase2 import PurchaseChain, SIZES
from src.run_preparation import prepare_run
from src.tasks import stable_hash

DIAGNOSTIC_PROTOCOL = "phase2-diagnostic-v1"
RESTART_MODES = ("history", "state_only", "history_review")
REVIEW = (
    "Before finishing, review EVERY required line, including already saved selections, "
    "against its budget and the highest-rating rule. Keep a saved selection if it satisfies "
    "the rule. Apply any needed change by calling save_offer. Describing a replacement "
    "in a final answer does not change the stored order. Complete the remaining lines "
    "and give a final answer only when the stored selections satisfy the request."
)
LAUNCH_FLAGS = ("--ctx-size", "--parallel", "--batch-size", "--ubatch-size",
                "--cache-type-k", "--cache-type-v", "--flash-attn", "--n-gpu-layers",
                "--reasoning", "--host", "--port")


class IsolatedChoice(PurchaseChain):
    """A frozen, actually observed menu; no parent state or oracle answer is exposed."""
    def __init__(self, scenario, menu):
        super().__init__("dev", scenario, 1, "free")
        self.frozen_menu = copy.deepcopy(menu)
        self.line = menu["line"]
        self.query = (
            f"Select exactly one offer for line {self.line}. This is an independent single-line order. "
            "Choose the highest-rated offer whose price is at or below this line's budget. "
            "Use save_offer to store your selection. When the stored selection satisfies the request, "
            "finish with a short final answer."
        )

    def menu(self, line, *, canonical=False):
        if line != self.line:
            raise ValueError("Only the target line is available")
        return copy.deepcopy(self.frozen_menu)

    def view(self):
        menu = self.menu(self.line)
        menu["saved_offer"] = next((dict(o) for o in menu["offers"]
                                    if o["offer"] == self.saved.get(self.line)), None)
        return {"required_lines": [self.line], "unsaved_lines": [] if self.line in self.saved else [self.line],
                "saved": dict(self.saved), "lines": [menu]}

    def execute(self, name, args):
        if name == "inspect_order" and not args:
            return {"ok": True, **self.view()}
        if (name != "save_offer" or set(args) != {"line", "offer"}
                or type(args["line"]) is not int or args["line"] != self.line
                or args["offer"] not in [o["offer"] for o in self.frozen_menu["offers"]]):
            return {"ok": False, "error": "Unknown tool, arguments or unavailable offer."}
        self.saved[self.line] = args["offer"]
        return {"ok": True, "updated_line": self.line, **self.view()}

    def check_success(self, ignored=None):
        return self.saved.get(self.line) == self.best(self.line)

    def task(self):
        return replace(super().task(), task_id=f"{self.base}-isolated-line{self.line}",
                       dataset_version=DIAGNOSTIC_PROTOCOL)


class RestartProbe(PurchaseChain):
    def __init__(self, scenario, condition, mode):
        if mode not in RESTART_MODES:
            raise ValueError("Unsupported diagnostic mode")
        super().__init__("dev", scenario, 1, condition)
        self.mode = mode

    def prepare_condition(self, task, condition):
        messages, _ = super().prepare_condition(task, condition)
        if self.mode == "state_only":
            # Preserve current state, offers/order and the original request;
            # replace the historical transcript with a current-state observation.
            messages = [messages[0], {"role": "user", "content": self.query +
                                     "\nCurrent order: " + json.dumps(self.view())}]
        elif self.mode == "history_review":
            messages.append({"role": "user", "content": REVIEW})
        return messages, None


def build_probes(baseline):
    require_complete(baseline, "dev", "FP16")
    if baseline[0]["protocol_hash"] != fingerprint():
        raise ValueError("Baseline protocol code differs; use the matching checkout")
    selected = {(r["base"], r["h"], r["condition"]): r for r in baseline if r["rotation"] == 0}
    probes = []
    for i in range(SIZES["dev"]):
        base = f"dev-{i:03d}"
        row = selected[base, 5, "free"]
        env = PurchaseChain("dev", i, 5, "free")
        env.prepare_condition(env.task(), "free")
        first = {}
        for event in row["trace"]["events"]:
            if not event.get("tool_executed"):
                continue
            action = event["action"]
            if action["name"] == "save_offer":
                line = action["arguments"]["line"]
                if line not in first:
                    first[line] = (env.menu(line), action["arguments"]["offer"] == env.best(line), event["step"])
            result = env.execute(action["name"], action["arguments"])
            if stable_hash(result) != stable_hash(event["tool_result"]):
                raise ValueError(f"Baseline tool replay mismatch: {base}")
        if set(first) != {1, 2, 3, 4, 5}:
            raise ValueError(f"Baseline did not reach all five lines: {base}; do not select only observed successes")
        for line in range(1, 6):
            menu, correct, turn = first[line]
            probes.append((IsolatedChoice(i, menu), {
                "probe_id": f"{base}/isolated/line{line}", "mode": "isolated_choice",
                "source_line": line, "source_horizon": 5, "baseline_turn": turn,
                "baseline_first_save_correct": correct, "menu_sha256": stable_hash(menu),
            }))
        modes = RESTART_MODES[i % 3:] + RESTART_MODES[:i % 3]
        conditions = ("normal_restart", "error_restart") if i % 2 == 0 else ("error_restart", "normal_restart")
        for mode in modes:
            for condition in conditions:
                probes.append((RestartProbe(i, condition, mode), {
                    "probe_id": f"{base}/{mode}/{condition}", "mode": mode,
                    "baseline_success": selected[base, 1, condition]["success"],
                }))
    return probes


def probe_score(env, trace):
    result = score(env, trace)
    valid = result["success"] is not None
    if isinstance(env, IsolatedChoice):
        first = next((e for e in trace["events"] if e.get("tool_result", {}).get("updated_line") == env.line), None)
        result.update(first_save_present=bool(first),
                      first_save_correct=(bool(first and not first.get("semantic_error")) if valid else None))
    else:
        result["accessory_correct_at_end"] = env.saved.get(0) == env.best(0) if valid else None
        result["accessory_ever_damaged"] = (any(e.get("semantic_error") == "wrong_offer" and
            e.get("tool_result", {}).get("updated_line") == 0 for e in trace["events"])
            if env.condition == "normal_restart" and valid else None)
    return result


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["mode"], row["condition"]].append(row)
    cells = []
    for (mode, condition), group in sorted(groups.items()):
        valid = [r for r in group if r["success"] is not None]
        cell = {"mode": mode, "condition": condition, "n": len(group), "n_valid": len(valid),
                "successes": sum(r["success"] for r in valid),
                "format_rejections": sum(r["format_rejections"] for r in group),
                "budget_exhausted": sum(r["trace"]["termination_reason"] == "budget_exhausted" for r in valid)}
        if mode == "isolated_choice":
            cell.update(first_saves_correct=sum(r["first_save_correct"] for r in valid),
                        no_save=sum(not r["first_save_present"] for r in valid),
                        baseline_first_saves_correct=sum(r["baseline_first_save_correct"] for r in valid),
                        changed_wrong_to_right=sum(not r["baseline_first_save_correct"] and r["first_save_correct"] for r in valid),
                        changed_right_to_wrong=sum(r["baseline_first_save_correct"] and not r["first_save_correct"] for r in valid))
        else:
            cell.update(accessory_write_episodes=sum(r["accessory_write_count"] > 0 for r in valid),
                        accessory_correct_at_end=sum(r["accessory_correct_at_end"] for r in valid),
                        success_changed_from_baseline=sum(r["success"] != r["baseline_success"] for r in valid))
            if condition == "normal_restart":
                cell["accessory_ever_damaged_episodes"] = sum(r["accessory_ever_damaged"] for r in valid)
            else:
                cell["injected_error_resolved"] = sum(r["injected_error_resolved"] for r in valid)
        cells.append(cell)
    return {"diagnostic_protocol": DIAGNOSTIC_PROTOCOL, "n_scenarios": SIZES["dev"], "cells": cells,
            "interpretation": "Development diagnostics only; 60 isolated menus are nested within 12 scenarios. "
                              "State-only changes history and input presentation together. No main-study pass/fail gate."}


def launch_settings(argv):
    result = {}
    for flag in LAUNCH_FLAGS:
        if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
            raise ValueError(f"Use an explicit single {flag} in the server command")
        result[flag] = argv[argv.index(flag) + 1]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-log", type=Path, required=True)
    ap.add_argument("--model-path", type=Path)
    ap.add_argument("--base-url", default="http://127.0.0.1:18080")
    ap.add_argument("--server-pid", type=int)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="Validate and list the 132 probes without a model/server")
    args = ap.parse_args()
    baseline = read_runs(args.baseline_log)
    probes = build_probes(baseline)
    print(json.dumps({"diagnostic_protocol": DIAGNOSTIC_PROTOCOL, "total": len(probes),
                      "isolated_choice": sum(m["mode"] == "isolated_choice" for _, m in probes),
                      "restart_probes": sum(m["mode"] != "isolated_choice" for _, m in probes)}, indent=2), flush=True)
    if args.dry_run:
        return
    if not args.model_path or not args.server_pid or not args.output_dir:
        ap.error("Execution requires --model-path, --server-pid and a new --output-dir")
    if args.output_dir.exists():
        ap.error("Use a new output directory; existing diagnostic results are never overwritten")
    print("Verifying FP16 model and server configuration...", flush=True)
    digest = sha256_file(args.model_path)
    manifest = json.loads((args.model_path.parent / "manifest.json").read_text(encoding="utf-8"))
    family = model_family(manifest, "FP16", digest)
    if digest != baseline[0]["model_sha256"] or family != baseline[0]["source_sha256"]:
        ap.error("Use the exact FP16 model from the baseline pilot")
    lock_path = args.baseline_log.parent / "settings.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock["configuration_sha256"] != baseline[0]["settings_lock_sha256"] or
            stable_hash(lock["configuration"]) != lock["configuration_sha256"]):
        ap.error("Baseline settings lock mismatch")
    inference = lock["configuration"]["inference_settings"]
    client = LlamaCppClient(args.base_url, "local", temperature=inference["temperature"], seed=inference["seed"],
                            max_tokens=inference["max_tokens"], timeout_seconds=inference["timeout_seconds"],
                            cache_prompt=inference["cache_prompt"], accept_plain_call=True)
    props = client.get_server_properties()
    served = Path(props.get("model_path", ""))
    if not served.is_file() or sha256_file(served) != digest:
        ap.error("The server is not serving the supplied FP16 file")
    import psutil
    current_argv = psutil.Process(args.server_pid).cmdline()
    old_argv = shlex.split((args.baseline_log.parent / "server_command.txt").read_text(encoding="utf-8"))
    if launch_settings(current_argv) != launch_settings(old_argv):
        ap.error("Server launch settings differ from baseline; restore batch/ubatch/context/KV/offload/reasoning/host/port")
    from urllib.parse import urlparse
    address = urlparse(args.base_url)
    current_settings = launch_settings(current_argv)
    if address.hostname != current_settings["--host"] or (address.port or 80) != int(current_settings["--port"]):
        ap.error("The supplied PID and API address do not identify the same configured endpoint")
    args.output_dir.mkdir(parents=True)
    setup = prepare_run(client, expected_context_size=8192, warmup_requests=2, settings_lock_path=lock_path,
                        report_path=args.output_dir / "preflight.json",
                        hardware={"hostname": platform.node(), "platform": platform.platform(),
                                  "processor": platform.processor(), "machine": platform.machine()})
    (args.output_dir / "settings.json").write_text(lock_path.read_text(encoding="utf-8"), encoding="utf-8")
    metadata = {"diagnostic_protocol": DIAGNOSTIC_PROTOCOL, "primary_protocol_hash": fingerprint(),
                "diagnostic_source_sha256": sha256_file(Path(__file__)),
                "baseline_log_sha256": sha256_file(args.baseline_log), "model_sha256": digest,
                "settings_lock_sha256": setup["settings_lock_sha256"], "server_argv": current_argv,
                "baseline_log": str(args.baseline_log.resolve()), "planned_probes": [m for _, m in probes]}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    results = []
    with (args.output_dir / "diagnostics.jsonl").open("x", encoding="utf-8") as stream:
        for number, (env, plan) in enumerate(probes, 1):
            trace = run_agent(env.task(), client, env, condition=env.condition, max_calls=6, replay_rejected_output=False)
            row = {"diagnostic_protocol": DIAGNOSTIC_PROTOCOL, "base": env.base, "condition": env.condition,
                   "max_response_turns": 6, **plan, **probe_score(env, trace), "trace": trace}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            results.append(row)
            print(f"[{number}/{len(probes)}] {plan['probe_id']} success={row['success']}", flush=True)
            if row["success"] is None:
                raise SystemExit("Infrastructure/internal error: partial diagnostics retained; use a new directory after fixing")
    summary = summarize(results)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Diagnostic run complete; this does not authorize the main experiment.")


if __name__ == "__main__":
    main()
