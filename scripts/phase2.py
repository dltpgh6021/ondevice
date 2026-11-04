"""Run a complete phase-2 development or held-out series on one loaded GGUF."""
import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import run_agent
from src.client import LlamaCppClient
from src.experiment import GpuMemoryMonitor, ProcessRamMonitor, sha256_file
from src.phase2 import CONDITIONS, HORIZONS, PROTOCOL, SIZES, PurchaseChain, trial_design
from src.run_preparation import prepare_run
from src.tasks import stable_hash


PROTOCOL_SOURCES = (
    "scripts/phase2.py", "src/phase2.py", "src/client.py", "src/agent.py",
    "src/run_preparation.py", "src/tasks.py", "src/experiment.py",
)


def fingerprint(root=ROOT):
    # Git may check out the same Python sources with LF or CRLF.
    return stable_hash({name: (root / name).read_text(encoding="utf-8") for name in PROTOCOL_SOURCES})


def read_runs(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def require_complete(rows, split, precision):
    expected = {(f"{split}-{i:03d}", h, c, k) for i, h, c, k in trial_design(split)}
    keys = {(r["base"], r["h"], r["condition"], r.get("rotation")) for r in rows}
    if keys != expected or len(rows) != len(expected):
        raise ValueError("Missing or duplicate trials")
    if any(r.get("protocol") != PROTOCOL for r in rows):
        raise ValueError("Unsupported experiment protocol")
    for field in ("model_sha256", "settings_lock_sha256", "protocol_hash", "source_sha256"):
        if len({r[field] for r in rows}) != 1:
            raise ValueError("Mixed run metadata: " + field)
    if any(r["split"] != split or r["precision"] != precision or type(r["success"]) is not bool for r in rows):
        raise ValueError("Wrong series or invalid infrastructure/internal trial")

    for row in rows:
        env = PurchaseChain(split, int(row["base"].split("-")[1]), row["h"], row["condition"], row["rotation"])
        messages, _ = env.prepare_condition(env.task(), row["condition"])
        trace = row["trace"]
        if (stable_hash(trace["initial_messages"]) != stable_hash(messages)
                or stable_hash(trace["initial_state"]) != env.state_hash()):
            raise ValueError(f"Incorrect starting input or state: {row['base']}, order {row['rotation']}")


def model_family(manifest, precision, digest):
    source = manifest["source_model"]
    if source["quantization"] not in {"F16", "FP16"}:
        raise ValueError("Phase 2 needs a registered FP16 source")
    allowed = {"FP16": {"F16", "FP16"}, "Q8": {"Q8_0"}, "Q4": {"Q4_K_M"}}[precision]
    variants = [source] if precision == "FP16" else manifest["variants"]
    if not any(v["quantization"] in allowed and v["sha256"] == digest for v in variants):
        raise ValueError(f"GGUF hash does not match the {precision} variant in the quantization manifest")
    return source["sha256"]


def position_summary(rows):
    """Repeated orders are paired within scenario; absent writes stay in the denominator."""
    split = rows[0]["split"]
    groups = {}
    for r in rows:
        if r["h"] == 1:
            groups.setdefault((r["base"], r["condition"]), {})[r["rotation"]] = r

    result = {}
    for condition in CONDITIONS:
        slots = {i: Counter() for i in (1, 2, 3)}
        next_slot_saves = next_slot_opportunities = 0
        complete_choices = same_offer = always_first = same_outcome = triads = 0
        for (base, c), orders in groups.items():
            if c != condition or set(orders) != {0, 1, 2}:
                continue
            triads += 1
            choices, positions = [], []
            for k, row in orders.items():
                env = PurchaseChain(split, int(base.split("-")[1]), 1, c, k)
                line = env.audit_line
                ids = [o["offer"] for o in env.menu(line)["offers"]]
                gold = env.best(line)
                counts = slots[ids.index(gold) + 1]
                counts["trials"] += 1
                counts["final_successes"] += row["success"]
                event = next((e for e in row["trace"]["events"]
                              if e.get("tool_result", {}).get("updated_line") == line), None)
                if event is None:
                    counts["no_save"] += 1
                    continue
                choice = event["action"]["arguments"]["offer"]
                if c != "free":
                    initial = row["trace"]["initial_state"]
                    saved = initial[0] if 0 in initial else initial["0"]
                    saved_slot = ids.index(saved)
                    if saved_slot < len(ids) - 1:
                        next_slot_opportunities += 1
                        next_slot_saves += choice == ids[saved_slot + 1]
                choices.append(choice)
                positions.append(ids.index(choice) + 1)
                counts["first_save_correct"] += choice == gold
                counts["first_save_in_first_slot"] += choice == ids[0]
            same_outcome += len({r["success"] for r in orders.values()}) == 1
            if len(choices) == 3:
                complete_choices += 1
                same_offer += len(set(choices)) == 1
                always_first += positions == [1, 1, 1]
        result[condition] = {"scenario_triples": triads, "triples_with_three_saves": complete_choices,
                             "same_first_saved_offer_all_orders": same_offer,
                             "first_slot_saved_all_orders": always_first,
                             "final_success_flips_across_orders": triads - same_outcome,
                             "first_accessory_save_in_next_slot": next_slot_saves,
                             "accessory_saves_with_a_following_slot": next_slot_opportunities,
                             "by_gold_slot": {k: {name:v[name] for name in (
                                 "trials", "final_successes", "no_save", "first_save_correct",
                                 "first_save_in_first_slot")} for k,v in slots.items()}}
    return result


def dev_rule_scores():
    """Score cheap rules on full free episodes; choose offers from visible menu fields."""
    rules = {
        "first_slot": lambda menu: menu["offers"][0],
        "cheapest": lambda menu: min(menu["offers"], key=lambda o: o["price"]),
        "highest_rating": lambda menu: max(menu["offers"], key=lambda o: o["rating"]),
        "middle_rating": lambda menu: sorted(menu["offers"], key=lambda o: o["rating"])[1],
        "request_rule": lambda menu: max((o for o in menu["offers"] if o["price"] <= menu["budget"]),
                                         key=lambda o: o["rating"]),
    }
    successes = {}
    for name, choose in rules.items():
        scores = Counter()
        for h in HORIZONS:
            for i in range(SIZES["dev"]):
                env = PurchaseChain("dev", i, h, "free")
                for line in range(1, h + 1):
                    offer = choose(env.menu(line))
                    env.execute("save_offer", {"line": line, "offer": offer["offer"]})
                scores[h] += int(env.check_success())
        successes[name] = dict(scores)
    return successes


def pilot_assessment(rows):
    require_complete(rows, "dev", "FP16")
    primary = [r for r in rows if r["rotation"] == 0]
    wins = {f"{c}/h{h}": sum(r["success"] for r in primary if r["h"] == h and r["condition"] == c)
            for h in HORIZONS for c in CONDITIONS}
    audit = position_summary(rows)
    issues = []
    for h in HORIZONS:
        for condition in CONDITIONS:
            observed = wins[f"{condition}/h{h}"]
            threshold = 4 if condition == "error_restart" else 8
            if observed < threshold:
                issues.append(f"h{h} {condition}: {observed}/12; need {threshold}/12")
    for slot, counts in audit["free"]["by_gold_slot"].items():
        if counts["final_successes"] < 8:
            issues.append(f"h1 free, gold slot {slot}: {counts['final_successes']}/12; need 8/12")
    ceiling_review = all(wins[f"free/h{h}"] >= 11 for h in (3, 5))
    return {"n_base_scenarios": SIZES["dev"], "primary_success_counts": wins,
            "dev_free_rule_success_counts": dev_rule_scores(),
            "minimum_performance_issues": issues,
            "review_for_ceiling_or_external_validation": ceiling_review,
            "position_audit": audit}


def pilot_gate(rows):
    issues = pilot_assessment(rows)["minimum_performance_issues"]
    if issues:
        raise ValueError("FP16 pilot gate: " + ", ".join(issues))


def score(env, trace):
    events = trace["events"]
    valid = not (trace["infrastructure_error"] or trace["internal_error"])
    reached = [e["step"] for e in events if e.get("success_after_action")]
    errors = [e["step"] for e in events if e.get("semantic_error") or e.get("tool_error")
              or e.get("error") in {"model_output_error","invalid_action","invalid_arguments"}]
    success = env.check_success()
    if trace["finished"] and not success:
        errors.append(events[-1]["step"])
    semantic = [e["step"] for e in events if e.get("semantic_error")]
    if trace["finished"] and not success:
        semantic.append(events[-1]["step"])
    writes = [e for e in events if e.get("tool_result", {}).get("updated_line") == 0]
    corrected = [e["step"] for e in writes if not e.get("semantic_error")]
    first_saves = {}
    for event in events:
        line = event.get("tool_result", {}).get("updated_line")
        if line is not None and line >= 1:
            first_saves.setdefault(line, {"turn":event["step"], "correct":not bool(event.get("semantic_error"))})
    return {"accessory_write_count":len(writes), "bundle_first_saves":first_saves,
            "first_repair_turn":min(corrected, default=None) if env.condition == "error_restart" else None,
            "semantic_error_turns": sorted(set(semantic)), "success": success if valid else None, "error_turns": sorted(set(errors)),
            "first_goal_turn": min(reached, default=None), "goal_lost_at_end": bool(reached) and not success,
            "injected_error_resolved": (env.saved.get(0) == env.best(0)
                                        if valid and env.condition == "error_restart" else None),
            "executed_calls": sum(bool(e.get("tool_executed")) for e in events),
            "format_rejections": sum(e.get("error") == "model_output_error" for e in events)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=SIZES, required=True)
    ap.add_argument("--precision", choices=["FP16","Q8","Q4"], required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, help="Default: manifest.json beside --model-path")
    ap.add_argument("--base-url", default="http://127.0.0.1:18080")
    ap.add_argument("--api-model", default="local")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--settings-lock", type=Path, default=Path("results/phase2/settings.json"))
    ap.add_argument("--dev-log", type=Path)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--server-pid", type=int)
    ap.add_argument("--gpu-index", type=int)
    a = ap.parse_args()
    protocol_hash = fingerprint()
    if a.output.exists() or not a.model_path.is_file():
        ap.error("Choose a new output path and an existing GGUF")
    digest = sha256_file(a.model_path)
    manifest_path = a.manifest or a.model_path.parent / "manifest.json"
    family = model_family(json.loads(manifest_path.read_text(encoding="utf-8")), a.precision, digest)
    baseline = None
    if a.split == "dev" and a.precision != "FP16":
        ap.error("Development pilot requires FP16")
    if a.split == "main":
        if not a.dev_log:
            ap.error("Held-out runs require --dev-log")
        dev = read_runs(a.dev_log)
        pilot_gate(dev)
        baseline = dev[0]
        if baseline["protocol_hash"] != protocol_hash or baseline["model_sha256"] != family:
            ap.error("Pilot protocol or FP16 model changed; repeat pilot")
    client = LlamaCppClient(a.base_url, a.api_model, max_tokens=a.max_tokens, accept_plain_call=True)
    props = client.get_server_properties()
    served = props.get("model_path")
    if not served or not Path(served).is_file():
        ap.error("Cannot verify the served GGUF locally. Start llama-server with an absolute --model path.")
    if not Path(served).samefile(a.model_path) and sha256_file(served) != digest:
        ap.error("Loaded model file differs from supplied file")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    setup = prepare_run(client, expected_context_size=8192, warmup_requests=2, settings_lock_path=a.settings_lock,
                        report_path=a.output.with_suffix(".preflight.json"),
                        hardware={"hostname":platform.node(),"platform":platform.platform(),"processor":platform.processor(),"machine":platform.machine()})
    if baseline and baseline["settings_lock_sha256"] != setup["settings_lock_sha256"]:
        ap.error("Pilot settings differ")
    wins = Counter()
    with a.output.open("x", encoding="utf-8") as out:
        for i, h, c, rotation in trial_design(a.split):
            env = PurchaseChain(a.split,i,h,c,rotation)
            ram = ProcessRamMonitor(a.server_pid)
            gpu = GpuMemoryMonitor(a.gpu_index)
            ram.start()
            gpu.start()
            try:
                trace = run_agent(env.task(),client,env,condition=c,max_calls=h+5,replay_rejected_output=False)
            finally:
                peak_ram, peak_gpu = ram.stop(), gpu.stop()
            row = {"protocol":PROTOCOL,"protocol_hash":protocol_hash,"precision":a.precision,"split":a.split,
                   "rotation":rotation,"base":env.base,"h":h,"condition":c,"max_response_turns":h+5,
                   "settings_lock_sha256":setup["settings_lock_sha256"], "model_sha256":digest, "source_sha256":family,
                   "file_bytes":a.model_path.stat().st_size,"peak_server_ram_mb":peak_ram,"peak_device_vram_mb":peak_gpu,
                   **score(env,trace),"trace":trace}
            out.write(json.dumps(row,ensure_ascii=False)+"\n")
            out.flush()
            if rotation == 0:
                wins[h,c] += int(row["success"] is True)
            if row["success"] is None:
                raise ValueError("Infrastructure/internal failure. Incomplete series retained; use a new output after fixing.")
    for (h,c),n in wins.items():
        print(f"h{h} {c}: {n}/{SIZES[a.split]}")
    if a.split == "dev":
        rows = read_runs(a.output)
        diagnostics = pilot_assessment(rows)
        print("FP16 pilot quality review:")
        print(json.dumps(diagnostics, indent=2))
        if diagnostics["minimum_performance_issues"]:
            raise SystemExit("FP16 pilot did not pass; inspect the report above")
        print("FP16 development gate passed")


if __name__ == "__main__":
    main()
