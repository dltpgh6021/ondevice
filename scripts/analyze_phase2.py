"""Summarize complete matched phase-2 runs. Differences and intervals are in percentage points."""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase2 import read_runs, require_complete, position_summary, pilot_assessment
from src.phase2 import CONDITIONS, HORIZONS, SIZES
from src.precision_comparison import bootstrap_ci


def effect_summary(label, metric, values):
    interval = bootstrap_ci(values) if len(set(values)) > 1 else None
    return {"precision_vs_fp16": label, "metric": metric, "mean_pp": mean(values),
            "paired_95pct_interval_pp": interval, "n_base_scenarios": len(values)}


def analyze(files):
    models = {}
    position_audits = {}
    locks = set()
    hashes = set()
    for path in files:
        rows = read_runs(path)
        if not rows:
            raise ValueError("Empty series")
        precision = rows[0]["precision"]
        require_complete(rows, "main", precision)
        if precision in models or rows[0]["model_sha256"] in hashes:
            raise ValueError("Duplicate model series or weights")
        hashes.add(rows[0]["model_sha256"])
        locks.add((rows[0]["protocol_hash"], rows[0]["settings_lock_sha256"], rows[0]["source_sha256"]))
        position_audits[precision] = position_summary(rows)
        models[precision] = {(r["base"],r["h"],r["condition"]):r for r in rows if r["rotation"] == 0}
    if set(models) != {"FP16","Q8","Q4"} or len(locks)!=1:
        raise ValueError("Need three precision series with matching configuration")
    cells, comparisons = [], []
    for precision, model in models.items():
        by_horizon = {}
        for h in HORIZONS:
            for c in CONDITIONS:
                sample = [r for (_,length,cond),r in model.items() if length==h and cond==c]
                saves = [item for r in sample for item in r["bundle_first_saves"].values()]
                responses = sum(sum(bool(e.get("response_received")) for e in r["trace"]["events"]) for r in sample)
                rejections = sum(r["format_rejections"] for r in sample)
                rejection_pairs = [(prev.get("error") == "model_output_error", cur.get("error") == "model_output_error")
                                   for r in sample for prev, cur in zip(r["trace"]["events"], r["trace"]["events"][1:])
                                   if prev.get("response_received") and cur.get("response_received")]
                followups = sum(prev for prev, _ in rejection_pairs)
                repeated = sum(prev and cur for prev, cur in rejection_pairs)
                cells.append({"precision":precision,"h":h,"condition":c,"n":len(sample),
                              "success_rate":mean(r["success"] for r in sample),
                              "semantic_error_episode_rate":mean(bool(r["semantic_error_turns"]) for r in sample),
                              "first_save_opportunities":len(saves),
                              "first_saves_correct":sum(item["correct"] for item in saves),
                              "unreached_bundle_lines":h * len(sample) - len(saves),
                              "accessory_rewrite_rate":mean(r["accessory_write_count"] > 0 for r in sample) if c!="free" else None,
                              "budget_exhausted_rate":mean(r["trace"]["termination_reason"]=="budget_exhausted" for r in sample),
                              "model_responses":responses, "format_rejections":rejections,
                              "pooled_format_rejection_rate":rejections / responses if responses else None,
                              "responses_after_rejection":followups,
                              "repeat_rejection_rate":repeated / followups if followups else None,
                              "mean_format_rejections":mean(r["format_rejections"] for r in sample),
                              "first_goal_reached_rate":mean(r["first_goal_turn"] is not None for r in sample),
                              "goal_lost_at_end_count":sum(r["goal_lost_at_end"] for r in sample),
                              "mean_ms":mean(r["trace"]["elapsed_ms"] for r in sample),
                              "error_resolved_rate":mean(r["injected_error_resolved"] for r in sample) if c=="error_restart" else None})
            if precision=="FP16":
                continue
            effects={"free_success_difference_pp":[], "additional_error_loss_difference_pp":[]}
            ref=models["FP16"]
            for i in range(SIZES["main"]):
                keys=[(f"main-{i:03d}",h,c) for c in CONDITIONS]
                f,n,e=keys
                effects["free_success_difference_pp"].append(100*(model[f]["success"]-ref[f]["success"]))
                effects["additional_error_loss_difference_pp"].append(100*((model[n]["success"]-model[e]["success"])-
                                                                           (ref[n]["success"]-ref[e]["success"])))
            by_horizon[h] = effects
            for metric,values in effects.items():
                comparisons.append({"h": h, **effect_summary(precision, metric, values)})
        if by_horizon:
            for metric in by_horizon[1]:
                change = [long - short for long, short in zip(by_horizon[5][metric], by_horizon[1][metric])]
                comparisons.append({"contrast": "h5_minus_h1", **effect_summary(precision, metric, change)})
    return {"cells":cells,"paired_effects":comparisons,"h1_position_audits":position_audits}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series", nargs="*", type=Path)
    parser.add_argument("--pilot", type=Path, help="Summarize FP16 dev results and position/ceiling diagnostics")
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.pilot:
        if args.series:
            parser.error("Pass either --pilot or three main series")
        report = pilot_assessment(read_runs(args.pilot))
    else:
        if len(args.series) != 3:
            parser.error("Supply three main series or --pilot DEV_LOG")
        report = analyze(args.series)
    with args.output.open("x",encoding="utf-8") as out:
        json.dump(report,out,indent=2)
