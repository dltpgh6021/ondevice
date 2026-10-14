from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import load_records
from src.metrics import aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate JSONL experiment scores")
    parser.add_argument("paths", nargs="+", help="One or more JSONL result files")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    records = []
    for path in args.paths:
        records.extend(load_records(path))
    scores = []
    for record in records:
        score = dict(record["score"])
        score.setdefault("model_label", record.get("metadata", {}).get("model_label", "unknown"))
        scores.append(score)
    condition_summaries = aggregate(scores)
    by_key = {
        (row["model_label"], row["horizon"], row["condition"]): row
        for row in condition_summaries
    }
    recovery_comparison = []
    for (model_label, horizon, condition), normal_row in by_key.items():
        if condition != "free":
            continue
        recovery_row = by_key.get((model_label, horizon, "recovery"))
        if not recovery_row:
            continue
        normal_rate = normal_row.get("success_rate")
        recovery_rate = recovery_row.get("recovery_rate")
        recovery_comparison.append(
            {
                "model_label": model_label,
                "horizon": horizon,
                "normal_success_rate": normal_rate,
                "recovery_rate": recovery_rate,
                "recovery_gap": round(normal_rate - recovery_rate, 4)
                if normal_rate is not None and recovery_rate is not None
                else None,
                "mean_extra_calls": recovery_row.get("mean_extra_calls"),
            }
        )
    result = {
        "condition_summaries": condition_summaries,
        "recovery_comparison": recovery_comparison,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

