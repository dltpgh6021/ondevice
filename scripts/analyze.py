from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import load_records
from src.metrics import aggregate


def _storage_by_label(manifest_path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Create convenient result-label aliases from models/manifest.json."""
    if not manifest_path:
        return {}
    path = Path(manifest_path)
    if not path.is_file():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    items: list[dict[str, Any]] = []
    for key in ("source_model", "variants", "reference_variants"):
        value = manifest.get(key)
        if isinstance(value, dict):
            items.append(value)
        elif isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))

    storage: dict[str, dict[str, Any]] = {}
    for item in items:
        quantization = str(item.get("quantization", "")).upper()
        label = str(item.get("label", ""))
        aliases = {label} if label else set()
        is_reference = "official" in label.lower() or "reference" in label.lower()
        if quantization in {"F16", "F32", "BF16"}:
            aliases.update({"FP16", "self-FP16"})
        elif quantization.startswith("Q8"):
            aliases.add("reference-Q8" if is_reference else "self-Q8")
            aliases.add("Q8")
        elif quantization.startswith("Q4"):
            aliases.add("reference-Q4" if is_reference else "self-Q4")
            aliases.add("Q4")
        info = {
            "filename": item.get("filename"),
            "quantization": item.get("quantization"),
            "size_bytes": item.get("size_bytes"),
            "size_mb": item.get("size_mb"),
            "sha256": item.get("sha256"),
        }
        for alias in aliases:
            storage[alias] = info
    return storage


def _write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(summaries: list[dict[str, Any]], output_dir: str | Path) -> list[str]:
    """Write optional PNG figures. Missing matplotlib should not break analysis."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("그래프를 만들려면 matplotlib를 설치하세요: python -m pip install matplotlib", file=sys.stderr)
        return []

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    models = sorted({str(row.get("model_label", "unknown")) for row in summaries})
    files: list[str] = []

    def plot_metric(filename: str, title: str, metric: str, condition: str, ylabel: str) -> None:
        plotted = False
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for model in models:
            rows = [
                row
                for row in summaries
                if row.get("model_label") == model
                and row.get("condition") == condition
                and isinstance(row.get(metric), (int, float))
            ]
            rows.sort(key=lambda row: row.get("horizon", 0))
            if not rows:
                continue
            axis.plot(
                [row["horizon"] for row in rows],
                [row[metric] for row in rows],
                marker="o",
                label=model,
            )
            plotted = True
        if not plotted:
            plt.close(figure)
            return
        axis.set_title(title)
        axis.set_xlabel("Horizon")
        axis.set_ylabel(ylabel)
        axis.set_xticks(sorted({row["horizon"] for row in summaries}))
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = directory / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        files.append(str(path))

    plot_metric("success_rate_free_by_horizon.png", "Execution success rate by horizon", "success_rate", "free", "Success rate")
    plot_metric("natural_error_rate_free_by_horizon.png", "Natural error rate by horizon", "natural_error_rate", "free", "Natural error rate")
    plot_metric("recovery_rate_by_horizon.png", "Recovery rate by horizon", "recovery_rate", "recovery", "Recovery rate")
    plot_metric("elapsed_ms_free_by_horizon.png", "Execution time by horizon", "mean_elapsed_ms", "free", "Mean elapsed time (ms)")
    plot_metric("server_ram_free_by_horizon.png", "Peak server RAM by horizon", "mean_peak_server_ram_mb", "free", "Mean peak server RAM (MB)")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate JSONL experiment scores and optionally create CSV/plots")
    parser.add_argument("paths", nargs="+", help="One or more JSONL result files")
    parser.add_argument("--output", default=None, help="JSON summary output path")
    parser.add_argument("--manifest", default=str(ROOT / "models" / "manifest.json"), help="Model manifest used to add storage size")
    parser.add_argument("--csv-output", default=None, help="Optional flat summary CSV output path")
    parser.add_argument("--plot-dir", default=None, help="Optional directory for PNG figures")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for path in args.paths:
        records.extend(load_records(path))
    scores: list[dict[str, Any]] = []
    for record in records:
        score = dict(record["score"])
        score.setdefault("model_label", record.get("metadata", {}).get("model_label", "unknown"))
        scores.append(score)

    condition_summaries = aggregate(scores)
    storage = _storage_by_label(args.manifest)
    for summary in condition_summaries:
        info = storage.get(str(summary.get("model_label")))
        summary["model_storage_mb"] = info.get("size_mb") if info else None
        summary["model_quantization"] = info.get("quantization") if info else None

    by_key = {(row["model_label"], row["horizon"], row["condition"]): row for row in condition_summaries}
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
    result = {"condition_summaries": condition_summaries, "recovery_comparison": recovery_comparison}
    if args.csv_output:
        _write_csv(condition_summaries, args.csv_output)
    if args.plot_dir:
        result["plots"] = _write_plots(condition_summaries, args.plot_dir)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
