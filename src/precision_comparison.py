"""Paired comparisons between precisions on the same tasks (standard library only).

Each (condition, horizon) cell pairs a model's runs with the reference model's
runs on identical task IDs. Differences are model minus reference. Runs with
infrastructure or internal failures are excluded on both sides of a pair.
No multiple-comparison correction is applied; report it as exploratory.
"""

from __future__ import annotations

import random
from collections import defaultdict
from math import comb
from typing import Any, Iterable

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 6)


def bootstrap_ci(values: list[float], *, resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED) -> list[float] | None:
    """Percentile 95% interval of the mean, resampling paired tasks."""
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(resamples))
    return [round(means[int(0.025 * resamples)], 6), round(means[int(0.975 * resamples) - 1], 6)]


def _label(record: dict[str, Any]) -> str:
    return str(record.get("metadata", {}).get("model_label") or record.get("score", {}).get("model_label") or "unknown")


def _outcomes(records: Iterable[dict[str, Any]]) -> dict[tuple, dict[str, dict[str, bool]]]:
    cells: dict[tuple, dict[str, dict[str, bool]]] = defaultdict(lambda: defaultdict(dict))
    for record in records:
        score, trace = record.get("score", {}), record.get("trace", {})
        if score.get("valid_for_behavior_metrics") is False or not isinstance(score.get("strict_success"), bool):
            continue
        condition = trace.get("condition") or score.get("condition")
        condition = "error_restart" if condition == "recovery" else condition
        protocol = trace.get("protocol_version") or score.get("protocol_version", "legacy-v1")
        task_id = record.get("task", {}).get("id") or score.get("task_id")
        key = (protocol, condition, score.get("horizon"))
        label = _label(record)
        if task_id in cells[key][label]:
            raise ValueError(f"duplicate {condition} record for {label}/{task_id}; separate repeated runs first")
        cells[key][label][task_id] = score["strict_success"]
    return cells


def default_reference_label(labels: Iterable[str]) -> str | None:
    candidates = sorted({label for label in labels if "fp16" in label.lower()})
    return candidates[0] if len(candidates) == 1 else None


def compare_precisions(records: list[dict[str, Any]], reference_label: str) -> dict[str, Any]:
    cells = _outcomes(records)
    success_rows = []
    for (protocol, condition, horizon), by_label in sorted(cells.items(), key=lambda item: tuple(map(str, item[0]))):
        reference = by_label.get(reference_label)
        if not reference:
            continue
        for label, outcomes in sorted(by_label.items()):
            if label == reference_label:
                continue
            common = sorted(set(reference) & set(outcomes))
            if not common:
                continue
            differences = [int(outcomes[task]) - int(reference[task]) for task in common]
            only_reference = sum(reference[task] and not outcomes[task] for task in common)
            only_model = sum(outcomes[task] and not reference[task] for task in common)
            success_rows.append({
                "protocol_version": protocol, "condition": condition, "horizon": horizon,
                "reference_label": reference_label, "model_label": label, "n_paired": len(common),
                "reference_success_rate": round(sum(reference[task] for task in common) / len(common), 4),
                "model_success_rate": round(sum(outcomes[task] for task in common) / len(common), 4),
                "success_difference": round(sum(differences) / len(differences), 4),
                "difference_ci95": bootstrap_ci(differences),
                "only_reference_succeeded": only_reference, "only_model_succeeded": only_model,
                "mcnemar_exact_p": mcnemar_exact_p(only_reference, only_model),
            })

    # Precision x condition interaction: does the recovery penalty grow with quantization?
    penalty_rows = []
    by_horizon: dict[tuple, dict[str, dict[str, dict[str, bool]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for (protocol, condition, horizon), by_label in cells.items():
        if condition not in {"normal_restart", "error_restart"}:
            continue
        for label, outcomes in by_label.items():
            for task, success in outcomes.items():
                by_horizon[(protocol, horizon)][label][task][condition] = success

    def penalties(pairs: dict[str, dict[str, bool]]) -> dict[str, int]:
        return {task: int(item["normal_restart"]) - int(item["error_restart"])
                for task, item in pairs.items() if len(item) == 2}

    for (protocol, horizon), by_label in sorted(by_horizon.items(), key=lambda item: tuple(map(str, item[0]))):
        if reference_label not in by_label:
            continue
        reference = penalties(by_label[reference_label])
        for label, pairs in sorted(by_label.items()):
            if label == reference_label:
                continue
            model = penalties(pairs)
            common = sorted(set(reference) & set(model))
            if not common:
                continue
            differences = [model[task] - reference[task] for task in common]
            penalty_rows.append({
                "protocol_version": protocol, "horizon": horizon, "reference_label": reference_label,
                "model_label": label, "n_paired": len(common),
                "reference_recovery_penalty": round(sum(reference[task] for task in common) / len(common), 4),
                "model_recovery_penalty": round(sum(model[task] for task in common) / len(common), 4),
                "penalty_difference": round(sum(differences) / len(differences), 4),
                "difference_ci95": bootstrap_ci(differences),
            })
    return {
        "reference_label": reference_label,
        "success_differences": success_rows,
        "recovery_penalty_differences": penalty_rows,
        "method": {"pairing": "identical task IDs within condition and horizon",
                   "ci": f"percentile bootstrap over paired tasks, {BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}",
                   "test": "two-sided exact McNemar on discordant pairs",
                   "limitations": ["No multiple-comparison correction.",
                                   "Each cell has at most 20 paired tasks; treat results as exploratory."]},
    }
