"""Paired comparisons between precisions on the same tasks (standard library only).

Three families of comparisons, all model minus reference:

1. ``success_differences``: per (condition, horizon) cell, success difference
   on identical task IDs, bootstrap CI, exact McNemar p and a Holm-adjusted p.
2. ``horizon_interactions``: the core amplification test. For each base
   scenario, the precision gap at a longer horizon minus the gap at a shorter
   horizon, ``(m_long - r_long) - (m_short - r_short)``. A negative mean means
   quantization hurts more when more steps remain. Tested with an exact
   sign-flip permutation test over scenarios.
3. ``recovery_penalty_differences``: precision x condition interaction for
   each error branch (silent ``error_restart`` and ``signaled_error_restart``).

Runs with infrastructure or internal failures are excluded on both sides of a
pair. All records must come from one hardware/settings/dataset configuration;
mixing machines raises unless explicitly allowed.
"""

from __future__ import annotations

import itertools
import random
import re
from collections import defaultdict
from math import comb
from typing import Any, Iterable

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
PERMUTATION_SEED = 0
PERMUTATION_RESAMPLES = 20000
ERROR_BRANCHES = ("error_restart", "signaled_error_restart")

# Fields that must be identical across every compared label. Model identity
# (label, path, hash) and the KV cache type are the experimental factors.
SHARED_CONFIGURATION_FIELDS = (
    "hostname", "platform", "processor", "machine", "seed", "inference_settings",
    "settings_lock_sha256", "server_configuration_sha256", "dataset_version", "dataset_sha256",
    "metrics_version", "agent_policy_version", "max_calls", "shared_extra_action_turns",
    "preparation_protocol", "warmup_settings",
)
VARIABLE_FIELDS = ("model_label", "model_path", "model_sha256", "kv_cache_type")


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 6)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = round(running, 6)
    return adjusted


def bootstrap_ci(values: list[float], *, resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED) -> list[float] | None:
    """Percentile 95% interval of the mean, resampling paired units."""
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(resamples))
    return [round(means[int(0.025 * resamples)], 6), round(means[int(0.975 * resamples) - 1], 6)]


def sign_flip_p(values: list[float], *, resamples: int = PERMUTATION_RESAMPLES,
                seed: int = PERMUTATION_SEED) -> float:
    """Two-sided paired permutation test of mean zero by flipping signs.

    Zero differences carry no information and are dropped. Integer differences
    (all paired success comparisons here) get the exact null distribution by
    dynamic programming over sign assignments; anything else falls back to a
    seeded Monte Carlo estimate with the +1 correction.
    """
    nonzero = [value for value in values if value != 0]
    if not nonzero:
        return 1.0
    if all(float(value).is_integer() for value in nonzero):
        integers = [int(value) for value in nonzero]
        observed = abs(sum(integers))
        distribution: dict[int, int] = {0: 1}
        for value in integers:
            following: dict[int, int] = defaultdict(int)
            for total, count in distribution.items():
                following[total + value] += count
                following[total - value] += count
            distribution = following
        extreme = sum(count for total, count in distribution.items() if abs(total) >= observed)
        return round(extreme / 2 ** len(integers), 6)
    observed = abs(sum(nonzero))
    rng = random.Random(seed)
    extreme = sum(abs(sum(value if rng.random() < 0.5 else -value for value in nonzero)) >= observed - 1e-12
                  for _ in range(resamples))
    return round((extreme + 1) / (resamples + 1), 6)


def _label(record: dict[str, Any]) -> str:
    return str(record.get("metadata", {}).get("model_label") or record.get("score", {}).get("model_label") or "unknown")


def _condition(record: dict[str, Any]) -> str | None:
    """Condition, qualified by the error kind for error branches (e.g. error_restart:wrong_option)."""
    condition = record.get("trace", {}).get("condition") or record.get("score", {}).get("condition")
    condition = "error_restart" if condition == "recovery" else condition
    if condition in ERROR_BRANCHES:
        kind = (record.get("trace", {}).get("error_kind") or record.get("score", {}).get("error_kind")
                or record.get("task", {}).get("error_kind"))
        if kind:
            condition = f"{condition}:{kind}"
    return condition


def scenario_id(record: dict[str, Any]) -> str:
    """Base scenario shared by every horizon and condition (e.g. shopping_001)."""
    task, score = record.get("task", {}), record.get("score", {})
    base = str(task.get("base_id") or score.get("base_id") or task.get("id") or score.get("task_id"))
    return re.sub(r"_h\d+$", "", base.removeprefix("recovery_"))


def check_shared_configuration(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Raise when compared labels ran under different machines/settings/data.

    Returns the per-label values of the variable factors for reporting.
    """
    seen: dict[str, dict[str, Any]] = {}
    factors: dict[str, dict[str, Any]] = {}
    conflicts: dict[str, set[str]] = defaultdict(set)
    for record in records:
        metadata = record.get("metadata", {})
        label = _label(record)
        factors.setdefault(label, {field: metadata.get(field) for field in VARIABLE_FIELDS})
        for field in SHARED_CONFIGURATION_FIELDS:
            value = metadata.get(field)
            if value is None and field in {"metrics_version", "agent_policy_version"}:
                value = record.get("trace", {}).get(field) or record.get("score", {}).get(field)
            if field in seen and seen[field]["value"] != value:
                conflicts[field].update({seen[field]["label"], label})
            seen.setdefault(field, {"value": value, "label": label})
    if conflicts:
        detail = "; ".join(f"{field}: {', '.join(sorted(labels))}" for field, labels in sorted(conflicts.items()))
        raise ValueError("Compared results come from different machines/settings/datasets (" + detail + "). "
                         "Run every precision on the same machine and settings lock, or pass "
                         "--allow-mixed-configuration to compare anyway (not recommended).")
    return factors


def _usable(record: dict[str, Any]) -> bool:
    score = record.get("score", {})
    return score.get("valid_for_behavior_metrics") is not False and isinstance(score.get("strict_success"), bool)


def _protocol(record: dict[str, Any]) -> str:
    return record.get("trace", {}).get("protocol_version") or record.get("score", {}).get("protocol_version", "legacy-v1")


def _outcomes(records: Iterable[dict[str, Any]]) -> dict[tuple, dict[str, dict[str, bool]]]:
    cells: dict[tuple, dict[str, dict[str, bool]]] = defaultdict(lambda: defaultdict(dict))
    for record in records:
        if not _usable(record):
            continue
        score = record["score"]
        condition = _condition(record)
        task_id = record.get("task", {}).get("id") or score.get("task_id")
        key = (_protocol(record), condition, score.get("horizon"))
        label = _label(record)
        if task_id in cells[key][label]:
            raise ValueError(f"duplicate {condition} record for {label}/{task_id}; separate repeated runs first")
        cells[key][label][task_id] = score["strict_success"]
    return cells


def _scenario_outcomes(records: Iterable[dict[str, Any]]) -> dict[tuple, dict[str, dict[int, dict[str, bool]]]]:
    """(protocol, condition) -> label -> horizon -> scenario -> success."""
    table: dict[tuple, dict[str, dict[int, dict[str, bool]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict)))
    for record in records:
        if not _usable(record):
            continue
        horizon = record["score"].get("horizon")
        if not isinstance(horizon, int) or isinstance(horizon, bool):
            continue
        table[(_protocol(record), _condition(record))][_label(record)][horizon][scenario_id(record)] = \
            record["score"]["strict_success"]
    return table


def default_reference_label(labels: Iterable[str]) -> str | None:
    candidates = sorted({label for label in labels if "fp16" in label.lower()})
    return candidates[0] if len(candidates) == 1 else None


def compare_precisions(records: list[dict[str, Any]], reference_label: str, *,
                       allow_mixed_configuration: bool = False) -> dict[str, Any]:
    records = list(records)
    factors = None if allow_mixed_configuration else check_shared_configuration(records)
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
    for row, adjusted in zip(success_rows, holm_adjust([row["mcnemar_exact_p"] for row in success_rows])):
        row["mcnemar_holm_p"] = adjusted

    # Precision x horizon: does the precision gap grow when more steps remain?
    interaction_rows = []
    for (protocol, condition), by_label in sorted(_scenario_outcomes(records).items(),
                                                  key=lambda item: tuple(map(str, item[0]))):
        reference = by_label.get(reference_label)
        if not reference:
            continue
        for label, by_horizon in sorted(by_label.items()):
            if label == reference_label:
                continue
            horizons = sorted(set(reference) & set(by_horizon))
            for short, long in itertools.combinations(horizons, 2):
                scenarios = sorted(set(reference[short]) & set(reference[long])
                                   & set(by_horizon[short]) & set(by_horizon[long]))
                if not scenarios:
                    continue
                gap_short = [int(by_horizon[short][s]) - int(reference[short][s]) for s in scenarios]
                gap_long = [int(by_horizon[long][s]) - int(reference[long][s]) for s in scenarios]
                interaction = [b - a for a, b in zip(gap_short, gap_long)]
                interaction_rows.append({
                    "protocol_version": protocol, "condition": condition,
                    "short_horizon": short, "long_horizon": long,
                    "reference_label": reference_label, "model_label": label, "n_scenarios": len(scenarios),
                    "gap_short": round(sum(gap_short) / len(scenarios), 4),
                    "gap_long": round(sum(gap_long) / len(scenarios), 4),
                    "interaction": round(sum(interaction) / len(scenarios), 4),
                    "interaction_ci95": bootstrap_ci(interaction),
                    "sign_flip_p": sign_flip_p(interaction),
                })
    for row, adjusted in zip(interaction_rows, holm_adjust([row["sign_flip_p"] for row in interaction_rows])):
        row["sign_flip_holm_p"] = adjusted

    # Precision x condition: does the recovery penalty grow with quantization?
    penalty_rows = []
    by_horizon: dict[tuple, dict[str, dict[str, dict[str, bool]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict)))
    for (protocol, condition, horizon), by_label in cells.items():
        if condition is None or condition.split(":")[0] not in {"normal_restart", *ERROR_BRANCHES}:
            continue
        for label, outcomes in by_label.items():
            for task, success in outcomes.items():
                by_horizon[(protocol, horizon)][label][task][condition] = success

    def penalties(pairs: dict[str, dict[str, bool]], branch: str) -> dict[str, int]:
        return {task: int(item["normal_restart"]) - int(item[branch])
                for task, item in pairs.items() if "normal_restart" in item and branch in item}

    for (protocol, horizon), by_label in sorted(by_horizon.items(), key=lambda item: tuple(map(str, item[0]))):
        if reference_label not in by_label:
            continue
        branches = sorted({condition for pairs in by_label.values() for item in pairs.values()
                           for condition in item if condition != "normal_restart"})
        for branch in branches:
            reference = penalties(by_label[reference_label], branch)
            for label, pairs in sorted(by_label.items()):
                if label == reference_label:
                    continue
                model = penalties(pairs, branch)
                common = sorted(set(reference) & set(model))
                if not common:
                    continue
                differences = [model[task] - reference[task] for task in common]
                penalty_rows.append({
                    "protocol_version": protocol, "horizon": horizon, "error_condition": branch,
                    "reference_label": reference_label, "model_label": label, "n_paired": len(common),
                    "reference_recovery_penalty": round(sum(reference[task] for task in common) / len(common), 4),
                    "model_recovery_penalty": round(sum(model[task] for task in common) / len(common), 4),
                    "penalty_difference": round(sum(differences) / len(differences), 4),
                    "difference_ci95": bootstrap_ci(differences),
                    "sign_flip_p": sign_flip_p(differences),
                })
    for row, adjusted in zip(penalty_rows, holm_adjust([row["sign_flip_p"] for row in penalty_rows])):
        row["sign_flip_holm_p"] = adjusted
    return {
        "reference_label": reference_label,
        "label_factors": factors,
        "success_differences": success_rows,
        "horizon_interactions": interaction_rows,
        "recovery_penalty_differences": penalty_rows,
        "method": {
            "pairing": "identical task IDs within condition and horizon; base scenarios across horizons",
            "ci": f"percentile bootstrap over paired units, {BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}",
            "tests": ["two-sided exact McNemar on discordant pairs (success differences)",
                      "two-sided sign-flip permutation over paired units (interactions and penalties), "
                      "exact for integer differences"],
            "interaction_reading": "negative interaction = the quantized model loses more at the longer horizon",
            "multiple_comparisons": "Holm adjustment within each of the three families",
            "limitations": ["Each task runs once at temperature 0, so within-task sampling variance is not estimated.",
                            "Horizon conditions of one base scenario are matched observations, not independent tasks."],
        },
    }
