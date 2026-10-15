"""Pair recovery-v2 continuations without mixing tasks, devices, or budgets."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Iterable

PROTOCOL = "extra-saved-record-v2"
CONFIG_FIELDS = (
    "model_sha256", "model_label", "hostname", "platform", "processor", "machine",
    "seed", "tasks_sha256", "inference_settings", "shared_extra_action_turns", "max_calls",
    "dataset_language", "dataset_version", "dataset_sha256",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _mean(values: list[float | int | bool]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def compare_recovery_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compare normal/error restarts only when their fixture and configuration match.

    Missing configuration fields remain explicit nulls in the signature and are
    reported as limitations. Missing pairing artifacts cannot form a comparison.
    Duplicate records raise rather than silently weighting a task more than once.
    """
    pairs: dict[tuple[str, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    diagnostics: dict[str, Any] = {
        "total_matched_pairs": 0, "valid_pairs": 0, "excluded_invalid_pairs": 0,
        "excluded_invalid_records": 0, "ignored_legacy_or_free_records": 0,
        "unpairable_records": 0, "unpairable_record_ids": [],
        "unmatched_normal_count": 0, "unmatched_error_count": 0,
        "unmatched_normal_ids": [], "unmatched_error_ids": [],
        "missing_metadata_fields": [], "limitations": [],
    }
    missing_metadata: set[str] = set()
    for record in records:
        task, trace = record.get("task", {}), record.get("trace", {})
        score, metadata = record.get("score", {}), record.get("metadata", {})
        condition = trace.get("condition", score.get("condition"))
        if condition == "recovery":
            condition = "error_restart"
        schema = task.get("schema_version", trace.get("schema_version"))
        if (trace.get("protocol_version") != PROTOCOL or schema != "recovery-v2"
                or condition not in {"normal_restart", "error_restart"}):
            diagnostics["ignored_legacy_or_free_records"] += 1
            continue
        pair_id = task.get("id") or trace.get("pair_id")
        fixture = trace.get("fixture_sha256")
        checkpoint = trace.get("checkpoint_state_hash")
        goal = task.get("success_condition")
        horizon = task.get("horizon", trace.get("horizon"))
        actual_budget = trace.get("max_calls")
        if (not pair_id or not fixture or not checkpoint or not isinstance(goal, dict) or not goal
                or not isinstance(horizon, int) or isinstance(horizon, bool)
                or not isinstance(actual_budget, int) or isinstance(actual_budget, bool)):
            diagnostics["unpairable_records"] += 1
            if len(diagnostics["unpairable_record_ids"]) < 20:
                diagnostics["unpairable_record_ids"].append(str(pair_id or "unknown"))
            continue
        label = metadata.get("model_label") or score.get("model_label") or "unknown"
        config = {field: metadata.get(field) for field in CONFIG_FIELDS}
        config["model_label"] = label
        config["effective_max_calls"] = actual_budget
        for field in CONFIG_FIELDS:
            value = config[field]
            # Explicit None means an automatic per-horizon budget; its actual
            # value is independently required and matched above.
            if field not in metadata or (field != "max_calls" and (value is None or value == "")):
                missing_metadata.add(field)
        if label == "unknown":
            missing_metadata.add("model_label")
        config_text = _canonical(config)
        key = (str(label), config_text, str(horizon), str(pair_id), str(fixture),
               str(checkpoint), _canonical(goal))
        if condition in pairs[key]:
            raise ValueError(f"duplicate {condition} record for {pair_id}; "
                             "separate repeated runs before paired analysis")
        pairs[key][condition] = record

    groups: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for key, conditions in pairs.items():
        if len(conditions) != 2:
            condition = next(iter(conditions))
            name = "normal" if condition == "normal_restart" else "error"
            diagnostics[f"unmatched_{name}_count"] += 1
            if len(diagnostics[f"unmatched_{name}_ids"]) < 20:
                diagnostics[f"unmatched_{name}_ids"].append(key[3])
            continue
        diagnostics["total_matched_pairs"] += 1
        normal, error = conditions["normal_restart"], conditions["error_restart"]
        valid = []
        for record in (normal, error):
            trace, score = record["trace"], record.get("score", {})
            valid.append(trace.get("runtime_error") is None
                         and score.get("valid_for_behavior_metrics") is not False
                         and isinstance(score.get("strict_success"), bool))
        diagnostics["excluded_invalid_records"] += valid.count(False)
        if not all(valid):
            diagnostics["excluded_invalid_pairs"] += 1
            continue
        diagnostics["valid_pairs"] += 1
        groups[key[:3]].append((normal, error))

    summaries = []
    for (label, config_text, horizon), matched in sorted(groups.items()):
        config = json.loads(config_text)
        normal_scores = [normal["score"] for normal, _ in matched]
        error_scores = [error["score"] for _, error in matched]
        successful_error = [score for score in error_scores if score["strict_success"]]

        def extra_mean(name: str) -> float | None:
            return _mean([score[name] for score in successful_error
                          if isinstance(score.get(name), (int, float))
                          and not isinstance(score[name], bool)])

        error_rate = _mean([score["strict_success"] for score in error_scores])
        summaries.append({
            "model_label": label, "protocol_version": PROTOCOL, "horizon": int(horizon),
            "comparison_group": hashlib.sha256(config_text.encode("utf-8")).hexdigest()[:12],
            "hostname": config["hostname"], "configuration": config,
            "n_pairs": len(matched), "n_successful_error_runs": len(successful_error),
            "normal_restart_success_rate": _mean([score["strict_success"] for score in normal_scores]),
            "error_restart_success_rate": error_rate, "recovery_rate": error_rate,
            "recovery_penalty": _mean([int(normal["strict_success"]) - int(error["strict_success"])
                                       for normal, error in zip(normal_scores, error_scores)]),
            "mean_extra_calls_over_normal": extra_mean("extra_calls_over_normal"),
            "mean_extra_calls_over_oracle": extra_mean("extra_calls_over_oracle"),
        })
    diagnostics["missing_metadata_fields"] = sorted(missing_metadata)
    if missing_metadata:
        diagnostics["limitations"].append(
            "Some configuration metadata is missing. Matching null fields cannot establish that those settings were identical.")
    if diagnostics["unpairable_records"]:
        diagnostics["limitations"].append(
            "Records lacking fixture/checkpoint/goal/horizon/budget artifacts were excluded from pairing.")
    return {"summaries": summaries, "diagnostics": diagnostics}
