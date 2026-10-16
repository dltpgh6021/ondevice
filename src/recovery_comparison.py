"""Pair recovery-v3 continuations without mixing tasks, devices, or budgets.

Each error branch (silent error_restart and signaled_error_restart) is paired
with normal_restart of the same fixture and configuration.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Iterable

from .execution_environment import RECOVERY_PROTOCOL

PROTOCOL = RECOVERY_PROTOCOL
ERROR_BRANCHES = ("error_restart", "signaled_error_restart")
CONFIG_FIELDS = (
    "model_sha256", "model_label", "kv_cache_type", "hostname", "platform", "processor", "machine",
    "seed", "tasks_sha256", "inference_settings", "shared_extra_action_turns", "max_calls",
    "dataset_language", "dataset_version", "dataset_sha256",
    "metrics_version", "agent_policy_version",
    "preparation_protocol", "server_configuration_sha256", "warmup_settings", "settings_lock_sha256",
)

VERSION_FIELDS = {"metrics_version", "agent_policy_version"}


def _configuration_value(record: dict[str, Any], field: str) -> Any:
    """Version tags can live in metadata, trace, or a previously scored row."""
    if field in VERSION_FIELDS:
        for section in ("metadata", "trace", "score"):
            value = record.get(section, {}).get(field)
            if value is not None and value != "":
                return value
        return "legacy-unversioned"
    return record.get("metadata", {}).get(field)


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
        "conflicting_version_records": 0, "conflicting_version_record_ids": [],
        "unmatched_normal_count": 0, "unmatched_error_count": 0, "unmatched_signaled_error_count": 0,
        "unmatched_normal_ids": [], "unmatched_error_ids": [], "unmatched_signaled_error_ids": [],
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
        if (trace.get("protocol_version") != PROTOCOL or schema != "recovery-v3"
                or condition not in {"normal_restart", *ERROR_BRANCHES}):
            diagnostics["ignored_legacy_or_free_records"] += 1
            continue
        pair_id = task.get("id") or trace.get("pair_id")
        conflicting_versions = []
        for field in VERSION_FIELDS:
            present = [record.get(section, {}).get(field)
                       for section in ("metadata", "trace", "score")
                       if record.get(section, {}).get(field) is not None
                       and record.get(section, {}).get(field) != ""]
            if any(not isinstance(value, str) for value in present) or len(set(map(_canonical, present))) > 1:
                conflicting_versions.append(field)
        if conflicting_versions:
            diagnostics["unpairable_records"] += 1
            diagnostics["conflicting_version_records"] += 1
            if len(diagnostics["conflicting_version_record_ids"]) < 20:
                diagnostics["conflicting_version_record_ids"].append({
                    "task_id": str(pair_id or "unknown"), "fields": sorted(conflicting_versions)})
            continue
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
        config = {field: _configuration_value(record, field) for field in CONFIG_FIELDS}
        config["model_label"] = label
        config["effective_max_calls"] = actual_budget
        for field in CONFIG_FIELDS:
            value = config[field]
            # Explicit None means an automatic per-horizon budget; its actual
            # value is independently required and matched above.
            if field in VERSION_FIELDS:
                if value == "legacy-unversioned":
                    missing_metadata.add(field)
            elif field not in metadata or (field != "max_calls" and (value is None or value == "")):
                missing_metadata.add(field)
        if label == "unknown":
            missing_metadata.add("model_label")
        config_text = _canonical(config)
        key = (str(label), config_text, str(horizon), str(pair_id), str(fixture),
               str(checkpoint), _canonical(goal))
        kind = trace.get("error_kind") or score.get("error_kind") or task.get("error_kind")
        slot = condition if condition == "normal_restart" else f"{condition}:{kind or 'unspecified'}"
        if slot in pairs[key]:
            raise ValueError(f"duplicate {slot} record for {pair_id}; "
                             "separate repeated runs before paired analysis")
        pairs[key][slot] = record

    groups: dict[tuple[str, str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)

    def usable(record: dict[str, Any]) -> bool:
        trace, score = record["trace"], record.get("score", {})
        return (trace.get("runtime_error") is None
                and trace.get("infrastructure_error") is None
                and trace.get("internal_error") is None
                and score.get("valid_for_behavior_metrics") is not False
                and isinstance(score.get("strict_success"), bool))

    unmatched_names = {"normal_restart": "normal", "error_restart": "error",
                       "signaled_error_restart": "signaled_error"}
    for key, conditions in pairs.items():
        branches = sorted(slot for slot in conditions if slot != "normal_restart")
        if "normal_restart" not in conditions or not branches:
            for condition in conditions:
                name = unmatched_names[condition.split(":")[0]]
                diagnostics[f"unmatched_{name}_count"] += 1
                if len(diagnostics[f"unmatched_{name}_ids"]) < 20:
                    diagnostics[f"unmatched_{name}_ids"].append(key[3])
            continue
        normal = conditions["normal_restart"]
        for branch in branches:
            error = conditions[branch]
            diagnostics["total_matched_pairs"] += 1
            valid = [usable(normal), usable(error)]
            diagnostics["excluded_invalid_records"] += valid.count(False)
            if not all(valid):
                diagnostics["excluded_invalid_pairs"] += 1
                continue
            diagnostics["valid_pairs"] += 1
            groups[(*key[:3], branch)].append((normal, error))

    summaries = []
    for (label, config_text, horizon, slot), matched in sorted(groups.items()):
        branch, _, error_kind = slot.partition(":")
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
            "error_condition": branch, "error_kind": error_kind or None,
            "recovery_min_calls": sorted({score.get("recovery_min_calls") for score in
                                          (error["score"] for _, error in matched)} - {None}),
            "metrics_version": config["metrics_version"],
            "agent_policy_version": config["agent_policy_version"],
            "comparison_group": hashlib.sha256(config_text.encode("utf-8")).hexdigest()[:12],
            "hostname": config["hostname"], "kv_cache_type": config.get("kv_cache_type"),
            "configuration": config,
            "n_pairs": len(matched), "n_successful_error_runs": len(successful_error),
            "normal_restart_success_rate": _mean([score["strict_success"] for score in normal_scores]),
            "error_restart_success_rate": error_rate, "recovery_rate": error_rate,
            "recovery_penalty": _mean([int(normal["strict_success"]) - int(error["strict_success"])
                                       for normal, error in zip(normal_scores, error_scores)]),
            "recovery_error_resolved_rate": _mean([score.get("recovery_error_resolved") is True for score in error_scores]),
            "mean_extra_calls_over_normal": extra_mean("extra_calls_over_normal"),
            "mean_extra_calls_over_oracle": extra_mean("extra_calls_over_oracle"),
            "error_restart_budget_exhausted_rate": _mean([score.get("budget_exhausted") is True for score in error_scores]),
            "normal_restart_budget_exhausted_rate": _mean([score.get("budget_exhausted") is True for score in normal_scores]),
            # Share of error runs whose first reply simply declares the task done.
            "immediate_finish_rate": _mean([score.get("first_error_kind") == "semantic:premature_finish"
                                            and score.get("first_error_turn") == 1 for score in error_scores]),
        })
    diagnostics["missing_metadata_fields"] = sorted(missing_metadata)
    if missing_metadata:
        diagnostics["limitations"].append(
            "Some configuration metadata is missing. Matching null fields cannot establish that those settings were identical.")
    if diagnostics["unpairable_records"]:
        diagnostics["limitations"].append(
            "Records lacking pairing artifacts or carrying inconsistent version tags were excluded from pairing.")
    return {"summaries": summaries, "diagnostics": diagnostics}
