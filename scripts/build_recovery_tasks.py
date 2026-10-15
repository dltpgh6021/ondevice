"""Build 60 matched recovery fixtures without changing execution-v2 tasks.

The runtime reads checked-in JSON. Use --check to validate, --emit-json to
inspect regenerated data, or --write to atomically regenerate recovery data.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.execution_environment import WORKFLOWS, resource_id
from src.tasks import stable_hash


def candidate_is_eligible(candidate: dict, spec: dict) -> bool:
    summary = candidate["summary"]
    return all(
        summary[key[4:]] >= value if key.startswith("min_") else summary[key[4:]] <= value
        for key, value in spec["request"]["constraints"].items()
    )


def build_recovery_tasks(executions: list[dict]) -> list[dict]:
    if len(executions) != 60 or any(task.get("schema_version") != "execution-v2" for task in executions):
        raise ValueError("Expected the 60 checked-in execution-v2 tasks.")
    recoveries = []
    for execution in executions:
        spec = copy.deepcopy(execution["execution_spec"])
        wrong_candidates = [item for item in spec["candidates"] if not candidate_is_eligible(item, spec)]
        if len(wrong_candidates) != 1:
            raise ValueError(f"{execution['id']}: expected exactly one ineligible candidate")
        wrong_candidate = wrong_candidates[0]
        wrong_options = [item for item in wrong_candidate["options"]
                         if item["available"] and item["kind"] != spec["request"]["required_option_kind"]]
        if len(wrong_options) != 1:
            raise ValueError(f"{execution['id']}: expected exactly one available undesired option")
        wrong_option = wrong_options[0]
        stale_request_id = f"request_{stable_hash([spec['request_id'], 'stale_request'])[:16]}"
        stale_draft_id = resource_id(spec, "stale_draft")
        wrong_completion = {
            "request_id": stale_request_id, "domain": spec["domain"],
            "candidate_id": wrong_candidate["candidate_id"], "option_id": wrong_option["option_id"],
            "result": {**spec["request"]["result_fields"], **wrong_candidate["summary"],
                       "option_kind": wrong_option["kind"]},
        }
        # An imported stale draft exists in both branches. It is confirmed only
        # in the error branch, so no target workflow resources are pre-created.
        spec["initial_drafts"] = {stale_draft_id: wrong_completion}
        remove_stale = {"tool": "discard_saved_result", "arguments": {"request_id": stale_request_id}}
        normal_actions = copy.deepcopy(execution["gold_actions"])
        if len(normal_actions) != execution["horizon"]:
            raise ValueError(f"{execution['id']}: horizon and normal actions differ")
        recoveries.append({
            "id": f"recovery_{execution['id']}", "base_id": execution["id"],
            "domain": execution["domain"], "task_type": "recovery", "schema_version": "recovery-v2",
            "language": execution["language"], "dataset_version": execution["dataset_version"],
            "horizon": execution["horizon"], "query": execution["query"],
            "initial_actions": copy.deepcopy(execution["initial_actions"]),
            "gold_actions": normal_actions, "checkpoint_step": 0,
            "error_injection": {"type": "extra_saved_record", "action": {
                "tool": WORKFLOWS[execution["domain"]]["names"][-1],
                "arguments": {"draft_id": stale_draft_id},
            }},
            "recovery_actions": [remove_stale] + copy.deepcopy(normal_actions),
            "normal_remaining_calls": execution["horizon"],
            "recovery_min_calls": execution["horizon"] + 1,
            "error_description": "Evaluator note only: an ineligible candidate with an undesired option is saved under a different request ID. Delete this extra record so that only the requested result remains.",
            "success_condition": copy.deepcopy(execution["success_condition"]), "execution_spec": spec,
        })
    return recoveries


def catalog_markdown(executions: list[dict]) -> str:
    lines = [
        "# Recovery tasks v2: 20 scenarios x horizons 1, 3, and 5", "",
        "The 60 recovery fixtures retain the original question, target, and correct history of their matched execution-v2 tasks.",
        "Both branches use the original requirement to leave exactly one saved result satisfying the request.",
        "No extra user message reveals the error or the reference answer.", "",
        "| Nominal horizon | Minimum normal calls | Minimum repair calls |",
        "|---|---|---|", "| 1 | 1 | 2 |", "| 3 | 3 | 4 |", "| 5 | 5 | 6 |", "",
        "Horizon counts remaining normal workflow dependencies. The error branch additionally needs one cleanup call.",
        "Record normal_remaining_calls and recovery_min_calls separately.", "",
        "## Matched checkpoint branches", "",
        "A stale draft for another request is a supplied artifact in the common starting fixture. It does not pre-create the target draft.",
        "Replay the same initial_actions to reach the common checkpoint. The normal branch starts there. The error branch additionally confirms the stale draft and appends the real assistant call and tool result to the history.",
        "The stale draft contains an ineligible candidate and an undesired option under a distinct opaque request_id.",
        "Saving the correct target does not remove the stale result. The agent must discard_saved_result using the stale request_id. Cleanup before or after the normal workflow is valid.",
        "Both branches share the checkpoint, correct history, tool schemas, original question, final goal, and response-turn budget. Only the error branch receives the injected saved result and its observable history.",
        "One model therefore needs 120 recovery continuations: 60 normal_restart and 60 error_restart.", "",
        "## Scenario list", "", "Every base scenario has matched h1, h3, and h5 recovery fixtures.", "",
        "| Base scenario | Domain | Common injected error |", "|---|---|---|",
    ]
    for task in executions:
        if task["horizon"] == 1:
            lines.append(f"| {task['base_id']} | {task['domain']} | Extra saved record violating the request |")
    lines += ["", "## Interpretation and limits", "",
              "This is one controlled error family: an extra saved result for a wrong request. Do not generalize to every error type or naturally generated agent failure.",
              "The model must recognize and remove the erroneous record. Argument correction, exception handling, and other recovery mechanisms are outside this fixture family.",
              "Compute recovery penalty only between normal_restart and error_restart of the same recovery fixture, not by subtracting recovery success from execution free success.",
              "The horizon conditions are matched observations of 20 base scenarios, not 60 independent scenarios.",
              "All supplied questions, request descriptions, candidate labels, titles, tool schemas, and injected observations use English. Metadata records language=en and dataset_version=workflow-en-v1.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(ROOT / "tasks" / "tasks.json"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--emit-json", action="store_true", help="Print unchanged execution + regenerated recovery JSON")
    mode.add_argument("--emit-doc", action="store_true", help="Print the matched recovery catalog")
    mode.add_argument("--write", action="store_true", help="Regenerate recovery fixtures, preserving every execution record")
    args = parser.parse_args()
    original = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    executions = [task for task in original if task.get("task_type", "execution") == "execution"]
    generated = build_recovery_tasks(executions)
    if args.emit_doc:
        print(catalog_markdown(executions))
    elif args.check:
        checked_in = [task for task in original if task.get("task_type") == "recovery"]
        if checked_in != generated:
            raise SystemExit("Recovery fixtures differ from the matched execution definitions.")
        print("Recovery fixtures match all 60 execution tasks (20 scenarios × 3 horizons).")
    elif args.write:
        source = Path(args.tasks).resolve()
        replacement = executions + generated
        if [task for task in replacement if task["task_type"] == "execution"] != executions:
            raise SystemExit("Refusing to alter execution fixtures.")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=source.parent,
                                         prefix="recovery-v2-", suffix=".tmp", delete=False) as handle:
            json.dump(replacement, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(source)
        print("Regenerated 60 recovery-v2 fixtures; all 60 execution records preserved.")
    else:
        print(json.dumps(executions + generated, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
