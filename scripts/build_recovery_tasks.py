"""Build 40 matched recovery fixtures (h1 and h3) without changing execution tasks.

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

from src.execution_environment import candidate_is_eligible, select_target, workflow_chain

# h5 has no checkpoint history: its normal branch would equal free h5 and any
# injected wrong workflow would leave the h3 situation. Recovery uses h1 and h3.
RECOVERY_HORIZONS = (1, 3)

ERROR_DESCRIPTION = (
    "Evaluator note only: from the checkpoint the history continues with the constraint-violating candidate "
    "and saves it for this same request. The saved result must be replaced by the correct result."
)


def recovery_plan(spec: dict, horizon: int, execution_seeds: list[dict]) -> dict:
    """Common checkpoint, injected wrong continuation and minimal repair for one horizon.

    The checkpoint history is the matched free task's provided context, replayed
    as the model's own calls, so both conditions hold the same information.
    The injected error is the natural free-execution mistake: a request-violating
    candidate is confirmed for the same request (``saved_result_violates_request``).
    Repair needs exactly the normal remaining work, so recovery cost equals horizon.
    """
    if horizon not in RECOVERY_HORIZONS:
        raise ValueError(f"recovery fixtures exist only for horizons {RECOVERY_HORIZONS}")
    best, option = select_target(spec)
    chain = workflow_chain(spec, best, option)
    ineligible = [item for item in spec["candidates"] if not candidate_is_eligible(item, spec)]
    if len(ineligible) != 1:
        raise ValueError(f"{spec['request_id']}: expected exactly one ineligible candidate")
    required = spec["request"]["required_option_kind"]
    wrong_option = next(item for item in ineligible[0]["options"] if item["available"] and item["kind"] == required)
    wrong_chain = workflow_chain(spec, ineligible[0], wrong_option)
    if horizon == 1:
        # All three drafts are already prepared (as in free h1); the history then
        # confirms the constraint-violating draft instead of the correct one.
        if wrong_chain[3] not in execution_seeds:
            raise ValueError(f"{spec['request_id']}: h1 seeds lack the constraint-violating draft")
        injected = wrong_chain[4:]
    else:
        # The history continues from the checkpoint down the wrong candidate's path.
        injected = wrong_chain[5 - horizon:]
    normal = chain[5 - horizon:]
    return {"initial_actions": copy.deepcopy(execution_seeds), "gold_actions": normal,
            "injected_actions": injected, "recovery_actions": copy.deepcopy(normal)}


def build_recovery_tasks(executions: list[dict]) -> list[dict]:
    if len(executions) != 60 or any(task.get("schema_version") != "execution-v2" for task in executions):
        raise ValueError("Expected the 60 checked-in execution-v2 tasks.")
    recoveries = []
    for execution in executions:
        if execution["horizon"] not in RECOVERY_HORIZONS:
            continue
        spec = copy.deepcopy(execution["execution_spec"])
        plan = recovery_plan(spec, execution["horizon"], execution["initial_actions"])
        if plan["gold_actions"] != execution["gold_actions"]:
            raise ValueError(f"{execution['id']}: recovery normal oracle differs from execution")
        recoveries.append({
            "id": f"recovery_{execution['id']}", "base_id": execution["id"],
            "domain": execution["domain"], "task_type": "recovery", "schema_version": "recovery-v2",
            "language": execution["language"], "dataset_version": execution["dataset_version"],
            "horizon": execution["horizon"], "query": execution["query"],
            "initial_actions": plan["initial_actions"],
            "gold_actions": plan["gold_actions"], "checkpoint_step": 0,
            "error_injection": {"type": "wrong_saved_result", "actions": plan["injected_actions"]},
            "recovery_actions": plan["recovery_actions"],
            "normal_remaining_calls": execution["horizon"],
            "recovery_min_calls": len(plan["recovery_actions"]),
            "error_description": ERROR_DESCRIPTION,
            "success_condition": copy.deepcopy(execution["success_condition"]), "execution_spec": spec,
        })
    return recoveries


def catalog_markdown(executions: list[dict]) -> str:
    lines = [
        "# Recovery tasks v3: 20 scenarios x horizons 1 and 3", "",
        "The 40 recovery fixtures retain the original question, target, and provided context of their matched h1 and h3 execution tasks.",
        "Both branches use the original requirement to leave exactly one saved result satisfying the request.",
        "No extra user message reveals the error or the reference answer.", "",
        "| Horizon | Correct history before the checkpoint | Injected wrong continuation | Minimum normal calls | Minimum repair calls |",
        "|---|---|---|---|---|",
        "| 1 | Request, candidates, two details, and three prepared drafts (one correct), as in free h1 | Confirm the constraint-violating draft | 1 | 1 (confirm the correct draft) |",
        "| 3 | Request and candidates, as in free h3 | Wrong details -> wrong draft -> confirm | 3 | 3 (correct details -> draft -> confirm) |", "",
        "Horizon counts remaining normal workflow dependencies from the common checkpoint. Minimum repair calls equal the normal remaining calls, so recovery cost does not differ from normal work by construction.", "",
        "There is no h5 recovery fixture. At h5 the checkpoint has no history, so normal_restart would equal free h5, and a fully injected wrong workflow would leave the same request and candidates as the h3 error branch. h5 is analyzed with free execution only.", "",
        "## Matched checkpoint branches", "",
        "Both branches replay the matched free task's provided context as the model's own earlier tool calls, so free and normal_restart hold the same information and differ only in delivery.",
        "The error branch additionally continues from the checkpoint with the constraint-violating candidate (with the requested option kind) and confirms it for the same request.",
        "This is the same mistake that free execution scores as saved_result_violates_request, so injected recovery and natural self-correction can be compared.",
        "Confirming a correct draft for the same request replaces the wrong saved result. Discarding first and then confirming is also valid.",
        "Both branches share the checkpoint, correct prefix, tool schemas, original question, final goal, and response-turn budget.",
        "One model therefore needs 80 recovery continuations: 40 normal_restart and 40 error_restart.", "",
        "## Scenario list", "", "Every base scenario has matched h1 and h3 recovery fixtures.", "",
        "| Base scenario | Domain | Common injected error |", "|---|---|---|",
    ]
    for task in executions:
        if task["horizon"] == 1:
            lines.append(f"| {task['base_id']} | {task['domain']} | Constraint-violating candidate saved for the same request |")
    lines += ["", "## Interpretation and limits", "",
              "This is one controlled error family: a request-violating candidate saved for the requested ID. Do not generalize to every error type.",
              "In every scenario the violating candidate ranks first when constraints are ignored, so the injected error is the plausible mistake of skipping a constraint check.",
              "The injected wrong continuation is supplied history, not a naturally generated session; its length differs by horizon (1 and 3 calls).",
              "Compute recovery penalty only between normal_restart and error_restart of the same recovery fixture, not by subtracting recovery success from execution free success.",
              "The horizon conditions are matched observations of 20 base scenarios, not 40 independent scenarios.",
              f"All supplied questions, request descriptions, candidate labels, titles, tool schemas, and injected observations use English. Metadata records language=en and dataset_version={executions[0]['dataset_version']}.", ""]
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
        print("Recovery fixtures match all 40 h1/h3 execution tasks (20 scenarios x 2 horizons).")
    elif args.write:
        source = Path(args.tasks).resolve()
        replacement = executions + generated
        if [task for task in replacement if task["task_type"] == "execution"] != executions:
            raise SystemExit("Refusing to alter execution fixtures.")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=source.parent,
                                         prefix="recovery-v3-", suffix=".tmp", delete=False) as handle:
            json.dump(replacement, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(source)
        print("Regenerated 40 recovery fixtures; all 60 execution records preserved.")
    else:
        print(json.dumps(executions + generated, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
