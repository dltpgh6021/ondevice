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

from src.execution_environment import candidate_is_eligible, select_target, workflow_chain

ERROR_DESCRIPTION = (
    "Evaluator note only: from the checkpoint the history continues with the constraint-violating candidate "
    "and saves it for this same request. The saved result must be replaced by the correct result."
)


def recovery_plan(spec: dict, horizon: int) -> dict:
    """Common checkpoint, injected wrong continuation and minimal repair for one horizon.

    The injected error is the natural free-execution mistake: a request-violating
    candidate is confirmed for the same request (``saved_result_violates_request``).
    Re-confirming a correct draft for the same request replaces it.
    """
    best, option = select_target(spec)
    chain = workflow_chain(spec, best, option)
    ineligible = [item for item in spec["candidates"] if not candidate_is_eligible(item, spec)]
    if len(ineligible) != 1:
        raise ValueError(f"{spec['request_id']}: expected exactly one ineligible candidate")
    required = spec["request"]["required_option_kind"]
    wrong_option = next(item for item in ineligible[0]["options"] if item["available"] and item["kind"] == required)
    wrong_chain = workflow_chain(spec, ineligible[0], wrong_option)
    prefix = chain[:5 - horizon]
    if horizon == 1:
        # The correct draft is already prepared; the history then inspects the
        # wrong candidate and confirms it instead. Confirming the prepared
        # correct draft repairs the saved result.
        injected, repair = wrong_chain[2:], chain[4:]
    else:
        # The history continues from the checkpoint down the wrong candidate's
        # path. Repair needs the correct details, a draft and a confirmation.
        injected, repair = wrong_chain[5 - horizon:], chain[2:]
    return {"initial_actions": prefix, "gold_actions": chain[5 - horizon:],
            "injected_actions": injected, "recovery_actions": repair}


def build_recovery_tasks(executions: list[dict]) -> list[dict]:
    if len(executions) != 60 or any(task.get("schema_version") != "execution-v2" for task in executions):
        raise ValueError("Expected the 60 checked-in execution-v2 tasks.")
    recoveries = []
    for execution in executions:
        spec = copy.deepcopy(execution["execution_spec"])
        plan = recovery_plan(spec, execution["horizon"])
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
        "# Recovery tasks v2: 20 scenarios x horizons 1, 3, and 5", "",
        "The 60 recovery fixtures retain the original question, target, and correct history of their matched execution-v2 tasks.",
        "Both branches use the original requirement to leave exactly one saved result satisfying the request.",
        "No extra user message reveals the error or the reference answer.", "",
        "| Nominal horizon | Correct history before the checkpoint | Injected wrong continuation | Minimum normal calls | Minimum repair calls |",
        "|---|---|---|---|---|",
        "| 1 | Request, candidates, correct details, correct draft | Wrong details -> wrong draft -> confirm | 1 | 1 (confirm the correct draft) |",
        "| 3 | Request and candidates | Wrong details -> wrong draft -> confirm | 3 | 3 (correct details -> draft -> confirm) |",
        "| 5 | None | Full wrong workflow (request -> candidates -> wrong details -> wrong draft -> confirm) | 5 | 3 (correct details -> draft -> confirm) |", "",
        "Horizon counts remaining normal workflow dependencies from the common checkpoint. Record normal_remaining_calls and recovery_min_calls separately.", "",
        "## Matched checkpoint branches", "",
        "Both branches replay the same correct prefix as the model's own earlier tool calls. The normal branch starts there.",
        "The error branch additionally continues from the checkpoint with the constraint-violating candidate (with the requested option kind) and confirms it for the same request.",
        "This is the same mistake that free execution scores as saved_result_violates_request, so injected recovery and natural self-correction can be compared.",
        "Confirming a correct draft for the same request replaces the wrong saved result. Discarding first and then confirming is also valid.",
        "Both branches share the checkpoint, correct prefix, tool schemas, original question, final goal, and response-turn budget. The error branch never needs more calls than the normal branch.",
        "One model therefore needs 120 recovery continuations: 60 normal_restart and 60 error_restart.", "",
        "## Scenario list", "", "Every base scenario has matched h1, h3, and h5 recovery fixtures.", "",
        "| Base scenario | Domain | Common injected error |", "|---|---|---|",
    ]
    for task in executions:
        if task["horizon"] == 1:
            lines.append(f"| {task['base_id']} | {task['domain']} | Constraint-violating candidate saved for the same request |")
    lines += ["", "## Interpretation and limits", "",
              "This is one controlled error family: a request-violating candidate saved for the requested ID. Do not generalize to every error type.",
              "The injected wrong continuation is supplied history, not a naturally generated session; its length differs by horizon (3, 3, and 5 calls).",
              "Compute recovery penalty only between normal_restart and error_restart of the same recovery fixture, not by subtracting recovery success from execution free success.",
              "The horizon conditions are matched observations of 20 base scenarios, not 60 independent scenarios.",
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
