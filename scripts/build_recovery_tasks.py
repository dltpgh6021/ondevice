"""Build matched recovery fixtures (h1 and h3) without changing execution tasks.

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

from src.dataset_spec import BASE_SCENARIOS, ERROR_KINDS, EXECUTION_TASKS, RECOVERY_HORIZONS, RECOVERY_TASKS
from src.execution_environment import SIGNALED_ERROR_MESSAGE, candidate_is_eligible, select_target, workflow_chain

# h5 has no checkpoint history: its normal branch would equal free h5 and any
# injected wrong workflow would leave the h3 situation. Recovery uses h1 and h3.

ERROR_DESCRIPTIONS = {
    "violating_candidate": (
        "Evaluator note only: from the checkpoint the history continues with the constraint-violating candidate "
        "(with the requested option) and saves it for this same request. The saved result must be replaced by "
        "the correct result."
    ),
    "wrong_option": (
        "Evaluator note only: from the checkpoint the history continues with the correct candidate but the other "
        "option kind and saves it for this same request. The saved result must be replaced by the correct result."
    ),
}


def recovery_plan(spec: dict, horizon: int, execution_seeds: list[dict], kind: str) -> dict:
    """Common checkpoint, injected wrong continuation and minimal repair for one horizon.

    The checkpoint history is the matched free task's provided context, replayed
    as the model's own calls, so every condition holds the same information.
    Both error families are mistakes free execution can make and scores as
    ``saved_result_violates_request``:

    - violating_candidate: the constraint-violating candidate (which ranks first
      when constraints are ignored) is saved with the requested option. Repair
      costs the normal remaining work (h1 1 call, h3 3 calls).
    - wrong_option: the correct candidate is saved with the other option kind.
      Its detail already exists, so repair costs h1 1 call, h3 2 calls
      (prepare -> confirm). Report recovery per error kind.
    """
    if horizon not in RECOVERY_HORIZONS:
        raise ValueError(f"recovery fixtures exist only for horizons {RECOVERY_HORIZONS}")
    if kind not in ERROR_KINDS:
        raise ValueError(f"unknown error kind {kind}")
    best, option = select_target(spec)
    chain = workflow_chain(spec, best, option)
    required = spec["request"]["required_option_kind"]
    if kind == "violating_candidate":
        ineligible = [item for item in spec["candidates"] if not candidate_is_eligible(item, spec)]
        if len(ineligible) != 1:
            raise ValueError(f"{spec['request_id']}: expected exactly one ineligible candidate")
        wrong_candidate = ineligible[0]
        wrong_option = next(item for item in wrong_candidate["options"] if item["available"] and item["kind"] == required)
    else:
        wrong_candidate = best
        wrong_option = next(item for item in best["options"] if item["available"] and item["kind"] != required)
    wrong_chain = workflow_chain(spec, wrong_candidate, wrong_option)
    if horizon == 1:
        # All three drafts are already prepared (as in free h1); the history then
        # confirms one of the two wrong drafts instead of the correct one.
        if wrong_chain[3] not in execution_seeds:
            raise ValueError(f"{spec['request_id']}: h1 seeds lack the {kind} draft")
        injected = wrong_chain[4:]
        repair = chain[4:]
    else:
        # The history continues from the checkpoint down the wrong path.
        injected = wrong_chain[5 - horizon:]
        repair = chain[5 - horizon:] if kind == "violating_candidate" else chain[3:]
    normal = chain[5 - horizon:]
    return {"initial_actions": copy.deepcopy(execution_seeds), "gold_actions": normal,
            "injected_actions": injected, "recovery_actions": copy.deepcopy(repair)}


def build_recovery_tasks(executions: list[dict]) -> list[dict]:
    if len(executions) != EXECUTION_TASKS or any(task.get("schema_version") != "execution-v2" for task in executions):
        raise ValueError(f"Expected the {EXECUTION_TASKS} checked-in execution-v2 tasks.")
    recoveries = []
    for execution in executions:
        if execution["horizon"] not in RECOVERY_HORIZONS:
            continue
        spec = copy.deepcopy(execution["execution_spec"])
        variants = {}
        for kind in ERROR_KINDS:
            plan = recovery_plan(spec, execution["horizon"], execution["initial_actions"], kind)
            if plan["gold_actions"] != execution["gold_actions"]:
                raise ValueError(f"{execution['id']}: recovery normal oracle differs from execution")
            variants[kind] = {
                "error_injection": {"type": "wrong_saved_result", "error_kind": kind,
                                    "actions": plan["injected_actions"]},
                "recovery_actions": plan["recovery_actions"],
                "recovery_min_calls": len(plan["recovery_actions"]),
                "error_description": ERROR_DESCRIPTIONS[kind],
            }
        recoveries.append({
            "id": f"recovery_{execution['id']}", "base_id": execution["id"],
            "domain": execution["domain"], "task_type": "recovery", "schema_version": "recovery-v3",
            "language": execution["language"], "dataset_version": execution["dataset_version"],
            "horizon": execution["horizon"], "query": execution["query"],
            "initial_actions": copy.deepcopy(execution["initial_actions"]),
            "gold_actions": copy.deepcopy(execution["gold_actions"]), "checkpoint_step": 0,
            "normal_remaining_calls": execution["horizon"],
            "error_variants": variants,
            "success_condition": copy.deepcopy(execution["success_condition"]), "execution_spec": spec,
        })
    return recoveries


def catalog_markdown(executions: list[dict]) -> str:
    runs = RECOVERY_TASKS * (1 + 2 * len(ERROR_KINDS))
    lines = [
        f"# Recovery tasks v5: {BASE_SCENARIOS} scenarios x horizons 1 and 3 x {len(ERROR_KINDS)} error kinds", "",
        f"The {RECOVERY_TASKS} recovery fixtures retain the original question, target, and provided context of their matched h1 and h3 execution tasks.",
        f"Each fixture holds {len(ERROR_KINDS)} error variants injected at the same checkpoint: " + ", ".join(ERROR_KINDS) + ".",
        "Every branch uses the original requirement to leave exactly one saved result satisfying the request.",
        "In normal_restart and error_restart no extra user message reveals the error or the reference answer.",
        "signaled_error_restart is the signaled-error control: the same error history followed by one fixed, generic audit message that names no candidate, option, constraint, or answer:",
        f"> {SIGNALED_ERROR_MESSAGE}", "",
        "## Error kinds", "",
        "| Error kind | Horizon | Correct history before the checkpoint | Injected wrong continuation | Minimum normal calls | Minimum repair calls |",
        "|---|---|---|---|---|---|",
        "| violating_candidate | 1 | Request, candidates, two details, three prepared drafts (one correct), as in free h1 | Confirm the constraint-violating draft | 1 | 1 (confirm the correct draft) |",
        "| violating_candidate | 3 | Request and candidates, as in free h3 | Violating candidate details -> draft -> confirm | 3 | 3 (correct details -> draft -> confirm) |",
        "| wrong_option | 1 | Same as above | Confirm the correct candidate's draft with the other option kind | 1 | 1 (confirm the correct draft) |",
        "| wrong_option | 3 | Same as above | Correct candidate details -> draft with the other option kind -> confirm | 3 | 2 (draft with the requested option -> confirm) |", "",
        "Horizon counts remaining normal workflow dependencies from the common checkpoint. For violating_candidate the minimum repair equals the normal remaining calls.",
        "For wrong_option at h3 the correct detail is already in the history, so the minimum repair is 2 calls. Report recovery separately per error kind and do not pool the two kinds across horizons.", "",
        "There is no h5 recovery fixture. At h5 the checkpoint has no history, so normal_restart would equal free h5, and a fully injected wrong workflow would leave the same request and candidates as the h3 error branch. h5 is analyzed with free execution only.", "",
        "## Matched checkpoint branches", "",
        "Every branch replays the matched free task's provided context as the model's own earlier tool calls, so free and normal_restart hold the same information and differ only in delivery.",
        "The error branches additionally continue from the checkpoint with one wrong continuation and confirm it for the same request.",
        "Both error kinds are mistakes that free execution scores as saved_result_violates_request, so injected recovery and natural self-correction can be compared.",
        "Confirming a correct draft for the same request replaces the wrong saved result. Discarding first and then confirming is also valid.",
        "All branches share the checkpoint, correct prefix, tool schemas, original question, final goal, and response-turn budget.",
        "error_restart (silent error) and signaled_error_restart differ only by the final audit message, so their difference measures how much an explicit error signal helps.",
        f"One model therefore needs {runs} recovery continuations: {RECOVERY_TASKS} normal_restart, plus {RECOVERY_TASKS} each of error_restart and signaled_error_restart for every error kind.", "",
        "## Scenario list", "", "Every base scenario has matched h1 and h3 recovery fixtures with both error kinds.", "",
        "| Base scenario | Domain |", "|---|---|",
    ]
    for task in executions:
        if task["horizon"] == 1:
            lines.append(f"| {task['base_id']} | {task['domain']} |")
    lines += ["", "## Interpretation and limits", "",
              "These are two controlled error families of one kind of mistake (a wrong result saved for the requested ID). Do not generalize to every error type.",
              "In every scenario the violating candidate ranks first when constraints are ignored, so violating_candidate is the plausible mistake of skipping a constraint check.",
              "wrong_option keeps the correct candidate and changes only the option kind, so it tests whether the option requirement is rechecked.",
              "The injected wrong continuation is supplied history, not a naturally generated session; its length differs by horizon (1 and 3 calls).",
              "Compute recovery penalty only between normal_restart and an error branch of the same recovery fixture, not by subtracting recovery success from execution free success.",
              f"The horizon conditions are matched observations of {BASE_SCENARIOS} base scenarios, not {RECOVERY_TASKS} independent scenarios.",
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
        print(f"Recovery fixtures match all {RECOVERY_TASKS} h1/h3 execution tasks ({BASE_SCENARIOS} scenarios x 2 horizons).")
    elif args.write:
        source = Path(args.tasks).resolve()
        replacement = executions + generated
        if [task for task in replacement if task["task_type"] == "execution"] != executions:
            raise SystemExit("Refusing to alter execution fixtures.")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=source.parent,
                                         prefix="recovery-v4-", suffix=".tmp", delete=False) as handle:
            json.dump(replacement, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(source)
        print(f"Regenerated {RECOVERY_TASKS} recovery fixtures; all {EXECUTION_TASKS} execution records preserved.")
    else:
        print(json.dumps(executions + generated, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
