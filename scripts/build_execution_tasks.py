"""Build 60 matched execution fixtures from 20 inspectable scenario definitions.

The checked-in JSON is the runtime input. This script prints regenerated data
or checks that it matches; it never overwrites recovery tasks or results.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.execution_environment import WORKFLOWS, candidate_is_eligible, select_target, workflow_chain
from src.tasks import stable_hash

DATASET_VERSION = "workflow-en-v2"


def h1_seed_actions(spec: dict, index: int) -> list[dict]:
    """Seeds for h1: lookups plus three prepared drafts, exactly one correct.

    The model still has to choose which draft to confirm. The two wrong drafts
    are the natural mistakes: the best candidate with the other option kind,
    and the constraint-violating candidate with the requested option kind.
    """
    best, option = select_target(spec)
    required = spec["request"]["required_option_kind"]
    wrong_option = next(item for item in best["options"] if item["available"] and item["kind"] != required)
    ineligible = next(item for item in spec["candidates"] if not candidate_is_eligible(item, spec))
    ineligible_option = next(item for item in ineligible["options"] if item["available"] and item["kind"] == required)
    correct = workflow_chain(spec, best, option)
    wrong_kind = workflow_chain(spec, best, wrong_option)
    wrong_candidate = workflow_chain(spec, ineligible, ineligible_option)
    prepares = [correct[3], wrong_kind[3], wrong_candidate[3]]
    rotation = index % len(prepares)
    prepares = prepares[rotation:] + prepares[:rotation]
    return correct[:3] + [wrong_candidate[2]] + prepares


def scenario(domain: str, index: int) -> dict:
    base_id = f"{domain}_{index + 1:03d}"
    request_id = f"request_{stable_hash(base_id)[:12]}"
    if domain == "shopping":
        label, category, budget, minimum = [
            ("Keyboard", "keyboard", 100, 4.0), ("Mouse", "mouse", 60, 4.2),
            ("Headset", "headset", 150, 4.3), ("Webcam", "webcam", 90, 4.1),
            ("USB hub", "usb_hub", 70, 4.0),
        ][index]
        cheap_first = index in (1, 4)
        constraints = {"max_price": budget, "min_rating": minimum}
        ranking = [{"field": "price" if cheap_first else "rating", "direction": "min" if cheap_first else "max"},
                   {"field": "rating" if cheap_first else "price", "direction": "max" if cheap_first else "min"}]
        values = [
            {"price": budget + 20, "rating": 4.9},
            {"price": budget - 15, "rating": 4.8},
            {"price": budget - 30, "rating": 4.4},
        ]
        option_kinds = ["standard_delivery", "express_delivery"]
        required = option_kinds[index % 2]
        rule = "the lowest-priced product, breaking ties by the highest rating" if cheap_first else "the highest-rated product, breaking ties by the lowest price"
        description = f"For the {category} category, consider products with a base price of at most {budget} and a rating of at least {minimum}. Select {rule}. Use the {required} delivery option."
        result_fields = {"category": category, "quantity": 1, "delivery_city": "Seoul"}
        goal = "Select the product and delivery option that satisfy the purchase request, then confirm and save the purchase item."
    elif domain == "schedule":
        title, date, minimum_capacity, earliest, latest = [
            ("Team meeting", "2026-10-20", 4, 9, 17),
            ("Paper reading group", "2026-10-21", 3, 10, 18),
            ("Experiment review meeting", "2026-10-22", 5, 9, 16),
            ("Project presentation rehearsal", "2026-10-23", 5, 13, 19),
            ("Weekly retrospective", "2026-10-24", 4, 11, 18),
        ][index]
        constraints = {"min_capacity": minimum_capacity, "min_start_hour": earliest, "max_start_hour": latest}
        walk_first = index in (1, 3)
        ranking = [{"field": "walking_min" if walk_first else "start_hour", "direction": "min"},
                   {"field": "start_hour" if walk_first else "walking_min", "direction": "min"}]
        values = [
            {"capacity": minimum_capacity - 1, "start_hour": earliest, "walking_min": 2},
            {"capacity": minimum_capacity + 1, "start_hour": earliest + 1, "walking_min": 12},
            {"capacity": minimum_capacity + 2, "start_hour": earliest + 2, "walking_min": 5},
        ]
        option_kinds = ["in_person", "hybrid"]
        required = option_kinds[index % 2]
        rule = "the slot with the shortest walking time, breaking ties by the earliest start hour" if walk_first else "the slot with the earliest start hour, breaking ties by the shortest walking time"
        description = f"Schedule '{title}' for one hour on {date}. Consider available slots with capacity of at least {minimum_capacity} and a start hour from {earliest} through {latest}, inclusive. Select {rule}. Use the {required} meeting format."
        result_fields = {"title": title, "date": date, "duration_min": 60, "city": "Seoul"}
        label = title
        goal = "Confirm and save an event that satisfies the date, start-hour, capacity, and meeting-format requirements in the scheduling request."
    elif domain == "travel":
        origin, destination, date, budget, earliest = [
            ("Seoul", "Busan", "2026-10-20", 100, 8),
            ("Seoul", "Daejeon", "2026-10-21", 60, 9),
            ("Busan", "Daegu", "2026-10-22", 50, 10),
            ("Daejeon", "Seoul", "2026-10-23", 70, 13),
            ("Daegu", "Seoul", "2026-10-24", 90, 11),
        ][index]
        constraints = {"max_price": budget, "min_depart_hour": earliest, "max_depart_hour": earliest + 4}
        cheap_first = index in (1, 4)
        ranking = [{"field": "price" if cheap_first else "duration_min", "direction": "min"},
                   {"field": "duration_min" if cheap_first else "price", "direction": "min"}]
        values = [
            {"price": budget + 15, "depart_hour": earliest, "duration_min": 65},
            {"price": budget - 10, "depart_hour": earliest + 1, "duration_min": 90},
            {"price": budget - 25, "depart_hour": earliest + 2, "duration_min": 140},
        ]
        option_kinds = ["refundable", "nonrefundable"]
        required = option_kinds[index % 2]
        rule = "the option with the lowest base fare, breaking ties by the shortest travel duration" if cheap_first else "the option with the shortest travel duration, breaking ties by the lowest base fare"
        description = f"Travel from {origin} to {destination} on {date}. Consider options with a base fare of at most {budget} and a departure hour from {earliest} through {earliest + 4}, inclusive. Select {rule}. Use the {required} fare option."
        result_fields = {"origin": origin, "destination": destination, "date": date, "passengers": 1}
        label = f"{origin} to {destination}"
        goal = "Confirm and save an itinerary that satisfies the route, date, base-fare, departure-hour, and fare-option requirements in the travel request."
    else:
        label, category, city, budget, max_walk = [
            ("Museum", "museum", "Seoul", 30, 25), ("Cafe", "cafe", "Seoul", 20, 20),
            ("Restaurant", "restaurant", "Busan", 40, 30), ("Park", "park", "Seoul", 10, 25),
            ("Gallery", "gallery", "Busan", 25, 20),
        ][index]
        constraints = {"max_fee": budget, "max_walking_min": max_walk}
        walk_first = index in (1, 3)
        ranking = [{"field": "walking_min" if walk_first else "rating", "direction": "min" if walk_first else "max"},
                   {"field": "rating" if walk_first else "walking_min", "direction": "max" if walk_first else "min"}]
        values = [
            {"fee": budget + 10, "walking_min": 3, "rating": 4.9},
            {"fee": max(0, budget - 5), "walking_min": max_walk - 3, "rating": 4.8},
            {"fee": max(0, budget - 8), "walking_min": 6, "rating": 4.4},
        ]
        option_kinds = ["indoor", "outdoor"]
        required = option_kinds[index % 2]
        rule = "the place with the shortest walking time, breaking ties by the highest rating" if walk_first else "the highest-rated place, breaking ties by the shortest walking time"
        description = f"Find a {category} in {city}. Consider candidates with a base admission fee of at most {budget} and a walking time of at most {max_walk} minutes. Select {rule}. Choose the {required} visit area."
        result_fields = {"city": city, "category": category, "date": f"2026-10-{20 + index}", "visitors": 1}
        goal = "Confirm and save a visit plan that satisfies the location, admission-fee, walking-time, and visit-area requirements in the visit request."

    # Vary the violated constraint and include primary-score ties that require
    # the stated secondary rule, instead of repeating one trivial selection.
    if domain == "shopping":
        if index == 1:
            values[0] = {"price": budget - 35, "rating": minimum - 0.1}
        if index == 2:
            values[2]["rating"] = values[1]["rating"]
        if index == 4:
            values[2]["price"] = values[1]["price"]
    elif domain == "schedule":
        if index == 1:
            values[0] = {"capacity": minimum_capacity + 1, "start_hour": earliest - 1, "walking_min": 2}
        if index == 2:
            values[2]["start_hour"] = values[1]["start_hour"]
        if index == 3:
            values[1]["walking_min"] = values[2]["walking_min"]
    elif domain == "travel":
        if index == 1:
            values[0] = {"price": budget - 30, "depart_hour": earliest - 1, "duration_min": 65}
        if index == 2:
            values[2]["duration_min"] = values[1]["duration_min"]
        if index == 4:
            values[2]["price"] = values[1]["price"]
    else:
        if index == 1:
            values[0] = {"fee": max(0, budget - 10), "walking_min": max_walk + 5, "rating": 4.9}
        if index == 2:
            values[2]["rating"] = values[1]["rating"]
        if index == 3:
            values[1]["walking_min"] = values[2]["walking_min"]

    request = {
        "description": description, "constraints": constraints, "ranking": ranking,
        "ranking_note": "Apply all constraints first. Compare ranking fields in listed order, using later fields to break ties. The full ranking tuple uniquely selects one eligible candidate.",
        "required_option_kind": required, "result_fields": result_fields,
    }
    candidates = []
    for number, value in enumerate(values):
        candidate_id = f"candidate_{stable_hash([base_id, number])[:12]}"
        options = [{"option_id": f"option_{stable_hash([base_id, number, kind])[:12]}",
                    "kind": kind, "available": True} for kind in option_kinds]
        # Change option and candidate ordering deterministically across scenarios.
        if (index + number) % 2:
            options.reverse()
        label_number = (number + index + list(WORKFLOWS).index(domain)) % 3
        candidates.append({"candidate_id": candidate_id, "summary": {"label": f"{label} candidate {chr(65 + label_number)}", **value}, "options": options})
    rotation = index % len(candidates)
    candidates = candidates[rotation:] + candidates[:rotation]
    return {"base_id": base_id, "domain": domain, "request_id": request_id,
            "query": f"Request ID: {request_id}. {goal} Continue from any information or tool results already provided. When finished, the entire saved result set must contain only one record: the correct result for this request.",
            "request": request, "candidates": candidates}


def build_execution_tasks() -> list[dict]:
    tasks = []
    for horizon in (1, 3, 5):
        for domain in WORKFLOWS:
            for index in range(5):
                case = scenario(domain, index)
                spec = {key: copy.deepcopy(value) for key, value in case.items() if key not in {"base_id", "query"}}
                selected, option = select_target(spec)
                chain = workflow_chain(spec, selected, option)
                target = {"request_id": spec["request_id"], "domain": domain, "candidate_id": selected["candidate_id"], "option_id": option["option_id"],
                          "result": {**spec["request"]["result_fields"], **selected["summary"], "option_kind": option["kind"]}}
                # initial_actions are state seeds shown to the model as user-provided
                # information, never as its own earlier tool calls.
                seeds = h1_seed_actions(spec, index) if horizon == 1 else chain[:5 - horizon]
                tasks.append({
                    "id": f"{case['base_id']}_h{horizon}", "base_id": case["base_id"],
                    "domain": domain, "task_type": "execution", "schema_version": "execution-v2",
                    "language": "en", "dataset_version": DATASET_VERSION,
                    "horizon": horizon, "query": case["query"], "initial_actions": seeds,
                    "gold_actions": chain[5 - horizon:], "checkpoint_step": {1: 0, 3: 1, 5: 2}[horizon],
                    "success_condition": {"completed_request_equals": target}, "execution_spec": spec,
                })
    return tasks


def catalog_markdown() -> str:
    lines = [
        "# Execution tasks v2: 20 scenarios x horizons 1, 3, and 5", "",
        "The 60 tasks are three starting-information conditions for the same 20 scenarios, rather than 60 independent questions.",
        "Each scenario keeps the same original question. Shorter conditions seed earlier lookup results into the environment and show them in the user message.",
        "They are never presented as the model's own earlier tool calls; that format is reserved for the matched recovery fixtures (normal_restart / error_restart).", "",
        "| Horizon | Information provided in the user message | Required model calls |", "|---|---|---|",
        "| 1 | Request, candidates, two details, and three prepared drafts (one correct) | Choose and confirm the correct draft |",
        "| 3 | Request and candidates | Read details -> prepare -> confirm |",
        "| 5 | None | Read request -> find candidates -> read details -> prepare -> confirm |", "",
        "At h1 the two wrong drafts are the best candidate with the other option kind and the constraint-violating candidate with the requested option kind, so a selection decision remains.",
        "Confirmation requires an existing draft_id created by preparation. Lookup identifiers must likewise be issued by the preceding tools before those resources exist.",
        "All horizons expose the same tool list. Calling inspect_state after completion does not change the final success judgment.", "",
        "Every row below has three tasks with the suffixes `_h1`, `_h3`, and `_h5`. Actual questions, tool results, and reference answers are stored in tasks.json.", "",
        "| Base ID | Request |", "|---|---|",
    ]
    for domain in WORKFLOWS:
        for index in range(5):
            case = scenario(domain, index)
            lines.append(f"| {case['base_id']} | {case['request']['description']} |")
    lines += ["", "## Interpretation and limits", "",
              "This design controls the number of remaining dependent calls in the same workflow. Starting information also differs, so it does not isolate only the length of a general real-world task.",
              "Error accumulation and first-error position from a cold start are analyzed with h5. Treat the 20 base scenarios as the analysis units and compare horizon conditions as matched observations.",
              "At h5 free execution and normal_restart start from the same empty state and therefore present identical model inputs by definition; at h1 and h3 they differ in how earlier results are delivered.",
              "All four domains currently use the same read-request -> search -> details -> prepare -> confirm workflow. Broader claims about agent behavior require more diverse tasks.",
              "Free execution does not inject errors. Compare recovery only between normal_restart and error_restart within the matched fixture. Do not directly subtract recovery success from free-execution success.",
              f"The model-facing dataset language is English, recorded as language=en and dataset_version={DATASET_VERSION}.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(ROOT / "tasks" / "tasks.json"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--emit-json", action="store_true", help="Print regenerated execution + unchanged recovery JSON")
    mode.add_argument("--emit-doc", action="store_true", help="Print the scenario catalog")
    args = parser.parse_args()
    if args.emit_doc:
        print(catalog_markdown())
        return 0
    original = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    generated = build_execution_tasks()
    if args.check:
        checked_in = [item for item in original if item.get("task_type", "execution") == "execution"]
        if checked_in != generated:
            raise SystemExit("Execution fixtures differ from the scenario definitions.")
        print("Execution fixtures match all 20 scenario definitions (60 tasks).")
        return 0
    recovery = [item for item in original if item.get("task_type") == "recovery"]
    print(json.dumps(generated + recovery, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
