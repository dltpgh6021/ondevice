"""Build matched execution fixtures (h1/h3/h5) from inspectable scenario definitions.

The checked-in JSON is the runtime input. This script prints regenerated data
or checks that it matches; it never overwrites recovery tasks or results.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.execution_environment import WORKFLOWS, candidate_is_eligible, select_target, workflow_chain
from src.dataset_spec import BASE_SCENARIOS, DATASET_VERSION, EXECUTION_TASKS, SCENARIOS_PER_DOMAIN
from src.tasks import stable_hash


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
    # Balanced over each domain and decorrelated from the candidate position.
    domain_index = list(WORKFLOWS).index(spec["domain"])
    rotation = (index + index // len(prepares) + 2 * domain_index + 1) % len(prepares)
    prepares = prepares[rotation:] + prepares[:rotation]
    return correct[:3] + [wrong_candidate[2]] + prepares


# Fifteen rows per domain. The structural pattern (index % 5) decides the
# ranking rule, which constraint the trap violates and whether a tie must be
# broken by the secondary rule; the rows vary the content and magnitudes.
SHOPPING = [
    ("Keyboard", "keyboard", 100, 4.0), ("Mouse", "mouse", 60, 4.2), ("Headset", "headset", 150, 4.3),
    ("Webcam", "webcam", 90, 4.1), ("USB hub", "usb_hub", 70, 4.0), ("Monitor", "monitor", 220, 4.2),
    ("Laptop stand", "laptop_stand", 55, 4.1), ("Microphone", "microphone", 130, 4.3),
    ("Speaker", "speaker", 85, 4.0), ("Charger", "charger", 65, 4.2), ("Desk lamp", "desk_lamp", 50, 4.1),
    ("External SSD", "external_ssd", 140, 4.0), ("Router", "router", 120, 4.2),
    ("Tablet case", "tablet_case", 45, 4.1), ("Ring light", "ring_light", 75, 4.3),
]
SCHEDULE = [
    ("Team meeting", "2026-10-20", 4, 9, 17), ("Paper reading group", "2026-10-21", 3, 10, 18),
    ("Experiment review meeting", "2026-10-22", 5, 9, 16), ("Project presentation rehearsal", "2026-10-23", 5, 13, 19),
    ("Weekly retrospective", "2026-10-24", 4, 11, 18), ("Design critique", "2026-10-26", 3, 10, 15),
    ("Budget planning session", "2026-10-27", 6, 8, 14), ("Lab safety briefing", "2026-10-28", 4, 14, 18),
    ("Mentoring session", "2026-10-29", 2, 9, 12), ("Code review meeting", "2026-10-30", 5, 15, 20),
    ("Thesis progress check", "2026-11-02", 3, 11, 16), ("Hackathon kickoff", "2026-11-03", 8, 9, 13),
    ("Client demo preparation", "2026-11-04", 4, 13, 18), ("Reading club", "2026-11-05", 6, 16, 21),
    ("Quarterly planning", "2026-11-06", 7, 10, 17),
]
TRAVEL = [
    ("Seoul", "Busan", "2026-10-20", 100, 8), ("Seoul", "Daejeon", "2026-10-21", 60, 9),
    ("Busan", "Daegu", "2026-10-22", 50, 10), ("Daejeon", "Seoul", "2026-10-23", 70, 13),
    ("Daegu", "Seoul", "2026-10-24", 90, 11), ("Seoul", "Gwangju", "2026-10-26", 80, 7),
    ("Incheon", "Busan", "2026-10-27", 110, 12), ("Gwangju", "Seoul", "2026-10-28", 75, 15),
    ("Busan", "Gangneung", "2026-10-29", 95, 9), ("Suwon", "Daegu", "2026-10-30", 65, 14),
    ("Seoul", "Jeonju", "2026-11-02", 55, 10), ("Daegu", "Gwangju", "2026-11-03", 45, 8),
    ("Gangneung", "Seoul", "2026-11-04", 85, 16), ("Jeonju", "Busan", "2026-11-05", 70, 11),
    ("Ulsan", "Seoul", "2026-11-06", 120, 6),
]
PLACE = [
    ("Museum", "museum", "Seoul", 30, 25), ("Cafe", "cafe", "Seoul", 20, 20),
    ("Restaurant", "restaurant", "Busan", 40, 30), ("Park", "park", "Seoul", 10, 25),
    ("Gallery", "gallery", "Busan", 25, 20), ("Aquarium", "aquarium", "Busan", 35, 30),
    ("Bookstore", "bookstore", "Seoul", 15, 15), ("Botanical garden", "botanical_garden", "Daejeon", 20, 35),
    ("Night market", "night_market", "Busan", 12, 18), ("Science center", "science_center", "Daejeon", 28, 22),
    ("Palace", "palace", "Seoul", 18, 25), ("Bakery", "bakery", "Jeonju", 16, 14),
    ("Concert hall", "concert_hall", "Seoul", 45, 20), ("Library", "library", "Daegu", 10, 16),
    ("Observatory", "observatory", "Gangneung", 22, 28),
]
for _table in (SHOPPING, SCHEDULE, TRAVEL, PLACE):
    assert len(_table) == SCENARIOS_PER_DOMAIN, "one row per scenario index"


def _rating(value: float) -> float:
    """One decimal place, so model-facing JSON never shows float artifacts."""
    return round(value, 1)


def scenario(domain: str, index: int) -> dict:
    base_id = f"{domain}_{index + 1:03d}"
    request_id = f"request_{stable_hash(base_id)[:12]}"
    pattern = index % 5
    if domain == "shopping":
        label, category, budget, minimum = SHOPPING[index]
        cheap_first = pattern in (1, 4)
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
        title, day, minimum_capacity, earliest, latest = SCHEDULE[index]
        constraints = {"min_capacity": minimum_capacity, "min_start_hour": earliest, "max_start_hour": latest}
        walk_first = pattern in (1, 3)
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
        description = f"Schedule '{title}' for one hour on {day}. Consider available slots with capacity of at least {minimum_capacity} and a start hour from {earliest} through {latest}, inclusive. Select {rule}. Use the {required} meeting format."
        result_fields = {"title": title, "date": day, "duration_min": 60, "city": "Seoul"}
        label = title
        goal = "Confirm and save an event that satisfies the date, start-hour, capacity, and meeting-format requirements in the scheduling request."
    elif domain == "travel":
        origin, destination, day, budget, earliest = TRAVEL[index]
        constraints = {"max_price": budget, "min_depart_hour": earliest, "max_depart_hour": earliest + 4}
        cheap_first = pattern in (1, 4)
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
        description = f"Travel from {origin} to {destination} on {day}. Consider options with a base fare of at most {budget} and a departure hour from {earliest} through {earliest + 4}, inclusive. Select {rule}. Use the {required} fare option."
        result_fields = {"origin": origin, "destination": destination, "date": day, "passengers": 1}
        label = f"{origin} to {destination}"
        goal = "Confirm and save an itinerary that satisfies the route, date, base-fare, departure-hour, and fare-option requirements in the travel request."
    else:
        label, category, city, budget, max_walk = PLACE[index]
        constraints = {"max_fee": budget, "max_walking_min": max_walk}
        walk_first = pattern in (1, 3)
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
        visit_day = (date(2026, 10, 20) + timedelta(days=index)).isoformat()
        result_fields = {"city": city, "category": category, "date": visit_day, "visitors": 1}
        goal = "Confirm and save a visit plan that satisfies the location, admission-fee, walking-time, and visit-area requirements in the visit request."

    # Vary the violated constraint and include primary-score ties that require
    # the stated secondary rule, instead of repeating one trivial selection.
    # The violating candidate (values[0]) always ranks first when constraints
    # are ignored, so choosing it is the plausible "forgot a constraint" mistake.
    if domain == "shopping":
        if pattern in (1, 4):
            values[0] = {"price": budget - 35 if pattern == 1 else budget - 40, "rating": _rating(minimum - 0.1)}
        if pattern == 2:
            values[2]["rating"] = values[1]["rating"]
        if pattern == 4:
            values[2]["price"] = values[1]["price"]
    elif domain == "schedule":
        if pattern == 1:
            values[0] = {"capacity": minimum_capacity + 1, "start_hour": earliest - 1, "walking_min": 2}
        if pattern == 2:
            values[2]["start_hour"] = values[1]["start_hour"]
        if pattern == 3:
            values[1]["walking_min"] = values[2]["walking_min"]
    elif domain == "travel":
        if pattern in (1, 4):
            values[0] = {"price": budget - 30, "depart_hour": earliest - 1, "duration_min": 65}
        if pattern == 2:
            values[2]["duration_min"] = values[1]["duration_min"]
        if pattern == 4:
            values[2]["price"] = values[1]["price"]
    else:
        if pattern == 0:
            values[0] = {"fee": max(0, budget - 10), "walking_min": max_walk + 5, "rating": 4.9}
        if pattern == 2:
            values[2]["rating"] = values[1]["rating"]
        if pattern == 3:
            values[1]["walking_min"] = values[2]["walking_min"]

    request = {
        "description": description, "constraints": constraints, "ranking": ranking,
        "ranking_note": "Apply all constraints first. Compare ranking fields in listed order, using later fields to break ties. The full ranking tuple uniquely selects one eligible candidate.",
        "required_option_kind": required, "result_fields": result_fields,
    }
    domain_index = list(WORKFLOWS).index(domain)
    candidates = []
    for number, value in enumerate(values):
        candidate_id = f"candidate_{stable_hash([base_id, number])[:12]}"
        options = [{"option_id": f"option_{stable_hash([base_id, number, kind])[:12]}",
                    "kind": kind, "available": True} for kind in option_kinds]
        label_number = (number + index + domain_index) % 3
        candidates.append({"candidate_id": candidate_id, "summary": {"label": f"{label} candidate {chr(65 + label_number)}", **value}, "options": options})

    # Place the correct candidate and its requested option at positions that
    # are balanced within each domain, so no list position predicts the answer.
    probe = {"domain": domain, "request_id": request_id, "request": request, "candidates": candidates}
    target, _ = select_target(probe)
    violating = next(item for item in candidates if not candidate_is_eligible(item, probe))
    other = next(item for item in candidates if item is not target and item is not violating)
    target_slot = (index + domain_index) % 3
    violating_slot = (target_slot + 1 + (index // 3 + domain_index) % 2) % 3
    other_slot = 3 - target_slot - violating_slot
    ordered: list[dict] = [{}, {}, {}]
    ordered[target_slot], ordered[violating_slot], ordered[other_slot] = target, violating, other
    for number, candidate in enumerate(ordered):
        if candidate is target:
            required_first = (index + domain_index) % 2 == 0
        else:
            required_first = (index + number + domain_index) % 2 == 1
        candidate["options"].sort(key=lambda item: (item["kind"] != required) if required_first else (item["kind"] == required))
    return {"base_id": base_id, "domain": domain, "request_id": request_id,
            "query": f"Request ID: {request_id}. {goal} Continue from any information or tool results already provided. When finished, the entire saved result set must contain only one record: the correct result for this request.",
            "request": request, "candidates": ordered}


def build_execution_tasks() -> list[dict]:
    tasks = []
    for horizon in (1, 3, 5):
        for domain in WORKFLOWS:
            for index in range(SCENARIOS_PER_DOMAIN):
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
        f"# Execution tasks v2: {BASE_SCENARIOS} scenarios x horizons 1, 3, and 5", "",
        f"The {EXECUTION_TASKS} tasks are three starting-information conditions for the same {BASE_SCENARIOS} scenarios, rather than {EXECUTION_TASKS} independent questions.",
        "Each scenario keeps the same original question. Shorter conditions seed earlier lookup results into the environment and show them in the user message.",
        "They are never presented as the model's own earlier tool calls; that format is reserved for the matched recovery fixtures (normal_restart / error_restart).", "",
        "| Horizon | Information provided in the user message | Required model calls |", "|---|---|---|",
        "| 1 | Request, candidates, two details, and three prepared drafts (one correct) | Choose and confirm the correct draft |",
        "| 3 | Request and candidates | Read details -> prepare -> confirm |",
        "| 5 | None | Read request -> find candidates -> read details -> prepare -> confirm |", "",
        "At h1 the two wrong drafts are the best candidate with the other option kind and the constraint-violating candidate with the requested option kind, so a selection decision remains.",
        "Within every domain the positions of the correct candidate, the trap, the requested option and the correct h1 draft are balanced, so no list position predicts the answer.",
        "Confirmation requires an existing draft_id created by preparation. Lookup identifiers must likewise be issued by the preceding tools before those resources exist.",
        "All horizons expose the same tool list. Calling inspect_state after completion does not change the final success judgment.", "",
        "Every row below has three tasks with the suffixes `_h1`, `_h3`, and `_h5`. Actual questions, tool results, and reference answers are stored in tasks.json.", "",
        "| Base ID | Request |", "|---|---|",
    ]
    for domain in WORKFLOWS:
        for index in range(SCENARIOS_PER_DOMAIN):
            case = scenario(domain, index)
            lines.append(f"| {case['base_id']} | {case['request']['description']} |")
    lines += ["", "## Interpretation and limits", "",
              "This design controls the number of remaining dependent calls in the same workflow. Starting information also differs, so it does not isolate only the length of a general real-world task.",
              f"Error accumulation and first-error position from a cold start are analyzed with h5. Treat the {BASE_SCENARIOS} base scenarios as the analysis units and compare horizon conditions as matched observations.",
              "Recovery fixtures exist only for h1 and h3. An h5 normal_restart would start from the same empty state as free h5, so h5 is analyzed with free execution only.",
              "In every scenario the constraint-violating candidate ranks first when the constraints are ignored, so saving it is the plausible mistake of skipping a constraint check.",
              "All four domains currently use the same read-request -> search -> details -> prepare -> confirm workflow. Broader claims about agent behavior require more diverse tasks.",
              "The only selection decisions (candidate and option) occur at h3 and h5 alike; the two extra h5 calls pass identifiers without a choice. Report h3-h5 differences as effects of longer dialogue and identifier passing, not of additional reasoning steps.",
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
        print(f"Execution fixtures match all {BASE_SCENARIOS} scenario definitions ({EXECUTION_TASKS} tasks).")
        return 0
    recovery = [item for item in original if item.get("task_type") == "recovery"]
    print(json.dumps(generated + recovery, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
