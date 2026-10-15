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

from src.execution_environment import WORKFLOWS, resource_id
from src.tasks import stable_hash


def choose_candidate(spec: dict) -> dict:
    def eligible(candidate: dict) -> bool:
        summary = candidate["summary"]
        return all(
            summary[key[4:]] >= value if key.startswith("min_") else summary[key[4:]] <= value
            for key, value in spec["request"]["constraints"].items()
        )

    candidates = [item for item in spec["candidates"] if eligible(item)]
    ranking = spec["request"]["ranking"]
    return min(candidates, key=lambda item: tuple(
        item["summary"][rule["field"]] * (1 if rule["direction"] == "min" else -1)
        for rule in ranking
    ))


def scenario(domain: str, index: int) -> dict:
    base_id = f"{domain}_{index + 1:03d}"
    request_id = f"request_{stable_hash(base_id)[:12]}"
    if domain == "shopping":
        label, category, budget, minimum = [
            ("키보드", "keyboard", 100, 4.0), ("마우스", "mouse", 60, 4.2),
            ("헤드셋", "headset", 150, 4.3), ("웹캠", "webcam", 90, 4.1),
            ("USB 허브", "usb_hub", 70, 4.0),
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
        rule = "가격이 가장 낮은 상품을 고르고, 동률이면 평점이 높은 상품" if cheap_first else "평점이 가장 높은 상품을 고르고, 동률이면 가격이 낮은 상품"
        description = f"{label} 중 기본 가격 {budget} 이하이고 평점 {minimum} 이상인 상품에서 {rule}을 선택한다. 배송 방식은 {required}로 지정한다."
        result_fields = {"category": category, "quantity": 1, "delivery_city": "Seoul"}
        goal = "구매 요청의 조건에 맞는 상품과 배송 방식을 선택해 구매 항목을 확정·저장해줘."
    elif domain == "schedule":
        title, date, minimum_capacity, earliest, latest = [
            ("팀 회의", "2026-10-20", 4, 9, 17),
            ("논문 읽기 모임", "2026-10-21", 3, 10, 18),
            ("실험 점검 회의", "2026-10-22", 5, 9, 16),
            ("프로젝트 발표 연습", "2026-10-23", 5, 13, 19),
            ("주간 회고", "2026-10-24", 4, 11, 18),
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
        rule = "도보 이동시간이 가장 짧은 시간대, 동률이면 시작 시간이 빠른 시간대" if walk_first else "시작 시간이 가장 빠른 시간대, 동률이면 도보 이동시간이 짧은 시간대"
        description = f"{date}에 '{title}'를 1시간 등록한다. 정원 {minimum_capacity} 이상, 시작 시각 {earliest}시 이상 {latest}시 이하인 빈 시간대 중 {rule}를 선택한다. 진행 방식은 {required}이다."
        result_fields = {"title": title, "date": date, "duration_min": 60, "city": "Seoul"}
        label = title
        goal = "일정 요청에 적힌 날짜·시간·정원 조건과 진행 방식에 맞는 일정을 확정·저장해줘."
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
        rule = "기본 운임이 가장 낮은 이동편, 동률이면 이동시간이 짧은 이동편" if cheap_first else "이동시간이 가장 짧은 이동편, 동률이면 기본 운임이 낮은 이동편"
        description = f"{date} {origin}에서 {destination}으로 간다. 기본 운임 {budget} 이하, 출발 시각 {earliest}시 이상 {earliest + 4}시 이하인 후보 중 {rule}을 선택한다. 운임 옵션은 {required}이다."
        result_fields = {"origin": origin, "destination": destination, "date": date, "passengers": 1}
        label = f"{origin}→{destination}"
        goal = "이동 요청의 구간·날짜·기본 운임·출발 시각 조건과 운임 옵션에 맞는 여행 일정을 확정·저장해줘."
    else:
        label, category, city, budget, max_walk = [
            ("박물관", "museum", "Seoul", 30, 25), ("카페", "cafe", "Seoul", 20, 20),
            ("식당", "restaurant", "Busan", 40, 30), ("공원", "park", "Seoul", 10, 25),
            ("전시관", "gallery", "Busan", 25, 20),
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
        rule = "도보 이동시간이 가장 짧은 장소, 동률이면 평점이 높은 장소" if walk_first else "평점이 가장 높은 장소, 동률이면 도보 이동시간이 짧은 장소"
        description = f"{city}의 {label} 중 기본 이용료 {budget} 이하, 도보 이동시간 {max_walk}분 이하인 후보에서 {rule}를 선택한다. 방문 구역은 {required}로 지정한다."
        result_fields = {"city": city, "category": category, "date": f"2026-10-{20 + index}", "visitors": 1}
        goal = "방문 요청의 지역·이용료·이동시간 조건과 방문 구역에 맞는 장소 방문 계획을 확정·저장해줘."

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
        "ranking_note": "Apply all constraints first, then compare ranking fields in listed order. All ranked scores in this fixture are distinct.",
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
        candidates.append({"candidate_id": candidate_id, "summary": {"label": f"{label} 후보 {chr(65 + label_number)}", **value}, "options": options})
    rotation = index % len(candidates)
    candidates = candidates[rotation:] + candidates[:rotation]
    return {"base_id": base_id, "domain": domain, "request_id": request_id,
            "query": f"요청 ID {request_id}. {goal} 이미 제공된 도구 결과가 있으면 이어서 진행하고, 저장된 결과는 요청에 맞는 한 개만 남겨줘.",
            "request": request, "candidates": candidates}


def build_execution_tasks() -> list[dict]:
    tasks = []
    for horizon in (1, 3, 5):
        for domain in WORKFLOWS:
            for index in range(5):
                case = scenario(domain, index)
                spec = {key: copy.deepcopy(value) for key, value in case.items() if key not in {"base_id", "query"}}
                selected = choose_candidate(spec)
                option = next(item for item in selected["options"] if item["kind"] == spec["request"]["required_option_kind"])
                names = WORKFLOWS[domain]["names"]
                chain = [
                    {"tool": names[0], "arguments": {"request_id": spec["request_id"]}},
                    {"tool": names[1], "arguments": {"context_id": resource_id(spec, "context")}},
                    {"tool": names[2], "arguments": {"offer_id": resource_id(spec, "offer", selected["candidate_id"])}},
                    {"tool": names[3], "arguments": {"detail_id": resource_id(spec, "detail", selected["candidate_id"]), "option_id": option["option_id"]}},
                    {"tool": names[4], "arguments": {"draft_id": resource_id(spec, "draft", selected["candidate_id"], option["option_id"])}},
                ]
                target = {"request_id": spec["request_id"], "domain": domain, "candidate_id": selected["candidate_id"], "option_id": option["option_id"],
                          "result": {**spec["request"]["result_fields"], **selected["summary"], "option_kind": option["kind"]}}
                tasks.append({
                    "id": f"{case['base_id']}_h{horizon}", "base_id": case["base_id"],
                    "domain": domain, "task_type": "execution", "schema_version": "execution-v2",
                    "horizon": horizon, "query": case["query"], "initial_actions": chain[:5 - horizon],
                    "gold_actions": chain[5 - horizon:], "checkpoint_step": {1: 0, 3: 1, 5: 2}[horizon],
                    "success_condition": {"completed_request_equals": target}, "execution_spec": spec,
                })
    return tasks


def catalog_markdown() -> str:
    lines = [
        "# Execution task v2: 20개 시나리오 × 1·3·5단계", "",
        "60개는 서로 독립인 60개 질문이 아니라, 같은 20개 시나리오의 시작 정보 3조건입니다.",
        "각 기본 질문은 동일하고, 짧은 조건에는 정상적으로 미리 수행된 도구 결과가 대화 기록으로 제공됩니다.", "",
        "| Horizon | 미리 제공된 결과 | 모델이 수행하는 필수 호출 |", "|---|---|---|",
        "| 1 | 요청·후보·상세·준비 결과 | 확정 |",
        "| 3 | 요청·후보 결과 | 상세 확인 → 준비 → 확정 |",
        "| 5 | 없음 | 요청 확인 → 후보 조회 → 상세 확인 → 준비 → 확정 |", "",
        "준비 단계에서 생성된 draft_id가 없으면 확정할 수 없습니다. 조회용 식별자 역시 앞 도구에서 발급되어야 존재합니다.",
        "각 조건의 도구 목록은 동일합니다. 완료 후 inspect_state를 호출해도 최종 성공 판정은 유지됩니다.", "",
        "아래 각 행에 `_h1`, `_h3`, `_h5`가 붙은 task 3개가 있습니다. 실제 질문·도구 결과·정답은 tasks.json에 저장됩니다.", "",
        "| 기본 ID | 요청 내용 |", "|---|---|",
    ]
    for domain in WORKFLOWS:
        for index in range(5):
            case = scenario(domain, index)
            lines.append(f"| {case['base_id']} | {case['request']['description']} |")
    lines += ["", "## 해석 범위", "",
              "이 설계는 같은 작업 흐름의 남은 의존 단계 수를 통제합니다. 시작 시 제공되는 정보량과 완료된 작업도 달라지므로, 일반적인 현실 과제의 길이만 바꾼 실험이라고 해석하지 않습니다.",
              "h1은 올바르게 준비된 결과를 확정하는 조건이므로 후보 선택 능력은 주로 h3·h5에서 비교합니다. 20개 기본 시나리오를 분석 단위로 보고 각 h조건은 대응 비교합니다.",
              "현재 4개 분야 모두 같은 요청→조회→상세→준비→확정 구조를 사용합니다. 다양한 Agent 행동 전반에 일반화하려면 후속 데이터 확장이 필요합니다.",
              "execution-v2는 오류를 주입하지 않습니다. 기존 recovery 60개는 legacy-v1이며 새 execution과 의미상 짝을 이루지 않으므로 recovery_gap을 계산하면 안 됩니다.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(ROOT / "tasks" / "tasks.json"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--emit-json", action="store_true", help="Print regenerated execution + unchanged legacy recovery JSON")
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
