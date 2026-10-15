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
            "error_description": "실험자용: 조건에 맞지 않는 후보·옵션의 결과가 별도 요청 ID로 추가 저장됨. 원래 요청 결과만 남기려면 이 기록을 삭제해야 함.",
            "success_condition": copy.deepcopy(execution["success_condition"]), "execution_spec": spec,
        })
    return recoveries


def catalog_markdown(executions: list[dict]) -> str:
    lines = [
        "# Recovery task v2: 20개 시나리오 × 1·3·5단계", "",
        "execution-v2 60개 각각과 원래 질문·정답·정상 작업 기록이 같은 회복 fixture 60개입니다.",
        "원래 질문의 ‘저장된 결과는 요청에 맞는 한 개만 남겨줘’ 조건을 두 분기에서 동일하게 사용합니다.",
        "오류 내용이나 정답을 별도의 사용자 메시지로 알려주지 않습니다.", "",
        "| Horizon 표기 | 정상 분기 남은 최소 호출 | 오류 분기 남은 최소 호출 |",
        "|---|---|---|", "| 1 | 1 | 2 |", "| 3 | 3 | 4 |", "| 5 | 5 | 6 |", "",
        "horizon은 정상 작업에 남은 의존 단계 수입니다. 오류 분기는 잘못된 저장 결과 삭제에 1회가 더 필요합니다.",
        "recovery_min_calls와 normal_remaining_calls를 구분해 기록하며 추가 호출 비용은 각 분기의 최소 호출을 기준으로 해석합니다.", "",
        "## 같은 checkpoint에서 두 분기 비교", "",
        "공통 시작 상태에는 원래 요청과 무관한 오래된 draft 하나가 가져온 자료로 존재합니다. 정상 경로에서 필요한 target draft를 미리 생성하지는 않습니다.",
        "initial_actions를 동일하게 재생한 지점이 checkpoint입니다. 정상 분기는 그대로 시작하고, 오류 분기는 오래된 draft를 확정하는 도구 호출 1회와 실제 결과를 대화 기록에 추가합니다.",
        "오래된 draft는 조건에 맞지 않는 후보와 요구되지 않은 옵션으로 만들며, 원래 요청과 다른 불투명 request_id를 사용합니다.",
        "따라서 올바른 원래 요청을 확정해도 오래된 결과가 남습니다. discard_saved_result로 잘못된 기록을 삭제해야 최종 상태가 성공 조건과 일치합니다.",
        "두 분기는 같은 checkpoint·정상 기록·도구·원래 질문을 공유합니다. 오류 호출 및 관측은 오류를 드러내기 위해 오류 분기에만 추가됩니다.",
        "60개 fixture를 normal_restart와 error_restart로 각각 실행하므로 모델 하나당 회복 비교 실행은 120회입니다.", "",
        "## 시나리오 목록", "", "각 행마다 `_h1`, `_h3`, `_h5`에 대응하는 회복 task 3개가 있습니다.", "",
        "| 원래 기본 시나리오 | 분야 | 공통 주입 오류 |", "|---|---|---|",
    ]
    for task in executions:
        if task["horizon"] == 1:
            lines.append(f"| {task['base_id']} | {task['domain']} | 조건을 위반한 오래된 결과의 추가 저장 |")
    lines += ["", "## 해석 범위", "",
              "하나의 고정 오류 유형(추가 저장 기록)과 통제된 대화 기록을 이용한 실험입니다. 모든 오류 유형이나 자연 발생 오류의 회복 능력으로 일반화하지 않습니다.",
              "모델은 기록의 오류를 알아차리고 삭제할 수 있는지 평가받습니다. 도구 오류 예외 처리·잘못된 인자 수정 등 다른 회복 유형은 이번 fixture에 포함하지 않습니다.",
              "회복 손실은 같은 회복 fixture의 normal_restart 성공률과 error_restart 성공률 사이에서 계산합니다. execution 자유 수행과 직접 빼지 않습니다.",
              "20개 기본 시나리오의 세 horizon은 대응 조건입니다. 60개의 독립 시나리오처럼 취급하지 않습니다.", ""]
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
