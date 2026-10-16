# Quantized Agent Stability

**양자화(FP16 → Q8 → Q4)가 소형 LLM Agent의 다단계 도구 사용에서 오류를 얼마나 더 만들고, 이미 생긴 오류를 얼마나 덜 고치는지**를 같은 상태에서 재개하는 통제 실험으로 측정하는 저장소입니다.

| 항목 | 현재 값 |
|---|---|
| 모델 | Qwen3-1.7B (원본 F16 GGUF → 같은 원본에서 직접 Q8_0·Q4_K_M 생성) |
| 추론 서버 | llama.cpp `llama-server` (OpenAI 호환 `/v1/chat/completions`) |
| 데이터 버전 | `workflow-en-v4` (모델 입력은 영어) |
| 과제 수 | 기본 시나리오 60개 → execution 180 + recovery fixture 120 = **300 task** |
| 모델당 실행 수 | **780회** (FP16·Q8·Q4 전체 2,340회) |
| 프로토콜 버전 | execution `execution-v4-provided-context`, recovery `wrong-saved-result-v5` (schema `recovery-v3`) |
| 지표 / Agent 정책 | `workflow-metrics-v6` / `single-action-feedback-v2` |
| 외부 의존성 | 핵심 기능은 Python 표준 라이브러리만 사용. 선택: `psutil`(RAM), `matplotlib`(그래프) |

> 이 README의 모든 검증 결과는 **정답 재생과 코드 검증**입니다. 실제 Qwen의 성능이나 논문 결론을 검증했다는 뜻이 아닙니다.

---

## 목차

1. [연구 질문과 가설](#1-연구-질문과-가설)
2. [실험 설계 한눈에 보기](#2-실험-설계-한눈에-보기)
3. [과제 세계: 4개 분야 × 5단계 작업 흐름](#3-과제-세계-4개-분야--5단계-작업-흐름)
4. [Horizon: 남은 단계 수](#4-horizon-남은-단계-수)
5. [실험 조건 4가지](#5-실험-조건-4가지)
6. [오류 종류 2가지와 복구 비용](#6-오류-종류-2가지와-복구-비용)
7. [모델이 실제로 받는 입력](#7-모델이-실제로-받는-입력)
8. [채점 규칙](#8-채점-규칙)
9. [지표 사전](#9-지표-사전)
10. [통계 분석](#10-통계-분석)
11. [저장소 구조](#11-저장소-구조)
12. [환경 준비 (처음 한 번)](#12-환경-준비-처음-한-번)
13. [llama.cpp 준비](#13-llamacpp-준비)
14. [모델 준비와 양자화](#14-모델-준비와-양자화)
15. [실행 전 점검](#15-실행-전-점검)
16. [실험 실행 절차 (runbook)](#16-실험-실행-절차-runbook)
17. [결과 파일 구조](#17-결과-파일-구조)
18. [분석 실행과 산출물](#18-분석-실행과-산출물)
19. [결과 해석과 보고서 작성 가이드](#19-결과-해석과-보고서-작성-가이드)
20. [팀 협업과 결과 공유](#20-팀-협업과-결과-공유)
21. [데이터셋 수정 방법](#21-데이터셋-수정-방법)
22. [문제 해결 FAQ](#22-문제-해결-faq)
23. [변경 이력](#23-변경-이력)
24. [알려진 한계](#24-알려진-한계)

---

## 1. 연구 질문과 가설

선행연구에서 "양자화된 Agent는 단계가 길어질수록 오류가 증폭된다"는 주장이 있었습니다. 이 저장소는 그 주장을 **같은 상태에서 재개하는 통제 실험**으로 다시 확인하고, 두 가지 방향으로 확장합니다.

1. **오류 발생과 오류 회복을 분리**합니다. 최종 성공률이 낮아진 이유가 "실수를 더 많이 해서"인지 "실수를 덜 고쳐서"인지 구분합니다.
2. **조용한 오류**(도구가 성공을 반환해 겉으로는 문제없어 보이는 잘못된 저장)를 다룹니다. 실제 Agent에서 가장 위험한 형태입니다.

| 가설 | 내용 | 검정에 쓰는 결과 |
|---|---|---|
| H1 증폭 | 양자화 모델의 성공률 손실(모델 − FP16)은 남은 단계가 많을수록(h1 → h3 → h5) 커진다 | `horizon_interactions` (정밀도 × horizon 상호작용) |
| H2 발생 | 양자화할수록 자연 발생 오류(`natural_error_rate`)가 늘고, 첫 오류 위치가 앞당겨진다 | `free` 조건 요약, `success_differences` |
| H3 회복 | 같은 오류가 주어졌을 때 양자화 모델의 회복 손실(normal − error)이 더 크다 | `recovery_penalty_differences` |
| H4 신호 | 오류가 있다는 사실만 알려줘도 회복률이 오르며, 그 이득은 정밀도마다 다르다 | `error_restart` vs `signaled_error_restart` |
| H5 효율 | 양자화는 모델 크기·메모리·토큰 속도를 개선한다. 위 신뢰성 손실과 함께 trade-off로 보고한다 | `model_storage_mb`, 토큰/초, 메모리 |

---

## 2. 실험 설계 한눈에 보기

```text
60개 기본 시나리오 (4개 분야 × 15)
 ├─ execution 과제 (조건: free)            h1, h3, h5   → 180 task
 └─ recovery fixture (h1, h3)                            → 120 task
       ├─ normal_restart                  정상 기록에서 재개              120회
       ├─ error_restart          × 오류 2종  조용한 오류 뒤 재개          240회
       └─ signaled_error_restart × 오류 2종  같은 오류 + 감사 메시지 1줄   240회
                                              free 180회 포함 모델당 합계 780회
```

비교 축은 네 가지이고, 나머지는 모두 고정합니다.

| 축 | 수준 | 고정되는 것 |
|---|---|---|
| 정밀도 | FP16 / Q8_0 / Q4_K_M (선택: Q6_K 등 추가, KV cache 양자화) | 같은 원본 GGUF, 같은 서버 빌드, 같은 장비 |
| Horizon | 남은 필수 호출 1 / 3 / 5 | 같은 질문·후보·정답 |
| 조건 | free / normal_restart / error_restart / signaled_error_restart | 같은 checkpoint 상태·도구·예산 |
| 오류 종류 | violating_candidate / wrong_option | 같은 checkpoint, 같은 요청 ID |

고정 추론 설정: temperature 0, seed 42, 응답당 최대 512 토큰, thinking 끔, 한 응답에 도구 호출 1개, context 8192, slot 1.

---

## 3. 과제 세계: 4개 분야 × 5단계 작업 흐름

모든 과제는 결정론적인 가짜(mock) 도구 환경에서 실행됩니다. 4개 분야 모두 **같은 5단계 구조**이고 도구 이름만 다릅니다.

| 단계 | 쇼핑 | 일정 | 이동 | 장소 | 입력 | 판단 필요 |
|---:|---|---|---|---|---|---|
| 1 | `read_purchase_request` | `read_schedule_request` | `read_travel_request` | `read_visit_request` | 질문에 있는 `request_id` | 없음 |
| 2 | `find_product_offers` | `find_available_slots` | `find_travel_options` | `find_place_offers` | 1의 `context_id` | 없음 |
| 3 | `get_product_details` | `get_slot_details` | `get_fare_details` | `get_visit_details` | 2의 `offer_id` | **있음**: 조건·순위로 후보 선택 |
| 4 | `prepare_purchase` | `prepare_event` | `prepare_itinerary` | `prepare_visit` | 3의 `detail_id` + `option_id` | **있음**: 옵션 종류 선택 |
| 5 | `confirm_purchase` | `confirm_event` | `confirm_itinerary` | `confirm_visit` | 4의 `draft_id` | 없음 |

보조 도구 2개가 항상 함께 노출됩니다.

- `discard_saved_result(request_id)`: 저장된 결과 하나 삭제
- `inspect_state()`: 저장된 최종 결과만 읽기(상태를 바꾸지 않음)

**리소스 의존성.** 각 도구는 앞 도구가 실제로 만든 리소스 ID만 받습니다. 정답 ID를 추측해 넣어도 그 리소스가 아직 만들어지지 않았으면 `*_not_found`로 거절됩니다. 그래서 단계를 건너뛸 수 없고, 정답 순서를 문자열로 비교하는 채점도 필요 없습니다. 추가 조회나 다른 후보 탐색은 허용됩니다.

### 시나리오 설계 규칙

각 시나리오에는 후보 3개와 후보별 옵션 2개가 있습니다.

- **적격 후보 2개 + 조건 위반 후보 1개.** 위반 후보는 모든 시나리오에서 **조건을 무시하면 순위 1등**입니다. 그래서 이 후보를 고르는 것은 "조건 확인을 빠뜨린" 그럴듯한 실수입니다.
- 위반되는 조건은 시나리오마다 다릅니다(가격 초과, 평점 미달, 인원 부족, 시작 시각이 너무 이름, 도보 시간 초과 등).
- 일부 시나리오는 **1순위 값이 동점**이라 명시된 2순위 기준을 적용해야 정답이 나옵니다.
- **위치 균형.** 분야마다 정답 후보·위반 후보·요청 옵션·h1 정답 draft의 목록 위치가 고르게 분포합니다(개수 차이 1 이하). 위치만으로 정답을 맞힐 수 없습니다.
- 모든 숫자는 소수 1자리 이하로 정리되어 `4.1000000000000005` 같은 표기가 모델 입력에 들어가지 않습니다.

### 예시: `shopping_001`

요청: "키보드 중 가격 100 이하, 평점 4.0 이상. 평점 최고를 고르고 동점이면 가격이 낮은 것. 일반 배송(`standard_delivery`)."

| 후보 | 가격 | 평점 | 판정 |
|---|---:|---:|---|
| Keyboard candidate A | 120 | 4.9 | 가격 초과 → **위반 후보**(조건 무시 시 1등) |
| Keyboard candidate B | 85 | 4.8 | **정답** |
| Keyboard candidate C | 70 | 4.4 | 적격이지만 평점 낮음 |

정답은 "B + standard_delivery"를 확정해 저장한 상태입니다. 금액은 가상 단위이고, `start_hour`·`depart_hour`는 그날의 24시간제 정수 시각입니다.

---

## 4. Horizon: 남은 단계 수

`horizon`은 시작 상태에서 완료까지 **남아 있는 필수 도구 호출 수**입니다. 미리 제공된 결과와 종료 응답은 포함하지 않습니다.

| Horizon | 시작할 때 이미 있는 정보 | 모델이 해야 할 일 |
|---:|---|---|
| 5 | 질문만 | 요청 조회 → 후보 조회 → 상세 → draft → 확정 |
| 3 | 요청 결과, 후보 목록 | 상세 → draft → 확정 |
| 1 | 요청, 후보, 상세 2개, **draft 3개**(정답 1, 같은 후보·다른 옵션 1, 위반 후보 1) | 3개 중 맞는 draft를 골라 확정 |

h1에도 선택이 남아 있도록 오답 draft 2개를 함께 줍니다. 그렇지 않으면 h1은 "하나뿐인 draft를 확정"하는 의미 없는 과제가 됩니다.

**해석상 주의.** 판단 지점(후보 선택, 옵션 선택)은 h3와 h5에 똑같이 있습니다. h5에 추가된 두 호출(요청 조회, 후보 조회)은 판단 없는 ID 전달입니다. 따라서 h3와 h5의 차이는 "추론 단계 증가"가 아니라 **"대화 길이와 ID 전달 증가"의 효과**로 보고합니다. 또 horizon이 달라지면 시작 정보량도 함께 달라지므로, "1-step 과제 대 5-step 과제"가 아니라 "올바른 정보가 주어진 지점에서 남은 단계 수"의 효과로 표현합니다.

---

## 5. 실험 조건 4가지

| 조건 | 과제 | 앞 단계 정보 전달 방식 | 오류 | 측정 대상 |
|---|---|---|---|---|
| `free` | execution 180 | **사용자 메시지**에 JSON으로 첨부 (h5는 없음) | 없음 | 자연 발생 오류, 성공률, 첫 오류 위치 |
| `normal_restart` | recovery 120 | **모델 자신의 과거 도구 호출** 기록 | 없음 | 같은 정상 상태에서의 다음 행동·성공률(회복 손실의 기준점) |
| `error_restart` | recovery 120 × 오류 2종 | 위 기록 + 잘못된 결과를 확정한 기록 | **조용한 오류** | 신호 없이 스스로 발견·수정하는지 |
| `signaled_error_restart` | recovery 120 × 오류 2종 | `error_restart`와 같은 기록 + 감사 메시지 1줄 | 신호 있는 오류 | 오류가 있다는 사실만 알려줬을 때 고치는지 |

- `free`와 `normal_restart`는 **같은 정보**(checkpoint 상태가 동일)를 **다른 방식**으로 받습니다. 검증기가 두 조건의 상태가 같고 모델 입력은 다른지 확인합니다.
- `error_restart`의 도구 결과는 모두 `"ok": true`입니다. 모델은 저장된 결과를 직접 다시 읽고 요청 조건과 비교해야만 오류를 알 수 있습니다.
- 감사 메시지는 다음 한 줄이며, 후보·옵션·조건·정답을 전혀 언급하지 않습니다.

  ```text
  Automated audit notice: the result currently saved for this request does not satisfy the request. Continue working on the original request.
  ```

- `--condition recovery`는 `error_restart`의 옛 이름(별칭)이고, 결과에는 `error_restart`로 기록됩니다.
- **h5에는 회복 fixture가 없습니다.** h5 checkpoint에는 이전 기록이 없어 `normal_restart`가 `free` h5와 완전히 같고, 잘못된 전체 경로를 주입하면 h3 오류 분기와 같은 상황이 되기 때문입니다. h5는 `free`로만 분석합니다.

---

## 6. 오류 종류 2가지와 복구 비용

두 종류 모두 "같은 요청 ID로 잘못된 결과가 저장된 상태"(`wrong_saved_result`)입니다. `free`에서 모델이 스스로 낼 수 있는 실수와 같은 종류라서, 주입 실험의 회복률과 `free`에서 스스로 고친 비율(`wrong_save_recovery_rate`)을 나란히 비교할 수 있습니다.

| 오류 종류 | 무엇이 저장됐나 | h1 주입 기록 | h3 주입 기록 | h1 최소 복구 | h3 최소 복구 |
|---|---|---|---|---:|---:|
| `violating_candidate` | 조건 위반 후보 + 요청 옵션 | draft 3개 중 위반 draft 확정 | 위반 후보 상세 → draft → 확정 | 1 | 3 (정답 상세 → draft → 확정) |
| `wrong_option` | **정답 후보** + 다른 옵션 종류 | draft 3개 중 옵션 오류 draft 확정 | 정답 후보 상세 → 다른 옵션 draft → 확정 | 1 | 2 (요청 옵션 draft → 확정) |

- 정상 분기의 최소 호출은 항상 horizon(h1 1, h3 3)입니다.
- `violating_candidate`는 복구 비용이 정상 작업량과 같아 두 분기의 예산 여유가 같습니다.
- `wrong_option`의 h3 복구는 정답 상세가 이미 기록에 있어 2회면 됩니다. 그래서 **회복 결과는 오류 종류별로 따로 보고**하고, 두 종류를 합쳐 horizon끼리 비교하지 않습니다.
- 복구 방법은 제한하지 않습니다. 같은 요청으로 올바른 draft를 다시 확정하면 잘못된 결과가 교체되고, `discard_saved_result`로 지운 뒤 확정해도 됩니다. 처음부터 정상 경로를 다시 밟아도 성공입니다.
- h1 checkpoint에는 `free` h1과 같은 draft 3개가 있어 정답 draft ID가 기록에 보입니다. 이것은 모든 조건이 같은 정보를 갖게 하려는 선택이며, 정답만 따로 드러나지는 않습니다.

### 예시: `recovery_shopping_001_h3`, `wrong_option`, `signaled_error_restart`

```text
[USER]  Request ID: request_97c9ac5f82d4. Select the product and delivery option ... (원래 질문)
[CALL]  read_purchase_request {"request_id": "request_97c9ac5f82d4"}           ← 정상 기록 (모델 자신의 호출로 제시)
  ->    {"ok": true, "context_id": "context_9447...", "request": {...}}
[CALL]  find_product_offers {"context_id": "context_9447..."}                  ← 정상 기록
  ->    {"ok": true, "offers": [B(85, 4.8), A(120, 4.9), C(70, 4.4)]}
[CALL]  get_product_details {"offer_id": "offer_ae54..."}                       ← 주입: 정답 후보 B
[CALL]  prepare_purchase {"detail_id": "detail_9c4b...", "option_id": "option_e70b..."}   ← 주입: express_delivery (틀림)
[CALL]  confirm_purchase {"draft_id": "draft_d1f0..."}                          ← 주입: 확정, "ok": true
[USER]  Automated audit notice: the result currently saved ... does not satisfy the request. ...
--- 모델 차례 ---
최소 복구: prepare_purchase(detail_9c4b..., standard_delivery 옵션) → confirm_purchase(새 draft)
```

`error_restart`는 마지막 `[USER]` 감사 메시지가 없는 것만 다르고 나머지는 글자 하나까지 같습니다.

---

## 7. 모델이 실제로 받는 입력

### 시스템 프롬프트 (모든 조건·정밀도 공통)

```text
You are a tool-using Agent. Use the available tools to complete the user request. Inspect the current state when needed.
There is no finish tool: when the requested final state has been achieved, reply with a short final answer that contains
no tool call. A reply without a tool call ends the task. Use English for all responses. Return at most one tool call per
response; multiple calls in one response are rejected without executing any of them. Invalid output receives only generic
format feedback and consumes a response turn. Resource IDs must come from the user context or tool results. Apply the
structured request constraints and ranking rules, including tie-breakers. Prepared drafts are not final until confirmed.
```

### 질문 (원래 질문은 모든 조건에서 동일)

```text
Request ID: request_97c9ac5f82d4. Select the product and delivery option that satisfy the purchase request, then confirm
and save the purchase item. Continue from any information or tool results already provided. When finished, the entire
saved result set must contain only one record: the correct result for this request.
```

`free` h1·h3에서는 이 질문 아래에 `Information already available from earlier lookups for this request (tool name and its result, JSON):`라는 머리말과 앞 단계 도구 결과 JSON이 붙습니다. 질문에는 정답 후보·옵션·draft ID가 없으며, 검증기가 이를 확인합니다.

### 응답 처리 규칙

| 모델 응답 | 처리 | 기록 |
|---|---|---|
| 도구 호출 1개 | 실행하고 결과를 대화에 추가 | `tool_executed=true` |
| 도구 호출 2개 이상 | 하나도 실행하지 않고 거절 | `model_error=multiple_tool_calls` |
| 인자가 JSON 객체가 아님 | 거절 | `invalid_arguments` |
| `finish_reason=length`(512토큰 잘림) | 실행 전 거절 | `output_truncated` |
| 툴콜처럼 보이는 텍스트(`<tool_call>`, `{"name":…,"arguments":…}`) | **종료가 아니라** 형식 오류로 거절 | `unparsed_tool_call` |
| 서버가 모델 출력을 파싱하지 못한 HTTP 500(`failed to parse input`) | 형식 오류로 거절 | `server_output_parse_failure` |
| 그 밖의 도구 호출 없는 텍스트 | **종료** | 목표 미달이면 `premature_finish` |

거절된 응답 뒤에는 모델이 실제로 낸 텍스트와 함께 고정 안내문이 대화에 들어갑니다. 안내문은 정답이나 고친 인자를 절대 알려주지 않으며, 거절된 응답도 응답 턴 1개를 소모합니다.

```text
Your last response could not be used (<종류>). Return exactly one tool call with a JSON object for its arguments,
or a final answer if the task is complete. No action from that response was executed.
```

### 응답 턴 예산

- 기본 예산 = **정상 최소 호출(horizon) + `--recovery-budget`(기본 3)**. 모든 조건에 같은 규칙을 적용합니다(`free` h5는 8턴, 회복 h3는 6턴).
- 종료 응답과 형식 오류 응답도 1턴을 씁니다.
- `--max-calls N`을 주면 모든 과제에 고정 상한 N을 씁니다. 이 경우 모든 조건·정밀도에 같은 값을 줘야 합니다.
- 미리 제공한 기록과 주입한 오류 호출은 모델 호출·오류 수·시간·토큰에 포함하지 않습니다.

---

## 8. 채점 규칙

### 성공 판정

성공은 **최종 저장 상태**만 봅니다. 마지막 도구 이름이나 `last_result`는 사용하지 않습니다.

- 저장된 결과 전체(`completed_requests`)가 **정답 결과 하나와 정확히 같아야** 성공입니다.
- 정답을 저장한 뒤 `inspect_state`로 확인해도 성공이 유지됩니다.
- 정답을 저장했다가 다시 망가뜨리면 실패입니다(`recovery_completion_step`에는 처음 도달한 시점이 남습니다).

### 오류 분류

한 응답에 오류 표지가 여러 개 있어도 1회로 셉니다. 첫 오류의 종류(`first_error_kind`)는 아래 접두사 중 하나입니다.

| 분류 | 예 | 의미 |
|---|---|---|
| `model_output:*` | `multiple_tool_calls`, `invalid_arguments`, `output_truncated`, `unparsed_tool_call`, `server_output_parse_failure`, `invalid_action` | 형식을 지키지 못한 응답 |
| `tool_error:*` | `unknown_tool`, `invalid_arguments`, `offer_not_found`, `draft_not_found` 등 | 도구가 거절한 호출 |
| `semantic:*` | `saved_result_violates_request`, `removed_requested_result`, `premature_finish` | 실행은 됐지만 요청과 어긋난 행동 |

**ID 복사 오류와 선택 오류를 분리합니다.**

- **ID 복사 오류**(`identifier_error_*`): 앞 결과가 발급하지 않은 ID를 쓴 경우입니다. 도구 오류 `unknown_request`, `context_not_found`, `offer_not_found`, `detail_not_found`, `unknown_option`, `draft_not_found`가 여기에 해당합니다. 16자리 해시 ID를 여러 번 옮겨 적어야 하므로 긴 문자열 복사 능력과 관련됩니다.
- **선택 오류**(`wrong_save_*`): 존재하는 ID지만 틀린 후보·옵션을 저장한 경우(`saved_result_violates_request`)입니다. 판단 능력과 관련됩니다.

이 둘을 나눠야 "Q4가 판단을 더 틀린다"와 "Q4가 긴 ID를 더 잘못 옮긴다"를 구분할 수 있습니다.

다음은 오류가 아닙니다: 추가 조회, 다른 후보 탐색, `inspect_state`, 삭제 순서의 차이.

### 인프라 장애는 분모에서 제외

| 상황 | 분류 | 처리 |
|---|---|---|
| HTTP 오류, 타임아웃, 연결 실패, 서버 응답 JSON 형식 오류 | `infrastructure_error` | 실행 중단, 행동 지표 분모에서 제외 |
| 예상하지 못한 코드 예외 | `internal_error` | 같음 |
| 단, HTTP 500 본문이 `failed to parse input` | 모델 출력 오류 | **제외하지 않음** (형식을 더 자주 깨는 정밀도가 분모에서 빠져 유리해지는 것을 막기 위함) |

짝지은 비교에서는 한쪽이라도 장애가 있으면 그 쌍을 제외하고 제외 개수를 보고합니다. 제외가 많거나 특정 정밀도에 몰리면 서버를 먼저 점검하고 해당 과제를 다시 수집합니다. 자동 재시도는 없습니다.

---

## 9. 지표 사전

### 과제 단위 점수 (`score`, 결과 JSONL 한 줄마다)

| 지표 | 정의 |
|---|---|
| `strict_success` | 최종 저장 상태가 정답과 정확히 같고 인프라 장애가 없음 |
| `valid_for_behavior_metrics` | 인프라/코드 장애가 없어 행동 지표에 포함됨 |
| `termination_reason` | `finished` / `budget_exhausted` / `infrastructure_error` / `internal_error` |
| `natural_error` | (`free`만) 오류를 한 번 이상 냈는지 |
| `natural_error_recovered` | (`free`만) 오류를 냈지만 최종 성공했는지 |
| `error_count`, `error_turn_rate` | 오류 응답 수, 전체 응답 대비 비율 |
| `model_output_error_count` | 형식 오류 응답 수 (7절 표의 모든 형식 오류) |
| `tool_error_action_count`, `tool_action_error_rate` | 실제 실행된 호출 중 거절·잘못된 상태 변경 수와 비율 |
| `identifier_error_count`, `identifier_error_episode` | ID 복사 오류 수, 한 번이라도 있었는지 |
| `wrong_save_count`, `wrong_save_episode`, `wrong_save_recovered` | 요청 위반 결과 저장 수, 있었는지, 그 뒤 고쳐서 성공했는지 |
| `first_error_turn` (=`first_error_step`) | 재개 후 1부터 센 첫 오류 응답 턴 |
| `first_error_kind`, `first_error_tool` | 첫 오류의 분류와 도구 |
| `first_error_workflow_stage` | 첫 오류가 5단계 중 몇 번째 단계 도구에서 났는지(1~5, 보조 도구면 null). horizon이 달라도 비교 가능 |
| `next_action_correct` | 첫 응답이 기준 경로의 첫 호출(도구·인자 모두)과 같은지 |
| `next_action_valid` | 첫 응답이 오류 없이 실행된 도구 호출인지(다른 유효한 조회 포함) |
| `response_turn_count` | 모델 응답 수(종료·형식 오류 포함) |
| `attempted_tool_call_count` | 모델이 낸 도구 호출 시도 수(거절된 다중 호출 포함) |
| `executed_tool_call_count` (=`tool_call_count`) | 실제 도구 실행기에 전달된 호출 수 |
| `extra_calls_over_normal` | 실행 호출 − 정상 최소 호출 |
| `extra_calls_over_oracle` | 실행 호출 − 해당 분기의 최소 호출(오류 분기는 최소 복구 호출) |
| `budget_exhausted` | 응답 턴 예산을 모두 씀 |

### 회복 전용 점수 (오류 분기만)

| 지표 | 정의 |
|---|---|
| `error_kind` | 주입된 오류 종류 |
| `recovery_success` | 잘못된 저장을 바로잡고 원래 목표 달성 |
| `recovery_error_resolved` | 최종 상태에 주입된 잘못된 결과가 없음. 지우기만 하고 정답을 저장하지 않으면 true이면서 실패일 수 있음 |
| `recovery_first_resolution_turn` (=`recovery_resolution_step`) | 잘못된 결과가 처음 사라진 응답 턴 |
| `recovery_stable_resolution_turn` | 마지막 재발 이후 제거가 유지되기 시작한 턴 |
| `recovery_completion_step` | 목표 상태에 처음 도달한 턴 |
| `recovery_error_count`, `recovery_first_error_step` | 재개 후 모델이 **새로** 낸 오류 수와 첫 위치(주입된 오류는 세지 않음) |
| `normal_remaining_calls`, `recovery_min_calls` | 정상 최소 호출, 최소 복구 호출 |

### 효율 점수

| 지표 | 정의 |
|---|---|
| `elapsed_ms` | 과제 전체 시간(워밍업 제외) |
| `ms_per_response_turn` | 응답 턴당 평균 시간 |
| `prompt_eval_tokens`, `prompt_eval_ms`, `generated_tokens`, `generation_ms` | llama.cpp 응답의 `timings` 합계 |
| `prompt_tokens_per_second`, `generation_tokens_per_second` | 위 값으로 계산한 처리량 |
| `prompt_tokens`, `completion_tokens` | 서버가 보고한 사용 토큰 |
| `peak_server_ram_mb` | `--server-pid`로 지정한 서버 프로세스의 최대 RSS(psutil 필요) |
| `peak_gpu_memory_mb` | `--gpu-index`를 준 경우 NVIDIA GPU **전체** 사용 메모리 최대값 |
| `max_ram_mb` | Python 실행기의 과제 종료 시점 RSS |

### 요약표 (`condition_summaries`, 모델 × 조건 × horizon × 오류 종류마다 한 행)

| 지표 | 정의 |
|---|---|
| `n`, `n_valid`, `n_runtime_errors` | 전체 / 행동 지표에 포함 / 장애로 제외된 실행 수 |
| `success_rate` | 유효 실행 중 성공 비율 |
| `natural_error_rate` | (`free`) 오류를 한 번이라도 낸 **과제의 비율**. 호출당 오류율이 아님 |
| `natural_error_recovery_rate` | (`free`) 오류를 낸 과제 중 결국 성공한 비율 |
| `wrong_save_episode_rate`, `wrong_save_recovery_rate` | 위반 결과를 저장한 과제 비율, 그중 스스로 고쳐 성공한 비율 |
| `identifier_error_episode_rate`, `mean_identifier_error_count` | ID 복사 오류가 있었던 과제 비율, 평균 횟수 |
| `behavior_error_episode_rate` | 오류가 한 번이라도 있었던 과제 비율(모든 조건) |
| `pooled_error_turn_rate`, `pooled_tool_action_error_rate` | 분자·분모를 과제 전체에서 합친 비율 |
| `first_error_stage_counts` | 첫 오류가 난 작업 단계(1~5)별 과제 수 |
| `next_action_accuracy`, `next_action_valid_rate` | 첫 응답 정확도 |
| `recovery_rate`, `recovery_error_resolved_rate` | (오류 분기) 회복 성공률, 잘못된 결과 제거율 |
| `mean_extra_calls_over_normal` / `_over_oracle` | **성공한 실행만** 평균한 추가 호출 |
| `mean_extra_calls_over_normal_all_runs` | 실패 포함 전체 평균(생존 편향 없는 보조 지표) |
| `mean_elapsed_ms` | 일찍 실패한 실행도 포함한 평균 시간. **정밀도 간 속도 비교에 쓰지 않음** |
| `mean_elapsed_ms_successful` | 성공한 실행만의 평균 시간 |
| `mean_ms_per_response_turn` | 응답 턴당 지연 |
| `pooled_prompt_tokens_per_second`, `pooled_generation_tokens_per_second` | 전체 토큰 합 ÷ 전체 시간 합 |
| `mean_peak_server_ram_mb`, `mean_peak_gpu_memory_mb` | 메모리 |
| `model_storage_mb`, `model_quantization` | `models/manifest.json`에서 찾은 모델 파일 크기와 양자화 형식 |

### 짝지은 회복 비교 (`recovery_comparison`, 모델 × horizon × 오류 분기 × 오류 종류마다 한 행)

| 지표 | 정의 |
|---|---|
| `normal_restart_success_rate` | 공통 정상 checkpoint에서의 성공률 |
| `error_restart_success_rate` (=`recovery_rate`) | 오류 분기 성공률 |
| `recovery_penalty` | 같은 fixture 쌍의 (정상 성공 − 오류 성공) 평균. **회복 손실** |
| `recovery_error_resolved_rate` | 잘못된 결과 제거율 |
| `immediate_finish_rate` | 오류 분기의 첫 응답이 곧바로 "완료"한 비율(조용한 오류를 전혀 의심하지 않은 정도) |
| `mean_extra_calls_over_normal` / `_over_oracle` | 성공한 오류 실행의 추가 호출(`n_successful_error_runs`와 함께) |
| `error_restart_budget_exhausted_rate`, `normal_restart_budget_exhausted_rate` | 예산 소진 비율 |
| `recovery_min_calls` | 이 행의 최소 복구 호출 |

장비·모델 hash·task hash·seed·추론 설정·예산·평가 버전·Agent 정책 버전·KV cache 형식이 다른 기록은 같은 쌍으로 묶지 않습니다. 같은 실행을 중복 입력하면 오류로 알립니다.

---

## 10. 통계 분석

### 분석 단위

60개 기본 시나리오가 독립 단위입니다. 같은 시나리오의 h1·h3·h5, 같은 fixture의 정상·오류 분기는 **짝지은 관측**이지 독립 표본이 아닙니다. 모든 비교는 같은 과제(또는 같은 시나리오)끼리 짝지어 계산합니다. temperature 0으로 과제당 한 번 실행하므로 과제 안의 샘플링 변동은 추정하지 않습니다.

### `precision_comparison`의 세 가지 비교 (모두 모델 − 기준 모델)

**① `success_differences`: 칸별 성공률 차이**

조건 × horizon(회복은 `조건:오류종류`) 칸마다, 같은 task ID끼리 짝지어 계산합니다.

- `success_difference`: 평균 차이
- `difference_ci95`: 과제를 재표집한 percentile bootstrap 95% 구간(2,000회, seed 0)
- `only_reference_succeeded`, `only_model_succeeded`: 엇갈린 쌍의 수
- `mcnemar_exact_p`: 엇갈린 쌍에 대한 양측 정확 McNemar 검정

**② `horizon_interactions`: 핵심 가설 H1 검정**

기본 시나리오마다 다음 값을 계산합니다.

```text
interaction = (모델 성공 − 기준 성공)@긴 horizon − (모델 성공 − 기준 성공)@짧은 horizon
```

- 평균이 **음수**면 양자화 손실이 긴 horizon에서 더 크다는 뜻(증폭)입니다.
- `gap_short`, `gap_long`: 각 horizon에서의 평균 성공률 차이
- `sign_flip_p`: 양측 정확 sign-flip 순열 검정. 시나리오별 값의 부호를 무작위로 뒤집은 분포와 비교합니다(0인 시나리오는 정보가 없어 제외). 정수 차이이므로 전체 분포를 정확히 계산합니다.
- 조합: `free`는 (1,3), (1,5), (3,5), 회복 조건은 (1,3)

**③ `recovery_penalty_differences`: 가설 H3 검정**

fixture마다 `(정상 − 오류)@모델 − (정상 − 오류)@기준`을 계산합니다. 양수면 양자화 모델의 회복 손실이 더 큽니다. 오류 분기(조용한/신호)와 오류 종류별로 따로 계산하고, 같은 sign-flip 검정을 씁니다.

### 다중 비교 보정

세 묶음 각각 안에서 Holm-Bonferroni 보정 p값(`mcnemar_holm_p`, `sign_flip_holm_p`)을 함께 줍니다. 논문에는 보정된 p값을 쓰고, 사전에 정한 주 검정과 탐색적 검정을 구분합니다.

**권장 주 검정(실험 전에 팀이 확정):**

1. `free`, Q4 vs FP16, (h1, h5) 상호작용 → H1
2. `error_restart:violating_candidate`, Q4 vs FP16, h3 회복 손실 차이 → H3

나머지는 탐색적 결과로 보고합니다.

### 검정력 감각

엇갈린 쌍이 모두 한 방향일 때 양측 정확 검정이 p < 0.05가 되려면 엇갈린 쌍이 최소 6개 필요합니다(2 × 0.5⁶ ≈ 0.031). 60개 짝이면 약 10%p 차이부터 잡힐 수 있지만, 엇갈림이 양방향으로 섞이면 더 큰 차이가 필요합니다. FP16과 Q8의 차이는 작을 가능성이 높으므로 "유의한 차이 없음"을 "차이 없음"으로 쓰지 말고 신뢰구간으로 보고합니다.

### 혼합 장비 방지

정밀도 비교는 모든 라벨의 다음 값이 같아야 실행됩니다: hostname, OS, CPU, machine, seed, 추론 설정, 설정 잠금 hash, 서버 설정 hash, 데이터 버전·hash, 지표·정책 버전, 예산, 워밍업 설정. 모델 경로·hash와 KV cache 형식은 비교 대상이라 제외합니다. 다르면 어느 필드가 다른지 알려주고 중단합니다. 꼭 필요할 때만 `--allow-mixed-configuration`을 주고, 보고서에 그 사실을 적습니다.

---

## 11. 저장소 구조

```text
ondevice/
├─ README.md
├─ requirements.txt                    # 표준 라이브러리만 필요 (psutil·matplotlib는 선택)
├─ .github/workflows/validate-dataset.yml  # push·PR 때 데이터 검증·생성기 일치·테스트·smoke 자동 실행
├─ models/
│  └─ manifest.json                    # 모델 파일 경로·크기·SHA-256·양자화 명령 (모델 파일은 Git에 올리지 않음)
├─ tasks/
│  ├─ tasks.json                       # 생성된 300개 task (직접 수정 금지)
│  ├─ EXECUTION_TASKS.md               # 생성된 정상 시나리오 카탈로그
│  └─ RECOVERY_TASKS.md                # 생성된 회복 설계·목록
├─ src/
│  ├─ dataset_spec.py                  # 데이터 버전·시나리오 수·horizon·오류 종류의 단일 정의
│  ├─ tasks.py                         # Task 구조, 로딩, 오류 종류 선택(with_error_kind)
│  ├─ dataset_validation.py            # JSON·형식·영어·정보 유출·의미·생성기 일치 통합 검증
│  ├─ execution_validation.py          # 정답 추론, 우회 차단, 위치 균형, 부동소수점 검사
│  ├─ recovery_validation.py           # checkpoint 일치, 오류 2종 독립 계산, 분기 입력 분리
│  ├─ execution_environment.py         # 5단계 리소스 도구, 조건별 대화 생성, 오류 주입, 감사 메시지
│  ├─ environment.py                   # 시스템 프롬프트, 구버전 도구(우회 차단용)
│  ├─ client.py                        # llama.cpp HTTP client, 응답 파싱, 형식 오류 분류
│  ├─ run_preparation.py               # 서버 /props 검사, 설정 잠금, 워밍업
│  ├─ agent.py                         # 공통 Agent 실행 루프, 형식 피드백, timings 수집
│  ├─ workflow_metrics.py              # 과제 단위 점수
│  ├─ metrics.py                       # 요약표 집계
│  ├─ recovery_comparison.py           # 정상/오류 분기 짝 비교
│  ├─ precision_comparison.py          # 정밀도 비교·상호작용·McNemar·sign-flip·Holm
│  └─ experiment.py                    # batch 실행, 메모리 측정, 메타데이터 기록
├─ scripts/
│  ├─ run_experiment.py                # 실제 llama.cpp 실행
│  ├─ analyze.py                       # 결과 집계, CSV, 그래프
│  ├─ quantize_models.py               # 같은 원본에서 Q8·Q4(+추가 단계) 생성, manifest 기록
│  ├─ validate_tasks.py                # 데이터 검증 보고서
│  ├─ smoke_test.py                    # 모델 없이 모든 분기(780개) 정답 재생
│  ├─ build_tasks.py                   # 두 생성기를 함께 실행·검사(--write / --check)
│  ├─ build_execution_tasks.py         # 60개 시나리오 정의
│  └─ build_recovery_tasks.py          # 회복 fixture(오류 2종) 정의
└─ tests/                              # 회귀 테스트 162개
```

Git에 올리지 않는 것: `models/*.gguf`, `.venv/`, `logs/`, `results/`.

---

## 12. 환경 준비 (처음 한 번)

필요한 것: **Python 3.11 이상**, Git, llama.cpp 빌드(13절).

### Windows (PowerShell)

```powershell
cd $env:USERPROFILE\ondevice
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1          # 실행 정책 오류가 나면: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -r requirements.txt
python -m pip install "psutil>=6.0,<8.0" "matplotlib>=3.8,<4.0"   # 선택: 서버 RAM 측정, 그래프
```

### Windows (CMD)

```cmd
cd /d "%USERPROFILE%\ondevice"
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### WSL / Linux / macOS

```bash
cd ~/ondevice
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

가상환경이 켜지면 명령줄 앞에 `(.venv)`가 붙습니다. 이후 모든 명령은 저장소 최상위 폴더에서 실행합니다. 경로는 슬래시(`/`)를 써도 Windows에서 그대로 동작합니다.

---

## 13. llama.cpp 준비

`llama-server`(또는 통합 실행 파일 `llama`)와 `llama-quantize`가 필요합니다. **세 정밀도 모두 같은 빌드, 같은 장비, 같은 옵션**으로 실행해야 비교가 공정합니다. 서버의 `build_info`가 설정 잠금에 기록되므로 빌드가 바뀌면 실행이 거부됩니다.

[llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases)에서 장비에 맞는 Windows x64 패키지를 받아 압축을 풉니다.

| 장비 | 패키지 | `--n-gpu-layers` |
|---|---|---|
| NVIDIA GPU | CUDA 빌드 (릴리스 설명의 CUDA DLL도 함께) | `99` (전체 GPU) |
| Intel Arc / 내장 GPU, AMD | Vulkan 빌드 | `99` (전체 GPU) |
| GPU 없음 | CPU 빌드 | `0` |

```powershell
# 예: C:\Users\<이름>\llama-bin-vk 에 풀었을 때
$env:PATH += ";$env:USERPROFILE\llama-bin-vk"
llama-server --version
llama-quantize --help
```

소스 빌드가 필요하면 [공식 빌드 문서](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)를 따릅니다(CUDA: `-DGGML_CUDA=ON`, Vulkan: `-DGGML_VULKAN=ON`). 빌드된 실행 파일은 보통 `build\bin\Release\`에 있습니다.

### 서버 실행 명령 (모든 정밀도 공통 옵션)

```text
llama-server -m models/Qwen3-1.7B-f16.gguf --jinja --reasoning off --host 127.0.0.1 --port 18080 --ctx-size 8192 --parallel 1 --batch-size 512 --ubatch-size 512 --cache-type-k f16 --cache-type-v f16 --flash-attn off --n-gpu-layers 99
```

| 옵션 | 이유 |
|---|---|
| `--jinja` | 모델 내장 chat template로 도구 호출을 처리 |
| `--reasoning off` | Qwen3의 thinking 끄기. 빌드가 이 옵션을 모르면 지워도 됩니다(요청마다 `enable_thinking=false`를 보냅니다) |
| `--ctx-size 8192 --parallel 1` | 실행기가 `/props`에서 context 8192, slot 1을 요구합니다 |
| `--cache-type-k/v f16`, `--flash-attn off` | 이번 비교의 공통 기준. 최적 속도 설정이라는 뜻은 아님 |
| `--n-gpu-layers` | 위 표 참고. FP16이 장비 메모리에 다 올라가지 않으면 세 정밀도 모두 같은 고정 레이어 수로 맞춥니다 |

통합 실행 파일 배포본은 `llama-server` 대신 `llama serve`로 같은 옵션을 씁니다. 서버 시작 로그에서 실제로 GPU에 올라간 레이어 수와 KV cache 크기를 확인하고, 로그를 결과와 함께 보관합니다.

---

## 14. 모델 준비와 양자화

### 원본 F16 GGUF 받기 (약 4GB)

```powershell
curl.exe -L -o models\Qwen3-1.7B-f16.gguf "https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-f16.gguf"
```

### 같은 원본에서 Q8·Q4 만들기

```text
python scripts/quantize_models.py --source models/Qwen3-1.7B-f16.gguf --output-dir models
```

```text
원본 F16 GGUF ─┬─ llama-quantize → Qwen3-1.7B-f16-Q8_0.gguf
               └─ llama-quantize → Qwen3-1.7B-f16-Q4_K_M.gguf
```

- Q8을 다시 Q4로 바꾸지 않습니다. 모든 단계를 **같은 원본**에서 독립적으로 만듭니다.
- `models/manifest.json`에 각 파일의 경로, 크기, SHA-256, 양자화 형식, 실행한 명령, llama.cpp 버전이 기록됩니다.
- 실행 파일이 PATH에 없으면 `--quantizer C:\...\llama-quantize.exe --llama C:\...\llama-server.exe`로 경로를 줍니다.
- 명령만 미리 보려면 `--dry-run`을 붙입니다.

### trade-off 곡선용 추가 단계 (선택)

```text
python scripts/quantize_models.py --source models/Qwen3-1.7B-f16.gguf --output-dir models --extra-types Q6_K,Q5_K_M,Q3_K_M --force
```

- 기존 manifest는 지우지 않고 **병합**합니다. 원본 hash가 다른 manifest에는 쓰지 않습니다.
- 이미 있는 파일을 다시 만들 때는 `--force`가 필요합니다(같은 원본·같은 명령이면 결과는 같습니다).
- 결과 라벨에 양자화 이름(`self-Q6_K`처럼)을 넣으면 분석이 manifest에서 파일 크기를 찾아 `model_storage_mb`에 넣습니다.

### 공식 배포 Q8·Q4와의 대조 (선택)

양자화 경로가 결과에 영향을 주는지 보려면 공식 파일(`ggml-org/Qwen3-1.7B-GGUF:Q8_0`, `:Q4_K_M`)도 실행해 `reference-Q8`, `reference-Q4` 라벨로 비교합니다. `--reference-q8`, `--reference-q4` 옵션으로 manifest에 기록할 수 있습니다. 공식 파일이 다른 체크포인트나 다른 형식에서 만들어졌다면 같은 정밀도의 순수 비교가 아니므로 따로 해석합니다.

---

## 15. 실행 전 점검

모델 없이 몇 분 안에 끝납니다. 하나라도 실패하면 실험을 시작하지 마세요.

```text
python scripts/validate_tasks.py --output results/dataset_validation.json
python scripts/build_tasks.py --check
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

| 명령 | 성공 표시 | 확인하는 것 |
|---|---|---|
| `validate_tasks.py` | `"valid": true` | 아래 검증 항목 전체 |
| `build_tasks.py --check` | `All 300 English tasks match ...` | `tasks.json`·카탈로그가 생성기와 일치 |
| `unittest` | `OK` (162개) | 결함 회귀 검사 |
| `smoke_test.py` | `SMOKE TEST PASSED` (실행 180/180, 회복 600/600) | 모든 분기의 정답 재생과 채점 |

### 데이터 검증 항목

**구성**
- 300개 task, execution 180 / recovery 120, 유형·horizon별 60개, 분야별 균형, 중복 ID 없음
- 모든 h1·h3 execution에 대응하는 recovery fixture, fixture마다 오류 2종(순서·라벨 일치)

**정답과 우회 차단**
- 요청의 조건과 순위 규칙만으로 유일한 정답을 독립 계산하고, 저장된 정답과 일치
- 180개 정답 경로 실행 성공, 완료 후 `inspect_state`에도 성공 유지
- 정답 경로의 모든 부분 경로, 인접 순서 교환, 구버전 장바구니·일정 도구로는 성공 불가
- 다른 후보·옵션으로 완료하면 실패
- h1의 오답 draft 2개를 확정하면 실패하고, 이후 정답 draft를 확정하면 성공

**설계 품질**
- 모든 시나리오에서 위반 후보가 조건 무시 시 **단독 1등**
- 분야마다 정답 후보·위반 후보·요청 옵션·h1 정답 draft 위치가 균형
- 모델 입력 숫자에 부동소수점 잔재 없음

**조건 분리**
- `free`는 앞 단계 결과를 사용자 메시지로만 받음
- `free`와 `normal_restart`의 checkpoint 상태가 같고 모델 입력은 다름
- 모든 조건·오류 종류의 모델 입력(호출 ID 이름을 무시하고 비교)이 서로 다름
- `signaled_error_restart` = `error_restart` + 감사 메시지 1줄과 정확히 같음, 감사 메시지에 정답 ID 없음

**회복**
- 주입 기록이 공개 규칙으로 독립 계산한 경로와 일치(위반 후보 / 정답 후보 + 다른 옵션)
- 주입 결과가 같은 요청 ID로 저장되고 요청을 위반하며, 모델 입력에 실제 호출·관측으로 보임
- 최소 복구 경로·삭제 후 재확정·정상 경로 재수행이 모두 성공, 복구 경로의 부분 생략과 잘못된 재확정은 실패
- 최소 복구 호출: `violating_candidate` = horizon, `wrong_option` = h1 1, h3 2

**형식과 언어**
- JSON 중복 키, NaN/Infinity, 잘못된 자료형(`1` 대신 `true`나 `"1"`) 거부
- 질문에 요청 ID 정확히 1회, 정답 후보·옵션·draft ID 유출 없음
- 모든 모델 입력(시스템·사용자 메시지, 도구 정의, 주입 관측, 정답 경로의 반환값)이 ASCII 영어

이 검사는 정해진 경로와 부분 경로를 확인하는 것이지, 가능한 모든 행동열을 전수 탐색한 것은 아닙니다. 영어 문장의 자연스러움은 사람이 검토해야 합니다.

---

## 16. 실험 실행 절차 (runbook)

### 16.1 실행 순서 개요

```text
[모델 하나마다]
서버 시작 → PID 확인 → (첫 모델만) 파일럿 → free → normal_restart → error_restart → signaled_error_restart → 서버 종료
[모든 모델이 끝나면]
분석 → 결과 공유
```

`run_experiment.py`는 매번 다음 순서로 진행합니다.

1. 300개 task 전체 검증(`--limit`을 줘도 전체 검증)
2. 서버 `/props` 조회 → context 8192, slot 1, chat template, build 확인
3. 설정 잠금 파일과 비교(없으면 새로 생성)
4. 과제와 무관한 더미 요청 2회로 워밍업(시간·토큰·메모리 지표에서 제외)
5. 과제 실행, 한 과제가 끝날 때마다 결과 한 줄을 바로 기록

### 16.2 서버 시작과 PID 확인

터미널 1:

```text
llama-server -m models/Qwen3-1.7B-f16.gguf --jinja --reasoning off --host 127.0.0.1 --port 18080 --ctx-size 8192 --parallel 1 --batch-size 512 --ubatch-size 512 --cache-type-k f16 --cache-type-v f16 --flash-attn off --n-gpu-layers 99
```

터미널 2에서 PID를 확인합니다. 서버를 다시 시작하면 PID가 바뀝니다.

```powershell
Get-Process llama* | Select-Object Id, ProcessName      # PowerShell
tasklist | findstr /I "llama"                           # CMD
pgrep -af llama                                         # WSL / macOS
```

### 16.3 첫 모델에서 파일럿 (반드시 먼저)

FP16으로 각 조건 몇 개만 돌려 **바닥·천장 효과**를 확인합니다.

```text
python scripts/run_experiment.py --model-label pilot-self-FP16 --condition free --horizon 5 --limit 8 --model-path models/Qwen3-1.7B-f16.gguf --server-pid <PID> --settings-lock results/pilot/settings.json --output results/pilot/fp16_free_h5.jsonl
python scripts/run_experiment.py --model-label pilot-self-FP16 --condition error_restart --horizon 3 --limit 4 --model-path models/Qwen3-1.7B-f16.gguf --server-pid <PID> --settings-lock results/pilot/settings.json --output results/pilot/fp16_error_h3.jsonl
python scripts/run_experiment.py --model-label pilot-self-FP16 --condition signaled_error_restart --horizon 3 --limit 4 --model-path models/Qwen3-1.7B-f16.gguf --server-pid <PID> --settings-lock results/pilot/settings.json --output results/pilot/fp16_signaled_h3.jsonl
python scripts/analyze.py results/pilot/fp16_free_h5.jsonl results/pilot/fp16_error_h3.jsonl results/pilot/fp16_signaled_h3.jsonl
```

- `--limit 4`인 오류 조건은 오류 2종을 모두 실행하므로 결과가 8행입니다.
- `--limit`은 데이터 순서(분야: 쇼핑 → 일정 → 이동 → 장소)대로 앞에서 자르므로, 위 명령은 쇼핑 과제만 고릅니다. 분야를 고루 보려면 `--task-id schedule_003_h5`처럼 골라 여러 번 실행합니다.

| 파일럿 결과 | 판단 |
|---|---|
| FP16 `free` h5 성공률이 매우 낮음(예: 0~10%) | 바닥 효과. 정밀도 차이가 안 보이므로 `--max-tokens`, 서버 옵션, chat template을 먼저 점검하고 팀과 과제 난이도를 논의 |
| FP16 `free` 전 horizon이 100% | 천장 효과 가능성. Q4까지 100%면 차이를 못 봄 |
| `error_restart` 회복률이 거의 0이고 첫 응답에서 바로 끝냄 | 조용한 오류를 전혀 의심하지 않음. 본실험에서 `signaled_error_restart`를 주 회복 지표로 보고 |
| `n_runtime_errors`가 0이 아님 | 서버·네트워크 문제. 해결 후 본실험 |
| `model_output_error_count`가 많음 | 툴콜 형식 문제. 결과 JSONL의 `raw_response`를 보고 chat template·`--jinja` 확인 |

파일럿 결과는 본실험과 **다른 설정 잠금·폴더**(`results/pilot/`)에 둡니다. 소수의 파일럿 결과로 결론을 내리지 않습니다.

### 16.4 본실험: 모델 하나 실행

PowerShell 예시입니다. `$PID`는 PowerShell 예약 변수이므로 `$serverPid`처럼 다른 이름을 씁니다.

```powershell
$label     = "self-FP16"
$model     = "models/Qwen3-1.7B-f16.gguf"
$serverPid = 12345                              # 16.2에서 확인한 값
$serverCmd = "llama-server -m $model --jinja --reasoning off --host 127.0.0.1 --port 18080 --ctx-size 8192 --parallel 1 --batch-size 512 --ubatch-size 512 --cache-type-k f16 --cache-type-v f16 --flash-attn off --n-gpu-layers 99"

foreach ($condition in "free", "normal_restart", "error_restart", "signaled_error_restart") {
    python scripts/run_experiment.py --model-label $label --condition $condition `
        --model-path $model --server-pid $serverPid --kv-cache-type f16 --server-command $serverCmd `
        --output "results/main/${label}_${condition}.jsonl"
    if ($LASTEXITCODE -ne 0) { Write-Host "중단: $condition"; break }
}
```

NVIDIA GPU라면 `--gpu-index 0`을 추가해 GPU 메모리를 기록합니다. 끝나면 서버를 `Ctrl+C`로 종료하고, 다음 모델 파일로 서버를 다시 띄운 뒤 `$label`, `$model`, `$serverPid`만 바꿔 같은 블록을 실행합니다.

| 라벨 | 모델 파일 |
|---|---|
| `self-FP16` | `models/Qwen3-1.7B-f16.gguf` |
| `self-Q8` | `models/Qwen3-1.7B-f16-Q8_0.gguf` |
| `self-Q4` | `models/Qwen3-1.7B-f16-Q4_K_M.gguf` |

**라벨 규칙.** 기준 모델 라벨에만 `FP16`이 들어가야 분석이 자동으로 기준을 찾습니다. 라벨에 양자화 이름(`Q8`, `Q4`, `Q4_K_M`, `Q6_K`)을 넣어야 파일 크기를 찾습니다.

### 16.5 나눠서 실행하기

한 번에 780회를 못 돌리면 나눠도 됩니다. 같은 설정 잠금을 쓰는 한 나중에 합쳐 분석할 수 있습니다.

```text
--horizon 1|3|5                      # 한 horizon만
--error-kind violating_candidate     # 오류 분기에서 한 종류만 (기본 all)
--task-id recovery_place_004_h3      # 과제 하나만
--limit N                            # 선택된 과제 중 앞에서 N개
```

- 나눠 실행할 때는 파일 이름을 겹치지 않게 지정합니다. 기존 결과·진단 파일은 **절대 덮어쓰지 않고** 오류로 멈춥니다.
- 같은 과제·조건을 두 번 실행한 결과를 함께 분석에 넣으면 "duplicate" 오류가 납니다. 재실행이 필요하면 이전 파일을 분석 입력에서 빼고, 왜 재실행했는지 기록합니다. 실패한 결과를 성공한 재실행으로 슬쩍 바꾸지 않습니다.

### 16.6 설정 잠금

첫 실행이 `results/experiment_settings.json`(기본 경로)에 모델과 무관한 공통 설정을 잠급니다. 추론 설정, 서버 build, chat template hash, context, slot, 기본 생성 파라미터, 장비 정보, 워밍업 정책이 들어갑니다. 이후 모든 정밀도·조건은 같은 잠금과 일치해야 실행됩니다.

- 다르다는 오류가 나면 **잠금 파일을 지워 우회하지 말고** 어느 필드가 다른지(오류 메시지에 나옴) 확인해 원래 설정으로 되돌립니다.
- 의도적으로 다른 실험(예: `--recovery-budget 6`, `--cache-prompt`, KV cache 양자화)을 할 때만 `--settings-lock results/<새시리즈>/settings.json`과 새 결과 폴더를 씁니다.

### 16.7 추가 실험 시리즈 (선택)

| 시리즈 | 바꾸는 것 | 목적 |
|---|---|---|
| 예산 민감도 | `--recovery-budget 6` | 예산이 부족해서 회복을 못 한 것인지 확인 |
| 배포형 지연 | `--cache-prompt` | KV cache 재사용 시 실제 응답 지연 |
| KV cache 양자화 | 서버 `--cache-type-k q8_0 --cache-type-v q8_0` (V 양자화는 `--flash-attn on`이 필요할 수 있음), 실행기 `--kv-cache-type q8_0`, 라벨 `self-Q4-kvq8` | 긴 대화에서 커지는 KV 메모리를 줄였을 때의 신뢰성 변화 |
| 양자화 단계 확장 | `--extra-types Q6_K,Q5_K_M,Q3_K_M` 모델 | 크기 대비 성공률 곡선 |

KV cache 시리즈는 먼저 기존 잠금으로 실행해 보고, 서버 설정 차이로 거부되면 오류 메시지의 차이가 KV·flash-attn 관련뿐인지 확인한 뒤 새 잠금을 씁니다. 이 경우 정밀도 비교에 `--allow-mixed-configuration`이 필요할 수 있으며, 보고서에 그 사실을 적습니다.

### 16.8 메모리 측정

| 상황 | 방법 | 주의 |
|---|---|---|
| 전체 GPU 배치, NVIDIA | `--gpu-index 0` → `peak_gpu_memory_mb` | Windows에서는 프로세스별 값이 어려워 **GPU 전체 사용량**. 측정 중 다른 GPU 프로그램 종료 |
| CPU 실행 | `--server-pid <PID>` → `peak_server_ram_mb` (psutil 필요) | llama.cpp는 모델을 mmap하므로 RSS가 작게 보일 수 있음. 서버를 `--no-mmap`으로 띄워 측정 |
| Intel Arc 등 내장 GPU | `nvidia-smi`가 없어 GPU 값은 `null` | 보고서에는 manifest의 모델 파일 크기 + 서버 시작 로그의 KV cache 크기를 1차 지표로 쓰고, 작업 관리자의 "공유 GPU 메모리"는 참고값으로만 |

메모리 측정은 워밍업이 끝난 뒤 과제 구간에만 동작합니다. 모델을 처음 불러오는 순간의 최대값(cold start)은 별도 실험입니다. 서로 다른 장비(GPU PC와 노트북)의 시간·메모리 수치는 합치지 않습니다.

---

## 17. 결과 파일 구조

한 번 실행하면 세 종류의 파일이 생깁니다.

| 파일 | 내용 |
|---|---|
| `results/.../<이름>.jsonl` | 과제 하나당 한 줄. 아래 구조 |
| `results/.../<이름>.preflight.json` | 서버 `/props` 원본, 실제 요청 설정, 워밍업 응답별 시간·토큰·상태. 실패한 준비도 기록 |
| `results/experiment_settings.json` (또는 지정한 잠금) | 공통 설정과 그 hash |

`--output`을 주지 않으면 결과는 `results/<라벨>_<조건>_<실행시각>.jsonl`에 저장됩니다.

JSONL 한 줄의 구조:

```text
{
  "metadata": { model_label, model_path, model_sha256, hostname, platform, processor, seed,
                condition, task_selection, inference_settings, server_configuration, llama_cpp_version,
                settings_lock_sha256, warmup, dataset_version, dataset_sha256, kv_cache_type,
                server_command, error_kind_selection, metrics_version, agent_policy_version, ... },
  "task":     { id, base_id, domain, horizon, query, initial_actions, success_condition,
                error_kind, error_injection, recovery_actions, recovery_min_calls, ... },
  "trace":    { condition, initial_messages, messages, events[...], final_state,
                injected_actions, error_kind, error_signal, checkpoint_state_hash,
                server_timings, termination_reason, max_calls, ... },
  "score":    { 9절의 과제 단위 점수 }
}
```

`trace.events`에는 응답마다 모델 원문(`raw_response`), 파싱된 행동, 도구 결과, 오류 표지, 실행 후 상태 hash, 그 시점 성공 여부가 들어갑니다. 특정 과제가 왜 실패했는지 보려면 이 배열을 읽으면 됩니다.

---

## 18. 분석 실행과 산출물

```text
python scripts/analyze.py results/main/self-FP16_free.jsonl results/main/self-FP16_normal_restart.jsonl ... --output results/main/summary.json --csv-output results/main/summary.csv --plot-dir results/main/plots
```

PowerShell에서는 와일드카드를 직접 넘기지 못하므로 파일 목록을 펼쳐서 넘깁니다.

```powershell
python scripts/analyze.py (Get-ChildItem results/main/*.jsonl).FullName --output results/main/summary.json --csv-output results/main/summary.csv --plot-dir results/main/plots
```

- 여러 모델 결과를 한 번에 넣으면 정밀도 비교까지 계산합니다. 기준은 라벨에 `FP16`이 들어간 유일한 라벨이고, 아니면 `--reference-label self-FP16`처럼 지정합니다.
- 같은 라벨·프로토콜 안에 다른 장비·설정이 섞이면 분석을 중단합니다. 장비가 다르면 라벨을 구분해 따로 분석합니다.

### 산출물

| 파일 | 내용 |
|---|---|
| `summary.json` | 아래 전부 |
| `summary.csv` | `condition_summaries` 요약표 |
| `summary_paired.csv` | `recovery_comparison` 짝 비교 |
| `summary_precision.csv` | `success_differences` |
| `summary_interaction.csv` | `horizon_interactions` (핵심 가설) |
| `summary_penalty.csv` | `recovery_penalty_differences` |
| `plots/*.png` | 아래 그래프(matplotlib 필요) |

`summary.json`의 최상위 키: `condition_summaries`, `recovery_comparison`, `recovery_pairing`(짝 진단: 짝 없는 기록, 제외된 쌍, 누락 metadata), `precision_comparison`.

### 그래프 (x축은 모두 horizon, 선은 모델)

| 파일 | 지표 |
|---|---|
| `success_rate_free_by_horizon.png` | `free` 성공률 |
| `natural_error_rate_free_by_horizon.png` | 자연 오류율 |
| `error_turn_rate_free_by_horizon.png`, `tool_action_error_rate_free_by_horizon.png` | 응답당·실행 호출당 오류율 |
| `wrong_save_recovery_free_by_horizon.png` | `free`에서 스스로 고친 비율 |
| `normal_restart_success_by_horizon.png`, `next_action_accuracy_normal_by_horizon.png` | 정상 재개 기준점 |
| `recovery_rate_by_horizon.png` | 조용한 오류 회복률(오류 종류별 선, `wrong_option`은 점선) |
| `signaled_recovery_rate_by_horizon.png` | 신호 있는 오류 회복률 |
| `error_turn_rate_recovery_by_horizon.png` | 회복 중 오류율 |
| `elapsed_ms_successful_free_by_horizon.png`, `ms_per_turn_free_by_horizon.png` | 시간 |
| `generation_tps_free_by_horizon.png`, `prompt_tps_free_by_horizon.png` | 토큰/초 |
| `server_ram_free_by_horizon.png`, `gpu_memory_free_by_horizon.png` | 메모리 |

값이 수집되지 않은 지표의 그래프는 만들지 않습니다.

---

## 19. 결과 해석과 보고서 작성 가이드

### 결과 표 구성 예

| 표 | 행 | 열 |
|---|---|---|
| 표 1 효율 | FP16/Q8/Q4 | 파일 크기, 토큰/초(prompt·생성), 턴당 지연, 메모리 |
| 표 2 실행 | 정밀도 × horizon | `free` 성공률, 자연 오류율, ID 복사 오류율, 선택 오류율, 첫 오류 단계 분포 |
| 표 3 증폭 | 정밀도 | `horizon_interactions`의 (1,5) interaction, 95% CI, Holm p |
| 표 4 회복 | 정밀도 × horizon × 오류 종류 | 정상 성공률, 조용한 오류 회복률, 신호 있는 오류 회복률, 회복 손실, `immediate_finish_rate` |
| 그림 | 파일 크기(x) vs 성공률(y) | trade-off 곡선 (추가 양자화 단계가 있으면 점이 늘어남) |

### 이렇게 쓰고, 이렇게 쓰지 않습니다

| 써도 되는 표현 | 쓰면 안 되는 표현 |
|---|---|
| "올바른 정보가 주어진 지점에서 남은 단계가 많을수록 Q4의 손실이 X%p 더 컸다" | "Q4는 5-step 과제에서 X% 더 나쁘다"(시작 정보량도 다름) |
| "h3→h5 차이는 대화 길이와 ID 전달 증가의 효과" | "h5는 추론 단계가 더 많다" |
| "통제된 4개 분야, 5단계 템플릿, 2가지 저장 오류에 대한 결과" | "양자화 Agent 일반에 대해" |
| "유의한 차이를 찾지 못했다(95% CI [a, b])" | "차이가 없다" |
| "60개 기본 시나리오(짝지은 관측)" | "180개 독립 표본" |
| "Q4는 Q4_K_M(일부 텐서는 더 높은 정밀도를 쓰는 혼합 양자화)" | "4비트 모델" |

### 오류 발생과 회복을 연결해 읽기

- `free`의 `wrong_save_recovery_rate`(스스로 낸 실수를 고친 비율)와 `error_restart:violating_candidate`의 회복률(주입된 같은 실수를 고친 비율)을 나란히 놓으면, 주입 실험이 자연 상황을 얼마나 반영하는지 볼 수 있습니다.
- `free` 성공률에서 회복 성공률을 빼는 식의 비교는 하지 않습니다. 회복 손실은 반드시 같은 fixture의 `normal_restart`와 짝지어 계산합니다.
- `recovery_error_resolved`는 true인데 `recovery_success`가 false면 "잘못된 결과를 지우기만 하고 정답을 저장하지 않은" 경우입니다.
- `error_restart`와 `signaled_error_restart`의 회복률 차이는 "오류를 발견하는 능력"과 "발견한 뒤 고치는 능력"을 나눠 보여줍니다. 신호가 있어도 못 고치면 발견이 아니라 수정 단계가 문제입니다.

### 교수 요구사항(수치·trade-off 결론)과의 연결

결론은 "Q4는 모델 크기를 X% 줄이고 생성 속도를 Y배 높였지만, 남은 단계가 4개 늘 때 성공률 손실이 Z%p 커지고 조용한 오류 회복률은 W%p 낮았다" 형태로 수치와 trade-off를 함께 씁니다.

---

## 20. 팀 협업과 결과 공유

**가장 중요한 규칙: 한 정밀도 비교에 들어가는 모든 모델은 같은 PC에서, 같은 llama.cpp 빌드와 같은 설정 잠금으로 실행합니다.** 팀원이 나눠 맡을 때는 "정밀도별로" 나누지 말고 "PC 하나가 모든 정밀도를 실행"하는 방식으로 나눕니다(예: A는 노트북에서 FP16·Q8·Q4 전체, B는 GPU PC에서 FP16·Q8·Q4 전체). 서로 다른 PC 결과는 장비별로 따로 분석하고, 같은 방향인지 비교합니다.

GitHub에는 코드·task·manifest만 올립니다. 결과는 팀 Google Drive에 다음 구조로 올립니다.

```text
<날짜>_<장비명>_<시리즈>/            예: 20261020_laptop-arc_main
├─ *.jsonl                         모든 조건의 결과
├─ *.preflight.json
├─ settings.json                   설정 잠금 파일
├─ summary.json, summary*.csv, plots/
├─ manifest.json                   models/manifest.json 사본
├─ server_logs/                    각 정밀도의 서버 시작 로그
└─ run_info.txt                    OS, CPU/GPU, RAM, VRAM, llama.cpp 버전, 서버 명령, 실행 날짜, 특이사항
```

`results/`는 `.gitignore`에 있으므로 `git add .`를 해도 올라가지 않습니다.

---

## 21. 데이터셋 수정 방법

`tasks/tasks.json`과 두 카탈로그는 **생성된 파일**입니다. 직접 고치면 실행 전 검증에서 "생성기 원본과 다르다"며 멈춥니다.

| 바꾸려는 것 | 수정할 곳 |
|---|---|
| 시나리오 수, 데이터 버전, horizon, 오류 종류 목록 | `src/dataset_spec.py` |
| 시나리오 내용(상품·일정·노선·장소 표, 조건, 순위 규칙, 함정 패턴, 위치 배치) | `scripts/build_execution_tasks.py` |
| 회복 오류 설계, 복구 경로, 카탈로그 문구 | `scripts/build_recovery_tasks.py` |
| 감사 메시지, 시스템 프롬프트 뒷부분 | `src/execution_environment.py` (`SIGNALED_ERROR_MESSAGE`, `AGENT_INSTRUCTIONS`) |
| 시스템 프롬프트 앞부분 | `src/environment.py` (`SYSTEM_PROMPT`) |

수정한 뒤:

```text
python scripts/build_tasks.py --write
python scripts/validate_tasks.py
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

`--write`는 생성 결과를 메모리에서 먼저 검증하고 통과할 때만 파일을 바꿉니다. 모델 입력이 바뀌는 수정이면 `DATASET_VERSION`을 올리고, 이전 결과와 섞지 않습니다. 시나리오 수를 바꾸면 생성기의 표(분야마다 행 수 = `SCENARIOS_PER_DOMAIN`)도 함께 맞춰야 합니다.

---

## 22. 문제 해결 FAQ

| 증상 | 원인과 해결 |
|---|---|
| `server context size must be 8192; observed ...` | 서버를 `--ctx-size 8192`로 띄우지 않음. 다른 값을 쓰려면 모든 실행에 `--expected-context-size`를 같은 값으로 주고 새 잠금 사용 |
| `server total_slots must be 1` | 서버에 `--parallel 1` 추가 |
| `/props is missing ...` | 너무 오래된 llama.cpp. 팀 공통 최신 빌드로 맞춤 |
| `settings differ from the existing lock (...)` | 괄호 안 필드가 달라짐(빌드, template, 장비, 추론 설정 등). 잠금을 지우지 말고 원래대로 되돌리거나 새 시리즈로 분리 |
| `Output/preflight report already exists` | 같은 이름의 결과가 있음. `--output`을 새 이름으로 |
| `could not reach llama-server` | 서버가 꺼졌거나 포트가 다름. `--base-url` 확인 |
| `No tasks match --task-type, --horizon and --task-id` | 회복 조건에 `--horizon 5`를 줬거나 ID 접두사(`recovery_`)가 틀림 |
| 결과에 `n_runtime_errors`가 많음 | 타임아웃(`--timeout-seconds` 늘림), 서버 메모리 부족, 서버 재시작 여부 확인 후 해당 과제 재수집 |
| `model_output_error_count`가 많고 `unparsed_tool_call`이 대부분 | 서버가 툴콜을 파싱하지 못함. `--jinja` 사용 여부, 빌드의 Qwen3 tool-call 지원 확인. 정밀도 간 형식 오류 차이 자체도 결과로 보고 |
| `server_output_parse_failure` | 서버가 모델 출력 파싱 중 500을 냄. 모델 행동으로 집계되며 정상 동작임 |
| `peak_server_ram_mb`가 `null` | psutil 미설치, `--server-pid` 누락, 또는 PID가 바뀜 |
| `peak_gpu_memory_mb`가 `null` | `--gpu-index` 누락, 또는 NVIDIA가 아님(Intel/AMD는 정상적으로 null) |
| 분석에서 `Compared results come from different machines/settings/datasets` | 다른 PC나 다른 잠금의 결과가 섞임. 장비별로 나눠 분석 |
| 분석에서 `duplicate ... record` | 같은 과제·조건을 두 번 넣음. 재실행 전 파일을 입력에서 제외 |
| `model_storage_mb`가 비어 있음 | 라벨에 양자화 이름이 없거나 manifest에 그 파일이 없음 |
| PowerShell에서 `$pid` 관련 오류 | `$PID`는 예약 변수. `$serverPid` 사용 |
| 그래프가 안 생김 | `python -m pip install matplotlib` |

---

## 23. 변경 이력

이전 버전(한국어 파일럿, `workflow-en-v1`~`v3`)의 결과는 새 버전과 합치지 않습니다. 모델 파일은 그대로 써도 되지만 결과는 새로 수집합니다.

### workflow-en-v4 (현재)

| # | 문제 | 수정 |
|---|---|---|
| 1 | 시스템 프롬프트가 존재하지 않는 `finish` 도구를 지시 | "finish 도구 없음, 도구 호출 없는 답이 종료"로 수정 |
| 2 | 파싱되지 않은 툴콜 텍스트가 정상 종료로 처리되어 형식 실패가 판단 실패로 둔갑 | `unparsed_tool_call` 형식 오류로 분류 |
| 3 | 서버의 모델 출력 파싱 실패(HTTP 500)가 인프라 장애로 분모에서 제외 | `server_output_parse_failure` 모델 출력 오류로 분류 |
| 4 | `"rating": 4.1000000000000005` 같은 부동소수점 잔재 | 반올림, 검증기가 거부 |
| 5 | 기본 시나리오 20개로는 검정력 부족 | 60개(300 task), 개수는 `src/dataset_spec.py`에서 관리 |
| 6 | 정답 후보 위치 편향(3번째 자리 0/20 등) | 분야별 위치 균형, 검증기가 검사 |
| 7 | 오류 종류가 1개뿐 | `wrong_option` 추가 |
| 8 | 조용한 오류만 있어 회복률 바닥 효과 위험 | `signaled_error_restart` 대조 조건 |
| 9 | 핵심 가설(증폭)의 직접 검정 없음 | 정밀도 × horizon 상호작용, sign-flip 검정, Holm 보정 |
| 10 | 다른 PC 결과를 경고 없이 비교 | 장비·설정 불일치 시 정밀도 비교 중단 |
| 11 | 평균 시간이 일찍 실패한 실행까지 포함, 토큰 속도 없음 | 토큰/초, 턴당 지연, 성공 실행 평균 시간 |
| 12 | 효율 축이 가중치 양자화 3점뿐 | KV cache 형식 기록, 추가 양자화 단계 생성, manifest 병합 |
| 13 | 형식 오류 피드백이 모델 응답을 JSON 덤프로 보여줌 | 모델이 실제로 낸 텍스트를 그대로 보여줌 |

### workflow-en-v2 설계 검토 체크리스트 대응

| 검토 항목 | 반영 | 버전 |
|---|---|---|
| 1·2 회복 h5 fixture가 다른 조건의 복사본 | 회복 h5 제외, h5는 `free`로만 분석 | v3 |
| 3 `normal_restart` h1 draft 1개 | `free` h1과 같은 draft 3개 | v3 |
| 4 horizon별 복구 비용 차이 | `violating_candidate` 복구 = horizon. `wrong_option`은 h3 복구 2회라 `recovery_min_calls`·`error_kind`별로 분리 보고 | v3·v4 |
| 재발 방지: 조건 간 입력 동일성 검사 | 호출 ID를 무시한 입력 비교(모든 조건·오류 종류) | v3·v4 |
| 5 함정 강도 불균일 | 모든 시나리오에서 위반 후보가 조건 무시 시 단독 1등 | v3 |
| 5 "잘못된 배송 옵션" 오류 종류 추가 | `wrong_option` | v4 |
| 6 h3·h5 차이 해석 | "대화 길이·ID 전달 효과"로 보고 | v3 |
| 7 ID 복사 오류와 선택 오류 분리 | `identifier_error_*` vs `wrong_save_*` | v3 |
| 8 시나리오 40~60개로 확대 | 60개 | v4 |

### workflow-en-v3

회복 h5 fixture 삭제, 회복 checkpoint = 대응 `free` 과제의 제공 정보, h1 오류 흐름을 "draft 3개 준비 후 위반 draft 확정"으로 변경, 위반 후보 함정 강도 통일, ID 복사 오류 지표 추가.

### workflow-en-v2

자연스러운 오류(조건 위반 후보를 같은 요청으로 확정) 주입, `free`는 앞 단계 결과를 사용자 메시지로 제공, 다음 행동 정확도·작업 단계별 첫 오류·자기 회복률 지표, 정밀도 간 짝지은 비교, GPU 메모리 측정.

---

## 24. 알려진 한계

- **과제 다양성.** 4개 분야 모두 같은 5단계 템플릿이고 후보는 3개뿐입니다. 실제 Agent 작업 전체를 대표하지 않습니다.
- **오류 다양성.** 회복 오류는 "같은 요청으로 잘못된 결과가 저장됨" 계열의 두 종류입니다. 다른 오류(잘못된 도구, 오래된 정보, 중간 단계 누락)로 일반화하지 않습니다.
- **주입된 기록.** 오류 분기의 과거 기록은 미리 만든 이력이며, 해당 모델이 실제로 만든 세션이 아닙니다. 자기가 만든 기록인지 여부가 회복에 영향을 줄 수 있습니다.
- **단일 실행.** temperature 0으로 과제당 한 번 실행하므로 샘플링 변동은 측정하지 않습니다. greedy decoding이라도 장비가 다르면 출력이 다를 수 있습니다.
- **짧은 대화.** 가장 긴 과제도 입력이 수천 자 수준입니다. 수십 단계에 걸친 장기 Agent에 대한 결론은 아닙니다.
- **단일 모델.** Qwen3-1.7B 하나만 다룹니다. 모델 크기에 따른 차이는 다루지 않습니다.
- **설정 검증 범위.** 서버가 `/props`로 보고하는 값만 자동 검사합니다. 실제 GPU 배치, 스레드, KV cache 정밀도는 서버 시작 로그로 사람이 확인해야 합니다.
- **외부 검증.** 일반적인 tool-calling 성능이 같은 방향인지는 BFCL 등 외부 벤치마크로 내부 실험이 안정된 뒤 별도로 확인합니다.
- **resume 없음.** 실행이 중간에 끊기면 이어서 하는 기능이 없습니다. 이미 기록된 줄은 남으므로, 남은 과제를 `--task-id`/`--horizon`으로 골라 새 파일로 실행합니다.
