# 회복 실험 개선 순서와 원격 실행 가이드

현재 Granite FP16은 h3 정상 재개 2/3, 오류 재개 0/3이다. 오류 조건에서 0번 수정 호출이 없었다. 다음은 **모델을 더 바꾸거나 양자화를 확대하기 전에 실패 지점을 분리하는 단계**다. 기존 질문·채점·결과를 보존하고, 이미 내려받은 Granite와 같은 개발 시나리오 세 개를 사용한다.

## 전체 순서

| 순서 | 확인하거나 개선할 것 | 사용자가 할 일 | 다음으로 넘어가는 판단 |
|---|---|---|---|
| 1 | 0번의 조건 위반 판단과 명시적 저장 변경 | 아래 `basic` 진단 12회 실행·공유 | 잘못 고르거나 지정된 수정도 못한 사례를 먼저 확인 |
| 2 | 검증 도구 결과가 회복에 도움이 되는지 | 1단계 확인 후 `validation` 진단 12회 실행·공유 | 결과 제공 후 수정되는지, 정상 상태를 훼손하지 않는지 확인 |
| 3 | 질문·상태 표시·오류 알림 조건 정리 | 진단 결과를 바탕으로 새 과제 버전 결정 | 실패 원인에 맞는 변경을 선택하고 작은 대응 비교 |
| 4 | FP16 기준 성능 확보 | 확정한 새 버전으로 dev 전체 평가 | 정상 선택과 회복이 모두 가능한지 확인 |
| 5 | 정밀도 본실험 | 같은 FP16에서 Q8·Q4 직접 생성 후 미사용 과제 평가 | 모델·질문·도구·예산·채점을 고정해 비교 |

**이번에 바로 실행할 것은 1단계다.** 2단계 코드도 준비돼 있지만 1단계에서 무엇이 막히는지 확인한 뒤 실행한다. 3단계 이후의 과제 변경과 전체 dev 실행기는 진단 결과에 따라 달라지므로 아직 기존 코드에 적용하지 않았다.

## 진단이 하는 일과 필요한 이유

첫 단계는 남은 작업을 제거하고 0번만 분리한다. 따라서 그 성공률을 원래 h3 과제 성공률과 직접 비교하지 않는다. 입력은 영어이며 여기서 목적을 한국어로 설명한다.

첫 단계에서는 과거 도구 대화 없이 현재 0번 상태만 보여준다. 이 분리 진단이 성공하더라도 과거 대화 하나가 기존 실패의 원인이라고 단정할 수는 없다.

| 단계와 모드 | 모델이 받는 요청 | 측정하는 것 |
|---|---|---|
| 1 `judge_saved` | 현재 0번이 규칙을 만족하는지 판단하고, 예산 이하 최고 평점 상품 ID를 제출 | 위반 여부와 올바른 대안을 판단할 수 있는가 |
| 1 `apply_exact` | 지정한 정답 상품 ID로 0번을 맞추고, 이미 그 상품이면 유지 | 정답을 아는 상태에서 실제 도구로 수정할 수 있는가 |
| 2 `validation_optional` | 기존 h3 질문과 상태에 `validate_order` 도구를 추가 | 모델이 스스로 검증 도구를 쓰는지와 최종 회복 결과 |
| 2 `validation_supplied` | 같은 h3 질문·상태·도구를 사용하되, 검증 결과 한 개를 실행기가 미리 제공 | 위반 정보가 주어진 뒤 올바르게 수정하고 남은 작업을 끝내는가 |

각 모드는 세 시나리오의 정상·오류 두 상태를 모두 사용해 6회다. 단계별 두 모드이므로 **12회 + 12회**다. 정상 과제를 포함하는 이유는 올바른 저장을 괜히 바꾸는지도 확인하기 위해서다.

`apply_exact`는 정답을 알려주는 기능 점검이다. 자연스러운 회복 능력의 본실험 점수에 넣지 않는다. `judge_saved`는 판정만 제출하고 주문을 수정하지 않는다.

## 검증 도구의 의미

`validate_order`는 현재 저장된 항목이 예산·평점 규칙을 위반하는지 확인한다. 예를 들어 예산 88에 가격 100이 저장됐다면 다음 정보를 반환한다.

```json
{"line": 0, "rule": "budget_exceeded", "selected_price": 100, "budget": 88}
```

정답 상품 ID는 알려주지 않는다. 아직 선택하지 않은 항목은 별도로 표시하며, 위반 목록이 비었다고 전체 과제가 완료됐다고 말하지 않는다. 두 모드 모두 같은 검증 도구를 사용할 수 있다.

`validation_supplied`의 첫 검증 결과는 **실행기가 제공한 개입**이다. 모델이 스스로 검증 도구를 선택했다는 뜻이 아니다. 이 1회는 `preloaded_validation_calls`로 기록하고 모델의 응답·호출 횟수와 분리한다. 생성 응답 상한은 두 모드 모두 8회다. 이 진단으로 속도·효율성의 우열을 주장하지 않는다.

현재 상태에는 이미 가격과 예산이 보이므로 `validation_optional`에서 도구를 호출하지 않고 직접 비교해 고쳐도 성공이다. 도구를 안 썼다는 사실만으로 오류 발견 실패라고 채점하지 않는다. 검증이 필수인 새 관측 구조를 만들려면 3단계에서 정보 제공 방식과 필요한 호출 수를 따로 설계해야 한다.

## 0. 로컬 코드 업로드

먼저 **로컬 Windows CMD**에서 이번 수정 파일을 올린다. 수정 위치는 다음 저장소다.

```cmd
cd /d "%USERPROFILE%\Documents\Codex\2026-09-30\referenced-chatgpt-conversation-this-is-an\ondevice-phase2"
git branch --show-current
git status --short
git add README.md PHASE2_RECOVERY_STEPS.md src/recovery_diagnostic.py scripts/diagnose_phase2_recovery.py tests/test_phase2_recovery_diagnostic.py
git commit -m "Add staged recovery diagnostics and experiment guide"
git push origin experiment/phase2-position-controls
```

브랜치는 `experiment/phase2-position-controls`다. 기존 모델·질문·실행 파일을 덮어쓰지 않는다. 약 1~2분이며, Git 업로드는 사용자가 한다.

## 1. 원격 WSL 환경과 기존 결과 확인

이후 명령은 **Granite를 실행했던 같은 원격 PC의 WSL bash**에서 실행한다. 기존 `.venv-hf`와 모델을 재사용한다. 새 모델 다운로드나 llama.cpp 재빌드는 필요 없다.

```bash
cd "$HOME/ondevice-phase2"
git pull --ff-only origin experiment/phase2-position-controls
git log -1 --oneline
ls scripts/diagnose_phase2_recovery.py src/recovery_diagnostic.py
.venv-hf/bin/python -m unittest discover -s tests -p "test_phase2*.py" -v
```

예상 결과는 **48개 테스트와 `OK`**, 약 5~15초다.

```bash
GRANITE_RUN="$(cat results/phase2/latest_granite_check.txt)"
ls "$GRANITE_RUN/plan.json" "$GRANITE_RUN/model_manifest.json" "$GRANITE_RUN/granite/trials.jsonl"
cat "$GRANITE_RUN/granite/summary.json"
nvidia-smi
```

기준 결과는 정상 2/3·오류 0/3이다. 기존 `llama-server`나 다른 모델이 GPU에 남아 있다면 [Granite 가이드 4절](PHASE2_GRANITE.md#4-gpu-메모리-비우기)에 따라 확인 후 종료한다. 새 PID나 서버 주소를 지정할 필요는 없다.

새 진단 묶음의 폴더를 만들고 위치를 저장한다. 아래 `basic`·`validation` 하위 폴더는 실행기가 생성하므로 미리 만들지 않는다.

```bash
DIAG_ROOT="$PWD/results/phase2/recovery_components_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DIAG_ROOT"
printf '%s\n' "$DIAG_ROOT" > results/phase2/latest_recovery_components.txt
```

## 2. 첫 단계 판단·수정 진단 12회

계획부터 확인한다.

```bash
.venv-hf/bin/python scripts/diagnose_phase2_recovery.py run --stage basic --reference-run "$GRANITE_RUN" --output-dir "$DIAG_ROOT/basic" --dry-run
```

약 1~5초 후 `trials: 12`, 모드 `judge_saved`, `apply_exact`, `max_generated_turns: 3`이 표시되면 실행한다.

```bash
set -o pipefail
.venv-hf/bin/python -u scripts/diagnose_phase2_recovery.py run --stage basic --reference-run "$GRANITE_RUN" --output-dir "$DIAG_ROOT/basic" 2>&1 | tee "$DIAG_ROOT/basic_console.log"
```

모델 해시 확인·로딩·워밍업 2회 후 `[1/12]`부터 `[12/12]`까지 출력된다. 예상 추론 시간은 **약 1~5분**이며 새 진단의 실제 시간은 아직 측정하지 않았다. 마지막에 `COMPLETED: 12 valid diagnostic trials.`가 나와야 한다. 과제 실패 `False`는 기록하고 계속 진행하지만 인프라·내부 오류는 중단한다.

```bash
.venv-hf/bin/python scripts/diagnose_phase2_recovery.py report --run-dir "$DIAG_ROOT/basic"
```

보고서는 초기 입력과 도구 결과를 재생한 뒤 저장된 집계를 확인한다. `summary.json`의 각 모드·조건별 `n_valid`는 3이어야 한다.

| 첫 단계 결과 | 우선 확인할 것 |
|---|---|
| `apply_exact` 오류 조건에서 실패 | 지정된 상품 ID가 실제 도구 호출에 사용됐는지, 형식 반려·조기 종료가 있었는지 확인 |
| `judge_saved`의 `validity_correct`가 낮음 | 저장된 상품이 규칙을 만족하는지 판단하는 문제 |
| `validity_correct`는 맞지만 `best_offer_correct`가 낮음 | 오류를 알아도 대안을 고르는 문제 |
| 두 모드의 정상·오류 사례를 모두 해결 | 기본 판단과 수정 동작이 가능한 근거. 다음 검증 결과 제공 비교로 진행 |

세 문제의 성패로 모델의 일반 능력을 확정하지 않는다. 틀린 사례가 있으면 그 trace를 먼저 읽고 원인을 확인한다. 정상 조건의 저장 유지와 오류 조건의 실제 수정 여부를 구분한다.

**이 단계가 끝나면 먼저 ZIP을 공유하면 된다.** 결과 해석이 불분명한 상태에서 두 번째 단계를 연달아 실행할 필요는 없다.

## 3. 두 번째 단계 검증 결과 제공 비교 12회

첫 단계에서 기본 판단·수정 동작을 확인한 뒤 실행한다.

```bash
.venv-hf/bin/python scripts/diagnose_phase2_recovery.py run --stage validation --reference-run "$GRANITE_RUN" --output-dir "$DIAG_ROOT/validation" --dry-run
.venv-hf/bin/python -u scripts/diagnose_phase2_recovery.py run --stage validation --reference-run "$GRANITE_RUN" --output-dir "$DIAG_ROOT/validation" 2>&1 | tee "$DIAG_ROOT/validation_console.log"
.venv-hf/bin/python scripts/diagnose_phase2_recovery.py report --run-dir "$DIAG_ROOT/validation"
```

모드는 `validation_optional`, `validation_supplied`이고 각각 정상 3회·오류 3회다. 생성 응답 상한은 8회다. 예상 시간은 **약 3~10분**이며 실제 시간은 실행 후 기록한다.

| 관측 결과 | 해석과 다음 행동 |
|---|---|
| 제공 조건에서만 오류 해소·전체 성공이 나타남 | 검증 결과를 제공한 개입이 도움이 됨. 검증 필요성 판단과 제공된 위반 정보 활용을 분리한 설계 검토 |
| 제공 조건에서 0번은 고치지만 전체 과제 실패 | 회복은 가능하나 남은 항목의 선택·종료가 병목. `accessory_correct_at_end`와 `successes`를 구분 |
| 제공 조건에서도 0번 수정이 없음 | 도구 결과 활용과 여러 항목 속 0번 재검토를 확인. 모델의 내부 인지 상태로 단정하지 않음 |
| 자율 조건에서도 회복함 | 실제 검증 호출 시점과 수정 경로 확인. 추가 도구의 존재와 호출이 무엇을 바꿨는지 구체적으로 분석 |
| 정상 조건에서 0번을 잘못 바꿈 | 오류 조건의 이득과 함께 정상 상태 훼손을 검토 |

`model_validation_calls`는 모델이 실제 호출한 횟수, `preloaded_validation_calls`는 실행기가 미리 제공한 횟수다. 기본 판단·수정 진단의 성공률과 h3 전체 과제 성공률을 하나로 합치지 않는다.

검증 도구를 호출했더라도 수정한 뒤 확인했을 수 있다. `first_model_validation_turn`과 `first_correct_accessory_write_turn` 및 원문 trace를 함께 읽어 검증 결과를 받은 뒤 수정했는지 확인한다.

## 4. 결과를 공유하는 방법

첫 단계만 끝났든 두 단계가 모두 끝났든 같은 명령을 사용한다. 참조 Granite 로그와 모델 manifest의 복사본은 진단 폴더에 자동으로 포함된다. 모델 가중치와 가상환경은 들어가지 않는다.

```bash
.venv-hf/bin/python -m zipfile -c "$HOME/phase2_recovery_components.zip" "$DIAG_ROOT"
explorer.exe "$(wslpath -w "$HOME")"
```

`phase2_recovery_components.zip`을 공유한다. `results`는 Git에서 제외되므로 ZIP 또는 팀 드라이브로 전달한다. ZIP을 다시 만들 때 같은 이름의 기존 ZIP은 새 내용으로 교체되므로 이전 공유본을 보존하려면 이름을 바꾼다.

다른 터미널에서 실시간으로 보려면 다음 명령을 사용한다.

```bash
cd "$HOME/ondevice-phase2"
DIAG_ROOT="$(cat results/phase2/latest_recovery_components.txt)"
tail -f "$DIAG_ROOT/basic_console.log"
```

두 번째 단계를 보는 경우 파일명을 `validation_console.log`로 바꾼다. 이 tail 창의 `Ctrl+C`는 화면만 닫는다. 실제 실험 터미널의 `Ctrl+C`는 실험을 중단한다.

## 5. 진단 이후의 과제 수정 방향

아래는 **결과에 따라 선택할 수정안**이다. 아직 적용된 기능이나 지금 실행할 명령이 아니다.

| 결과에 맞는 수정 | 구체적인 변경 | 연구에서 설명할 범위 |
|---|---|---|
| 필수 항목을 명확히 제시 | 질문 시작부터 전체 필수 항목을 동등하게 보여주고, 저장 여부와 규칙 충족 여부를 구분 | 같은 의미의 문항 표현·관측 형식 개선. 정상·오류 양쪽에 같은 변경 적용 |
| 검증 이후 회복을 주 조건으로 선택 | 공통 흐름에서 검증 결과를 얻은 뒤 후보 선택·수정·남은 작업을 평가 | 검증 결과가 제공된 상태의 회복. 스스로 오류를 찾는 능력과 구분 |
| 스스로 확인하는 과정을 주 조건으로 선택 | 무엇을 조회해야 새 정보를 얻는지 설계하고 검증·수정 호출과 h1·h3·h5의 관계를 정의 | 필요한 확인 행동과 그 뒤의 회복. 검증 도구를 안 썼다는 이유만으로 임의 실패 처리하지 않음 |
| 후보 판단 자체가 어려움 | 조건·관측 표현을 개발 세트에서 검토하고 필요하면 다른 FP16 후보를 소규모 평가 | 모델과 과제의 적합성. 현재 모델이 통과하도록 정답이나 채점을 느슨하게 바꾸지 않음 |

새 표현은 새 버전으로 관리하고 이전 결과와 섞지 않는다. 오류 조건에만 유리한 설명을 더하지 않는다. 검증 도구로 오류 위치·원인을 알려준 조건은 오류 알림 없는 조건과 구분한다. 정답 상품을 지정한 `apply_exact`는 계속 기능 점검으로만 둔다.

다음으로 확정한 모델·관측·도구·질문을 사용해 dev 전체를 실행한다. 기존 진행 기준은 horizon별 free·normal 성공 8/12 이상, error 성공 4/12 이상과 h1 정답 위치 점검이다. 이 기준을 결과가 낮다는 이유로 완화하지 않는다. 새 조건의 연구 의도가 달라지면 진행 기준도 본실험 전에 근거와 함께 다시 정한다.

현재 `scripts/phase2.py`는 기존 GGUF·서버·프로토콜을 전제로 한다. 새 Granite 조건을 전체 dev에 적용하는 실행기는 3단계에서 선택한 설계에 맞춰 추가해야 한다. 새 조건과 다른 기존 명령으로 전체 실험을 돌리지 않는다.

FP16 기준 성능을 확보한 뒤 같은 원본에서 Q8_0·Q4_K_M을 생성하고, 같은 장비·도구·설정·응답 예산으로 미사용 본실험 과제를 평가한다. 검증 호출·수정 호출·남은 필수 작업 수와 총 응답 수를 구분해 기록한다.

검증 단계를 추가한 새 버전에서는 h5까지 포함해 필수 호출과 종료 응답이 상한 안에 들어가는지 확인하고, 재시도 여유도 함께 고정한다. 정상·오류 조건은 필요한 수정 호출 수가 다르다는 점을 명시한다.

## 오류나 새 터미널에서 이어가기

```bash
cd "$HOME/ondevice-phase2"
GRANITE_RUN="$(cat results/phase2/latest_granite_check.txt)"
DIAG_ROOT="$(cat results/phase2/latest_recovery_components.txt)"
```

인프라 오류로 중단됐다면 콘솔과 부분 로그를 보존해 공유한다. 원인을 해결한 뒤 새 `DIAG_ROOT`를 만들고 재실행한다. 12회 진단에는 중간 재개 기능을 넣지 않았다. 기존 결과 폴더에는 덮어쓰지 않는다. 원격 코드나 모델 해시가 다르다고 나오면 출력을 확인하고 새 준비 폴더를 사용한다.

로컬에서는 자동 검증과 가짜 응답을 이용한 실행 흐름을 확인한다. 실제 모델 성능은 원격 GPU에서 위 절차로 확인한다.
