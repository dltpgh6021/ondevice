# Granite FP16으로 검증 결과 제공 효과 비교하기

**실행 순서**는 로컬 Windows에서 코드 업로드 → 기존 Granite 원격 WSL에서 48회 실행 → 결과 검증 → ZIP 공유다. 앞서 3개 시나리오를 돌렸던 PC의 모델과 가상환경을 그대로 사용한다. 원격 GPU에서는 아래 2절부터 따라 하면 된다.

## 1. 무엇을 비교하고 어떤 파일을 올릴까?

h3는 **이미 저장된 0번 상품을 검토하면서, 남은 1·2·3번 상품을 선택하는 과제**다. 같은 개발 시나리오 12개를 다음 네 조건에서 처음부터 각각 실행한다.

| 검증 결과 제시 방식 | 0번에 올바른 상품이 저장됨 | 0번에 예산을 넘긴 상품이 저장됨 |
|---|---|---|
| 선택형 `validation_optional` | 정상 항목을 보존하는지 12회 | 스스로 문제를 찾아 고치는지 12회 |
| 사전 제공형 `validation_supplied` | ‘저장 항목 위반 없음’ 보고서를 받고 보존하는지 12회 | 위반 위치·가격·예산을 보고 대안을 골라 고치는지 12회 |

두 방식은 같은 질문·후보·도구를 사용한다. 어느 쪽에서든 모델이 직접 가격을 비교하거나 검증 도구를 호출할 수 있다. 사전 제공형에서는 첫 응답 전에 실행기가 검증 보고서 한 건을 대화에 추가한다. 보고서에는 정답 상품 ID가 없다. 모델이 생성할 수 있는 응답은 두 방식 모두 최대 8회다. 이 차이로 **위반 정보가 제공됐을 때 회복 행동이 어떻게 달라지는지** 본다.

이전에 본 세 과제(`dev-000`~`002`)도 이번 설정에서 다시 실행한다. 결과는 이번 48건으로 집계한다. 나머지 아홉 과제(`dev-003`~`011`)는 이번 검증 결과 제공 비교에서 추가되는 사례다.

**로컬 Windows CMD**에서 아래 명령을 실행한다. 과제 목록, 실행·집계 코드, 짝 비교 테스트, 이 가이드가 함께 올라간다.

```cmd
cd /d "%USERPROFILE%\Documents\Codex\2026-09-30\referenced-chatgpt-conversation-this-is-an\ondevice-phase2"
git branch --show-current
git status --short
```

브랜치가 `experiment/phase2-position-controls`이면 이어서 실행한다.

```cmd
git add src/recovery_diagnostic.py scripts/diagnose_phase2_recovery.py tests/test_phase2_recovery_diagnostic.py PHASE2_H3_VALIDATION_COMPARE.md
git commit -m "Compare validation feedback on twelve paired h3 scenarios"
git push origin experiment/phase2-position-controls
git log -1 --oneline
```

## 2. 원격 WSL에서 코드와 GPU 확인

이제부터 **Granite 진단을 돌렸던 원격 PC의 VS Code WSL 터미널**을 사용한다. 같은 터미널에서 순서대로 실행한다.

```bash
cd "$HOME/ondevice-phase2"
git branch --show-current
```

위와 같은 브랜치인지 확인한 뒤 실행한다.

```bash
git pull --ff-only origin experiment/phase2-position-controls
git log -1 --oneline
.venv-hf/bin/python -m unittest discover -s tests -p "test_phase2*.py" -q
GRANITE_RUN="$(cat results/phase2/latest_granite_check.txt)"
ls "$GRANITE_RUN/plan.json" "$GRANITE_RUN/model_manifest.json" "$GRANITE_RUN/granite/trials.jsonl"
nvidia-smi
```

로컬과 원격의 마지막 커밋 번호가 같고 테스트 마지막에 `OK`가 떠야 한다. 여기서 읽는 기준 폴더는 **최초 Granite 6회 결과**다. 직전에 만든 24회 진단 ZIP을 지정하면 경로가 맞지 않는다. 추론은 Python이 GPU 모델을 직접 로드한다. 이전 서버가 GPU를 사용 중이면 [Granite 가이드의 GPU 확인 절차](PHASE2_GRANITE.md)를 먼저 따른다.

## 3. 계획 확인 후 48회 실행

아래에서 고유한 상위 폴더를 만든다. 하위 `run` 폴더는 실제 실행 시 자동 생성된다.

```bash
OUT="$(mktemp -d "$PWD/results/phase2/h3_validation_XXXXXXXX")"
printf '%s\n' "$OUT" > results/phase2/latest_h3_validation.txt
git rev-parse HEAD > "$OUT/project_commit.txt"
.venv-hf/bin/python scripts/diagnose_phase2_recovery.py run --stage validation_dev12 --reference-run "$GRANITE_RUN" --output-dir "$OUT/run" --dry-run
```

계획에 `"trials": 48`, `dev-000`부터 `dev-011`, `"max_generated_turns": 8`이 나오면 **아래 블록을 한 번만** 실행한다.

```bash
set -o pipefail
.venv-hf/bin/python -u scripts/diagnose_phase2_recovery.py run --stage validation_dev12 --reference-run "$GRANITE_RUN" --output-dir "$OUT/run" 2>&1 | tee "$OUT/console.log"
```

모델 파일 확인과 워밍업 2회 후 `[1/48]`부터 진행된다. 이전 동일 PC에서 h3 12회의 과제 실행시간 합계가 163초였다. 같은 속도면 **추론 약 11분**이고, 모델 로딩과 과제별 응답 길이에 따라 실제 소요 시간은 달라진다. `success=False`도 유효한 모델 결과라서 다음 과제를 계속 실행한다. 완료 문구는 `COMPLETED: 48 valid diagnostic trials.`다.

다른 WSL 터미널에서 진행 상황만 보고 싶을 때는 다음을 입력한다.

```bash
cd "$HOME/ondevice-phase2"
OUT="$(cat results/phase2/latest_h3_validation.txt)"
tail -f "$OUT/console.log"
```

이 조회 창은 `Ctrl+C`로 닫을 수 있다. 실제 추론을 실행한 터미널은 완료될 때까지 열어 둔다. 중간에 서버·GPU·내부 오류로 멈추면 `console.log`와 부분 결과를 보존하고 원인을 확인한다. 다시 실행할 때는 3절 첫 블록부터 새 폴더를 만든다.

## 4. 결과 검증과 해석

48회 완료 후 원격 WSL에서 아래 명령을 실행한다. `report`는 저장된 도구 호출을 재생해 시작 상태·도구 결과·최종 상태·집계가 일치하는지 검사한다. 모델 추론을 다시 할 필요가 없다.

```bash
cd "$HOME/ondevice-phase2"
OUT="$(cat results/phase2/latest_h3_validation.txt)"
.venv-hf/bin/python scripts/diagnose_phase2_recovery.py report --run-dir "$OUT/run"
```

`$OUT/run/summary.json`에 네 조건의 `n_valid`가 각각 **12**인지 먼저 확인한다.

| 보고 싶은 결과 | 읽을 필드 | 분모와 해석 |
|---|---|---|
| 주어진 오류를 실제로 고쳤는가 | 오류 조건의 `accessory_correct_at_end` | 12건 중 종료 시 0번이 정답인 건수 |
| 정상 항목을 보존했는가 | 정상 조건의 같은 필드 | 12건 중 종료 시 0번이 여전히 정답인 건수. 중간에 손댄 건수는 `accessory_write_episodes`로 확인 |
| 남은 작업을 얼마나 했는가 | `bundle_correct_lines`, `bundle_all_correct` | 정답인 1·2·3번 항목 수는 최대 36개. 세 항목이 모두 맞은 과제 수는 최대 12건. 종료 상태에서 각 항목의 실제 후보로 판정 |
| 주문 전체를 완료했는가 | `successes` | 12건 중 0~3번 모두 정답인 건수 |
| 모델이 직접 검증을 요청했는가 | `model_validation_episodes` | 12건 중 직접 한 번이라도 검증한 건수. 실행기가 미리 준 보고서는 `preloaded_validation_episodes`로 따로 집계 |

**먼저 오류 조건 두 행에서 0번 복구 건수를 비교하고, 그다음 같은 두 행의 묶음 완료와 주문 전체 완료를 본다.** 예를 들어 0번을 고쳤는데 마지막 상품을 잘못 골랐다면 복구는 성공하고 묶음과 전체 완료는 실패한다. 선택형에서 검증 도구 없이 스스로 정확히 고쳐도 복구 성공이다.

`paired`에는 같은 시나리오끼리 짝지은 결과가 들어 있다. 각 조건·지표별로 `gained_with_report`는 선택형 실패 → 제공형 성공, `lost_with_report`는 선택형 성공 → 제공형 실패인 시나리오 ID다. 두 목록을 함께 읽고, **이번에 추가한 아홉 사례에서도 변화가 나타났는지** ID로 확인한다. 특정 사례의 수정 순서는 `trials.jsonl`의 `first_correct_accessory_write_turn`과 실제 도구 호출에서 확인할 수 있다.

정상 항목이 잘 보존되고 추가 사례에서도 제공형의 복구가 반복되면 같은 조건의 h1·h3·h5 FP16 검증을 검토한다. 0번 복구는 늘었지만 전체 완료가 계속 낮다면 `bundle_correct_lines`와 실패 사례부터 살펴본다. 이번 측정 범위는 Granite **FP16·h3·개발 과제**이므로 양자화 효과는 이후 정밀도 비교에서 판단한다.

## 5. 공유할 ZIP 만들기

4절의 재생 검증이 성공한 뒤 같은 WSL에서 실행한다.

```bash
OUT="$(cat results/phase2/latest_h3_validation.txt)"
ZIP="$HOME/$(basename "$OUT").zip"
.venv-hf/bin/python -m zipfile -c "$ZIP" "$OUT"
.venv-hf/bin/python -m zipfile -t "$ZIP"
ls -lh "$ZIP"
explorer.exe "$(wslpath -w "$HOME")"
```

탐색기에서 방금 출력된 이름의 **`h3_validation_….zip` 하나**를 공유하면 된다. 콘솔 로그·커밋·모델 정보·48회 원시 기록·조건별 결과표가 들어 있다.
