# 마지막 개인 테스트 · Granite FP16 h3 주문 표시 비교

**한 번 실행하면 정상 재개 12회와 오류 재개 12회가 진행된다.** 같은 PC에서 이전 h3 기록을 기준으로 사용한다. 완료 후 비교표와 ZIP을 팀에 공유하면 개인 테스트는 끝이다. 실제 추론 명령은 3절에 한 줄로 적었다.

## 비교하는 내용

12개 개발 주문 각각에서 0번 상품은 올바르게 저장돼 있거나 예산을 초과한 상태다. 두 조건 모두 이전처럼 저장 항목의 검증 보고서를 받고, 남은 1·2·3번 상품을 골라야 한다.

이번에는 주문 상태를 보여줄 때 **아직 저장하지 않은 첫 번째 묶음 줄**의 번호·예산·상품 후보를 `current_bundle_menu`에 모아 맨 앞에 표시한다. 0번을 포함해 나머지 보이는 줄은 `other_visible_lines`에 표시한다. 세 줄을 모두 저장하면 현재 메뉴가 비게 된다. 이 표시는 저장 여부로 결정되므로, 잘못된 선택도 저장돼 있다면 별도로 검토해야 한다. 질문, 검증 보고서, 상품 순서, 도구, 최종 상태 채점, 모델과 응답 상한(8회)은 동일하다.

**비교 기준은 지난 48회 중 보고서를 사전 제공한 24건**이다. 이전에 한 줄씩 풀었던 36개 메뉴의 결과는 이번 변경을 선택한 근거다.

| 기준 지표 | 정상 재개 12건 | 오류 재개 12건 |
|---|---:|---:|
| 주문 전체 성공 | 4/12 | 4/12 |
| 종료 시 0번 정답 (정상 보존·오류 복구) | 12/12 | 9/12 |
| 1·2·3번 첫 저장 정답 | 24/36 | 26/36 |
| 1·2·3번 모두 종료 시 정답 | 4/12 | 5/12 |

## 1. 로컬 Windows CMD에서 업로드

실행기와 자동 테스트, 이 문서를 실험 브랜치에 올린다. 앞서 작업한 로컬 저장소에서 아래 명령을 실행한다.

```cmd
cd /d "%USERPROFILE%\Documents\Codex\2026-09-30\referenced-chatgpt-conversation-this-is-an\ondevice-phase2"
git branch --show-current
git status --short
```

브랜치가 `experiment/phase2-position-controls`이면 이어서 입력한다.

```cmd
git add scripts/check_phase2_current_menu.py tests/test_phase2_current_menu.py PHASE2_FINAL_H3_GUIDE.md
git commit -m "Add final paired h3 current-menu display check"
git push origin experiment/phase2-position-controls
git log -1 --oneline
```

## 2. 원격 WSL에서 기준 확인

이후에는 **기존 Granite 실험을 돌린 원격 PC의 VS Code WSL 터미널**을 사용한다. 기존 가상환경과 GPU 모델을 그대로 사용한다.

```bash
cd "$HOME/ondevice-phase2"
git branch --show-current
```

1절과 같은 브랜치인지 확인한 뒤 아래를 입력한다.

```bash
git pull --ff-only origin experiment/phase2-position-controls
git log -1 --oneline
.venv-hf/bin/python -m unittest discover -s tests -p "test_phase2*.py" -q
SRC="$PWD/results/phase2/h3_validation_7wZ8VQF3/run"
.venv-hf/bin/python scripts/check_phase2_current_menu.py plan --source "$SRC"
nvidia-smi
```

로컬·원격의 마지막 커밋 번호가 같고 테스트 마지막에 `OK`가 뜨는지 확인한다. `plan`은 GPU 추론 없이 기존 기록을 재생한다. 위 표와 같은 기준 숫자와 아래 문구가 나오면 준비된 상태다.

```text
Plan: 24 new h3 trials; 12 normal, 12 error.
```

경로를 옮겨서 파일을 찾지 못하면 지난 48회 ZIP을 풀어 둔 폴더의 `run`을 `SRC`로 지정하고 `plan`을 다시 실행한다. 이 폴더에는 `metadata.json`, `trials.jsonl`, `summary.json`, `chat_template.jinja`, `reference/`가 있어야 한다.

## 3. 정상·오류 24회 실행

같은 WSL 터미널에서 아래 블록을 한 번 실행한다. 상위 폴더와 이름은 자동 생성된다. 이전 같은 장비에서 기준 24건의 과제 수행시간 합계는 **약 5분 28초**였다. 이번 실행에는 모델 파일 확인·로딩·워밍업 시간이 추가되고 응답 길이에 따라 소요 시간이 달라진다.

```bash
OUT="$(mktemp -d "$PWD/results/phase2/final_h3_XXXXXXXX")"
printf '%s\n' "$OUT" > results/phase2/latest_final_h3.txt
printf '%s\n' "$SRC" > "$OUT/source_path.txt"
git rev-parse HEAD > "$OUT/project_commit.txt"
set -o pipefail
.venv-hf/bin/python -u scripts/check_phase2_current_menu.py run --source "$SRC" --out "$OUT/run" 2>&1 | tee "$OUT/console.log"
```

`[1/24]`에서 `[24/24]`까지 진행된다. 예를 들어 `error_restart order=False line0=True`는 **주어진 오류는 고쳤으나 주문 전체는 실패했다**는 유효한 결과다. 성공·실패에 관계없이 정상적으로 끝나면 마지막에 `Verified 24 paired trials.`가 뜬다. 프로그램이 예외로 중단되면 부분 기록과 콘솔을 보존한다. 장비·경로 등 원인을 해결하고 실행을 다시 해야 할 때는 새 폴더를 사용한다.

## 4. 결과 확인 후 공유하고 마치기

아래 명령은 새 WSL 터미널에서도 실행할 수 있다. 모델을 다시 돌리지 않고 24쌍의 기록을 검증한다.

```bash
cd "$HOME/ondevice-phase2"
OUT="$(cat results/phase2/latest_final_h3.txt)"
SRC="$(cat "$OUT/source_path.txt")"
.venv-hf/bin/python scripts/check_phase2_current_menu.py check --source "$SRC" --out "$OUT/run"
```

`run/comparison.json`의 `before`는 기존 표시, `after`는 새 표시, `delta`는 건수 차이다. 팀에는 다음 순서로 설명하면 된다.

1. **주문 전체 성공** — `order_correct`가 정상 **8/12 이상**, 오류 **4/12 이상**을 모두 충족했는지 확인한다. 화면의 `H3 order thresholds met: True/False`가 이 판단이다. 오류 조건은 기존에도 4/12였으므로 개선 여부는 건수 차이도 함께 본다.
2. **상품 선택** — `first_save_correct_lines`에서 정상·오류 각각 36개 중 첫 저장 정답이 얼마나 달라졌는지 확인한다. `lines_with_save`가 36보다 작으면 시도하지 못한 항목도 있다. `bundle_correct_lines`는 최종 정답 항목 수다. 앞줄 선택이 달라지면 뒷줄 후보도 달라질 수 있으므로 동일한 **주문**끼리 짝지어 해석한다.
3. **0번 보존·복구** — `line0_correct`가 정상 12/12, 오류 9/12에서 어떻게 변했는지 확인한다. 주문 성공 기준을 충족해도 복구가 줄었다면 두 결과를 함께 보고한다. `pairs`에서 같은 주문의 전후 결과를 확인할 수 있다.

검증 완료 후 ZIP 하나를 만든다.

```bash
ZIP="$HOME/$(basename "$OUT").zip"
.venv-hf/bin/python -m zipfile -c "$ZIP" "$OUT"
.venv-hf/bin/python -m zipfile -t "$ZIP"
ls -lh "$ZIP"
explorer.exe "$(wslpath -w "$HOME")"
```

탐색기에서 방금 생성된 **`final_h3_….zip`**을 공유한다. 여기에는 새 실행 원문, 전후 비교표, 커밋 번호와 콘솔 기록이 들어 있다. 팀이 전후를 재검증할 때는 기존 `h3_validation_7wZ8VQF3.zip`도 함께 사용한다. **기준 충족 여부와 관계없이 이 결과 공유로 개인 실험을 마친다.**
