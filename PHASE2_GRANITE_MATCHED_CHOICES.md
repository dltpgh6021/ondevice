# Granite FP16에서 같은 상품을 한 줄씩 고르는 비교

**이번에 직접 돌릴 실험은 한 묶음이다.** 원격 PC에서 한 번 실행하면 36개 메뉴를 자동으로 처리한다. 지난 h3 48회 결과와 이미 설치한 Granite를 사용한다. 새 모델 다운로드, 양자화, 서버 실행은 필요 없다.

## 무엇을 확인하나

지난 실험에서 **정상 재개 + 검증 보고서 사전 제공** 조건을 선택한다. 시작 시 0번이 이미 올바르므로, 0번 복구의 영향을 줄이고 평소 상품 선택에 집중할 수 있다. 12개 주문의 1·2·3번 상품에서 첫 저장은 36건 중 24건이 정답이었다. 예산 초과가 4건, 예산 안에서 더 높은 평점의 상품을 놓친 경우가 8건이었다.

이번에는 각 첫 저장 **직전에 모델이 실제로 본** 예산, 상품 ID, 가격, 평점, 후보 순서를 로그에서 복사한다. 이전 선택에 따라 달라지는 2·3번 메뉴도 당시의 후보를 사용한다. 정답인 경우와 틀린 경우를 모두 포함한다.

| 같은 메뉴에서 비교하는 값 | 측정 방법 |
|---|---|
| 기존 h3의 첫 선택 | 이전 로그에서 해당 줄에 처음 저장한 상품을 채점 |
| 단독 과제의 첫 선택 | 같은 후보 한 줄만 제시하고 처음 저장한 상품을 채점. 저장 없이 끝내면 `no_save`로 포함 |
| 단독 과제의 최종 선택 | 중간에 상품을 바꿨는지 확인하는 보조 지표 |

모델·정밀도·도구 세 종류·응답 상한(8회)·상품 선택 규칙은 동일하다. 단독 과제는 빈 상태에서 한 줄만 요청하고 이전 주문 대화와 사전 보고서를 제거한다. 따라서 차이가 나면 **한 줄로 제시한 조건에서 판단이 달라졌다**고 해석한다. 대화 길이 하나만의 영향으로 단정하기는 어렵다. 이 비교의 단위는 12개 주문에 속한 36개 선택이다.

## 1. 로컬 Windows CMD에서 업로드

추가 파일은 실행기, 첫 선택과 재선택을 구분하는 테스트, 이 가이드다. 기존 48회 실험 코드는 그대로 사용한다.

```cmd
cd /d "%USERPROFILE%\Documents\Codex\2026-09-30\referenced-chatgpt-conversation-this-is-an\ondevice-phase2"
git branch --show-current
git status --short
```

현재 브랜치가 `experiment/phase2-position-controls`인지 확인한 뒤 실행한다.

```cmd
git add scripts/compare_phase2_granite_choices.py tests/test_phase2_granite_choices.py PHASE2_GRANITE_MATCHED_CHOICES.md
git commit -m "Compare Granite single-line decisions with observed h3 choices"
git push origin experiment/phase2-position-controls
git log -1 --oneline
```

## 2. 원격 WSL에서 지난 결과 확인

**지난 48회를 돌린 원격 PC**에서 아래 명령을 실행한다. 최초 Granite 6회 폴더와 구분하기 위해 이번 비교는 h3 48회 폴더 안의 `run`을 입력으로 받는다.

```bash
cd "$HOME/ondevice-phase2"
git branch --show-current
```

브랜치가 1절과 같으면 이어서 입력한다.

```bash
git pull --ff-only origin experiment/phase2-position-controls
git log -1 --oneline
.venv-hf/bin/python -m unittest discover -s tests -p "test_phase2*.py" -q
SRC="$(cat results/phase2/latest_h3_validation.txt)/run"
ls "$SRC/summary.json" "$SRC/trials.jsonl"
.venv-hf/bin/python scripts/compare_phase2_granite_choices.py plan --source "$SRC"
nvidia-smi
```

테스트 마지막은 `OK`여야 한다. `plan`은 GPU 추론 없이 48회 기록을 재생하고 다음 숫자를 출력한다. 항목 순서는 바뀔 수 있다.

```text
Menus: 36. Original h3 first choices: {'correct': 24, 'over_budget': 4, 'lower_rating': 8}
```

경로나 숫자가 다르면 사용한 이전 결과를 확인한다. 이번에 공유한 ZIP의 폴더명은 `h3_validation_7wZ8VQF3`이다. 원격에 그 폴더가 그대로 있으면 `SRC="$PWD/results/phase2/h3_validation_7wZ8VQF3/run"`으로 지정하고 `plan`을 다시 실행할 수 있다.

## 3. 단독 메뉴 36개 실행

아래 블록을 같은 원격 WSL 터미널에서 한 번 실행한다. 이전 실험과 동일한 모델 파일·설정·템플릿을 검사하고 워밍업 2회 후 시작한다.

```bash
OUT="$(mktemp -d "$PWD/results/phase2/granite_single_XXXXXXXX")"
printf '%s\n' "$OUT" > results/phase2/latest_granite_single.txt
printf '%s\n' "$SRC" > "$OUT/source_path.txt"
git rev-parse HEAD > "$OUT/project_commit.txt"
set -o pipefail
.venv-hf/bin/python -u scripts/compare_phase2_granite_choices.py run --source "$SRC" --out "$OUT/run" 2>&1 | tee "$OUT/console.log"
```

진행률은 `[1/36]`부터 `[36/36]`까지다. `lower_rating`, `over_budget`, `no_save`도 기록할 결과다. 마지막에 **`Verified 36 paired menus.`**가 표시된다. 실제 실행 시간은 아직 측정 전이다. 첫 몇 건이 지난 뒤 콘솔 진행률로 남은 시간을 추정할 수 있다. 중단 시에는 부분 로그를 보존하고 원인을 확인한 뒤 이 절의 첫 줄부터 새 폴더에서 재실행한다.

## 4. 확인하고 ZIP 공유

새 WSL 터미널에서도 아래 블록을 사용할 수 있다. `check`는 모델을 다시 실행하지 않고 기록을 재생해 메뉴·선택·집계를 대조한다.

```bash
cd "$HOME/ondevice-phase2"
OUT="$(cat results/phase2/latest_granite_single.txt)"
SRC="$(cat "$OUT/source_path.txt")"
.venv-hf/bin/python scripts/compare_phase2_granite_choices.py check --source "$SRC" --out "$OUT/run"
```

검증이 성공하면 ZIP을 만든다.

```bash
ZIP="$HOME/$(basename "$OUT").zip"
.venv-hf/bin/python -m zipfile -c "$ZIP" "$OUT"
.venv-hf/bin/python -m zipfile -t "$ZIP"
ls -lh "$ZIP"
explorer.exe "$(wslpath -w "$HOME")"
```

방금 생성한 **`granite_single_….zip`**을 공유하면 된다. 메뉴, 이전 첫 선택, 새 대화 기록, 결과표가 들어 있다. 원격 재검증에 쓸 지난 h3 결과 폴더도 보관한다.

## 5. 어떻게 읽고 어디서 멈출까?

`run/summary.json`에서 먼저 `outcomes.h3_first`와 `outcomes.isolated_first`를 비교한다. `changes.gained`는 h3에서 틀렸지만 단독으로 맞힌 메뉴, `changes.lost`는 반대 방향이다. `both_failed`는 두 번 모두 틀린 메뉴다. 총 정답 수가 같아도 어떤 과제들이 서로 뒤바뀌었는지 본다. `decisions`에서 각 메뉴의 오류 종류와 단독 최종 결과도 확인할 수 있다.

| 관측 결과 | 다음 행동 |
|---|---|
| 단독으로 제시하면 여러 주문에서 오답이 줄고 반대로 틀리는 사례는 적음 | 주문 상태에서 현재 선택할 줄과 후보를 더 명확히 보여주는 수정 한 가지를 고른다. 선택 규칙과 정답을 유지하고 정상·오류 양쪽에 같이 적용한다. |
| 단독에서도 같은 예산 초과나 낮은 평점 선택이 반복됨 | 이 모델·선택 규칙 조합의 한계로 기록한다. 팀이 과제 표현 한 가지를 바꿔 볼 근거가 있다고 합의할 때만 FP16으로 확인한다. |
| 개선과 악화가 섞이거나, 저장 없이 끝내는 사례가 많음 | 대표 사례와 도구 기록을 팀에 공유한다. 동일 입력을 반복해서 점수가 오르기를 기다리는 방식은 여기서 끝낸다. |

**개인 GPU 테스트는 이번 1회 + 필요할 때 확인 1회로 마치는 것을 권한다.** 이는 업무량 계획이다. 양자화 성능 차이가 나오거나 논문 가설이 지지될 때까지 필요한 실험 횟수는 미리 정할 수 없다.

- **이번 결과만으로 다음 설계를 선택하기 어렵다면** 36개 비교까지로 개인 진단을 마친다. 같은 모델·같은 입력의 반복 실행은 새로운 설계 근거를 거의 더하지 않는다.
- **바꿀 근거가 있다면** 결과를 보기 전에 수정 한 가지와 성공 기준을 기록한다. 같은 12개 개발 주문의 h3 정상·오류 재개를 한 묶음으로 다시 확인한다. 이전 계획의 주문 전체 성공 기준은 정상 **8/12**, 오류 **4/12**다. 0번 복구와 정상 항목 보존도 별도 보고한다. 이 후속 실행은 수정안이 정해진 뒤 준비한다.
- **그 확인 뒤에는 결과와 무관하게 팀에 인계한다.** 기준에 도달하면 팀이 고정한 설계로 h1·h3·h5 FP16 전체 개발 검증을 수행하고 양자화 본실험 진행 여부를 결정한다. 미달이면 현재 기준 성능과 남은 한계를 공유하고 팀이 범위를 결정한다. 이번까지 여러 번 본 개발 과제는 최종 논문 평가용 미사용 과제와 구분한다.
