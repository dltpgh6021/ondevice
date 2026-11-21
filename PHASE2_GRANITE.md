# Granite FP16 소규모 비교 실행 가이드

기존 Qwen3-4B-Instruct-2507에서 사용한 **h3 개발 시나리오 3개 × 정상·오류 재개 2조건 = 6회**를 IBM Granite-4.0-Micro로 실행한다. 이전 공식 Qwen 원본의 6회 결과를 재사용하므로 Qwen을 다시 다운로드하거나 실행하지 않는다. 같은 PC에서 진행한다.

현재 목적은 양자화할 기준 모델을 선별하는 것이다. Granite의 선택 정확도와 기존 오류 수정 행동이 개선되는지 확인한다. 이번 소수 결과만으로 본실험을 시작하거나 모델의 일반 성능을 결론 내리지는 않는다.

## 사용할 모델과 파일

모델은 **`ibm-granite/granite-4.0-micro`**다. 이름에 `base` 또는 `h-micro`가 붙는 모델과 구분한다. 공식 리비전을 `56111ae135df9c53a78c99028e7bc24035a9e979`로 고정했다. 배포 체크포인트는 BF16이고, 이번에는 **FP16으로 로드**한다. Q8·Q4는 기준 성능을 확인한 뒤 만든다.

공식 파일 기준 가중치는 약 **6.81GB**다. 가중치 외에 대화 기록과 실행용 GPU 메모리가 필요하다. 기존 RTX 3080 10GB, WSL RAM 약 15GB에서 실제 로딩과 추론을 확인한다. [공식 모델 설명](https://huggingface.co/ibm-granite/granite-4.0-micro), [고정한 모델 설정](https://huggingface.co/ibm-granite/granite-4.0-micro/blob/56111ae135df9c53a78c99028e7bc24035a9e979/config.json)

| 파일 | 하는 일과 필요한 이유 |
|---|---|
| `src/granite_client.py` | Granite의 공식 대화 형식과 종료 토큰을 사용해 FP16으로 실행한다. 원문·토큰 기록과 단일 도구 호출 파싱은 기존 비교 방식으로 남긴다. |
| `scripts/check_phase2_granite.py` | 모델 다운로드·해시 기록, 기존 Qwen 입력 검증, Granite 6회 실행, 대응 결과 비교를 수행한다. |
| `tests/test_phase2_granite.py` | 잘못된 모델 설정, 누락·중복 실행, 시작 상태 변경, 도구 결과·점수 불일치를 검사한다. |
| 이 가이드 | 로컬 코드 업로드부터 원격 결과 ZIP 공유까지 안내한다. |

사용자 질문·후보·시작 상태·도구 정의·채점은 바꾸지 않는다. 두 모델 모두 greedy 선택, seed 42, **응답당 512토큰·과제당 최대 응답 8회**, 워밍업 2회다. 도구가 연속 호출되면 각각 한 응답을 사용한다. 정상 상태의 0번은 유지하고, 오류 상태의 0번은 수정하면서 남은 묶음 항목 세 개를 완료해야 한다.

모델별 공식 대화 템플릿은 다르다. 같은 의미의 메시지와 상태를 각각의 공식 템플릿으로 전달하며, 서로 다른 모델의 렌더링 문자열까지 같게 만들지는 않는다. 이번에는 **Transformers로 공식 모델을 실행**하므로 llama.cpp 재설치·재빌드는 필요 없다.

## 0. 로컬 Windows에서 코드 업로드

이 절만 **로컬 Windows CMD**에서 실행한다. 아래는 이번 수정 파일이 있는 저장소 위치다. `Desktop/ondevice`에 있는 다른 복사본과 혼동하지 않는다.

```cmd
cd /d "%USERPROFILE%\Documents\Codex\2026-09-30\referenced-chatgpt-conversation-this-is-an\ondevice-phase2"
git branch --show-current
git status --short
```

브랜치는 `experiment/phase2-position-controls`여야 한다. 해당 브랜치에서 이번 다섯 파일만 올린다.

```cmd
git add README.md PHASE2_GRANITE.md src/granite_client.py scripts/check_phase2_granite.py tests/test_phase2_granite.py
git commit -m "Add Granite FP16 candidate screening and WSL guide"
git push origin experiment/phase2-position-controls
```

예상 시간은 약 1~2분이다. 업로드가 완료돼야 원격 PC가 수정 파일을 받을 수 있다. 코드 업로드는 사용자가 수행한다.

## 1. 원격 WSL에서 코드와 환경 확인

이후 명령은 모두 **이전 공식 Qwen 비교를 실행했던 원격 PC의 Ubuntu/WSL bash**에서 실행한다. VS Code의 WSL 터미널을 사용하면 된다. 가상환경 활성화 여부와 관계없이 `.venv-hf/bin/python`을 직접 지정한다.

```bash
cd "$HOME/ondevice-phase2"
git switch experiment/phase2-position-controls
git pull --ff-only origin experiment/phase2-position-controls
git log -1 --oneline
ls scripts/check_phase2_granite.py src/granite_client.py PHASE2_GRANITE.md
ls -l .venv-hf/bin/python
```

파일 세 개가 보여야 한다. `git pull`이 로컬 수정 때문에 거절되면 강제 초기화하지 말고 그 출력을 확인한다. `.venv-hf`가 이미 있으면 다음 두 명령으로 환경과 GPU를 확인한다.

```bash
.venv-hf/bin/python -m pip install -r requirements-transformers.txt
.venv-hf/bin/python -c "import torch, transformers; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

기존 환경의 예상 출력은 torch `2.7.1+cu126`, Transformers `4.57.6`, `CUDA: True`, RTX 3080이다. 기존 환경에서는 약 1~2분이다.

`.venv-hf/bin/python`이 없을 때만 아래 설치를 먼저 한다. 패키지 다운로드 때문에 약 5~20분 이상 걸릴 수 있다.

```bash
python3 -m venv .venv-hf
.venv-hf/bin/python -m pip install --upgrade pip
.venv-hf/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.venv-hf/bin/python -m pip install -r requirements-transformers.txt
```

## 2. 자동 검증과 이전 Qwen 결과 찾기

```bash
.venv-hf/bin/python -m unittest discover -s tests -p "test_phase2*.py" -v
```

이번 코드 기준 예상 결과는 **40개 테스트와 `OK`**다. 약 5~15초이며 모델 추론은 하지 않는다.

```bash
CHECK_DIR="$(cat results/phase2/latest_backend_check.txt)"
REFERENCE_DIR="$CHECK_DIR/hf"
printf 'Qwen reference: %s\n' "$REFERENCE_DIR"
ls -lh "$REFERENCE_DIR/metadata.json" "$REFERENCE_DIR/trials.jsonl"
cat "$REFERENCE_DIR/summary.json"
```

앞서 공유한 결과라면 정상 재개는 `successes: 1`, 오류 재개는 `successes: 0`이고 각 조건의 `n`은 3이다. `latest_backend_check.txt`가 없다면 먼저 실제 폴더를 찾는다.

```bash
find "$PWD/results/phase2" -path '*/hf/trials.jsonl' -print
```

해당 결과의 `hf` 폴더 절대경로를 `REFERENCE_DIR`로 지정한다. 폴더에는 `metadata.json`, `trials.jsonl`이 있고, 부모 폴더에는 `check.json`이 있어야 한다. 결과 ZIP만 남아 있다면 이 구조를 유지해 풀어서 사용한다. 모델 파일을 ZIP에 넣을 필요는 없다.

다른 PC의 결과를 그대로 기준으로 지정하면 실행기가 장비 불일치로 중단한다. 그 경우 [공식 원본 비교 가이드](PHASE2_BACKEND_CHECK.md)에 따라 새 PC에서 Qwen 기준을 먼저 준비한다.

## 3. 실행 계획 확인과 Granite 다운로드

새 결과 폴더 이름을 정한다. 이 시점에는 `mkdir`로 결과 폴더를 만들지 않는다. 실행기가 준비가 끝난 후 생성한다.

```bash
RUN_DIR="$PWD/results/phase2/granite_$(date +%Y%m%d_%H%M%S)"
.venv-hf/bin/python scripts/check_phase2_granite.py prepare --reference-dir "$REFERENCE_DIR" --output-dir "$RUN_DIR" --dry-run
```

약 1~5초 후 `model_id: ibm-granite/granite-4.0-micro`, `trials: 6`, `horizon: 3`, `max_response_turns: 8`, `max_tokens_per_response: 512`가 나오면 입력 검증이 통과한 것이다. 모델 다운로드와 추론은 아직 하지 않는다.

```bash
.venv-hf/bin/python -u scripts/check_phase2_granite.py prepare --reference-dir "$REFERENCE_DIR" --output-dir "$RUN_DIR"
```

공식 모델은 `models/granite-4.0-micro-hf`에 저장된다. 가중치 두 파일과 설정·토크나이저를 다운로드하고 SHA256을 기록한다. 기존 네트워크 약 1.7MiB/s라면 **약 60~75분**, 더 빠른 회선에서는 짧아진다. 같은 리비전의 다운로드 캐시가 있으면 재사용한다. 예상 마지막 출력은 다음과 같다.

```text
PREPARED: six paired Granite trials. No GPU inference has run yet.
```

이 문구가 나온 뒤 위치를 저장한다.

```bash
printf '%s\n' "$RUN_DIR" > results/phase2/latest_granite_check.txt
cat "$RUN_DIR/model_manifest.json"
```

`model_manifest.json`은 모델 ID·고정 리비전·원본 BF16/실행 FP16·각 파일 해시·가중치 크기를 기록한 식별표다. 여기까지는 GPU에 모델을 올리지 않으므로 기존 서버가 있어도 다운로드할 수 있다.

## 4. GPU 메모리 비우기

Granite와 기존 Qwen 서버를 동시에 GPU에 올리지 않는다. 현재 사용 중인 프로세스를 확인한다.

```bash
nvidia-smi
pgrep -af '[l]lama-server'
```

이전 실험용 llama-server가 남아 있다면 표시된 PID와 모델 경로를 확인한다. **그 서버를 종료해도 되는 경우에만** 아래 명령에서 PID를 입력한다. 서버가 없고 GPU 메모리가 해제돼 있으면 이 블록을 건너뛴다.

```bash
read -r -p "Previous llama-server PID to stop: " SERVER_PID
ps -ww -p "$SERVER_PID" -o pid=,args=
kill "$SERVER_PID"
nvidia-smi
```

다른 사용자의 작업이 GPU를 사용 중이면 그 작업이 끝난 뒤 진행한다. 예상 시간은 수 초다. Granite는 이번 Python 실행기가 직접 로드하므로 새 llama-server나 PID를 지정하지 않는다.

## 5. Granite FP16 6회 실행

```bash
set -o pipefail
.venv-hf/bin/python -u scripts/check_phase2_granite.py run --run-dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/console.log"
```

해시 검증 → FP16 GPU 로딩 → 워밍업 2회 → 과제 6회 순서다. 준비된 모델의 추론은 **약 3~15분으로 추정**하며 이 장비에서의 Granite 실측 시간은 아직 없다. 응답이 길어지면 더 걸릴 수 있다. 다음 형태로 진행 상황이 출력된다.

```text
[1/6] dev-000 normal_restart success=True 또는 False
[2/6] dev-000 error_restart success=True 또는 False
...
[6/6] dev-002 error_restart success=True 또는 False
COMPLETED: six valid trials. Success=False is a model outcome, not an execution error.
```

위 성공 여부는 출력 형식 예시다. 실제 성공 건수는 실행 후 확인한다. `False`는 과제 실패 결과이므로 다음 과제를 계속 실행한다. 인프라·내부 오류는 부분 로그를 보존하고 중단하며, 성공률 분모에 정상 결과처럼 섞지 않는다.

실행 중 다른 터미널에서 확인하려면 프로젝트 폴더로 이동한 뒤 아래 명령을 사용한다.

```bash
RUN_DIR="$(cat results/phase2/latest_granite_check.txt)"
tail -f "$RUN_DIR/console.log"
```

`Ctrl+C`는 이 `tail` 화면만 닫는다. 모델을 실행 중인 원래 터미널에서 누르면 실험이 중단되므로 창을 구분한다.

## 6. Qwen과 비교하고 결과 공유

`COMPLETED`를 확인한 후 실행한다.

```bash
.venv-hf/bin/python scripts/check_phase2_granite.py compare --run-dir "$RUN_DIR"
```

약 1~5초 후 조건별 집계와 여섯 대응 결과가 출력되고 `comparison.json`으로 저장된다. 시작 메시지·상태와 도구 결과를 다시 검사하므로 불완전한 결과를 정상 비교로 처리하지 않는다. 비교 명령은 새 파일을 만들며 기존 `comparison.json`이 있으면 덮어쓰지 않는다. 이미 만든 결과는 `cat "$RUN_DIR/comparison.json"`으로 본다.

| 지표 | 해석 |
|---|---|
| `successes` | 0번과 남은 세 항목을 모두 올바르게 완료한 과제 수 |
| `accessory_write_episodes` | 0번을 다시 저장한 실행 수. 수정 시도 자체이며 올바르게 고쳤다는 뜻은 아니다. |
| 오류 조건의 `accessory_correct_at_end` | 잘못 주어진 0번을 최종적으로 올바르게 복구한 과제 수 |
| 정상 조건의 `accessory_correct_at_end` | 원래 올바른 0번을 끝까지 올바르게 유지한 과제 수 |
| `first_bundle_saves_correct` | 새로 수행한 항목별 첫 선택의 정답 수. 조건별 분모는 9회다. |
| `format_rejections` | 모델 출력이 행동 형식에 맞지 않아 거절된 횟수 |

먼저 정상 선택과 실제 회복 행동이 함께 개선되는지 본다. 0번 수정이 늘어도 잘못 바꾸거나 최종 과제를 끝내지 못할 수 있어 전체 성공을 함께 확인한다. 개선 가능성이 보이면 기존 dev 전체 평가로 넘어갈지 판단한다. 이번 결과는 기준 모델 선별용이고 모델별 속도나 양자화 효과의 본실험 자료로 사용하지 않는다.

결과는 Git에서 제외되므로 ZIP으로 공유한다. 이전 Qwen 비교 결과도 함께 넣어 입력과 대응 결과를 다시 확인할 수 있게 한다. 모델과 가상환경은 포함하지 않는다.

```bash
.venv-hf/bin/python -m zipfile -c "$HOME/phase2_granite_check.zip" "$RUN_DIR" "$CHECK_DIR"
explorer.exe "$(wslpath -w "$HOME")"
```

약 수 초 후 Windows 탐색기에 `phase2_granite_check.zip`이 보인다. ZIP을 공유하거나 팀 드라이브에 올린다. `REFERENCE_DIR`를 직접 지정했다면 ZIP 명령 전에 `CHECK_DIR="$(dirname "$REFERENCE_DIR")"`로 부모 폴더를 지정한다.

## 창을 새로 열거나 오류가 났을 때

새 터미널에서 실행을 이어갈 때는 변수를 복원한다.

```bash
cd "$HOME/ondevice-phase2"
RUN_DIR="$(cat results/phase2/latest_granite_check.txt)"
CHECK_DIR="$(cat results/phase2/latest_backend_check.txt)"
REFERENCE_DIR="$CHECK_DIR/hf"
```

- **다운로드가 끊김:** 같은 3절 `prepare` 명령을 다시 실행한다. 준비가 완료되기 전에는 결과 폴더를 만들지 않으며, 다운로드 캐시를 재사용한다.
- **CUDA out of memory:** `console.log`와 `nvidia-smi` 결과를 공유한다. 과제 실패로 해석하지 않는다. GPU에 다른 모델이 남아 있는지 먼저 확인한다.
- **실행 중 중단되거나 로딩 실패:** 기존 결과 폴더를 보존한다. 원인을 해결한 후 3절에서 새 `RUN_DIR`를 만들어 prepare부터 재실행한다. 다운로드한 모델은 재사용하며 이번 6회 실행에는 중간 재개 기능을 넣지 않았다.
- **코드·모델·참조 결과가 변경됨:** 실행기가 해시 불일치로 중단한다. 예전 결과에 새 코드를 섞지 말고 새 준비 폴더를 사용한다.
- **정상·오류 조건 모두 낮은 성공률:** 명령 오류와 구분한다. ZIP의 선택·수정·종료 기록을 확인한 뒤 다음 모델이나 과제 표현 검토 여부를 결정한다.

원격 PC에서 실제 Granite GPU 추론은 사용자가 위 절차로 수행한다. 로컬 검증 결과와 실제 모델 성능을 구분한다.
