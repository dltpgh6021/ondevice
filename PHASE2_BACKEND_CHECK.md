# 공식 원본과 GGUF FP16 소규모 비교

같은 3개 개발 시나리오 `dev-000~002`의 h3 정상·오류 재개를 실행한다. 경로별 6회, 합계 12회다. 기존 사용자 질문을 사용한다. 같은 초기 메시지·저장 상태·후보·도구 정의·응답 상한 8회를 적용하고, 경로별 후속 관측은 실제 선택에 따라 생성한다.

현재 GGUF의 템플릿 소스는 공식 원본의 소스와 다르다. 같은 Jinja 환경에서 기존 72개 시작 입력을 렌더링하면 결과 문자열은 모두 같았다. 템플릿 소스의 차이가 이번 실패의 원인이라고 단정할 근거는 확보되지 않았다. 비교 실행에서도 서버의 `/apply-template`와 공식 tokenizer가 생성한 프롬프트를 저장한다.

공식 체크포인트는 BF16으로 배포되며 이번에는 **FP16으로 로드**한다. 사용한 GGUF 역시 F16이다. 공식 리비전은 `cdbee75f17c01a7cc42f958dc650907174af0554`로 고정한다. 두 경로 모두 greedy, seed 42, 응답당 512토큰을 사용한다. 비교는 변환·런타임·도구 출력 처리의 영향을 함께 포함한다. [공식 원본](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [Transformers 도구 호출](https://huggingface.co/docs/transformers/en/chat_extras)

## 준비하는 파일

| 파일 | 역할 |
|---|---|
| `src/transformers_client.py` | 공식 모델의 GPU 실행과 Qwen 도구 태그를 공통 행동으로 파싱 |
| `scripts/compare_phase2_backends.py` | 6개 입력 고정, 두 경로 실행, 입력 일치 검사와 결과 집계 |
| `requirements-transformers.txt` | 비교에 쓰는 추가 Python 패키지 |

각 모델 응답의 원문, 생성 종료 이유, 파싱된 도구 호출과 실제 도구 실행 결과를 로그에 담는다. 공식 원본은 구조화된 태그를 생성한 뒤 공통 단일 행동 규칙으로 파싱한다. llama.cpp는 서버의 파서·생성 제약도 관여하므로 두 경로의 형식 오류 차이도 함께 확인한다.

## 1. 코드 업데이트와 별도 가상환경

원격 WSL Ubuntu의 프로젝트 루트에서 실행한다. Git 업로드 후 가져온다.

```bash
cd "$HOME/ondevice-phase2"
git pull --ff-only origin experiment/phase2-position-controls
python3 -m venv .venv-hf
.venv-hf/bin/python -m pip install --upgrade pip
.venv-hf/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.venv-hf/bin/python -m pip install -r requirements-transformers.txt
.venv-hf/bin/python -m unittest discover -s tests -p "test_phase2*.py" -v
```

현재 예상 결과는 33개 테스트와 `OK`다.

PyTorch GPU wheel은 CUDA 실행 라이브러리를 포함하며 Windows NVIDIA 드라이버와 함께 동작한다. CUDA Toolkit 12.4로 빌드한 기존 llama.cpp는 그 설정으로 사용한다. 새 환경에서 실제 GPU가 보이는지 확인한다. [PyTorch 2.7.1 설치 명령](https://pytorch.org/get-started/previous-versions/#v271)

```bash
.venv-hf/bin/python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

원본 모델 가중치 약 8GB와 PyTorch 패키지를 추가로 내려받는다. 기존 네트워크 속도라면 원본 가중치만 약 80분이 걸릴 수 있다. 추론 시간은 두 경로를 합쳐 대략 3~10분을 예상하지만 실제 GPU 실행으로 확인해야 한다.

## 2. 실행 계획과 공식 tokenizer 준비

```bash
BASELINE_DIR="$(cat results/phase2/latest_v4_4b_dev.txt)"
CHECK_DIR="$PWD/results/phase2/backend_check_$(date +%Y%m%d_%H%M%S)"
.venv-hf/bin/python scripts/compare_phase2_backends.py prepare --baseline-log "$BASELINE_DIR/dev.jsonl" --output-dir "$CHECK_DIR" --dry-run
```

예상 출력은 `trials_per_backend: 6`, `total: 12`다. 준비 명령을 실행하면 모델 설정과 tokenizer 파일을 내려받고 입력 계획을 저장한다.

```bash
.venv-hf/bin/python scripts/compare_phase2_backends.py prepare --baseline-log "$BASELINE_DIR/dev.jsonl" --output-dir "$CHECK_DIR"
printf '%s\n' "$CHECK_DIR" > results/phase2/latest_backend_check.txt
```

`CHECK_DIR`는 실행기가 생성한다. 공식 원본 파일은 `models/qwen3-4b-instruct-2507-hf`에 저장되고, 여기에는 뒤의 hf 실행 단계에서 가중치도 다운로드한다. 모델 폴더는 결과 ZIP에서 제외한다.

## 3. 기존 GGUF FP16에서 6회 실행

기존 서버의 현재 PID를 확인한다. 서버를 다시 시작했다면 현재 PID를 지정한다. 기본 파일럿과 같은 서버 실행 명령은 README 5절에 있다.

```bash
MODEL_FILE="$PWD/models/qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-F16.gguf"
SERVER_PID="$(cat "$BASELINE_DIR/server.pid")"
ps -ww -p "$SERVER_PID" -o pid=,args=
curl -fsS --max-time 10 http://127.0.0.1:18080/health
```

해당 프로세스가 4B F16 서버이고 health가 `{"status":"ok"}`이면 실행한다.

```bash
.venv-hf/bin/python -u scripts/compare_phase2_backends.py run --backend gguf --check-dir "$CHECK_DIR" --model-path "$MODEL_FILE" --server-pid "$SERVER_PID"
```

`[6/6]`까지 완료되면 `$CHECK_DIR/gguf`에 `trials.jsonl`, `summary.json`, `metadata.json`, 서버 템플릿과 preflight를 저장한다. 행동 실패는 그대로 기록하고 다음 과제로 진행한다. 인프라·내부 오류는 부분 로그를 남기고 종료한다.

## 4. 서버 종료 후 공식 원본에서 6회 실행

지금 실행 중인 GGUF 서버를 확인한 뒤 종료한다. 두 모델을 한꺼번에 GPU에 올리면 10GB VRAM을 초과한다.

```bash
ps -ww -p "$SERVER_PID" -o pid=,args=
kill "$SERVER_PID"
nvidia-smi
```

프로세스가 종료돼 GPU 메모리가 해제됐는지 확인한다. 그다음 공식 원본을 실행한다.

```bash
.venv-hf/bin/python -u scripts/compare_phase2_backends.py run --backend hf --check-dir "$CHECK_DIR"
```

처음에는 고정 리비전의 공식 가중치를 내려받고 해시를 기록한다. 그 뒤 FP16·SDPA로 GPU에 로드하고 워밍업 2회와 과제 6회를 실행한다. `[6/6]`까지 완료되면 `$CHECK_DIR/hf`에 결과와 모델·tokenizer·라이브러리·생성 설정을 저장한다. CUDA 메모리 부족이 발생하면 traceback을 공유한다. 과제 실패와 메모리 오류의 분모를 섞지 않는다.

## 5. 두 경로 비교와 결과 공유

```bash
.venv-hf/bin/python scripts/compare_phase2_backends.py compare --check-dir "$CHECK_DIR"
.venv-hf/bin/python -m zipfile -c "$HOME/phase2_backend_check.zip" "$CHECK_DIR"
explorer.exe "$(wslpath -w "$HOME")"
```

`comparison.json`은 두 경로의 시작 입력·상태가 같은지 검사한 뒤 조건별 전체 성공, 0번 최종 정답·재저장, 묶음 최초 선택 정확도, 형식 반려를 집계한다. `initial_rendered_prompts_equal`은 동일 시작 메시지를 실제 실행 경로의 템플릿으로 만든 문자열의 일치 여부다. 서버가 `/apply-template`를 제공하지 않으면 null과 원인을 기록하며, 문자열 일치는 그 결과를 통해 확인해야 한다.

탐색기에서 `phase2_backend_check.zip`을 공유한다. 공식 원본에서 개선이 보이면 템플릿·토큰화·변환·출력 파싱을 살핀다. 양쪽 성능이 낮으면 같은 개발 과제에서 교체 모델 후보를 검토한다. 3개 시나리오는 실행 경로를 점검하는 개발용 표본이며, 최종 정밀도 비교의 결론은 더 큰 미사용 과제에서 확인한다.

새 터미널에서 이어갈 때는 프로젝트 루트로 이동하고 `CHECK_DIR="$(cat results/phase2/latest_backend_check.txt)"`를 다시 지정한다. 이미 결과가 있는 backend 폴더에는 덮어쓰기를 막는다. 인프라 오류를 해결한 뒤 재실행하려면 새 CHECK_DIR로 prepare부터 진행한다. 같은 모델 파일은 캐시에서 재사용한다.
