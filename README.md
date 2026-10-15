# Quantized Agent Stability

양자화가 장기 Agent의 수행 안정성에 미치는 영향을 비교하는 실험 저장소입니다.

**현재 단계:** execution 60개를 `execution-v2`로 재설계했습니다. 기존 recovery 60개는 아직 `legacy-v1`이며 새 execution과 의미상 대응하지 않습니다. 회복 실험과 오류 지표 보완이 끝나기 전에는 본실험이나 `recovery_gap` 해석을 시작하지 않습니다. 아래 recovery 실행법은 구버전 동작을 확인하는 개발용 설명입니다.

이 프로젝트는 FP16·Q8·Q4 모델을 같은 Agent framework와 동일한 mock tool 환경에서 실행합니다. 최종 성공률만 비교하지 않고, 다음 두 현상을 분리해 측정합니다.

1. 모델이 오류를 얼마나 자주 발생시키는가
2. 같은 오류가 주어졌을 때 얼마나 잘 회복하는가

## 실험 조건과 task 유형

- `free`: `execution` task를 초기 상태에서 처음부터 수행합니다. 양자화 수준과 horizon에 따른 도구 선택·정상 수행을 측정합니다.
- `recovery`: `recovery` task를 이미 잘못된 상태에서 시작합니다. 같은 오류를 받은 뒤 복구하는 능력을 측정합니다.
- `normal_restart`: 모든 모델에 동일한 올바른 중간 상태를 제공합니다.

execution-v2의 `horizon`은 제공된 정상 시작 상태에서 완료까지 **남아 있는 필수 도구 호출 수**입니다. 5단계 작업 흐름 중 h1에는 앞 4단계, h3에는 앞 2단계의 정상 도구 결과가 대화 기록으로 제공됩니다. h5는 요청 조회부터 모델이 수행합니다. 미리 제공한 호출과 종료 응답은 horizon에 포함하지 않습니다. 실행 중 실제 호출 수·추가 호출 수는 별도로 기록합니다.

## 현재 구현 범위

현재 파일에는 새 정상 수행 60개와 구버전 회복 수행 60개, 총 120개 task가 있습니다. execution-v2의 60개는 20개 기본 시나리오의 시작 정보 3조건이며, 서로 독립인 60개 질문이 아닙니다. [시나리오 전체 목록](tasks/EXECUTION_TASKS.md)에서 실제 요청 내용을 확인할 수 있습니다.

- 결정론적 mock tool 환경
- 정상 상태와 오류 상태의 checkpoint 생성
- 오류 상태 주입
- 오류 상태에서 사용하는 회복 전용 질문(`recovery_query`)
- FakeClient를 이용한 모델 없는 smoke test
- llama.cpp 서버와 통신하는 HTTP client
- 성공률·오류 위치·회복률·추가 호출 수 집계
- 실행 환경과 task/model hash 기록

새 execution은 같은 기본 시나리오의 질문·후보·선택 규칙·최종 목표를 h1·h3·h5에서 동일하게 유지합니다. 시작할 때 이미 제공되는 정상 작업 결과의 깊이만 달라집니다. recovery와의 ID 대응 관계는 남아 있지만, 실제 내용의 대응 관계는 다음 회복 task 재설계에서 맞출 예정입니다.

현재 `tasks/tasks.json`에는 다음 120개가 포함되어 있습니다.

| Task 유형 | Horizon | 쇼핑 | 일정 | 이동 | 장소 | 합계 |
|---:|---:|---:|---:|---:|---:|---:|
| 정상 수행 | 1 | 5 | 5 | 5 | 5 | 20 |
| 정상 수행 | 3 | 5 | 5 | 5 | 5 | 20 |
| 정상 수행 | 5 | 5 | 5 | 5 | 5 | 20 |
| 회복 수행 | 1 | 5 | 5 | 5 | 5 | 20 |
| 회복 수행 | 3 | 5 | 5 | 5 | 5 | 20 |
| 회복 수행 | 5 | 5 | 5 | 5 | 5 | 20 |
| 합계 | - | 30 | 30 | 30 | 30 | 120 |

`task_type`이 `execution`이면 정상 수행 task이고, `recovery`이면 회복 task입니다. 회복 task는 checkpoint를 재생하지 않고 처음부터 오류를 주입한 상태에서 시작합니다. 오류 유형은 현재 장바구니의 잘못된 상품, 잘못된 일정, stale 결과를 사용합니다.

이 회복 동작은 구버전 설명이며, 현재 연구용 회복 프로토콜로 검증된 상태는 아닙니다.

## execution-v2의 실제 단계 의존성

| 분야 | 5단계 흐름 |
|---|---|
| 쇼핑 | 구매 요청 확인 → 상품 후보 조회 → 상품 상세 → 구매 견적 준비 → 구매 항목 확정 |
| 일정 | 일정 요청 확인 → 빈 시간대 조회 → 시간대 상세 → 일정 예약 준비 → 일정 확정 |
| 이동 | 이동 요청 확인 → 이동편 후보 조회 → 운임 상세 → 여행 일정 준비 → 여행 일정 확정 |
| 장소 | 방문 요청 확인 → 장소 후보 조회 → 방문 상세 → 방문 계획 준비 → 방문 계획 확정 |

앞 호출이 생성한 검색 컨텍스트·후보·상세·견적/예약 리소스가 다음 호출의 입력입니다. 존재하지 않는 리소스의 식별자를 알아맞혀 넣어도 실행되지 않습니다. 단순히 정답 순서나 조회 횟수를 검사하는 방식은 사용하지 않습니다. 반복 조회나 다른 후보 확인도 허용되며, 최종 저장값이 요청 조건을 만족하는지 채점합니다.

예를 들어 `shopping_001_h5`는 구매 요청을 조회해야 예산 100·평점 4.0 이상·평점 우선 선택·일반 배송이라는 조건을 알 수 있습니다. 후보 결과에서 조건에 맞는 상품을 선택하고, 상세 결과의 배송 옵션을 이용해 견적을 만든 뒤 해당 견적을 확정합니다. 질문에는 정답 상품 ID가 없습니다. h3에는 요청·후보 결과를, h1에는 올바른 견적 준비까지의 결과를 제공하므로 각각 3회·1회가 남습니다.

최종 판정은 저장된 구매/일정/이동/방문 결과를 읽습니다. `last_result`나 마지막 도구 이름은 사용하지 않으며, 완료 후 `inspect_state`를 호출해도 성공이 유지됩니다. 일정의 날짜·시간 조건·인원·진행 방식 등 필요한 정보는 요청/후보/상세 결과에 모두 포함됩니다.

각 시나리오는 적격 후보 2개와 조건을 위반하는 후보 1개, 후보별 옵션 2개를 가집니다. 일부 시나리오는 1순위 점수가 같아 명시된 2순위 기준을 적용해야 합니다. 금액은 가상 단위이고 `start_hour`·`depart_hour`는 해당 날짜의 24시간제 정수 시각입니다.

현재 `natural_error`·`first_error_step`는 여전히 정답 경로와의 순서·인자 비교 방식입니다. 유효한 추가 조회까지 오류로 잡을 수 있으므로 연구용 오류 지표로 사용하기 전에 별도 수정해야 합니다. 이번 작업은 execution 과제와 최종 성공 판정의 재설계이며, 기존 오류 지표·회복 프로토콜·서버 측정 설정의 보완 완료를 뜻하지 않습니다.

이 설계는 **동일 작업 흐름에서 남은 의존 단계 수**의 효과를 비교합니다. 시작 정보량과 이미 완료된 작업도 달라지므로 일반 과제의 길이만 바꾼 결과라고 해석하지 않습니다. h1은 올바르게 준비된 결과를 확정하므로 후보 선택 능력은 h3·h5에서 주로 비교합니다. 현재 4개 분야는 같은 5단계 구조를 사용하며, 다양한 현실 Agent 작업 전체를 대표하지는 않습니다.

## 폴더 구조

```text
ondevice/
├─ README.md
├─ requirements.txt
├─ tasks/tasks.json
├─ src/
│  ├─ tasks.py          # task 로딩과 검증
│  ├─ client.py         # FakeClient와 llama.cpp client
│  ├─ environment.py    # 결정론적 mock tool과 상태 관리
│  ├─ execution_environment.py # 리소스 의존성이 있는 execution-v2 도구
│  ├─ execution_validation.py  # 정답·우회 경로·최종 상태 검증
│  ├─ agent.py          # 공통 Agent 실행 루프
│  ├─ metrics.py        # 성공·오류·회복 지표
│  └─ experiment.py     # batch 실행과 메타데이터 기록
└─ scripts/
   ├─ smoke_test.py     # 모델 없이 전체 흐름 검사
   ├─ validate_tasks.py  # task 중복·누락·대응 관계 검증
   ├─ build_execution_tasks.py # 20개 시나리오 정의와 JSON 일치 검사
   ├─ run_experiment.py # 실제 llama.cpp 실행
   └─ analyze.py        # JSONL 결과 집계
```

모델 파일(`.gguf`), 가상환경, 실행 로그는 Git에 올리지 않습니다.

## 새 PC에서 팀원이 시작하는 순서

이 저장소에는 실험 코드·task·설정만 올립니다. `.gguf` 모델, `.venv`, `logs/`, `results/`는 용량이 크거나 PC별로 달라질 수 있으므로 GitHub에서 내려받지 않습니다. 새 PC에서는 다음 순서로 환경을 다시 만듭니다.

### 0. 저장소 복제

현재 저장소 주소를 예시로 사용했습니다. 저장소 주소가 바뀌면 `git clone`의 주소만 바꾸면 됩니다.

```bash
git clone https://github.com/dltpgh6021/ondevice.git
cd ondevice
```

Windows CMD에서는 다음처럼 실행해도 됩니다.

```cmd
git clone https://github.com/dltpgh6021/ondevice.git
cd ondevice
```

### 1. Python 환경 준비

Python 3.11 이상과 Git을 설치한 뒤, 이 README의 가상환경·패키지 설치 절차를 실행합니다. GPU PC에서도 Python 가상환경은 CPU PC와 같은 방식으로 만듭니다.

```text
python -m pip install -r requirements.txt
python scripts\smoke_test.py
```

`SMOKE TEST PASSED`가 출력되면 모델 없이도 task와 Agent 실행 흐름이 정상입니다. 이 단계가 실패하면 모델을 준비하기 전에 `tasks/`, `src/`, Python 버전을 먼저 확인합니다.

## task 세트 검증

본 실험을 시작하기 전에 다음 명령으로 120개 task의 구성을 검사합니다.

```text
python scripts\validate_tasks.py
```

WSL/macOS에서는 `python scripts/validate_tasks.py`를 사용합니다. `valid: true`는 전체 구성과 새 execution 검증이 통과했다는 뜻이며, 구버전 recovery의 실험 타당성을 보장하지 않습니다. 출력의 `recovery_semantics`에도 재설계 대기 상태가 표시됩니다.

검증 항목은 다음과 같습니다.

- 전체 120개 task
- execution 60개와 recovery 60개
- 각 task 유형·horizon 조합별 20개
- 중복 task ID 없음
- 모든 execution task에 대응하는 recovery task 존재
- 대응 task의 horizon과 domain 일치
- execution의 20개 기본 시나리오가 h1·h3·h5에서 같은 질문·환경·목표를 유지
- 요청의 조건과 우선순위로 유일한 정답을 독립적으로 계산
- 60개 정답 경로 실행 성공 및 완료 후 조회에도 성공 유지
- 정답 경로의 모든 부분 경로 780개, 인접 순서 교환 120개에서 우회 성공 없음
- 기존 장바구니/일정 도구를 통한 우회 차단, 다른 후보·옵션의 잘못된 완료 거부

이 검사는 정답 부분 경로와 리소스 의존성을 검사합니다. 임의의 모든 도구 인자·행동열을 전수 탐색한 검사는 아닙니다. JSON과 시나리오 정의의 일치는 Windows에서 `python scripts\build_execution_tasks.py --check`, WSL/macOS에서 `python scripts/build_execution_tasks.py --check`로 확인합니다. 모델 없이 정답을 재생하는 검사는 모델의 실제 수행 능력 검증과 구분합니다.

### 2. llama.cpp 준비

새 PC에서 `llama`와 `llama-quantize` 명령이 인식되어야 합니다.

```text
llama version
llama-quantize --help
```

GPU PC에서는 CUDA가 활성화된 llama.cpp 빌드 또는 CUDA 지원 배포본을 사용합니다. FP16·Q8·Q4를 모두 같은 GPU PC에서 실행해야 실행 시간과 메모리를 공정하게 비교할 수 있습니다. `llama` 명령 대신 `llama-server`를 사용하는 배포본이라면 README의 서버 명령에서 실행 파일 이름만 해당 이름으로 바꿉니다.

#### Windows NVIDIA GPU: 사전 빌드 바이너리 사용(권장)

1. NVIDIA 드라이버가 설치되어 있는지 확인합니다.

```cmd
nvidia-smi
```

2. [llama.cpp 공식 Releases](https://github.com/ggml-org/llama.cpp/releases)에서 PC에 맞는 **Windows x64 CUDA** 패키지를 다운로드합니다. NVIDIA GPU라면 CUDA 12 패키지를 우선 사용하고, 릴리스 설명에 표시된 CUDA DLL을 함께 보관합니다.
3. 압축을 예를 들어 `C:\tools\llama-cuda`에 풉니다.
4. 압축을 푼 폴더에서 다음 파일이 있는지 확인합니다. 릴리스에 따라 이름이 `llama.exe` 또는 `llama-server.exe`일 수 있습니다.

```cmd
llama-server.exe --version
llama-quantize.exe --help
```

5. 매번 전체 경로를 쓰지 않으려면 해당 폴더를 PATH에 추가합니다. PATH를 수정하지 않는 경우에는 아래 명령에서 실행 파일의 전체 경로를 사용합니다.

```cmd
set PATH=%PATH%;C:\tools\llama-cuda
llama-server.exe --version
```

#### Windows에서 소스 빌드

사전 빌드 패키지를 사용할 수 없을 때만 다음 방법을 사용합니다. 공식 빌드 문서의 CUDA 옵션은 `GGML_CUDA=ON`입니다.

```cmd
git clone https://github.com/ggml-org/llama.cpp.git C:\tools\llama.cpp
cd /d C:\tools\llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

빌드 후 실행 파일은 보통 `build\bin\Release\` 아래에 있습니다. 이 경로의 `llama-server.exe`, `llama-quantize.exe`를 사용하고, `llama` 통합 명령이 없다면 `quantize_models.py`에 `--llama`와 `--quantizer` 전체 경로를 전달합니다.

공식 서버 문서와 CUDA 빌드 참고:

- [llama.cpp server 사용법](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [llama.cpp CUDA 빌드 문서](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

### 3. 모델 준비

모델은 GitHub에서 복제하지 않습니다. 새 PC에서 다음 중 하나를 수행합니다.

1. 원본 FP16/BF16 GGUF를 받은 뒤 `quantize_models.py`로 self-Q8·self-Q4를 생성합니다.
2. 비교용 reference Q8·Q4는 `llama serve -hf`로 자동 다운로드합니다.

직접 양자화한 파일에는 `self-` 접두사를 사용하고, 제공처에서 미리 받은 파일에는 `reference-` 접두사를 사용하면 결과 파일을 쉽게 구분할 수 있습니다.

```text
self-FP16      = 원본 GGUF
self-Q8        = 같은 원본에서 직접 생성한 Q8_0
self-Q4        = 같은 원본에서 직접 생성한 Q4_K_M
reference-Q8   = 제공처에서 미리 양자화한 Q8_0
reference-Q4   = 제공처에서 미리 양자화한 Q4_K_M
```

모델 파일의 실제 경로와 SHA-256 hash는 `models/manifest.json`에 기록합니다. 모델을 별도 경로에 보관하는 경우에도 실행 시 `--model-path`를 지정하면 결과 로그에 모델 hash를 남길 수 있습니다.

#### 원본 FP16 GGUF를 models에 저장

원본은 [ggml-org/Qwen3-1.7B-GGUF의 Qwen3-1.7B-f16.gguf](https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF/blob/main/Qwen3-1.7B-f16.gguf)를 사용합니다. 파일 크기가 약 4GB이므로 충분한 디스크 공간을 확보합니다.

Windows CMD:

```cmd
curl.exe -L -o models\Qwen3-1.7B-f16.gguf "https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-f16.gguf"
```

WSL / Linux / macOS:

```bash
curl -L -o models/Qwen3-1.7B-f16.gguf "https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-f16.gguf"
```

다운로드 뒤 다음처럼 직접 Q8과 Q4를 생성합니다.

```cmd
python scripts\quantize_models.py --source "models\Qwen3-1.7B-f16.gguf" --output-dir models
```

생성 결과는 모두 로컬 `models/`에 보관합니다.

```text
models/
├─ Qwen3-1.7B-f16.gguf
├─ Qwen3-1.7B-f16-Q8_0.gguf
├─ Qwen3-1.7B-f16-Q4_K_M.gguf
└─ manifest.json
```

#### reference Q8/Q4 준비

reference 모델은 같은 공식 저장소에서 자동으로 받을 수 있습니다. 아래 명령은 한 번에 하나씩 실행하며, 첫 번째 서버를 `Ctrl+C`로 종료한 뒤 다음 모델을 실행합니다. 처음 실행할 때 모델이 로컬 cache에 다운로드됩니다.

```cmd
llama serve -hf ggml-org/Qwen3-1.7B-GGUF:Q8_0 --host 127.0.0.1 --port 18080 --ctx-size 8192
llama serve -hf ggml-org/Qwen3-1.7B-GGUF:Q4_K_M --host 127.0.0.1 --port 18080 --ctx-size 8192
```

reference 모델도 hash와 파일 크기를 manifest에 남길 경우, 다운로드된 `.gguf` 파일의 실제 경로를 확인해 `quantize_models.py`의 `--reference-q8`, `--reference-q4` 옵션으로 기록합니다. reference 모델과 cache도 GitHub에는 올리지 않습니다.

### 4. GPU PC에서 본 실험 실행

서버는 한 번에 한 모델만 실행합니다. 첫 요청은 모델·CUDA kernel warm-up에 사용하고, 비교 결과에서는 warm-up을 제외합니다. 서버를 종료하려면 서버 터미널에서 `Ctrl+C`를 누릅니다.

```text
llama serve -m "C:\path\to\Qwen3-1.7B-f16.gguf" --host 127.0.0.1 --port 18080 --ctx-size 8192
```

다른 터미널에서 먼저 1개 task로 확인합니다.

```text
python scripts\run_experiment.py --model-label self-FP16 --condition free --model-path "C:\path\to\Qwen3-1.7B-f16.gguf" --base-url http://127.0.0.1:18080 --limit 1
```

정상이면 `--limit 1`을 제거해 60개 execution task를 실행합니다. 이후 서버를 종료하고 같은 방식으로 self-Q8, self-Q4, reference-Q8, reference-Q4를 실행합니다. 모든 모델에서 prompt, tool schema, seed, temperature, context size, Agent loop, llama.cpp 버전을 동일하게 유지합니다.

#### 서버 PID와 peak RAM 측정

서버를 새로 시작할 때마다 PID가 바뀝니다. 반드시 현재 실행 중인 서버의 PID를 다시 확인합니다.

Windows CMD:

```cmd
tasklist | findstr /I "llama"
```

WSL / Linux / macOS:

```bash
pgrep -af llama
```

예를 들어 현재 PID가 `32356`이면 `--server-pid 32356`을 추가합니다. 다음에 서버를 재시작하면 숫자를 새 PID로 바꿉니다.

```cmd
python scripts\run_experiment.py --model-label self-Q8 --condition free --model-path "models\Qwen3-1.7B-f16-Q8_0.gguf" --base-url http://127.0.0.1:18080 --server-pid 32356 --limit 1
```

`mean_peak_server_ram_mb`가 모델 서버의 peak RAM이고, `mean_max_ram_mb`는 Python 실행기의 RAM입니다. 서버 PID를 생략하면 서버 RAM 값은 `null`로 남습니다.

GPU와 CPU는 실행 시간·메모리가 다르므로 서로의 효율성 수치를 직접 합치지 않습니다. 팀원이 여러 PC에서 실행한다면 결과 로그의 장비 정보와 `device_label`을 보존하고, 성공률·오류율과 효율성 지표를 각각 해석합니다.

### 5. 팀원이 공유해야 하는 파일

GitHub에는 다음 파일을 공유합니다.

```text
README.md
requirements.txt
src/
scripts/
tasks/tasks.json
models/manifest.json
```

다음 항목은 각 PC에만 보관합니다.

```text
models/*.gguf
.venv/
logs/
results/
```

실험 결과는 GitHub가 아니라 팀 Google Drive의 공유 폴더에 올립니다. `results/`는 `.gitignore`에 포함되어 있으므로 `git add .`를 해도 GitHub에 업로드되지 않습니다.

모델별로 다음 명령을 실행해 요약표와 그래프를 만듭니다.

```cmd
python scripts\analyze.py results\self-Q8_free_실행시각.jsonl results\self-Q8_recovery_실행시각.jsonl --output results\self-Q8_summary.json --csv-output results\self-Q8_summary.csv --plot-dir results\self-Q8_plots
```

Google Drive에는 다음 파일을 함께 올립니다.

```text
<날짜>_<장비명>_<모델명>/
├─ execution 결과 JSONL
├─ recovery 결과 JSONL
├─ summary.json
├─ summary.csv
├─ plots/
├─ models/manifest.json 사본
└─ run_info.txt  (OS, CPU/GPU, RAM, VRAM, llama.cpp 버전, 서버 PID)
```

예시 폴더명은 `20261015_rtx4070_self-Q8`입니다. 모델 파일 자체는 용량이 크므로 Google Drive에도 필요할 때만 별도 공유하고, 결과 폴더에는 보통 hash와 manifest만 포함합니다.

## 1. 프로젝트 폴더로 이동

### Windows CMD

```cmd
cd /d "%USERPROFILE%\Desktop\ondevice"
```

### Windows PowerShell

```powershell
Set-Location "$env:USERPROFILE\Desktop\ondevice"
```

### WSL

Windows의 사용자 폴더 아래 `Desktop\ondevice`를 WSL에서 사용할 때는 다음처럼 이동합니다. `<WindowsUser>`는 본인의 Windows 사용자 이름으로 바꿉니다.

```bash
cd "/mnt/c/Users/<WindowsUser>/Desktop/ondevice"
```

### macOS

프로젝트를 저장한 실제 경로로 이동합니다.

```bash
cd ~/Desktop/ondevice
```

## 2. 가상환경 만들기

이미 `.venv`가 있다면 이 단계는 건너뛰어도 됩니다.

### Windows CMD

```cmd
py -3 -m venv .venv
.venv\Scripts\activate
```

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### WSL / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

가상환경이 활성화되면 명령줄 앞에 `(.venv)`가 표시됩니다.

## 3. 패키지 설치

현재 핵심 기능은 Python 표준 라이브러리만 사용합니다.

```text
python -m pip install -r requirements.txt
```

최대 RAM 측정이 필요하면 선택적으로 `psutil`을 설치합니다.

```text
python -m pip install "psutil>=6.0,<8.0"
```

`mean_max_ram_mb`는 Python 실행기 자체의 RSS입니다. 실제 모델 서버의 메모리를 측정하려면 llama 서버를 실행한 뒤 PID를 확인하고 `--server-pid`를 함께 전달합니다.

Windows CMD:

```cmd
tasklist | findstr /I "llama"
```

예를 들어 PID가 `32356`이면 다음처럼 실행합니다. 서버를 다시 시작하면 PID가 바뀌므로 매번 새 PID를 확인해야 합니다.

```cmd
python scripts\run_experiment.py --model-label self-Q8 --condition free --model-path "C:\path\to\Qwen3-1.7B-f16-Q8_0.gguf" --base-url http://127.0.0.1:18080 --server-pid 32356 --limit 1
```

이때 결과에는 다음 두 메모리 값이 별도로 기록됩니다.

- `max_ram_mb`: Python 실행기의 task 종료 시점 RSS
- `peak_server_ram_mb`: task 실행 중 llama 서버 프로세스에서 관측한 최대 RSS

`mean_peak_server_ram_mb`를 모델별로 비교용 peak RAM 지표로 사용합니다. `--ram-sample-interval-ms`로 측정 간격을 바꿀 수 있으며 기본값은 100ms입니다.

분석 결과를 CSV와 PNG 그래프로 저장하려면 선택적으로 matplotlib를 설치합니다.

```text
python -m pip install "matplotlib>=3.8,<4.0"
```

분석 예시는 다음과 같습니다.

```text
python scripts\analyze.py results\self-Q8_free_실행시각.jsonl --csv-output results\summary.csv --plot-dir results\plots
```

`--plot-dir`를 지정하면 horizon별 성공률·자연 오류율·회복률·실행 시간·peak server RAM 그래프가 생성됩니다. 모델 저장공간은 `models/manifest.json`에서 읽어 `model_storage_mb`로 요약표에 추가됩니다.

## 4. 모델 없이 smoke test 실행

실제 모델이나 llama.cpp 서버 없이 Agent 흐름을 확인하는 단계입니다.

### Windows CMD / PowerShell

```text
python scripts\smoke_test.py
```

### WSL / macOS

```bash
python scripts/smoke_test.py
```

정상 실행되면 다음 메시지가 출력됩니다.

```text
SMOKE TEST PASSED
```

이 테스트는 다음을 확인합니다.

- task 로딩
- 정상 checkpoint 생성
- 오류 상태 주입
- Agent tool call 실행
- 최종 상태 채점
- free·normal_restart·recovery 조건
- 새 execution 60개 전체의 정답 실행과 필수 호출 수

## 모델 생성 및 manifest 기록

직접 양자화 실험은 이미 만들어진 Q8·Q4 파일을 내려받는 방식이 아니라, 하나의 양자화되지 않은 FP16/BF16 GGUF에서 Q8과 Q4를 각각 생성하는 방식입니다.

```text
원본 FP16/BF16 GGUF
       ├─ llama-quantize → Q8_0
       └─ llama-quantize → Q4_K_M
```

Windows CMD / PowerShell:

```cmd
python scripts\quantize_models.py --source "C:\path\to\qwen3-1.7b-f16.gguf" --output-dir models
```

WSL / macOS:

```bash
python scripts/quantize_models.py --source /path/to/qwen3-1.7b-f16.gguf --output-dir models
```

실행하면 다음 파일이 생성됩니다.

```text
models/
├─ qwen3-1.7b-f16-Q8_0.gguf
├─ qwen3-1.7b-f16-Q4_K_M.gguf
└─ manifest.json
```

`manifest.json`에는 원본·생성 모델의 경로, 파일 크기, SHA-256 hash, quantization type, 양자화 명령어, llama.cpp 버전이 기록됩니다. 비교용 공식 Q8·Q4 파일이 있다면 다음 옵션으로 함께 기록할 수 있습니다.

```text
--reference-q8 "C:\path\to\official-q8.gguf" --reference-q4 "C:\path\to\official-q4.gguf"
```

`--source`는 반드시 양자화되지 않은 GGUF여야 합니다. Hugging Face의 safetensors 원본만 가지고 있다면, 먼저 llama.cpp의 Hugging Face→GGUF 변환 도구로 F16/BF16 GGUF를 만든 뒤 이 스크립트를 실행합니다. Q8 파일을 다시 Q4로 바꾸지 않는 것이 중요합니다.

## 직접 양자화 모델과 공식 모델의 비교

주 비교는 이 저장소의 동일한 `execution` task 60개로 진행합니다. 같은 task, prompt, tool schema, seed, temperature, context size, llama.cpp 버전을 FP16·직접 생성 Q8·직접 생성 Q4·공식 Q8·공식 Q4에 동일하게 적용합니다.

이 비교는 양자화 방식 또는 변환 경로가 결과에 영향을 주는지 확인하기 위한 대조 실험입니다. 공식 Q8·Q4가 다른 체크포인트나 다른 quantization type에서 만들어졌다면, 같은 정밀도의 순수 비교가 아니므로 manifest에 출처 차이를 남기고 결과를 별도로 해석해야 합니다.

BFCL은 주 실험을 대체하지 않고 외부 검증용으로 추가하는 편이 적절합니다. 이 프로젝트의 핵심 질문은 horizon별 오류 위치와 회복 과정까지 통제하는 것이므로, 먼저 자체 execution/recovery task로 비교한 다음 BFCL을 통해 일반적인 tool-calling 성능이 같은 방향으로 나타나는지 확인합니다.

## 5. llama.cpp 서버 실행

Ollama를 사용하지 않고 `llama.cpp`의 서버를 직접 사용합니다. 서버는 별도의 터미널에서 계속 실행해 둡니다.

```text
llama serve -hf Qwen/Qwen3-1.7B-GGUF:Q8_0 --reasoning off --host 127.0.0.1 --port 18080 --ctx-size 8192
```

Windows, WSL, macOS에서 명령어 형식은 동일합니다. `llama` 명령이 인식되지 않으면 llama.cpp가 설치된 경로를 사용하거나 `llama-server` 실행 파일 경로를 확인해야 합니다.

## 6. 실제 모델로 조건 실행

서버를 실행한 상태에서 별도의 터미널을 열고 다음을 실행합니다.

### Windows CMD / PowerShell

```text
python scripts\run_experiment.py --model-label Q8 --condition free --base-url http://127.0.0.1:18080
```

### WSL / macOS

```bash
python scripts/run_experiment.py --model-label Q8 --condition free --base-url http://127.0.0.1:18080
```

`free`는 정상 수행 task 60개를 실행하고, `recovery`는 별도 회복 task 60개를 실행합니다.

```text
--condition recovery
```

정상 중간 상태에서 재개하는 진단 조건도 필요할 때 사용할 수 있습니다.

```text
--condition normal_restart
```

예를 들어 Q4의 회복 실험은 다음과 같습니다.

```text
python scripts\run_experiment.py --model-label Q4 --condition recovery --base-url http://127.0.0.1:18080 --recovery-budget 3
```

FP16·Q8·Q4를 비교할 때는 다음 항목을 동일하게 유지해야 합니다.

- task 파일
- prompt와 tool 정의
- seed와 temperature
- context size
- Agent loop
- llama.cpp 버전

## 7. 결과 집계

실험 결과는 `results/` 아래 JSONL 파일로 저장됩니다.

### Windows CMD / PowerShell

```text
python scripts\analyze.py results\Q8_free_실행시각.jsonl
```

### WSL / macOS

```bash
python scripts/analyze.py results/Q8_free_실행시각.jsonl
```

### 회복률 조사 방법

회복률은 `recovery` 조건에서 처음부터 같은 오류 상태를 받은 뒤, Agent가 제한된 호출 횟수 안에 정확한 최종 상태를 만드는 비율입니다. 단순히 tool을 한 번 호출했는지가 아니라 최종 상태가 `success_condition`과 일치하는지를 기준으로 합니다.

각 모델에서 `free`와 `recovery`를 같은 seed·서버 설정으로 실행합니다. `free`는 `execution` task를, `recovery`는 대응하는 `recovery` task를 자동으로 선택합니다.

#### Windows CMD / PowerShell

```text
python scripts\run_experiment.py --model-label Q4 --condition free --base-url http://127.0.0.1:18080
python scripts\run_experiment.py --model-label Q4 --condition recovery --base-url http://127.0.0.1:18080 --recovery-budget 3
python scripts\analyze.py results\Q4_free_실행시각.jsonl results\Q4_recovery_실행시각.jsonl
```

#### WSL / macOS

```bash
python scripts/run_experiment.py --model-label Q4 --condition free --base-url http://127.0.0.1:18080
python scripts/run_experiment.py --model-label Q4 --condition recovery --base-url http://127.0.0.1:18080 --recovery-budget 3
python scripts/analyze.py results/Q4_free_실행시각.jsonl results/Q4_recovery_실행시각.jsonl
```

FP16·Q8·Q4 각각에 대해 같은 두 조건을 실행한 뒤, 결과 파일을 한 번에 분석할 수도 있습니다.

```text
python scripts\analyze.py results\FP16_free_실행시각.jsonl results\FP16_recovery_실행시각.jsonl results\Q8_free_실행시각.jsonl results\Q8_recovery_실행시각.jsonl results\Q4_free_실행시각.jsonl results\Q4_recovery_실행시각.jsonl
```

구버전 분석 코드는 `recovery_comparison`에 다음 값을 출력합니다. 새 execution-v2와 구버전 recovery를 함께 분석한 `recovery_gap`에는 의미가 없으므로 사용하지 않습니다. 같은 checkpoint에서의 정상/오류 비교로 회복 실험을 수정한 뒤 해당 집계도 바꿀 예정입니다.

- `normal_success_rate`: 정상 초기 상태에서의 완료율
- `recovery_rate`: 같은 오류 상태에서의 회복 완료율
- `recovery_gap`: `normal_success_rate - recovery_rate`
- `mean_extra_calls`: 회복에 추가로 사용한 평균 tool call 수

`recovery` 실행에서는 회복 task의 `recovery_query`가 사용됩니다. 이 질문에는 이미 발생한 오류 상황과 복구해야 할 목표 상태가 명시되어 있습니다. 다만 정확히 어떤 tool을 어떤 순서로 호출할지는 알려주지 않으므로, 모델이 상태를 확인하고 스스로 복구해야 합니다. 모든 정밀도 조건에는 같은 오류 설명과 같은 recovery query가 제공됩니다.

`--recovery-budget 3`은 회복 task의 필수 tool call 수에 추가로 최대 3회의 호출을 허용한다는 뜻입니다. 예를 들어 horizon 5 회복 task는 기본 허용량 5회 + 회복 예산 3회 = 8회입니다. 모든 모델과 task에 같은 회복 예산을 사용해야 합니다. `--max-calls`를 지정하면 이 자동 계산 대신 고정된 호출 상한을 사용합니다.

주요 집계 항목은 다음과 같습니다.

- 전체 성공률
- free 조건의 자연 오류율
- 최초 오류 위치
- normal_restart의 다음 행동 정확도(선택적 진단 조건)
- recovery의 회복률
- 회복에 사용된 추가 호출 수
- recovery 오류 총 횟수(`recovery_error_count`)
- recovery tool 오류·invalid action 횟수
- recovery 최초 오류 단계(`recovery_first_error_step`)
- 성공 상태에 처음 도달한 단계(`recovery_completion_step`)
- 실행 시간과 생성 토큰 수
- 선택적으로 최대 RAM 사용량

## 8. 연구용 task 구성

현재 `tasks/tasks.json`에 연구용 task 120개가 이미 포함되어 있습니다.

```text
20개 기본 시나리오 × horizon 1·3·5 × 2(task 유형) = 120개 task
```

ID·horizon·domain상 대응 관계는 검사하지만, 현재 새 정상 수행 task와 구버전 회복 task는 의미상 대응하는 쌍이 아닙니다. 회복 task 재설계가 필요합니다.

- `execution`: 오류가 없는 정상 시작 상태와 제공된 조회/준비 기록에서 남은 1·3·5단계를 수행
- `recovery`: 처음부터 동일한 오류 상태를 주고 오류를 복구

각 task에는 다음 정보가 포함되어 있습니다.

- 사용자 질문
- 정답 tool trajectory
- 필수 horizon
- checkpoint 단계
- 오류 주입 방식
- 오류 상황 설명
- 회복 전용 질문
- 최종 성공 조건

execution-v2에는 `schema_version`, `execution_spec`(요청과 후보 환경), `initial_actions`(미리 제공할 정상 호출 기록)가 추가됩니다. `gold_actions`는 모델이 앞으로 수행해야 할 호출만 담습니다. `checkpoint_step`은 이 남은 경로 안에서 선택적으로 `normal_restart`를 시험할 지점이며, 미리 제공된 `initial_actions`의 개수와는 다릅니다. 실행 기록에는 시작 대화·상태·제공된 호출·스키마 버전·최종 성공 조건이 남습니다.

회복 task의 `checkpoint_step`은 0이며, 오류가 모델의 첫 호출 전에 주입됩니다. 따라서 정상 수행의 자연 오류율과 회복 수행의 회복률을 서로 다른 실험으로 측정할 수 있습니다.

horizon별 회복 경로는 다음처럼 구성했습니다. horizon 1은 즉시 교정 또는 stale 결과 재조회 1회, horizon 3은 상태 확인·오류 제거·목표 상태 적용, horizon 5는 추가 정보 조회를 포함한 뒤 오류 제거·목표 상태 적용의 순서입니다. `replace_cart_item`과 `replace_event`는 한 번의 교정 호출을 검증하기 위한 mock tool입니다.

BFCL은 내부 task가 안정화된 뒤 외부 검증 데이터로 별도 추가합니다. 처음부터 BFCL 실행 코드를 섞지 않는 이유는 핵심 실험의 상태·오류·회복 흐름을 단순하게 유지하기 위해서입니다.
