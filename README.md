# Quantized Agent Stability

양자화가 장기 Agent의 수행 안정성에 미치는 영향을 비교하는 실험 저장소입니다.

**현재 단계:** execution 60개와 recovery 60개를 v2로 재설계하고 영어 데이터 자동 검증과 회복·오류 지표 v3를 구현했습니다. 회복 비교는 같은 회복 fixture의 `normal_restart`와 `error_restart`를 비교합니다. 아래 검증은 정답 재생과 코드 검증이며 실제 Qwen 성능 검증은 아닙니다. 워밍업·서버 설정 고정·VRAM·반복 실행 관리는 여전히 본실험 전에 보완해야 합니다.

현재 모델 입력은 **영어**, 데이터 버전은 `workflow-en-v1`입니다. 질문·요청 조건 설명·후보 이름·일정 제목·도구 정의·오류 관측을 영어로 통일했습니다. 실제 실험 명령은 전체 데이터셋 자동 검증을 먼저 통과해야 시작합니다. 팀원용 README는 한국어이며, 생성되는 task 카탈로그는 영어입니다.

이 변경이 Qwen의 영어 성능 우위를 측정한 결과는 아닙니다. 이번 비교에서 언어 조건을 고정하기 위한 선택입니다. 기존 한국어 파일럿 결과와 새 영어 결과는 합치지 않습니다. 모델 파일과 양자화 결과는 그대로 사용할 수 있지만, 영어 조건의 결과는 새로 수집해야 합니다.

이 프로젝트는 FP16·Q8·Q4 모델을 같은 Agent framework와 동일한 mock tool 환경에서 실행합니다. 최종 성공률만 비교하지 않고, 다음 두 현상을 분리해 측정합니다.

1. 모델이 오류를 얼마나 자주 발생시키는가
2. 같은 오류가 주어졌을 때 얼마나 잘 회복하는가

## 실험 조건과 task 유형

- `free`: `execution` task를 초기 상태에서 처음부터 수행합니다. 양자화 수준과 horizon에 따른 도구 선택·정상 수행을 측정합니다.
- `normal_restart`: 회복 task 60개를 공통의 정상 checkpoint에서 재개합니다.
- `error_restart`: 같은 회복 task 60개에서 잘못된 추가 저장 결과를 만든 도구 호출·관측을 제공한 뒤 재개합니다. `recovery`는 이 조건의 호환 별칭입니다.

execution-v2의 `horizon`은 제공된 정상 시작 상태에서 완료까지 **남아 있는 필수 도구 호출 수**입니다. 5단계 작업 흐름 중 h1에는 앞 4단계, h3에는 앞 2단계의 정상 도구 결과가 대화 기록으로 제공됩니다. h5는 요청 조회부터 모델이 수행합니다. 미리 제공한 호출과 종료 응답은 horizon에 포함하지 않습니다. 실행 중 실제 호출 수·추가 호출 수는 별도로 기록합니다.

## 현재 구현 범위

현재 파일에는 정상 수행 60개와 대응 회복 fixture 60개, 총 120개 task가 있습니다. 60개는 20개 기본 시나리오의 시작 정보 3조건이며, 서로 독립인 60개 질문이 아닙니다. [정상 시나리오](tasks/EXECUTION_TASKS.md)와 [회복 설계·목록](tasks/RECOVERY_TASKS.md)에서 요청과 오류 조건을 확인할 수 있습니다.

- 결정론적 mock tool 환경
- 정상 상태와 오류 상태의 checkpoint 생성
- 오류 상태 주입
- 원래 질문을 유지하며 실제 오류 호출·관측이 보이는 checkpoint 기록
- FakeClient를 이용한 모델 없는 smoke test
- llama.cpp 서버와 통신하는 HTTP client
- 성공률·오류 위치·회복률·추가 호출 수 집계
- 실행 환경과 task/model hash 기록

execution은 같은 기본 시나리오의 질문·후보·선택 규칙·최종 목표를 h1·h3·h5에서 동일하게 유지합니다. 회복 fixture도 이 질문·목표·정상 기록을 공유하며, 오래된 draft가 공통 초기 자료로 추가됩니다. 정상/오류 분기는 같은 fixture에서 갈라집니다.

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

`task_type`이 `execution`이면 정상 수행 task이고, `recovery`이면 정상/오류 두 분기에 공통으로 사용할 회복 fixture입니다. `task_type`과 실제 실행 `condition`을 구분합니다. 모델당 execution 60회 + 회복 정상 분기 60회 + 오류 분기 60회 = 180회이며, FP16·Q8·Q4를 모두 실행하면 반복 전 총 540회입니다.

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

v2 task의 `natural_error`·`first_error_step`는 정답 경로의 순서를 비교하지 않습니다. 도구/인자 오류, 요청을 위반한 결과 확정, 올바른 결과 삭제, 미완료 상태의 종료와 잘못된 모델 출력을 셉니다. 유효한 추가 조회·다른 후보 탐색·삭제 순서 차이는 오류가 아닙니다. 준비만 한 잘못된 draft 등 모든 계획 오류를 포착하는 지표는 아닙니다. 지표 v3에서는 잘못된 모델 JSON·다중 도구 출력·토큰 상한 잘림은 행동 평가에 포함하고, HTTP·연결·서버 응답 형식 장애 및 예상하지 못한 코드 예외는 별도 표시하여 분모에서 제외합니다.

이 설계는 **동일 작업 흐름에서 남은 의존 단계 수**의 효과를 비교합니다. 시작 정보량과 이미 완료된 작업도 달라지므로 일반 과제의 길이만 바꾼 결과라고 해석하지 않습니다. h1은 올바르게 준비된 결과를 확정하므로 후보 선택 능력은 h3·h5에서 주로 비교합니다. 현재 4개 분야는 같은 5단계 구조를 사용하며, 다양한 현실 Agent 작업 전체를 대표하지는 않습니다.

## 폴더 구조

```text
ondevice/
├─ README.md
├─ requirements.txt
├─ .github/workflows/validate-dataset.yml # GitHub 업로드·PR 시 오프라인 검사
├─ tests/test_dataset_validation.py # 잘못된 데이터가 거부되는지 회귀 검사
├─ tests/test_behavior_metrics.py # 출력 오류·서버 장애·회복 지표 회귀 검사
├─ tasks/tasks.json
├─ src/
│  ├─ tasks.py          # task 로딩과 검증
│  ├─ dataset_validation.py # JSON·형식·영어·정보 누락·의미를 묶은 통합 검증
│  ├─ client.py         # FakeClient와 llama.cpp client
│  ├─ environment.py    # 결정론적 mock tool과 상태 관리
│  ├─ execution_environment.py # 리소스 의존성이 있는 execution-v2 도구
│  ├─ execution_validation.py  # 정답·우회 경로·최종 상태 검증
│  ├─ recovery_validation.py   # 대응 checkpoint·복구 필요성 검증
│  ├─ recovery_comparison.py   # 동일 조건의 정상/오류 쌍 비교
│  ├─ workflow_metrics.py      # v2 task의 상태·행동 기반 v3 지표
│  ├─ agent.py          # 공통 Agent 실행 루프
│  ├─ metrics.py        # 성공·오류·회복 지표
│  └─ experiment.py     # batch 실행과 메타데이터 기록
└─ scripts/
   ├─ smoke_test.py     # 모델 없이 전체 흐름 검사
   ├─ validate_tasks.py  # task 중복·누락·대응 관계 검증
   ├─ build_tasks.py     # 두 영어 task 유형과 카탈로그를 함께 검증·재생성
   ├─ build_execution_tasks.py # 20개 시나리오 정의와 JSON 일치 검사
   ├─ build_recovery_tasks.py  # execution과 대응하는 회복 fixture 생성·검사
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

WSL/macOS에서는 `python scripts/validate_tasks.py`를 사용합니다. `valid: true`와 `execution_semantics.valid`, `recovery_semantics.valid`가 모두 true이면 구성과 두 task 유형의 코드 검증이 통과한 것입니다. 실제 모델의 능력이나 논문 결론이 검증됐다는 뜻은 아닙니다.

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
- 회복 두 분기의 공통 checkpoint·원래 질문·실제 오류 관측 기록 일치
- 잘못된 추가 결과를 남기면 실패, 삭제를 먼저/나중에 해도 성공
- 복구 경로의 부분 생략 1,620개에서 우회 성공 없음, 정상/오류 정답 재생 120회

추가 검증 항목은 JSON의 중복 키·NaN/Infinity·잘못된 자료형, 빈 질문, 질문에 필요한 요청 ID 누락, 정답 후보/옵션/견적 ID 유출, 기본 시나리오 내용 중복, 생성기 원본과의 문구·조건 불일치입니다. 숫자 `1`을 `true`나 문자열 `"1"`로 바꾸는 경우도 통과하지 못합니다.

모델에 제공되는 시스템/사용자 메시지·도구 정의·주입된 관측·정답 경로의 후속 반환값까지 180개 조건에서 검사합니다. 영어 데이터는 `language=en`, `dataset_version=workflow-en-v1`로 표시하며 비ASCII 문자열 혼입을 차단합니다. 이 검사는 영어 문장의 논리적 정확성을 증명하는 언어 판별기는 아니므로 시나리오 내용은 사람이 검토해야 합니다.

독립 검증과 보고서 저장, 결함 회귀 검사는 다음과 같습니다. Windows CMD/PowerShell·WSL·macOS에서 같은 슬래시 경로를 사용할 수 있습니다.

```text
python scripts/validate_tasks.py --output results/dataset_validation_en.json
python -m unittest discover -s tests -v
```

성공 시 `valid: true`와 오류 없는 보고서를 출력합니다. 실패 시 `valid: false`, 실패한 `stage`, `errors`를 출력하고 종료 코드 1로 끝납니다. 보고서의 `dataset_sha256`은 내용 기준, `tasks_file_sha256`은 파일 바이트 기준 hash입니다. `results/` 보고서는 Git에 올리지 않고 결과 공유 폴더에 함께 보관합니다.

`run_experiment.py`도 같은 전체 검증을 서버 접속 전에 자동 수행합니다. `--limit 1`을 넣어도 검증은 120개 전체에 적용합니다. 실패하면 모델을 실행하지 않습니다. 실행 metadata에는 데이터 언어·버전·내용 hash·검증기 버전이 남습니다. 입력 task 파일을 결과 파일로 덮어쓰는 `--output` 지정도 거부합니다.

GitHub용 `.github/workflows/validate-dataset.yml`도 추가했습니다. 저장소에 업로드된 후 push·pull request 때 Python 표준 라이브러리만으로 데이터 검증·생성기 일치·회귀 검사·smoke test를 수행하도록 설정되어 있습니다. 실제 GitHub 실행 성공은 업로드 후 Actions 탭에서 확인합니다.

이 검사는 정답 부분 경로와 리소스 의존성을 검사합니다. 임의의 모든 도구 인자·행동열을 전수 탐색한 검사는 아닙니다. 기존 두 생성기의 `--check`도 사용할 수 있으며, 두 유형을 함께 확인하는 명령은 `python scripts/build_tasks.py --check`입니다.

### task를 수정할 때

`tasks/tasks.json`은 생성된 파일입니다. 질문·조건은 `scripts/build_execution_tasks.py`, 회복 오류 설계는 `scripts/build_recovery_tasks.py`에서 수정한 뒤 다음을 실행합니다.

```text
python scripts/build_tasks.py --write
python scripts/validate_tasks.py
python -m unittest discover -s tests -v
```

`--write`는 생성한 120개를 메모리에서 검증한 뒤 JSON과 두 카탈로그를 함께 갱신합니다. JSON만 직접 바꾸면 생성기 원본과 다르다는 이유로 실험이 중단됩니다. 연구 조건이 바뀌면 데이터 버전을 갱신하고 결과를 구분해 보관합니다.

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

`--plot-dir`를 지정하면 horizon별 성공률·자연 오류율·응답별/실제 도구 호출별 오류율·회복률·실행 시간·peak server RAM 그래프가 생성됩니다. 값이 수집되지 않은 지표의 그림은 만들지 않습니다. 모델 저장공간은 `models/manifest.json`에서 읽어 `model_storage_mb`로 요약표에 추가됩니다.

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
- free·normal_restart·error_restart 조건
- 새 execution 60개 전체의 정답 실행과 필수 호출 수
- 회복 60개 × 정상/오류 두 분기의 정답 실행과 추가 비용 지표

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

서버 실행 후 별도 터미널에서 실행합니다. 아래 슬래시 경로는 Windows CMD/PowerShell·WSL·macOS 공통입니다. 예시는 self-Q4이며, 실제 FP16·Q8·Q4 모델을 서버에서 교체할 때 라벨·모델 경로·서버 PID도 맞춥니다.

```text
python scripts/run_experiment.py --model-label self-Q4 --condition free --task-type execution --base-url http://127.0.0.1:18080
python scripts/run_experiment.py --model-label self-Q4 --condition normal_restart --task-type recovery --base-url http://127.0.0.1:18080 --recovery-budget 3
python scripts/run_experiment.py --model-label self-Q4 --condition error_restart --task-type recovery --base-url http://127.0.0.1:18080 --recovery-budget 3
```

モデル hash와 메모리를 기록하려면 세 명령 모두에 실제 `--model-path models/모델파일.gguf --server-pid 실제PID`를 추가합니다. 처음 확인할 때는 `--limit 1`을 사용합니다.

`free`는 execution 60개, `normal_restart`와 `error_restart`는 같은 recovery 60개를 각각 선택합니다. `--condition recovery`를 입력해도 `error_restart`로 기록됩니다. 구버전 execution checkpoint 진단은 `--condition normal_restart --task-type execution`으로 명시해야 하며 회복 비교에는 포함되지 않습니다.

세 정밀도에서 task·프롬프트·도구·seed·context size·thinking 설정·llama.cpp 버전·장비·호출 예산을 같게 유지합니다. 서버의 완전한 설정 수집, 워밍업, GPU VRAM, resume/반복 번호 지원은 후속 보완 대상입니다. 현재 client의 temperature/max_tokens/seed/timeout 값은 metadata의 inference_settings에 기록되지만 CLI로 변경하는 기능은 아직 없습니다.

## 7. 회복 실험과 집계

### 공통 오류와 두 분기

오류 유형은 `extra_saved_record` 하나입니다. 두 분기 모두 공통 checkpoint 자료로 오래된 draft 하나를 가지며, 오류 분기에서만 이 draft를 확정한 실제 호출과 반환값이 대화 기록에 추가됩니다. 이 결과는 요청 조건과 다른 후보·옵션이며 별도의 요청 ID로 저장됩니다.

원래 질문의 ‘저장된 결과는 요청에 맞는 한 개만 남겨줘’ 조건을 그대로 사용합니다. 새로운 회복 질문이나 정답 지시는 제공하지 않습니다. 모델은 기록에서 추가 저장 결과를 확인하고 `discard_saved_result`로 삭제해야 합니다. 원래 정답을 저장해도 잘못된 추가 기록이 남으면 실패합니다. 삭제는 작업 전이나 후에 해도 됩니다.

정상·오류 분기는 원래 질문·기본 환경·정상 작업 기록·도구·최종 목표·호출 예산을 공유합니다. `checkpoint_state_hash`는 주입 전 공통 상태, `initial_state_hash`는 각 분기의 재개 상태입니다. 오류 분기에는 추가 관측과 잘못된 저장 상태가 함께 주어지며, 이 조건 묶음의 영향을 측정합니다. 전체 과거 세션을 자연스럽게 생성한 데이터는 아닙니다.

| Horizon 표기 | 정상 최소 호출 | 복구 최소 호출 |
|---|---:|---:|
| 1 | 1 | 2 |
| 3 | 3 | 4 |
| 5 | 5 | 6 |

horizon은 정상 작업에 남은 필수 단계이며, 복구에는 삭제 1회가 더 필요합니다. `normal_remaining_calls`와 `recovery_min_calls`를 따로 기록합니다.

`--recovery-budget 3`이면 두 분기 모두 horizon + 3이라는 같은 응답 턴 상한을 사용합니다. 현재 구현에서는 도구 호출뿐 아니라 종료 응답·잘못된 형식의 응답도 한 턴을 사용합니다. h5는 두 분기 모두 최대 8턴이며, 정상 정답은 5호출+종료, 회복 정답은 6호출+종료입니다. `--max-calls`를 지정하면 두 분기에 같은 고정 상한을 전달해야 합니다. 오류 분기에만 최소 복구 길이 기준으로 예산을 늘리면 안 됩니다.

미리 제공한 호출과 오류 주입 호출은 새 모델 호출·오류 횟수·실행시간·생성 토큰에 포함하지 않습니다.

### 결과 비교

실행 결과는 `results/` 아래에 JSONL로 저장됩니다. 실제 생성된 파일명으로 아래 예시를 바꿉니다.

```text
python scripts/analyze.py results/self-Q4_normal_restart_실행시각.jsonl results/self-Q4_error_restart_실행시각.jsonl --output results/recovery_summary.json --csv-output results/recovery_summary.csv --plot-dir results/plots
```

`recovery_comparison`은 같은 회복 fixture의 정상/오류 결과만 짝지어 다음을 계산합니다.

- `normal_restart_success_rate`: 공통 정상 checkpoint에서의 완료율
- `error_restart_success_rate`: 오류를 받은 뒤의 최종 완료율
- `recovery_penalty`: 동일한 정상/오류 쌍의 성공률 차이
- `mean_extra_calls_over_normal`: 정상 최소 호출보다 얼마나 더 사용했는지
- `mean_extra_calls_over_oracle`: 해당 오류의 최소 복구 호출보다 얼마나 더 사용했는지

추가 호출 평균은 완료한 오류 실행에 대해서만 계산합니다. 실패 실행이 빨리 끝났다고 효율적인 회복으로 해석하지 않습니다. 서로 다른 장비·모델 hash·task hash·seed·수집된 client 설정·예산·평가 버전·Agent 정책 버전은 같은 비교 쌍으로 합치지 않습니다. 동일한 실행을 중복 입력하면 오류로 알리며, 짝이 없는 결과와 서버/코드 예외가 있는 쌍은 진단 정보로 남깁니다. 잘못된 모델 출력은 대응 비교에서도 제외하지 않습니다. 기록되지 않은 서버 설정 차이까지 자동 검출하는 것은 아닙니다.

`--csv-output results/recovery_summary.csv`를 지정하면 조건별 표와 함께 `results/recovery_summary_paired.csv`에 대응 비교 표를 저장합니다. 회복률 그림은 `error_restart` 결과를 사용합니다. 일반 표·그림은 같은 모델 라벨·프로토콜에 다른 수집 설정이 들어오면 분석을 중단합니다. 장비나 seed가 다른 결과는 별도로 분석하거나 라벨을 구분합니다. 누락된 metadata나 미수집 서버 설정까지 자동 검출할 수는 없습니다.

execution의 `free` 결과는 전체 수행 분석용입니다. 이 성공률에서 회복 성공률을 빼는 구버전 `recovery_gap`은 사용하지 않습니다. 구버전 결과와 새 결과를 섞지 말고 별도 폴더에 보관합니다.

### 회복·오류 지표 읽기

- `recovery_success`: 잘못된 기록 제거와 원래 목표 달성을 모두 완료했는지
- `recovery_error_resolved`: 추가된 잘못된 기록을 제거했는지. 이것만 true이고 최종 과제는 실패할 수 있음
- `recovery_resolution_step`: 잘못된 기록이 처음 제거된 재개 이후 응답 단계
- `recovery_stable_resolution_turn`: 마지막 재발 이후 제거가 유지되기 시작한 응답 턴. 최종 상태에 잘못된 기록이 있으면 null
- `recovery_completion_step`: 목표 상태에 처음 도달한 응답 단계. 이후 다시 망가뜨리면 최종 성공은 false
- `recovery_error_count`: 재개 후 도구/인자 오류·잘못된 확정/삭제·미완료 종료·출력 형식 오류를 낸 응답 단계 수. 한 응답에서 여러 오류 표지가 있어도 1회로 계산
- `recovery_first_error_step`: 주입된 오류가 아니라 모델이 재개 후 새로 낸 첫 오류 단계
- `attempted_tool_call_count`: 모델이 출력한 도구 호출 시도 수. 한 응답에 여러 호출을 내면 모두 시도로 기록하되 실행하지 않음
- `tool_call_count`, `executed_tool_call_count`: 실제 도구 실행기에 전달한 호출 수. 도구가 인자를 거절한 호출도 포함하며, 파싱 실패로 실행하지 못한 출력은 제외
- `response_turn_count`: 모델 응답을 받은 횟수. 종료·출력 형식 오류도 포함하며, 응답 없이 끝난 서버/코드 예외 단계는 제외
- `error_count`, `error_turn_rate`: 오류를 낸 응답 수와 전체 응답 수 대비 비율
- `tool_error_action_count`, `tool_action_error_rate`: 실제 도구 호출 중 도구 거절 또는 잘못된 상태 변경이 발생한 호출 수와 비율
- `natural_error_rate`: `free`에서 오류를 한 번 이상 낸 **과제의 비율**. 호출별 오류율이 아니며, 위 응답별/호출별 비율과 구분
- `first_error_turn` (`first_error_step`과 동일): 재개 후 1부터 센 응답 턴. `first_error_tool_call_index`는 그 오류 응답이 실제 도구를 실행한 경우의 1-based 호출 번호이며 파싱·종료 오류이면 null
- `extra_calls_over_normal`, `extra_calls_over_oracle`: 정상 기준/분기별 최소 호출 기준의 추가 비용
- `model_output_error_count`: 잘못된 JSON·동시 다중 호출·토큰 상한 잘림 등으로 사용할 수 없었던 응답 수
- `termination_reason`: `finished` / `budget_exhausted` / `infrastructure_error` / `internal_error`. 정상 조회만 반복하다 예산이 끝나면 오류 횟수 0인 실패도 가능
- `n_runtime_errors`: 서버 장애 또는 코드 예외로 행동 성공률에서 제외된 실행 수. `n_infrastructure_errors`와 `n_internal_errors`로 세부 구분

`recovery_error_resolved`는 **최종 상태**에서 잘못된 기록이 없는지 확인합니다. `recovery_resolution_step`은 첫 제거 시점이므로, 이후 잘못된 기록을 다시 저장하면 첫 제거 시점이 있어도 최종 회복은 실패합니다. 주입된 호출과 제공된 과거 기록은 새 오류·호출 비용에 포함하지 않습니다. 선택적 조회와 유효한 다른 행동 순서는 오류가 아니며 `next_action_accuracy`는 사용하지 않아 null입니다. 비율의 분모가 0이면 null로 남깁니다. 집계의 `pooled_error_turn_rate`와 `pooled_tool_action_error_rate`는 유효한 과제들의 분자·분모를 각각 합친 비율입니다.

### 출력 오류와 서버 장애 처리 정책

모든 정밀도는 한 응답에 도구 호출 하나만 내도록 같은 시스템 지시를 받습니다. 여러 호출을 내면 첫 번째만 조용히 실행하지 않고 전체 응답을 거절합니다. 잘못된 모델 출력은 원문·토큰 사용량·`finish_reason`·오류 종류를 로그에 남기고, 정답을 알려주지 않는 고정된 형식 안내만 제공합니다. 이 응답도 동일한 응답 턴 예산을 소비합니다. 잘린 응답(`finish_reason=length`)은 도구 실행 전에 거절합니다. 파싱 오류 뒤 올바르게 복구하면 최종 성공은 가능하지만 발생한 출력 오류는 지워지지 않습니다.

HTTP 오류·타임아웃·연결 실패·잘못된 서버 응답 형식은 `infrastructure_error`, 예상하지 못한 코드 예외는 `internal_error`로 종료합니다. 이 실행은 모델 성공률 분모에서 제외하며, 대응 비교에서도 어느 한쪽에 장애가 있으면 그 쌍을 제외하고 개수를 보고합니다. 제외가 많거나 정밀도별로 치우치면 서버·설정 문제를 먼저 점검하고 해당 과제를 다시 수집해야 합니다. 자동 재시도는 아직 없으므로 중복 결과를 그대로 합치지 않습니다.

현재 `metrics_version=workflow-metrics-v3`, `agent_policy_version=single-action-feedback-v1`을 metadata·trace·score에 기록합니다. 기존 로그의 모델 파싱 오류는 분리되지 않았으므로 새 지표로 소급 해석하지 않고 별도로 보관합니다. 새 데이터셋과 같은 모델 파일을 쓰더라도 평가·Agent 정책이 바뀌었으므로 본실험 결과는 새로 수집합니다.

## 8. 연구용 task 구성과 검증

```text
20개 기본 시나리오 × 정상 horizon 1·3·5 = execution 60개
각 execution과 대응하는 recovery fixture = recovery 60개
모델당 free 60회 + normal_restart 60회 + error_restart 60회 = 180회
```

각 task에는 원래 질문·환경 fixture·정답 경로·정상 제공 기록·최종 성공 조건이 있습니다. 회복 task에는 오류 주입 호출, 별도 복구 정답 경로, 정상/복구 최소 호출도 있습니다.

`gold_actions`는 정상 분기의 남은 호출이고, `recovery_actions`는 잘못된 기록 삭제가 포함된 복구 경로입니다. `initial_actions`는 미리 제공할 정상 기록입니다. 회복 task의 `checkpoint_step=0`은 그 남은 경로에서 추가 재생하지 않는다는 뜻입니다. 실제 전체 작업 흐름의 checkpoint 깊이는 `resume_workflow_step=len(initial_actions)`이며 h1·h3·h5에서 각각 4·2·0입니다.

```text
python scripts/build_tasks.py --check
python scripts/validate_tasks.py
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

검증에는 60개 execution 재생, 60개 회복 정상 분기와 60개 오류 분기의 재생, 오류 관측·공통 checkpoint 검사, 정답만 저장하고 삭제하지 않는 실패 사례, 순서가 다른 합법적인 복구, 부분 경로 우회 검사가 포함됩니다. 실제 모델 테스트와 구분합니다.

[정상 시나리오](tasks/EXECUTION_TASKS.md), [회복 설계·목록](tasks/RECOVERY_TASKS.md)을 함께 확인합니다. 20개 기본 시나리오의 대응 조건을 독립 표본 60개처럼 분석하지 않습니다. 이번 회복 조건은 ‘잘못된 요청의 결과가 추가 저장된 오류’ 하나이며 모든 오류 유형에 일반화하지 않습니다. BFCL은 내부 실험이 안정화된 뒤 외부 검증용으로 별도 추가합니다.
