# Quantized Agent Stability

양자화가 장기 Agent의 수행 안정성에 미치는 영향을 비교하는 실험 저장소입니다.

이 프로젝트는 FP16·Q8·Q4 모델을 같은 Agent framework와 동일한 mock tool 환경에서 실행합니다. 최종 성공률만 비교하지 않고, 다음 두 현상을 분리해 측정합니다.

1. 모델이 오류를 얼마나 자주 발생시키는가
2. 같은 오류가 주어졌을 때 얼마나 잘 회복하는가

## 실험 조건과 task 유형

- `free`: `execution` task를 초기 상태에서 처음부터 수행합니다. 양자화 수준과 horizon에 따른 도구 선택·정상 수행을 측정합니다.
- `recovery`: `recovery` task를 이미 잘못된 상태에서 시작합니다. 같은 오류를 받은 뒤 복구하는 능력을 측정합니다.
- `normal_restart`: 모든 모델에 동일한 올바른 중간 상태를 제공합니다.

`horizon`은 과제를 완료하는 데 필요한 **정답 tool call 수**입니다. 재시도와 오류 회복에 사용된 추가 호출 수는 horizon과 별도로 기록합니다.

## 현재 구현 범위

초기에는 쇼핑 예제 하나로 코드 구조를 검증했으며, 현재는 정상 수행 60개와 회복 수행 60개, 총 120개 task로 구성되어 있습니다.

- 결정론적 mock tool 환경
- 정상 상태와 오류 상태의 checkpoint 생성
- 오류 상태 주입
- 오류 상태에서 사용하는 회복 전용 질문(`recovery_query`)
- FakeClient를 이용한 모델 없는 smoke test
- llama.cpp 서버와 통신하는 HTTP client
- 성공률·오류 위치·회복률·추가 호출 수 집계
- 실행 환경과 task/model hash 기록

이 구조를 바탕으로 20개 기본 시나리오를 horizon 1·3·5로 확장한 뒤, 각 시나리오에 정상 수행과 회복 수행을 한 세트씩 두었습니다.

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
│  ├─ agent.py          # 공통 Agent 실행 루프
│  ├─ metrics.py        # 성공·오류·회복 지표
│  └─ experiment.py     # batch 실행과 메타데이터 기록
└─ scripts/
   ├─ smoke_test.py     # 모델 없이 전체 흐름 검사
   ├─ run_experiment.py # 실제 llama.cpp 실행
   └─ analyze.py        # JSONL 결과 집계
```

모델 파일(`.gguf`), 가상환경, 실행 로그는 Git에 올리지 않습니다.

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

분석 결과의 `recovery_comparison`에서 다음 값을 확인합니다.

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
- 실행 시간과 생성 토큰 수
- 선택적으로 최대 RAM 사용량

## 8. 연구용 task 구성

현재 `tasks/tasks.json`에 연구용 task 120개가 이미 포함되어 있습니다.

```text
20개 기본 시나리오 × horizon 1·3·5 × 2(task 유형) = 120개 task
```

정상 수행 task와 회복 task는 서로 대응하는 쌍입니다.

- `execution`: 오류가 없는 초기 상태에서 원래 질문을 수행
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

회복 task의 `checkpoint_step`은 0이며, 오류가 모델의 첫 호출 전에 주입됩니다. 따라서 정상 수행의 자연 오류율과 회복 수행의 회복률을 서로 다른 실험으로 측정할 수 있습니다.

horizon별 회복 경로는 다음처럼 구성했습니다. horizon 1은 즉시 교정 또는 stale 결과 재조회 1회, horizon 3은 상태 확인·오류 제거·목표 상태 적용, horizon 5는 추가 정보 조회를 포함한 뒤 오류 제거·목표 상태 적용의 순서입니다. `replace_cart_item`과 `replace_event`는 한 번의 교정 호출을 검증하기 위한 mock tool입니다.

BFCL은 내부 task가 안정화된 뒤 외부 검증 데이터로 별도 추가합니다. 처음부터 BFCL 실행 코드를 섞지 않는 이유는 핵심 실험의 상태·오류·회복 흐름을 단순하게 유지하기 위해서입니다.
