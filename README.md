# Quantized Agent Stability

현재 브랜치는 **`experiment/phase2-position-controls`**이며, 현재 작업은 **IBM Granite-4.0-Micro FP16 후보의 소규모 비교 6회**다. 원격 WSL 실행은 **[PHASE2_GRANITE.md](PHASE2_GRANITE.md)의 0~6절**을 따른다. 이전 Qwen3-4B-Instruct-2507은 비교 기준으로 보존하며, 최종 양자화 기준 모델은 후보 평가 후 결정한다. 마지막 수정일은 2026-11-21이다.

연구 목표는 같은 원본에서 만든 FP16·Q8_0·Q4_K_M Agent의 오류 발생, 오류 회복과 효율성을 비교하는 것이다. Qwen FP16 파일럿과 입력 진단 132회 이후 h3 문구 비교 72회에서 세 문구 모두 정상 성공 3/12·오류 성공 0/12였다. 후속 공식 원본·GGUF 비교는 경로별 6회에서 정상 성공 1/3·오류 성공 0/3, 도구 선택도 같았다. 과제와 채점을 유지하고 다른 모델의 기준 성능을 확인한다. 아래 본문의 Qwen 설치·서버 명령은 이전 실험을 재현할 때 사용한다.

| 항목 | 현재 실행 기준 |
|---|---|
| 모델 | 후보: IBM Granite-4.0-Micro, 공식 BF16 체크포인트를 FP16으로 로드 |
| 모델 파일 | `models/granite-4.0-micro-hf/`에 고정 리비전의 공식 파일 다운로드 |
| 현재 명령 | `.venv-hf/bin/python scripts/check_phase2_granite.py prepare / run / compare` |
| 실행 수 | h3에서 개발 시나리오 3개 × 정상/오류 2조건 = Granite 6회. 이전 공식 Qwen 6회와 비교 |
| 과제 | `dependent-purchases-v4`, 모델 입력은 영어 |
| horizon | 같은 오류를 수정하면서 완료할 남은 묶음 항목 수 1·3·5 |
| 추론 설정 | thinking off, temperature 0, seed 42, 응답당 512토큰, 워밍업 2회 |
| 현재 실행 방식 | Transformers FP16·SDPA, 입력과 출력 합계 최대 8192토큰. llama-server 재빌드 불필요 |
| 확인한 실험 장비 | Ubuntu 22.04 / WSL2, RTX 3080 10GB, RAM 약 15GB |
| 4B 모델 준비 상태 | FP16과 manifest 준비 완료. Q8·Q4는 기준 모델·설정 확정 후 생성 |

모델 원본 설명과 사용하는 GGUF 배포 위치는 아래에서 확인할 수 있다. 다운로드에는 기존 파일럿과 같은 고정 리비전을 사용한다. [Qwen 모델 설명](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [사용한 F16 GGUF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/blob/3325f405027441dd81d27cc87479e6d60656fc92/Qwen3-4B-Instruct-2507-F16.gguf)

## 실행 경로 선택

지금은 **[Granite FP16 비교 안내](PHASE2_GRANITE.md)**를 따른다. 같은 PC의 이전 공식 Qwen 결과를 재사용한다. 기존 llama-server가 GPU에 남아 있다면 확인 후 종료하고 Granite를 로드한다. 코드·가이드 업로드는 사용자가 수행한다.

이전 공식 원본·GGUF 비교를 재현할 때는 [전용 실행 안내](PHASE2_BACKEND_CHECK.md)의 1~5절을 따른다. 아래 Qwen 준비 경로는 그 재현을 위한 안내다.

- **기존 원격 PC**라면 1절에서 코드를 업데이트하고 3절의 파일 존재 여부를 확인한 뒤 5절에서 기존 파일럿 폴더와 서버를 확인한다. 이후 7절을 실행한다.
- **새 PC**라면 1~6절에서 환경·모델·그 장비의 FP16 개발 파일럿을 준비한 뒤 7절을 실행한다. 기존 설정 잠금에는 장비·서버 빌드·템플릿 정보가 들어 있다.
- 후속 정밀도 비교 설계는 [PHASE2.md](PHASE2.md), 이전 입력 진단 132회와 추가 지표는 [PHASE2_DIAGNOSTICS.md](PHASE2_DIAGNOSTICS.md)에 있다.
- Qwen3-1.7B와 네 분야 작업 흐름을 사용한 1차 실험 안내는 [README_PHASE1.md](README_PHASE1.md)에 보관했다.

## 1. 코드와 Python 환경

필요한 도구는 Git, Python 3.10 이상, llama.cpp다. 아래 1~8절은 **Ubuntu/WSL의 bash** 기준이다. Windows CMD와 macOS의 차이는 9절에 있다. 명령은 한 줄씩 실행하고 오류가 나면 해당 단계의 출력을 확인한다.

새 Ubuntu에서 기본 도구가 없으면 먼저 설치한다.

```bash
sudo apt-get update
sudo apt-get install -y git build-essential python3-venv curl libcurl4-openssl-dev
```

처음 가져오는 PC에서 실행한다.

```bash
git clone --branch experiment/phase2-position-controls --single-branch https://github.com/dltpgh6021/ondevice.git "$HOME/ondevice-phase2"
cd "$HOME/ondevice-phase2"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install "psutil>=6.0,<8.0"
```

이미 저장소가 있다면 업데이트한다. Git이 로컬 수정 때문에 전환이나 병합을 거절하면 해당 출력을 확인한다.

```bash
cd "$HOME/ondevice-phase2"
git fetch origin
git switch experiment/phase2-position-controls
git pull --ff-only origin experiment/phase2-position-controls
source .venv/bin/activate
python -m pip install "psutil>=6.0,<8.0"
git log -1 --oneline
```

두 경로 모두 다음 검증을 실행한다.

```bash
python -m unittest discover -s tests -p "test_phase2*.py" -v
```

현재 예상 결과는 **40개 테스트와 `OK`**다. `psutil`은 서버 PID와 실행 옵션 확인에도 사용한다. 이후 명령은 프로젝트 루트와 활성화된 가상환경에서 실행한다. 창을 새로 열면 `cd`와 가상환경 활성화를 다시 한다.

## 2. NVIDIA GPU용 llama.cpp 준비

기존 실험 PC에서는 아래 명령으로 설치 상태를 확인한다.

```bash
nvidia-smi
nvcc --version
"$HOME/llama.cpp/build/bin/llama-server" --version
```

기존 4B 로그의 서버 빌드는 `b1-e95426f`다. 새 PC에는 아래 소스 리비전으로 빌드한다. CUDA Toolkit 설치가 필요하며, 기존 WSL 장비에서는 12.4를 사용했다. `nvcc`가 없다면 장비에 맞는 CUDA Toolkit부터 설치한다.

```bash
python -m pip install "cmake>=3.28"
git clone https://github.com/ggml-org/llama.cpp.git "$HOME/llama.cpp"
git -C "$HOME/llama.cpp" checkout e95426f
cmake -S "$HOME/llama.cpp" -B "$HOME/llama.cpp/build" -DGGML_CUDA=ON
cmake --build "$HOME/llama.cpp/build" --config Release -j 4 --target llama-server llama-quantize
"$HOME/llama.cpp/build/bin/llama-server" --version
```

빌드 후 같은 코드를 쓰더라도 장비별 FP16 로드와 메모리 여유는 확인한다. RTX 3080 10GB에서는 ubatch 512로 추론 중 메모리 오류가 발생해 **128**을 사용했다. [llama.cpp 빌드 안내](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md), [기준 소스 리비전](https://github.com/ggml-org/llama.cpp/commit/e95426f)

## 3. 4B FP16 모델과 manifest

기존 PC라면 두 파일을 확인한다.

```bash
ls -lh models/qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-F16.gguf
ls -l models/qwen3-4b-instruct-2507/manifest.json
```

새 PC는 모델 폴더를 만들고 다운로드한다. 파일은 약 **8.05GB, 7.50GiB**다. 기존 장비의 약 1.7MiB/s 네트워크에서는 약 78분 걸렸으며 실제 시간은 속도에 따라 달라진다.

```bash
mkdir -p models/qwen3-4b-instruct-2507
MODEL_FILE="$PWD/models/qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-F16.gguf"
curl -L --fail --retry 3 -C - -o "$MODEL_FILE.part" "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/3325f405027441dd81d27cc87479e6d60656fc92/Qwen3-4B-Instruct-2507-F16.gguf"
printf '%s  %s\n' '336228a2810f17724ecd6e9e2e40e8381eee81e886316fee311ade2899c31030' "$MODEL_FILE.part" | sha256sum -c -
```

해시 검증에서 **`OK`가 나오면** 파일을 확정한다.

```bash
mv -n "$MODEL_FILE.part" "$MODEL_FILE"
```

manifest는 모델의 식별표로 원본, 크기, 형식, SHA256과 배포 위치를 기록한다. 아래 Python 블록 전체를 프로젝트 루트에서 실행하면 해시를 검사하고 FP16 manifest를 모델 옆에 만든다. 기존 manifest가 있으면 원본 해시를 확인한다.

```bash
python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
from scripts.quantize_models import file_info

model = Path("models/qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-F16.gguf").resolve()
expected = "336228a2810f17724ecd6e9e2e40e8381eee81e886316fee311ade2899c31030"
info = file_info(model, label="Qwen3-4B-Instruct-2507-FP16", quantization="F16")
if info["sha256"] != expected:
    raise SystemExit("FP16 hash mismatch")
info.update(base_model="Qwen/Qwen3-4B-Instruct-2507",
            repository="unsloth/Qwen3-4B-Instruct-2507-GGUF",
            revision="3325f405027441dd81d27cc87479e6d60656fc92")
target = model.parent / "manifest.json"
manifest = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_model": info, "variants": [info], "reference_variants": [],
            "toolchain": {"commands": []}}
if target.exists():
    previous = json.loads(target.read_text(encoding="utf-8"))
    if previous["source_model"]["sha256"] != expected:
        raise SystemExit("Existing manifest has a different source")
    print("Existing manifest verified:", target)
else:
    with target.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print("Manifest saved:", target)
PY
```

## 4. 회복 과제 이해하기

각 상품 항목에서 **그 항목의 예산 이하인 후보 중 평점이 가장 높은 상품**을 저장한다.

- `normal_restart`는 0번에 정답이 저장돼 있고, 1번부터 h번까지의 묶음 선택이 남아 있다.
- `error_restart`는 0번에 예산 초과 상품이 저장돼 있고, 같은 묶음 선택이 남아 있다.
- `save_offer`를 같은 항목에 호출하면 기존 선택이 교체된다. 독립된 0번은 묶음 작업 전후에 수정할 수 있다.
- 묶음 1번을 저장하면 2번의 예산과 후보가 도구 결과에 나타난다. 같은 방식으로 다음 항목을 처리한다.
- `ok: true`는 도구 실행 성공이며, 채점은 마지막 저장 상태에 적용한다. `unsaved_lines`는 저장 여부를 표시한다.
- h3의 최소 도구 호출은 정상 3회, 오류는 수정 포함 4회다. 두 조건의 응답 상한은 8회이며 최종 답변과 형식 반려도 포함한다.

## 5. 기준 폴더와 현재 서버 확인

같은 PC에 완료한 4B FP16 파일럿이 있으면 폴더를 확인한다.

```bash
BASELINE_DIR="$(cat results/phase2/latest_v4_4b_dev.txt)"
ls -l "$BASELINE_DIR/dev.jsonl" "$BASELINE_DIR/settings.json" "$BASELINE_DIR/server_command.txt"
wc -l "$BASELINE_DIR/dev.jsonl"
```

`dev.jsonl`은 **180행**이다. 개발용 기본 108회와 h1 후보 위치 비교 72회의 합계다. 새 PC에 이 폴더가 없으면 아래 서버를 시작하고 6절로 간다.

기존 서버를 사용할 경우 실제 프로세스를 확인한다.

```bash
MODEL_FILE="$PWD/models/qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-F16.gguf"
SERVER_PID="$(cat "$BASELINE_DIR/server.pid")"
ps -ww -p "$SERVER_PID" -o pid=,args=
curl -fsS --max-time 10 http://127.0.0.1:18080/health
```

프로세스가 4B F16 모델을 사용하고 health가 **`{"status":"ok"}`**이면 7절로 진행한다. 서버를 재시작하면 PID가 바뀌므로 `SERVER_PID`에 현재 값을 지정한다.

서버가 종료돼 있거나 새 PC라면 다음 공통 명령으로 시작한다. 이후 로그에서 로드 완료를 확인한다.

```bash
mkdir -p results/phase2
RUN_DIR="$(mktemp -d "$PWD/results/phase2/4b_XXXXXXXX")"
MODEL_FILE="$PWD/models/qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-F16.gguf"
CUDA_VISIBLE_DEVICES=0 "$HOME/llama.cpp/build/bin/llama-server" -m "$MODEL_FILE" --alias local --jinja --reasoning off --host 127.0.0.1 --port 18080 --ctx-size 8192 --parallel 1 --batch-size 512 --ubatch-size 128 --cache-type-k f16 --cache-type-v f16 --flash-attn off --n-gpu-layers 99 > "$RUN_DIR/server_FP16.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$RUN_DIR/server.pid"
ps -ww -p "$SERVER_PID" -o args= > "$RUN_DIR/server_command.txt"
git rev-parse HEAD > "$RUN_DIR/project_commit.txt"
git -C "$HOME/llama.cpp" rev-parse HEAD > "$RUN_DIR/llama_cpp_commit.txt"
cp "$(dirname "$MODEL_FILE")/manifest.json" "$RUN_DIR/model_manifest.json"
```

기존 GPU에서 로드는 약 7초 걸렸다. 다음 명령으로 실제 상태를 확인한다.

```bash
tail -n 20 "$RUN_DIR/server_FP16.log"
kill -0 "$SERVER_PID" && curl -fsS --max-time 10 http://127.0.0.1:18080/health
```

기존 파일럿의 서버만 다시 시작한 경우 `BASELINE_DIR`는 원래 180행 폴더로 두고 현재 PID를 사용해 7절로 간다. 프로그램이 서버 옵션과 원본 설정을 비교한다.

## 6. 새 PC의 FP16 개발 파일럿

5절에서 서버를 시작한 같은 터미널에서 실행한다.

```bash
python -u scripts/phase2.py --split dev --precision FP16 --model-path "$MODEL_FILE" --server-pid "$SERVER_PID" --gpu-index 0 --settings-lock "$RUN_DIR/settings.json" --output "$RUN_DIR/dev.jsonl"
```

기존 RTX 3080의 180회는 과제 시간 합계가 약 14분이었다. 준비와 응답 편차를 포함하면 **15~25분**이 예상된다. `phase2.py`는 실행 완료 후 조건별 요약을 출력한다. 중간 진행은 다른 터미널에서 실제 폴더를 지정해 `wc -l <실행폴더>/dev.jsonl`로 확인할 수 있다.

마지막에 `FP16 pilot did not pass`가 나와도 180회가 완성되면 질문 진단에 사용할 수 있다. 인프라·내부 오류로 중단된 경우에는 원인을 해결하고 새 출력 폴더에서 실행한다. 완성 여부는 아래 명령으로 검사한다.

```bash
python -c "import sys; from pathlib import Path; from scripts.phase2 import read_runs, require_complete; p=Path(sys.argv[1]); require_complete(read_runs(p/'dev.jsonl'), 'dev', 'FP16'); print('Complete FP16 pilot: 180 trials')" "$RUN_DIR"
```

검사가 성공하면 기준 폴더를 기록한다.

```bash
printf '%s\n' "$RUN_DIR" > results/phase2/latest_v4_4b_dev.txt
BASELINE_DIR="$RUN_DIR"
```

## 7. 이번 h3 질문 문구 비교 72회

| 문구 | 변경 내용 |
|---|---|
| `original` | 원문을 같은 실행에서 재평가 |
| `next_offer` | 저장 결과에 다음 항목의 예산·후보가 나온다는 한 문장 추가 |
| `review_all` | 다음 후보 안내, 전체 필수 항목, 0번의 독립성, 수정·종료 조건을 명확하게 설명 |

먼저 실행 계획을 확인한다.

```bash
python scripts/check_phase2_prompt.py --horizon 3 --baseline-log "$BASELINE_DIR/dev.jsonl" --dry-run
```

출력에서 **`"trials": 72`**를 확인한다. 같은 터미널의 `MODEL_FILE`과 현재 `SERVER_PID`로 실행한다. 아래 출력 폴더는 프로그램이 만든다.

```bash
PROMPT_DIR="$BASELINE_DIR/prompt_h3_$(date +%Y%m%d_%H%M%S)"
python -u scripts/check_phase2_prompt.py --horizon 3 --baseline-log "$BASELINE_DIR/dev.jsonl" --model-path "$MODEL_FILE" --server-pid "$SERVER_PID" --output-dir "$PROMPT_DIR"
```

실행 중 `[1/72] ... success=True` 형식으로 진행 상황이 나온다. `False`도 관측할 행동 결과이므로 마지막까지 기다린다. 기존 장비의 평균 시간으로 **약 8~12분**을 예상한다. 모델과 설정 확인, 워밍업도 진행한다.

| 결과 파일 | 내용 |
|---|---|
| `trials.jsonl` | 72회의 입력·응답·도구 호출·상태·채점 |
| `summary.json` | 문구·정상/오류별 결과, 원문과의 대응 비교 |
| `metadata.json` | 모델·코드·기준 로그 해시, 서버 명령 |
| `settings.json`, `preflight.json` | 공통 설정과 준비·워밍업 기록 |

```bash
cat "$PROMPT_DIR/summary.json"
```

오류 조건의 `accessory_correct_at_end`는 0번을 최종적으로 올바르게 고친 건수이고, `successes`는 남은 묶음까지 포함한 전체 성공 건수다. 정상 조건의 성공과 `accessory_ever_damaged_episodes`도 확인한다. `review_all`은 여러 설명을 함께 바꾸므로 결과를 문구 전체의 효과로 해석한다.

h3 결과를 공유한 뒤 h5 확인 여부를 판단한다. h5도 72회이며 준비를 포함해 **약 12~18분**을 예상한다.

## 8. 결과 공유와 서버 종료

같은 터미널에서 결과 폴더와 기준 로그를 ZIP으로 묶는다.

```bash
python -m zipfile -c "$HOME/phase2_h3_prompt_check.zip" "$PROMPT_DIR" "$BASELINE_DIR/dev.jsonl" "$BASELINE_DIR/server_command.txt"
```

**ZIP을 Google Drive 등에 올리고 링크를 팀에 공유한다.** Git은 `results/`, GGUF, 가상환경을 제외한다. 새 PC의 실행에는 그 PC에서 만든 기준 FP16 로그도 포함된다. ZIP과 함께 OS·GPU·RAM, 실행 브랜치를 알려준다.

서버를 종료할 때는 자신이 시작한 현재 PID를 확인하고 종료한다.

```bash
ps -ww -p "$SERVER_PID" -o pid=,args=
kill "$SERVER_PID"
```

## 9. Windows CMD와 macOS

Windows CMD에서 코드를 내려받아 검증하는 예시다.

```cmd
git clone --branch experiment/phase2-position-controls --single-branch https://github.com/dltpgh6021/ondevice.git "%USERPROFILE%/ondevice-phase2"
cd /d "%USERPROFILE%/ondevice-phase2"
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install "psutil>=6.0,<8.0"
python -m unittest discover -s tests -p "test_phase2*.py" -v
```

Windows CMD에서 1~8절의 GPU 실행 절차를 사용하려면 `wsl`로 Ubuntu에 들어가 진행한다. `$HOME`, `$PWD`, `source`, `&`는 bash 문법이다. 가상환경은 WSL 안에서 새로 만든다. 네이티브 Windows 설치·서버 실행은 [기존 플랫폼 안내](README_PHASE1.md#12-환경-준비-처음-한-번)를 참고하고 현재 4B 경로와 ubatch 128을 적용한다.

macOS에서는 1절의 Git·가상환경 명령을 사용할 수 있다. Xcode Command Line Tools와 CMake를 설치하고, 소스 빌드 시 `-DGGML_CUDA=ON`을 `-DGGML_METAL=ON`으로 바꾼다. 모델 해시 검증은 `sha256sum -c -` 대신 `shasum -a 256 -c -`를 쓴다. `--gpu-index 0`은 생략한다. 5절의 `ps -ww -p ... -o args=`에서 형식을 지원하지 않으면 `python -c "import psutil, shlex, sys; print(shlex.join(psutil.Process(int(sys.argv[1])).cmdline()))" "$SERVER_PID"`로 서버 명령을 기록한다. 새 장비에서 새 파일럿과 설정 잠금을 만든 뒤 같은 Python 진단 명령으로 실행한다.

## 10. 자주 멈추는 지점

| 표시·상황 | 확인할 내용 |
|---|---|
| `check_phase2_prompt.py`를 찾지 못함 | 브랜치, `git pull` 성공 여부와 최신 커밋 확인 |
| `latest_v4_4b_dev.txt`가 없음 | 새 PC는 6절에서 파일럿 생성, 기존 PC는 완료한 180행 폴더 지정 |
| `settings differ` | 같은 장비·빌드·템플릿·설정인지 확인. 장비가 바뀌었다면 그 PC의 파일럿과 설정 생성 |
| `No such process`, health 연결 실패 | 서버 프로세스, 현재 PID, 포트 18080과 서버 로그 확인 |
| CUDA 메모리 부족 | `nvidia-smi`, 서버 로그, 4B F16·ubatch 128·parallel 1 설정 확인 |
| 포트 18080을 사용할 수 없음 | 기존 서버를 확인해 재사용하거나 자신이 시작한 서버를 종료 후 재시작 |
| `Infrastructure/internal error` | 부분 로그와 오류를 공유하고 원인 해결 후 새 출력 폴더에서 실행 |
| `success=False` | 선택·종료 행동의 채점 결과이므로 72회 완료까지 진행 |

## 후속 정밀도 비교

FP16 개발 기준을 확보한 모델·설정에서 같은 원본 F16으로 Q8_0·Q4_K_M을 직접 생성한다. 양자화 명령에는 4B 라벨과 별도 모델 폴더를 명시한다.

```bash
python scripts/quantize_models.py --source "$MODEL_FILE" --source-label Qwen3-4B-Instruct-2507-FP16 --output-dir models/qwen3-4b-instruct-2507 --quantizer "$HOME/llama.cpp/build/bin/llama-quantize" --llama "$HOME/llama.cpp/build/bin/llama-server"
```

본실험 계획은 60개 시나리오의 기본 540회와 위치 비교 72회를 합친 모델당 612회다. 세 정밀도를 같은 PC·서버 설정에서 실행하고 서버 RAM, 장치 VRAM, 시간과 토큰을 기록한다. 구체적인 개발 통과 기준과 본실험 명령은 [PHASE2.md](PHASE2.md)에 있다.
