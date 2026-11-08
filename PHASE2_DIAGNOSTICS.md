# 4B FP16 개발용 진단 실행

현재 목적은 긴 과제의 선택 오류와 저장된 오류를 수정하지 않는 행동을 분리해 확인하는 것이다. Qwen3-4B-Instruct-2507 FP16과 기존 개발 시나리오 12개를 사용한다. `dependent-purchases-v4`의 과제 생성·채점·본실험 통과 기준은 변경하지 않는다. 진단 버전은 `phase2-diagnostic-v1`이며 본실험 로그와 합치지 않는다.

## 무엇을 실행하는가

| 조건 | 횟수 | 제공하는 내용과 목적 |
|---|---:|---|
| isolated_choice | 60 | 기존 h5 자유 수행의 12개 시나리오에서 각 5개 항목을 첫 저장 직전에 보았던 후보·가격·평점·예산·순서 그대로 한 항목씩 제시한다. 올바른 경로로 새 후보를 만들거나 틀린 항목만 골라내지 않는다. 독립적인 선택과 실제 긴 대화 속 첫 선택을 비교한다. |
| history | 24 | h1 정상/오류 재개 각각 12개. 기존 입력을 그대로 다시 실행해 같은 실행 안의 대조군으로 사용한다. |
| state_only | 24 | 같은 h1 정상/오류 상태를 유지하고 과거 assistant 저장 호출·tool 반환 기록 대신 현재 상태를 사용자 입력에 제시한다. 원래 요청, 후보 순서와 실제 저장 상태를 유지한다. |
| history_review | 24 | 기존 h1 입력과 대화 기록 뒤에 전체 필수 항목 재검토와 실제 도구로 수정을 적용하라는 요청을 추가한다. 정상/오류 양쪽에 정확히 같은 문장을 쓰고 오류 존재·위치·정답은 알려주지 않는다. |
| 합계 | 132 | 새로운 본실험 문항을 만드는 작업이 아니라 기존 개발 데이터의 진단이다. |

모든 진단은 응답 상한 6회다. h1 대조군의 기존 상한과 같다. 독립 선택에서는 처음 저장한 선택의 정확도와 최종 성공을 따로 기록한다. 형식 반려와 조회도 응답 상한에 포함한다. 최초 정답 도달 시 자동으로 종료하지 않는다. 최종 답변에 교체 상품을 적어도 `save_offer`로 실제 상태를 바꾸지 않으면 수정으로 채점하지 않는다.

정상·오류 조건의 순서는 시나리오마다 번갈아 실행하고, 세 재개 방식의 순서도 시나리오마다 순환한다. 진단은 같은 모델의 FP16 파일과 기본 추론 설정, 서버의 batch/ubatch·KV·GPU offload 설정을 기존 파일럿과 맞춰 실행한다. 원본 로그는 성공률 기준에 미달해도 되지만, 오류 없이 180회를 완료하고 모든 h5 자유 수행 항목에 도달한 기록이어야 한다.

## 먼저 코드 반영과 오프라인 확인

로컬 변경을 GitHub에 올린 뒤 원격 Ubuntu/WSL 실험 PC의 프로젝트 루트에서 실행한다. 아래 명령은 한 줄씩 입력한다. 이 단계에는 모델 서버가 필요 없다.

```bash
cd "$HOME/ondevice-phase2"
git pull --ff-only origin experiment/phase2-position-controls
.venv/bin/python -m unittest discover -s tests -p "test_phase2*.py" -v
BASELINE_DIR="$(cat results/phase2/latest_v4_4b_dev.txt)"
.venv/bin/python scripts/diagnose_phase2.py --baseline-log "$BASELINE_DIR/dev.jsonl" --dry-run
```

예상 결과는 테스트 `OK`와 아래 목록이다. dry-run은 서버 요청이나 파일 쓰기를 수행하지 않는다.

```json
{
  "diagnostic_protocol": "phase2-diagnostic-v1",
  "total": 132,
  "isolated_choice": 60,
  "restart_probes": 72
}
```

원본 폴더에는 `dev.jsonl`, `settings.json`, `server_command.txt`가 필요하다. `latest_v4_4b_dev.txt`는 메모리 오류가 났던 14줄 실행이 아니라 ubatch 128로 완료한 180줄 폴더를 가리켜야 한다.

## 실제 실행

기존 4B 서버가 살아 있는 경우 같은 원격 터미널에서 아래 명령을 사용한다. 서버를 재시작했다면 `SERVER_PID`는 새 서버의 PID로 지정한다. 서버는 원래 파일럿과 같은 설정으로 실행한다: context 8192, parallel 1, batch 512, ubatch 128, KV f16/f16, flash-attn off, n-gpu-layers 99, reasoning off, host 127.0.0.1, port 18080. 서버 CLI에는 위 옵션을 긴 이름으로 명시한다. 실행기가 실제 PID의 명령과 원본 `server_command.txt`를 비교한다.

```bash
MODEL_FILE="$PWD/models/qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-F16.gguf"
SERVER_PID="$(cat "$BASELINE_DIR/server.pid")"
kill -0 "$SERVER_PID" && curl -fsS --max-time 10 http://127.0.0.1:18080/health
```

`status: ok`를 확인한 뒤 새 출력 경로로 실행한다. `DIAG_DIR`는 실행기가 생성하므로 미리 mkdir 하지 않는다.

```bash
DIAG_DIR="$BASELINE_DIR/diagnostic_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python -u scripts/diagnose_phase2.py --baseline-log "$BASELINE_DIR/dev.jsonl" --model-path "$MODEL_FILE" --server-pid "$SERVER_PID" --output-dir "$DIAG_DIR"
```

모델 파일 해시를 확인한 뒤 워밍업 2회를 수행한다. 기존 설정 잠금을 재사용하며 설정이 다르면 멈춘다. 과제마다 `[1/132] ... success=True`처럼 진행 상황을 출력한다. 예상 시간은 약 5~20분이며 실제 응답 길이·재시도에 따라 달라진다. 132회를 마치면 `summary.json`을 자동 저장한다. 이 진단에는 본실험 성공률 문턱을 적용하지 않으므로 행동 실패만으로 실행을 중단하지 않는다. 인프라·내부 오류가 발생하면 부분 기록을 보존하고 종료한다. 수정 후 새 출력 폴더에서 다시 시작한다.

## 결과 해석과 공유

`summary.json`에서 다음을 읽는다.

- isolated_choice: `first_saves_correct/60`을 원본의 `baseline_first_saves_correct/60`과 비교한다. `changed_wrong_to_right`, `changed_right_to_wrong`, `no_save`도 함께 확인한다. 단독 입력은 대화 맥락과 주의 대상도 바꾸므로 길이만의 순수 인과 효과라고 주장하지 않는다. 60개 판단은 12개 시나리오에 속한다.
- 각 재개 조건: `successes/12`는 부속품과 후속 항목이 모두 맞은 비율이다. 오류 조건의 `injected_error_resolved/12`는 주입 오류 자체를 해결한 비율이다. 두 지표를 구분한다.
- 정상 상태: `accessory_write_episodes`는 재저장 여부이며, `accessory_ever_damaged_episodes`와 `accessory_correct_at_end`로 정상 상태를 오히려 망가뜨리는지 확인한다. 올바른 부속품을 그대로 두는 것은 정상 행동이다.
- history 대조군: 원본 h1 실행과 성공 결과가 달라졌는지 `success_changed_from_baseline`으로 확인한다. 반복 결과가 다르면 trace를 비교한다.

프로그램은 최종 문장의 표현만으로 오류 인지를 자동 판정하지 않는다. 글에 오류나 정답을 언급했지만 실제 수정하지 않은 사례는 전체 trace로 별도 검토한다. state_only는 대화 기록과 관측 표현을 함께 바꾸며, history_review는 새 사용자 턴과 검토 지시를 추가한다. 이 효과들을 모델 내부 원인으로 단정하지 않는다.

공유할 파일은 다음과 같다. 모델 가중치는 포함되지 않는다.

```bash
cat "$DIAG_DIR/summary.json"
.venv/bin/python -m zipfile -c "$HOME/phase2_4b_diagnostics.zip" "$DIAG_DIR"
```

ZIP에는 `diagnostics.jsonl`(전체 trace), `summary.json`, `metadata.json`(원본 로그·모델·코드 해시와 실제 서버 명령), `settings.json`, `preflight.json`이 담긴다. results는 Git에 올리지 않고 별도로 공유한다.

진단에서 개선된 입력 구성이 확인되면 변경 이유와 범위를 고정하고 새 버전으로 전체 개발 파일럿 180회를 실행한다. h1에서 회복이 개선됐다는 사실만으로 h3·h5 통과를 가정하지 않는다. 정상·오류 재개 양쪽에 공통 지시를 적용하고 본실험 기준은 유지한다. 기준 성능을 확보한 뒤 선정한 동일 FP16에서 Q8_0·Q4_K_M을 직접 생성하고 정밀도 비교로 넘어간다. 모든 진단에서 실패가 지속되면 모델이나 평가 설계를 재검토한다. 문턱이나 정답 채점을 결과에 맞춰 바꾸지 않는다.
