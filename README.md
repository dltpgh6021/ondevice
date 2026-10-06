# Quantization Effects on Long-Horizon Agent Performance

Windows-first starter project for comparing one Qwen3-1.7B checkpoint at three precisions:

- `F16/BF16` reference (record the actual GGUF type)
- `Q8_0`
- `Q4_K_M`

The first milestone is deliberately small: one deterministic refund workflow, four dependent tool calls, and a JSONL trajectory log. The same runner can use a scripted client to verify the harness or a local `llama.cpp` OpenAI-compatible server to test a model.

## Why Windows first

The current machine has `wsl.exe` but no usable WSL distribution was visible, while the project will eventually use Windows `llama-server.exe` and native CPU measurements. Keeping the runner, model files, and logs on the same Windows filesystem avoids path and timing differences during the first experiment. WSL can be added later if it becomes useful for model conversion or automation.

## Setup

Install Python 3.12 for Windows, then from this directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The smoke harness itself uses only the Python standard library. `langgraph` is listed for the next step, when the runner loop is moved into the project's LangGraph graph without changing the task or grading format.

## Run the harness without a model

```powershell
python scripts\run_smoke.py --demo
```

This executes the fixed tool sequence and writes a trajectory log under `logs\smoke`.

## Run against llama.cpp

Start a local server for one GGUF model. The exact executable and model path depend on the llama.cpp build and the files you create:

```powershell
llama-server.exe -m ..\models\qwen3-1.7b-f16.gguf --host 127.0.0.1 --port 8080 --ctx-size 8192
```

In another PowerShell window:

```powershell
python scripts\run_smoke.py --endpoint http://127.0.0.1:8080/v1/chat/completions --model qwen3-1.7b
```

Run the same task and prompt settings for the F16/BF16, Q8_0, and Q4_K_M servers. Do not compare results until all three models produce parseable actions and complete at least one smoke run.

## Initial research protocol

The main study will later use 20 base scenarios, each expanded to required horizons 1, 3, and 5, for 60 tasks. The primary analysis is the precision-by-horizon change in task success. Secondary logs include first error step, invalid tool calls, recovery, generated tokens, elapsed time, and peak memory.

