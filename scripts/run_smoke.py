from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_runner import AgentRunner
from src.llm_client import LlamaCppClient, ScriptedClient


def load_task() -> dict:
    with (ROOT / "tasks" / "smoke_refund.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the first Agent harness smoke test.")
    parser.add_argument("--demo", action="store_true", help="Use a scripted client; no model server required.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", default="qwen3-1.7b")
    args = parser.parse_args()
    if args.demo and args.endpoint != "http://127.0.0.1:8080/v1/chat/completions":
        parser.error("use either --demo or --endpoint, not both")

    if args.demo:
        client = ScriptedClient(actions=[
            {"action": "tool", "tool": "lookup_customer", "arguments": {"email": "alice@example.com"}},
            {"action": "tool", "tool": "get_order", "arguments": {"order_id": "O100"}},
            {"action": "tool", "tool": "approve_refund", "arguments": {"order_id": "O100"}},
            {"action": "tool", "tool": "refund_order", "arguments": {"order_id": "O100"}},
        ])
        label = "demo"
    else:
        client = LlamaCppClient(endpoint=args.endpoint, model=args.model)
        label = args.model

    result = AgentRunner(client=client).run(load_task())
    result["runner_label"] = label
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    output_dir = ROOT / "logs" / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{label.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"success": result["success"], "steps": result["steps_executed"], "log": str(output_file)}, ensure_ascii=False))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

