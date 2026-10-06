from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class LLMClient(Protocol):
    def next_action(
        self,
        instruction: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ...


def parse_action(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
    try:
        action = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_json_action:{exc}") from exc
    if not isinstance(action, dict) or action.get("action") not in {"tool", "finish"}:
        raise ValueError("invalid_action_shape")
    if action["action"] == "tool" and not isinstance(action.get("tool"), str):
        raise ValueError("tool_action_missing_tool")
    if not isinstance(action.get("arguments", {}), dict):
        raise ValueError("arguments_must_be_object")
    return action


@dataclass
class LlamaCppClient:
    endpoint: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 256
    timeout_seconds: int = 180

    def next_action(
        self,
        instruction: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system = (
            "You are a tool-using agent. Choose exactly one next action. "
            "Return exactly one valid JSON object and no prose. "
            'Use either {"action":"tool","tool":"name","arguments":{}} or '
            '{"action":"finish","tool":null,"arguments":{},"message":"..."}. '
            "Use only the listed tools and use exactly the argument keys in each schema. "
            "For get_order, approve_refund, and refund_order, the argument key is "
            "order_id, never customer_id or name. Use values returned by earlier tool_result entries. "
            "Always call approve_refund successfully before calling refund_order. "
            "Do not use refund_order as a substitute for approval. "
            "Do not choose finish unless the requested task is complete and the latest tool_result "
            "confirms the final state. "
            "If a previous action has an error, correct that action and continue; do not finish until "
            "the requested task is complete. Do not include a reasoning or markdown block."
        )
        user = {
            "instruction": instruction,
            "available_tools": tools,
            "trajectory": history,
            "format_reminder": "JSON only; one action per turn.",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        return parse_action(result["choices"][0]["message"]["content"])


@dataclass
class ScriptedClient:
    actions: list[dict[str, Any]]
    index: int = 0

    def next_action(
        self,
        instruction: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.index >= len(self.actions):
            return {"action": "finish", "tool": None, "arguments": {}, "message": "done"}
        action = self.actions[self.index]
        self.index += 1
        return action

