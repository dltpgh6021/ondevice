"""Model clients with one normalized action format."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def _normalize_arguments(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool arguments are not valid JSON: {value}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must decode to an object")
        return parsed
    raise ValueError("unsupported tool arguments")


def parse_chat_response(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("chat response did not contain choices")
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        call = tool_calls[0]
        function = call.get("function") or {}
        return {
            "type": "tool_call",
            "name": function.get("name"),
            "arguments": _normalize_arguments(function.get("arguments")),
        }

    legacy_call = message.get("function_call")
    if legacy_call:
        return {
            "type": "tool_call",
            "name": legacy_call.get("name"),
            "arguments": _normalize_arguments(legacy_call.get("arguments")),
        }

    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{"):
            try:
                candidate = json.loads(stripped)
            except json.JSONDecodeError:
                candidate = None
            if isinstance(candidate, dict) and candidate.get("type") == "tool_call":
                return {
                    "type": "tool_call",
                    "name": candidate.get("name"),
                    "arguments": _normalize_arguments(candidate.get("arguments")),
                }
    return {"type": "finish", "answer": str(content)}


class ScriptedClient:
    """Deterministic fake client for smoke tests and offline development."""

    def __init__(self, actions: list[dict[str, Any]]) -> None:
        self.actions = list(actions)
        self.calls = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        del messages, tools
        self.calls += 1
        if self.actions:
            action = self.actions.pop(0)
        else:
            action = {"type": "finish", "answer": "script exhausted"}
        return {"action": action, "usage": {}}


class LlamaCppClient:
    """Small OpenAI-compatible client for a local llama-server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 512,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.endpoint = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        request_body = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not reach llama-server at {self.endpoint}: {exc}") from exc
        return {
            "action": parse_chat_response(payload),
            "usage": payload.get("usage") or {},
        }

