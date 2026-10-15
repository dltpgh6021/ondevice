"""Model clients with one normalized action format."""

from __future__ import annotations

import json
import http.client
import math
import urllib.error
import urllib.request
from typing import Any


def _reject_nonfinite(value: str) -> Any:
    raise ValueError(f"nonfinite JSON number: {value}")


def _parse_finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite JSON number: {value}")
    return number


class InfrastructureError(RuntimeError):
    """An unusable server/transport response, not a model behavior failure."""

    def __init__(self, message: str, *, kind: str = "transport_error") -> None:
        super().__init__(message)
        self.kind = kind


def _response_usage(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage")
    if usage is None:
        return {}
    if not isinstance(usage, dict) or any(
        key in usage and (type(usage[key]) is not int or usage[key] < 0)
        for key in ("prompt_tokens", "completion_tokens")
    ):
        raise InfrastructureError("invalid token usage in server response", kind="response_protocol")
    return usage


class ModelOutputError(ValueError):
    """An observable model response that violates the single-action protocol."""

    def __init__(self, message: str, *, kind: str = "invalid_arguments",
                 payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.payload = payload or {}
        self.usage = _response_usage(self.payload)
        choices = self.payload.get("choices") or []
        choice = choices[0] if choices else {}
        self.finish_reason = choice.get("finish_reason")
        message_value = choice.get("message") or {}
        calls = message_value.get("tool_calls") or []
        self.attempted_tool_call_count = len(calls) if isinstance(calls, list) else 0
        if not self.attempted_tool_call_count and message_value.get("function_call"):
            self.attempted_tool_call_count = 1


def _normalize_arguments(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ModelOutputError("tool arguments are not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ModelOutputError("tool arguments must decode to an object")
        return parsed
    raise ModelOutputError("unsupported tool arguments")


def parse_chat_response(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise InfrastructureError("chat response is not an object", kind="response_protocol")
    choices = payload.get("choices")
    if (not isinstance(choices, list) or not choices or not isinstance(choices[0], dict)
            or not isinstance(choices[0].get("message"), dict)):
        raise InfrastructureError("chat response did not contain a valid choice/message", kind="response_protocol")
    message = choices[0]["message"]
    _response_usage(payload)
    if choices[0].get("finish_reason") == "length":
        raise ModelOutputError("model output reached the token limit", kind="output_truncated", payload=payload)
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        if (not isinstance(tool_calls, list) or any(not isinstance(call, dict) for call in tool_calls)):
            raise ModelOutputError("tool_calls must be a list of call objects", kind="invalid_action", payload=payload)
        if len(tool_calls) != 1:
            raise ModelOutputError("exactly one tool call per response is allowed", kind="multiple_tool_calls", payload=payload)
        call = tool_calls[0]
        function = call.get("function") or {}
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
            raise ModelOutputError("tool call needs a function name", kind="invalid_action", payload=payload)
        try:
            arguments = _normalize_arguments(function.get("arguments"))
        except ModelOutputError as exc:
            raise ModelOutputError(str(exc), kind=exc.kind, payload=payload) from exc
        return {
            "type": "tool_call",
            "name": function.get("name"),
            "arguments": arguments,
        }

    legacy_call = message.get("function_call")
    if legacy_call:
        if not isinstance(legacy_call, dict) or not isinstance(legacy_call.get("name"), str) or not legacy_call["name"]:
            raise ModelOutputError("function_call needs a function name", kind="invalid_action", payload=payload)
        try:
            arguments = _normalize_arguments(legacy_call.get("arguments"))
        except ModelOutputError as exc:
            raise ModelOutputError(str(exc), kind=exc.kind, payload=payload) from exc
        return {
            "type": "tool_call",
            "name": legacy_call.get("name"),
            "arguments": arguments,
        }

    content = message.get("content")
    if content is None:
        content = ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{"):
            try:
                candidate = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ModelOutputError("JSON action is not valid JSON", kind="invalid_action", payload=payload) from exc
            if isinstance(candidate, dict) and candidate.get("type") == "tool_call":
                if not isinstance(candidate.get("name"), str) or not candidate["name"]:
                    raise ModelOutputError("JSON action needs a function name", kind="invalid_action", payload=payload)
                try:
                    arguments = _normalize_arguments(candidate.get("arguments"))
                except ModelOutputError as exc:
                    failure = ModelOutputError(str(exc), kind=exc.kind, payload=payload)
                    failure.attempted_tool_call_count = 1
                    raise failure from exc
                return {
                    "type": "tool_call",
                    "name": candidate.get("name"),
                    "arguments": arguments,
                }
            if isinstance(candidate, dict) and candidate.get("type") == "finish":
                return {"type": "finish", "answer": str(candidate.get("answer", ""))}
            raise ModelOutputError("JSON response is not a supported action", kind="invalid_action", payload=payload)
    else:
        raise ModelOutputError("response content must be text", kind="invalid_action", payload=payload)
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
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature) or temperature < 0):
            raise ValueError("temperature must be finite and nonnegative")
        if type(seed) is not int or not 0 <= seed < 2 ** 32 - 1:
            raise ValueError("seed must be an integer from 0 through 4294967294 (not random)")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be finite and positive")
        self.base_url = base_url.rstrip("/")
        self.endpoint = self.base_url + "/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def request_settings(self) -> dict[str, Any]:
        """Exact, model-independent settings sent on EVERY inference request.

        Deliberately uncached, non-thinking and one-action-at-a-time. Only the
        temperature sampler is active, so inherited filters/penalties cannot
        silently change the sampling chain between servers.
        """
        return {
            "temperature": self.temperature, "seed": self.seed, "max_tokens": self.max_tokens,
            "top_p": 1.0, "top_k": 0, "min_p": 0.0,
            "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "samplers": ["temperature"], "dynatemp_range": 0.0, "mirostat": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none",
            "parallel_tool_calls": False, "cache_prompt": False, "stream": False, "n": 1,
            "tool_choice": "auto",
        }

    def _request_json(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"), parse_constant=_reject_nonfinite,
                                     parse_float=_parse_finite_float)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise InfrastructureError(f"llama-server HTTP {exc.code}: {body}", kind="http_error") from exc
        except urllib.error.URLError as exc:
            raise InfrastructureError(f"could not reach llama-server at {request.full_url}: {exc}", kind="connection_error") from exc
        except (TimeoutError, OSError, http.client.HTTPException) as exc:
            raise InfrastructureError(f"llama-server transport failed: {exc}", kind="transport_error") from exc
        except (ValueError, UnicodeError) as exc:
            raise InfrastructureError("llama-server returned invalid response JSON", kind="response_protocol") from exc
        if not isinstance(payload, dict):
            raise InfrastructureError("llama-server response is not an object", kind="response_protocol")
        return payload

    def get_server_properties(self) -> dict[str, Any]:
        """Read-only properties; router/multi-model mode is outside this runner."""
        return self._request_json(urllib.request.Request(self.base_url + "/props", method="GET"))

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        request_body = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            **self.request_settings(),
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        payload = self._request_json(request)
        action = parse_chat_response(payload)
        return {
            "action": action,
            "usage": payload.get("usage") or {},
            "raw_response": payload,
            "finish_reason": payload["choices"][0].get("finish_reason"),
            "attempted_tool_call_count": int(action["type"] == "tool_call"),
        }

