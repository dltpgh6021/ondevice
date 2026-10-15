"""Verify a common experiment configuration and warm the server without tasks."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .client import ModelOutputError
from .tasks import stable_hash


PROTOCOL = "fixed-inference-warmup-v1"
_HARDWARE_FIELDS = ("hostname", "platform", "processor", "machine")
_MODEL_KEYS = {"model", "model_path", "model_alias", "model_aliases", "alias",
               "model_file_type", "model_size", "quantization"}
_WARMUP_MESSAGES = [
    {"role": "system", "content": "Use English. Respond with exactly one tool call."},
    {"role": "user", "content": "Call warmup_ping once with no arguments."},
]
_WARMUP_TOOLS = [{
    "type": "function",
    "function": {
        "name": "warmup_ping", "description": "A harmless warmup operation with no task data.",
        "parameters": {"type": "object", "properties": {}, "required": [],
                       "additionalProperties": False},
    },
}]


class PreparationError(RuntimeError):
    """The experiment cannot safely start under this configuration."""


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def _without_model_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_model_keys(item) for key, item in value.items()
                if key not in _MODEL_KEYS}
    if isinstance(value, list):
        return [_without_model_keys(item) for item in value]
    return value


def _server_configuration(props: Any, expected_context_size: int) -> dict[str, Any]:
    if not isinstance(props, dict):
        raise PreparationError("/props must return an object")
    if not isinstance(props.get("build_info"), str) or not props["build_info"].strip():
        raise PreparationError("/props is missing nonempty build_info; use a supported llama.cpp build")
    template = props.get("chat_template")
    if not isinstance(template, str) or not template.strip():
        raise PreparationError("/props is missing a nonempty chat_template")
    generation = props.get("default_generation_settings")
    if not isinstance(generation, dict) or not isinstance(generation.get("params"), dict):
        raise PreparationError("/props must contain default_generation_settings.params")
    n_ctx = generation.get("n_ctx")
    if type(n_ctx) is not int or n_ctx != expected_context_size:
        raise PreparationError(f"server context size must be {expected_context_size}; observed {n_ctx!r}")
    slots = props.get("total_slots")
    if type(slots) is not int or slots != 1:
        raise PreparationError(f"server total_slots must be 1; observed {slots!r}")
    configuration = {
        "build_info": copy.deepcopy(props["build_info"]),
        "n_ctx": n_ctx, "total_slots": slots,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "chat_template_caps": copy.deepcopy(props.get("chat_template_caps")),
        "default_generation_params": _without_model_keys(generation["params"]),
    }
    tool_template = props.get("chat_template_tool_use")
    if tool_template is not None:
        if not isinstance(tool_template, str):
            raise PreparationError("chat_template_tool_use must be text when present")
        configuration["chat_template_tool_use_sha256"] = hashlib.sha256(tool_template.encode("utf-8")).hexdigest()
    return configuration


def _load_lock(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise PreparationError(f"settings lock cannot be read: {path}") from exc
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise PreparationError("settings lock has an unsupported schema")
    configuration = value.get("configuration")
    if not isinstance(configuration, dict) or value.get("configuration_sha256") != stable_hash(configuration):
        raise PreparationError("settings lock configuration hash is invalid")
    return value


def _require_matching_lock(existing: dict[str, Any] | None, expected: dict[str, Any]) -> None:
    if existing is not None and existing["configuration_sha256"] != expected["configuration_sha256"]:
        changed = [key for key in expected["configuration"]
                   if stable_hash(existing["configuration"].get(key)) != stable_hash(expected["configuration"][key])]
        raise PreparationError(
            f"settings differ from the existing lock ({', '.join(changed) or 'configuration fields'}); restore the common server/request/hardware "
            "configuration, or intentionally choose a separate lock and result series"
        )


def prepare_run(
    client: Any, *, expected_context_size: int, warmup_requests: int,
    settings_lock_path: Path | str, report_path: Path | str, hardware: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed on setup differences; all warmup work stays outside task traces."""
    if type(expected_context_size) is not int or expected_context_size <= 0:
        raise PreparationError("expected_context_size must be a positive integer")
    if type(warmup_requests) is not int or warmup_requests < 1:
        raise PreparationError("warmup_requests must be an integer of at least 1")
    lock_path, destination = Path(settings_lock_path).resolve(), Path(report_path).resolve()
    if lock_path == destination:
        raise PreparationError("settings lock and preflight report must have different paths")
    if destination.exists():
        raise PreparationError(f"preflight report already exists; choose a new path: {destination}")
    report: dict[str, Any] = {
        "schema_version": 1, "preparation_protocol": PROTOCOL, "status": "preparing",
        "warmup": [],
        "verification_limits": [
            "Server-reported context, slot, template, build and generation settings are checked.",
            "This does not independently verify GPU offload, device identity or actual KV-cache allocation.",
            "Warmup uses a synthetic prompt, not evaluation tasks; correctness is not a preflight gate.",
        ],
    }
    try:
        existing = _load_lock(lock_path)
        props = client.get_server_properties()
        report["server_properties"] = copy.deepcopy(props)
        server = _server_configuration(props, expected_context_size)
        inference = dict(client.request_settings())
        inference["timeout_seconds"] = client.timeout_seconds
        warmup_settings = {
            "protocol": PROTOCOL, "requests": warmup_requests,
            "prompt_sha256": stable_hash({"messages": _WARMUP_MESSAGES, "tools": _WARMUP_TOOLS}),
        }
        configuration = {
            "inference_settings": inference, "server_configuration": server,
            "hardware": {key: hardware.get(key) for key in _HARDWARE_FIELDS},
            "warmup_settings": warmup_settings,
        }
        # Reject non-finite/non-JSON configuration values before computing or saving a lock.
        try:
            json.dumps(configuration, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise PreparationError("server/request configuration must contain finite JSON values") from exc
        lock = {"schema_version": 1, "configuration": configuration,
                "configuration_sha256": stable_hash(configuration)}
        report.update({"configuration": configuration, "settings_lock_sha256": lock["configuration_sha256"]})
        _require_matching_lock(existing, lock)
        for number in range(1, warmup_requests + 1):
            started = time.perf_counter()
            diagnostic: dict[str, Any] = {"request_number": number}
            report["warmup"].append(diagnostic)
            try:
                response = client.chat(copy.deepcopy(_WARMUP_MESSAGES), copy.deepcopy(_WARMUP_TOOLS))
                diagnostic.update({"status": "response_received", "usage": response.get("usage", {}),
                                   "finish_reason": response.get("finish_reason"),
                                   "action_type": response.get("action", {}).get("type")})
            except ModelOutputError as exc:
                diagnostic.update({"status": "model_output_error", "kind": exc.kind,
                                   "message": str(exc), "usage": exc.usage,
                                   "finish_reason": exc.finish_reason})
            except Exception as exc:
                diagnostic.update({"status": "failed", "kind": getattr(exc, "kind", type(exc).__name__),
                                   "message": str(exc)})
                raise
            finally:
                diagnostic["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        if existing is None:
            try:
                _write_new_json(lock_path, lock)
            except FileExistsError:
                # A concurrently started run may create the same lock; never replace it.
                concurrent = _load_lock(lock_path)
                if concurrent is None:
                    raise PreparationError("settings lock disappeared during concurrent creation")
                _require_matching_lock(concurrent, lock)
        report["status"] = "ready"
        _write_new_json(destination, report)
        return {
            "server_configuration": server, "server_configuration_sha256": stable_hash(server),
            "llama_cpp_version": copy.deepcopy(props["build_info"]),
            "inference_settings": inference, "preparation_protocol": PROTOCOL,
            "warmup_settings": warmup_settings, "warmup": report["warmup"],
            "preflight_report_path": str(destination),
            "settings_lock_sha256": lock["configuration_sha256"],
        }
    except Exception as exc:
        report.update({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)}})
        if not destination.exists():
            _write_new_json(destination, report)
        raise
