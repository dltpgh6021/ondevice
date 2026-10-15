"""Small resource-based workflows with observable, unavoidable dependencies.

Search contexts, offers, details and drafts are actual server resources. An
identifier guessed from a gold answer does not exist before its creating call.
No exact gold order, chosen candidate or call count is checked by the tools.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from .environment import MockEnvironment, SYSTEM_PROMPT
from .tasks import Task, stable_hash


WORKFLOWS = {
    "shopping": {
        "names": ("read_purchase_request", "find_product_offers", "get_product_details", "prepare_purchase", "confirm_purchase"),
        "noun": "purchase",
    },
    "schedule": {
        "names": ("read_schedule_request", "find_available_slots", "get_slot_details", "prepare_event", "confirm_event"),
        "noun": "calendar event",
    },
    "travel": {
        "names": ("read_travel_request", "find_travel_options", "get_fare_details", "prepare_itinerary", "confirm_itinerary"),
        "noun": "travel itinerary",
    },
    "place": {
        "names": ("read_visit_request", "find_place_offers", "get_visit_details", "prepare_visit", "confirm_visit"),
        "noun": "visit plan",
    },
}


def resource_id(spec: dict[str, Any], kind: str, *parts: str) -> str:
    """Stable opaque IDs, identical across the three horizons of one scenario."""
    return f"{kind}_{stable_hash([spec['request_id'], kind, *parts])[:16]}"


def completion_condition_matches(state: dict[str, Any], condition: dict[str, Any]) -> bool:
    required = condition["completed_request_equals"]
    completed = state.get("completed_requests", {})
    return completed == {required["request_id"]: required}


class ExecutionEnvironment(MockEnvironment):
    """The same five tool schemas are exposed at every horizon within a domain."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = copy.deepcopy(spec)
        self.names = WORKFLOWS[spec["domain"]]["names"]
        self.state: dict[str, Any] = {
            "contexts": {}, "offers": {}, "details": {}, "drafts": {},
            "completed_requests": {},
        }

    def tool_definitions(self) -> list[dict[str, Any]]:
        noun = WORKFLOWS[self.spec["domain"]]["noun"]
        definitions = (
            (self.names[0], "Read a request and create its search context. Returns constraints and context_id.", {"request_id": "Request ID from the user."}),
            (self.names[1], "Search using an existing context_id. Returns candidate summaries and offer_id values. Choose according to the request constraints and ranking rule.", {"context_id": "Context ID returned by the request tool."}),
            (self.names[2], "Read an existing offer. Returns detail_id and available option IDs with their conditions.", {"offer_id": "Offer ID returned by the search tool."}),
            (self.names[3], f"Prepare a {noun} draft using a detail_id and an option_id from that detail. Returns draft_id and the full proposed result. Does not complete the task.", {"detail_id": "Detail ID returned by the detail tool.", "option_id": "Option ID listed in that detail."}),
            (self.names[4], f"Confirm an existing draft_id to save the final {noun}. Reconfirming another draft for the same request replaces its earlier result.", {"draft_id": "Draft ID returned by the preparation tool."}),
            ("inspect_state", "Read only the saved final result. Does not expose private fixtures or create workflow resources.", {}),
        )
        return [
            {"type": "function", "function": {
                "name": name, "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {key: {"type": "string", "description": text} for key, text in props.items()},
                    "required": list(props), "additionalProperties": False,
                },
            }}
            for name, description, props in definitions
        ]

    def _failed(self, error: str, **extra: Any) -> dict[str, Any]:
        return {"ok": False, "error": error, **extra}

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        schema = next((item["function"]["parameters"] for item in self.tool_definitions()
                       if item["function"]["name"] == tool_name), None)
        if schema is None:
            return self._failed("unknown_tool", tool=tool_name)
        if set(arguments) != set(schema["required"]) or any(not isinstance(value, str) for value in arguments.values()):
            return self._failed("invalid_arguments")
        request_id = self.spec["request_id"]

        if tool_name == self.names[0]:
            if arguments["request_id"] != request_id:
                return self._failed("unknown_request")
            context_id = resource_id(self.spec, "context")
            request = copy.deepcopy(self.spec["request"])
            self.state["contexts"][context_id] = request
            return {"ok": True, "request_id": request_id, "context_id": context_id, "request": request}

        if tool_name == self.names[1]:
            if arguments["context_id"] not in self.state["contexts"]:
                return self._failed("context_not_found", hint="Read the request first.")
            offers = []
            for candidate in self.spec["candidates"]:
                offer_id = resource_id(self.spec, "offer", candidate["candidate_id"])
                self.state["offers"][offer_id] = copy.deepcopy(candidate)
                offers.append({"offer_id": offer_id, "candidate_id": candidate["candidate_id"], **candidate["summary"]})
            return {"ok": True, "offers": offers}

        if tool_name == self.names[2]:
            candidate = self.state["offers"].get(arguments["offer_id"])
            if candidate is None:
                return self._failed("offer_not_found", hint="Use an offer_id from the search result.")
            detail_id = resource_id(self.spec, "detail", candidate["candidate_id"])
            self.state["details"][detail_id] = copy.deepcopy(candidate)
            return {"ok": True, "detail_id": detail_id, "candidate_id": candidate["candidate_id"],
                    **copy.deepcopy(candidate["summary"]), "options": copy.deepcopy(candidate["options"])}

        if tool_name == self.names[3]:
            candidate = self.state["details"].get(arguments["detail_id"])
            if candidate is None:
                return self._failed("detail_not_found", hint="Read the selected offer details first.")
            option = next((item for item in candidate["options"] if item["option_id"] == arguments["option_id"]), None)
            if option is None:
                return self._failed("unknown_option")
            if not option["available"]:
                return self._failed("option_unavailable")
            draft_id = resource_id(self.spec, "draft", candidate["candidate_id"], option["option_id"])
            completion = {"request_id": request_id, "domain": self.spec["domain"],
                          "candidate_id": candidate["candidate_id"], "option_id": option["option_id"],
                          "result": {**self.spec["request"]["result_fields"], **candidate["summary"],
                                     "option_kind": option["kind"]}}
            self.state["drafts"][draft_id] = completion
            return {"ok": True, "draft_id": draft_id, "proposed_result": copy.deepcopy(completion)}

        if tool_name == self.names[4]:
            completion = self.state["drafts"].get(arguments["draft_id"])
            if completion is None:
                return self._failed("draft_not_found", hint="Prepare a draft first.")
            self.state["completed_requests"][request_id] = copy.deepcopy(completion)
            return {"ok": True, "saved_result": copy.deepcopy(completion)}

        return {"ok": True, "completed_requests": copy.deepcopy(self.state["completed_requests"])}

    def prepare_condition(self, task: Task, condition: str) -> tuple[list[dict[str, Any]], str | None]:
        if condition not in {"free", "normal_restart"}:
            raise ValueError("execution-v2 only supports free and normal_restart; recovery redesign is pending")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT + " Resource IDs must come from the user context or tool results. Use the request ranking rule, including its tie-breaker. Prepared drafts are not final until confirmed."},
            {"role": "user", "content": task.query},
        ]
        prefix = list(task.initial_actions or [])
        if condition == "normal_restart":
            prefix += task.gold_actions[:task.checkpoint_step]
        for index, action in enumerate(prefix):
            result = self.execute(action["tool"], action.get("arguments", {}))
            if not result["ok"]:
                raise ValueError(f"{task.task_id}: invalid provided context: {result}")
            call_id = f"provided_{index + 1}"
            messages.append({"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {
                "name": action["tool"], "arguments": json.dumps(action.get("arguments", {}), ensure_ascii=False),
            }}]})
            messages.append({"role": "tool", "tool_call_id": call_id, "name": action["tool"],
                             "content": json.dumps(result, ensure_ascii=False)})
        remaining = task.gold_actions[task.checkpoint_step:] if condition == "normal_restart" else task.gold_actions
        return messages, remaining[0]["tool"] if remaining else None

    def check_success(self, condition: dict[str, Any]) -> bool:
        return completion_condition_matches(self.state, condition)
