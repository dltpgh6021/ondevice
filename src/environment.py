"""Deterministic mock tools used for controlled Agent comparisons."""

from __future__ import annotations

import copy
import json
from typing import Any

from .tasks import Task, stable_hash


SYSTEM_PROMPT = (
    "You are a tool-using Agent. Use the available tools to complete the user "
    "request. Inspect the current state when needed. There is no finish tool: "
    "when the requested final state has been achieved, reply with a short final "
    "answer that contains no tool call. A reply without a tool call ends the task."
)


class MockEnvironment:
    """A deterministic shopping, schedule, travel, and place-search world."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {
            "budget": 120.0,
            "products": {
                "keyboard_01": {"name": "Compact Keyboard", "category": "keyboard", "price": 45.0, "rating": 4.2},
                "keyboard_02": {"name": "Quiet Mechanical Keyboard", "category": "keyboard", "price": 79.0, "rating": 4.8},
                "keyboard_99": {"name": "Unrequested Keyboard", "category": "keyboard", "price": 35.0, "rating": 3.1},
                "mouse_01": {"name": "Ergonomic Mouse", "category": "mouse", "price": 39.0, "rating": 4.6},
                "headset_01": {"name": "Noise-Canceling Headset", "category": "headset", "price": 99.0, "rating": 4.7},
                "webcam_01": {"name": "Full HD Webcam", "category": "webcam", "price": 55.0, "rating": 4.4},
                "usb_hub_01": {"name": "Seven-Port USB Hub", "category": "usb_hub", "price": 32.0, "rating": 4.5},
            },
            "cart": [],
            "calendar": [],
            "weather": {
                "Seoul|2026-10-20": {"weather_id": "seoul_20261020_sunny", "condition": "sunny", "high": 23, "low": 15},
                "Seoul|2026-10-21": {"weather_id": "seoul_20261021_rain", "condition": "rain", "high": 18, "low": 12},
                "Seoul|2026-10-22": {"weather_id": "seoul_20261022_cloudy", "condition": "cloudy", "high": 20, "low": 13},
                "Busan|2026-10-20": {"weather_id": "busan_20261020_sunny", "condition": "sunny", "high": 25, "low": 17},
                "Busan|2026-10-21": {"weather_id": "busan_20261021_cloudy", "condition": "cloudy", "high": 22, "low": 16},
            },
            "places": {
                "Seoul|museum": [
                    {"place_id": "national_museum", "name": "National Museum", "category": "museum", "rating": 4.9}
                ],
                "Seoul|cafe": [
                    {"place_id": "quiet_cafe", "name": "Quiet Cafe", "category": "cafe", "rating": 4.7}
                ],
                "Seoul|restaurant": [
                    {"place_id": "green_table", "name": "Green Table", "category": "restaurant", "rating": 4.8}
                ],
                "Seoul|park": [
                    {"place_id": "han_river_park", "name": "Han River Park", "category": "park", "rating": 4.6}
                ],
                "Busan|beach": [
                    {"place_id": "gwangalli_beach", "name": "Gwangalli Beach", "category": "beach", "rating": 4.8}
                ],
            },
            "routes": {
                "home|national_museum": {"route_id": "home_national_museum", "duration_min": 32, "distance_km": 8.4},
                "home|quiet_cafe": {"route_id": "home_quiet_cafe", "duration_min": 18, "distance_km": 4.1},
                "home|green_table": {"route_id": "home_green_table", "duration_min": 24, "distance_km": 5.7},
                "home|han_river_park": {"route_id": "home_han_river_park", "duration_min": 27, "distance_km": 6.8},
                "station|han_river_park": {"route_id": "station_han_river_park", "duration_min": 25, "distance_km": 6.2},
                "office|green_table": {"route_id": "office_green_table", "duration_min": 21, "distance_km": 5.0},
                "hostel|gwangalli_beach": {"route_id": "hostel_gwangalli_beach", "duration_min": 14, "distance_km": 3.2},
            },
            "last_result": None,
        }

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.state = copy.deepcopy(snapshot)

    def state_hash(self) -> str:
        return stable_hash(self.state)

    def _remember(self, result: dict[str, Any]) -> dict[str, Any]:
        self.state["last_result"] = copy.deepcopy(result)
        return result

    def tool_definitions(self) -> list[dict[str, Any]]:
        def function(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
            schema: dict[str, Any] = {"type": "object", "properties": properties}
            if required:
                schema["required"] = required
            return {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}

        return [
            function("get_budget", "Return the available shopping budget.", {}),
            function("search_product", "Find products by category.", {"category": {"type": "string"}}, ["category"]),
            function("get_cart", "Return the current cart contents.", {}),
            function("add_to_cart", "Add one product to the cart.", {"product_id": {"type": "string"}}, ["product_id"]),
            function("remove_from_cart", "Remove one product from the cart.", {"product_id": {"type": "string"}}, ["product_id"]),
            function("replace_cart_item", "Replace an incorrect cart item with the requested product.", {"wrong_product_id": {"type": "string"}, "product_id": {"type": "string"}}, ["wrong_product_id", "product_id"]),
            function("get_calendar", "Return calendar events for a date.", {"date": {"type": "string"}}, ["date"]),
            function("create_event", "Create a calendar event.", {"title": {"type": "string"}, "date": {"type": "string"}, "time": {"type": "string"}, "location": {"type": "string"}}, ["title", "date", "time"]),
            function("delete_event", "Delete a matching calendar event.", {"title": {"type": "string"}, "date": {"type": "string"}, "time": {"type": "string"}}, ["title", "date", "time"]),
            function("replace_event", "Replace an incorrect event with the requested event.", {"old_title": {"type": "string"}, "old_date": {"type": "string"}, "old_time": {"type": "string"}, "title": {"type": "string"}, "date": {"type": "string"}, "time": {"type": "string"}, "location": {"type": "string"}}, ["old_title", "old_date", "old_time", "title", "date", "time"]),
            function("get_weather", "Return weather for a city and date.", {"city": {"type": "string"}, "date": {"type": "string"}}, ["city", "date"]),
            function("search_place", "Find places by city and category.", {"city": {"type": "string"}, "category": {"type": "string"}}, ["city", "category"]),
            function("get_route", "Return a route between two locations.", {"origin": {"type": "string"}, "destination": {"type": "string"}}, ["origin", "destination"]),
            function("inspect_state", "Inspect the current cart and calendar when recovering from an error.", {}),
        ]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "get_budget":
            return self._remember({"ok": True, "budget": self.state["budget"]})

        if tool_name == "search_product":
            category = arguments.get("category")
            products = [{"product_id": product_id, **product} for product_id, product in self.state["products"].items() if product["category"] == category]
            products.sort(key=lambda item: (-item["rating"], item["price"]))
            return self._remember({"ok": True, "products": products})

        if tool_name == "get_cart":
            return self._remember({"ok": True, "cart": list(self.state["cart"])})

        if tool_name == "add_to_cart":
            product_id = arguments.get("product_id")
            product = self.state["products"].get(product_id)
            if product is None:
                return self._remember({"ok": False, "error": "unknown_product", "product_id": product_id})
            self.state["cart"].append(product_id)
            return self._remember({"ok": True, "added": product_id, "cart": list(self.state["cart"])})

        if tool_name == "remove_from_cart":
            product_id = arguments.get("product_id")
            if product_id not in self.state["cart"]:
                return self._remember({"ok": False, "error": "not_in_cart", "product_id": product_id})
            self.state["cart"].remove(product_id)
            return self._remember({"ok": True, "removed": product_id, "cart": list(self.state["cart"])})

        if tool_name == "replace_cart_item":
            wrong_product_id = arguments.get("wrong_product_id")
            product_id = arguments.get("product_id")
            if product_id not in self.state["products"]:
                return self._remember({"ok": False, "error": "unknown_product", "product_id": product_id})
            if wrong_product_id not in self.state["cart"]:
                return self._remember({"ok": False, "error": "wrong_product_not_in_cart", "product_id": wrong_product_id})
            self.state["cart"] = [product_id if item == wrong_product_id else item for item in self.state["cart"]]
            return self._remember({"ok": True, "replaced": wrong_product_id, "added": product_id, "cart": list(self.state["cart"])})

        if tool_name == "get_calendar":
            date = arguments.get("date")
            events = [event for event in self.state["calendar"] if event["date"] == date]
            return self._remember({"ok": True, "date": date, "events": events})

        if tool_name == "create_event":
            event = {key: arguments[key] for key in ("title", "date", "time") if key in arguments}
            if arguments.get("location"):
                event["location"] = arguments["location"]
            self.state["calendar"].append(event)
            return self._remember({"ok": True, "created": event})

        if tool_name == "delete_event":
            target = {key: arguments[key] for key in ("title", "date", "time") if key in arguments}
            before = len(self.state["calendar"])
            self.state["calendar"] = [event for event in self.state["calendar"] if not all(event.get(key) == value for key, value in target.items())]
            removed = before - len(self.state["calendar"])
            return self._remember({"ok": removed > 0, "removed": removed})

        if tool_name == "replace_event":
            old_target = {key: arguments[key] for key in ("old_title", "old_date", "old_time") if key in arguments}
            before = len(self.state["calendar"])
            self.state["calendar"] = [
                event for event in self.state["calendar"]
                if not all(event.get(key.removeprefix("old_")) == value for key, value in old_target.items())
            ]
            removed = before - len(self.state["calendar"])
            if removed == 0:
                return self._remember({"ok": False, "error": "old_event_not_found"})
            event = {key: arguments[key] for key in ("title", "date", "time") if key in arguments}
            if arguments.get("location"):
                event["location"] = arguments["location"]
            self.state["calendar"].append(event)
            return self._remember({"ok": True, "replaced": removed, "created": event})

        if tool_name == "get_weather":
            key = f"{arguments.get('city')}|{arguments.get('date')}"
            weather = self.state["weather"].get(key)
            if weather is None:
                return self._remember({"ok": False, "error": "unknown_weather_key", "key": key})
            return self._remember({"ok": True, "city": arguments["city"], "date": arguments["date"], **weather})

        if tool_name == "search_place":
            key = f"{arguments.get('city')}|{arguments.get('category')}"
            places = sorted(self.state["places"].get(key, []), key=lambda item: -item["rating"])
            return self._remember({"ok": bool(places), "city": arguments["city"], "category": arguments["category"], "places": places, "top_place_id": places[0]["place_id"] if places else None})

        if tool_name == "get_route":
            key = f"{arguments.get('origin')}|{arguments.get('destination')}"
            route = self.state["routes"].get(key)
            if route is None:
                return self._remember({"ok": False, "error": "unknown_route", "key": key})
            return self._remember({"ok": True, "origin": arguments["origin"], "destination": arguments["destination"], **route})

        if tool_name == "inspect_state":
            return self._remember({"ok": True, "cart": list(self.state["cart"]), "calendar": copy.deepcopy(self.state["calendar"])})

        return self._remember({"ok": False, "error": "unknown_tool", "tool": tool_name})

    def inject_error(self, error_spec: dict[str, Any] | None) -> None:
        if not error_spec:
            return
        error_type = error_spec.get("type")
        if error_type == "wrong_cart_item":
            wrong_product_id = error_spec["wrong_product_id"]
            if wrong_product_id not in self.state["products"]:
                raise ValueError(f"unknown injected product: {wrong_product_id}")
            self.state["cart"] = [wrong_product_id]
            return
        if error_type == "wrong_event":
            self.state["calendar"] = [dict(error_spec["wrong_event"])]
            return
        if error_type == "stale_result":
            self.state["last_result"] = copy.deepcopy(error_spec["stale_result"])
            return
        raise ValueError(f"unsupported error injection: {error_type}")

    def prepare_condition(self, task: Task, condition: str) -> tuple[list[dict[str, Any]], str | None]:
        if condition not in {"free", "normal_restart", "recovery"}:
            raise ValueError(f"unsupported condition: {condition}")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": task.recovery_query if condition == "recovery" and task.recovery_query else task.query,
            },
        ]
        expected_next: str | None = None
        prefix_end = 0 if condition in {"free", "recovery"} else task.checkpoint_step
        for action in task.gold_actions[:prefix_end]:
            result = self.execute(action["tool"], action.get("arguments", {}))
            call_id = f"gold_{len(messages)}"
            messages.append({"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": action["tool"], "arguments": json.dumps(action.get("arguments", {}), ensure_ascii=False)}}]})
            messages.append({"role": "tool", "tool_call_id": call_id, "name": action["tool"], "content": json.dumps(result, ensure_ascii=False)})

        if condition == "recovery":
            self.inject_error(task.error_injection)
        if condition == "free":
            prefix_end = 0
        if prefix_end < task.horizon:
            expected_next = task.gold_actions[prefix_end]["tool"]
        return messages, expected_next

    def check_success(self, condition: dict[str, Any]) -> bool:
        if "cart_contains" in condition:
            return condition["cart_contains"] in self.state["cart"]
        if "cart_equals" in condition:
            return list(condition["cart_equals"]) == self.state["cart"]
        if "calendar_contains" in condition:
            required = condition["calendar_contains"]
            return any(all(event.get(key) == value for key, value in required.items()) for event in self.state["calendar"])
        if "last_result_field_equals" in condition:
            spec = condition["last_result_field_equals"]
            return (self.state["last_result"] or {}).get(spec["field"]) == spec["value"]
        raise ValueError(f"unsupported success condition: {condition}")


def environment_for_task(task: Task) -> MockEnvironment:
    """Keep legacy recovery fixtures separate from the redesigned execution API."""
    if task.schema_version in {"execution-v2", "recovery-v3"}:
        from .execution_environment import ExecutionEnvironment

        return ExecutionEnvironment(task.execution_spec or {})
    return MockEnvironment()

