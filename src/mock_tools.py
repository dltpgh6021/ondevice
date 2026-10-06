from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


class ToolError(Exception):
    """A tool-level error that should be visible in the trajectory."""


@dataclass
class MockToolEnvironment:
    state: dict[str, Any]

    @classmethod
    def from_task(cls, task: dict[str, Any]) -> "MockToolEnvironment":
        return cls(state=deepcopy(task["initial_state"]))

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self.state)

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "lookup_customer": self.lookup_customer,
            "get_order": self.get_order,
            "approve_refund": self.approve_refund,
            "refund_order": self.refund_order,
        }
        if name not in handlers:
            raise ToolError(f"unknown_tool:{name}")
        return handlers[name](arguments)

    def lookup_customer(self, arguments: dict[str, Any]) -> dict[str, Any]:
        email = arguments.get("email")
        customer = self.state["customers"].get(email)
        if customer is None:
            raise ToolError("customer_not_found")
        return deepcopy(customer)

    def get_order(self, arguments: dict[str, Any]) -> dict[str, Any]:
        order_id = arguments.get("order_id")
        order = self.state["orders"].get(order_id)
        if order is None:
            raise ToolError("order_not_found")
        return deepcopy(order)

    def approve_refund(self, arguments: dict[str, Any]) -> dict[str, Any]:
        order_id = arguments.get("order_id")
        order = self.state["orders"].get(order_id)
        if order is None:
            raise ToolError("order_not_found")
        if not order["eligible_for_refund"]:
            raise ToolError("order_not_eligible")
        order["refund_approved"] = True
        return {"order_id": order_id, "refund_approved": True}

    def refund_order(self, arguments: dict[str, Any]) -> dict[str, Any]:
        order_id = arguments.get("order_id")
        order = self.state["orders"].get(order_id)
        if order is None:
            raise ToolError("order_not_found")
        if not order["refund_approved"]:
            raise ToolError("refund_not_approved")
        if not order["refunded"]:
            order["refunded"] = True
            self.state["refunded_order_ids"].append(order_id)
        return {"order_id": order_id, "refunded": True}

    def is_success(self, expected_final: dict[str, Any]) -> bool:
        for key, expected in expected_final.items():
            if self.state.get(key) != expected:
                return False
        return True

