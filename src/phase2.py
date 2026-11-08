"""Paired purchasing decisions for phase 2. Keys and grading stay outside model messages."""
import copy
import hashlib
import json
import random

from .tasks import Task, stable_hash

HORIZONS = (1, 3, 5)
CONDITIONS = ("free", "normal_restart", "error_restart")
SIZES = {"dev": 12, "main": 60}
PROTOCOL = "dependent-purchases-v4"

# (number affordable, quality ranks in ascending price order). Each six-case
# block contains every profile, independently shuffled for each order line.
PROFILES = (
    (2, (2, 1, 0)), (3, (0, 2, 1)), (3, (0, 1, 2)),
    (2, (1, 0, 2)), (2, (0, 1, 2)), (1, (0, 1, 2)),
)
TOOLS = [{"type": "function", "function": {
    "name": "save_offer", "description": (
        "Select exactly one offer for a line. Saving again on the same line replaces its previous selection; "
        "it does not add another item. Changing an earlier bundle line clears dependent later lines. "
        "An ok result means the update was applied, not that the selection satisfies the user's rule."),
    "parameters": {"type": "object", "properties": {"line": {"type": "integer"}, "offer": {"type": "string"}},
                   "required": ["line", "offer"], "additionalProperties": False}}}]


TOOLS.append({"type":"function","function":{"name":"inspect_order","description":(
    "Read required line numbers, unsaved line numbers, saved selections and available offers. "
    "An empty unsaved_lines list means every line has a selection, not that those selections satisfy the user's rule."),
              "parameters":{"type":"object","properties":{},"additionalProperties":False}}})


def trial_design(split):
    """Base order for primary outcomes, plus paired h1 presentation orders."""
    for i in range(SIZES[split]):
        audit = split == "dev" or i % 5 == 0
        for h in HORIZONS:
            for condition in CONDITIONS:
                rotations = [(i + k) % 3 for k in range(3)] if h == 1 and audit else [0]
                for rotation in rotations:
                    yield i, h, condition, rotation


class PurchaseChain:
    def __init__(self, split, scenario, horizon, condition, rotation=0):
        if rotation not in (0, 1, 2) or (horizon != 1 and rotation):
            raise ValueError("Only h1 supports the order audit")
        self.rotation = rotation
        self.audit_line = 1 if condition == "free" else 0
        self.base = f"{split}-{scenario:03d}"
        self.scenario = scenario
        self.h = horizon
        self.condition = condition
        self.start = 1 if condition == "free" else 0
        self.saved = {}
        bundle_lines = list(range(1, horizon + 1))
        self.query = (
            "Complete the order with exactly one selected offer per required line. "
            "On each line choose the highest-rated offer whose price is at or below that line's budget. "
            f"The required bundle line numbers are {bundle_lines}. Complete them in order, including line {horizon}. " +
            ("Line 0 is an additional accessory, not one of the bundle lines. It must also satisfy the same selection rule. "
             if self.start == 0 else "") +
            "Review saved selections using their prices, ratings and budgets. Keep selections that already satisfy the rule; "
            "replace a selection if it does not. Saving another offer on the same line replaces the previous one. "
            "Changing a bundle selection clears dependent later selections, which must then be made again. "
            "The order state lists unsaved_lines; these lines still need a selection. "
            "When every required line has a selection satisfying the rule, stop calling tools and finish with a short final answer.")

    def menu(self, line, *, canonical=False):
        parent = self.saved.get(line - 1, "") if line > 1 else ""
        key = f"{self.base}/{line}/{parent}"
        rng = random.Random(key)
        split = self.base.split("-")[0]
        profiles = list(PROFILES)
        random.Random(f"{split}/{self.scenario // len(profiles)}/{line}/profiles").shuffle(profiles)
        affordable, ranks = profiles[self.scenario % len(profiles)]
        if line == 0:
            # The paired checkpoint always supports a silent over-budget edit.
            affordable = min(affordable, 2)
        prices = sorted(rng.sample(range(30, 160), 3))
        ratings = sorted(rng.sample(range(30, 51), 3))
        upper = prices[affordable] if affordable < 3 else prices[-1] + 12
        budget = rng.randrange(prices[affordable - 1], upper)
        winner = max(range(affordable), key=lambda i: ranks[i])
        offset = random.Random(f"{split}/{self.scenario // 3}/{line}").randrange(3)
        slot = (offset + self.scenario) % 3
        order = [i for i in range(3) if i != winner]
        rng.shuffle(order)
        order.insert(slot, winner)
        offers = []
        for position, i in enumerate(order):
            ident = hashlib.sha256(f"{key}/{position}".encode()).hexdigest()[:10]
            offers.append({"offer": ident, "price": prices[i], "rating": ratings[ranks[i]] / 10})
        if line == self.audit_line and not canonical:
            offers = offers[self.rotation:] + offers[:self.rotation]
        return {"line": line, "budget": budget, "offers": offers}

    def best(self, line):
        menu = self.menu(line)
        return max((o for o in menu["offers"] if o["price"] <= menu["budget"]), key=lambda o:o["rating"])["offer"]

    def view(self):
        required = list(range(self.start, self.h + 1))
        lines = []
        for line in required:
            if line > 1 and line - 1 not in self.saved:
                continue
            menu = self.menu(line)
            # Expose only existing menu facts. Do not label a selection as correct
            # or affordable, sort the candidates, or include the hidden answer.
            menu["saved_offer"] = next(
                (dict(offer) for offer in menu["offers"] if offer["offer"] == self.saved.get(line)), None)
            lines.append(menu)
        return {"required_lines": required,
                "unsaved_lines": [line for line in required if line not in self.saved],
                "saved": dict(self.saved), "lines": lines}

    def snapshot(self):
        return copy.deepcopy(self.saved)

    def state_hash(self):
        return stable_hash(self.saved)

    def check_success(self, ignored=None):
        return all(self.saved.get(i) == self.best(i) for i in range(self.start, self.h+1))

    def tool_definitions(self):
        return TOOLS

    def execute(self, name, args):
        if name == "inspect_order" and not args:
            return {"ok": True, **self.view()}
        line, offer = args.get("line"), args.get("offer")
        if (name != "save_offer" or set(args) != {"line","offer"} or type(line) is not int
                or not self.start <= line <= self.h or (line > 1 and line-1 not in self.saved)):
            return {"ok": False, "error": "Unknown tool, arguments or unavailable line."}
        if offer not in [o["offer"] for o in self.menu(line)["offers"]]:
            return {"ok": False, "error": "Offer unavailable on this line."}
        if line > 0 and self.saved.get(line) != offer:
            self.saved = {i:v for i,v in self.saved.items() if i <= line}
        self.saved[line] = offer
        return {"ok": True, "updated_line": line, **self.view()}

    def assess_action_result(self, result):
        if "updated_line" not in result:
            return None
        line = result["updated_line"]
        return "wrong_offer" if self.saved[line] != self.best(line) else None

    def prepare_condition(self, task, condition):
        self.saved = {}
        messages = [{"role":"system","content":"Use one tool call per response or a final answer. Use English."},
                    {"role":"user","content":self.query + "\nCurrent order: " + json.dumps(self.view())}]
        if condition != "free":
            menu = self.menu(0, canonical=True)
            offer = (self.best(0) if condition == "normal_restart" else
                     next(o["offer"] for o in menu["offers"] if o["price"] > menu["budget"]))
            args = {"line":0, "offer":offer}
            result = self.execute("save_offer", args)
            messages += [{"role":"assistant","tool_calls":[{"id":"previous","type":"function","function":{
                "name":"save_offer","arguments":json.dumps(args)}}]},
                {"role":"tool","tool_call_id":"previous","name":"save_offer","content":json.dumps(result)}]
        return messages, None

    def task(self):
        return Task(task_id=f"{self.base}-h{self.h}",base_id=self.base,domain="shopping",
                    task_type="execution" if self.start else "recovery",horizon=self.h,query=self.query,
                    recovery_query=None,error_description=None,gold_actions=[],checkpoint_step=0,
                    error_injection=None,success_condition={},schema_version="phase2",language="en",
                    dataset_version=PROTOCOL,normal_remaining_calls=self.h,
                    recovery_min_calls=self.h+1 if self.condition=="error_restart" else None)
