from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tasks import load_tasks, validate_task_set


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the project task set")
    parser.add_argument("--tasks", default=str(ROOT / "tasks" / "tasks.json"))
    parser.add_argument("--expected-total", type=int, default=120)
    parser.add_argument("--expected-per-type-horizon", type=int, default=20)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    summary = validate_task_set(
        tasks,
        expected_total=args.expected_total,
        expected_counts={"execution": 60, "recovery": 60},
        required_horizons={1, 3, 5},
        expected_per_type_horizon=args.expected_per_type_horizon,
        require_pairs=True,
    )
    print(json.dumps({"valid": True, **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
