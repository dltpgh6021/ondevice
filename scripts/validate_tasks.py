from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_validation import DatasetValidationError, validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate all 120 English tasks offline (no model/server required)")
    parser.add_argument("--tasks", default=str(ROOT / "tasks" / "tasks.json"))
    parser.add_argument("--output", default=None, help="Optional JSON validation report")
    args = parser.parse_args()
    if args.output and Path(args.output).resolve() == Path(args.tasks).resolve():
        parser.error("--output must not overwrite the input dataset")
    try:
        report = validate_dataset(args.tasks)
    except DatasetValidationError as exc:
        report = exc.report
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
