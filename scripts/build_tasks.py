"""Regenerate both English task families together, after offline validation."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_execution_tasks import build_execution_tasks, catalog_markdown as execution_catalog
from scripts.build_recovery_tasks import build_recovery_tasks, catalog_markdown as recovery_catalog
from src.dataset_validation import validate_dataset_records


def _write_generated(path: Path, content: str) -> None:
    """Atomically replace one explicitly selected generated artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                     prefix="generated-task-", suffix=".tmp", delete=False) as handle:
        handle.write(content.rstrip("\r\n") + "\n")
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Validate and compare the checked-in dataset")
    mode.add_argument("--write", action="store_true", help="Regenerate tasks.json and the two generated catalogs")
    parser.add_argument("--tasks", default=str(ROOT / "tasks" / "tasks.json"))
    args = parser.parse_args()
    execution = build_execution_tasks()
    expected = execution + build_recovery_tasks(execution)
    _, report = validate_dataset_records(expected)
    source = Path(args.tasks).resolve()
    if args.check:
        from src.dataset_validation import load_validated_tasks

        _, actual_report = load_validated_tasks(source)
        if actual_report["dataset_sha256"] != report["dataset_sha256"]:
            raise SystemExit("Dataset ordering differs from the canonical generated output.")
        catalogs = {"EXECUTION_TASKS.md": execution_catalog(), "RECOVERY_TASKS.md": recovery_catalog(execution)}
        for name, content in catalogs.items():
            path = source.parent / name
            if not path.exists() or path.read_text(encoding="utf-8") != content.rstrip("\r\n") + "\n":
                raise SystemExit(f"{name} differs from the generated English catalog.")
        print("All 120 English tasks match the validated scenario definitions.")
        return 0
    _write_generated(source, json.dumps(expected, ensure_ascii=False, indent=2) + "\n")
    _write_generated(source.parent / "EXECUTION_TASKS.md", execution_catalog() + "\n")
    _write_generated(source.parent / "RECOVERY_TASKS.md", recovery_catalog(execution) + "\n")
    print(json.dumps({"written": str(source), "tasks": 120, "language": report["language"],
                      "dataset_version": report["dataset_version"],
                      "dataset_sha256": report["dataset_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
