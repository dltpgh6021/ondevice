"""Create Q8/Q4 GGUF variants from one FP16/BF16 GGUF source.

This script deliberately quantizes both variants from the same unquantized
source file. It also writes a manifest so that model provenance, file sizes,
hashes, and toolchain information are preserved for later comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path, *, label: str, quantization: str) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "filename": path.name,
        "quantization": quantization,
        "size_bytes": path.stat().st_size,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
        "sha256": sha256_file(path),
    }


def command_string(command: list[str]) -> str:
    if sys.platform.startswith("win"):
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def executable(value: str) -> str:
    resolved = shutil.which(value)
    if resolved:
        return resolved
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    raise FileNotFoundError(f"실행 파일을 찾을 수 없습니다: {value}")


def tool_version(llama_executable: str) -> str | None:
    try:
        result = subprocess.run(
            [llama_executable, "version"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or result.stderr.strip() or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="하나의 FP16/BF16 GGUF에서 Q8과 Q4 GGUF를 생성하고 manifest를 기록합니다."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="양자화하지 않은 FP16/BF16/F32 GGUF 파일 경로",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "models"),
        help="생성 모델과 manifest를 저장할 폴더",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="manifest 경로; 기본값은 output-dir/manifest.json",
    )
    parser.add_argument("--source-label", default="Qwen3-1.7B-FP16")
    parser.add_argument("--source-quantization", default="F16")
    parser.add_argument("--q8-type", default="Q8_0")
    parser.add_argument("--q4-type", default="Q4_K_M")
    parser.add_argument("--quantizer", default="llama-quantize")
    parser.add_argument("--llama", default="llama", help="llama version 조회에 사용할 실행 파일")
    parser.add_argument("--threads", type=int, default=0, help="quantizer thread 수; 0이면 기본값 사용")
    parser.add_argument("--reference-q8", default=None, help="비교용 공식 Q8 GGUF 경로(선택)")
    parser.add_argument("--reference-q4", default=None, help="비교용 공식 Q4 GGUF 경로(선택)")
    parser.add_argument("--force", action="store_true", help="이미 존재하는 출력 파일을 덮어쓰기")
    parser.add_argument("--dry-run", action="store_true", help="명령만 출력하고 실제 변환하지 않음")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise SystemExit(f"source GGUF를 찾을 수 없습니다: {source}")
    if source.suffix.lower() != ".gguf":
        raise SystemExit("--source는 GGUF 파일이어야 합니다. Hugging Face safetensors는 먼저 GGUF로 변환하세요.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else output_dir / "manifest.json"
    quantizer = executable(args.quantizer)
    llama = executable(args.llama)

    stem = source.stem
    q8_path = output_dir / f"{stem}-{args.q8_type}.gguf"
    q4_path = output_dir / f"{stem}-{args.q4_type}.gguf"
    outputs = [(q8_path, args.q8_type), (q4_path, args.q4_type)]
    if not args.force:
        existing = [str(path) for path, _ in outputs if path.exists()]
        if existing:
            raise SystemExit(
                "출력 파일이 이미 존재합니다. 덮어쓰려면 --force를 사용하세요:\n" + "\n".join(existing)
            )

    commands: list[dict[str, Any]] = []
    for output, quantization in outputs:
        command = [quantizer, str(source), str(output), quantization]
        if args.threads > 0:
            command.append(str(args.threads))
        commands.append({"quantization": quantization, "output": str(output), "argv": command, "command": command_string(command)})

    print(f"source: {source}")
    for item in commands:
        print(item["command"])
    if args.dry_run:
        print("dry-run: 실제 변환과 manifest 저장을 생략했습니다.")
        return 0

    for item in commands:
        subprocess.run(item["argv"], check=True)

    variants = [file_info(source, label=args.source_label, quantization=args.source_quantization)]
    for output, quantization in outputs:
        variants.append(file_info(output, label=f"{args.source_label}-{quantization}", quantization=quantization))

    reference_variants: list[dict[str, Any]] = []
    for reference, label, quantization in (
        (args.reference_q8, "official-Q8", args.q8_type),
        (args.reference_q4, "official-Q4", args.q4_type),
    ):
        if not reference:
            continue
        reference_path = Path(reference).expanduser().resolve()
        if not reference_path.is_file():
            raise SystemExit(f"비교용 공식 모델을 찾을 수 없습니다: {reference_path}")
        reference_variants.append(file_info(reference_path, label=label, quantization=quantization))

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python_version": sys.version,
        },
        "source_model": variants[0],
        "variants": variants,
        "reference_variants": reference_variants,
        "toolchain": {
            "llama_executable": llama,
            "llama_version": tool_version(llama),
            "quantizer_executable": quantizer,
            "q8_type": args.q8_type,
            "q4_type": args.q4_type,
            "threads": args.threads if args.threads > 0 else "default",
            "commands": commands,
        },
        "experiment_note": "Q8과 Q4는 동일한 unquantized source GGUF에서 각각 독립적으로 생성했습니다.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
