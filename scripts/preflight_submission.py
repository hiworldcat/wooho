from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path


REQUIRED_MODULES = ("numpy", "pandas", "pyarrow", "PIL", "cv2")


def available_memory_bytes() -> int | None:
    if os.name == "posix":
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    return None


def dataset_summary(root: Path) -> dict[str, object]:
    meta = root / "meta"
    parquet_files = sorted(root.glob("data/**/*.parquet"))
    if not parquet_files:
        parquet_files = sorted(root.glob("**/*.parquet"))
    total_bytes = sum(path.stat().st_size for path in parquet_files)
    largest_bytes = max((path.stat().st_size for path in parquet_files), default=0)
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "info_exists": (meta / "info.json").is_file(),
        "tasks_exists": (meta / "tasks.jsonl").is_file(),
        "episodes_exists": (meta / "episodes.jsonl").is_file(),
        "parquet_count": len(parquet_files),
        "parquet_total_bytes": total_bytes,
        "largest_parquet_bytes": largest_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check dependencies, dataset structure, memory and disk without loading frames."
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/target/v2"))
    parser.add_argument("--expected-episodes", type=int, default=20)
    return parser.parse_args()


def gib(value: int | None) -> float | None:
    return None if value is None else round(value / (1024**3), 2)


def main() -> None:
    args = parse_args()
    reference_root = args.reference_root.resolve()
    target_root = args.target_root.resolve()
    output_root = args.output_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if reference_root == target_root:
        errors.append("reference-root and target-root must be different")

    missing_modules = [
        module for module in REQUIRED_MODULES if importlib.util.find_spec(module) is None
    ]
    if missing_modules:
        errors.append("missing Python modules: " + ", ".join(missing_modules))

    reference = dataset_summary(reference_root)
    target = dataset_summary(target_root)
    for label, summary in (("reference", reference), ("target", target)):
        if not summary["exists"]:
            errors.append(f"{label} dataset directory not found")
            continue
        for field in ("info_exists", "tasks_exists", "episodes_exists"):
            if not summary[field]:
                errors.append(f"{label} dataset missing metadata: {field}")
        if int(summary["parquet_count"]) == 0:
            errors.append(f"{label} dataset contains no parquet episodes")

    if int(target["parquet_count"]) != args.expected_episodes:
        errors.append(
            f"target parquet count must be {args.expected_episodes}, got {target['parquet_count']}"
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    free_disk = shutil.disk_usage(output_root.parent).free
    input_bytes = int(reference["parquet_total_bytes"]) + int(target["parquet_total_bytes"])
    recommended_disk = max(2 * input_bytes, 2 * 1024**3)
    if free_disk < recommended_disk:
        errors.append(
            f"free disk {gib(free_disk)} GiB is below recommended {gib(recommended_disk)} GiB"
        )

    available_memory = available_memory_bytes()
    largest_episode = max(
        int(reference["largest_parquet_bytes"]), int(target["largest_parquet_bytes"])
    )
    recommended_memory = max(6 * largest_episode, 2 * 1024**3)
    if available_memory is not None and available_memory < recommended_memory:
        warnings.append(
            f"available memory {gib(available_memory)} GiB is below conservative estimate "
            f"{gib(recommended_memory)} GiB; close other applications before running"
        )

    payload = {
        "status": "FAIL" if errors else "PASS",
        "reference": reference,
        "target": target,
        "resources": {
            "free_disk_gib": gib(free_disk),
            "recommended_disk_gib": gib(recommended_disk),
            "available_memory_gib": gib(available_memory),
            "recommended_memory_gib": gib(recommended_memory),
        },
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
