from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen target pipeline and validate the official result."
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/target/v2"))
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument(
        "--geometry-config",
        type=Path,
        default=ROOT / "scripts" / "geometry_config.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_root = args.reference_root.resolve()
    target_root = args.target_root.resolve()
    output_root = args.output_root.resolve()

    if reference_root == target_root:
        raise SystemExit("reference-root and target-root must be different")
    if not reference_root.is_dir():
        raise SystemExit(f"reference dataset not found: {reference_root}")
    if not target_root.is_dir():
        raise SystemExit(f"target dataset not found: {target_root}")
    if not args.geometry_config.is_file():
        raise SystemExit(f"geometry config not found: {args.geometry_config}")

    pipeline_command = [
        sys.executable,
        str(ROOT / "run_v2_pipeline.py"),
        "--reference-root",
        str(reference_root),
        "--target-root",
        str(target_root),
        "--output-root",
        str(output_root),
        "--geometry-config",
        str(args.geometry_config.resolve()),
    ]
    readiness_command = [
        sys.executable,
        str(ROOT / "scripts" / "check_submission_readiness.py"),
        "--output-root",
        str(output_root),
        "--expected-episodes",
        str(args.expected_episodes),
    ]
    preflight_command = [
        sys.executable,
        str(ROOT / "scripts" / "preflight_submission.py"),
        "--reference-root",
        str(reference_root),
        "--target-root",
        str(target_root),
        "--output-root",
        str(output_root),
        "--expected-episodes",
        str(args.expected_episodes),
    ]
    export_metrics_command = [
        sys.executable,
        str(ROOT / "scripts" / "export_ppt_metrics.py"),
        "--output-root",
        str(output_root),
    ]

    print("[1/4] Checking environment and dataset structure", flush=True)
    subprocess.run(preflight_command, cwd=ROOT, check=True)
    print("[2/4] Running frozen target pipeline", flush=True)
    subprocess.run(pipeline_command, cwd=ROOT, check=True)
    print("[3/4] Checking submission readiness", flush=True)
    subprocess.run(readiness_command, cwd=ROOT, check=True)
    print("[4/4] Exporting PPT metrics", flush=True)
    subprocess.run(export_metrics_command, cwd=ROOT, check=True)
    print("FINAL TARGET RUN: PASS", flush=True)


if __name__ == "__main__":
    main()
