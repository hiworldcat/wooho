from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "diagnostics/findings_v2.json",
    "diagnostics/geometry_constraints_v2.json",
    "diagnostics/problem_standards_v2.json",
    "diagnostics/reference_baselines_v2.json",
    "diagnostics/run_manifest_v2.json",
    "reports/dataset_quality_report_v2.json",
    "reports/dataset_quality_report_v2.md",
    "reports/dataset_quality_report_v2_zh.md",
    "reports/episode_quality_report_v2_detail.md",
    "reports/episode_scores_v2.csv",
)
EXPECTED_SCORING_VERSION = "official_70_20_10_v1"
REFERENCE_MARKERS = ("参考集", "鍙傝€冮泦")
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a frozen V2 target result before numbers are copied into the PPT."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/target/v2"),
        help="V2 target output directory.",
    )
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument(
        "--allow-reference-only",
        action="store_true",
        help=(
            "Accept an official reference-only run when the competition provides no "
            "separate target set. The run must still be marked as negative_control."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    for relative_path in REQUIRED_FILES:
        if not (output_root / relative_path).is_file():
            errors.append(f"missing required file: {relative_path}")

    if errors:
        print("SUBMISSION READINESS: FAIL")
        for message in errors:
            print(f"ERROR: {message}")
        raise SystemExit(1)

    manifest = load_json(output_root / "diagnostics/run_manifest_v2.json")
    report = load_json(output_root / "reports/dataset_quality_report_v2.json")
    findings = load_json(output_root / "diagnostics/findings_v2.json")

    with (output_root / "reports/episode_scores_v2.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        score_rows = list(csv.DictReader(handle))

    reference_only_run = (
        args.allow_reference_only
        and manifest.get("dataset_role") == "negative_control"
        and manifest.get("same_dataset") is True
    )
    if reference_only_run:
        warnings.append(
            "accepted official reference-only run; do not describe it as an independent target set"
        )
    else:
        if manifest.get("dataset_role") != "target":
            errors.append(f"dataset_role must be target, got {manifest.get('dataset_role')!r}")
        if manifest.get("same_dataset") is not False:
            errors.append("reference and target datasets were not kept separate")
    if manifest.get("scoring_version") != EXPECTED_SCORING_VERSION:
        errors.append("run manifest scoring version is not the frozen 70/20/10 version")

    run = report.get("run", {})
    if run != manifest:
        errors.append("report run metadata does not match run_manifest_v2.json")
    if report.get("scoring", {}).get("version") != EXPECTED_SCORING_VERSION:
        errors.append("report scoring version is not the frozen 70/20/10 version")

    dataset = report.get("dataset", {})
    episode_count = int(dataset.get("episodes", -1))
    if episode_count != args.expected_episodes:
        errors.append(
            f"expected {args.expected_episodes} episodes, report contains {episode_count}"
        )
    if len(score_rows) != episode_count:
        errors.append(
            f"episode score row count {len(score_rows)} does not match report count {episode_count}"
        )
    if len(report.get("episodes", [])) != episode_count:
        errors.append("JSON episode detail count does not match report count")

    summary = report.get("summary", {})
    if int(summary.get("finding_count", -1)) != len(findings):
        errors.append("finding_count does not match findings_v2.json")
    issue_counts = Counter(str(item.get("issue_type")) for item in findings)
    if dict(issue_counts) != summary.get("by_issue", {}):
        errors.append("by_issue summary does not match findings_v2.json")

    score = dataset.get("dataset_quality_score")
    if not isinstance(score, (int, float)) or not 0 <= float(score) <= 100:
        errors.append(f"dataset_quality_score is invalid: {score!r}")

    for row_number, row in enumerate(score_rows, start=2):
        file_value = str(row.get("file", ""))
        if not file_value:
            errors.append(f"CSV row {row_number} has an empty file path")
        if WINDOWS_ABSOLUTE_PATH.match(file_value) or file_value.startswith("/"):
            errors.append(f"CSV row {row_number} contains an absolute path: {file_value}")
        if any(marker in file_value for marker in REFERENCE_MARKERS) and not reference_only_run:
            errors.append(f"CSV row {row_number} points to the reference dataset: {file_value}")

    geometry_counts = summary.get("geometry_status_counts", {})
    if int(geometry_counts.get("fail", 0)) > 0:
        errors.append("one or more episodes have failed geometry modules")
    if int(geometry_counts.get("unavailable", 0)) > 0:
        warnings.append("one or more episodes have unavailable geometry modules")

    phase_counts = summary.get("phase_status_counts", {})
    if int(phase_counts.get("fail", 0)) > 0:
        errors.append("one or more episodes have failed phase detection")
    if int(phase_counts.get("unavailable", 0)) > 0:
        warnings.append("one or more episodes have unavailable phase detection")

    status = "PASS" if not errors else "FAIL"
    print(f"SUBMISSION READINESS: {status}")
    print(f"output_root: {output_root}")
    print(f"episodes: {episode_count}")
    print(f"findings: {len(findings)}")
    print(f"dataset_quality_score: {score}")
    for message in warnings:
        print(f"WARNING: {message}")
    for message in errors:
        print(f"ERROR: {message}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
