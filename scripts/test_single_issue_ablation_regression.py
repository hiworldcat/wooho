from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import geometry_constraints
from generate_single_issue_ablation_data import DEFAULT_OUTPUT_ROOT, ISSUES, generate_cases
from v2_quality_pipeline import (
    Finding,
    FindingFactory,
    collect_reference_baselines,
    image_views_from_info,
    inspect_episode,
    load_dataset,
    merge_findings,
    score_episode,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "outputs" / "v2" / "diagnostics" / "single_issue_ablation_regression.json"
HARD_FAMILIES = {"vision_illegal", "state_illegal", "temporal_illegal"}
FAMILY_ALIASES = {
    "vision_single": {"vision_single", "vision_illegal"},
    "vision_illegal": {"vision_illegal", "vision_single"},
    "state_temporal": {"state_temporal", "vision_state_temporal"},
    "vision_state_temporal": {"vision_state_temporal", "state_temporal", "vision_temporal"},
}


def finding_family(finding: Finding) -> str:
    if finding.category_id == "1.1.1.A":
        return "vision_illegal"
    if finding.category_id == "1.1.1.B":
        return "state_illegal"
    if finding.category_id == "1.1.2.A":
        return "vision_single"
    if finding.category_id == "1.1.2.B":
        return "vision_vision"
    if finding.category_id == "1.1.2.C":
        return "vision_state_vision"
    if finding.category_id == "1.1.2.D":
        return "state_vision_state"
    if finding.category_id == "1.2.1":
        return "temporal_illegal"
    if finding.category_id == "1.2.2.A":
        return "vision_temporal"
    if finding.category_id == "1.2.2.B":
        return "state_temporal"
    if finding.category_id == "1.2.2.C":
        return "vision_state_temporal"
    return "unknown"


def expected_families(expected: str) -> set[str]:
    return FAMILY_ALIASES.get(expected, {expected})


def span_iou(a: tuple[int | None, int | None], b: tuple[int | None, int | None]) -> float:
    a0, a1 = a
    b0, b1 = b
    if a0 is None or a1 is None or b0 is None or b1 is None:
        return 1.0 if a0 is None and b0 is None else 0.0
    left = max(int(a0), int(b0))
    right = min(int(a1), int(b1))
    intersection = max(0, right - left + 1)
    union = max(int(a1), int(b1)) - min(int(a0), int(b0)) + 1
    return float(intersection / union) if union > 0 else 0.0


def load_manifest(base: Path, regenerate: bool) -> dict[str, Any]:
    manifest_path = base / "manifest.json"
    if regenerate or not manifest_path.exists():
        return generate_cases(base, [issue.issue_id for issue in ISSUES], clean=regenerate, seed=20260825)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_reference() -> tuple[dict[str, Any], dict[int, dict[str, Any]], list[str], dict[str, Any]]:
    info, _tasks, episodes, parquet_files = load_dataset(ROOT)
    episode_meta = {int(row["episode_index"]): row for row in episodes}
    views = image_views_from_info(info)
    baselines = collect_reference_baselines(info, episode_meta, parquet_files, views)
    geometry_config = geometry_constraints.default_geometry_config(info)
    config_path = ROOT / "scripts" / "geometry_config.json"
    if config_path.exists():
        geometry_config = geometry_constraints.load_geometry_config(config_path, geometry_config)
    baselines["geometry_config"] = geometry_config
    baselines["geometry_constraints"] = geometry_constraints.fit_geometry_reference(parquet_files, geometry_config)
    expected_low_dim_shape: dict[str, int] = {}
    for column in ["state", "actions"]:
        spec = info.get("features", {}).get(column)
        if spec and spec.get("shape"):
            expected_low_dim_shape[column] = int(spec["shape"][0])
    baselines["expected_low_dim_shape"] = expected_low_dim_shape
    return info, episode_meta, views, baselines


def evaluate_case(case: dict[str, Any], info: dict[str, Any], episode_meta: dict[int, dict[str, Any]], views: list[str], baselines: dict[str, Any]) -> dict[str, Any]:
    case_path = Path(str(case["case_path"]).replace("\\", "/"))
    path = case_path if case_path.is_absolute() else ROOT / case_path
    expected = str(case["expected_detector_family"])
    allowed = expected_families(expected)
    evidence = dict(case.get("evidence", {}))
    expected_span = (evidence.get("frame_start"), evidence.get("frame_end"))
    source_episode = int(case.get("source_episode_index", 0))
    factory = FindingFactory()
    meta, findings = inspect_episode(path, info, episode_meta.get(source_episode), views, baselines, factory)
    if meta.get("episode_index") is None:
        meta["episode_index"] = source_episode
    merged = merge_findings(findings)
    score = score_episode(meta, merged)
    family_counts = Counter(finding_family(item) for item in merged)
    expected_hits = [item for item in merged if finding_family(item) in allowed]
    ious = [span_iou((item.frame_start, item.frame_end), expected_span) for item in expected_hits]
    best_iou = max(ious, default=0.0)
    detected = bool(expected_hits)
    localized = detected and (expected in HARD_FAMILIES or expected_span[0] is None or best_iou >= 0.05)
    return {
        "issue_id": case["issue_id"],
        "expected_detector_family": expected,
        "allowed_detector_families": sorted(allowed),
        "detected": detected,
        "localized": localized,
        "best_interval_iou": round(best_iou, 4),
        "expected_span": list(expected_span),
        "finding_count": len(merged),
        "family_counts": dict(sorted(family_counts.items())),
        "official_score_total": score.score_total,
        "legacy_score_total": score.legacy_score_total,
        "phase_status": score.phase_status,
        "case_path_resolved": str(path),
        "case_path_exists": path.exists(),
        "top_findings": [
            {
                "family": finding_family(item),
                "issue_type": item.issue_type,
                "confidence": item.confidence_level,
                "frame_start": item.frame_start,
                "frame_end": item.frame_end,
                "severity": item.severity_score,
                "evidence": item.evidence,
            }
            for item in sorted(merged, key=lambda item: (-item.severity_score, item.finding_id))[:8]
        ],
    }


def evaluate_negative_control(info: dict[str, Any], episode_meta: dict[int, dict[str, Any]], views: list[str], baselines: dict[str, Any]) -> dict[str, Any]:
    _info, _tasks, _episodes, parquet_files = load_dataset(ROOT)
    path = parquet_files[0]
    episode = int(path.stem.split("_")[-1])
    factory = FindingFactory()
    meta, findings = inspect_episode(path, info, episode_meta.get(episode), views, baselines, factory)
    merged = merge_findings(findings)
    family_counts = Counter(finding_family(item) for item in merged)
    return {
        "path": str(path.relative_to(ROOT)),
        "finding_count": len(merged),
        "family_counts": dict(sorted(family_counts.items())),
        "max_severity": max((item.severity_score for item in merged), default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 23 single-issue ablation regression checks.")
    parser.add_argument("--case-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--min-recall", type=float, default=0.55)
    parser.add_argument("--min-f1", type=float, default=0.50)
    parser.add_argument("--min-mean-iou", type=float, default=0.50)
    args = parser.parse_args()

    manifest = load_manifest(args.case_root, args.regenerate)
    info, episode_meta, views, baselines = build_reference()
    rows = [evaluate_case(case, info, episode_meta, views, baselines) for case in manifest["cases"]]
    negative_control = evaluate_negative_control(info, episode_meta, views, baselines)

    tp = sum(1 for row in rows if row["localized"])
    fn = len(rows) - tp
    fp = 0
    for row in rows:
        expected = set(row["allowed_detector_families"])
        for family, count in row["family_counts"].items():
            if family not in expected and family != "unknown":
                fp += int(count)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    mean_iou = sum(float(row["best_interval_iou"]) for row in rows if row["detected"]) / max(1, sum(1 for row in rows if row["detected"]))

    result = {
        "case_count": len(rows),
        "expected_case_count": len(ISSUES),
        "negative_control": negative_control,
        "metrics": {
            "true_positive_cases": tp,
            "false_negative_cases": fn,
            "auxiliary_false_positive_findings": fp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_interval_iou_on_detected": round(mean_iou, 4),
        },
        "thresholds": {
            "min_recall": args.min_recall,
            "min_f1": args.min_f1,
            "min_mean_iou": args.min_mean_iou,
        },
        "cases": rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"wrote: {OUTPUT_PATH}")

    assert len(rows) == len(ISSUES), f"expected {len(ISSUES)} cases, got {len(rows)}"
    assert recall >= args.min_recall, f"recall {recall:.3f} below {args.min_recall:.3f}"
    assert f1 >= args.min_f1, f"f1 {f1:.3f} below {args.min_f1:.3f}"
    assert mean_iou >= args.min_mean_iou, f"mean IoU {mean_iou:.3f} below {args.min_mean_iou:.3f}"


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"REGRESSION_ASSERTION_FAILED: {exc}", file=sys.stderr)
        raise
