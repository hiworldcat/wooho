from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from v2_quality_pipeline import (
    FindingFactory,
    collect_reference_baselines,
    geometry_constraints,
    image_views_from_info,
    inspect_episode,
    load_dataset,
    merge_findings,
    score_episode,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "v2" / "reports" / "batch_benchmark.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "v2" / "reports" / "batch_benchmark_summary.json"


def load_case_paths(root: Path) -> list[Path]:
    if (root / "manifest.json").exists():
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return [ROOT / str(case["case_path"]).replace("\\", "/") for case in manifest["cases"]]
    return sorted(root.rglob("*.parquet"))


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V2 batch benchmark with failure isolation.")
    parser.add_argument("--input-root", type=Path, default=ROOT / "outputs" / "ablations" / "single_issue_cases")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    info, episode_meta, views, baselines = build_reference()
    case_paths = [p for p in load_case_paths(args.input_root) if p.is_file()]
    case_paths = case_paths[: args.limit]
    factory = FindingFactory()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    start_total = time.perf_counter()
    for idx, path in enumerate(case_paths, start=1):
        case_start = time.perf_counter()
        try:
            meta, findings = inspect_episode(path, info, None, views, baselines, factory)
            if meta.get("episode_index") is None:
                stem = path.stem
                if stem.startswith("episode_") and stem[8:14].isdigit():
                    meta["episode_index"] = int(stem[8:14])
            merged = merge_findings(findings)
            score = score_episode(meta, merged)
            rows.append({
                "path": str(path.relative_to(ROOT)),
                "episode_index": meta.get("episode_index"),
                "finding_count": len(merged),
                "score_total": score.score_total,
                "legacy_score_total": score.legacy_score_total,
                "geometry_status": score.geometry_status,
                "phase_status": score.phase_status,
                "elapsed_sec": round(time.perf_counter() - case_start, 3),
                "status": "ok",
            })
        except Exception as exc:
            failures.append({
                "path": str(path.relative_to(ROOT)),
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_sec": round(time.perf_counter() - case_start, 3),
            })
            rows.append({
                "path": str(path.relative_to(ROOT)),
                "episode_index": None,
                "finding_count": None,
                "score_total": None,
                "legacy_score_total": None,
                "geometry_status": "fail",
                "phase_status": "unavailable",
                "elapsed_sec": round(time.perf_counter() - case_start, 3),
                "status": "fail",
            })
        print(f"[{idx}/{len(case_paths)}] {path.name} -> {rows[-1]['status']} ({rows[-1]['elapsed_sec']}s)")

    output_csv = args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["path"])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    elapsed_total = round(time.perf_counter() - start_total, 3)
    summary = {
        "input_root": str(args.input_root),
        "case_count": len(case_paths),
        "success_count": sum(1 for row in rows if row.get("status") == "ok"),
        "failure_count": len(failures),
        "elapsed_total_sec": elapsed_total,
        "throughput_cases_per_sec": round(len(case_paths) / elapsed_total, 3) if elapsed_total else None,
        "mean_case_sec": round(sum(row["elapsed_sec"] for row in rows) / max(1, len(rows)), 3),
        "failures": failures[:20],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote: {output_csv}")
    print(f"wrote: {args.summary_json}")


if __name__ == "__main__":
    main()
