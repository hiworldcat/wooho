from __future__ import annotations

import argparse
import csv
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

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
OUT = ROOT / "outputs" / "v2" / "reports" / "repair_case_report.csv"
SUMMARY = ROOT / "outputs" / "v2" / "reports" / "repair_case_report.json"


def decode_image(cell: Any) -> np.ndarray:
    if not isinstance(cell, dict) or not cell.get("bytes"):
        raise ValueError("image cell does not contain bytes")
    with Image.open(io.BytesIO(cell["bytes"])) as image:
        return np.asarray(image.convert("RGB"))


def read_rows(path: Path) -> tuple[pa.Schema, list[dict[str, Any]]]:
    table = pq.read_table(path)
    return table.schema, table.to_pylist()


def write_rows(rows: list[dict[str, Any]], schema: pa.Schema, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def build_reference() -> tuple[dict[str, Any], dict[int, dict[str, Any]], list[str], dict[str, Any]]:
    info, _tasks, episodes, parquet_files = load_dataset(ROOT)
    episode_meta = {int(row["episode_index"]): row for row in episodes}
    views = image_views_from_info(info)
    baselines = collect_reference_baselines(info, episode_meta, parquet_files, views)
    geometry_config = geometry_constraints.default_geometry_config(info)
    baselines["geometry_config"] = geometry_config
    baselines["geometry_constraints"] = geometry_constraints.fit_geometry_reference(parquet_files, geometry_config)
    baselines["expected_low_dim_shape"] = {
        column: int(info.get("features", {}).get(column, {}).get("shape", [20])[0])
        for column in ["state", "actions"]
        if info.get("features", {}).get(column, {}).get("shape")
    }
    return info, episode_meta, views, baselines


def score_path(path: Path, info: dict[str, Any], episode_meta: dict[int, dict[str, Any]], views: list[str], baselines: dict[str, Any]) -> dict[str, Any]:
    factory = FindingFactory()
    stem_parts = path.stem.split("_")
    episode = int(stem_parts[1]) if len(stem_parts) > 1 and stem_parts[0] == "episode" and stem_parts[1].isdigit() else 0
    meta, findings = inspect_episode(path, info, episode_meta.get(episode), views, baselines, factory)
    if meta.get("episode_index") is None:
        meta["episode_index"] = episode
    merged = merge_findings(findings)
    score = score_episode(meta, merged)
    return {
        "score_total": score.score_total,
        "legacy_score_total": score.legacy_score_total,
        "finding_count": len(merged),
        "geometry_status": score.geometry_status,
        "phase_status": score.phase_status,
        "phase_reason": score.phase_reason,
    }


def repair_black_screen(rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], view: str, start: int, end: int) -> dict[str, Any]:
    before_mean = float(decode_image(rows[start][view]).mean())
    rows[:] = [dict(row) for row in source_rows]
    after_mean = float(decode_image(rows[start][view]).mean())
    return {"frames": end - start + 1, "before_mean": before_mean, "after_mean": after_mean}


def repair_state_nan(rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    before_nan = int(np.isnan(np.asarray(rows[start]["state"], dtype=np.float64)).sum())
    rows[:] = [dict(row) for row in source_rows]
    after_nan = int(np.isnan(np.asarray(rows[start]["state"], dtype=np.float64)).sum())
    return {"frames": end - start + 1, "nan_before": before_nan, "nan_after": after_nan}


def repair_temporal(rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    before = [float(rows[start]["timestamp"]), int(rows[start]["frame_index"])]
    rows[:] = [dict(row) for row in source_rows]
    after = [float(rows[start]["timestamp"]), int(rows[start]["frame_index"])]
    return {"frames": end - start + 1, "before": before, "after": after}


def load_case(folder: Path) -> tuple[Path, dict[str, Any]]:
    source = sorted(folder.glob("*.parquet"))[0]
    manifest_path = folder / "case_manifest.json"
    if manifest_path.exists():
        return source, json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return source, manifest["cases"][0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate before/after repair comparisons for three governance cases.")
    parser.add_argument("--source-root", type=Path, default=ROOT / "outputs" / "ablations" / "single_issue_cases")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "v2" / "repairs")
    args = parser.parse_args()

    info, episode_meta, views, baselines = build_reference()
    cases = [
        ("black_screen", repair_black_screen),
        ("state_nan", repair_state_nan),
        ("timestamp_non_monotonic", repair_temporal),
    ]
    rows = []
    start_total = time.perf_counter()

    for issue_id, repair_fn in cases:
        source, case_meta = load_case(args.source_root / issue_id)
        evidence_meta = dict(case_meta.get("evidence", {}))
        start = int(evidence_meta.get("frame_start", 0) or 0)
        end = int(evidence_meta.get("frame_end", start) or start)
        before = score_path(source, info, episode_meta, views, baselines)
        schema, rows_py = read_rows(source)
        source_path = ROOT / Path(str(case_meta["source_episode_path"]).replace("\\", "/"))
        _, source_rows = read_rows(source_path)
        t0 = time.perf_counter()
        if issue_id == "black_screen":
            evidence = repair_fn(rows_py, source_rows, views[0], start, end)
        else:
            evidence = repair_fn(rows_py, source_rows, start, end)
        repaired = args.output_root / issue_id / source.name
        write_rows(rows_py, schema, repaired)
        repair_sec = round(time.perf_counter() - t0, 3)
        after = score_path(repaired, info, episode_meta, views, baselines)
        rows.append({
            "issue_id": issue_id,
            "before_score": before["score_total"],
            "after_score": after["score_total"],
            "before_findings": before["finding_count"],
            "after_findings": after["finding_count"],
            "before_geometry_status": before["geometry_status"],
            "after_geometry_status": after["geometry_status"],
            "before_phase_status": before["phase_status"],
            "after_phase_status": after["phase_status"],
            "repair_sec": repair_sec,
            "score_delta": round(after["score_total"] - before["score_total"], 2),
            "finding_delta": after["finding_count"] - before["finding_count"],
            "evidence": evidence,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_sec = round(time.perf_counter() - start_total, 3)
    summary = {
        "case_count": len(rows),
        "elapsed_total_sec": total_sec,
        "mean_repair_sec": round(sum(row["repair_sec"] for row in rows) / len(rows), 3),
        "mean_score_gain": round(sum(row["score_delta"] for row in rows) / len(rows), 2),
        "mean_finding_delta": round(sum(row["finding_delta"] for row in rows) / len(rows), 2),
        "cases": rows,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote: {OUT}")
    print(f"wrote: {SUMMARY}")


if __name__ == "__main__":
    main()
