from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import geometry_constraints
from v2_quality_pipeline import FindingFactory, image_views_from_info, load_dataset


CASE_NAMES = {
    "state_sensor_delay_small",
    "state_sensor_delay_large",
    "action_response_delay",
    "action_high_freq_jitter",
    "state_high_freq_jitter",
}


def read_parquet(path: Path) -> pd.DataFrame:
    for engine in ("fastparquet", "pyarrow"):
        try:
            return pd.read_parquet(path, engine=engine)
        except Exception:
            continue
    return pd.read_parquet(path)


def case_name(path: Path) -> str:
    for part in path.parts:
        if part in CASE_NAMES:
            return part
    return path.parent.name


def summarize_findings(findings: list[dict]) -> dict[str, object]:
    by_type: dict[str, int] = defaultdict(int)
    spans: list[int] = []
    evidence: list[dict[str, object]] = []
    for finding in findings:
        if is_dataclass(finding):
            item = asdict(finding)
        elif isinstance(finding, dict):
            item = finding
        else:
            item = dict(getattr(finding, "__dict__", {}))
        issue_type = str(item.get("issue_type", "unknown"))
        by_type[issue_type] += 1
        start = item.get("start_frame", item.get("frame_start"))
        end = item.get("end_frame", item.get("frame_end"))
        if isinstance(start, int) and isinstance(end, int):
            spans.append(max(1, end - start + 1))
        ev = item.get("evidence") or {}
        evidence.append({
            "issue_type": issue_type,
            "frames": [start, end],
            "direction_reversals": ev.get("direction_reversals"),
            "path_efficiency": ev.get("path_efficiency"),
            "path_length": ev.get("path_length"),
            "min_path_length": ev.get("min_path_length"),
        })
    return {
        "finding_count": len(findings),
        "by_type": dict(sorted(by_type.items())),
        "span_frames": spans,
        "evidence": evidence[:8],
    }


def inspect_path(path: Path, views: list[str], reference: dict, config: dict) -> dict[str, object]:
    df = read_parquet(path)
    frames = df["frame_index"].to_numpy(dtype=np.int64) if "frame_index" in df else np.arange(len(df), dtype=np.int64)
    image_context = {view: {"motion": np.array([], dtype=np.float64)} for view in views}
    result = geometry_constraints.inspect_episode_geometry(
        df=df,
        frames=frames,
        episode=int(df["episode_index"].iloc[0]) if "episode_index" in df else 0,
        task_index=int(df["task_index"].iloc[0]) if "task_index" in df else 0,
        views=views,
        image_context=image_context,
        reference=reference,
        config=config,
        factory=FindingFactory(),
    )
    summary = summarize_findings(result["findings"])
    summary["path"] = str(path)
    summary["status"] = result["diagnostics"].get("status")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    info, _tasks, _episodes, parquet_files = load_dataset()
    config = geometry_constraints.default_geometry_config(info)
    config_path = root / "scripts" / "geometry_config.json"
    if config_path.exists():
        config = geometry_constraints.load_geometry_config(config_path, config)
    reference = geometry_constraints.fit_geometry_reference(parquet_files[:3], config)
    views = image_views_from_info(info)

    base = root / "outputs" / "ablations" / "single_issue_cases"
    paths = sorted(base.rglob("*.parquet"))
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in paths:
        name = case_name(path)
        if name in CASE_NAMES:
            grouped[name].append(inspect_path(path, views, reference, config))

    compact: dict[str, object] = {}
    for name in sorted(CASE_NAMES):
        rows = grouped.get(name, [])
        issue_counts: dict[str, int] = defaultdict(int)
        spans: list[int] = []
        evidence: list[dict[str, object]] = []
        for row in rows:
            for issue_type, count in row.get("by_type", {}).items():
                issue_counts[str(issue_type)] += int(count)
            spans.extend(int(v) for v in row.get("span_frames", []))
            evidence.extend(row.get("evidence", []))
        compact[name] = {
            "files": len(rows),
            "finding_count": sum(issue_counts.values()),
            "by_type": dict(sorted(issue_counts.items())),
            "span_frames": spans,
            "sample_evidence": evidence[:6],
        }

    print(json.dumps({
        "ablation_base": str(base),
        "parquet_files_seen": len(paths),
        "cases": compact,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

