from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "初赛数据"
OUTPUT_ROOT = ROOT / "outputs" / "diagnostics"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

EXPECTED_COLUMNS = {
    "image",
    "left_wrist_image",
    "right_wrist_image",
    "state",
    "actions",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}


@dataclass
class Finding:
    episode_index: int | None
    issue_type: str
    severity: str
    frame_start: int | None
    frame_end: int | None
    evidence: dict[str, Any]


def is_real_parquet(path: Path) -> bool:
    return path.is_file() and not path.name.startswith("._") and "__MACOSX" not in str(path)


def find_parquet_files() -> list[Path]:
    return sorted(p for p in DATA_ROOT.rglob("*.parquet") if is_real_parquet(p))


def find_meta_root() -> Path:
    for candidate in DATA_ROOT.rglob("info.json"):
        if "__MACOSX" not in str(candidate):
            return candidate.parent
    raise FileNotFoundError("info.json not found")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def as_numeric_matrix(series: pd.Series) -> np.ndarray:
    values = []
    for value in series:
        values.append(np.asarray(value, dtype=np.float64).reshape(-1))
    if not values:
        return np.empty((0, 0), dtype=np.float64)
    widths = {row.size for row in values}
    if len(widths) != 1:
        return np.array(values, dtype=object)
    return np.stack(values)


def contiguous_ranges(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    indices = np.sort(np.unique(indices.astype(int)))
    split_points = np.where(np.diff(indices) > 1)[0]
    starts = np.r_[0, split_points + 1]
    ends = np.r_[split_points, len(indices) - 1]
    return [(int(indices[start]), int(indices[end])) for start, end in zip(starts, ends)]


def add_finding(
    findings: list[Finding],
    episode: int | None,
    issue_type: str,
    severity: str,
    frame_start: int | None,
    frame_end: int | None,
    **evidence: Any,
) -> None:
    findings.append(
        Finding(
            episode_index=episode,
            issue_type=issue_type,
            severity=severity,
            frame_start=frame_start,
            frame_end=frame_end,
            evidence=evidence,
        )
    )


def inspect_episode(path: Path, expected: dict[str, Any] | None) -> list[Finding]:
    table = pq.read_table(path)
    df = table.to_pandas()
    episode = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns and len(df) else None
    frame = df["frame_index"].to_numpy(dtype=np.int64) if "frame_index" in df.columns else np.arange(len(df))
    findings: list[Finding] = []

    missing_columns = sorted(EXPECTED_COLUMNS - set(df.columns))
    extra_columns = sorted(set(df.columns) - EXPECTED_COLUMNS)
    if missing_columns:
        add_finding(findings, episode, "missing_columns", "critical", None, None, columns=missing_columns)
    if extra_columns:
        add_finding(findings, episode, "unexpected_columns", "info", None, None, columns=extra_columns)

    if expected is not None and len(df) != int(expected.get("length", len(df))):
        add_finding(
            findings,
            episode,
            "length_mismatch",
            "high",
            None,
            None,
            expected_length=int(expected["length"]),
            actual_length=int(len(df)),
        )

    if len(df) == 0:
        add_finding(findings, episode, "empty_episode", "critical", None, None)
        return findings

    for column in ["frame_index", "index", "episode_index", "task_index"]:
        if column not in df.columns:
            continue
        values = df[column].to_numpy()
        if not np.isfinite(values).all():
            bad = np.where(~np.isfinite(values))[0]
            ranges = contiguous_ranges(frame[bad])
            for start, end in ranges:
                add_finding(findings, episode, f"non_finite_{column}", "high", start, end)

    if "frame_index" in df.columns:
        diffs = np.diff(frame)
        bad = np.where(diffs != 1)[0]
        for idx in bad:
            issue = "duplicate_frame_index" if diffs[idx] == 0 else "frame_index_gap_or_reorder"
            severity = "high" if diffs[idx] < 0 or diffs[idx] > 3 else "medium"
            add_finding(
                findings,
                episode,
                issue,
                severity,
                int(frame[idx]),
                int(frame[idx + 1]),
                previous=int(frame[idx]),
                current=int(frame[idx + 1]),
                delta=int(diffs[idx]),
            )

    if "index" in df.columns:
        values = df["index"].to_numpy(dtype=np.int64)
        diffs = np.diff(values)
        bad = np.where(diffs != 1)[0]
        for idx in bad:
            add_finding(
                findings,
                episode,
                "index_gap_or_reorder",
                "medium",
                int(frame[idx]),
                int(frame[idx + 1]),
                previous=int(values[idx]),
                current=int(values[idx + 1]),
                delta=int(diffs[idx]),
            )

    if "timestamp" in df.columns:
        timestamp = df["timestamp"].to_numpy(dtype=np.float64)
        diffs = np.diff(timestamp)
        finite_diffs = diffs[np.isfinite(diffs)]
        target = float(np.median(finite_diffs)) if finite_diffs.size else 0.1
        tolerance = max(0.02, target * 0.25)
        bad = np.where((diffs <= 0) | (np.abs(diffs - target) > tolerance))[0]
        for idx in bad:
            severity = "high" if diffs[idx] <= 0 or abs(diffs[idx] - target) > target else "medium"
            add_finding(
                findings,
                episode,
                "timestamp_anomaly",
                severity,
                int(frame[idx]),
                int(frame[idx + 1]),
                previous_timestamp=float(timestamp[idx]),
                current_timestamp=float(timestamp[idx + 1]),
                delta=float(diffs[idx]),
                expected_delta=target,
            )

    for column in ["state", "actions"]:
        if column not in df.columns:
            continue
        matrix = as_numeric_matrix(df[column])
        if matrix.dtype == object:
            add_finding(findings, episode, f"inconsistent_{column}_shape", "high", None, None)
            continue
        if matrix.shape[1] != 20:
            add_finding(
                findings,
                episode,
                f"invalid_{column}_shape",
                "critical",
                None,
                None,
                actual_shape=list(matrix.shape),
                expected_shape=[len(df), 20],
            )
        non_finite = ~np.isfinite(matrix).all(axis=1)
        for start, end in contiguous_ranges(frame[np.where(non_finite)[0]]):
            add_finding(findings, episode, f"non_finite_{column}", "high", start, end)

        if len(matrix) > 2:
            # A large first difference is expected during normal robot motion.
            # Use a second difference to target isolated kinematic spikes instead.
            second_delta = np.linalg.norm(matrix[2:] - 2.0 * matrix[1:-1] + matrix[:-2], axis=1)
            median = float(np.median(second_delta))
            mad = float(np.median(np.abs(second_delta - median)))
            floor = 0.25 if column == "state" else 0.40
            threshold = max(median + 12.0 * mad, floor)
            spikes = np.where(second_delta > threshold)[0]
            for idx in spikes:
                add_finding(
                    findings,
                    episode,
                    f"{column}_continuity_spike",
                    "medium",
                    int(frame[idx + 1]),
                    int(frame[idx + 1]),
                    second_delta_norm=float(second_delta[idx]),
                    baseline_median=median,
                    threshold=float(threshold),
                )

    if "episode_index" in df.columns and not (df["episode_index"] == episode).all():
        add_finding(findings, episode, "episode_index_inconsistent", "high", None, None)
    if "task_index" in df.columns and df["task_index"].nunique(dropna=False) != 1:
        add_finding(findings, episode, "task_index_inconsistent", "medium", None, None)

    return findings


def main() -> None:
    meta_root = find_meta_root()
    episodes = {int(row["episode_index"]): row for row in load_jsonl(meta_root / "episodes.jsonl")}
    findings: list[Finding] = []
    for path in find_parquet_files():
        findings.extend(inspect_episode(path, episodes.get(int(path.stem.split("_")[-1]))))

    payload = [asdict(finding) for finding in findings]
    output = OUTPUT_ROOT / "structural_findings.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"episodes_checked: {len(find_parquet_files())}")
    print(f"findings: {len(payload)}")
    if payload:
        by_type: dict[str, int] = {}
        for item in payload:
            by_type[item["issue_type"]] = by_type.get(item["issue_type"], 0) + 1
        print("by_type:", by_type)
    else:
        print("No structural, timestamp, or numeric anomalies detected.")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
