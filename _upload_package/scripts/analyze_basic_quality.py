from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "初赛数据"
OUTPUT_ROOT = ROOT / "outputs"
OUTPUT_ROOT.mkdir(exist_ok=True)


def is_real_parquet(path: Path) -> bool:
    return path.is_file() and not path.name.startswith("._") and "__MACOSX" not in str(path)


def find_meta_root() -> Path:
    for candidate in DATA_ROOT.rglob("info.json"):
        if "__MACOSX" not in str(candidate):
            return candidate.parent
    raise FileNotFoundError("info.json not found")


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_parquet_files() -> list[Path]:
    return sorted(p for p in DATA_ROOT.rglob("*.parquet") if is_real_parquet(p))


def summarize_episode(path: Path) -> dict:
    table = pq.read_table(path)
    df = table.to_pandas()
    summary = {
        "file": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "task_index": int(df["task_index"].iloc[0]) if "task_index" in df.columns and len(df) else None,
        "episode_index": int(df["episode_index"].iloc[0]) if "episode_index" in df.columns and len(df) else None,
        "frame_index_min": int(df["frame_index"].min()) if "frame_index" in df.columns and len(df) else None,
        "frame_index_max": int(df["frame_index"].max()) if "frame_index" in df.columns and len(df) else None,
        "timestamp_min": float(df["timestamp"].min()) if "timestamp" in df.columns and len(df) else None,
        "timestamp_max": float(df["timestamp"].max()) if "timestamp" in df.columns and len(df) else None,
        "timestamp_gap_median": None,
        "timestamp_gap_max": None,
        "timestamp_monotonic": None,
        "nan_count": 0,
        "inf_count": 0,
    }
    if "timestamp" in df.columns and len(df) > 1:
        ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
        gaps = np.diff(ts)
        summary["timestamp_gap_median"] = float(np.nanmedian(gaps))
        summary["timestamp_gap_max"] = float(np.nanmax(gaps))
        summary["timestamp_monotonic"] = bool(np.all(gaps >= 0))
    for col in ["state", "actions"]:
        if col in df.columns:
            arr = np.stack(df[col].to_numpy())
            summary["nan_count"] += int(np.isnan(arr).sum())
            summary["inf_count"] += int(np.isinf(arr).sum())
    return summary


def main() -> None:
    meta_root = find_meta_root()
    info = json.loads((meta_root / "info.json").read_text(encoding="utf-8"))
    episodes = load_jsonl(meta_root / "episodes.jsonl")
    parquet_files = find_parquet_files()

    print("=== PARQUET SCHEMA OF FIRST EPISODE ===")
    first = parquet_files[0]
    schema = pq.read_schema(first)
    print(schema)
    print()

    print("=== BASIC SUMMARY ===")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"episode_meta_count: {len(episodes)}")
    print(f"parquet_count: {len(parquet_files)}")
    print()

    rows = [summarize_episode(p) for p in parquet_files]
    rows_sorted = sorted(rows, key=lambda x: x["episode_index"] if x["episode_index"] is not None else -1)

    total_rows = sum(r["rows"] for r in rows_sorted)
    lengths = [r["rows"] for r in rows_sorted]
    print("length stats:", {
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "total": int(total_rows),
    })
    print()

    task_counts = Counter(r["task_index"] for r in rows_sorted)
    print("task counts:", dict(task_counts))
    print()

    print("=== EPISODE HEALTH ===")
    for row in rows_sorted:
        print(
            {
                "episode": row["episode_index"],
                "task": row["task_index"],
                "rows": row["rows"],
                "timestamp_monotonic": row["timestamp_monotonic"],
                "gap_median": row["timestamp_gap_median"],
                "gap_max": row["timestamp_gap_max"],
                "nan_count": row["nan_count"],
                "inf_count": row["inf_count"],
            }
        )

    out = OUTPUT_ROOT / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "basic_quality_summary.json").write_text(
        json.dumps(rows_sorted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"wrote: {out / 'basic_quality_summary.json'}")


if __name__ == "__main__":
    main()
