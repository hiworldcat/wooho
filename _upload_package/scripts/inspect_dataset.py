from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "初赛数据"


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


def main() -> None:
    meta_root = find_meta_root()
    info = json.loads((meta_root / "info.json").read_text(encoding="utf-8"))
    tasks = load_jsonl(meta_root / "tasks.jsonl")
    episodes = load_jsonl(meta_root / "episodes.jsonl")
    parquet_files = find_parquet_files()

    print("=== DATA OVERVIEW ===")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print()
    print(f"tasks: {len(tasks)}")
    for row in tasks:
        print(row)
    print()
    print(f"episodes_meta: {len(episodes)}")
    lengths = [row.get("length", 0) for row in episodes]
    print(
        {
            "min_length": min(lengths),
            "max_length": max(lengths),
            "mean_length": round(sum(lengths) / len(lengths), 2),
        }
    )
    print()
    print(f"parquet_files: {len(parquet_files)}")
    print("first 3 parquet files:")
    for p in parquet_files[:3]:
        print(" ", p)

    print()
    print("=== EPISODE LENGTHS ===")
    for row in episodes[:10]:
        print(f"episode {row['episode_index']:02d}: length={row['length']}, tasks={row['tasks']}")

    task_counts = Counter(tuple(row.get("tasks", [])) for row in episodes)
    print()
    print("task distribution:")
    for key, value in task_counts.items():
        print(value, key)

    print()
    print("ready for parquet inspection")


if __name__ == "__main__":
    main()
