from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from generate_single_issue_ablation_data import (
    ISSUES,
    build_context,
    find_episode_files,
    find_meta_root,
    load_episode,
    load_jsonl,
    write_design_doc,
    write_episode,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "ablations" / "test_dataset_100"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate about 100 deterministic single-issue test trajectories.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    meta_root = find_meta_root()
    info = json.loads((meta_root / "info.json").read_text(encoding="utf-8"))
    tasks = load_jsonl(meta_root / "tasks.jsonl")
    source_files = find_episode_files(meta_root)
    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    cases = []
    global_index = 0
    for case_index in range(args.count):
        spec = ISSUES[case_index % len(ISSUES)]
        source_path = source_files[case_index % len(source_files)]
        source_df, source_schema = load_episode(source_path)
        context = build_context(info, source_path)
        rng = np.random.default_rng(args.seed + case_index)
        df = source_df.copy(deep=True)
        evidence = spec.apply(df, context, rng)
        if "episode_index" in df.columns:
            df["episode_index"] = case_index
        if "index" in df.columns:
            df["index"] = np.arange(global_index, global_index + len(df), dtype=np.int64)
        global_index += len(df)
        case_dir = args.output_root / spec.issue_id
        case_path = case_dir / f"episode_{case_index:06d}_{spec.issue_id}.parquet"
        write_episode(df, source_schema, case_path)
        cases.append({
            "case_index": case_index,
            "issue_id": spec.issue_id,
            "axis": spec.axis,
            "hardness": spec.hardness,
            "category": spec.category,
            "target": spec.target,
            "description": spec.description,
            "expected_detector_family": spec.expected_detector_family,
            "source_episode_index": context["source_episode_index"],
            "source_episode_path": context["source_episode_path"],
            "case_path": str(case_path.relative_to(ROOT)),
            "rows": len(df),
            "evidence": evidence,
        })

    manifest = {
        "generator": str(Path(__file__).relative_to(ROOT)),
        "seed": args.seed,
        "requested_count": args.count,
        "case_count": len(cases),
        "source_dataset": {
            "codebase_version": info.get("codebase_version"),
            "robot_type": info.get("robot_type"),
            "fps": info.get("fps"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "tasks": tasks,
        },
        "policy": {
            "granularity": "one injected issue per generated parquet",
            "source_reference_modified": False,
            "issue_schedule": "round_robin_over_23_issue_specs_and_available_source_episodes",
        },
        "cases": cases,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_design_doc(args.output_root, manifest)
    print(f"generated_cases: {len(cases)}")
    print(f"manifest: {args.output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
