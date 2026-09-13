from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frozen report values into a PPT-friendly JSON and Markdown sheet."
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/target/v2"))
    parser.add_argument("--destination", type=Path, default=Path("submission_work/ppt_metrics"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    report = load_json(output_root / "reports/dataset_quality_report_v2.json")
    findings = load_json(output_root / "diagnostics/findings_v2.json")
    manifest = load_json(output_root / "diagnostics/run_manifest_v2.json")

    episodes = report.get("episodes", [])
    abnormal_episode_ids = sorted(
        {
            int(item["episode_index"])
            for item in findings
            if item.get("episode_index") is not None
        }
    )
    finding_counts = Counter(str(item.get("issue_type", "unknown")) for item in findings)
    ranked_findings = sorted(
        findings,
        key=lambda item: (
            bool(item.get("illegal")),
            float(item.get("severity_score", 0)),
            float(item.get("quality_penalty", 0)),
        ),
        reverse=True,
    )
    top_case = ranked_findings[0] if ranked_findings else None
    summary = report.get("summary", {})
    dataset = report.get("dataset", {})

    metrics = {
        "source": {
            "dataset_role": manifest.get("dataset_role"),
            "target_dataset": manifest.get("target_dataset"),
            "scoring_version": manifest.get("scoring_version"),
        },
        "slide_07_case": {
            "episode_index": None if top_case is None else top_case.get("episode_index"),
            "frame_start": None if top_case is None else top_case.get("frame_start"),
            "frame_end": None if top_case is None else top_case.get("frame_end"),
            "issue_type": None if top_case is None else top_case.get("issue_type"),
            "confidence_level": None if top_case is None else top_case.get("confidence_level"),
            "severity_score": None if top_case is None else top_case.get("severity_score"),
        },
        "slide_10_governance": {
            "before_finding_count": len(findings),
            "after_finding_count": None,
            "before_score": dataset.get("finder_quality_score", dataset.get("dataset_quality_score")),
            "after_score": None,
            "note": "治理后数值必须来自同配置复检，不得估算。",
        },
        "slide_11_overview": {
            "completed_episodes": len(episodes),
            "abnormal_episodes": len(abnormal_episode_ids),
            "dataset_quality_score": dataset.get("dataset_quality_score"),
            "finding_count": int(summary.get("finding_count", len(findings))),
            "throughput_episodes_per_second": None,
        },
        "diagnostics": {
            "abnormal_episode_ids": abnormal_episode_ids,
            "finding_counts_by_type": dict(finding_counts.most_common()),
            "geometry_status_counts": summary.get("geometry_status_counts", {}),
            "phase_status_counts": summary.get("phase_status_counts", {}),
        },
    }

    args.destination.mkdir(parents=True, exist_ok=True)
    json_path = args.destination / "ppt_metrics.json"
    md_path = args.destination / "PPT_结果回填表.md"
    json_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    case = metrics["slide_07_case"]
    overview = metrics["slide_11_overview"]
    governance = metrics["slide_10_governance"]
    markdown = f"""# PPT结果回填表

## 第7页：正式异常案例

- 轨迹：{case['episode_index']}
- 帧段：{case['frame_start']}–{case['frame_end']}
- 根因：{case['issue_type']}
- 置信度：{case['confidence_level']}
- 严重度：{case['severity_score']}

## 第10页：治理前后

- 治理前问题数：{governance['before_finding_count']}
- 治理前得分：{governance['before_score']}
- 治理后问题数：待同配置复检
- 治理后得分：待同配置复检

## 第11页：20条正式结果总览

- 完成轨迹：{overview['completed_episodes']} / 20
- 异常轨迹：{overview['abnormal_episodes']} / 20
- 质量得分：{overview['dataset_quality_score']}
- 问题数：{overview['finding_count']}
- 运行效率：待正式运行计时
"""
    md_path.write_text(markdown, encoding="utf-8")
    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")


if __name__ == "__main__":
    main()
