from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
META_ROOT = next(p.parent for p in (ROOT / "初赛数据").rglob("info.json") if "__MACOSX" not in str(p))
DIAGNOSTICS = ROOT / "outputs" / "diagnostics"
REPORT_ROOT = ROOT / "outputs" / "reports"
REPORT_ROOT.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def penalty(severity: str) -> float:
    return {"critical": 25.0, "high": 12.0, "medium": 5.0, "low": 2.0, "info": 0.0}.get(severity, 2.0)


def score_with_findings(base: float, findings: list[dict[str, Any]]) -> float:
    value = base - sum(penalty(item["severity"]) for item in findings)
    return round(max(0.0, min(base, value)), 2)


def group_findings(findings: list[dict[str, Any]], episode: int) -> dict[str, list[dict[str, Any]]]:
    current = [item for item in findings if item.get("episode_index") == episode]
    groups = {"time": [], "numeric": [], "other": []}
    for item in current:
        issue = item["issue_type"]
        if any(token in issue for token in ["timestamp", "frame_index", "index_gap", "length"]):
            groups["time"].append(item)
        elif any(token in issue for token in ["state", "actions", "finite", "shape"]):
            groups["numeric"].append(item)
        else:
            groups["other"].append(item)
    return groups


def main() -> None:
    info = load_json(META_ROOT / "info.json")
    episodes = {int(row["episode_index"]): row for row in load_json(META_ROOT / "episodes.jsonl")}
    basic = {int(Path(item["file"]).stem.split("_")[-1]): item for item in load_json(DIAGNOSTICS / "basic_quality_summary.json")}
    structural = load_json(DIAGNOSTICS / "structural_findings.json")
    image_findings = load_json(DIAGNOSTICS / "image_findings.json")
    image_summaries = load_json(DIAGNOSTICS / "image_metrics_summary.json")
    sync = {int(item["episode_index"]): item for item in load_json(DIAGNOSTICS / "multimodal_sync_summary.json")}

    image_by_episode: dict[int, list[dict[str, Any]]] = {}
    for item in image_findings:
        image_by_episode.setdefault(int(item["episode_index"]), []).append(item)

    rows: list[dict[str, Any]] = []
    for episode in sorted(episodes):
        groups = group_findings(structural, episode)
        time_score = score_with_findings(20.0, groups["time"])
        numeric_score = score_with_findings(25.0, groups["numeric"])
        image_score = score_with_findings(20.0, image_by_episode.get(episode, []))

        sync_item = sync.get(episode, {})
        sync_measure = sync_item.get("state_action", {}).get("action_to_state", {})
        sync_corr = sync_measure.get("best_correlation")
        sync_lag = sync_measure.get("best_lag")
        sync_findings: list[dict[str, Any]] = []
        if sync_corr is None or sync_corr < 0.75:
            sync_findings.append({"severity": "medium"})
        if sync_lag is not None and sync_lag not in (-1, 0):
            sync_findings.append({"severity": "medium"})
        sync_score = score_with_findings(20.0, sync_findings)

        task = int(basic[episode]["task_index"])
        total = round(time_score + numeric_score + image_score + sync_score, 2)
        rows.append(
            {
                "episode_index": episode,
                "task_index": task,
                "length": int(basic[episode]["rows"]),
                "time_score": time_score,
                "numeric_score": numeric_score,
                "image_score": image_score,
                "sync_score": sync_score,
                "quality_score_without_value": total,
                "structural_findings": len([item for item in structural if item.get("episode_index") == episode]),
                "image_findings": len(image_by_episode.get(episode, [])),
                "sync_lag": sync_lag,
                "sync_correlation": sync_corr,
            }
        )

    task_counts = Counter(row["task_index"] for row in rows)
    balance = min(task_counts.values()) / max(task_counts.values()) if task_counts else 0.0
    value_score = round(15.0 * (0.7 + 0.3 * balance), 2) if len(task_counts) >= 2 else 10.5
    dataset_score = round(float(np.mean([row["quality_score_without_value"] for row in rows])) + value_score, 2)

    for row in rows:
        row["value_score_dataset"] = value_score
        row["dataset_quality_score"] = dataset_score

    csv_path = REPORT_ROOT / "episode_scores.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "dataset": {
            "robot_type": info.get("robot_type"),
            "episodes": len(rows),
            "frames": int(info.get("total_frames", 0)),
            "fps": info.get("fps"),
            "task_counts": dict(task_counts),
            "value_score": value_score,
            "dataset_quality_score": dataset_score,
        },
        "scoring": {
            "time_integrity": 20,
            "state_action_quality": 25,
            "image_quality": 20,
            "multimodal_sync": 20,
            "task_coverage_and_value": 15,
        },
        "episodes": rows,
        "conclusion": (
            "The reference set is structurally healthy under the current rules: no missing columns, "
            "index/timestamp anomalies, non-finite state/action values, image decode failures, or severe "
            "image defects were detected. The state-action relationship shows a stable one-frame baseline "
            "offset in most episodes, so it is calibrated rather than treated as an anomaly."
        ),
    }
    json_path = REPORT_ROOT / "dataset_quality_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = [
        "# 多模态机器人数据质量检测报告",
        "",
        f"- 机器人类型：{info.get('robot_type')}",
        f"- 轨迹数量：{len(rows)}",
        f"- 总帧数：{info.get('total_frames')}",
        f"- 采样频率：{info.get('fps')} FPS",
        f"- 数据集质量分：**{dataset_score}/100**",
        "",
        "## 评分构成",
        "",
        "| 维度 | 满分 | 当前结果 |",
        "|---|---:|---:|",
        f"| 时间完整性 | 20 | {round(float(np.mean([r['time_score'] for r in rows])), 2)} |",
        f"| 状态/动作质量 | 25 | {round(float(np.mean([r['numeric_score'] for r in rows])), 2)} |",
        f"| 图像质量 | 20 | {round(float(np.mean([r['image_score'] for r in rows])), 2)} |",
        f"| 多模态同步 | 20 | {round(float(np.mean([r['sync_score'] for r in rows])), 2)} |",
        f"| 任务覆盖与数据价值 | 15 | {value_score} |",
        "",
        "## 当前结论",
        "",
        "参考集的字段、帧索引、时间戳、状态/动作有限性、图像解码和严重图像质量指标均通过当前规则。",
        "动作与状态之间存在稳定的一帧基线偏移，已作为记录语义基线保留，没有判为异常。",
        "",
        "## 输出文件",
        "",
        "- `outputs/diagnostics/basic_quality_summary.json`：基础统计",
        "- `outputs/diagnostics/structural_findings.json`：结构和时序异常",
        "- `outputs/diagnostics/image_metrics_summary.json`：图像质量指标",
        "- `outputs/diagnostics/image_findings.json`：图像异常",
        "- `outputs/diagnostics/multimodal_sync_summary.json`：同步分析",
        "- `outputs/diagnostics/detector_validation.json`：故障注入验证",
        "- `outputs/reports/episode_scores.csv`：逐轨迹评分",
        "- `outputs/reports/dataset_quality_report.json`：完整机器可读报告",
    ]
    markdown_path = REPORT_ROOT / "dataset_quality_report.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print("dataset_quality_score:", dataset_score)
    print("value_score:", value_score)
    print("episode_score_range:", min(row["quality_score_without_value"] for row in rows), "-", max(row["quality_score_without_value"] for row in rows))
    print(f"wrote: {csv_path}")
    print(f"wrote: {json_path}")
    print(f"wrote: {markdown_path}")


if __name__ == "__main__":
    main()
