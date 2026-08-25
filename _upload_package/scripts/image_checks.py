from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "初赛数据"
OUTPUT_ROOT = ROOT / "outputs" / "diagnostics"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
VIEWS = ["image", "left_wrist_image", "right_wrist_image"]


@dataclass
class Finding:
    episode_index: int | None
    view: str
    issue_type: str
    severity: str
    frame_start: int | None
    frame_end: int | None
    evidence: dict[str, Any]


def is_real_parquet(path: Path) -> bool:
    return path.is_file() and not path.name.startswith("._") and "__MACOSX" not in str(path)


def find_parquet_files() -> list[Path]:
    return sorted(p for p in DATA_ROOT.rglob("*.parquet") if is_real_parquet(p))


def decode_image(value: Any) -> np.ndarray:
    if not isinstance(value, dict) or not value.get("bytes"):
        raise ValueError("image cell does not contain embedded bytes")
    with Image.open(io.BytesIO(value["bytes"])) as image:
        return np.asarray(image.convert("RGB"))


def image_metrics(image: np.ndarray) -> dict[str, float]:
    image_float = image.astype(np.float32)
    gray = image_float.mean(axis=2)
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    histogram, _ = np.histogram(gray, bins=32, range=(0, 255), density=True)
    histogram = histogram[histogram > 0]
    entropy = float(-(histogram * np.log2(histogram)).sum()) if histogram.size else 0.0
    return {
        "mean": float(image_float.mean()),
        "std": float(image_float.std()),
        "sharpness": float(laplacian.var()),
        "black_fraction": float((image_float < 5).all(axis=2).mean()),
        "white_fraction": float((image_float > 250).all(axis=2).mean()),
        "entropy": entropy,
    }


def add_finding(
    findings: list[Finding],
    episode: int | None,
    view: str,
    issue_type: str,
    severity: str,
    start: int | None,
    end: int | None,
    **evidence: Any,
) -> None:
    findings.append(Finding(episode, view, issue_type, severity, start, end, evidence))


def inspect_view(df, episode: int, view: str) -> tuple[dict[str, Any], list[Finding]]:
    frames = df["frame_index"].to_numpy(dtype=np.int64)
    findings: list[Finding] = []
    metrics: list[dict[str, float]] = []
    hashes: list[str] = []
    decoded: list[np.ndarray | None] = []

    for frame, value in zip(frames, df[view]):
        try:
            image = decode_image(value)
            decoded.append(image)
            metrics.append(image_metrics(image))
            hashes.append(hashlib.sha1(image.tobytes()).hexdigest())
            if image.shape != (224, 224, 3):
                add_finding(findings, episode, view, "invalid_image_shape", "high", int(frame), int(frame), shape=list(image.shape))
        except Exception as exc:  # keep scanning the remaining frames
            decoded.append(None)
            metrics.append({})
            hashes.append("")
            add_finding(findings, episode, view, "image_decode_error", "critical", int(frame), int(frame), error=str(exc))

    for idx, metric in enumerate(metrics):
        if not metric:
            continue
        frame = int(frames[idx])
        if (metric["mean"] < 5 and metric["std"] < 5) or metric["black_fraction"] > 0.98:
            add_finding(findings, episode, view, "black_screen", "high", frame, frame, **metric)
        elif (metric["mean"] > 250 and metric["std"] < 5) or metric["white_fraction"] > 0.98:
            add_finding(findings, episode, view, "white_screen", "high", frame, frame, **metric)
        elif metric["sharpness"] < 50:
            add_finding(findings, episode, view, "severe_blur", "medium", frame, frame, **metric)

    # Exact repeated frames are a conservative freeze detector. A short run is allowed.
    start = 0
    while start < len(hashes):
        end = start
        while end + 1 < len(hashes) and hashes[end + 1] and hashes[end + 1] == hashes[start]:
            end += 1
        run_length = end - start + 1
        if hashes[start] and run_length >= 8:
            add_finding(
                findings,
                episode,
                view,
                "frozen_image_run",
                "medium",
                int(frames[start]),
                int(frames[end]),
                run_length=run_length,
            )
        start = end + 1

    valid_metrics = [item for item in metrics if item]
    summary: dict[str, Any] = {"frames": len(frames), "decoded_frames": len(valid_metrics)}
    if valid_metrics:
        for key in valid_metrics[0]:
            values = np.array([item[key] for item in valid_metrics], dtype=np.float64)
            summary[f"{key}_median"] = float(np.median(values))
            summary[f"{key}_min"] = float(np.min(values))
            summary[f"{key}_max"] = float(np.max(values))
    summary["finding_count"] = len(findings)
    return summary, findings


def main() -> None:
    all_findings: list[Finding] = []
    summaries: list[dict[str, Any]] = []
    for path in find_parquet_files():
        table = pq.read_table(path, columns=["frame_index", *VIEWS])
        df = table.to_pandas()
        episode = int(path.stem.split("_")[-1])
        for view in VIEWS:
            summary, findings = inspect_view(df, episode, view)
            summaries.append({"episode_index": episode, "view": view, **summary})
            all_findings.extend(findings)

    (OUTPUT_ROOT / "image_metrics_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "image_findings.json").write_text(
        json.dumps([asdict(item) for item in all_findings], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"episodes_checked: {len(find_parquet_files())}")
    print(f"views_checked: {len(summaries)}")
    print(f"findings: {len(all_findings)}")
    if all_findings:
        counts: dict[str, int] = {}
        for finding in all_findings:
            counts[finding.issue_type] = counts.get(finding.issue_type, 0) + 1
        print("by_type:", counts)
    print(f"wrote: {OUTPUT_ROOT / 'image_metrics_summary.json'}")
    print(f"wrote: {OUTPUT_ROOT / 'image_findings.json'}")


if __name__ == "__main__":
    main()
