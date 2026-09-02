from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - optional fallback when pyarrow is unavailable
    pq = None
try:
    import fastparquet  # type: ignore
except Exception:  # pragma: no cover - optional fallback when fastparquet is unavailable
    fastparquet = None
from PIL import Image

import geometry_constraints


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "v2"
DIAG_ROOT = OUTPUT_ROOT / "diagnostics"
REPORT_ROOT = OUTPUT_ROOT / "reports"
DIAG_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_ROOT.mkdir(parents=True, exist_ok=True)

def parquet_column_names(path: Path) -> list[str]:
    if pq is not None:
        return list(pq.read_schema(path).names)
    if fastparquet is not None:
        return list(fastparquet.ParquetFile(path).columns)
    raise ImportError("No parquet engine available")

def read_parquet_frame(path: Path, columns: list[str] | None = None) -> Any:
    if pq is not None:
        table = pq.read_table(path, columns=columns)
        return table.to_pandas()
    if fastparquet is not None:
        parquet_file = fastparquet.ParquetFile(path)
        return parquet_file.to_pandas(columns=columns)
    raise ImportError("No parquet engine available")

DEFAULT_IMAGE_VIEWS = ["image", "left_wrist_image", "right_wrist_image"]
LOW_DIM_COLUMNS = ["state", "actions"]
CONSISTENCY_LOW_DIM_COLUMNS = ["state"]
INDEX_COLUMNS = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
FREEZE_GRACE_FRAMES = 10
MIN_FREEZE_FRAMES = FREEZE_GRACE_FRAMES + 1
FREEZE_MILD_FRAMES = 50
FREEZE_SEVERE_FRAMES = 100
MIN_FREEZE_REPORT_SEVERITY = 55.0
REFERENCE_ENVELOPE_MARGIN = 1.05
REFERENCE_SOFT_DIM_CAPS = {
    "vision_single": 0.8,
    "vision_vision": 0.6,
    "temporal": 2.0,
    "cross_modal": 1.2,
}
STATE_JITTER_P99_MULTIPLIER = 1.5
STATE_JITTER_MAD_MULTIPLIER = 8.0
VISION_JITTER_P99_MULTIPLIER = 2.0
VISION_JITTER_MAD_MULTIPLIER = 10.0
STATE_JITTER_MIN_CLUSTER = 2
STATE_JITTER_CLUSTER_GAP = 2
STATE_JITTER_SINGLE_SPIKE_MIN_SCORE = 95.0
PHASE_NAMES = ("start", "active", "end")
MIN_PHASE_FRAMES = 8
PHASE_ACTIVITY_P95_MULTIPLIER = 0.65
PHASE_ACTIVITY_MAD_MULTIPLIER = 2.5


@dataclass
class Finding:
    finding_id: str
    category_id: str
    category_name: str
    issue_type: str
    issue_name: str
    definition: str
    decision_rule: str
    object_level: str
    modality: str
    confidence_level: str
    illegal: bool
    severity_score: float
    quality_penalty: float
    episode_index: int | None = None
    task_index: int | None = None
    view: str | None = None
    column: str | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    merged_count: int = 1


@dataclass
class EpisodeResult:
    episode_index: int
    task_index: int | None
    length: int
    file: str
    score_total: float
    legacy_score_total: float
    score_detection_quality: float
    score_data_value: float
    score_engineering_reliability: float
    scoring_version: str
    data_value_reasons: list[str]
    engineering_reliability_reasons: list[str]
    score_structural: float
    score_vision_single: float
    score_vision_vision: float
    score_state: float
    score_temporal: float
    score_cross_modal: float
    soft_penalty_total: float
    soft_penalties: dict[str, float]
    finding_count: int
    critical_count: int
    high_confidence_count: int
    suspicious_count: int
    ood_count: int
    geometry_status: str = "ok"
    geometry_reason: str | None = None
    geometry_module_statuses: dict[str, Any] = field(default_factory=dict)
    phase_status: str = "unavailable"
    phase_reason: str | None = None
    phase_segments: list[dict[str, Any]] = field(default_factory=list)


STATUS_COLUMNS = list(geometry_constraints.MODULE_STATUS_VALUES)
GEOMETRY_REPORT_MODULES = [
    ("geometry", lambda item: item.geometry_status),
    ("arms.left", lambda item: dict(item.geometry_module_statuses.get("arms", {})).get("left", "unavailable")),
    ("arms.right", lambda item: dict(item.geometry_module_statuses.get("arms", {})).get("right", "unavailable")),
    ("bimanual", lambda item: item.geometry_module_statuses.get("bimanual", "unavailable")),
    ("state_vision.left", lambda item: dict(item.geometry_module_statuses.get("state_vision", {})).get("left", "unavailable")),
    ("state_vision.right", lambda item: dict(item.geometry_module_statuses.get("state_vision", {})).get("right", "unavailable")),
]


def normalized_status_counts(counter: Counter[str]) -> dict[str, int]:
    return {status: int(counter.get(status, 0)) for status in STATUS_COLUMNS}


def geometry_module_status(value: Any) -> str:
    status = str(value or "unavailable")
    return status if status in geometry_constraints.MODULE_STATUS_SET else "fail"


STANDARDS: dict[str, dict[str, str]] = {
    "vision_illegal": {
        "category_id": "1.1.1.A",
        "category_name": "Vision data illegal problem",
        "definition": "Vision data violates file, schema, decoding, shape, channel, dtype, coverage, or frame alignment constraints.",
        "decision_rule": "Any required view is missing, any image cell cannot be decoded, decoded shape does not match metadata, or image frame coverage cannot align with the trajectory frame.",
    },
    "state_illegal": {
        "category_id": "1.1.1.B",
        "category_name": "State data illegal problem",
        "definition": "Low-dimensional state data violates schema, shape, finite-value, length, or index constraints.",
        "decision_rule": "Any low-dimensional field is missing, has inconsistent row shape, contains NaN/Inf, or cannot align with the episode frame count.",
    },
    "vision_single": {
        "category_id": "1.1.2.A",
        "category_name": "Vision single-frame quality problem",
        "definition": "A decodable single image has abnormal visual quality or content distribution.",
        "decision_rule": "Brightness, contrast, sharpness, black/white ratio, entropy, or reference-set deviation crosses calibrated thresholds for the same camera.",
    },
    "vision_vision": {
        "category_id": "1.1.2.B",
        "category_name": "Vision-Vision consistency problem",
        "definition": "Multiple camera views at the same time are mutually inconsistent in quality, motion, synchronization, or coarse scene relation.",
        "decision_rule": "Reserved for hard multi-camera calibration metadata. The v2 detector does not directly compare camera pixels without a state-supported overlap gate.",
    },
    "vision_state_vision": {
        "category_id": "1.1.2.C",
        "category_name": "Vision-State-Vision geometry/scale problem",
        "definition": "Visual spatial change and robot state geometry are inconsistent, suggesting broken geometry, scale, crop, or coarse cross-modal relation.",
        "decision_rule": "Known camera semantics and robot state first indicate that two views should overlap; only then is weak visual motion correlation used as verification.",
    },
    "state_vision_state": {
        "category_id": "1.1.2.D",
        "category_name": "State-Vision-State consistency problem",
        "definition": "Robot state values or local transitions are inconsistent with visual evidence or neighboring state context.",
        "decision_rule": "Robot state value/delta/acceleration is a calibrated outlier and visual evidence does not support the same local change.",
    },
    "temporal_illegal": {
        "category_id": "1.2.1",
        "category_name": "Temporal illegal problem",
        "definition": "Trajectory order, timestamp, index, episode boundary, or modal length violates hard temporal constraints.",
        "decision_rule": "Frame index is duplicated, skipped, or reordered; timestamp is non-monotonic; episode/task index is inconsistent; metadata length disagrees with parquet rows.",
    },
    "vision_temporal": {
        "category_id": "1.2.2.A",
        "category_name": "Vision frozen/fast/jitter problem",
        "definition": "Vision sequence is legally structured but has abnormal temporal dynamics such as freezing, sudden fast motion, or jitter.",
        "decision_rule": "Exact repeated frames, near-zero visual motion, extreme visual motion, or high-frequency visual acceleration persists beyond calibrated window thresholds.",
    },
    "state_temporal": {
        "category_id": "1.2.2.B",
        "category_name": "State frozen/fast/jitter problem",
        "definition": "Low-dimensional sequence is legally structured but has abnormal temporal dynamics such as freezing, fast transition, spike, or jitter.",
        "decision_rule": "State delta, acceleration, local variance, or unchanged-run length crosses calibrated task/column thresholds.",
    },
    "vision_state_temporal": {
        "category_id": "1.2.2.C",
        "category_name": "Vision-State temporal consistency problem",
        "definition": "Vision and robot state are locally out of sync or show unstable response timing.",
        "decision_rule": "Best lag between visual motion and robot state motion deviates from the calibrated reference lag, or the best correlation falls below the reference baseline.",
    },
}


DIMENSION_POINTS = {
    "structural": 25.0,
    "vision_single": 20.0,
    "vision_vision": 10.0,
    "state": 15.0,
    "temporal": 15.0,
    "cross_modal": 15.0,
}
LEGACY_DIMENSION_POINTS = dict(DIMENSION_POINTS)
OFFICIAL_SCORING_VERSION = "official_70_20_10_v1"
OFFICIAL_SCORE_POINTS = {
    "detection_quality": 70.0,
    "data_value": 20.0,
    "engineering_reliability": 10.0,
}

CATEGORY_TO_DIMENSION = {
    "1.1.1.A": "structural",
    "1.1.1.B": "structural",
    "1.2.1": "structural",
    "1.1.2.A": "vision_single",
    "1.1.2.B": "vision_vision",
    "1.1.2.C": "cross_modal",
    "1.1.2.D": "cross_modal",
    "1.2.2.A": "temporal",
    "1.2.2.B": "temporal",
    "1.2.2.C": "cross_modal",
}


def is_real_path(path: Path) -> bool:
    return path.is_file() and not path.name.startswith("._") and "__MACOSX" not in str(path)


def find_meta_root(search_root: Path | None = None) -> Path:
    root = (search_root or ROOT).resolve()
    candidates: list[Path] = []
    for info_path in root.rglob("info.json"):
        if not is_real_path(info_path):
            continue
        if "outputs" in info_path.parts:
            continue
        candidates.append(info_path.parent)
    unique_candidates = sorted({candidate.resolve() for candidate in candidates})
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    if not unique_candidates:
        raise FileNotFoundError(f"Could not find LeRobot info.json under {root}")
    raise FileNotFoundError(
        f"Multiple info.json candidates under {root}; pass --reference-root and --target-root explicitly."
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dataset(dataset_root: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    meta_root = find_meta_root(dataset_root)
    info = json.loads((meta_root / "info.json").read_text(encoding="utf-8"))
    tasks = load_jsonl(meta_root / "tasks.jsonl")
    episodes = load_jsonl(meta_root / "episodes.jsonl")
    data_root = meta_root.parent
    parquet_files = sorted(path for path in data_root.rglob("*.parquet") if is_real_path(path) and "outputs" not in path.parts)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet episodes found under {data_root}")
    return info, tasks, episodes, parquet_files


def image_views_from_info(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    views = [name for name, spec in features.items() if spec.get("dtype") == "image"]
    return views or DEFAULT_IMAGE_VIEWS


def expected_columns_from_info(info: dict[str, Any]) -> set[str]:
    return set(info.get("features", {}).keys()) or set(DEFAULT_IMAGE_VIEWS + LOW_DIM_COLUMNS + INDEX_COLUMNS)


def safe_episode_from_path(path: Path) -> int | None:
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return None


def to_float_matrix(values: Any) -> np.ndarray | None:
    rows: list[np.ndarray] = []
    widths: set[int] = set()
    for value in values:
        try:
            row = np.asarray(value, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        rows.append(row)
        widths.add(row.size)
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    if len(widths) != 1:
        return None
    return np.stack(rows)


def contiguous_ranges(indices: np.ndarray, max_gap: int = 1) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    values = np.sort(np.unique(indices.astype(int)))
    split_points = np.where(np.diff(values) > max_gap)[0]
    starts = np.r_[0, split_points + 1]
    ends = np.r_[split_points, len(values) - 1]
    return [(int(values[start]), int(values[end])) for start, end in zip(starts, ends)]


def decode_image(value: Any) -> np.ndarray:
    if not isinstance(value, dict) or not value.get("bytes"):
        raise ValueError("image cell does not contain embedded bytes")
    with Image.open(io.BytesIO(value["bytes"])) as image:
        return np.asarray(image.convert("RGB"))


def downsample_gray(image: np.ndarray, stride: int = 8) -> np.ndarray:
    gray = image.astype(np.float32).mean(axis=2)
    return gray[::stride, ::stride]


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
    histogram, _ = np.histogram(gray, bins=32, range=(0, 255), density=False)
    histogram = histogram.astype(np.float64)
    histogram = histogram[histogram > 0]
    probabilities = histogram / float(histogram.sum()) if histogram.size else histogram
    entropy = float(-(probabilities * np.log2(probabilities)).sum()) if probabilities.size else 0.0
    return {
        "mean": float(image_float.mean()),
        "std": float(image_float.std()),
        "sharpness": float(laplacian.var()),
        "black_fraction": float((image_float < 5).all(axis=2).mean()),
        "white_fraction": float((image_float > 250).all(axis=2).mean()),
        "entropy": entropy,
    }


def robust_stats(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "median": None,
            "mad": None,
            "mean": None,
            "std": None,
            "p01": None,
            "p05": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return {
        "count": int(arr.size),
        "median": median,
        "mad": mad,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p01": float(np.quantile(arr, 0.01)),
        "p05": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def stat_value(stats: dict[str, Any], key: str, default: float) -> float:
    value = stats.get(key)
    if value is None or not np.isfinite(value):
        return default
    return float(value)


def robust_z(value: float, stats: dict[str, Any]) -> float:
    median = stat_value(stats, "median", value)
    mad = stat_value(stats, "mad", 0.0)
    std = stat_value(stats, "std", 0.0)
    scale = max(1.4826 * mad, std, 1e-9)
    return abs(float(value) - median) / scale


def lower_outlier_score(value: float, stats: dict[str, Any], weak_z: float = 4.0, strong_z: float = 8.0) -> float:
    if stats.get("count", 0) < 5:
        return 0.0
    median = stat_value(stats, "median", value)
    if value >= median:
        return 0.0
    z = robust_z(value, stats)
    return normalized_score(z, weak_z, strong_z)


def upper_outlier_score(value: float, stats: dict[str, Any], weak_z: float = 4.0, strong_z: float = 8.0) -> float:
    if stats.get("count", 0) < 5:
        return 0.0
    median = stat_value(stats, "median", value)
    if value <= median:
        return 0.0
    z = robust_z(value, stats)
    return normalized_score(z, weak_z, strong_z)


def upper_tail_pressure(value: float, stats: dict[str, Any], start_key: str = "p95") -> float:
    if stats.get("count", 0) < 5:
        return 0.0
    start = stat_value(stats, start_key, value)
    upper = stat_value(stats, "max", value)
    if value <= start or upper <= start:
        return 0.0
    return min(1.0, (float(value) - start) / max(upper - start, 1e-9))


def lower_tail_pressure(value: float, stats: dict[str, Any], start_key: str = "p05") -> float:
    if stats.get("count", 0) < 5:
        return 0.0
    start = stat_value(stats, start_key, value)
    lower = stat_value(stats, "min", value)
    if value >= start or start <= lower:
        return 0.0
    return min(1.0, (start - float(value)) / max(start - lower, 1e-9))


def two_sided_tail_pressure(value: float, stats: dict[str, Any]) -> float:
    return max(upper_tail_pressure(value, stats), lower_tail_pressure(value, stats))


def two_sided_outlier_score(value: float, stats: dict[str, Any], weak_z: float = 4.0, strong_z: float = 8.0) -> float:
    if stats.get("count", 0) < 5:
        return 0.0
    return normalized_score(robust_z(value, stats), weak_z, strong_z)


def normalized_score(value: float, weak: float, strong: float) -> float:
    if not np.isfinite(value) or value <= weak:
        return 0.0
    if value >= strong:
        return 1.0
    return float((value - weak) / max(strong - weak, 1e-9))


def freeze_duration_severity(run_len: int, exact: bool = False) -> float:
    if run_len <= FREEZE_GRACE_FRAMES:
        return 0.0
    mild_start, mild_end = (30.0, 55.0) if exact else (20.0, 45.0)
    severe_end = 85.0 if exact else 80.0
    cap = 100.0 if exact else 95.0
    if run_len <= FREEZE_MILD_FRAMES:
        ratio = (run_len - FREEZE_GRACE_FRAMES) / max(FREEZE_MILD_FRAMES - FREEZE_GRACE_FRAMES, 1)
        return mild_start + ratio * (mild_end - mild_start)
    if run_len <= FREEZE_SEVERE_FRAMES:
        ratio = (run_len - FREEZE_MILD_FRAMES) / max(FREEZE_SEVERE_FRAMES - FREEZE_MILD_FRAMES, 1)
        return mild_end + ratio * (severe_end - mild_end)
    ratio = min(1.0, (run_len - FREEZE_SEVERE_FRAMES) / 100.0)
    return severe_end + ratio * (cap - severe_end)


def confidence_from_score(score: float, illegal: bool = False, ood: bool = False) -> str:
    if illegal:
        return "确定异常"
    if ood:
        return "分布外样本"
    if score >= 50:
        return "高置信异常"
    return "疑似异常"


def severity_to_penalty(score: float, illegal: bool, category_id: str, frame_start: int | None, frame_end: int | None) -> float:
    dimension = CATEGORY_TO_DIMENSION.get(category_id, "state")
    max_points = DIMENSION_POINTS.get(dimension, 10.0)
    coverage_factor = 1.0
    if frame_start is not None and frame_end is not None:
        coverage_factor = min(1.5, 1.0 + max(0, frame_end - frame_start) / 200.0)
    base = max_points * (score / 100.0) * coverage_factor
    if illegal:
        base = max(base, max_points * 0.65)
    return round(min(max_points, base), 3)


class FindingFactory:
    def __init__(self) -> None:
        self.counter = 0

    def make(
        self,
        standard_key: str,
        issue_type: str,
        issue_name: str,
        object_level: str,
        modality: str,
        severity_score: float,
        illegal: bool = False,
        ood: bool = False,
        episode_index: int | None = None,
        task_index: int | None = None,
        view: str | None = None,
        column: str | None = None,
        frame_start: int | None = None,
        frame_end: int | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> Finding:
        self.counter += 1
        standard = STANDARDS[standard_key]
        category_id = standard["category_id"]
        score = 100.0 if illegal else round(max(0.0, min(100.0, severity_score)), 2)
        return Finding(
            finding_id=f"V2F-{self.counter:06d}",
            category_id=category_id,
            category_name=standard["category_name"],
            issue_type=issue_type,
            issue_name=issue_name,
            definition=standard["definition"],
            decision_rule=standard["decision_rule"],
            object_level=object_level,
            modality=modality,
            confidence_level=confidence_from_score(score, illegal=illegal, ood=ood),
            illegal=illegal,
            severity_score=score,
            quality_penalty=severity_to_penalty(score, illegal, category_id, frame_start, frame_end),
            episode_index=episode_index,
            task_index=task_index,
            view=view,
            column=column,
            frame_start=frame_start,
            frame_end=frame_end,
            evidence=evidence or {},
        )



def collect_reference_baselines(
    info: dict[str, Any],
    episodes_meta: dict[int, dict[str, Any]],
    parquet_files: list[Path],
    views: list[str],
) -> dict[str, Any]:
    image_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    image_motion: dict[str, list[float]] = defaultdict(list)
    image_accel: dict[str, list[float]] = defaultdict(list)
    state_values: dict[str, list[float]] = defaultdict(list)
    state_row_abs: dict[str, list[float]] = defaultdict(list)
    state_delta: dict[str, list[float]] = defaultdict(list)
    state_accel: dict[str, list[float]] = defaultdict(list)

    expected_shape_by_view = {
        name: tuple(info.get("features", {}).get(name, {}).get("shape", [224, 224, 3]))
        for name in views
    }

    for path in parquet_files:
        columns = [column for column in set(views + LOW_DIM_COLUMNS + ["frame_index", "task_index"]) if column]
        try:
            column_names = parquet_column_names(path)
            df = read_parquet_frame(path, [c for c in columns if c in column_names])
        except Exception:
            continue
        if len(df) == 0:
            continue

        task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else None
        task_key = f"task:{task_index}" if task_index is not None else "task:unknown"

        visual_motions: dict[str, np.ndarray] = {}
        for view in views:
            if view not in df.columns:
                continue
            previous: np.ndarray | None = None
            motions: list[float] = []
            for value in df[view]:
                try:
                    image = decode_image(value)
                except Exception:
                    previous = None
                    continue
                metrics = image_metrics(image)
                if tuple(image.shape) == expected_shape_by_view.get(view):
                    for metric_name, metric_value in metrics.items():
                        image_values[f"global|{view}"][metric_name].append(metric_value)
                        image_values[f"{task_key}|{view}"][metric_name].append(metric_value)
                gray = downsample_gray(image)
                if previous is not None:
                    motions.append(float(np.abs(gray - previous).mean()))
                previous = gray
            if motions:
                motion_arr = np.asarray(motions, dtype=np.float64)
                visual_motions[view] = motion_arr
                image_motion[f"global|{view}"].extend(motion_arr.tolist())
                image_motion[f"{task_key}|{view}"].extend(motion_arr.tolist())
                if motion_arr.size > 2:
                    accel = np.abs(np.diff(motion_arr))
                    image_accel[f"global|{view}"].extend(accel.tolist())
                    image_accel[f"{task_key}|{view}"].extend(accel.tolist())


        state_motion: dict[str, np.ndarray] = {}
        for column in LOW_DIM_COLUMNS:
            if column not in df.columns:
                continue
            matrix = to_float_matrix(df[column])
            if matrix is None or matrix.size == 0 or not np.isfinite(matrix).all():
                continue
            state_values[f"global|{column}"].extend(matrix.reshape(-1).tolist())
            state_values[f"{task_key}|{column}"].extend(matrix.reshape(-1).tolist())
            if len(matrix) > 1:
                delta = np.linalg.norm(np.diff(matrix, axis=0), axis=1)
                state_delta[f"global|{column}"].extend(delta.tolist())
                state_delta[f"{task_key}|{column}"].extend(delta.tolist())
                state_motion[column] = delta
            if len(matrix) > 2:
                accel = np.linalg.norm(matrix[2:] - 2.0 * matrix[1:-1] + matrix[:-2], axis=1)
                state_accel[f"global|{column}"].extend(accel.tolist())
                state_accel[f"{task_key}|{column}"].extend(accel.tolist())


    phase_state_delta: dict[str, list[float]] = defaultdict(list)
    phase_state_accel: dict[str, list[float]] = defaultdict(list)
    phase_episode_summaries: list[dict[str, Any]] = []
    phase_reference_delta = {key: robust_stats(values) for key, values in state_delta.items()}
    for path in parquet_files:
        try:
            column_names = parquet_column_names(path)
            wanted = [name for name in ["state", "frame_index", "episode_index", "task_index"] if name in column_names]
            df = read_parquet_frame(path, wanted)
        except Exception:
            continue
        if len(df) == 0 or "state" not in df.columns:
            continue
        matrix = to_float_matrix(df["state"])
        if matrix is None or matrix.shape[1] < 20 or not np.isfinite(matrix).all():
            continue
        episode = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else None
        task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else None
        delta = np.linalg.norm(np.diff(matrix, axis=0), axis=1) if len(matrix) > 1 else np.array([], dtype=np.float64)
        accel = np.linalg.norm(matrix[2:] - 2.0 * matrix[1:-1] + matrix[:-2], axis=1) if len(matrix) > 2 else np.array([], dtype=np.float64)
        phase_result = phase_transition_labels(delta, phase_reference_delta.get("global|state", {"count": 0}))
        if phase_result.get("status") != "ok":
            phase_episode_summaries.append({
                "episode_index": episode,
                "task_index": task_index,
                "length": int(len(matrix)),
                "phase_status": phase_result.get("status"),
                "phase_reason": phase_result.get("reason"),
            })
            continue
        labels = np.asarray(phase_result.get("transition_labels", []), dtype=object)
        accel_labels = labels[1:] if len(labels) > 1 else np.array([], dtype=object)
        for phase in PHASE_NAMES:
            delta_mask = labels == phase
            accel_mask = accel_labels == phase
            if delta_mask.any():
                for scope in [f"global|phase:{phase}|state", f"task:{task_index}|phase:{phase}|state"]:
                    phase_state_delta[scope].extend(delta[delta_mask].tolist())
            if accel_mask.any():
                for scope in [f"global|phase:{phase}|state", f"task:{task_index}|phase:{phase}|state"]:
                    phase_state_accel[scope].extend(accel[accel_mask].tolist())
        phase_episode_summaries.append({
            "episode_index": episode,
            "task_index": task_index,
            "length": int(len(matrix)),
            "phase_status": "ok",
            "phase_reason": phase_result.get("reason"),
            "phase_segments": phase_result.get("segments", []),
        })

    return {
        "image_metrics": {
            key: {metric: robust_stats(values) for metric, values in metric_map.items()}
            for key, metric_map in image_values.items()
        },
        "image_motion": {key: robust_stats(values) for key, values in image_motion.items()},
        "image_accel": {key: robust_stats(values) for key, values in image_accel.items()},
        "state_values": {key: robust_stats(values) for key, values in state_values.items()},
        "state_row_abs": {key: robust_stats(values) for key, values in state_row_abs.items()},
        "state_delta": {key: robust_stats(values) for key, values in state_delta.items()},
        "state_accel": {key: robust_stats(values) for key, values in state_accel.items()},
        "phase_state_delta": {key: robust_stats(values) for key, values in phase_state_delta.items()},
        "phase_state_accel": {key: robust_stats(values) for key, values in phase_state_accel.items()},
        "phase_episode_summaries": phase_episode_summaries,
        "expected_image_shape": {key: list(value) for key, value in expected_shape_by_view.items()},
        "reference_policy": {
            "baseline_layers": ["task+view/column", "global+view/column"],
            "fallback": "If task-level baseline has fewer than 5 samples, the global baseline is used. If the global baseline is also too small, only hard rules are applied.",
            "statistics": "median, MAD, p01, p05, p95, p99, min, max",
        },
        "episodes_used": len(parquet_files),
        "episode_lengths": {
            int(ep): int(meta.get("length", 0))
            for ep, meta in episodes_meta.items()
        },
    }


def baseline_lookup(
    baselines: dict[str, Any],
    section: str,
    task_index: int | None,
    *parts: str,
) -> dict[str, Any]:
    key_task = "|".join([f"task:{task_index}" if task_index is not None else "task:unknown", *parts])
    key_global = "|".join(["global", *parts])
    section_data = baselines.get(section, {})
    task_stats = section_data.get(key_task)
    if isinstance(task_stats, dict) and task_stats.get("count", 0) >= 5:
        return task_stats
    global_stats = section_data.get(key_global)
    if isinstance(global_stats, dict):
        return global_stats
    return {"count": 0}


def reference_stats_ready(stats: dict[str, Any]) -> bool:
    return isinstance(stats, dict) and int(stats.get("count", 0)) >= 5


def reference_lower_threshold(stats: dict[str, Any], key: str = "p05", multiplier: float = 0.5, floor: float = 1e-8) -> float | None:
    if not reference_stats_ready(stats):
        return None
    value = stat_value(stats, key, float("nan"))
    if not np.isfinite(value):
        return None
    return max(floor, float(value) * float(multiplier))


def reference_upper_threshold(
    stats: dict[str, Any],
    key: str = "p95",
    quantile_multiplier: float = 1.5,
    mad_multiplier: float = 8.0,
) -> float | None:
    if not reference_stats_ready(stats):
        return None
    candidates: list[float] = []
    for candidate in [
        stat_value(stats, key, float("nan")) * float(quantile_multiplier),
        stat_value(stats, "median", 0.0) + float(mad_multiplier) * stat_value(stats, "mad", 0.0),
        stat_value(stats, "max", float("nan")) * REFERENCE_ENVELOPE_MARGIN,
    ]:
        if np.isfinite(candidate):
            candidates.append(float(candidate))
    return max(candidates) if candidates else None


def phase_transition_labels(delta: np.ndarray, delta_stats: dict[str, Any]) -> dict[str, Any]:
    if delta.size == 0:
        return {"status": "unavailable", "reason": "empty_delta", "transition_labels": []}
    quiet_threshold = reference_lower_threshold(delta_stats, key="p05", multiplier=0.5)
    active_threshold = reference_upper_threshold(
        delta_stats,
        key="p95",
        quantile_multiplier=PHASE_ACTIVITY_P95_MULTIPLIER,
        mad_multiplier=PHASE_ACTIVITY_MAD_MULTIPLIER,
    )
    if quiet_threshold is None or active_threshold is None or active_threshold <= quiet_threshold:
        return {
            "status": "unavailable",
            "reason": "reference_phase_baseline_missing",
            "transition_labels": ["unknown"] * int(delta.size),
        }
    active_idx = np.where(np.isfinite(delta) & (delta >= active_threshold))[0]
    if active_idx.size < MIN_PHASE_FRAMES:
        return {
            "status": "unavailable",
            "reason": "reference_active_window_missing",
            "quiet_threshold": float(quiet_threshold),
            "active_threshold": float(active_threshold),
            "transition_labels": ["unknown"] * int(delta.size),
        }
    first = int(active_idx[0])
    last = int(active_idx[-1])
    if last - first + 1 < MIN_PHASE_FRAMES:
        return {
            "status": "unavailable",
            "reason": "active_window_too_short",
            "quiet_threshold": float(quiet_threshold),
            "active_threshold": float(active_threshold),
            "transition_labels": ["unknown"] * int(delta.size),
        }
    labels = np.full(delta.shape, "start", dtype=object)
    labels[first : last + 1] = "active"
    labels[last + 1 :] = "end"
    segments: list[dict[str, Any]] = []
    for phase in PHASE_NAMES:
        phase_idx = np.where(labels == phase)[0]
        if phase_idx.size == 0:
            continue
        segments.append(
            {
                "phase": phase,
                "transition_start": int(phase_idx[0]),
                "transition_end": int(phase_idx[-1]),
                "transition_count": int(phase_idx.size),
            }
        )
    return {
        "status": "ok",
        "reason": "reference_delta_activity_split",
        "quiet_threshold": float(quiet_threshold),
        "active_threshold": float(active_threshold),
        "transition_labels": labels.tolist(),
        "segments": segments,
    }


def phase_specific_stats(baselines: dict[str, Any], task_index: int | None, phase: str, column: str, section: str = "phase_state_delta") -> dict[str, Any]:
    stats = baseline_lookup(baselines, section, task_index, f"phase:{phase}", column)
    if reference_stats_ready(stats):
        return stats
    fallback_section = "state_delta" if section == "phase_state_delta" else "state_accel"
    return baseline_lookup(baselines, fallback_section, task_index, column)


def metric_baseline_lookup(
    baselines: dict[str, Any],
    task_index: int | None,
    view: str,
    metric_name: str,
) -> dict[str, Any]:
    key_task = f"task:{task_index}|{view}" if task_index is not None else f"task:unknown|{view}"
    key_global = f"global|{view}"
    image_metrics = baselines.get("image_metrics", {})
    task_metric = image_metrics.get(key_task, {}).get(metric_name)
    if isinstance(task_metric, dict) and task_metric.get("count", 0) >= 5:
        return task_metric
    return image_metrics.get(key_global, {}).get(metric_name, {"count": 0})


def add_if_score(
    findings: list[Finding],
    factory: FindingFactory,
    score: float,
    threshold_score: float,
    **kwargs: Any,
) -> None:
    if score >= threshold_score:
        findings.append(factory.make(severity_score=score, **kwargs))


def inspect_episode(
    path: Path,
    info: dict[str, Any],
    expected_meta: dict[str, Any] | None,
    views: list[str],
    baselines: dict[str, Any],
    factory: FindingFactory,
) -> tuple[dict[str, Any], list[Finding]]:
    findings: list[Finding] = []
    expected_columns = expected_columns_from_info(info)
    expected_shapes = {
        name: tuple(spec.get("shape", []))
        for name, spec in info.get("features", {}).items()
    }

    try:
        df = read_parquet_frame(path)
    except Exception as exc:
        episode_from_path = safe_episode_from_path(path)
        findings.append(
            factory.make(
                "state_illegal",
                "parquet_read_error",
                "Parquet file cannot be read",
                "file",
                "file",
                100,
                illegal=True,
                episode_index=episode_from_path,
                evidence={"file": str(path), "error": str(exc)},
            )
        )
        return {"episode_index": episode_from_path, "task_index": None, "length": 0, "file": str(path)}, findings

    episode = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns and len(df) else safe_episode_from_path(path)
    task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns and len(df) else None
    length = int(len(df))
    frames = df["frame_index"].to_numpy(dtype=np.int64) if "frame_index" in df.columns and len(df) else np.arange(length)

    missing_columns = sorted(expected_columns - set(df.columns))
    if missing_columns:
        for column in missing_columns:
            key = "vision_illegal" if column in views else "state_illegal"
            findings.append(
                factory.make(
                    key,
                    "missing_required_column",
                    "Required column is missing",
                    "file",
                    "vision" if column in views else "state",
                    100,
                    illegal=True,
                    episode_index=episode,
                    task_index=task_index,
                    column=column,
                    evidence={"missing_column": column},
                )
            )

    if length == 0:
        findings.append(
            factory.make(
                "temporal_illegal",
                "empty_episode",
                "Episode has no frame",
                "episode",
                "all",
                100,
                illegal=True,
                episode_index=episode,
                task_index=task_index,
                evidence={"file": str(path)},
            )
        )
        return {
            "episode_index": episode,
            "task_index": task_index,
            "length": length,
            "file": str(path),
            "soft_penalties": {},
            "phase_detection": {"status": "unavailable", "reason": "empty_episode", "segments": []},
        }, findings

    if expected_meta is not None and length != int(expected_meta.get("length", length)):
        findings.append(
            factory.make(
                "temporal_illegal",
                "metadata_length_mismatch",
                "Episode row count differs from metadata length",
                "episode",
                "all",
                100,
                illegal=True,
                episode_index=episode,
                task_index=task_index,
                evidence={"expected_length": int(expected_meta.get("length", length)), "actual_length": length},
            )
        )

    inspect_indices(df, frames, episode, task_index, findings, factory, info)
    image_context = inspect_images(df, frames, episode, task_index, views, expected_shapes, baselines, findings, factory)
    state_context = inspect_low_dimensional(df, frames, episode, task_index, baselines, findings, factory)
    geometry_result = geometry_constraints.inspect_episode_geometry(
        df=df,
        frames=frames,
        episode=episode,
        task_index=task_index,
        views=views,
        image_context=image_context,
        reference=baselines.get("geometry_constraints", {}),
        config=baselines.get("geometry_config", geometry_constraints.default_geometry_config(info)),
        factory=factory,
    )
    findings.extend(geometry_result["findings"])
    soft_penalties = reference_soft_penalties(task_index, views, image_context, state_context, baselines)

    return {
        "episode_index": episode,
        "task_index": task_index,
        "length": length,
        "file": str(path),
        "soft_penalties": soft_penalties,
        "geometry_constraints": geometry_result["diagnostics"],
        "phase_detection": state_context.get("phase_detection", {"status": "unavailable", "reason": "state_not_processed", "segments": []}),
    }, findings


def inspect_indices(
    df: Any,
    frames: np.ndarray,
    episode: int | None,
    task_index: int | None,
    findings: list[Finding],
    factory: FindingFactory,
    info: dict[str, Any],
) -> None:
    for column in ["frame_index", "index", "episode_index", "task_index", "timestamp"]:
        if column not in df.columns:
            findings.append(
                factory.make(
                    "temporal_illegal",
                    "missing_temporal_column",
                    "Temporal/index column is missing",
                    "file",
                    "index",
                    100,
                    illegal=True,
                    episode_index=episode,
                    task_index=task_index,
                    column=column,
                    evidence={"missing_column": column},
                )
            )
            continue
        values = np.asarray(df[column])
        numeric = np.asarray(values, dtype=np.float64)
        non_finite = ~np.isfinite(numeric)
        for start, end in contiguous_ranges(frames[np.where(non_finite)[0]]):
            findings.append(
                factory.make(
                    "temporal_illegal",
                    f"non_finite_{column}",
                    "Index/timestamp column has non-finite values",
                    "frame",
                    "index",
                    100,
                    illegal=True,
                    episode_index=episode,
                    task_index=task_index,
                    column=column,
                    frame_start=start,
                    frame_end=end,
                    evidence={"column": column},
                )
            )

    if "frame_index" in df.columns and len(df) > 1:
        diffs = np.diff(frames)
        for idx in np.where(diffs != 1)[0]:
            issue = "duplicate_frame_index" if diffs[idx] == 0 else "frame_index_gap_or_reorder"
            findings.append(
                factory.make(
                    "temporal_illegal",
                    issue,
                    "Frame index is duplicated, skipped, or reordered",
                    "frame",
                    "index",
                    100,
                    illegal=True,
                    episode_index=episode,
                    task_index=task_index,
                    frame_start=int(frames[idx]),
                    frame_end=int(frames[idx + 1]),
                    evidence={"previous": int(frames[idx]), "current": int(frames[idx + 1]), "delta": int(diffs[idx])},
                )
            )

    if "index" in df.columns and len(df) > 1:
        values = df["index"].to_numpy(dtype=np.int64)
        diffs = np.diff(values)
        for idx in np.where(diffs != 1)[0]:
            findings.append(
                factory.make(
                    "temporal_illegal",
                    "global_index_gap_or_reorder",
                    "Global index is duplicated, skipped, or reordered",
                    "frame",
                    "index",
                    100,
                    illegal=True,
                    episode_index=episode,
                    task_index=task_index,
                    frame_start=int(frames[idx]),
                    frame_end=int(frames[idx + 1]),
                    evidence={"previous": int(values[idx]), "current": int(values[idx + 1]), "delta": int(diffs[idx])},
                )
            )

    if "timestamp" in df.columns and len(df) > 1:
        timestamp = df["timestamp"].to_numpy(dtype=np.float64)
        diffs = np.diff(timestamp)
        fps = float(info.get("fps") or 0)
        expected_delta = 1.0 / fps if fps > 0 else float(np.nanmedian(diffs))
        tolerance = max(0.02, expected_delta * 0.35)
        bad = np.where((diffs <= 0) | (np.abs(diffs - expected_delta) > tolerance))[0]
        for idx in bad:
            findings.append(
                factory.make(
                    "temporal_illegal",
                    "timestamp_gap_or_reorder",
                    "Timestamp is non-monotonic or deviates from expected FPS",
                    "frame",
                    "timestamp",
                    100,
                    illegal=True,
                    episode_index=episode,
                    task_index=task_index,
                    frame_start=int(frames[idx]),
                    frame_end=int(frames[idx + 1]),
                    evidence={
                        "previous_timestamp": float(timestamp[idx]),
                        "current_timestamp": float(timestamp[idx + 1]),
                        "delta": float(diffs[idx]),
                        "expected_delta": expected_delta,
                        "tolerance": tolerance,
                    },
                )
            )

    if "episode_index" in df.columns and episode is not None and not (df["episode_index"] == episode).all():
        findings.append(
            factory.make(
                "temporal_illegal",
                "episode_index_inconsistent",
                "Episode index changes inside one parquet episode",
                "episode",
                "index",
                100,
                illegal=True,
                episode_index=episode,
                task_index=task_index,
            )
        )

    if "task_index" in df.columns and df["task_index"].nunique(dropna=False) != 1:
        findings.append(
            factory.make(
                "temporal_illegal",
                "task_index_inconsistent",
                "Task index changes inside one parquet episode",
                "episode",
                "index",
                100,
                illegal=True,
                episode_index=episode,
                task_index=task_index,
            )
        )


def inspect_images(
    df: Any,
    frames: np.ndarray,
    episode: int | None,
    task_index: int | None,
    views: list[str],
    expected_shapes: dict[str, tuple[int, ...]],
    baselines: dict[str, Any],
    findings: list[Finding],
    factory: FindingFactory,
) -> dict[str, dict[str, Any]]:
    context: dict[str, dict[str, Any]] = {}
    for view in views:
        if view not in df.columns:
            continue
        metrics_by_frame: list[dict[str, float] | None] = []
        grays: list[np.ndarray | None] = []
        hashes: list[str] = []
        motions: list[float] = []
        previous_gray: np.ndarray | None = None

        for idx, value in enumerate(df[view]):
            frame = int(frames[idx])
            try:
                image = decode_image(value)
                metric = image_metrics(image)
                metrics_by_frame.append(metric)
                grays.append(downsample_gray(image))
                hashes.append(hashlib.sha1(image.tobytes()).hexdigest())
                if expected_shapes.get(view) and tuple(image.shape) != expected_shapes[view]:
                    findings.append(
                        factory.make(
                            "vision_illegal",
                            "invalid_image_shape",
                            "Image shape does not match metadata",
                            "frame",
                            "vision",
                            100,
                            illegal=True,
                            episode_index=episode,
                            task_index=task_index,
                            view=view,
                            frame_start=frame,
                            frame_end=frame,
                            evidence={"actual_shape": list(image.shape), "expected_shape": list(expected_shapes[view])},
                        )
                    )
                if previous_gray is not None:
                    motions.append(float(np.abs(grays[-1] - previous_gray).mean()))
                previous_gray = grays[-1]
            except Exception as exc:
                metrics_by_frame.append(None)
                grays.append(None)
                hashes.append("")
                previous_gray = None
                findings.append(
                    factory.make(
                        "vision_illegal",
                        "image_decode_error",
                        "Image cell cannot be decoded",
                        "frame",
                        "vision",
                        100,
                        illegal=True,
                        episode_index=episode,
                        task_index=task_index,
                        view=view,
                        frame_start=frame,
                        frame_end=frame,
                        evidence={"error": str(exc)},
                    )
                )

        for idx, metric in enumerate(metrics_by_frame):
            if metric is None:
                continue
            frame = int(frames[idx])
            metric_scores: dict[str, float] = {}
            metric_scores["black"] = 1.0 if metric["black_fraction"] > 0.98 or (metric["mean"] < 5 and metric["std"] < 5) else 0.0
            metric_scores["white"] = 1.0 if metric["white_fraction"] > 0.98 or (metric["mean"] > 250 and metric["std"] < 5) else 0.0
            metric_scores["under_exposed"] = max(
                normalized_score(20.0 - metric["mean"], 0.0, 15.0),
                lower_outlier_score(metric["mean"], metric_baseline_lookup(baselines, task_index, view, "mean")),
            )
            metric_scores["over_exposed"] = max(
                normalized_score(metric["mean"] - 235.0, 0.0, 15.0),
                upper_outlier_score(metric["mean"], metric_baseline_lookup(baselines, task_index, view, "mean")),
            )
            metric_scores["low_contrast"] = max(
                normalized_score(12.0 - metric["std"], 0.0, 10.0),
                lower_outlier_score(metric["std"], metric_baseline_lookup(baselines, task_index, view, "std")),
            )
            metric_scores["blur"] = max(
                normalized_score(50.0 - metric["sharpness"], 0.0, 40.0),
                lower_outlier_score(metric["sharpness"], metric_baseline_lookup(baselines, task_index, view, "sharpness")),
            )
            metric_scores["low_entropy"] = max(
                normalized_score(2.5 - metric["entropy"], 0.0, 1.5),
                lower_outlier_score(metric["entropy"], metric_baseline_lookup(baselines, task_index, view, "entropy")),
            )
            issue_score = 100.0 * max(metric_scores.values())
            if issue_score >= 30:
                issue_type = max(metric_scores, key=metric_scores.get)
                findings.append(
                    factory.make(
                        "vision_single",
                        issue_type,
                        "Single image quality is abnormal",
                        "frame",
                        "vision",
                        issue_score,
                        episode_index=episode,
                        task_index=task_index,
                        view=view,
                        frame_start=frame,
                        frame_end=frame,
                        evidence={"metrics": metric, "component_scores": metric_scores},
                    )
                )

        motion_arr = np.asarray(motions, dtype=np.float64)
        context[view] = {
            "metrics": metrics_by_frame,
            "hashes": hashes,
            "motion": motion_arr,
            "valid_frames": int(sum(item is not None for item in metrics_by_frame)),
        }
        inspect_vision_temporal(frames, episode, task_index, view, motion_arr, hashes, baselines, findings, factory)
    return context


def inspect_vision_temporal(
    frames: np.ndarray,
    episode: int | None,
    task_index: int | None,
    view: str,
    motion: np.ndarray,
    hashes: list[str],
    baselines: dict[str, Any],
    findings: list[Finding],
    factory: FindingFactory,
) -> None:
    start = 0
    while start < len(hashes):
        end = start
        while end + 1 < len(hashes) and hashes[start] and hashes[end + 1] == hashes[start]:
            end += 1
        run_len = end - start + 1
        if hashes[start] and run_len >= MIN_FREEZE_FRAMES:
            severity = freeze_duration_severity(run_len, exact=True)
            if severity < MIN_FREEZE_REPORT_SEVERITY:
                start = end + 1
                continue
            findings.append(
                factory.make(
                    "vision_temporal",
                    "exact_frozen_image_run",
                    "Exact repeated visual frames",
                    "segment",
                    "vision",
                    severity,
                    episode_index=episode,
                    task_index=task_index,
                    view=view,
                    frame_start=int(frames[start]),
                    frame_end=int(frames[end]),
                    evidence={
                        "run_length": run_len,
                        "grace_frames": FREEZE_GRACE_FRAMES,
                        "mild_frames": FREEZE_MILD_FRAMES,
                        "severe_frames": FREEZE_SEVERE_FRAMES,
                    },
                )
            )
        start = end + 1

    if motion.size == 0:
        return
    motion_stats = baseline_lookup(baselines, "image_motion", task_index, view)
    accel_stats = baseline_lookup(baselines, "image_accel", task_index, view)
    low_threshold = max(0.2, stat_value(motion_stats, "p01", 0.2) * 0.5)
    high_threshold = max(
        stat_value(motion_stats, "p99", float(np.max(motion))) * 1.5,
        stat_value(motion_stats, "median", 0.0) + 8.0 * stat_value(motion_stats, "mad", 0.0),
        stat_value(motion_stats, "max", float(np.max(motion))) * REFERENCE_ENVELOPE_MARGIN,
    )

    near_static = np.where(motion <= low_threshold)[0]
    for start, end in contiguous_ranges(near_static, max_gap=1):
        run_len = end - start + 1
        if run_len >= MIN_FREEZE_FRAMES:
            severity = freeze_duration_severity(run_len)
            if severity < MIN_FREEZE_REPORT_SEVERITY:
                continue
            findings.append(
                factory.make(
                    "vision_temporal",
                    "low_motion_freeze_run",
                    "Visual motion is near zero for a long window",
                    "segment",
                    "vision",
                    severity,
                    episode_index=episode,
                    task_index=task_index,
                    view=view,
                    frame_start=int(frames[start]),
                    frame_end=int(frames[min(end + 1, len(frames) - 1)]),
                    evidence={
                        "run_length": run_len,
                        "low_motion_threshold": low_threshold,
                        "median_motion": float(np.median(motion[start : end + 1])),
                        "grace_frames": FREEZE_GRACE_FRAMES,
                        "mild_frames": FREEZE_MILD_FRAMES,
                        "severe_frames": FREEZE_SEVERE_FRAMES,
                    },
                )
            )

    fast = np.where(motion >= high_threshold)[0]
    for start, end in contiguous_ranges(fast, max_gap=1):
        max_motion = float(np.max(motion[start : end + 1]))
        score = 100.0 * max(0.35, upper_outlier_score(max_motion, motion_stats, weak_z=4.0, strong_z=10.0))
        findings.append(
            factory.make(
                "vision_temporal",
                "visual_fast_jump",
                "Visual sequence has an extreme fast jump",
                "segment",
                "vision",
                score,
                episode_index=episode,
                task_index=task_index,
                view=view,
                frame_start=int(frames[start]),
                frame_end=int(frames[min(end + 1, len(frames) - 1)]),
                evidence={"max_motion": max_motion, "high_motion_threshold": high_threshold},
            )
        )

    if motion.size > 2:
        accel = np.abs(np.diff(motion))
        accel_threshold = max(
            stat_value(accel_stats, "p99", float(np.max(accel))) * VISION_JITTER_P99_MULTIPLIER,
            stat_value(accel_stats, "median", 0.0) + VISION_JITTER_MAD_MULTIPLIER * stat_value(accel_stats, "mad", 0.0),
            stat_value(accel_stats, "max", float(np.max(accel))) * REFERENCE_ENVELOPE_MARGIN,
        )
        jitter = np.where(accel >= accel_threshold)[0]
        for start, end in contiguous_ranges(jitter, max_gap=2):
            if end - start + 1 >= 2:
                score = min(85.0, 35.0 + 10.0 * (end - start + 1))
                findings.append(
                    factory.make(
                        "vision_temporal",
                        "visual_high_frequency_jitter",
                        "Visual motion has high-frequency acceleration spikes",
                        "segment",
                        "vision",
                        score,
                        episode_index=episode,
                        task_index=task_index,
                        view=view,
                        frame_start=int(frames[start]),
                        frame_end=int(frames[min(end + 2, len(frames) - 1)]),
                        evidence={"max_motion_accel": float(np.max(accel[start : end + 1])), "accel_threshold": accel_threshold},
                    )
                )


def inspect_low_dimensional(
    df: Any,
    frames: np.ndarray,
    episode: int | None,
    task_index: int | None,
    baselines: dict[str, Any],
    findings: list[Finding],
    factory: FindingFactory,
) -> dict[str, dict[str, Any]]:
    context: dict[str, dict[str, Any]] = {}
    phase_detection: dict[str, Any] = {"status": "unavailable", "reason": "state_not_processed"}
    for column in LOW_DIM_COLUMNS:
        if column not in df.columns:
            continue
        matrix = to_float_matrix(df[column])
        if matrix is None:
            findings.append(
                factory.make(
                    "state_illegal",
                    "inconsistent_low_dim_shape",
                    "Low-dimensional row shape is inconsistent",
                    "episode",
                    "state",
                    100,
                    illegal=True,
                    episode_index=episode,
                    task_index=task_index,
                    column=column,
                )
            )
            continue
        expected_shape = baselines.get("expected_low_dim_shape", {}).get(column)
        if expected_shape is None:
            expected_shape = 20
        if matrix.ndim != 2 or matrix.shape[1] != expected_shape:
            findings.append(
                factory.make(
                    "state_illegal",
                    "invalid_low_dim_shape",
                    "Low-dimensional shape does not match expected schema",
                    "episode",
                    "state",
                    100,
                    illegal=True,
                    episode_index=episode,
                    task_index=task_index,
                    column=column,
                    evidence={"actual_shape": list(matrix.shape), "expected_width": expected_shape},
                )
            )
        non_finite = np.where(~np.isfinite(matrix).all(axis=1))[0]
        for start_idx, end_idx in contiguous_ranges(frames[non_finite]):
            findings.append(
                factory.make(
                    "state_illegal",
                    "non_finite_low_dim_value",
                    "Low-dimensional value contains NaN or Inf",
                    "frame",
                    "state",
                    100,
                    illegal=True,
                    episode_index=episode,
                    task_index=task_index,
                    column=column,
                    frame_start=start_idx,
                    frame_end=end_idx,
                )
            )
        impossible = np.where(np.isfinite(matrix).all(axis=1) & (np.max(np.abs(matrix), axis=1) > 1e6))[0]
        for start_idx, end_idx in contiguous_ranges(frames[impossible]):
            findings.append(
                factory.make(
                    "state_illegal",
                    "impossible_low_dim_magnitude",
                    "Low-dimensional value exceeds impossible magnitude guardrail",
                    "frame",
                    "state",
                    100,
                    illegal=True,
                    episode_index=episode,
                    task_index=task_index,
                    column=column,
                    frame_start=start_idx,
                    frame_end=end_idx,
                    evidence={"guardrail_abs_max": 1e6},
                )
            )
        if column in CONSISTENCY_LOW_DIM_COLUMNS:
            value_stats = baseline_lookup(baselines, "state_values", task_index, column)
            row_abs = np.max(np.abs(matrix), axis=1) if matrix.size else np.array([])
            row_abs_stats = baseline_lookup(baselines, "state_row_abs", task_index, column)
            for idx, row_max in enumerate(row_abs):
                if not np.isfinite(row_max):
                    continue
                score = max(
                    100.0 * upper_outlier_score(float(row_max), row_abs_stats, weak_z=6.0, strong_z=12.0),
                    100.0 * two_sided_outlier_score(float(np.median(matrix[idx])), value_stats, weak_z=6.0, strong_z=12.0),
                )
                if score >= 35:
                    findings.append(
                        factory.make(
                            "state_vision_state",
                            "low_dim_value_reference_outlier",
                            "Low-dimensional value deviates from reference distribution",
                            "frame",
                            "state",
                            score,
                            ood=score < 55,
                            episode_index=episode,
                            task_index=task_index,
                            column=column,
                            frame_start=int(frames[idx]),
                            frame_end=int(frames[idx]),
                            evidence={"row_abs_max": float(row_max), "value_baseline": value_stats},
                        )
                    )
        delta = np.linalg.norm(np.diff(matrix, axis=0), axis=1) if len(matrix) > 1 else np.array([])
        accel = np.linalg.norm(matrix[2:] - 2.0 * matrix[1:-1] + matrix[:-2], axis=1) if len(matrix) > 2 else np.array([])
        if column == "state":
            phase_detection = phase_transition_labels(delta, baseline_lookup(baselines, "state_delta", task_index, column))
        context[column] = {
            "matrix": matrix,
            "delta": delta,
            "accel": accel,
            "phase_detection": phase_detection if column == "state" else {"status": "unavailable", "reason": "not_state_column"},
        }
        inspect_state_temporal(
            frames,
            episode,
            task_index,
            column,
            delta,
            accel,
            baselines,
            findings,
            factory,
            phase_detection=phase_detection if column == "state" else None,
        )
    context["phase_detection"] = phase_detection
    return context


def inspect_state_temporal(
    frames: np.ndarray,
    episode: int | None,
    task_index: int | None,
    column: str,
    delta: np.ndarray,
    accel: np.ndarray,
    baselines: dict[str, Any],
    findings: list[Finding],
    factory: FindingFactory,
    phase_detection: dict[str, Any] | None = None,
) -> None:
    if delta.size == 0:
        return
    phase_labels = np.asarray(dict(phase_detection or {}).get("transition_labels", []), dtype=object)
    if phase_labels.size != delta.size or not np.isin(phase_labels, PHASE_NAMES).any():
        phase_labels = np.full(delta.shape, "active", dtype=object)
    for phase in PHASE_NAMES:
        phase_mask = phase_labels == phase
        if not phase_mask.any():
            continue
        phase_delta_stats = phase_specific_stats(baselines, task_index, phase, column, section="phase_state_delta")
        phase_accel_stats = phase_specific_stats(baselines, task_index, phase, column, section="phase_state_accel")
        low_threshold = reference_lower_threshold(phase_delta_stats, key="p05", multiplier=0.5)
        if low_threshold is None:
            low_threshold = reference_lower_threshold(baseline_lookup(baselines, "state_delta", task_index, column), key="p01", multiplier=0.5) or 1e-8
        high_threshold = reference_upper_threshold(phase_delta_stats, key="p99", quantile_multiplier=1.5, mad_multiplier=8.0)
        frozen = np.where(phase_mask & (delta <= low_threshold))[0]
        for start_idx, end_idx in contiguous_ranges(frozen, max_gap=1):
            run_len = end_idx - start_idx + 1
            if run_len >= MIN_FREEZE_FRAMES:
                severity = freeze_duration_severity(run_len)
                if severity < MIN_FREEZE_REPORT_SEVERITY:
                    continue
                findings.append(
                    factory.make(
                        "state_temporal",
                        f"low_dim_freeze_run_{phase}",
                        "Low-dimensional state is nearly unchanged for a long window",
                        "segment",
                        "state",
                        severity,
                        episode_index=episode,
                        task_index=task_index,
                        column=column,
                        frame_start=int(frames[start_idx]),
                        frame_end=int(frames[min(end_idx + 1, len(frames) - 1)]),
                        evidence={
                            "phase": phase,
                            "run_length": run_len,
                            "low_delta_threshold": low_threshold,
                            "median_delta": float(np.median(delta[start_idx : end_idx + 1])),
                            "grace_frames": FREEZE_GRACE_FRAMES,
                            "mild_frames": FREEZE_MILD_FRAMES,
                            "severe_frames": FREEZE_SEVERE_FRAMES,
                            "reference_count": phase_delta_stats.get("count", 0),
                        },
                    )
                )
        if high_threshold is not None:
            fast = np.where(phase_mask & (delta >= high_threshold))[0]
            for start_idx, end_idx in contiguous_ranges(fast, max_gap=1):
                max_delta = float(np.max(delta[start_idx : end_idx + 1]))
                score = max(35.0, 100.0 * upper_outlier_score(max_delta, phase_delta_stats, weak_z=4.0, strong_z=10.0))
                findings.append(
                    factory.make(
                        "state_temporal",
                        f"low_dim_fast_jump_{phase}",
                        "Low-dimensional state has an extreme fast jump",
                        "segment",
                        "state",
                        score,
                        episode_index=episode,
                        task_index=task_index,
                        column=column,
                        frame_start=int(frames[start_idx]),
                        frame_end=int(frames[min(end_idx + 1, len(frames) - 1)]),
                        evidence={
                            "phase": phase,
                            "max_delta": max_delta,
                            "high_delta_threshold": high_threshold,
                            "reference_count": phase_delta_stats.get("count", 0),
                        },
                    )
                )
        if accel.size:
            accel_phase_mask = phase_mask[1:] if len(phase_mask) > 1 else np.array([], dtype=bool)
            accel_threshold = reference_upper_threshold(
                phase_accel_stats,
                key="p99",
                quantile_multiplier=STATE_JITTER_P99_MULTIPLIER,
                mad_multiplier=STATE_JITTER_MAD_MULTIPLIER,
            )
            if accel_threshold is not None and accel_phase_mask.size:
                spikes = np.where(accel_phase_mask & (accel >= accel_threshold))[0]
                for start_idx, end_idx in contiguous_ranges(spikes, max_gap=STATE_JITTER_CLUSTER_GAP):
                    max_accel = float(np.max(accel[start_idx : end_idx + 1]))
                    score = max(35.0, 100.0 * upper_outlier_score(max_accel, phase_accel_stats, weak_z=4.0, strong_z=10.0))
                    cluster_len = end_idx - start_idx + 1
                    if cluster_len < STATE_JITTER_MIN_CLUSTER and score < STATE_JITTER_SINGLE_SPIKE_MIN_SCORE:
                        continue
                    findings.append(
                        factory.make(
                            "state_temporal",
                            f"low_dim_jitter_or_spike_{phase}",
                            "Low-dimensional state acceleration is an extreme outlier",
                            "segment",
                            "state",
                            score,
                            episode_index=episode,
                            task_index=task_index,
                            column=column,
                            frame_start=int(frames[start_idx + 1]),
                            frame_end=int(frames[min(end_idx + 1, len(frames) - 1)]),
                            evidence={
                                "phase": phase,
                                "max_accel": max_accel,
                                "accel_threshold": accel_threshold,
                                "cluster_len": cluster_len,
                                "min_cluster_len": STATE_JITTER_MIN_CLUSTER,
                                "single_spike_min_score": STATE_JITTER_SINGLE_SPIKE_MIN_SCORE,
                                "reference_count": phase_accel_stats.get("count", 0),
                            },
                        )
                    )


def capped_soft_penalty(values: list[float], cap: float) -> float:
    finite = [float(value) for value in values if np.isfinite(value) and value > 0]
    if not finite:
        return 0.0
    top = sorted(finite, reverse=True)[:5]
    return round(min(cap, cap * float(np.mean(top))), 3)


def reference_soft_penalties(
    task_index: int | None,
    views: list[str],
    image_context: dict[str, dict[str, Any]],
    state_context: dict[str, dict[str, Any]],
    baselines: dict[str, Any],
) -> dict[str, float]:
    pressures: dict[str, list[float]] = defaultdict(list)

    for view in views:
        ctx = image_context.get(view, {})
        metrics = [item for item in ctx.get("metrics", []) if item is not None]
        if metrics:
            for metric_name in ["mean", "std", "sharpness", "entropy"]:
                values = np.asarray([item[metric_name] for item in metrics], dtype=np.float64)
                stats = metric_baseline_lookup(baselines, task_index, view, metric_name)
                if metric_name == "mean":
                    pressures["vision_single"].append(two_sided_tail_pressure(float(np.median(values)), stats))
                else:
                    pressures["vision_single"].append(lower_tail_pressure(float(np.quantile(values, 0.05)), stats))
            for metric_name in ["black_fraction", "white_fraction"]:
                values = np.asarray([item[metric_name] for item in metrics], dtype=np.float64)
                stats = metric_baseline_lookup(baselines, task_index, view, metric_name)
                pressures["vision_single"].append(upper_tail_pressure(float(np.quantile(values, 0.95)), stats))

        motion = np.asarray(ctx.get("motion", []), dtype=np.float64)
        if motion.size:
            motion_stats = baseline_lookup(baselines, "image_motion", task_index, view)
            pressures["temporal"].append(upper_tail_pressure(float(np.max(motion)), motion_stats, start_key="p99") * 0.5)
            near_static = np.where(motion <= max(0.2, stat_value(motion_stats, "p01", 0.2) * 0.5))[0]
            if near_static.size:
                max_run = max((end - start + 1 for start, end in contiguous_ranges(near_static, max_gap=1)), default=0)
                pressures["temporal"].append(min(1.0, freeze_duration_severity(max_run) / max(MIN_FREEZE_REPORT_SEVERITY, 1e-9)))
            if motion.size > 2:
                accel = np.abs(np.diff(motion))
                accel_stats = baseline_lookup(baselines, "image_accel", task_index, view)
                pressures["temporal"].append(upper_tail_pressure(float(np.max(accel)), accel_stats, start_key="p99") * 0.75)

    for column in CONSISTENCY_LOW_DIM_COLUMNS:
        ctx = state_context.get(column, {})
        delta = np.asarray(ctx.get("delta", []), dtype=np.float64)
        if delta.size:
            delta_stats = baseline_lookup(baselines, "state_delta", task_index, column)
            pressures["temporal"].append(upper_tail_pressure(float(np.max(delta)), delta_stats, start_key="p99") * 0.5)
            frozen = np.where(delta <= max(1e-8, stat_value(delta_stats, "p01", 1e-8) * 0.5))[0]
            if frozen.size:
                max_run = max((end - start + 1 for start, end in contiguous_ranges(frozen, max_gap=1)), default=0)
                pressures["temporal"].append(min(1.0, freeze_duration_severity(max_run) / max(MIN_FREEZE_REPORT_SEVERITY, 1e-9)))
        accel = np.asarray(ctx.get("accel", []), dtype=np.float64)
        if accel.size:
            accel_stats = baseline_lookup(baselines, "state_accel", task_index, column)
            pressures["temporal"].append(upper_tail_pressure(float(np.max(accel)), accel_stats, start_key="p99") * 0.75)


    return {
        dimension: capped_soft_penalty(values, REFERENCE_SOFT_DIM_CAPS[dimension])
        for dimension, values in pressures.items()
        if dimension in REFERENCE_SOFT_DIM_CAPS
    }


def merge_findings(findings: list[Finding]) -> list[Finding]:
    groups: dict[tuple[Any, ...], list[Finding]] = defaultdict(list)
    for finding in findings:
        key = (
            finding.episode_index,
            finding.category_id,
            finding.issue_type,
            finding.modality,
            finding.view,
            finding.column,
        )
        groups[key].append(finding)

    merged: list[Finding] = []
    for _, items in groups.items():
        items = sorted(items, key=lambda item: (-1 if item.frame_start is None else item.frame_start, item.finding_id))
        current: Finding | None = None
        for item in items:
            if current is None:
                current = item
                continue
            can_merge = (
                current.frame_start is not None
                and current.frame_end is not None
                and item.frame_start is not None
                and item.frame_start <= current.frame_end + 2
            )
            if can_merge:
                current.frame_end = max(current.frame_end, item.frame_end if item.frame_end is not None else item.frame_start)
                current.severity_score = round(max(current.severity_score, item.severity_score), 2)
                current.quality_penalty = round(min(
                    DIMENSION_POINTS.get(CATEGORY_TO_DIMENSION.get(current.category_id, "state"), 10.0),
                    current.quality_penalty + item.quality_penalty * 0.35,
                ), 3)
                current.merged_count += item.merged_count
                current.evidence.setdefault("merged_evidence", []).append(item.evidence)
                if current.confidence_level != "确定异常" and item.confidence_level == "高置信异常":
                    current.confidence_level = "高置信异常"
            else:
                merged.append(current)
                current = item
        if current is not None:
            merged.append(current)
    return sorted(merged, key=lambda item: (item.episode_index if item.episode_index is not None else -1, item.category_id, item.frame_start or -1, item.finding_id))


def score_reasons(score: float, max_points: float, reasons: list[str]) -> list[str]:
    if score >= max_points - 1e-6 and not reasons:
        return ["full_credit"]
    return reasons or ["minor_reference_penalties"]


def module_status_penalty(status: str, fail: float = 1.5, unavailable: float = 1.0, warning: float = 0.4) -> float:
    if status == "fail":
        return fail
    if status == "unavailable":
        return unavailable
    if status == "warning":
        return warning
    return 0.0


def iter_geometry_statuses(value: Any) -> list[str]:
    statuses: list[str] = []
    if isinstance(value, dict):
        for nested in value.values():
            statuses.extend(iter_geometry_statuses(nested))
    else:
        statuses.append(geometry_module_status(value))
    return statuses


def compute_data_value_score(meta: dict[str, Any], findings: list[Finding], geometry: dict[str, Any]) -> tuple[float, list[str]]:
    score = OFFICIAL_SCORE_POINTS["data_value"]
    reasons: list[str] = []
    length = int(meta.get("length", 0) or 0)
    if length <= 0:
        return 0.0, ["empty_episode_has_no_training_value"]
    if length < 32:
        deduction = 3.0
        score -= deduction
        reasons.append(f"short_episode:-{deduction:g}")
    if meta.get("task_index") is None:
        deduction = 2.0
        score -= deduction
        reasons.append(f"missing_task_index:-{deduction:g}")

    missing_state = {item.column for item in findings if item.issue_type == "missing_required_column" and item.modality == "state"}
    if missing_state:
        deduction = min(5.0, 2.5 * len(missing_state))
        score -= deduction
        reasons.append(f"missing_low_dim_columns:{','.join(sorted(str(item) for item in missing_state))}:-{deduction:g}")
    missing_views = {item.column for item in findings if item.issue_type == "missing_required_column" and item.modality == "vision"}
    if missing_views:
        deduction = min(4.0, 1.5 * len(missing_views))
        score -= deduction
        reasons.append(f"missing_vision_columns:{','.join(sorted(str(item) for item in missing_views))}:-{deduction:g}")

    phase = dict(meta.get("phase_detection", {}))
    phase_status = str(phase.get("status") or "unavailable")
    if phase_status != "ok":
        deduction = 2.5
        score -= deduction
        reasons.append(f"phase_detection_{phase_status}:{phase.get('reason', '')}:-{deduction:g}")
    elif len(phase.get("segments", [])) < 3:
        deduction = 1.0
        score -= deduction
        reasons.append(f"incomplete_phase_segments:-{deduction:g}")

    module_statuses = iter_geometry_statuses(geometry.get("module_statuses", {}))
    unavailable_modules = sum(1 for status in module_statuses if status == "unavailable")
    failed_modules = sum(1 for status in module_statuses if status == "fail")
    if unavailable_modules or failed_modules:
        deduction = min(4.0, unavailable_modules * 0.75 + failed_modules * 1.25)
        score -= deduction
        reasons.append(f"geometry_submodules_degraded:unavailable={unavailable_modules},fail={failed_modules}:-{deduction:g}")

    severe_findings = sum(1 for item in findings if item.confidence_level in {"确定异常", "高置信异常"})
    if severe_findings:
        deduction = min(3.0, severe_findings * 0.25)
        score -= deduction
        reasons.append(f"severe_finding_density:{severe_findings}:-{deduction:g}")

    final = round(max(0.0, min(OFFICIAL_SCORE_POINTS["data_value"], score)), 2)
    return final, score_reasons(final, OFFICIAL_SCORE_POINTS["data_value"], reasons)


def compute_engineering_reliability_score(meta: dict[str, Any], findings: list[Finding], geometry: dict[str, Any], soft_penalty_total: float) -> tuple[float, list[str]]:
    score = OFFICIAL_SCORE_POINTS["engineering_reliability"]
    reasons: list[str] = []
    geometry_status = geometry_module_status(geometry.get("status", "ok"))
    top_penalty = module_status_penalty(geometry_status, fail=3.0, unavailable=2.0, warning=1.0)
    if top_penalty:
        score -= top_penalty
        reasons.append(f"geometry_status_{geometry_status}:-{top_penalty:g}")

    module_penalty = min(4.0, sum(module_status_penalty(status) for status in iter_geometry_statuses(geometry.get("module_statuses", {}))))
    if module_penalty:
        score -= module_penalty
        reasons.append(f"module_status_penalty:-{module_penalty:g}")

    illegal_count = sum(1 for item in findings if item.illegal)
    if illegal_count:
        deduction = min(2.0, illegal_count * 0.35)
        score -= deduction
        reasons.append(f"illegal_findings:{illegal_count}:-{deduction:g}")

    phase = dict(meta.get("phase_detection", {}))
    if str(phase.get("status") or "unavailable") == "unavailable":
        deduction = 1.0
        score -= deduction
        reasons.append(f"phase_unavailable:{phase.get('reason', '')}:-{deduction:g}")

    if soft_penalty_total > 0:
        deduction = min(1.5, soft_penalty_total * 0.25)
        score -= deduction
        reasons.append(f"reference_soft_penalties:{soft_penalty_total:g}:-{deduction:g}")

    final = round(max(0.0, min(OFFICIAL_SCORE_POINTS["engineering_reliability"], score)), 2)
    return final, score_reasons(final, OFFICIAL_SCORE_POINTS["engineering_reliability"], reasons)


def score_episode(meta: dict[str, Any], findings: list[Finding]) -> EpisodeResult:
    episode = meta["episode_index"]
    current = [finding for finding in findings if finding.episode_index == episode]
    penalties_by_dim: dict[str, float] = defaultdict(float)
    for finding in current:
        dimension = CATEGORY_TO_DIMENSION.get(finding.category_id, "state")
        penalties_by_dim[dimension] += finding.quality_penalty
    soft_penalties = {
        dimension: round(min(REFERENCE_SOFT_DIM_CAPS.get(dimension, 0.0), float(value)), 3)
        for dimension, value in dict(meta.get("soft_penalties", {})).items()
        if dimension in DIMENSION_POINTS
    }
    for dimension, value in soft_penalties.items():
        penalties_by_dim[dimension] += value
    scores = {
        dimension: round(max(0.0, points - min(points, penalties_by_dim.get(dimension, 0.0))), 2)
        for dimension, points in DIMENSION_POINTS.items()
    }
    legacy_total = round(sum(scores.values()), 2)
    detection_quality = round(OFFICIAL_SCORE_POINTS["detection_quality"] * legacy_total / 100.0, 2)
    geometry = dict(meta.get("geometry_constraints", {}))
    soft_penalty_total = round(float(sum(soft_penalties.values())), 3)
    data_value, data_value_reasons = compute_data_value_score(meta, current, geometry)
    engineering_reliability, engineering_reliability_reasons = compute_engineering_reliability_score(
        meta,
        current,
        geometry,
        soft_penalty_total,
    )
    official_total = round(min(100.0, detection_quality + data_value + engineering_reliability), 2)
    phase = dict(meta.get("phase_detection", {}))
    return EpisodeResult(
        episode_index=int(episode),
        task_index=meta.get("task_index"),
        length=int(meta.get("length", 0)),
        file=str(meta.get("file", "")),
        score_total=official_total,
        legacy_score_total=legacy_total,
        score_detection_quality=detection_quality,
        score_data_value=data_value,
        score_engineering_reliability=engineering_reliability,
        scoring_version=OFFICIAL_SCORING_VERSION,
        data_value_reasons=data_value_reasons,
        engineering_reliability_reasons=engineering_reliability_reasons,
        score_structural=scores["structural"],
        score_vision_single=scores["vision_single"],
        score_vision_vision=scores["vision_vision"],
        score_state=scores["state"],
        score_temporal=scores["temporal"],
        score_cross_modal=scores["cross_modal"],
        soft_penalty_total=soft_penalty_total,
        soft_penalties=soft_penalties,
        finding_count=len(current),
        critical_count=sum(1 for item in current if item.confidence_level == "确定异常"),
        high_confidence_count=sum(1 for item in current if item.confidence_level == "高置信异常"),
        suspicious_count=sum(1 for item in current if item.confidence_level == "疑似异常"),
        ood_count=sum(1 for item in current if item.confidence_level == "分布外样本"),
        geometry_status=geometry_module_status(geometry.get("status", "ok")),
        geometry_reason=geometry.get("reason"),
        geometry_module_statuses=dict(geometry.get("module_statuses", {})),
        phase_status=str(phase.get("status") or "unavailable"),
        phase_reason=phase.get("reason"),
        phase_segments=list(phase.get("segments", [])),
    )

def write_reports(
    info: dict[str, Any],
    tasks: list[dict[str, Any]],
    baselines: dict[str, Any],
    findings: list[Finding],
    episode_results: list[EpisodeResult],
) -> None:
    findings_payload = [asdict(item) for item in findings]
    (DIAG_ROOT / "findings_v2.json").write_text(json.dumps(findings_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (DIAG_ROOT / "reference_baselines_v2.json").write_text(json.dumps(baselines, ensure_ascii=False, indent=2), encoding="utf-8")
    (DIAG_ROOT / "problem_standards_v2.json").write_text(json.dumps(STANDARDS, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [asdict(item) for item in episode_results]
    csv_path = REPORT_ROOT / "episode_scores_v2.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    by_issue = Counter(item.issue_type for item in findings)
    by_confidence = Counter(item.confidence_level for item in findings)
    by_category = Counter(f"{item.category_id} {item.category_name}" for item in findings)
    dataset_score = round(float(np.mean([item.score_total for item in episode_results])) if episode_results else 0.0, 2)
    legacy_dataset_score = round(float(np.mean([item.legacy_score_total for item in episode_results])) if episode_results else 0.0, 2)
    official_component_means = {
        "detection_quality": round(float(np.mean([item.score_detection_quality for item in episode_results])) if episode_results else 0.0, 2),
        "data_value": round(float(np.mean([item.score_data_value for item in episode_results])) if episode_results else 0.0, 2),
        "engineering_reliability": round(float(np.mean([item.score_engineering_reliability for item in episode_results])) if episode_results else 0.0, 2),
    }
    phase_status_counts = Counter(item.phase_status for item in episode_results)
    soft_penalty_mean = round(float(np.mean([item.soft_penalty_total for item in episode_results])) if episode_results else 0.0, 3)
    soft_penalty_max = round(max((item.soft_penalty_total for item in episode_results), default=0.0), 3)
    soft_penalty_by_dimension = {
        dimension: round(float(np.mean([item.soft_penalties.get(dimension, 0.0) for item in episode_results])), 3)
        for dimension in sorted(REFERENCE_SOFT_DIM_CAPS)
    }
    geometry_status_counts = Counter(item.geometry_status for item in episode_results)
    geometry_module_status_counts: dict[str, dict[str, int]] = {}
    for module_name, getter in GEOMETRY_REPORT_MODULES:
        geometry_module_status_counts[module_name] = normalized_status_counts(
            Counter(geometry_module_status(getter(item)) for item in episode_results)
        )

    report = {
        "version": "v2",
        "dataset": {
            "codebase_version": info.get("codebase_version"),
            "robot_type": info.get("robot_type"),
            "episodes": len(episode_results),
            "frames": int(info.get("total_frames", 0)),
            "fps": info.get("fps"),
            "tasks": tasks,
            "dataset_quality_score": dataset_score,
            "legacy_dataset_quality_score": legacy_dataset_score,
        },
        "scoring": {
            "version": OFFICIAL_SCORING_VERSION,
            "official_points": OFFICIAL_SCORE_POINTS,
            "official_formula": "score_total = detection_quality(70) + data_value(20) + engineering_reliability(10)",
            "detection_quality": "70 * legacy six-dimension detector score / 100; finding penalties remain calibrated by legacy dimensions.",
            "data_value": "Episode utility score from task metadata, modality completeness, phase segmentation availability, geometry-module availability, and severe-finding density.",
            "engineering_reliability": "Runtime/report reliability score from module status aggregation, illegal findings, phase availability, and reference soft penalties.",
            "legacy_dimension_points": LEGACY_DIMENSION_POINTS,
            "legacy_status": "deprecated_breakdown_only",
            "episode_score": "official 70/20/10 total; legacy six-dimension score is retained as legacy_score_total.",
            "finding_levels": ["确定异常", "高置信异常", "疑似异常", "分布外样本"],
            "merge_rule": "Findings with same episode/category/type/modality/view/column and overlapping or adjacent frame ranges are merged.",
        },
        "summary": {
            "finding_count": len(findings),
            "by_issue": dict(by_issue),
            "by_confidence": dict(by_confidence),
            "by_category": dict(by_category),
            "episode_score_min": min((item.score_total for item in episode_results), default=0.0),
            "episode_score_max": max((item.score_total for item in episode_results), default=0.0),
            "official_component_means": official_component_means,
            "legacy_dataset_quality_score": legacy_dataset_score,
            "phase_status_counts": normalized_status_counts(phase_status_counts),
            "soft_penalty_mean": soft_penalty_mean,
            "soft_penalty_max": soft_penalty_max,
            "soft_penalty_by_dimension": soft_penalty_by_dimension,
            "geometry_status_counts": normalized_status_counts(geometry_status_counts),
            "geometry_module_status_counts": geometry_module_status_counts,
        },
        "episodes": rows,
        "standards": STANDARDS,
    }
    json_path = REPORT_ROOT / "dataset_quality_report_v2.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_lines = [
        "# V2 多模态机器人数据质量检测报告",
        "",
        f"- 数据版本: {info.get('codebase_version')}",
        f"- 机器人类型: {info.get('robot_type')}",
        f"- Episode 数: {len(episode_results)}",
        f"- 总帧数: {info.get('total_frames')}",
        f"- FPS: {info.get('fps')}",
        f"- 数据集质量分: **{dataset_score}/100**",
        f"- 正式评分口径: `{OFFICIAL_SCORING_VERSION}` = 检测质量 {official_component_means['detection_quality']}/70 + 数据价值 {official_component_means['data_value']}/20 + 工程可靠性 {official_component_means['engineering_reliability']}/10",
        f"- 旧六维均分: {legacy_dataset_score}/100（仅保留为 legacy breakdown）",
        "",
        "## V2 检测框架",
        "",
        "本程序按“基础全覆盖、增强只选一个”的思路实现基础全覆盖部分，并将增强能力收敛为参考增强检测、跨模态一致性和报警归并。",
        "",
        "| 维度 | 满分 | 当前均分 |",
        "|---|---:|---:|",
    ]
    for dimension, points in DIMENSION_POINTS.items():
        field_name = f"score_{dimension}"
        mean_score = round(float(np.mean([getattr(item, field_name) for item in episode_results])) if episode_results else 0.0, 2)
        markdown_lines.append(f"| {dimension} | {points:g} | {mean_score} |")

    markdown_lines.extend([
        "",
        "## 模块状态",
        "",
        "状态枚举: `ok` 正常完成，`warning` 核心完成但存在降级或局部不可用，`unavailable` 核心或子模块无法运行，`fail` 模块执行失败。",
        "",
        "| 模块 | ok | warning | unavailable | fail |",
        "|---|---:|---:|---:|---:|",
    ])
    for module_name, counts in geometry_module_status_counts.items():
        markdown_lines.append(
            f"| {module_name} | {counts['ok']} | {counts['warning']} | {counts['unavailable']} | {counts['fail']} |"
        )

    markdown_lines.extend([
        "",
        "## 异常汇总",
        "",
        f"- 合并后报警数: {len(findings)}",
        f"- 确定异常: {by_confidence.get('确定异常', 0)}",
        f"- 高置信异常: {by_confidence.get('高置信异常', 0)}",
        f"- 疑似异常: {by_confidence.get('疑似异常', 0)}",
        f"- 分布外样本: {by_confidence.get('分布外样本', 0)}",
        "",
        "## 问题类型 Top 10",
        "",
        "| 问题类型 | 数量 |",
        "|---|---:|",
    ])
    for issue, count in by_issue.most_common(10):
        markdown_lines.append(f"| {issue} | {count} |")

    markdown_lines.extend([
        "",
        "## 输出文件",
        "",
        "- `outputs/v2/diagnostics/findings_v2.json`: 标准化异常明细",
        "- `outputs/v2/diagnostics/reference_baselines_v2.json`: 正常参考集阈值与基线",
        "- `outputs/v2/diagnostics/problem_standards_v2.json`: 问题定义与判定标准",
        "- `outputs/v2/reports/episode_scores_v2.csv`: episode 级评分表",
        "- `outputs/v2/reports/dataset_quality_report_v2.json`: 机器可读完整报告",
        "- `outputs/v2/reports/dataset_quality_report_v2.md`: 人类可读摘要报告",
    ])
    md_path = REPORT_ROOT / "dataset_quality_report_v2.md"
    md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    total_critical = sum(item.critical_count for item in episode_results)
    total_high_confidence = sum(item.high_confidence_count for item in episode_results)
    total_suspicious = sum(item.suspicious_count for item in episode_results)
    total_ood = sum(item.ood_count for item in episode_results)
    clean_markdown_lines = [
        "# V2 Multimodal Robot Data Quality Report",
        "",
        f"- dataset version: {info.get('codebase_version')}",
        f"- robot type: {info.get('robot_type')}",
        f"- episodes: {len(episode_results)}",
        f"- frames: {info.get('total_frames')}",
        f"- FPS: {info.get('fps')}",
        f"- dataset quality score: **{dataset_score}/100**",
        f"- official scoring: `{OFFICIAL_SCORING_VERSION}` = detection quality {official_component_means['detection_quality']}/70 + data value {official_component_means['data_value']}/20 + engineering reliability {official_component_means['engineering_reliability']}/10",
        f"- legacy six-dimension mean: {legacy_dataset_score}/100 (breakdown only)",
        "",
        "## Detection Framework",
        "",
        "The v2 pipeline combines hard legality checks, single-view image quality, low-dimensional state/action temporal checks, and state-only cross-modal consistency checks.",
        "Direct Vision-Vision findings are disabled. Multi-view visual agreement is checked only after a state-supported overlap gate; actions are excluded from consistency checks because commands can naturally lead or lag observed state and vision.",
        "",
        "| dimension | max_points | current_mean |",
        "|---|---:|---:|",
    ]
    for dimension, points in DIMENSION_POINTS.items():
        field_name = f"score_{dimension}"
        mean_score = round(float(np.mean([getattr(item, field_name) for item in episode_results])) if episode_results else 0.0, 2)
        clean_markdown_lines.append(f"| {dimension} | {points:g} | {mean_score} |")

    clean_markdown_lines.extend(
        [
            "",
            "## Module Status",
            "",
            "Status enum: `ok` completed, `warning` core completed with degraded or unavailable submodules, `unavailable` core or submodule could not run, `fail` module execution failed.",
            "",
            "| module | ok | warning | unavailable | fail |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for module_name, counts in geometry_module_status_counts.items():
        clean_markdown_lines.append(
            f"| {module_name} | {counts['ok']} | {counts['warning']} | {counts['unavailable']} | {counts['fail']} |"
        )

    clean_markdown_lines.extend(
        [
            "",
            "## Finding Summary",
            "",
            f"- merged findings: {len(findings)}",
            f"- critical: {total_critical}",
            f"- high confidence: {total_high_confidence}",
            f"- suspicious: {total_suspicious}",
            f"- out of distribution: {total_ood}",
            "",
            "## Top 10 Issue Types",
            "",
            "| issue_type | count |",
            "|---|---:|",
        ]
    )
    for issue, count in by_issue.most_common(10):
        clean_markdown_lines.append(f"| {issue} | {count} |")

    clean_markdown_lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `outputs/v2/diagnostics/findings_v2.json`: normalized merged findings",
            "- `outputs/v2/diagnostics/reference_baselines_v2.json`: reference baselines and thresholds",
            "- `outputs/v2/diagnostics/problem_standards_v2.json`: problem definitions and decision rules",
            "- `outputs/v2/reports/episode_scores_v2.csv`: episode-level score table",
            "- `outputs/v2/reports/dataset_quality_report_v2.json`: machine-readable complete report",
            "- `outputs/v2/reports/dataset_quality_report_v2.md`: human-readable summary report",
            "- `outputs/v2/reports/episode_quality_report_v2_detail.md`: episode-level detail report",
        ]
    )
    md_path.write_text("\n".join(clean_markdown_lines) + "\n", encoding="utf-8")

    findings_by_episode: dict[int, list[Finding]] = defaultdict(list)
    for item in findings:
        if item.episode_index is not None:
            findings_by_episode[int(item.episode_index)].append(item)

    detail_lines = [
        "# V2 Episode detailed quality report",
        "",
        "This report is regenerated from the current merged findings.",
        "",
        "## Overview",
        "",
        "| episode | task | length | phase_status | geometry_status | geometry_reason | arms.left | arms.right | bimanual | state_vision.left | state_vision.right | detection_70 | data_value_20 | reliability_10 | official_total | legacy_total | findings |",
        "|---|---:|---:|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(episode_results, key=lambda item: item.episode_index):
        arms = dict(result.geometry_module_statuses.get("arms", {}))
        state_vision = dict(result.geometry_module_statuses.get("state_vision", {}))
        detail_lines.append(
            "| "
            f"{result.episode_index} | {result.task_index} | {result.length} | "
            f"{result.phase_status} | {result.geometry_status} | {result.geometry_reason or ''} | "
            f"{geometry_module_status(arms.get('left'))} | {geometry_module_status(arms.get('right'))} | "
            f"{geometry_module_status(result.geometry_module_statuses.get('bimanual'))} | "
            f"{geometry_module_status(state_vision.get('left'))} | {geometry_module_status(state_vision.get('right'))} | "
            f"{result.score_detection_quality:g} | {result.score_data_value:g} | {result.score_engineering_reliability:g} | "
            f"{result.score_total:g} | {result.legacy_score_total:g} | {result.finding_count} |"
        )

    detail_lines.extend(["", "## Episode Details", ""])
    for result in sorted(episode_results, key=lambda item: item.episode_index):
        detail_lines.extend(
            [
                f"### Episode {result.episode_index}",
                "",
                f"- task: {result.task_index}",
                f"- length: {result.length}",
                f"- score_total: {result.score_total:g}",
                f"- official_breakdown: detection_quality {result.score_detection_quality:g}/70, data_value {result.score_data_value:g}/20, engineering_reliability {result.score_engineering_reliability:g}/10",
                f"- legacy_score_total: {result.legacy_score_total:g}",
                f"- phase_status: {result.phase_status}",
                f"- phase_reason: {result.phase_reason or ''}",
                f"- phase_segments: {json.dumps(result.phase_segments, ensure_ascii=False, sort_keys=True)}",
                f"- geometry_status: {result.geometry_status}",
                f"- geometry_reason: {result.geometry_reason or ''}",
                f"- geometry_module_statuses: {json.dumps(result.geometry_module_statuses, ensure_ascii=False, sort_keys=True)}",
                (
                    "- legacy_breakdown: "
                    f"structural {result.score_structural:g}, "
                    f"vision_single {result.score_vision_single:g}, "
                    f"vision_vision {result.score_vision_vision:g}, "
                    f"state {result.score_state:g}, "
                    f"temporal {result.score_temporal:g}, "
                    f"cross_modal {result.score_cross_modal:g}"
                ),
                (
                    "- findings: "
                    f"{result.finding_count} | critical {result.critical_count} | "
                    f"high_confidence {result.high_confidence_count} | suspicious {result.suspicious_count} | "
                    f"ood {result.ood_count}"
                ),
            ]
        )
        episode_findings = sorted(
            findings_by_episode.get(result.episode_index, []),
            key=lambda item: (-1 if item.frame_start is None else item.frame_start, item.finding_id),
        )
        if not episode_findings:
            detail_lines.extend(["- findings detail: none", ""])
            continue
        detail_lines.append("- findings detail:")
        for finding in episode_findings:
            if finding.frame_start is None:
                frame_text = "episode-level"
            elif finding.frame_end is not None and finding.frame_end != finding.frame_start:
                frame_text = f"frames {finding.frame_start}-{finding.frame_end}"
            else:
                frame_text = f"frame {finding.frame_start}"
            view_text = f" | view {finding.view}" if finding.view else ""
            column_text = f" | column {finding.column}" if finding.column else ""
            detail_lines.append(
                "  - "
                f"[{finding.finding_id}] {finding.issue_type} - {finding.issue_name} | "
                f"category {finding.category_id} | {finding.modality} | {finding.object_level} | "
                f"{frame_text}{view_text}{column_text} | {finding.confidence_level} | "
                f"severity {finding.severity_score:g}"
            )
        detail_lines.append("")

    (REPORT_ROOT / "episode_quality_report_v2_detail.md").write_text(
        "\n".join(detail_lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v2 quality pipeline.")
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--target-root", type=Path, default=None)
    parser.add_argument("--geometry-config", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_root = args.reference_root
    target_root = args.target_root
    if reference_root is None and target_root is None:
        reference_root = ROOT
        target_root = ROOT
    elif reference_root is None:
        reference_root = target_root
    elif target_root is None:
        target_root = reference_root

    assert reference_root is not None and target_root is not None

    reference_info, reference_tasks, reference_episodes, reference_parquet_files = load_dataset(reference_root)
    target_info, tasks, episodes, parquet_files = load_dataset(target_root)
    reference_episodes_meta = {int(row["episode_index"]): row for row in reference_episodes}
    episodes_meta = {int(row["episode_index"]): row for row in episodes}
    views = image_views_from_info(target_info)
    reference_views = image_views_from_info(reference_info)
    factory = FindingFactory()

    baselines = collect_reference_baselines(reference_info, reference_episodes_meta, reference_parquet_files, reference_views)
    geometry_config = geometry_constraints.default_geometry_config(reference_info)
    if args.geometry_config is not None:
        geometry_config = geometry_constraints.load_geometry_config(args.geometry_config, geometry_config)
    baselines["geometry_config"] = geometry_config
    baselines["geometry_constraints"] = geometry_constraints.fit_geometry_reference(reference_parquet_files, geometry_config)
    expected_low_dim_shape = {}
    for column in LOW_DIM_COLUMNS:
        spec = target_info.get("features", {}).get(column)
        if spec and spec.get("shape"):
            expected_low_dim_shape[column] = int(spec["shape"][0])
    baselines["expected_low_dim_shape"] = expected_low_dim_shape

    episode_metas: list[dict[str, Any]] = []
    all_findings: list[Finding] = []
    for path in parquet_files:
        episode_from_path = safe_episode_from_path(path)
        expected_meta = episodes_meta.get(episode_from_path) if episode_from_path is not None else None
        episode_meta, findings = inspect_episode(path, target_info, expected_meta, views, baselines, factory)
        if episode_meta["episode_index"] is None and episode_from_path is not None:
            episode_meta["episode_index"] = episode_from_path
        episode_metas.append(episode_meta)
        all_findings.extend(findings)

    merged_findings = merge_findings(all_findings)
    episode_results = [score_episode(meta, merged_findings) for meta in sorted(episode_metas, key=lambda item: item["episode_index"])]
    geometry_payload = [meta.get("geometry_constraints", {}) for meta in sorted(episode_metas, key=lambda item: item["episode_index"])]
    (DIAG_ROOT / "geometry_constraints_v2.json").write_text(json.dumps(geometry_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(target_info, tasks, baselines, merged_findings, episode_results)

    dataset_score = round(float(np.mean([item.score_total for item in episode_results])) if episode_results else 0.0, 2)
    print(f"v2 episodes_checked: {len(episode_results)}")
    print(f"v2 merged_findings: {len(merged_findings)}")
    print(f"v2 dataset_quality_score: {dataset_score}/100")
    print(f"wrote: {DIAG_ROOT / 'findings_v2.json'}")
    print(f"wrote: {REPORT_ROOT / 'dataset_quality_report_v2.md'}")


if __name__ == "__main__":
    main()





