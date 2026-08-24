from __future__ import annotations

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
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "v2"
DIAG_ROOT = OUTPUT_ROOT / "diagnostics"
REPORT_ROOT = OUTPUT_ROOT / "reports"
DIAG_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_IMAGE_VIEWS = ["image", "left_wrist_image", "right_wrist_image"]
LOW_DIM_COLUMNS = ["state", "actions"]
INDEX_COLUMNS = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
MIN_FREEZE_FRAMES = 8
MAX_LAG = 5
ARM_POSITION_SLICES = {
    "left": slice(0, 3),
    "right": slice(10, 13),
}
ARM_ROTATION_6D_SLICES = {
    "left": slice(3, 9),
    "right": slice(13, 19),
}
WRIST_CAMERA_LOCAL_OPTICAL_AXIS = {
    # In this dataset the wrist cameras overlap the shared workspace when the
    # left wrist looks along local -Y and the right wrist along local +Y.
    # Real hand-eye extrinsics should replace this assumption if available.
    "left": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "right": np.array([0.0, 1.0, 0.0], dtype=np.float64),
}
BASE_CAMERA_OFFSET_FROM_WORKSPACE = np.array([-0.65, 0.0, 0.35], dtype=np.float64)
BASE_CAMERA_FOV_DEGREES = 95.0
WRIST_CAMERA_FOV_DEGREES = 95.0
BASE_CAMERA_MAX_RANGE_M = 2.0
WRIST_CAMERA_MAX_RANGE_M = 0.9
GEOMETRY_OVERLAP_MIN_SCORE = 0.10


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
    score_structural: float
    score_vision_single: float
    score_vision_vision: float
    score_state: float
    score_temporal: float
    score_cross_modal: float
    finding_count: int
    critical_count: int
    high_confidence_count: int
    suspicious_count: int
    ood_count: int


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
        "definition": "Visual spatial change and low-dimensional state change are inconsistent, suggesting broken geometry, scale, crop, or coarse cross-modal relation.",
        "decision_rule": "Known camera semantics and robot state first indicate that two views should overlap; only then is weak visual motion correlation used as verification.",
    },
    "state_vision_state": {
        "category_id": "1.1.2.D",
        "category_name": "State-Vision-State consistency problem",
        "definition": "Low-dimensional state values or local transitions are inconsistent with visual evidence or neighboring state context.",
        "decision_rule": "State value/delta/acceleration is a calibrated outlier and visual evidence does not support the same local change.",
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
        "definition": "Vision and low-dimensional state are locally out of sync or show unstable response timing.",
        "decision_rule": "Best lag between visual motion and state motion deviates from the calibrated reference lag, or the best correlation falls below the reference baseline.",
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


def find_meta_root() -> Path:
    for info_path in ROOT.rglob("info.json"):
        if is_real_path(info_path):
            return info_path.parent
    raise FileNotFoundError("Could not find LeRobot info.json under workspace")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dataset() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    meta_root = find_meta_root()
    info = json.loads((meta_root / "info.json").read_text(encoding="utf-8"))
    tasks = load_jsonl(meta_root / "tasks.jsonl")
    episodes = load_jsonl(meta_root / "episodes.jsonl")
    data_root = meta_root.parent
    parquet_files = sorted(path for path in data_root.rglob("*.parquet") if is_real_path(path))
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


def lagged_correlation(a: np.ndarray, b: np.ndarray, lag: int) -> float | None:
    if lag > 0:
        left, right = a[lag:], b[:-lag]
    elif lag < 0:
        left, right = a[:lag], b[-lag:]
    else:
        left, right = a, b
    if len(left) < 8 or len(right) < 8:
        return None
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int = MAX_LAG) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return {"best_lag": None, "best_correlation": None, "correlations": {}}
    a = (a - np.median(a)) / max(float(np.std(a)), 1e-12)
    b = (b - np.median(b)) / max(float(np.std(b)), 1e-12)
    correlations: dict[str, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        corr = lagged_correlation(a, b, lag)
        if corr is not None:
            correlations[str(lag)] = corr
    if not correlations:
        return {"best_lag": None, "best_correlation": None, "correlations": {}}
    lag_key = max(correlations, key=correlations.get)
    return {
        "best_lag": int(lag_key),
        "best_correlation": float(correlations[lag_key]),
        "correlations": correlations,
    }


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
    state_delta: dict[str, list[float]] = defaultdict(list)
    state_accel: dict[str, list[float]] = defaultdict(list)
    sync_corr: dict[str, list[float]] = defaultdict(list)
    sync_lag: dict[str, list[float]] = defaultdict(list)
    view_pair_corr: dict[str, list[float]] = defaultdict(list)

    expected_shape_by_view = {
        name: tuple(info.get("features", {}).get(name, {}).get("shape", [224, 224, 3]))
        for name in views
    }

    for path in parquet_files:
        columns = [column for column in set(views + LOW_DIM_COLUMNS + ["frame_index", "task_index"]) if column]
        try:
            table = pq.read_table(path, columns=[c for c in columns if c in pq.read_schema(path).names])
            df = table.to_pandas()
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

        for left_i, left in enumerate(views):
            for right in views[left_i + 1 :]:
                if left in visual_motions and right in visual_motions:
                    length = min(len(visual_motions[left]), len(visual_motions[right]))
                    if length >= 8 and np.std(visual_motions[left][:length]) > 1e-12 and np.std(visual_motions[right][:length]) > 1e-12:
                        corr = float(np.corrcoef(visual_motions[left][:length], visual_motions[right][:length])[0, 1])
                        if np.isfinite(corr):
                            view_pair_corr[f"global|{left}|{right}"].append(corr)
                            view_pair_corr[f"{task_key}|{left}|{right}"].append(corr)

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

        for view, visual_signal in visual_motions.items():
            for column, low_signal in state_motion.items():
                length = min(len(visual_signal), len(low_signal))
                if length < 8:
                    continue
                lag = best_lag(visual_signal[:length], low_signal[:length])
                if lag["best_correlation"] is not None:
                    key_global = f"global|{view}|{column}"
                    key_task = f"{task_key}|{view}|{column}"
                    sync_corr[key_global].append(float(lag["best_correlation"]))
                    sync_corr[key_task].append(float(lag["best_correlation"]))
                    if lag["best_lag"] is not None:
                        sync_lag[key_global].append(float(lag["best_lag"]))
                        sync_lag[key_task].append(float(lag["best_lag"]))

    return {
        "image_metrics": {
            key: {metric: robust_stats(values) for metric, values in metric_map.items()}
            for key, metric_map in image_values.items()
        },
        "image_motion": {key: robust_stats(values) for key, values in image_motion.items()},
        "image_accel": {key: robust_stats(values) for key, values in image_accel.items()},
        "state_values": {key: robust_stats(values) for key, values in state_values.items()},
        "state_delta": {key: robust_stats(values) for key, values in state_delta.items()},
        "state_accel": {key: robust_stats(values) for key, values in state_accel.items()},
        "sync_corr": {key: robust_stats(values) for key, values in sync_corr.items()},
        "sync_lag": {key: robust_stats(values) for key, values in sync_lag.items()},
        "view_pair_corr": {key: robust_stats(values) for key, values in view_pair_corr.items()},
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
        table = pq.read_table(path)
        df = table.to_pandas()
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
        return {"episode_index": episode, "task_index": task_index, "length": length, "file": str(path)}, findings

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
    inspect_state_gated_vision(frames, episode, task_index, views, image_context, state_context, baselines, findings, factory)
    inspect_cross_modal(frames, episode, task_index, views, image_context, state_context, baselines, findings, factory)

    return {"episode_index": episode, "task_index": task_index, "length": length, "file": str(path)}, findings


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
            findings.append(
                factory.make(
                    "vision_temporal",
                    "exact_frozen_image_run",
                    "Exact repeated visual frames",
                    "segment",
                    "vision",
                    min(100.0, 55.0 + 3.0 * (run_len - MIN_FREEZE_FRAMES)),
                    episode_index=episode,
                    task_index=task_index,
                    view=view,
                    frame_start=int(frames[start]),
                    frame_end=int(frames[end]),
                    evidence={"run_length": run_len},
                )
            )
        start = end + 1

    if motion.size == 0:
        return
    motion_stats = baseline_lookup(baselines, "image_motion", task_index, view)
    accel_stats = baseline_lookup(baselines, "image_accel", task_index, view)
    low_threshold = max(0.2, stat_value(motion_stats, "p01", 0.2) * 0.5)
    high_threshold = max(stat_value(motion_stats, "p99", float(np.max(motion))) * 1.5, stat_value(motion_stats, "median", 0.0) + 8.0 * stat_value(motion_stats, "mad", 0.0))

    near_static = np.where(motion <= low_threshold)[0]
    for start, end in contiguous_ranges(near_static, max_gap=1):
        run_len = end - start + 1
        if run_len >= MIN_FREEZE_FRAMES:
            findings.append(
                factory.make(
                    "vision_temporal",
                    "low_motion_freeze_run",
                    "Visual motion is near zero for a long window",
                    "segment",
                    "vision",
                    min(95.0, 45.0 + 2.5 * run_len),
                    episode_index=episode,
                    task_index=task_index,
                    view=view,
                    frame_start=int(frames[start]),
                    frame_end=int(frames[min(end + 1, len(frames) - 1)]),
                    evidence={"run_length": run_len, "low_motion_threshold": low_threshold, "median_motion": float(np.median(motion[start : end + 1]))},
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
        accel_threshold = max(stat_value(accel_stats, "p99", float(np.max(accel))) * 1.5, stat_value(accel_stats, "median", 0.0) + 8.0 * stat_value(accel_stats, "mad", 0.0))
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
        for start, end in contiguous_ranges(frames[non_finite]):
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
                    frame_start=start,
                    frame_end=end,
                )
            )
        impossible = np.where(np.isfinite(matrix).all(axis=1) & (np.max(np.abs(matrix), axis=1) > 1e6))[0]
        for start, end in contiguous_ranges(frames[impossible]):
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
                    frame_start=start,
                    frame_end=end,
                    evidence={"guardrail_abs_max": 1e6},
                )
            )

        value_stats = baseline_lookup(baselines, "state_values", task_index, column)
        row_abs = np.max(np.abs(matrix), axis=1) if matrix.size else np.array([])
        global_abs_stats = robust_stats(np.abs(matrix[np.isfinite(matrix)]).reshape(-1).tolist()) if matrix.size else {"count": 0}
        for idx, row_max in enumerate(row_abs):
            if not np.isfinite(row_max):
                continue
            score = max(
                100.0 * upper_outlier_score(float(row_max), global_abs_stats, weak_z=6.0, strong_z=12.0),
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
        context[column] = {"matrix": matrix, "delta": delta, "accel": accel}
        inspect_state_temporal(frames, episode, task_index, column, delta, accel, baselines, findings, factory)
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
) -> None:
    if delta.size == 0:
        return
    delta_stats = baseline_lookup(baselines, "state_delta", task_index, column)
    accel_stats = baseline_lookup(baselines, "state_accel", task_index, column)
    low_threshold = max(1e-8, stat_value(delta_stats, "p01", 1e-8) * 0.5)
    high_threshold = max(stat_value(delta_stats, "p99", float(np.max(delta))) * 1.5, stat_value(delta_stats, "median", 0.0) + 8.0 * stat_value(delta_stats, "mad", 0.0))

    frozen = np.where(delta <= low_threshold)[0]
    for start, end in contiguous_ranges(frozen, max_gap=1):
        run_len = end - start + 1
        if run_len >= MIN_FREEZE_FRAMES:
            findings.append(
                factory.make(
                    "state_temporal",
                    "low_dim_freeze_run",
                    "Low-dimensional state is nearly unchanged for a long window",
                    "segment",
                    "state",
                    min(90.0, 35.0 + 2.5 * run_len),
                    episode_index=episode,
                    task_index=task_index,
                    column=column,
                    frame_start=int(frames[start]),
                    frame_end=int(frames[min(end + 1, len(frames) - 1)]),
                    evidence={"run_length": run_len, "low_delta_threshold": low_threshold, "median_delta": float(np.median(delta[start : end + 1]))},
                )
            )

    fast = np.where(delta >= high_threshold)[0]
    for start, end in contiguous_ranges(fast, max_gap=1):
        max_delta = float(np.max(delta[start : end + 1]))
        score = max(35.0, 100.0 * upper_outlier_score(max_delta, delta_stats, weak_z=4.0, strong_z=10.0))
        findings.append(
            factory.make(
                "state_temporal",
                "low_dim_fast_jump",
                "Low-dimensional state has an extreme fast jump",
                "segment",
                "state",
                score,
                episode_index=episode,
                task_index=task_index,
                column=column,
                frame_start=int(frames[start]),
                frame_end=int(frames[min(end + 1, len(frames) - 1)]),
                evidence={"max_delta": max_delta, "high_delta_threshold": high_threshold},
            )
        )

    if accel.size:
        accel_threshold = max(stat_value(accel_stats, "p99", float(np.max(accel))) * 1.5, stat_value(accel_stats, "median", 0.0) + 8.0 * stat_value(accel_stats, "mad", 0.0))
        spikes = np.where(accel >= accel_threshold)[0]
        for start, end in contiguous_ranges(spikes, max_gap=1):
            max_accel = float(np.max(accel[start : end + 1]))
            score = max(35.0, 100.0 * upper_outlier_score(max_accel, accel_stats, weak_z=4.0, strong_z=10.0))
            findings.append(
                factory.make(
                    "state_temporal",
                    "low_dim_jitter_or_spike",
                    "Low-dimensional state acceleration is an extreme outlier",
                    "segment",
                    "state",
                    score,
                    episode_index=episode,
                    task_index=task_index,
                    column=column,
                    frame_start=int(frames[start + 1]),
                    frame_end=int(frames[min(end + 1, len(frames) - 1)]),
                    evidence={"max_accel": max_accel, "accel_threshold": accel_threshold},
                )
            )


def view_arm_hint(view: str) -> str | None:
    lowered = view.lower()
    if "left" in lowered:
        return "left"
    if "right" in lowered:
        return "right"
    return None


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12 or not np.isfinite(norm):
        return np.zeros_like(vector, dtype=np.float64)
    return vector / norm


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray | None:
    values = np.asarray(rotation_6d, dtype=np.float64).reshape(-1)
    if values.size < 6 or not np.isfinite(values[:6]).all():
        return None
    x_axis = normalize_vector(values[:3])
    y_raw = values[3:6]
    y_axis = normalize_vector(y_raw - float(np.dot(x_axis, y_raw)) * x_axis)
    if np.linalg.norm(x_axis) < 1e-12 or np.linalg.norm(y_axis) < 1e-12:
        return None
    z_axis = normalize_vector(np.cross(x_axis, y_axis))
    if np.linalg.norm(z_axis) < 1e-12:
        return None
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def geometry_state_matrix(state_context: dict[str, dict[str, Any]]) -> tuple[str | None, np.ndarray | None]:
    for column in ["state", "actions"]:
        item = state_context.get(column)
        matrix = item.get("matrix") if isinstance(item, dict) else None
        if matrix is None:
            continue
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim == 2 and matrix.shape[1] >= 13 and np.isfinite(matrix).all():
            return column, matrix
    return None, None


def arm_position_motion(matrix: np.ndarray, arm: str) -> tuple[np.ndarray | None, np.ndarray]:
    position_slice = ARM_POSITION_SLICES.get(arm)
    if position_slice is None or matrix.shape[1] < position_slice.stop:
        return None, np.array([], dtype=np.float64)
    positions = matrix[:, position_slice]
    if len(positions) < 2:
        return positions, np.array([], dtype=np.float64)
    motion = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return positions, motion


def active_motion_summary(motion: np.ndarray) -> dict[str, float]:
    motion = np.asarray(motion, dtype=np.float64)
    motion = motion[np.isfinite(motion)]
    if motion.size == 0:
        return {"threshold": 0.0, "active_fraction": 0.0, "median": 0.0, "max": 0.0}
    threshold = max(1e-6, float(np.quantile(motion, 0.75)))
    active_fraction = float(np.mean(motion > threshold))
    return {
        "threshold": threshold,
        "active_fraction": active_fraction,
        "median": float(np.median(motion)),
        "max": float(np.max(motion)),
    }


def workspace_geometry(matrix: np.ndarray) -> dict[str, Any]:
    arm_positions = []
    for arm in ["left", "right"]:
        positions, _ = arm_position_motion(matrix, arm)
        if positions is not None and len(positions) > 0:
            arm_positions.append(positions)
    if not arm_positions:
        return {"center": None, "extent": None}
    points = np.vstack(arm_positions)
    return {
        "center": np.mean(points, axis=0),
        "extent": np.ptp(points, axis=0),
    }


def camera_pose_for_view(
    view: str,
    row: np.ndarray,
    workspace_center: np.ndarray,
) -> dict[str, Any] | None:
    arm = view_arm_hint(view)
    if arm is None:
        position = workspace_center + BASE_CAMERA_OFFSET_FROM_WORKSPACE
        optical_axis = normalize_vector(workspace_center - position)
        return {
            "kind": "base",
            "arm": None,
            "position": position,
            "optical_axis": optical_axis,
            "fov_degrees": BASE_CAMERA_FOV_DEGREES,
            "max_range_m": BASE_CAMERA_MAX_RANGE_M,
            "local_optical_axis": None,
        }

    position_slice = ARM_POSITION_SLICES.get(arm)
    rotation_slice = ARM_ROTATION_6D_SLICES.get(arm)
    local_axis = WRIST_CAMERA_LOCAL_OPTICAL_AXIS.get(arm)
    if position_slice is None or rotation_slice is None or local_axis is None:
        return None
    if row.size < max(position_slice.stop, rotation_slice.stop):
        return None
    rotation = rotation_6d_to_matrix(row[rotation_slice])
    if rotation is None:
        return None
    position = np.asarray(row[position_slice], dtype=np.float64)
    optical_axis = normalize_vector(rotation @ local_axis)
    if np.linalg.norm(optical_axis) < 1e-12:
        return None
    return {
        "kind": "wrist",
        "arm": arm,
        "position": position,
        "optical_axis": optical_axis,
        "fov_degrees": WRIST_CAMERA_FOV_DEGREES,
        "max_range_m": WRIST_CAMERA_MAX_RANGE_M,
        "local_optical_axis": local_axis.tolist(),
    }


def point_in_camera_cone(point: np.ndarray, pose: dict[str, Any]) -> bool:
    point = np.asarray(point, dtype=np.float64)
    position = np.asarray(pose["position"], dtype=np.float64)
    optical_axis = normalize_vector(np.asarray(pose["optical_axis"], dtype=np.float64))
    vector = point - position
    distance = float(np.linalg.norm(vector))
    if distance < 0.02 or distance > float(pose["max_range_m"]):
        return False
    direction = vector / distance
    cos_limit = math.cos(math.radians(float(pose["fov_degrees"]) * 0.5))
    return float(np.dot(direction, optical_axis)) >= cos_limit


def shared_geometry_targets(row: np.ndarray, workspace_center: np.ndarray) -> list[np.ndarray]:
    targets = [np.asarray(workspace_center, dtype=np.float64)]
    left_position = row[ARM_POSITION_SLICES["left"]] if row.size >= ARM_POSITION_SLICES["left"].stop else None
    right_position = row[ARM_POSITION_SLICES["right"]] if row.size >= ARM_POSITION_SLICES["right"].stop else None
    if left_position is not None and right_position is not None:
        targets.append((np.asarray(left_position, dtype=np.float64) + np.asarray(right_position, dtype=np.float64)) * 0.5)
    return targets


def estimate_camera_overlap(
    left: str,
    right: str,
    matrix: np.ndarray,
) -> dict[str, Any]:
    geometry = workspace_geometry(matrix)
    workspace_center = geometry.get("center")
    if workspace_center is None:
        return {"enabled": False, "reason": "workspace_geometry_unavailable"}

    sample_count = min(32, len(matrix))
    if sample_count < 8:
        return {"enabled": False, "reason": "too_few_pose_samples", "sample_count": sample_count}
    sample_indices = np.unique(np.linspace(0, len(matrix) - 1, sample_count).astype(int))
    hits = 0
    checked = 0
    shared_distances: list[float] = []
    example_pose: dict[str, Any] | None = None

    for index in sample_indices:
        row = np.asarray(matrix[index], dtype=np.float64)
        left_pose = camera_pose_for_view(left, row, workspace_center)
        right_pose = camera_pose_for_view(right, row, workspace_center)
        if left_pose is None or right_pose is None:
            continue
        checked += 1
        targets = shared_geometry_targets(row, workspace_center)
        matched_target = None
        for target in targets:
            if point_in_camera_cone(target, left_pose) and point_in_camera_cone(target, right_pose):
                matched_target = target
                break
        if matched_target is not None:
            hits += 1
            shared_distances.append(float(np.linalg.norm(matched_target - np.asarray(left_pose["position"], dtype=np.float64))))
            if example_pose is None:
                example_pose = {
                    "sample_index": int(index),
                    "left_camera": {
                        "kind": left_pose["kind"],
                        "arm": left_pose["arm"],
                        "position": np.asarray(left_pose["position"]).round(4).tolist(),
                        "optical_axis": np.asarray(left_pose["optical_axis"]).round(4).tolist(),
                        "local_optical_axis": left_pose["local_optical_axis"],
                    },
                    "right_camera": {
                        "kind": right_pose["kind"],
                        "arm": right_pose["arm"],
                        "position": np.asarray(right_pose["position"]).round(4).tolist(),
                        "optical_axis": np.asarray(right_pose["optical_axis"]).round(4).tolist(),
                        "local_optical_axis": right_pose["local_optical_axis"],
                    },
                    "matched_target": np.asarray(matched_target).round(4).tolist(),
                }

    overlap_score = float(hits / checked) if checked else 0.0
    return {
        "enabled": overlap_score >= GEOMETRY_OVERLAP_MIN_SCORE,
        "reason": "estimated_frustum_overlap" if overlap_score >= GEOMETRY_OVERLAP_MIN_SCORE else "estimated_frustum_overlap_too_small",
        "overlap_score": overlap_score,
        "hits": int(hits),
        "samples_checked": int(checked),
        "workspace_center": np.asarray(workspace_center).round(4).tolist(),
        "workspace_extent": np.asarray(geometry.get("extent")).round(4).tolist() if geometry.get("extent") is not None else None,
        "median_shared_target_distance_m": float(np.median(shared_distances)) if shared_distances else None,
        "camera_model": {
            "robot": "Franka/Panda-like dual-arm state with end-effector pose",
            "base_camera_offset_from_workspace": BASE_CAMERA_OFFSET_FROM_WORKSPACE.tolist(),
            "base_fov_degrees": BASE_CAMERA_FOV_DEGREES,
            "wrist_fov_degrees": WRIST_CAMERA_FOV_DEGREES,
            "wrist_local_optical_axes": {key: value.tolist() for key, value in WRIST_CAMERA_LOCAL_OPTICAL_AXIS.items()},
            "hand_eye_calibration": "not provided; wrist optical axes are inferred from state/workspace consistency",
        },
        "example_pose": example_pose,
    }


def state_overlap_gate(
    left: str,
    right: str,
    state_context: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    geometry_column, matrix = geometry_state_matrix(state_context)
    left_arm = view_arm_hint(left)
    right_arm = view_arm_hint(right)
    evidence: dict[str, Any] = {
        "enabled": False,
        "reason": "state_geometry_unavailable",
        "geometry_column": geometry_column,
        "left_view_arm": left_arm,
        "right_view_arm": right_arm,
    }
    if matrix is None:
        return evidence

    overlap = estimate_camera_overlap(left, right, matrix)
    evidence.update(overlap)
    return evidence


def inspect_state_gated_vision(
    frames: np.ndarray,
    episode: int | None,
    task_index: int | None,
    views: list[str],
    image_context: dict[str, dict[str, Any]],
    state_context: dict[str, dict[str, Any]],
    baselines: dict[str, Any],
    findings: list[Finding],
    factory: FindingFactory,
) -> None:
    gated_pairs: list[tuple[str, str, dict[str, Any]]] = []
    for left_i, left in enumerate(views):
        for right in views[left_i + 1 :]:
            if left not in image_context or right not in image_context:
                continue
            gate = state_overlap_gate(left, right, state_context)
            if not gate.get("enabled"):
                continue
            gated_pairs.append((left, right, gate))
            left_motion = image_context[left]["motion"]
            right_motion = image_context[right]["motion"]
            length = min(len(left_motion), len(right_motion))
            if length < 8 or np.std(left_motion[:length]) < 1e-12 or np.std(right_motion[:length]) < 1e-12:
                continue
            corr = float(np.corrcoef(left_motion[:length], right_motion[:length])[0, 1])
            pair_stats = baseline_lookup(baselines, "view_pair_corr", task_index, left, right)
            score = 100.0 * lower_outlier_score(corr, pair_stats, weak_z=3.0, strong_z=6.0)
            if corr < 0.0:
                score = max(score, 45.0)
            elif corr < 0.05:
                score = max(score, 35.0)
            if score >= 30:
                findings.append(
                    factory.make(
                        "vision_state_vision",
                        "state_gated_view_pair_motion_inconsistency",
                        "State-supported overlapping camera pair has weak visual agreement",
                        "episode",
                        "vision-state",
                        score,
                        ood=score < 55,
                        episode_index=episode,
                        task_index=task_index,
                        view=f"{left}|{right}",
                        evidence={
                            "correlation": corr,
                            "pair_baseline": pair_stats,
                            "state_gate": gate,
                            "verification_policy": "Visual correlation is used only as weak verification after state-supported overlap.",
                        },
                    )
                )

    per_view_mean_motion = {
        view: float(np.median(ctx["motion"])) for view, ctx in image_context.items() if len(ctx["motion"]) > 0
    }
    gated_views = {view for pair in gated_pairs for view in pair[:2]}
    gated_motion = {view: value for view, value in per_view_mean_motion.items() if view in gated_views}
    if len(gated_motion) >= 2:
        values = np.asarray(list(gated_motion.values()), dtype=np.float64)
        if np.min(values) >= 0:
            ratio = float(np.max(values) / max(np.min(values), 1e-6))
            if ratio > 10.0:
                culprit = max(gated_motion, key=gated_motion.get)
                supporting_gates = [
                    gate for left, right, gate in gated_pairs if culprit in {left, right}
                ]
                findings.append(
                    factory.make(
                        "vision_state_vision",
                        "state_gated_view_motion_scale_mismatch",
                        "State-supported overlapping camera group has a large visual motion scale mismatch",
                        "episode",
                        "vision-state",
                        min(90.0, 35.0 + ratio),
                        ood=ratio < 20.0,
                        episode_index=episode,
                        task_index=task_index,
                        view=culprit,
                        evidence={
                            "per_view_median_motion": gated_motion,
                            "max_min_ratio": ratio,
                            "state_gates": supporting_gates,
                            "verification_policy": "Motion scale comparison is limited to state-supported overlapping views.",
                        },
                    )
                )


def inspect_cross_modal(
    frames: np.ndarray,
    episode: int | None,
    task_index: int | None,
    views: list[str],
    image_context: dict[str, dict[str, Any]],
    state_context: dict[str, dict[str, Any]],
    baselines: dict[str, Any],
    findings: list[Finding],
    factory: FindingFactory,
) -> None:
    visual_combined = None
    available_visual = [ctx["motion"] for ctx in image_context.values() if len(ctx["motion"]) > 0]
    if available_visual:
        min_len = min(len(item) for item in available_visual)
        visual_combined = np.mean(np.stack([item[:min_len] for item in available_visual]), axis=0)

    for column, state_item in state_context.items():
        delta = state_item.get("delta", np.array([]))
        if visual_combined is not None and len(delta) > 0:
            length = min(len(visual_combined), len(delta))
            visual = visual_combined[:length]
            low_dim = delta[:length]
            if length >= 8:
                visual_stats = robust_stats(visual.tolist())
                state_stats = baseline_lookup(baselines, "state_delta", task_index, column)
                visual_low = visual <= max(0.2, stat_value(visual_stats, "p05", 0.2) * 0.5)
                visual_high = visual >= stat_value(visual_stats, "p95", float(np.max(visual)))
                state_high_threshold = stat_value(state_stats, "p99", float(np.max(low_dim)))
                state_high = low_dim >= state_high_threshold
                state_low = low_dim <= max(1e-8, stat_value(state_stats, "p01", 1e-8) * 0.5)

                for start, end in contiguous_ranges(np.where(state_high & visual_low)[0], max_gap=1):
                    if end - start + 1 >= 2:
                        findings.append(
                            factory.make(
                                "vision_state_vision",
                                "state_moves_visual_static",
                                "State changes strongly but visual motion is static",
                                "segment",
                                "vision-state",
                                min(90.0, 45.0 + 8.0 * (end - start + 1)),
                                episode_index=episode,
                                task_index=task_index,
                                column=column,
                                frame_start=int(frames[start]),
                                frame_end=int(frames[min(end + 1, len(frames) - 1)]),
                                evidence={
                                    "max_state_delta": float(np.max(low_dim[start : end + 1])),
                                    "median_visual_motion": float(np.median(visual[start : end + 1])),
                                    "state_high_threshold": state_high_threshold,
                                },
                            )
                        )

                for start, end in contiguous_ranges(np.where(visual_high & state_low)[0], max_gap=1):
                    if end - start + 1 >= 2:
                        findings.append(
                            factory.make(
                                "state_vision_state",
                                "visual_moves_state_static",
                                "Visual motion is strong but low-dimensional state is static",
                                "segment",
                                "vision-state",
                                min(90.0, 45.0 + 8.0 * (end - start + 1)),
                                episode_index=episode,
                                task_index=task_index,
                                column=column,
                                frame_start=int(frames[start]),
                                frame_end=int(frames[min(end + 1, len(frames) - 1)]),
                                evidence={
                                    "max_visual_motion": float(np.max(visual[start : end + 1])),
                                    "median_state_delta": float(np.median(low_dim[start : end + 1])),
                                },
                            )
                        )

        for view in views:
            if view not in image_context or len(image_context[view]["motion"]) == 0 or len(delta) == 0:
                continue
            visual = image_context[view]["motion"]
            length = min(len(visual), len(delta))
            if length < 8:
                continue
            lag = best_lag(visual[:length], delta[:length])
            corr = lag["best_correlation"]
            best = lag["best_lag"]
            corr_stats = baseline_lookup(baselines, "sync_corr", task_index, view, column)
            lag_stats = baseline_lookup(baselines, "sync_lag", task_index, view, column)
            if corr is not None:
                corr_score = 100.0 * lower_outlier_score(float(corr), corr_stats, weak_z=3.0, strong_z=6.0)
                if corr < 0.1:
                    corr_score = max(corr_score, 45.0)
                if corr_score >= 30:
                    findings.append(
                        factory.make(
                            "vision_state_temporal",
                            "low_cross_modal_correlation",
                            "Vision-State motion correlation is lower than reference",
                            "episode",
                            "vision-state",
                            corr_score,
                            ood=corr_score < 55,
                            episode_index=episode,
                            task_index=task_index,
                            view=view,
                            column=column,
                            evidence={"best_lag": best, "best_correlation": corr, "correlation_baseline": corr_stats},
                        )
                    )
            if best is not None and lag_stats.get("count", 0) >= 5:
                expected_lag = round(stat_value(lag_stats, "median", float(best)))
                if abs(int(best) - expected_lag) >= 2:
                    findings.append(
                        factory.make(
                            "vision_state_temporal",
                            "cross_modal_lag_shift",
                            "Vision-State best lag deviates from reference lag",
                            "episode",
                            "vision-state",
                            min(85.0, 35.0 + 12.0 * abs(int(best) - expected_lag)),
                            ood=True,
                            episode_index=episode,
                            task_index=task_index,
                            view=view,
                            column=column,
                            evidence={"best_lag": best, "expected_lag": expected_lag, "lag_baseline": lag_stats},
                        )
                    )


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


def score_episode(meta: dict[str, Any], findings: list[Finding]) -> EpisodeResult:
    episode = meta["episode_index"]
    current = [finding for finding in findings if finding.episode_index == episode]
    penalties_by_dim: dict[str, float] = defaultdict(float)
    for finding in current:
        dimension = CATEGORY_TO_DIMENSION.get(finding.category_id, "state")
        penalties_by_dim[dimension] += finding.quality_penalty
    scores = {
        dimension: round(max(0.0, points - min(points, penalties_by_dim.get(dimension, 0.0))), 2)
        for dimension, points in DIMENSION_POINTS.items()
    }
    total = round(sum(scores.values()), 2)
    return EpisodeResult(
        episode_index=int(episode),
        task_index=meta.get("task_index"),
        length=int(meta.get("length", 0)),
        file=str(meta.get("file", "")),
        score_total=total,
        score_structural=scores["structural"],
        score_vision_single=scores["vision_single"],
        score_vision_vision=scores["vision_vision"],
        score_state=scores["state"],
        score_temporal=scores["temporal"],
        score_cross_modal=scores["cross_modal"],
        finding_count=len(current),
        critical_count=sum(1 for item in current if item.confidence_level == "确定异常"),
        high_confidence_count=sum(1 for item in current if item.confidence_level == "高置信异常"),
        suspicious_count=sum(1 for item in current if item.confidence_level == "疑似异常"),
        ood_count=sum(1 for item in current if item.confidence_level == "分布外样本"),
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
        },
        "scoring": {
            "dimension_points": DIMENSION_POINTS,
            "episode_score": "100 - merged finding penalties, capped by dimension",
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
        "",
        "## Detection Framework",
        "",
        "The v2 pipeline combines hard legality checks, single-view image quality, low-dimensional state checks, temporal checks, and cross-modal consistency checks.",
        "Direct Vision-Vision findings are disabled. Multi-view visual agreement is checked only after a state-supported overlap gate, and those findings are categorized as Vision-State-Vision.",
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
        "| episode | task | length | structural | vision_single | vision_vision | state | temporal | cross_modal | total | findings |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(episode_results, key=lambda item: item.episode_index):
        detail_lines.append(
            "| "
            f"{result.episode_index} | {result.task_index} | {result.length} | "
            f"{result.score_structural:g} | {result.score_vision_single:g} | {result.score_vision_vision:g} | "
            f"{result.score_state:g} | {result.score_temporal:g} | {result.score_cross_modal:g} | "
            f"{result.score_total:g} | {result.finding_count} |"
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
                (
                    "- breakdown: "
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


def main() -> None:
    info, tasks, episodes, parquet_files = load_dataset()
    episodes_meta = {int(row["episode_index"]): row for row in episodes}
    views = image_views_from_info(info)
    factory = FindingFactory()

    baselines = collect_reference_baselines(info, episodes_meta, parquet_files, views)
    expected_low_dim_shape = {}
    for column in LOW_DIM_COLUMNS:
        spec = info.get("features", {}).get(column)
        if spec and spec.get("shape"):
            expected_low_dim_shape[column] = int(spec["shape"][0])
    baselines["expected_low_dim_shape"] = expected_low_dim_shape

    episode_metas: list[dict[str, Any]] = []
    all_findings: list[Finding] = []
    for path in parquet_files:
        episode_from_path = safe_episode_from_path(path)
        expected_meta = episodes_meta.get(episode_from_path) if episode_from_path is not None else None
        episode_meta, findings = inspect_episode(path, info, expected_meta, views, baselines, factory)
        if episode_meta["episode_index"] is None and episode_from_path is not None:
            episode_meta["episode_index"] = episode_from_path
        episode_metas.append(episode_meta)
        all_findings.extend(findings)

    merged_findings = merge_findings(all_findings)
    episode_results = [score_episode(meta, merged_findings) for meta in sorted(episode_metas, key=lambda item: item["episode_index"])]
    write_reports(info, tasks, baselines, merged_findings, episode_results)

    dataset_score = round(float(np.mean([item.score_total for item in episode_results])) if episode_results else 0.0, 2)
    print(f"v2 episodes_checked: {len(episode_results)}")
    print(f"v2 merged_findings: {len(merged_findings)}")
    print(f"v2 dataset_quality_score: {dataset_score}/100")
    print(f"wrote: {DIAG_ROOT / 'findings_v2.json'}")
    print(f"wrote: {REPORT_ROOT / 'dataset_quality_report_v2.md'}")


if __name__ == "__main__":
    main()
