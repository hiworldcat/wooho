from __future__ import annotations

import io
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - optional for pure geometry tests
    pq = None
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


EPS = 1e-9
DEFAULT_FPS = 10.0
MIN_RUN_FRAMES = 11
MAX_LAG = 5
LAG_STABILITY_WINDOW = 21
LAG_STABILITY_STEP = 5
JITTER_WINDOW_FRAMES = 11
JITTER_MIN_DIRECTION_REVERSALS = 3
JITTER_MAX_PATH_EFFICIENCY = 0.35
JITTER_MIN_PATH_LENGTH_FACTOR = 4.0

MODULE_STATUS_VALUES = ("ok", "warning", "unavailable", "fail")
MODULE_STATUS_SET = frozenset(MODULE_STATUS_VALUES)


def normalize_module_status(value: Any) -> str:
    status = str(value or "ok")
    return status if status in MODULE_STATUS_SET else "fail"


def collect_module_statuses(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "arms": {
            arm: normalize_module_status(item.get("status", "ok"))
            for arm, item in dict(diagnostics.get("arms", {})).items()
            if isinstance(item, dict)
        },
        "bimanual": normalize_module_status(
            diagnostics.get("bimanual", {}).get("status", "ok")
            if isinstance(diagnostics.get("bimanual"), dict)
            else "ok"
        ),
        "state_vision": {
            arm: normalize_module_status(item.get("status", "ok"))
            for arm, item in dict(diagnostics.get("state_vision", {})).items()
            if isinstance(item, dict)
        },
    }


def unavailable_module_statuses(reason: str) -> dict[str, Any]:
    return {
        "arms": {arm: "unavailable" for arm in ARM_SPECS},
        "bimanual": "unavailable",
        "state_vision": {arm: "unavailable" for arm in ARM_SPECS},
    }


def iter_module_statuses(module_statuses: dict[str, Any]):
    for value in module_statuses.values():
        if isinstance(value, dict):
            for nested in value.values():
                if isinstance(nested, dict):
                    yield normalize_module_status(nested.get("status", "ok"))
                else:
                    yield normalize_module_status(nested)
        else:
            yield normalize_module_status(value)


def aggregate_module_statuses(core_status: str, module_statuses: dict[str, Any]) -> str:
    core = normalize_module_status(core_status)
    if core in {"fail", "unavailable"}:
        return core
    statuses = list(iter_module_statuses(module_statuses))
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status in {"warning", "unavailable"} for status in statuses):
        return "warning"
    return "ok"


def summarize_geometry_status(diagnostics: dict[str, Any]) -> str:
    module_statuses = diagnostics.get("module_statuses")
    if not isinstance(module_statuses, dict):
        module_statuses = collect_module_statuses(diagnostics)
    return aggregate_module_statuses(str(diagnostics.get("core_status", diagnostics.get("status", "ok"))), module_statuses)

@dataclass(frozen=True)
class ArmSpec:
    position: slice
    rotation_6d: slice
    gripper: int
    wrist_view: str


ARM_SPECS = {
    "left": ArmSpec(slice(0, 3), slice(3, 9), 9, "left_wrist_image"),
    "right": ArmSpec(slice(10, 13), slice(13, 19), 19, "right_wrist_image"),
}


def default_geometry_config(info: dict[str, Any] | None = None) -> dict[str, Any]:
    fps = DEFAULT_FPS
    if info and info.get("fps"):
        try:
            fps = float(info["fps"])
        except Exception:
            fps = DEFAULT_FPS
    return {
        "fps": fps,
        "state_frame_mode": "unknown",
        "position_unit": "unknown",
        "rotation6d_layout": "unknown",
        "rotation6d_min_norm": 1e-4,
        "rotation6d_min_sin_angle": 1e-3,
        "bimanual_enabled": True,
        "calibration_mode": "none",
        "state_vision_enabled": True,
        "max_lag": MAX_LAG,
        "vision_feature_max_frames": None,
        "lag_stability_window": LAG_STABILITY_WINDOW,
        "lag_stability_step": LAG_STABILITY_STEP,
        "jitter_window_frames": JITTER_WINDOW_FRAMES,
        "jitter_min_direction_reversals": JITTER_MIN_DIRECTION_REVERSALS,
        "jitter_max_path_efficiency": JITTER_MAX_PATH_EFFICIENCY,
        "jitter_min_path_length_factor": JITTER_MIN_PATH_LENGTH_FACTOR,
        "hardware_max_linear_speed": None,
        "hardware_max_linear_accel": None,
    }


def validate_geometry_config(config: dict[str, Any]) -> dict[str, Any]:
    validated = dict(config)
    state_frame_mode = str(validated.get("state_frame_mode", "unknown"))
    rotation_layout = str(validated.get("rotation6d_layout", "unknown"))
    position_unit = str(validated.get("position_unit", "unknown"))
    if state_frame_mode not in {"common_world", "per_arm_base", "unknown"}:
        raise ValueError(f"invalid state_frame_mode: {state_frame_mode}")
    if rotation_layout not in {"columns", "rows", "unknown"}:
        raise ValueError(f"invalid rotation6d_layout: {rotation_layout}")
    if position_unit not in {"meter", "millimeter", "dataset_unit", "unknown"}:
        raise ValueError(f"invalid position_unit: {position_unit}")
    validated["state_frame_mode"] = state_frame_mode
    validated["rotation6d_layout"] = rotation_layout
    validated["position_unit"] = position_unit
    validated["max_lag"] = max(0, int(validated.get("max_lag", MAX_LAG)))
    validated["lag_stability_window"] = max(MIN_RUN_FRAMES, int(validated.get("lag_stability_window", LAG_STABILITY_WINDOW)))
    validated["lag_stability_step"] = max(1, int(validated.get("lag_stability_step", LAG_STABILITY_STEP)))
    validated["jitter_window_frames"] = max(MIN_RUN_FRAMES, int(validated.get("jitter_window_frames", JITTER_WINDOW_FRAMES)))
    validated["jitter_min_direction_reversals"] = max(2, int(validated.get("jitter_min_direction_reversals", JITTER_MIN_DIRECTION_REVERSALS)))
    validated["jitter_max_path_efficiency"] = min(0.95, max(0.05, float(validated.get("jitter_max_path_efficiency", JITTER_MAX_PATH_EFFICIENCY))))
    validated["jitter_min_path_length_factor"] = max(1.0, float(validated.get("jitter_min_path_length_factor", JITTER_MIN_PATH_LENGTH_FACTOR)))
    for key in ["hardware_max_linear_speed", "hardware_max_linear_accel"]:
        value = validated.get(key)
        if value is not None:
            value = float(value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be a positive finite number or null")
            validated[key] = value
    return validated


def load_geometry_config(path: Path | str, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(base_config or default_geometry_config())
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("geometry config must be a JSON object")
    config.update(payload)
    return validate_geometry_config(config)


def finite_float_matrix(values: Any) -> np.ndarray | None:
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
    matrix = np.stack(rows)
    if matrix.ndim != 2:
        return None
    return matrix


def robust_stats(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
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
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def stat(stats: dict[str, Any], key: str, default: float) -> float:
    value = stats.get(key)
    if value is None:
        return default
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def robust_z(value: float, stats: dict[str, Any]) -> float:
    if stats.get("count", 0) < 5:
        return 0.0
    center = stat(stats, "median", value)
    scale = max(1.4826 * stat(stats, "mad", 0.0), stat(stats, "std", 0.0), EPS)
    return abs(float(value) - center) / scale


def score_from_z(z: float, weak: float = 4.0, strong: float = 8.0) -> float:
    if not np.isfinite(z) or z <= weak:
        return 0.0
    if z >= strong:
        return 100.0
    return float(100.0 * (z - weak) / max(strong - weak, EPS))


def contiguous_ranges(indices: np.ndarray, max_gap: int = 1) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    values = np.sort(np.unique(indices.astype(int)))
    split_points = np.where(np.diff(values) > max_gap)[0]
    starts = np.r_[0, split_points + 1]
    ends = np.r_[split_points, len(values) - 1]
    return [(int(values[start]), int(values[end])) for start, end in zip(starts, ends)]


def normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < EPS:
        return None
    return vector / norm


def rotation6d_precheck(values: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    rot = np.asarray(values, dtype=np.float64)
    a1 = rot[:, :3]
    a2 = rot[:, 3:6]
    n1 = np.linalg.norm(a1, axis=1)
    n2 = np.linalg.norm(a2, axis=1)
    cross_norm = np.linalg.norm(np.cross(a1, a2), axis=1)
    sin_angle = cross_norm / np.maximum(n1 * n2, EPS)
    finite = np.isfinite(rot).all(axis=1)
    min_norm = float(config.get("rotation6d_min_norm", 1e-4))
    min_sin = float(config.get("rotation6d_min_sin_angle", 1e-3))
    legal = finite & (n1 > min_norm) & (n2 > min_norm) & (sin_angle > min_sin)
    return {
        "n1": n1,
        "n2": n2,
        "sin_angle": sin_angle,
        "finite": finite,
        "legal": legal,
    }


def rotation6d_to_matrix(value: np.ndarray, layout: str) -> np.ndarray | None:
    vec = np.asarray(value, dtype=np.float64).reshape(-1)[:6]
    if vec.size != 6 or not np.isfinite(vec).all():
        return None
    a1, a2 = vec[:3], vec[3:6]
    b1 = normalize(a1)
    if b1 is None:
        return None
    temp = a2 - float(np.dot(b1, a2)) * b1
    b2 = normalize(temp)
    if b2 is None:
        return None
    b3 = normalize(np.cross(b1, b2))
    if b3 is None:
        return None
    if layout == "columns":
        return np.stack([b1, b2, b3], axis=1)
    if layout == "rows":
        return np.stack([b1, b2, b3], axis=0)
    return None


def rotation_sequence(values: np.ndarray, legal: np.ndarray, layout: str) -> list[np.ndarray | None]:
    if layout not in {"columns", "rows"}:
        return [None for _ in range(len(values))]
    rotations: list[np.ndarray | None] = []
    for row, ok in zip(values, legal):
        rotations.append(rotation6d_to_matrix(row, layout) if bool(ok) else None)
    return rotations


def so3_distance(left: np.ndarray, right: np.ndarray) -> float:
    cos_theta = (float(np.trace(left.T @ right)) - 1.0) * 0.5
    return float(math.acos(max(-1.0, min(1.0, cos_theta))))


def angular_motion(rotations: list[np.ndarray | None], fps: float) -> np.ndarray:
    values: list[float] = []
    for left, right in zip(rotations[:-1], rotations[1:]):
        if left is None or right is None:
            values.append(float("nan"))
        else:
            values.append(so3_distance(left, right) * fps)
    return np.asarray(values, dtype=np.float64)


def arm_motion_features(matrix: np.ndarray, arm: str, config: dict[str, Any]) -> dict[str, Any]:
    spec = ARM_SPECS[arm]
    fps = float(config.get("fps", DEFAULT_FPS))
    pos = matrix[:, spec.position]
    rot6d = matrix[:, spec.rotation_6d]
    gripper = matrix[:, spec.gripper] if matrix.shape[1] > spec.gripper else np.full(len(matrix), np.nan)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1) if len(pos) > 1 else np.array([], dtype=np.float64)
    speed = step * fps
    accel = np.linalg.norm(pos[2:] - 2.0 * pos[1:-1] + pos[:-2], axis=1) * fps * fps if len(pos) > 2 else np.array([], dtype=np.float64)
    precheck = rotation6d_precheck(rot6d, config)
    rotations = rotation_sequence(rot6d, precheck["legal"], str(config.get("rotation6d_layout", "unknown")))
    omega = angular_motion(rotations, fps)
    return {
        "position": pos,
        "rotation6d": rot6d,
        "gripper": gripper,
        "step": step,
        "speed": speed,
        "accel": accel,
        "rotation_precheck": precheck,
        "rotations": rotations,
        "omega": omega,
    }


def oscillation_windows(
    positions: np.ndarray,
    window_frames: int,
    min_direction_reversals: int,
    max_path_efficiency: float,
    min_path_length: float,
) -> list[dict[str, Any]]:
    """Find sustained back-and-forth motion, not isolated acceleration spikes."""
    positions = np.asarray(positions, dtype=np.float64)
    window_frames = max(MIN_RUN_FRAMES, int(window_frames))
    if positions.ndim != 2 or len(positions) < window_frames:
        return []

    candidates: list[dict[str, Any]] = []
    for start in range(0, len(positions) - window_frames + 1):
        window = positions[start : start + window_frames]
        steps = np.diff(window, axis=0)
        step_norms = np.linalg.norm(steps, axis=1)
        finite = np.isfinite(steps).all(axis=1) & np.isfinite(step_norms)
        if finite.sum() < window_frames - 1:
            continue
        valid_steps = steps[finite]
        valid_norms = step_norms[finite]
        path_length = float(np.sum(valid_norms))
        if path_length < min_path_length:
            continue
        net_displacement = float(np.linalg.norm(window[-1] - window[0]))
        path_efficiency = net_displacement / max(path_length, EPS)
        directions = valid_steps / np.maximum(valid_norms[:, None], EPS)
        reversals = int(np.sum(np.sum(directions[:-1] * directions[1:], axis=1) < -0.25))
        if reversals < min_direction_reversals or path_efficiency > max_path_efficiency:
            continue
        candidates.append(
            {
                "frame_start": int(start),
                "frame_end": int(start + window_frames - 1),
                "direction_reversals": reversals,
                "path_length": path_length,
                "net_displacement": net_displacement,
                "path_efficiency": path_efficiency,
                "min_path_length": float(min_path_length),
            }
        )
    return candidates


def bimanual_features(arms: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    state_frame_mode = str(config.get("state_frame_mode", "unknown"))
    if not config.get("bimanual_enabled", True):
        return {
            "status": "unavailable",
            "reason": "disabled_by_config",
            "state_frame_mode": state_frame_mode,
        }
    if state_frame_mode != "common_world":
        return {
            "status": "unavailable",
            "reason": "state_frame_mode_unknown" if state_frame_mode == "unknown" else "state_frame_mode_not_common_world",
            "state_frame_mode": state_frame_mode,
        }
    left = arms["left"]
    right = arms["right"]
    pos_delta = right["position"] - left["position"]
    distance = np.linalg.norm(pos_delta, axis=1)
    rel_pos_local: list[np.ndarray | None] = []
    rel_rot: list[np.ndarray | None] = []
    for idx, (left_r, right_r) in enumerate(zip(left["rotations"], right["rotations"])):
        if left_r is None or right_r is None:
            rel_pos_local.append(None)
            rel_rot.append(None)
            continue
        rel_pos_local.append(left_r.T @ pos_delta[idx])
        rel_rot.append(left_r.T @ right_r)
    rel_pos_arr = np.asarray(
        [row if row is not None else np.full(3, np.nan) for row in rel_pos_local],
        dtype=np.float64,
    )
    return {
        "status": "ok",
        "distance": distance,
        "relative_position_left_frame": rel_pos_arr,
        "relative_rotation": rel_rot,
    }


def decode_gray(value: Any) -> np.ndarray:
    if not isinstance(value, dict) or not value.get("bytes"):
        raise ValueError("image cell does not contain embedded bytes")
    with Image.open(io.BytesIO(value["bytes"])) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    return gray


def feature_motion_from_view(values: Any, max_frames: int | None = None) -> dict[str, Any]:
    if cv2 is None:
        return {"status": "unavailable", "reason": "opencv_unavailable", "motion": np.array([], dtype=np.float64)}
    motions: list[float] = []
    track_counts: list[int] = []
    inlier_counts: list[int] = []
    duplicate_flags: list[bool] = []
    previous: np.ndarray | None = None
    previous_hash: str | None = None
    frame_limit = len(values) if max_frames is None else min(len(values), max_frames)
    for value in list(values)[:frame_limit]:
        try:
            gray = decode_gray(value)
        except Exception:
            previous = None
            previous_hash = None
            continue
        if max(gray.shape[:2]) > 320:
            scale = 320.0 / float(max(gray.shape[:2]))
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        current_hash = __import__("hashlib").sha1(gray.tobytes()).hexdigest()
        if previous_hash is not None:
            duplicate_flags.append(current_hash == previous_hash)
        previous_hash = current_hash
        if previous is None:
            previous = gray
            continue
        points = cv2.goodFeaturesToTrack(previous, maxCorners=160, qualityLevel=0.01, minDistance=7)
        if points is None or len(points) < 12:
            motions.append(float("nan"))
            track_counts.append(0)
            inlier_counts.append(0)
            previous = gray
            continue
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
        if next_points is None or status is None:
            motions.append(float("nan"))
            track_counts.append(0)
            inlier_counts.append(0)
            previous = gray
            continue
        keep = status.reshape(-1).astype(bool)
        src = points.reshape(-1, 2)[keep]
        dst = next_points.reshape(-1, 2)[keep]
        track_counts.append(int(len(src)))
        if len(src) < 8:
            motions.append(float("nan"))
            inlier_counts.append(0)
            previous = gray
            continue
        _, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if inliers is not None and int(inliers.sum()) >= 6:
            inlier_mask = inliers.reshape(-1).astype(bool)
            displacement = np.linalg.norm(dst[inlier_mask] - src[inlier_mask], axis=1)
            inlier_counts.append(int(inlier_mask.sum()))
        else:
            displacement = np.linalg.norm(dst - src, axis=1)
            inlier_counts.append(0)
        motions.append(float(np.median(displacement)) if displacement.size else float("nan"))
        previous = gray
    motion = np.asarray(motions, dtype=np.float64)
    valid = np.isfinite(motion)
    return {
        "status": "ok" if valid.sum() >= 8 else "unavailable",
        "reason": "feature_tracks" if valid.sum() >= 8 else "too_few_feature_tracks",
        "motion": motion,
        "background_motion": motion,
        "valid_fraction": float(valid.mean()) if motion.size else 0.0,
        "median_track_count": float(np.median(track_counts)) if track_counts else 0.0,
        "median_inlier_count": float(np.median(inlier_counts)) if inlier_counts else 0.0,
        "duplicate_frame_fraction": float(np.mean(duplicate_flags)) if duplicate_flags else 0.0,
    }


def fallback_visual_motion(image_context: dict[str, Any], view: str) -> dict[str, Any]:
    item = image_context.get(view, {})
    motion = np.asarray(item.get("motion", []), dtype=np.float64)
    if motion.size == 0:
        return {"status": "unavailable", "reason": "visual_motion_unavailable", "motion": motion}
    return {
        "status": "unavailable",
        "reason": "feature_tracks_required",
        "motion": motion,
        "valid_fraction": float(np.isfinite(motion).mean()),
    }


def lagged_corr(left: np.ndarray, right: np.ndarray, lag: int) -> float | None:
    if lag > 0:
        a, b = left[lag:], right[:-lag]
    elif lag < 0:
        a, b = left[:lag], right[-lag:]
    else:
        a, b = left, right
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 8 or np.std(a) < EPS or np.std(b) < EPS:
        return None
    corr = float(np.corrcoef(a, b)[0, 1])
    return corr if np.isfinite(corr) else None


def lag_correlation_profile(visual: np.ndarray, state: np.ndarray, max_lag: int) -> dict[str, Any]:
    length = min(len(visual), len(state))
    visual = np.asarray(visual[:length], dtype=np.float64)
    state = np.asarray(state[:length], dtype=np.float64)
    valid = np.isfinite(visual) & np.isfinite(state)
    if valid.sum() < 12:
        return {"status": "unavailable", "reason": "too_few_valid_samples"}
    v_excitation = float(np.nanpercentile(visual, 90) - np.nanpercentile(visual, 10))
    s_excitation = float(np.nanpercentile(state, 90) - np.nanpercentile(state, 10))
    if v_excitation <= EPS or s_excitation <= EPS:
        return {"status": "unavailable", "reason": "low_motion_excitation"}
    visual_norm = (visual - np.nanmedian(visual)) / max(float(np.nanstd(visual)), EPS)
    state_norm = (state - np.nanmedian(state)) / max(float(np.nanstd(state)), EPS)
    correlations: dict[str, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        corr = lagged_corr(visual_norm, state_norm, lag)
        if corr is not None:
            correlations[str(lag)] = corr
    if not correlations:
        return {"status": "unavailable", "reason": "correlation_unavailable"}
    ranked = sorted(correlations.items(), key=lambda item: item[1], reverse=True)
    best_lag, best_corr = int(ranked[0][0]), float(ranked[0][1])
    second_corr = float(ranked[1][1]) if len(ranked) > 1 else float("-inf")
    peak_margin = best_corr - second_corr
    identifiable = best_corr >= 0.35 and peak_margin >= 0.08
    return {
        "status": "ok" if identifiable else "unavailable",
        "reason": "identifiable_peak" if identifiable else "ambiguous_or_weak_peak",
        "best_lag": best_lag,
        "best_corr": best_corr,
        "second_best_corr": second_corr if np.isfinite(second_corr) else None,
        "peak_margin": float(peak_margin) if np.isfinite(peak_margin) else None,
        "valid_fraction": float(valid.mean()),
        "visual_excitation": v_excitation,
        "state_excitation": s_excitation,
        "correlations": correlations,
    }


def lag_stability_report(visual: np.ndarray, state: np.ndarray, max_lag: int, window_size: int, step: int) -> dict[str, Any]:
    length = min(len(visual), len(state))
    if length < MIN_RUN_FRAMES:
        return {"status": "unavailable", "reason": "too_few_samples", "window_count": 0, "valid_window_count": 0}

    window_size = max(MIN_RUN_FRAMES, min(int(window_size), length))
    step = max(1, int(step))
    if length < window_size:
        return {"status": "unavailable", "reason": "window_too_long", "window_count": 0, "valid_window_count": 0}

    candidates: list[dict[str, Any]] = []
    total_windows = 0
    for start in range(0, length - window_size + 1, step):
        total_windows += 1
        window = lag_correlation_profile(visual[start : start + window_size], state[start : start + window_size], max_lag)
        if window.get("status") == "ok":
            candidates.append(window)

    if len(candidates) < 3:
        return {
            "status": "unavailable",
            "reason": "too_few_stable_windows",
            "window_count": int(total_windows),
            "valid_window_count": int(len(candidates)),
            "valid_window_fraction": float(len(candidates) / total_windows) if total_windows else 0.0,
        }

    lags = np.asarray([item["best_lag"] for item in candidates], dtype=np.float64)
    corrs = np.asarray([item["best_corr"] for item in candidates], dtype=np.float64)
    lag_median = float(np.median(lags))
    lag_mad = float(np.median(np.abs(lags - lag_median)))
    corr_median = float(np.median(corrs))
    stable = lag_mad <= 1.0 and corr_median >= 0.35
    return {
        "status": "ok" if stable else "unavailable",
        "reason": "stable_peak" if stable else "unstable_peak",
        "window_count": int(total_windows),
        "valid_window_count": int(len(candidates)),
        "valid_window_fraction": float(len(candidates) / total_windows) if total_windows else 0.0,
        "lag_median": lag_median,
        "lag_mad": lag_mad,
        "corr_median": corr_median,
        "window_lags": [int(item["best_lag"]) for item in candidates],
    }


def identifiable_lag(
    visual: np.ndarray,
    state: np.ndarray,
    max_lag: int,
    window_size: int = LAG_STABILITY_WINDOW,
    step: int = LAG_STABILITY_STEP,
) -> dict[str, Any]:
    profile = lag_correlation_profile(visual, state, max_lag)
    if profile.get("status") != "ok":
        return profile
    stability = lag_stability_report(
        visual,
        state,
        max_lag,
        int(window_size),
        int(step),
    )
    profile["lag_stability"] = stability
    if stability.get("status") != "ok":
        profile["status"] = "unavailable"
        profile["reason"] = stability.get("reason", "unstable_peak")
    return profile


def fit_geometry_reference(
    parquet_files: list[Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    config = validate_geometry_config(config)
    if pq is None:
        return {
            "config": config,
            "stats": {},
            "episodes_used": 0,
            "episode_summaries": [],
            "status": "unavailable",
            "reason": "pyarrow_unavailable",
        }
    buckets: dict[str, list[float]] = defaultdict(list)
    episode_summaries: list[dict[str, Any]] = []
    for path in parquet_files:
        try:
            schema = pq.read_schema(path).names
            if "state" not in schema:
                continue
            df = pq.read_table(path, columns=["state", "episode_index", "task_index"]).to_pandas()
        except Exception:
            continue
        matrix = finite_float_matrix(df["state"])
        if matrix is None or matrix.shape[1] < 20 or not np.isfinite(matrix).all():
            continue
        arms = {arm: arm_motion_features(matrix, arm, config) for arm in ARM_SPECS}
        episode = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns and len(df) else None
        task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns and len(df) else None
        for arm, features in arms.items():
            prefix = f"task:{task_index}|{arm}"
            for scope in [f"global|{arm}", prefix]:
                buckets[f"{scope}|position_step"].extend(features["step"][np.isfinite(features["step"])].tolist())
                buckets[f"{scope}|linear_speed"].extend(features["speed"][np.isfinite(features["speed"])].tolist())
                buckets[f"{scope}|linear_accel"].extend(features["accel"][np.isfinite(features["accel"])].tolist())
                buckets[f"{scope}|angular_speed"].extend(features["omega"][np.isfinite(features["omega"])].tolist())
                buckets[f"{scope}|rot6d_n1"].extend(features["rotation_precheck"]["n1"][np.isfinite(features["rotation_precheck"]["n1"])].tolist())
                buckets[f"{scope}|rot6d_n2"].extend(features["rotation_precheck"]["n2"][np.isfinite(features["rotation_precheck"]["n2"])].tolist())
                buckets[f"{scope}|rot6d_sin_angle"].extend(features["rotation_precheck"]["sin_angle"][np.isfinite(features["rotation_precheck"]["sin_angle"])].tolist())
        bimanual = bimanual_features(arms, config)
        if bimanual.get("status") == "ok":
            distance = bimanual["distance"]
            for scope in ["global|bimanual", f"task:{task_index}|bimanual"]:
                buckets[f"{scope}|eef_distance"].extend(distance[np.isfinite(distance)].tolist())
                if len(distance) > 1:
                    drift = np.abs(distance - np.nanmedian(distance))
                    buckets[f"{scope}|distance_drift"].extend(drift[np.isfinite(drift)].tolist())
        episode_summaries.append(
            {
                "episode_index": episode,
                "task_index": task_index,
                "length": int(len(matrix)),
                "rotation6d_legal_fraction": {
                    arm: float(np.mean(features["rotation_precheck"]["legal"])) for arm, features in arms.items()
                },
            }
        )
    return {
        "config": config,
        "status": "ok",
        "stats": {key: robust_stats(values) for key, values in buckets.items()},
        "episodes_used": len(episode_summaries),
        "episode_summaries": episode_summaries,
        "policy": {
            "rotation6d": "precheck finite values, vector norms, and non-collinearity before SO(3) conversion",
            "bimanual": "enabled only when state_frame_mode is common_world",
            "state_vision": "uses feature-track motion and duplicate-frame gating when OpenCV is available; otherwise reports unavailable",
            "calibration": "strict reprojection, epipolar, and triangulation are disabled when calibration_mode is none",
        },
    }


def stats_lookup(reference: dict[str, Any], task_index: int | None, group: str, metric: str) -> dict[str, Any]:
    stats = reference.get("stats", {})
    task_key = f"task:{task_index}|{group}|{metric}"
    global_key = f"global|{group}|{metric}"
    task_stats = stats.get(task_key)
    if isinstance(task_stats, dict) and task_stats.get("count", 0) >= 5:
        return task_stats
    global_stats = stats.get(global_key)
    if isinstance(global_stats, dict):
        return global_stats
    return {"count": 0}


def inspect_episode_geometry(
    df: Any,
    frames: np.ndarray,
    episode: int | None,
    task_index: int | None,
    views: list[str],
    image_context: dict[str, dict[str, Any]],
    reference: dict[str, Any],
    config: dict[str, Any],
    factory: Any,
) -> dict[str, Any]:
    findings: list[Any] = []
    config = validate_geometry_config(config)
    diagnostics: dict[str, Any] = {
        "episode_index": episode,
        "task_index": task_index,
        "status": "ok",
        "core_status": "ok",
        "status_enum": list(MODULE_STATUS_VALUES),
        "status_policy": {
            "ok": "core and enabled modules completed",
            "warning": "core completed but at least one submodule is degraded or unavailable",
            "unavailable": "core geometry could not run",
            "fail": "a module reported an execution-level failure",
        },
        "gates": {
            "rotation6d_layout": config.get("rotation6d_layout", "unknown"),
            "state_frame_mode": config.get("state_frame_mode", "unknown"),
            "calibration_mode": config.get("calibration_mode", "none"),
            "position_unit": config.get("position_unit", "unknown"),
            "temporal_alignment": "performed_in_geometry_module",
            "strict_calibrated_geometry": config.get("calibration_mode", "none") != "none",
        },
        "arms": {},
        "bimanual": {},
        "state_vision": {},
    }
    if "state" not in df.columns:
        diagnostics["core_status"] = "unavailable"
        diagnostics["status"] = "unavailable"
        diagnostics["reason"] = "state_column_missing"
        diagnostics["module_statuses"] = unavailable_module_statuses("core_unavailable")
        return {"findings": findings, "diagnostics": diagnostics}
    matrix = finite_float_matrix(df["state"])
    if matrix is None or matrix.ndim != 2 or matrix.shape[1] < 20:
        diagnostics["core_status"] = "unavailable"
        diagnostics["status"] = "unavailable"
        diagnostics["reason"] = "state_matrix_invalid"
        diagnostics["module_statuses"] = unavailable_module_statuses("core_unavailable")
        return {"findings": findings, "diagnostics": diagnostics}
    if len(matrix) != len(frames):
        diagnostics["core_status"] = "unavailable"
        diagnostics["status"] = "unavailable"
        diagnostics["reason"] = "state_frame_length_mismatch"
        diagnostics["module_statuses"] = unavailable_module_statuses("core_unavailable")
        return {"findings": findings, "diagnostics": diagnostics}

    arms = {arm: arm_motion_features(matrix, arm, config) for arm in ARM_SPECS}
    for arm, features in arms.items():
        precheck = features["rotation_precheck"]
        legal = precheck["legal"]
        diagnostics["arms"][arm] = {
            "rotation6d_legal_fraction": float(np.mean(legal)) if len(legal) else None,
            "median_step": float(np.nanmedian(features["step"])) if len(features["step"]) else 0.0,
            "median_speed": float(np.nanmedian(features["speed"])) if len(features["speed"]) else 0.0,
            "median_angular_speed": float(np.nanmedian(features["omega"])) if np.isfinite(features["omega"]).any() else None,
            "status": "ok",
        }
        bad = np.where(~legal)[0]
        for start, end in contiguous_ranges(frames[bad], max_gap=1):
            local = np.where((frames >= start) & (frames <= end))[0]
            findings.append(
                factory.make(
                    "state_illegal",
                    "geometry_rotation6d_degenerate",
                    "Rotation 6D input is non-finite, near-zero, or near-collinear",
                    "segment",
                    "state-geometry",
                    100,
                    illegal=True,
                    episode_index=episode,
                    task_index=task_index,
                    column=f"state.{arm}.rotation6d",
                    frame_start=int(start),
                    frame_end=int(end),
                    evidence={
                        "arm": arm,
                        "min_n1": float(np.nanmin(precheck["n1"][local])) if local.size else None,
                        "min_n2": float(np.nanmin(precheck["n2"][local])) if local.size else None,
                        "min_sin_angle": float(np.nanmin(precheck["sin_angle"][local])) if local.size else None,
                        "gate": diagnostics["gates"],
                    },
                )
            )

        # Ordinary speed and acceleration changes are expected during task execution.
        # Only sustained oscillation, or an explicitly configured hardware limit, is reported.
        finite_steps = features["step"][np.isfinite(features["step"])]
        median_step = float(np.median(finite_steps)) if finite_steps.size else 0.0
        jitter_cfg = {
            "window_frames": int(config.get("jitter_window_frames", JITTER_WINDOW_FRAMES)),
            "min_direction_reversals": int(config.get("jitter_min_direction_reversals", JITTER_MIN_DIRECTION_REVERSALS)),
            "max_path_efficiency": float(config.get("jitter_max_path_efficiency", JITTER_MAX_PATH_EFFICIENCY)),
        }
        min_path_length = max(
            EPS,
            median_step
            * jitter_cfg["window_frames"]
            * float(config.get("jitter_min_path_length_factor", JITTER_MIN_PATH_LENGTH_FACTOR)),
        )
        oscillations = oscillation_windows(
            features["position"],
            jitter_cfg["window_frames"],
            jitter_cfg["min_direction_reversals"],
            jitter_cfg["max_path_efficiency"],
            min_path_length,
        )
        for start, end in contiguous_ranges(
            np.asarray([item["frame_start"] for item in oscillations], dtype=np.int64),
            max_gap=2,
        ):
            matching = [item for item in oscillations if start <= item["frame_start"] <= end]
            strongest = max(matching, key=lambda item: (item["direction_reversals"], item["path_length"]))
            strength = min(
                100.0,
                55.0 + 8.0 * max(0, strongest["direction_reversals"] - jitter_cfg["min_direction_reversals"]),
            )
            findings.append(
                factory.make(
                    "state_temporal",
                    f"geometry_{arm}_oscillatory_jitter",
                    "Arm position exhibits sustained back-and-forth oscillation",
                    "segment",
                    "state-geometry",
                    strength,
                    ood=strength < 70.0,
                    episode_index=episode,
                    task_index=task_index,
                    column=f"state.{arm}.position",
                    frame_start=int(start),
                    frame_end=int(min(len(frames) - 1, end + jitter_cfg["window_frames"] - 1)),
                    evidence={
                        "arm": arm,
                        "metric": "oscillatory_jitter",
                        "window_frames": jitter_cfg["window_frames"],
                        "direction_reversals": strongest["direction_reversals"],
                        "path_length": strongest["path_length"],
                        "net_displacement": strongest["net_displacement"],
                        "path_efficiency": strongest["path_efficiency"],
                        "min_path_length": strongest["min_path_length"],
                        "gate": diagnostics["gates"],
                    },
                )
            )

        for metric, values, limit_key in [
            ("linear_speed", features["speed"], "hardware_max_linear_speed"),
            ("linear_accel", features["accel"], "hardware_max_linear_accel"),
        ]:
            hardware_limit = config.get(limit_key)
            if hardware_limit is None:
                continue
            bad_metric = np.where(np.isfinite(values) & (values > float(hardware_limit)))[0]
            for start_i, end_i in contiguous_ranges(bad_metric, max_gap=1):
                max_value = float(np.nanmax(values[start_i : end_i + 1]))
                score = min(100.0, max(60.0, 60.0 + 40.0 * (max_value / float(hardware_limit) - 1.0)))
                findings.append(
                    factory.make(
                        "state_temporal",
                        f"geometry_{arm}_{metric}_hardware_limit_exceeded",
                        "Arm motion exceeds an explicitly configured hardware limit",
                        "segment",
                        "state-geometry",
                        score,
                        ood=False,
                        episode_index=episode,
                        task_index=task_index,
                        column=f"state.{arm}.{metric}",
                        frame_start=int(frames[start_i]),
                        frame_end=int(frames[min(end_i + 1, len(frames) - 1)]),
                        evidence={
                            "arm": arm,
                            "metric": metric,
                            "max_value": max_value,
                            "hardware_limit": float(hardware_limit),
                            "gate": diagnostics["gates"],
                        },
                    )
                )

    bimanual = bimanual_features(arms, config)
    if bimanual.get("status") != "ok":
        diagnostics["bimanual"] = bimanual
    else:
        distance = bimanual["distance"]
        drift = np.abs(distance - np.nanmedian(distance))
        drift_stats = stats_lookup(reference, task_index, "bimanual", "distance_drift")
        diagnostics["bimanual"] = {
            "status": "ok",
            "median_eef_distance": float(np.nanmedian(distance)),
            "max_distance_drift": float(np.nanmax(drift)) if drift.size else 0.0,
            "reference_count": drift_stats.get("count", 0),
        }
        if drift_stats.get("count", 0) >= 5:
            threshold = max(
                stat(drift_stats, "p99", float(np.nanmax(drift))) * 1.4,
                stat(drift_stats, "median", 0.0) + 8.0 * stat(drift_stats, "mad", 0.0),
            )
            bad_drift = np.where(np.isfinite(drift) & (drift > threshold))[0]
            for start, end in contiguous_ranges(bad_drift, max_gap=2):
                max_drift = float(np.nanmax(drift[start : end + 1]))
                score = max(40.0, score_from_z(robust_z(max_drift, drift_stats), weak=4.0, strong=8.0))
                findings.append(
                    factory.make(
                        "state_vision_state",
                        "geometry_bimanual_relative_distance_drift",
                        "Bimanual relative end-effector distance drifts during one episode",
                        "segment",
                        "state-geometry",
                        score,
                        ood=score < 60.0,
                        episode_index=episode,
                        task_index=task_index,
                        column="state.bimanual.relative_distance",
                        frame_start=int(frames[start]),
                        frame_end=int(frames[end]),
                        evidence={
                            "max_distance_drift": max_drift,
                            "threshold": threshold,
                            "reference": drift_stats,
                            "gate": diagnostics["gates"],
                        },
                    )
                )

    if config.get("state_vision_enabled", True):
        inspect_state_vision_response(
            df,
            frames,
            episode,
            task_index,
            views,
            image_context,
            arms,
            config,
            factory,
            findings,
            diagnostics,
        )
    else:
        for arm, spec in ARM_SPECS.items():
            diagnostics["state_vision"][arm] = {
                "status": "unavailable",
                "reason": "disabled_by_config",
                "view": spec.wrist_view,
            }
    diagnostics["module_statuses"] = collect_module_statuses(diagnostics)
    diagnostics["status"] = summarize_geometry_status(diagnostics)
    if diagnostics["status"] == "warning" and "reason" not in diagnostics:
        diagnostics["reason"] = "partial_module_unavailable"
    return {"findings": findings, "diagnostics": diagnostics}


def inspect_state_vision_response(
    df: Any,
    frames: np.ndarray,
    episode: int | None,
    task_index: int | None,
    views: list[str],
    image_context: dict[str, dict[str, Any]],
    arms: dict[str, dict[str, Any]],
    config: dict[str, Any],
    factory: Any,
    findings: list[Any],
    diagnostics: dict[str, Any],
) -> None:
    max_lag = int(config.get("max_lag", MAX_LAG))
    for arm, spec in ARM_SPECS.items():
        view = spec.wrist_view
        if view not in views or view not in df.columns:
            diagnostics["state_vision"][arm] = {"status": "unavailable", "reason": "wrist_view_missing", "view": view}
            continue
        max_frames = config.get("vision_feature_max_frames")
        max_frames = int(max_frames) if max_frames is not None else None
        feature_result = feature_motion_from_view(df[view], max_frames=max_frames)
        visual = np.asarray(feature_result.get("motion", []), dtype=np.float64)
        state_motion = np.asarray(arms[arm]["speed"], dtype=np.float64)
        lag = identifiable_lag(
            visual,
            state_motion,
            max_lag,
            window_size=int(config.get("lag_stability_window", LAG_STABILITY_WINDOW)),
            step=int(config.get("lag_stability_step", LAG_STABILITY_STEP)),
        )
        diagnostics["state_vision"][arm] = {
            "view": view,
            "visual_motion_status": feature_result.get("status"),
            "visual_motion_reason": feature_result.get("reason"),
            "duplicate_frame_fraction": float(feature_result.get("duplicate_frame_fraction", 0.0)),
            "lag": lag,
            "status": "ok",
        }
        if feature_result.get("status") != "ok" or float(feature_result.get("duplicate_frame_fraction", 0.0)) >= 0.35:
            diagnostics["state_vision"][arm]["status"] = "unavailable"
            diagnostics["state_vision"][arm]["reason"] = feature_result.get("reason") if feature_result.get("status") != "ok" else "duplicate_frame_heavy"
            continue
        length = min(len(visual), len(state_motion))
        if length < MIN_RUN_FRAMES:
            continue
        visual = visual[:length]
        state_motion = state_motion[:length]
        visual_valid = visual[np.isfinite(visual)]
        state_valid = state_motion[np.isfinite(state_motion)]
        if visual_valid.size < MIN_RUN_FRAMES or state_valid.size < MIN_RUN_FRAMES:
            continue
        visual_low = np.nanquantile(visual_valid, 0.10)
        visual_high = np.nanquantile(visual_valid, 0.90)
        state_low = np.nanquantile(state_valid, 0.10)
        state_high = np.nanquantile(state_valid, 0.90)
        state_active_visual_static = np.where(np.isfinite(visual) & np.isfinite(state_motion) & (state_motion >= state_high) & (visual <= visual_low))[0]
        visual_active_state_static = np.where(np.isfinite(visual) & np.isfinite(state_motion) & (visual >= visual_high) & (state_motion <= state_low))[0]
        for start, end in contiguous_ranges(state_active_visual_static, max_gap=1):
            if end - start + 1 < MIN_RUN_FRAMES:
                continue
            findings.append(
                factory.make(
                    "vision_state_vision",
                    "geometry_wrist_state_moves_visual_static",
                    "Corresponding wrist state moves while wrist camera motion is static",
                    "segment",
                    "state-vision-geometry",
                    min(85.0, 40.0 + 5.0 * (end - start + 1)),
                    ood=True,
                    episode_index=episode,
                    task_index=task_index,
                    view=view,
                    column=f"state.{arm}.position",
                    frame_start=int(frames[start]),
                    frame_end=int(frames[min(end + 1, len(frames) - 1)]),
                    evidence={
                        "arm": arm,
                        "view": view,
                        "median_state_speed": float(np.nanmedian(state_motion[start : end + 1])),
                        "median_visual_motion": float(np.nanmedian(visual[start : end + 1])),
                        "visual_motion_source": feature_result.get("reason"),
                        "lag_gate": lag,
                    },
                )
            )
        for start, end in contiguous_ranges(visual_active_state_static, max_gap=1):
            if end - start + 1 < MIN_RUN_FRAMES:
                continue
            findings.append(
                factory.make(
                    "state_vision_state",
                    "geometry_wrist_visual_moves_state_static",
                    "Corresponding wrist camera moves while wrist state is static",
                    "segment",
                    "state-vision-geometry",
                    min(85.0, 40.0 + 5.0 * (end - start + 1)),
                    ood=True,
                    episode_index=episode,
                    task_index=task_index,
                    view=view,
                    column=f"state.{arm}.position",
                    frame_start=int(frames[start]),
                    frame_end=int(frames[min(end + 1, len(frames) - 1)]),
                    evidence={
                        "arm": arm,
                        "view": view,
                        "median_visual_motion": float(np.nanmedian(visual[start : end + 1])),
                        "median_state_speed": float(np.nanmedian(state_motion[start : end + 1])),
                        "visual_motion_source": feature_result.get("reason"),
                        "lag_gate": lag,
                    },
                )
            )



