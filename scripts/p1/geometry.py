from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pq = None

try:
    import fastparquet  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fastparquet = None

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

import geometry_constraints as gc


ROOT = Path(__file__).resolve().parents[2]
P1_ROOT = ROOT / "outputs" / "v2" / "p1"
DIAG_ROOT = P1_ROOT / "diagnostics"
REPORT_ROOT = P1_ROOT / "reports"


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


def parquet_column_names(path: Path) -> list[str]:
    if pq is not None:
        return list(pq.read_schema(path).names)
    if fastparquet is not None:
        return list(fastparquet.ParquetFile(path).columns)
    raise ImportError("No parquet engine available")


def read_parquet_frame(path: Path, columns: list[str] | None = None) -> Any:
    if pq is not None:
        return pq.read_table(path, columns=columns).to_pandas()
    if fastparquet is not None:
        return fastparquet.ParquetFile(path).to_pandas(columns=columns)
    raise ImportError("No parquet engine available")


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
    return views or ["image", "left_wrist_image", "right_wrist_image"]


def safe_episode_from_path(path: Path) -> int | None:
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return None


def make_finding(
    finding_id: str,
    issue_type: str,
    issue_name: str,
    severity_score: float,
    confidence_level: str,
    episode_index: int | None = None,
    task_index: int | None = None,
    view: str | None = None,
    column: str | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "issue_type": issue_type,
        "issue_name": issue_name,
        "severity_score": round(float(severity_score), 2),
        "confidence_level": confidence_level,
        "episode_index": episode_index,
        "task_index": task_index,
        "view": view,
        "column": column,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "evidence": evidence or {},
    }


@dataclass(frozen=True)
class P1Reference:
    state: dict[str, Any]
    p1: dict[str, Any]


def _grid_coverage(points: np.ndarray, shape: tuple[int, int], grid: tuple[int, int] = (4, 4)) -> float:
    if points.size == 0:
        return 0.0
    h, w = shape
    cols, rows = grid
    xs = np.clip(points[:, 0] / max(float(w), 1.0), 0.0, 0.999999)
    ys = np.clip(points[:, 1] / max(float(h), 1.0), 0.0, 0.999999)
    cell_x = np.floor(xs * cols).astype(int)
    cell_y = np.floor(ys * rows).astype(int)
    return float(len({(int(x), int(y)) for x, y in zip(cell_x, cell_y)})) / float(cols * rows)


def _resize_gray(gray: np.ndarray, limit: int = 320) -> np.ndarray:
    if cv2 is None:
        return gray
    if max(gray.shape[:2]) <= limit:
        return gray
    scale = limit / float(max(gray.shape[:2]))
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)



def visual_motion_from_view(values: Any, max_frames: int | None = None) -> dict[str, Any]:
    if cv2 is not None:
        return gc.feature_motion_from_view(values, max_frames=max_frames)

    motions: list[float] = []
    duplicate_flags: list[bool] = []
    previous: np.ndarray | None = None
    previous_hash: str | None = None
    frame_limit = len(values) if max_frames is None else min(len(values), max_frames)
    for value in list(values)[:frame_limit]:
        try:
            gray = gc.decode_gray(value).astype(np.float32)
        except Exception:
            previous = None
            previous_hash = None
            continue
        stride = max(1, int(max(gray.shape[:2]) / 80))
        sample = gray[::stride, ::stride]
        current_hash = hashlib.sha1(sample.astype(np.uint8).tobytes()).hexdigest()
        if previous_hash is not None:
            duplicate_flags.append(current_hash == previous_hash)
        previous_hash = current_hash
        if previous is not None:
            h = min(previous.shape[0], sample.shape[0])
            w = min(previous.shape[1], sample.shape[1])
            motions.append(float(np.mean(np.abs(sample[:h, :w] - previous[:h, :w]))))
        previous = sample
    motion = np.asarray(motions, dtype=np.float64)
    valid = np.isfinite(motion)
    return {
        "status": "ok" if int(valid.sum()) >= 8 else "unavailable",
        "reason": "downsampled_frame_difference" if int(valid.sum()) >= 8 else "too_few_frame_differences",
        "motion": motion,
        "background_motion": motion,
        "valid_fraction": float(valid.mean()) if motion.size else 0.0,
        "median_track_count": 0.0,
        "median_inlier_count": 0.0,
        "duplicate_frame_fraction": float(np.mean(duplicate_flags)) if duplicate_flags else 0.0,
    }


def _panel_proxy_features_fallback(values: Any, max_frames: int | None = None) -> dict[str, Any]:
    centers: list[tuple[float, float]] = []
    areas: list[float] = []
    angles: list[float] = []
    aspect_ratios: list[float] = []
    valid: list[bool] = []
    previous_center: tuple[float, float] | None = None
    center_deltas: list[float] = []
    frame_limit = len(values) if max_frames is None else min(len(values), max_frames)
    for value in list(values)[:frame_limit]:
        try:
            gray = gc.decode_gray(value).astype(np.float32)
        except Exception:
            centers.append((float("nan"), float("nan")))
            areas.append(float("nan"))
            angles.append(float("nan"))
            aspect_ratios.append(float("nan"))
            valid.append(False)
            continue
        threshold = float((np.quantile(gray, 0.10) + np.quantile(gray, 0.90)) * 0.5)
        ys, xs = np.where(gray <= threshold)
        image_area = float(gray.shape[0] * gray.shape[1])
        if xs.size == 0:
            centers.append((float("nan"), float("nan")))
            areas.append(float("nan"))
            angles.append(float("nan"))
            aspect_ratios.append(float("nan"))
            valid.append(False)
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        width = max(1, x1 - x0 + 1)
        height = max(1, y1 - y0 + 1)
        box_area = float(width * height)
        aspect = max(width, height) / max(min(width, height), 1)
        fill_fraction = xs.size / max(box_area, 1.0)
        ok = 0.03 * image_area <= box_area <= 0.95 * image_area and 1.05 <= aspect <= 12.0 and fill_fraction >= 0.15
        center = (float((x0 + x1) * 0.5), float((y0 + y1) * 0.5))
        centers.append(center if ok else (float("nan"), float("nan")))
        areas.append(box_area if ok else float("nan"))
        angles.append(float("nan"))
        aspect_ratios.append(float(aspect) if ok else float("nan"))
        valid.append(bool(ok))
        if ok and previous_center is not None:
            center_deltas.append(float(np.linalg.norm(np.asarray(center) - np.asarray(previous_center))))
        if ok:
            previous_center = center
    valid_mask = np.asarray(valid, dtype=bool)
    centers_arr = np.asarray(centers, dtype=np.float64)
    areas_arr = np.asarray(areas, dtype=np.float64)
    angles_arr = np.asarray(angles, dtype=np.float64)
    aspect_arr = np.asarray(aspect_ratios, dtype=np.float64)
    center_delta_arr = np.asarray(center_deltas, dtype=np.float64)
    return {
        "status": "ok" if int(valid_mask.sum()) >= 5 else "unavailable",
        "reason": "panel_proxy_dark_region" if int(valid_mask.sum()) >= 5 else "panel_proxy_unavailable",
        "valid_mask": valid_mask,
        "valid_fraction": float(valid_mask.mean()) if valid_mask.size else 0.0,
        "center": centers_arr,
        "area": areas_arr,
        "angle": angles_arr,
        "aspect_ratio": aspect_arr,
        "center_delta": center_delta_arr,
        "median_area": float(np.nanmedian(areas_arr)) if np.isfinite(areas_arr).any() else float("nan"),
        "median_aspect_ratio": float(np.nanmedian(aspect_arr)) if np.isfinite(aspect_arr).any() else float("nan"),
        "median_center_delta": float(np.nanmedian(center_delta_arr)) if center_delta_arr.size else float("nan"),
    }


def _pairwise_overlap_gate_fallback(values_a: Any, values_b: Any, max_frames: int | None = None) -> dict[str, Any]:
    frame_limit = min(len(values_a), len(values_b))
    if max_frames is not None:
        frame_limit = min(frame_limit, max_frames)
    if frame_limit < 4:
        return {"status": "unavailable", "reason": "too_few_frames"}
    sample_step = max(1, frame_limit // 30)
    sampled = 0
    ok_frames = 0
    corrs: list[float] = []
    for idx in range(0, frame_limit, sample_step):
        sampled += 1
        try:
            left = gc.decode_gray(values_a[idx]).astype(np.float32)
            right = gc.decode_gray(values_b[idx]).astype(np.float32)
        except Exception:
            continue
        stride = max(1, int(max(max(left.shape[:2]), max(right.shape[:2])) / 80))
        left = left[::stride, ::stride]
        right = right[::stride, ::stride]
        h = min(left.shape[0], right.shape[0])
        w = min(left.shape[1], right.shape[1])
        a = left[:h, :w].reshape(-1)
        b = right[:h, :w].reshape(-1)
        if a.size < 16 or float(np.std(a)) < 1e-6 or float(np.std(b)) < 1e-6:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(corr):
            corrs.append(corr)
            if corr >= 0.75:
                ok_frames += 1
    ok_fraction = float(ok_frames / sampled) if sampled else 0.0
    comparable = ok_frames >= 3 and ok_fraction >= 0.2
    return {
        "status": "ok" if comparable else "unavailable",
        "reason": "coarse_visual_similarity" if comparable else "opencv_required_for_feature_overlap",
        "sampled_frames": int(sampled),
        "ok_frames": int(ok_frames),
        "ok_fraction": ok_fraction,
        "median_matches": 0.0,
        "median_inliers": 0.0,
        "median_coverage": 0.0,
        "median_correlation": float(np.median(corrs)) if corrs else float("nan"),
    }
def panel_proxy_features(values: Any, max_frames: int | None = None) -> dict[str, Any]:
    if cv2 is None:
        return _panel_proxy_features_fallback(values, max_frames=max_frames)

    centers: list[tuple[float, float]] = []
    areas: list[float] = []
    angles: list[float] = []
    aspect_ratios: list[float] = []
    valid: list[bool] = []
    previous_center: tuple[float, float] | None = None
    center_deltas: list[float] = []
    frame_limit = len(values) if max_frames is None else min(len(values), max_frames)

    for value in list(values)[:frame_limit]:
        try:
            gray = gc.decode_gray(value)
        except Exception:
            centers.append((float("nan"), float("nan")))
            areas.append(float("nan"))
            angles.append(float("nan"))
            aspect_ratios.append(float("nan"))
            valid.append(False)
            continue
        gray = _resize_gray(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        image_area = float(gray.shape[0] * gray.shape[1])
        candidates: list[tuple[float, tuple[float, float], float, float, float]] = []

        def scan(mask: np.ndarray) -> None:
            raw = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask
            kernel = np.ones((5, 5), dtype=np.uint8)
            clean = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=2)
            clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)
            found = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = found[0] if len(found) == 2 else found[1]
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < 0.03 * image_area or area > 0.95 * image_area:
                    continue
                (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
                short = max(min(float(w), float(h)), 1e-6)
                aspect = max(float(w), float(h)) / short
                if aspect < 1.1 or aspect > 10.0:
                    continue
                candidates.append((area, (float(cx), float(cy)), float(angle), aspect, area / image_area))

        dark_threshold = float(np.quantile(blur, 0.35))
        scan(blur <= dark_threshold)
        if not candidates:
            edges = cv2.Canny(blur, 40, 120)
            scan(edges > 0)

        if not candidates:
            centers.append((float("nan"), float("nan")))
            areas.append(float("nan"))
            angles.append(float("nan"))
            aspect_ratios.append(float("nan"))
            valid.append(False)
            continue

        area, center, angle, aspect, _ = max(candidates, key=lambda item: item[0])
        centers.append(center)
        areas.append(area)
        angles.append(angle)
        aspect_ratios.append(aspect)
        valid.append(True)
        if previous_center is not None and np.isfinite(center).all():
            center_deltas.append(float(np.linalg.norm(np.asarray(center) - np.asarray(previous_center))))
        previous_center = center

    valid_mask = np.asarray(valid, dtype=bool)
    centers_arr = np.asarray(centers, dtype=np.float64)
    areas_arr = np.asarray(areas, dtype=np.float64)
    angles_arr = np.asarray(angles, dtype=np.float64)
    aspect_arr = np.asarray(aspect_ratios, dtype=np.float64)
    center_delta_arr = np.asarray(center_deltas, dtype=np.float64)
    return {
        "status": "ok" if int(valid_mask.sum()) >= 5 else "unavailable",
        "reason": "panel_proxy_detected" if int(valid_mask.sum()) >= 5 else "panel_proxy_unavailable",
        "valid_mask": valid_mask,
        "valid_fraction": float(valid_mask.mean()) if valid_mask.size else 0.0,
        "center": centers_arr,
        "area": areas_arr,
        "angle": angles_arr,
        "aspect_ratio": aspect_arr,
        "center_delta": center_delta_arr,
        "median_area": float(np.nanmedian(areas_arr)) if np.isfinite(areas_arr).any() else float("nan"),
        "median_aspect_ratio": float(np.nanmedian(aspect_arr)) if np.isfinite(aspect_arr).any() else float("nan"),
        "median_center_delta": float(np.nanmedian(center_delta_arr)) if center_delta_arr.size else float("nan"),
    }


def pairwise_overlap_gate(values_a: Any, values_b: Any, max_frames: int | None = None) -> dict[str, Any]:
    if cv2 is None:
        return _pairwise_overlap_gate_fallback(values_a, values_b, max_frames=max_frames)

    frame_limit = min(len(values_a), len(values_b))
    if max_frames is not None:
        frame_limit = min(frame_limit, max_frames)
    if frame_limit < 4:
        return {"status": "unavailable", "reason": "too_few_frames"}

    orb = cv2.ORB_create(nfeatures=600)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    sample_step = max(1, frame_limit // 30)
    sampled = 0
    ok_frames = 0
    matches_list: list[int] = []
    inliers_list: list[int] = []
    coverage_list: list[float] = []

    for idx in range(0, frame_limit, sample_step):
        sampled += 1
        try:
            gray_a = _resize_gray(gc.decode_gray(values_a[idx]))
            gray_b = _resize_gray(gc.decode_gray(values_b[idx]))
        except Exception:
            continue
        kp_a, des_a = orb.detectAndCompute(gray_a, None)
        kp_b, des_b = orb.detectAndCompute(gray_b, None)
        if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
            continue
        knn = matcher.knnMatch(des_a, des_b, k=2)
        good = [m for m, n in knn if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            continue
        pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])
        _H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 4.0)
        if mask is None:
            continue
        inlier_mask = mask.reshape(-1).astype(bool)
        inliers = int(inlier_mask.sum())
        if inliers < 8:
            continue
        inlier_pts = pts_a[inlier_mask]
        coverage = _grid_coverage(inlier_pts, gray_a.shape[:2])
        inlier_ratio = inliers / max(len(good), 1)
        matches_list.append(int(len(good)))
        inliers_list.append(inliers)
        coverage_list.append(coverage)
        if inlier_ratio >= 0.5 and coverage >= 0.20:
            ok_frames += 1

    ok_fraction = float(ok_frames / sampled) if sampled else 0.0
    comparable = ok_frames >= 3 and ok_fraction >= 0.2
    return {
        "status": "ok" if comparable else "unavailable",
        "reason": "shared_visual_structure" if comparable else "insufficient_shared_visual_structure",
        "sampled_frames": int(sampled),
        "ok_frames": int(ok_frames),
        "ok_fraction": ok_fraction,
        "median_matches": float(np.median(matches_list)) if matches_list else float("nan"),
        "median_inliers": float(np.median(inliers_list)) if inliers_list else float("nan"),
        "median_coverage": float(np.median(coverage_list)) if coverage_list else float("nan"),
    }


def pairwise_overlap_summary(df: Any, views: list[str], max_frames: int | None = None) -> dict[str, Any]:
    if len(views) < 2:
        return {"status": "unavailable", "reason": "too_few_views"}
    summary: dict[str, Any] = {"status": "ok", "pairs": {}}
    for i, left in enumerate(views):
        if left not in df.columns:
            continue
        for right in views[i + 1 :]:
            if right not in df.columns:
                continue
            result = pairwise_overlap_gate(df[left], df[right], max_frames=max_frames)
            summary["pairs"][f"{left}->{right}"] = result
    ok_pairs = [name for name, item in summary["pairs"].items() if item.get("status") == "ok"]
    summary["comparable_pairs"] = ok_pairs
    summary["comparable_pair_count"] = len(ok_pairs)
    summary["not_comparable_pairs"] = [name for name, item in summary["pairs"].items() if item.get("status") != "ok"]
    summary["all_pairs_unavailable"] = len(ok_pairs) == 0
    return summary


def _lookup_stats(reference: dict[str, Any], key: str) -> dict[str, Any]:
    stats = reference.get("stats", {})
    item = stats.get(key)
    return item if isinstance(item, dict) else {"count": 0}


def _threshold_from_stats(stats: dict[str, Any], key: str, default: float) -> float:
    if stats.get("count", 0) < 5:
        return default
    return gc.stat(stats, key, default)


def _finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(x), len(y))
    if length <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    left = np.asarray(x[:length], dtype=np.float64)
    right = np.asarray(y[:length], dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    return left[mask], right[mask]


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    left, right = _finite_pair(x, y)
    if left.size < gc.MIN_RUN_FRAMES:
        return float("nan")
    if float(np.nanstd(left)) < gc.EPS or float(np.nanstd(right)) < gc.EPS:
        return float("nan")
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else float("nan")


def motion_coupling_features(state_motion: np.ndarray, visual_motion: np.ndarray, max_lag: int) -> dict[str, Any]:
    state = np.asarray(state_motion, dtype=np.float64)
    visual = np.asarray(visual_motion, dtype=np.float64)
    length = min(len(state), len(visual))
    if length < gc.MIN_RUN_FRAMES:
        return {"status": "unavailable", "reason": "too_few_motion_samples"}
    state = state[:length]
    visual = visual[:length]
    lag_scores: dict[str, float | None] = {}
    best_lag: int | None = None
    best_corr = float("nan")
    for lag in range(-max(0, int(max_lag)), max(0, int(max_lag)) + 1):
        if lag < 0:
            x, y = state[-lag:], visual[: length + lag]
        elif lag > 0:
            x, y = state[: length - lag], visual[lag:]
        else:
            x, y = state, visual
        corr = _safe_corr(x, y)
        lag_scores[str(lag)] = round(corr, 6) if np.isfinite(corr) else None
        if np.isfinite(corr) and (not np.isfinite(best_corr) or abs(corr) > abs(best_corr)):
            best_corr = corr
            best_lag = lag

    finite_state = state[np.isfinite(state)]
    finite_visual = visual[np.isfinite(visual)]
    if finite_state.size < gc.MIN_RUN_FRAMES or finite_visual.size < gc.MIN_RUN_FRAMES:
        return {"status": "unavailable", "reason": "insufficient_finite_motion_samples"}
    state_active_threshold = float(np.nanquantile(finite_state, 0.75))
    visual_active_threshold = float(np.nanquantile(finite_visual, 0.75))
    state_quiet_threshold = float(np.nanquantile(finite_state, 0.25))
    visual_quiet_threshold = float(np.nanquantile(finite_visual, 0.25))
    active_state = state >= state_active_threshold
    active_visual = visual >= visual_active_threshold
    quiet_state = state <= state_quiet_threshold
    quiet_visual = visual <= visual_quiet_threshold
    state_active_visual_quiet = np.isfinite(state) & np.isfinite(visual) & active_state & quiet_visual
    visual_active_state_quiet = np.isfinite(state) & np.isfinite(visual) & active_visual & quiet_state
    zero_lag = _safe_corr(state, visual)
    return {
        "status": "ok" if np.isfinite(best_corr) else "unavailable",
        "reason": "state_visual_motion_coupling" if np.isfinite(best_corr) else "motion_correlation_degenerate",
        "sample_count": int(length),
        "zero_lag_correlation": round(zero_lag, 6) if np.isfinite(zero_lag) else None,
        "best_lag": best_lag,
        "best_abs_correlation": round(float(abs(best_corr)), 6) if np.isfinite(best_corr) else None,
        "best_signed_correlation": round(float(best_corr), 6) if np.isfinite(best_corr) else None,
        "lag_correlations": lag_scores,
        "state_active_visual_quiet_fraction": float(np.mean(state_active_visual_quiet)),
        "visual_active_state_quiet_fraction": float(np.mean(visual_active_state_quiet)),
    }

def fit_p1_reference(parquet_files: list[Path], config: dict[str, Any]) -> dict[str, Any]:
    config = gc.validate_geometry_config(config)
    if pq is None and fastparquet is None:
        return {"status": "unavailable", "reason": "no_parquet_engine", "stats": {}}

    buckets: dict[str, list[float]] = defaultdict(list)
    episode_summaries: list[dict[str, Any]] = []
    for path in parquet_files:
        try:
            cols = parquet_column_names(path)
            wanted = [name for name in ["image", "left_wrist_image", "right_wrist_image", "state", "episode_index", "task_index"] if name in cols]
            df = read_parquet_frame(path, columns=wanted)
        except Exception:
            continue
        if len(df) == 0 or "state" not in df.columns:
            continue
        matrix = gc.finite_float_matrix(df["state"])
        if matrix is None or matrix.shape[1] < 20 or not np.isfinite(matrix).all():
            continue
        arms = {arm: gc.arm_motion_features(matrix, arm, config) for arm in gc.ARM_SPECS}
        episode = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else None
        task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else None

        if "image" in df.columns:
            main = visual_motion_from_view(df["image"], max_frames=int(config.get("vision_feature_max_frames")) if config.get("vision_feature_max_frames") is not None else None)
            panel = panel_proxy_features(df["image"], max_frames=int(config.get("vision_feature_max_frames")) if config.get("vision_feature_max_frames") is not None else None)
            if main.get("status") == "ok":
                buckets["global|main_camera|background_motion"].extend(np.asarray(main["motion"], dtype=np.float64)[np.isfinite(main["motion"])].tolist())
                buckets["global|main_camera|duplicate_frame_fraction"].append(float(main.get("duplicate_frame_fraction", 0.0)))
            if panel.get("status") == "ok":
                buckets["global|main_camera|panel_valid_fraction"].append(float(panel.get("valid_fraction", 0.0)))
                buckets["global|main_camera|panel_area"].extend(np.asarray(panel["area"], dtype=np.float64)[np.isfinite(panel["area"])].tolist())
                buckets["global|main_camera|panel_center_delta"].extend(np.asarray(panel["center_delta"], dtype=np.float64)[np.isfinite(panel["center_delta"])].tolist())

        if "image" in df.columns and main.get("status") == "ok":
            left_speed = np.asarray(arms["left"]["speed"], dtype=np.float64)
            right_speed = np.asarray(arms["right"]["speed"], dtype=np.float64)
            arm_speed = np.maximum(left_speed[: len(right_speed)], right_speed[: len(left_speed)])
            coupling = motion_coupling_features(
                arm_speed,
                np.asarray(main.get("motion", []), dtype=np.float64),
                max_lag=int(config.get("max_lag", gc.MAX_LAG)),
            )
            if coupling.get("status") == "ok":
                buckets["global|state_visual_motion_coupling|best_abs_correlation"].append(float(coupling.get("best_abs_correlation", 0.0)))
                buckets["global|state_visual_motion_coupling|state_active_visual_quiet_fraction"].append(float(coupling.get("state_active_visual_quiet_fraction", 0.0)))
                buckets["global|state_visual_motion_coupling|visual_active_state_quiet_fraction"].append(float(coupling.get("visual_active_state_quiet_fraction", 0.0)))

        overlap = pairwise_overlap_summary(df, [view for view in ["image", "left_wrist_image", "right_wrist_image"] if view in df.columns], max_frames=int(config.get("vision_feature_max_frames")) if config.get("vision_feature_max_frames") is not None else None)
        for pair_name, pair in overlap.get("pairs", {}).items():
            if pair.get("status") == "ok":
                buckets[f"global|{pair_name}|ok_fraction"].append(float(pair.get("ok_fraction", 0.0)))
                buckets[f"global|{pair_name}|median_inliers"].append(float(pair.get("median_inliers", 0.0)))
                buckets[f"global|{pair_name}|median_matches"].append(float(pair.get("median_matches", 0.0)))
                buckets[f"global|{pair_name}|median_coverage"].append(float(pair.get("median_coverage", 0.0)))

        for arm, features in arms.items():
            buckets[f"global|{arm}|speed"].extend(features["speed"][np.isfinite(features["speed"])].tolist())
            buckets[f"global|{arm}|omega"].extend(features["omega"][np.isfinite(features["omega"])].tolist())

        episode_summaries.append(
            {
                "episode_index": episode,
                "task_index": task_index,
                "length": int(len(df)),
                "overlap_pair_count": int(overlap.get("comparable_pair_count", 0)),
                "panel_valid_fraction": float(panel.get("valid_fraction", 0.0)) if "image" in df.columns else None,
            }
        )

    return {
        "status": "ok",
        "config": config,
        "stats": {key: gc.robust_stats(values) for key, values in buckets.items()},
        "episodes_used": len(episode_summaries),
        "episode_summaries": episode_summaries,
    }


def inspect_episode_p1(
    df: Any,
    frames: np.ndarray,
    episode: int | None,
    task_index: int | None,
    views: list[str],
    reference: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    config = gc.validate_geometry_config(config)
    diagnostics: dict[str, Any] = {
        "episode_index": episode,
        "task_index": task_index,
        "status": "ok",
        "main_camera": {},
        "panel_proxy": {},
        "overlap_gates": {},
        "motion_coupling": {},
        "bimanual_rigidity": {},
        "root_cause_windows": [],
    }
    findings: list[dict[str, Any]] = []
    if "state" not in df.columns:
        diagnostics["status"] = "unavailable"
        diagnostics["reason"] = "state_column_missing"
        return {"findings": findings, "diagnostics": diagnostics}

    matrix = gc.finite_float_matrix(df["state"])
    if matrix is None or matrix.ndim != 2 or matrix.shape[1] < 20:
        diagnostics["status"] = "unavailable"
        diagnostics["reason"] = "state_matrix_invalid"
        return {"findings": findings, "diagnostics": diagnostics}
    if len(matrix) != len(frames):
        diagnostics["status"] = "unavailable"
        diagnostics["reason"] = "state_frame_length_mismatch"
        return {"findings": findings, "diagnostics": diagnostics}

    arms = {arm: gc.arm_motion_features(matrix, arm, config) for arm in gc.ARM_SPECS}
    left_speed = np.asarray(arms["left"]["speed"], dtype=np.float64)
    right_speed = np.asarray(arms["right"]["speed"], dtype=np.float64)
    arm_speed = np.maximum(left_speed[: len(right_speed)], right_speed[: len(left_speed)])
    bimanual = gc.bimanual_features(arms, config)
    if bimanual.get("status") == "ok":
        distance = np.asarray(bimanual.get("distance", []), dtype=np.float64)
        drift = np.abs(distance - np.nanmedian(distance)) if distance.size else np.array([], dtype=np.float64)
        drift_stats = reference.get("state", {}).get("stats", {}).get("global|bimanual|distance_drift", {})
        diagnostics["bimanual_rigidity"] = {
            "status": "ok",
            "median_eef_distance": float(np.nanmedian(distance)) if np.isfinite(distance).any() else None,
            "max_distance_drift": float(np.nanmax(drift)) if np.isfinite(drift).any() else None,
            "reference_count": int(drift_stats.get("count", 0)) if isinstance(drift_stats, dict) else 0,
            "constraint_level": "state_only_candidate",
        }
        if isinstance(drift_stats, dict) and drift_stats.get("count", 0) >= 5 and drift.size:
            threshold = max(
                gc.stat(drift_stats, "p99", float(np.nanmax(drift))) * 1.4,
                gc.stat(drift_stats, "median", 0.0) + 8.0 * gc.stat(drift_stats, "mad", 0.0),
            )
            bad = np.where(np.isfinite(drift) & (drift > threshold))[0]
            for start, end in gc.contiguous_ranges(bad, max_gap=2):
                if end - start + 1 < gc.MIN_RUN_FRAMES:
                    continue
                max_drift = float(np.nanmax(drift[start : end + 1]))
                findings.append(
                    make_finding(
                        f"P1F-{len(findings)+1:06d}",
                        "p1_bimanual_rigidity_drift_candidate",
                        "Bimanual relative distance drifts under the enabled common-world gate",
                        max(45.0, gc.score_from_z(gc.robust_z(max_drift, drift_stats), weak=4.0, strong=8.0)),
                        "weak",
                        episode,
                        task_index,
                        column="state.bimanual.relative_distance",
                        frame_start=int(frames[start]),
                        frame_end=int(frames[end]),
                        evidence={
                            "max_distance_drift": max_drift,
                            "threshold": threshold,
                            "reference": drift_stats,
                            "constraint_level": "state_only_candidate",
                        },
                    )
                )
    else:
        diagnostics["bimanual_rigidity"] = {
            "status": "unavailable",
            "reason": bimanual.get("reason", "bimanual_gate_unavailable"),
            "state_frame_mode": config.get("state_frame_mode", "unknown"),
            "constraint_level": "disabled_by_gate",
        }

    if "image" in df.columns:
        main_motion = visual_motion_from_view(
            df["image"],
            max_frames=int(config.get("vision_feature_max_frames")) if config.get("vision_feature_max_frames") is not None else None,
        )
        panel = panel_proxy_features(
            df["image"],
            max_frames=int(config.get("vision_feature_max_frames")) if config.get("vision_feature_max_frames") is not None else None,
        )
        overlap = pairwise_overlap_summary(
            df,
            [view for view in ["image", "left_wrist_image", "right_wrist_image"] if view in df.columns],
            max_frames=int(config.get("vision_feature_max_frames")) if config.get("vision_feature_max_frames") is not None else None,
        )
    else:
        main_motion = {"status": "unavailable", "reason": "main_view_missing", "motion": np.array([], dtype=np.float64)}
        panel = {"status": "unavailable", "reason": "main_view_missing", "valid_fraction": 0.0}
        overlap = {"status": "unavailable", "reason": "main_view_missing", "pairs": {}, "comparable_pairs": []}

    diagnostics["main_camera"] = {
        "status": main_motion.get("status"),
        "reason": main_motion.get("reason"),
        "duplicate_frame_fraction": float(main_motion.get("duplicate_frame_fraction", 0.0)),
        "median_background_motion": float(np.nanmedian(np.asarray(main_motion.get("motion", []), dtype=np.float64)))
        if np.isfinite(np.asarray(main_motion.get("motion", []), dtype=np.float64)).any()
        else None,
    }
    diagnostics["panel_proxy"] = {
        "status": panel.get("status"),
        "reason": panel.get("reason"),
        "valid_fraction": float(panel.get("valid_fraction", 0.0)),
        "median_area": panel.get("median_area"),
        "median_aspect_ratio": panel.get("median_aspect_ratio"),
        "median_center_delta": panel.get("median_center_delta"),
    }
    diagnostics["overlap_gates"] = overlap
    diagnostics["motion_coupling"] = motion_coupling_features(
        arm_speed,
        np.asarray(main_motion.get("motion", []), dtype=np.float64),
        max_lag=int(config.get("max_lag", gc.MAX_LAG)),
    )

    coupling = diagnostics["motion_coupling"]
    if coupling.get("status") == "ok":
        coupling_stats = reference.get("p1", {}).get("stats", {}).get("global|state_visual_motion_coupling|best_abs_correlation", {})
        sv_stats = reference.get("p1", {}).get("stats", {}).get("global|state_visual_motion_coupling|state_active_visual_quiet_fraction", {})
        vs_stats = reference.get("p1", {}).get("stats", {}).get("global|state_visual_motion_coupling|visual_active_state_quiet_fraction", {})
        corr = coupling.get("best_abs_correlation")
        sv = coupling.get("state_active_visual_quiet_fraction")
        vs = coupling.get("visual_active_state_quiet_fraction")
        if isinstance(corr, (float, int)) and coupling_stats.get("count", 0) >= 5:
            corr_floor = max(0.20, gc.stat(coupling_stats, "p05", 0.0) * 0.65)
            coupling["reference_corr_floor"] = corr_floor
            if float(corr) < corr_floor:
                findings.append(
                    make_finding(
                        f"P1F-{len(findings)+1:06d}",
                        "p1_state_visual_motion_decoupling_candidate",
                        "State motion and main-camera visual motion are weakly coupled",
                        max(35.0, gc.score_from_z(gc.robust_z(float(corr), coupling_stats), weak=2.5, strong=5.0)),
                        "weak",
                        episode,
                        task_index,
                        view="image",
                        frame_start=int(frames[0]) if len(frames) else None,
                        frame_end=int(frames[-1]) if len(frames) else None,
                        evidence={"best_abs_correlation": float(corr), "corr_floor": corr_floor, "reference": coupling_stats},
                    )
                )
        for label, value, stats in [
            ("state_active_visual_quiet", sv, sv_stats),
            ("visual_active_state_quiet", vs, vs_stats),
        ]:
            if isinstance(value, (float, int)) and stats.get("count", 0) >= 5:
                threshold = max(0.12, gc.stat(stats, "p95", 0.0) * 1.5, gc.stat(stats, "median", 0.0) + 5.0 * gc.stat(stats, "mad", 0.0))
                coupling[f"reference_{label}_ceiling"] = threshold
                if float(value) > threshold:
                    findings.append(
                        make_finding(
                            f"P1F-{len(findings)+1:06d}",
                            "p1_state_visual_activity_mismatch_candidate",
                            f"State and visual activity mismatch is high: {label}",
                            min(80.0, 40.0 + 250.0 * (float(value) - threshold)),
                            "weak",
                            episode,
                            task_index,
                            view="image",
                            frame_start=int(frames[0]) if len(frames) else None,
                            frame_end=int(frames[-1]) if len(frames) else None,
                            evidence={"mismatch_fraction": float(value), "threshold": threshold, "reference": stats, "mismatch_type": label},
                        )
                    )

    main_motion_arr = np.asarray(main_motion.get("motion", []), dtype=np.float64)
    if main_motion_arr.size:
        length = min(len(main_motion_arr), len(arm_speed))
        main_motion_arr = main_motion_arr[:length]
        arm_speed = arm_speed[:length]
        main_stats = reference.get("p1", {}).get("stats", {}).get("global|main_camera|background_motion", {})
        arm_stats = reference.get("state", {}).get("stats", {})
        left_stats = arm_stats.get("global|left|linear_speed", arm_stats.get("global|left|speed", {}))
        right_stats = arm_stats.get("global|right|linear_speed", arm_stats.get("global|right|speed", {}))
        state_stats = left_stats if left_stats.get("count", 0) >= right_stats.get("count", 0) else right_stats
        high_state = _threshold_from_stats(state_stats, "p95", float(np.nanquantile(arm_speed[np.isfinite(arm_speed)], 0.90)) if np.isfinite(arm_speed).any() else 0.0)
        low_main = _threshold_from_stats(main_stats, "p05", float(np.nanquantile(main_motion_arr[np.isfinite(main_motion_arr)], 0.10)) if np.isfinite(main_motion_arr).any() else 0.0)
        panel_area_stats = reference.get("p1", {}).get("stats", {}).get("global|main_camera|panel_area", {})
        panel_valid_stats = reference.get("p1", {}).get("stats", {}).get("global|main_camera|panel_valid_fraction", {})
        panel_area_threshold = _threshold_from_stats(panel_area_stats, "p05", float("nan"))
        panel_valid_threshold = _threshold_from_stats(panel_valid_stats, "p05", 0.0)
        panel_area_arr = np.asarray(panel.get("area", []), dtype=np.float64)
        panel_valid_arr = np.asarray(panel.get("valid_mask", []), dtype=bool)
        panel_center_delta = np.asarray(panel.get("center_delta", []), dtype=np.float64)
        panel_area_valid = panel_area_arr[np.isfinite(panel_area_arr)]
        panel_center_valid = panel_center_delta[np.isfinite(panel_center_delta)]
        if (
            np.isfinite(high_state)
            and np.isfinite(low_main)
            and np.isfinite(arm_speed).any()
            and np.isfinite(main_motion_arr).any()
        ):
            mask = np.isfinite(arm_speed) & np.isfinite(main_motion_arr) & (arm_speed >= high_state) & (main_motion_arr <= low_main)
            for start, end in gc.contiguous_ranges(np.where(mask)[0], max_gap=1):
                if end - start + 1 < gc.MIN_RUN_FRAMES:
                    continue
                severity = min(92.0, 55.0 + 4.0 * (end - start + 1))
                findings.append(
                    make_finding(
                        f"P1F-{len(findings)+1:06d}",
                        "p1_main_camera_freeze_candidate",
                        "Main camera background stays almost static while state moves",
                        severity,
                        "supported",
                        episode,
                        task_index,
                        view="image",
                        frame_start=int(frames[start]),
                        frame_end=int(frames[min(end + 1, len(frames) - 1)]),
                        evidence={
                            "median_state_speed": float(np.nanmedian(arm_speed[start : end + 1])),
                            "median_main_motion": float(np.nanmedian(main_motion_arr[start : end + 1])),
                            "state_threshold": high_state,
                            "main_threshold": low_main,
                        },
                    )
                )
                diagnostics["root_cause_windows"].append({"type": "camera_freeze_candidate", "frame_start": int(frames[start]), "frame_end": int(frames[min(end + 1, len(frames) - 1)])})

        if panel_valid_arr.size and float(panel.get("valid_fraction", 0.0)) >= max(0.10, panel_valid_threshold):
            if panel_area_valid.size and np.isfinite(panel_area_threshold):
                area_mask = np.isfinite(panel_area_arr) & (panel_area_arr <= panel_area_threshold)
                for start, end in gc.contiguous_ranges(np.where(area_mask)[0], max_gap=1):
                    if end - start + 1 < gc.MIN_RUN_FRAMES:
                        continue
                    findings.append(
                        make_finding(
                            f"P1F-{len(findings)+1:06d}",
                            "p1_panel_proxy_unstable_candidate",
                            "Main panel proxy is present but area collapses or becomes unstable",
                            min(78.0, 45.0 + 2.5 * (end - start + 1)),
                            "weak",
                            episode,
                            task_index,
                            view="image",
                            frame_start=int(frames[start]),
                            frame_end=int(frames[min(end + 1, len(frames) - 1)]),
                            evidence={
                                "median_panel_area": float(np.nanmedian(panel_area_arr[start : end + 1])),
                                "area_threshold": panel_area_threshold,
                                "panel_valid_fraction": float(panel.get("valid_fraction", 0.0)),
                                "median_center_delta": float(np.nanmedian(panel_center_valid)) if panel_center_valid.size else None,
                            },
                        )
                    )
                    diagnostics["root_cause_windows"].append({"type": "panel_proxy_unstable", "frame_start": int(frames[start]), "frame_end": int(frames[min(end + 1, len(frames) - 1)])})
        else:
            diagnostics["panel_proxy"]["status"] = "unavailable"
            diagnostics["panel_proxy"]["reason"] = "insufficient_panel_proxy_support"

    if overlap.get("all_pairs_unavailable", False):
        diagnostics["root_cause_windows"].append({"type": "not_comparable", "reason": "pairwise_overlap_gate_failed"})

    diagnostics["summary"] = {
        "finding_count": len(findings),
        "comparable_pair_count": int(overlap.get("comparable_pair_count", 0)),
        "panel_valid_fraction": float(panel.get("valid_fraction", 0.0)),
        "motion_coupling_status": diagnostics["motion_coupling"].get("status"),
        "motion_coupling_best_lag": diagnostics["motion_coupling"].get("best_lag"),
        "motion_coupling_best_abs_correlation": diagnostics["motion_coupling"].get("best_abs_correlation"),
        "state_active_visual_quiet_fraction": diagnostics["motion_coupling"].get("state_active_visual_quiet_fraction"),
        "visual_active_state_quiet_fraction": diagnostics["motion_coupling"].get("visual_active_state_quiet_fraction"),
    }
    return {"findings": findings, "diagnostics": diagnostics}


def fit_p1_reference_payload(reference_root: Path, geometry_config: dict[str, Any]) -> dict[str, Any]:
    info, tasks, episodes, parquet_files = load_dataset(reference_root)
    state_reference = gc.fit_geometry_reference(parquet_files, geometry_config)
    p1_reference = fit_p1_reference(parquet_files, geometry_config)
    return {
        "info": info,
        "tasks": tasks,
        "episodes": episodes,
        "state": state_reference,
        "p1": p1_reference,
        "geometry_config": geometry_config,
    }


def run_p1_pipeline(
    reference_root: Path | None = None,
    target_root: Path | None = None,
    geometry_config_path: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    reference_root = reference_root or ROOT
    target_root = target_root or reference_root
    output_root = output_root or P1_ROOT
    diag_root = output_root / 'diagnostics'
    report_root = output_root / 'reports'
    diag_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    reference_info, reference_tasks, reference_episodes, reference_parquet_files = load_dataset(reference_root)
    target_info, target_tasks, target_episodes, target_parquet_files = load_dataset(target_root)
    geometry_config = gc.default_geometry_config(reference_info)
    if geometry_config_path is not None:
        geometry_config = gc.load_geometry_config(geometry_config_path, geometry_config)
    geometry_config = gc.validate_geometry_config(geometry_config)
    reference_payload = fit_p1_reference_payload(reference_root, geometry_config)

    target_episode_meta = {int(row["episode_index"]): row for row in target_episodes}
    results: list[dict[str, Any]] = []
    episode_diagnostics: list[dict[str, Any]] = []
    all_findings: list[dict[str, Any]] = []
    for path in target_parquet_files:
        episode = safe_episode_from_path(path)
        episode_meta = target_episode_meta.get(episode, {}) if episode is not None else {}
        df = read_parquet_frame(path)
        if "frame_index" in df.columns:
            frames = df["frame_index"].to_numpy(dtype=np.int64)
        else:
            frames = np.arange(len(df), dtype=np.int64)
        views = [view for view in image_views_from_info(target_info) if view in df.columns]
        item = inspect_episode_p1(
            df=df,
            frames=frames,
            episode=episode,
            task_index=int(df["task_index"].iloc[0]) if "task_index" in df.columns and len(df) else (int(episode_meta["task_index"]) if episode_meta.get("task_index") is not None else None),
            views=views,
            reference=reference_payload,
            config=geometry_config,
        )
        findings = item["findings"]
        diagnostics = item["diagnostics"]
        all_findings.extend(findings)
        episode_diagnostics.append(diagnostics)
        results.append(
            {
                "episode_index": episode,
                "task_index": int(df["task_index"].iloc[0]) if "task_index" in df.columns and len(df) else episode_meta.get("task_index"),
                "length": int(len(df)),
                "finding_count": len(findings),
                "comparable_pair_count": diagnostics.get("summary", {}).get("comparable_pair_count", 0),
                "panel_valid_fraction": diagnostics.get("summary", {}).get("panel_valid_fraction", 0.0),
                "main_camera_status": diagnostics.get("main_camera", {}).get("status"),
                "panel_proxy_status": diagnostics.get("panel_proxy", {}).get("status"),
                "not_comparable": diagnostics.get("overlap_gates", {}).get("all_pairs_unavailable", False),
                "bimanual_status": diagnostics.get("bimanual_rigidity", {}).get("status"),
                "motion_coupling_status": diagnostics.get("motion_coupling", {}).get("status"),
                "motion_coupling_best_lag": diagnostics.get("motion_coupling", {}).get("best_lag"),
                "motion_coupling_best_abs_correlation": diagnostics.get("motion_coupling", {}).get("best_abs_correlation"),
                "state_active_visual_quiet_fraction": diagnostics.get("motion_coupling", {}).get("state_active_visual_quiet_fraction"),
                "visual_active_state_quiet_fraction": diagnostics.get("motion_coupling", {}).get("visual_active_state_quiet_fraction"),
            }
        )

    findings_by_type = Counter(item["issue_type"] for item in all_findings)
    report = {
        "dataset": {
            "reference_episodes": len(reference_episodes),
            "target_episodes": len(target_episodes),
            "reference_root": str(reference_root),
            "target_root": str(target_root),
            "fps": geometry_config.get("fps"),
        },
        "summary": {
            "finding_count": len(all_findings),
            "finding_types": dict(findings_by_type),
            "episodes_with_comparable_pairs": sum(1 for row in results if row["comparable_pair_count"] > 0),
            "episodes_with_panel_proxy": sum(1 for row in results if row["panel_proxy_status"] == "ok"),
            "episodes_without_comparable_pairs": sum(1 for row in results if row["not_comparable"]),
            "episodes_with_motion_coupling": sum(1 for row in results if row["motion_coupling_status"] == "ok"),
        },
        "episodes": results,
        "findings": all_findings,
        "reference": {"state": reference_payload["state"], "p1": reference_payload["p1"]},
        "geometry_config": geometry_config,
    }

    (diag_root / "p1_geometry_findings.json").write_text(json.dumps(all_findings, ensure_ascii=False, indent=2), encoding="utf-8")
    (diag_root / "p1_episode_diagnostics.json").write_text(json.dumps(episode_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    (diag_root / "p1_geometry_reference.json").write_text(
        json.dumps({"state": reference_payload["state"], "p1": reference_payload["p1"], "geometry_config": geometry_config}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_root / "p1_geometry_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_lines = ["episode_index,task_index,length,finding_count,comparable_pair_count,panel_valid_fraction,main_camera_status,panel_proxy_status,not_comparable,bimanual_status,motion_coupling_status,motion_coupling_best_lag,motion_coupling_best_abs_correlation,state_active_visual_quiet_fraction,visual_active_state_quiet_fraction"]
    for row in results:
        csv_lines.append(
            ",".join(
                [
                    str(row["episode_index"]),
                    str(row["task_index"]),
                    str(row["length"]),
                    str(row["finding_count"]),
                    str(row["comparable_pair_count"]),
                    f'{float(row["panel_valid_fraction"]):.6f}',
                    str(row["main_camera_status"]),
                    str(row["panel_proxy_status"]),
                    str(row["not_comparable"]),
                    str(row["bimanual_status"]),
                    str(row["motion_coupling_status"]),
                    str(row["motion_coupling_best_lag"]),
                    str(row["motion_coupling_best_abs_correlation"]),
                    str(row["state_active_visual_quiet_fraction"]),
                    str(row["visual_active_state_quiet_fraction"]),
                ]
            )
        )
    (report_root / "p1_episode_summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md_lines = [
        "# P1 Spatial Geometry Report",
        "",
        f"- reference episodes: {len(reference_episodes)}",
        f"- target episodes: {len(target_episodes)}",
        f"- findings: {len(all_findings)}",
        f"- episodes with comparable pairs: {report['summary']['episodes_with_comparable_pairs']}",
        f"- episodes with panel proxy: {report['summary']['episodes_with_panel_proxy']}",
        f"- episodes with motion coupling: {report['summary']['episodes_with_motion_coupling']}",
        "",
        "## Finding Types",
        "",
        "| issue_type | count |",
        "|---|---:|",
    ]
    for issue_type, count in findings_by_type.most_common():
        md_lines.append(f"| {issue_type} | {count} |")
    md_lines.extend(
        [
            "",
            "## Episode Summary",
            "",
            "| episode | task | length | findings | comparable pairs | panel valid | main | panel | overlap | bimanual | coupling | best lag | mismatch sv | mismatch vs |",
            "|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in results:
        md_lines.append(
            f"| {row['episode_index']} | {row['task_index']} | {row['length']} | {row['finding_count']} | "
            f"{row['comparable_pair_count']} | {row['panel_valid_fraction']:.3f} | {row['main_camera_status']} | "
            f"{row['panel_proxy_status']} | {row['not_comparable']} | {row['bimanual_status']} | "
            f"{row['motion_coupling_best_abs_correlation']} | {row['motion_coupling_best_lag']} | "
            f"{row['state_active_visual_quiet_fraction']} | {row['visual_active_state_quiet_fraction']} |"
        )
    md_lines.extend(["", "## Notes", "", "P1 is intentionally weaker than calibrated 3D geometry. It reports state-visual motion coupling even when cross-view feature overlap is unavailable. Strict reprojection, epipolar, and triangulation checks remain unavailable unless camera calibration is provided."])
    (report_root / "p1_geometry_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return report








