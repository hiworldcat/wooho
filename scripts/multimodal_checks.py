from __future__ import annotations

import io
import json
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


def is_real_parquet(path: Path) -> bool:
    return path.is_file() and not path.name.startswith("._") and "__MACOSX" not in str(path)


def find_parquet_files() -> list[Path]:
    return sorted(p for p in DATA_ROOT.rglob("*.parquet") if is_real_parquet(p))


def decode_gray(value: Any) -> np.ndarray:
    with Image.open(io.BytesIO(value["bytes"])) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    return gray[::8, ::8]


def matrix_from_column(values: Any) -> np.ndarray:
    return np.stack([np.asarray(value, dtype=np.float64).reshape(-1) for value in values])


def normalize(signal: np.ndarray) -> np.ndarray | None:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        return None
    center = float(np.median(signal))
    scale = float(np.std(signal))
    if not np.isfinite(scale) or scale < 1e-12:
        return None
    return (signal - center) / scale


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
    return float(np.corrcoef(left, right)[0, 1])


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 5) -> dict[str, Any]:
    a_norm = normalize(a)
    b_norm = normalize(b)
    if a_norm is None or b_norm is None:
        return {"best_lag": None, "best_correlation": None, "correlations": {}}
    correlations: dict[str, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        corr = lagged_correlation(a_norm, b_norm, lag)
        if corr is not None and np.isfinite(corr):
            correlations[str(lag)] = corr
    if not correlations:
        return {"best_lag": None, "best_correlation": None, "correlations": {}}
    lag = max(correlations, key=correlations.get)
    return {
        "best_lag": int(lag),
        "best_correlation": float(correlations[lag]),
        "correlations": correlations,
    }


def inspect_episode(path: Path) -> dict[str, Any]:
    table = pq.read_table(path, columns=["state", "actions", *VIEWS])
    df = table.to_pandas()
    state = matrix_from_column(df["state"])
    actions = matrix_from_column(df["actions"])
    state_motion = np.linalg.norm(np.diff(state, axis=0), axis=1)
    action_motion = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    combined_motion = state_motion + action_motion

    visual_signals: dict[str, np.ndarray] = {}
    for view in VIEWS:
        images = [decode_gray(value) for value in df[view]]
        visual_signals[view] = np.asarray(
            [np.abs(images[i] - images[i - 1]).mean() for i in range(1, len(images))],
            dtype=np.float64,
        )

    result: dict[str, Any] = {
        "episode_index": int(path.stem.split("_")[-1]),
        "frame_count": int(len(df)),
        "state_action": {
            "action_to_state": best_lag(action_motion, state_motion),
        },
        "views": {},
    }
    for view, visual_signal in visual_signals.items():
        result["views"][view] = {
            "visual_to_state": best_lag(visual_signal, state_motion),
            "visual_to_action": best_lag(visual_signal, action_motion),
            "visual_median_change": float(np.median(visual_signal)),
            "visual_p99_change": float(np.quantile(visual_signal, 0.99)),
            "motion_median": float(np.median(combined_motion)),
        }
    return result


def main() -> None:
    results = [inspect_episode(path) for path in find_parquet_files()]
    output = OUTPUT_ROOT / "multimodal_sync_summary.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"episodes_checked: {len(results)}")
    for row in results:
        print(
            row["episode_index"],
            "action_to_state_lag=",
            row["state_action"]["action_to_state"]["best_lag"],
            "corr=",
            row["state_action"]["action_to_state"]["best_correlation"],
            "image_lag=",
            row["views"]["image"]["visual_to_state"]["best_lag"],
            "image_corr=",
            row["views"]["image"]["visual_to_state"]["best_correlation"],
        )
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
