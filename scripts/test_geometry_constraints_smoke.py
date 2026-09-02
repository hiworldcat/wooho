from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import geometry_constraints


def core_checks() -> dict[str, object]:
    config = geometry_constraints.default_geometry_config()
    config = geometry_constraints.validate_geometry_config(config)

    rot = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    precheck = geometry_constraints.rotation6d_precheck(rot, config)
    matrix = geometry_constraints.rotation6d_to_matrix(rot[0], "columns")
    rng = np.random.default_rng(0)
    seq = rng.normal(size=40).astype(np.float64)
    lag = geometry_constraints.identifiable_lag(
        seq,
        seq.copy(),
        max_lag=2,
        window_size=15,
        step=4,
    )
    monotonic = np.stack([np.linspace(0.0, 0.1, 40), np.zeros(40), np.zeros(40)], axis=1)
    oscillating = monotonic.copy()
    oscillating[:, 0] = 0.02 * ((-1.0) ** np.arange(40))
    monotonic_windows = geometry_constraints.oscillation_windows(
        monotonic,
        config["jitter_window_frames"],
        config["jitter_min_direction_reversals"],
        config["jitter_max_path_efficiency"],
        0.01,
    )
    oscillating_windows = geometry_constraints.oscillation_windows(
        oscillating,
        config["jitter_window_frames"],
        config["jitter_min_direction_reversals"],
        config["jitter_max_path_efficiency"],
        0.01,
    )
    module_status_checks = {
        "all_ok": geometry_constraints.aggregate_module_statuses(
            "ok",
            {
                "arms": {"left": "ok", "right": "ok"},
                "bimanual": "ok",
                "state_vision": {"left": "ok", "right": "ok"},
            },
        ),
        "partial_unavailable": geometry_constraints.aggregate_module_statuses(
            "ok",
            {
                "arms": {"left": "ok", "right": "ok"},
                "bimanual": "ok",
                "state_vision": {"left": "ok", "right": "unavailable"},
            },
        ),
        "submodule_warning": geometry_constraints.aggregate_module_statuses(
            "ok",
            {
                "arms": {"left": "ok", "right": "ok"},
                "bimanual": "warning",
                "state_vision": {"left": "ok", "right": "ok"},
            },
        ),
        "submodule_fail": geometry_constraints.aggregate_module_statuses(
            "ok",
            {
                "arms": {"left": "ok", "right": "ok"},
                "bimanual": "ok",
                "state_vision": {"left": "fail", "right": "ok"},
            },
        ),
        "core_unavailable": geometry_constraints.aggregate_module_statuses(
            "unavailable",
            {
                "arms": {"left": "ok", "right": "ok"},
                "bimanual": "ok",
                "state_vision": {"left": "ok", "right": "ok"},
            },
        ),
    }
    assert module_status_checks == {
        "all_ok": "ok",
        "partial_unavailable": "warning",
        "submodule_warning": "warning",
        "submodule_fail": "fail",
        "core_unavailable": "unavailable",
    }
    return {
        "config": config,
        "rotation6d_legal": bool(precheck["legal"][0]),
        "rotation_matrix_shape": None if matrix is None else list(matrix.shape),
        "lag_status": lag.get("status"),
        "lag_best": lag.get("best_lag"),
        "lag_stability_status": lag.get("lag_stability", {}).get("status"),
        "monotonic_oscillation_windows": len(monotonic_windows),
        "sustained_oscillation_windows": len(oscillating_windows),
        "module_status_checks": module_status_checks,
    }


def parquet_smoke() -> dict[str, object]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return {"status": "skipped", "reason": "pyarrow_unavailable"}

    from v2_quality_pipeline import load_dataset, image_views_from_info, FindingFactory

    info, _tasks, _episodes, parquet_files = load_dataset()
    config = geometry_constraints.default_geometry_config(info)
    reference = geometry_constraints.fit_geometry_reference(parquet_files[:3], config)
    path = parquet_files[0]
    df = pq.read_table(path).to_pandas()
    frames = df["frame_index"].to_numpy(dtype=np.int64)
    views = image_views_from_info(info)
    image_context = {view: {"motion": np.array([], dtype=np.float64)} for view in views}
    result = geometry_constraints.inspect_episode_geometry(
        df=df,
        frames=frames,
        episode=int(df["episode_index"].iloc[0]),
        task_index=int(df["task_index"].iloc[0]),
        views=views,
        image_context=image_context,
        reference=reference,
        config=config,
        factory=FindingFactory(),
    )
    return {
        "status": result["diagnostics"]["status"],
        "reference_episodes_used": reference.get("episodes_used", 0),
        "finding_count": len(result["findings"]),
        "state_vision_arms": sorted(result["diagnostics"]["state_vision"].keys()),
    }


def inspect_episode_status_checks() -> dict[str, object]:
    import pandas as pd

    class Factory:
        def make(self, *args, **kwargs):
            return {"args": args, "kwargs": kwargs}

    frames = np.arange(16, dtype=np.int64)
    state_rows = []
    for idx in range(len(frames)):
        left_pos = [idx * 0.001, 0.0, 0.0]
        left_rot = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        right_pos = [0.2 + idx * 0.001, 0.0, 0.0]
        right_rot = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        state_rows.append(left_pos + left_rot + [0.0] + right_pos + right_rot + [0.0])
    df = pd.DataFrame(
        {
            "frame_index": frames,
            "episode_index": np.zeros(len(frames), dtype=np.int64),
            "task_index": np.zeros(len(frames), dtype=np.int64),
            "state": state_rows,
        }
    )
    config = geometry_constraints.default_geometry_config()
    config.update({"rotation6d_layout": "columns", "state_frame_mode": "common_world"})
    result = geometry_constraints.inspect_episode_geometry(
        df=df,
        frames=frames,
        episode=0,
        task_index=0,
        views=["image"],
        image_context={},
        reference={"stats": {}},
        config=config,
        factory=Factory(),
    )
    diagnostics = result["diagnostics"]
    module_statuses = diagnostics["module_statuses"]
    assert diagnostics["status"] == "warning", diagnostics
    assert diagnostics["core_status"] == "ok", diagnostics
    assert module_statuses["state_vision"] == {"left": "unavailable", "right": "unavailable"}, module_statuses
    assert module_statuses["arms"] == {"left": "ok", "right": "ok"}, module_statuses
    assert module_statuses["bimanual"] == "ok", module_statuses
    return {
        "top_level_status": diagnostics["status"],
        "reason": diagnostics.get("reason"),
        "module_statuses": module_statuses,
    }


def main() -> None:
    payload = {"core": core_checks(), "inspect_episode_status": inspect_episode_status_checks(), "parquet": parquet_smoke()}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
