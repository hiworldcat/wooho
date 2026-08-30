from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometry_constraints as gc
from p1 import geometry as p1_geometry


def image_cell(frame: int, shift: int = 0) -> dict[str, bytes]:
    image = Image.new("RGB", (160, 120), (225, 225, 225))
    draw = ImageDraw.Draw(image)
    x = 25 + frame + shift
    draw.rectangle((x, 25, x + 70, 90), fill=(35, 55, 70), outline=(10, 10, 10), width=2)
    draw.line((x + 8, 35, x + 60, 80), fill=(180, 190, 200), width=2)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {"bytes": buffer.getvalue()}


def state_rows(length: int = 32) -> list[list[float]]:
    rows = []
    for frame in range(length):
        left = [0.01 * frame, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        right = [0.01 * frame + 0.4, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        rows.append(left + right)
    return rows


def main() -> None:
    config = gc.validate_geometry_config(
        {
            **gc.default_geometry_config(),
            "state_frame_mode": "common_world",
            "rotation6d_layout": "columns",
        }
    )
    rows = state_rows()
    values = [image_cell(frame) for frame in range(len(rows))]

    panel = p1_geometry.panel_proxy_features(values)
    assert panel["status"] == "ok"
    assert panel["valid_fraction"] >= 0.5

    motion = p1_geometry.visual_motion_from_view(values)
    assert motion["status"] == "ok"
    assert motion["duplicate_frame_fraction"] == 0.0

    overlap = p1_geometry.pairwise_overlap_gate(values, values)
    assert overlap["status"] == "ok"

    matrix = gc.finite_float_matrix(rows)
    assert matrix is not None
    arms = {arm: gc.arm_motion_features(matrix, arm, config) for arm in gc.ARM_SPECS}
    bimanual = gc.bimanual_features(arms, config)
    assert bimanual["status"] == "ok"
    assert np.nanmedian(bimanual["distance"]) > 0.0

    frame = pd.DataFrame(
        {
            "state": rows,
            "image": values,
            "left_wrist_image": values,
            "right_wrist_image": values,
        }
    )
    result = p1_geometry.inspect_episode_p1(
        df=frame,
        frames=np.arange(len(rows), dtype=np.int64),
        episode=0,
        task_index=0,
        views=["image", "left_wrist_image", "right_wrist_image"],
        reference={"state": {"stats": {}}, "p1": {"stats": {}}},
        config=config,
    )
    assert result["diagnostics"]["panel_proxy"]["status"] in {"ok", "unavailable"}
    coupling = result["diagnostics"]["motion_coupling"]
    assert coupling["status"] in {"ok", "unavailable"}

    direct_coupling = p1_geometry.motion_coupling_features(
        np.linspace(0.0, 1.0, 32) ** 2,
        np.linspace(0.0, 1.0, 32) ** 2,
        max_lag=3,
    )
    assert direct_coupling["status"] == "ok"
    assert direct_coupling["best_abs_correlation"] > 0.99
    print("P1 geometry smoke test passed")


if __name__ == "__main__":
    main()


