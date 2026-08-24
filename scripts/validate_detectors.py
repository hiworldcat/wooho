from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from image_checks import inspect_view
from quality_checks import inspect_episode


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "初赛数据"
OUTPUT_ROOT = ROOT / "outputs" / "diagnostics"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def source_file() -> Path:
    return next(
        p for p in sorted(DATA_ROOT.rglob("episode_000000.parquet"))
        if not p.name.startswith("._") and "__MACOSX" not in str(p)
    )


def black_png() -> bytes:
    image = Image.new("RGB", (224, 224), (0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def inject_and_validate() -> dict:
    source = source_file()
    table = pq.read_table(source)
    df = table.to_pandas()
    black = {"bytes": black_png(), "path": None}

    # Inject independently detectable defects into a copy in memory.
    df.loc[50, "timestamp"] = float(df.loc[49, "timestamp"]) + 0.8
    state = np.asarray(df.loc[80, "state"], dtype=np.float32).copy()
    state[0] += 5.0
    df.at[80, "state"] = state
    df.at[100, "image"] = black
    frozen = df.loc[120, "image"]
    for index in range(121, 130):
        df.at[index, "image"] = frozen

    # Write the modified frame table to a temporary parquet file so the same
    # structural checker used for real data can inspect it unchanged.
    temp = OUTPUT_ROOT / "_injected_episode_000000.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), temp)
    structural = inspect_episode(temp, {"length": len(df)})
    image_summary, image_findings = inspect_view(df, 0, "image")
    temp.unlink(missing_ok=True)

    result = {
        "injected_defects": ["timestamp_gap", "state_spike", "black_screen", "frozen_image_run"],
        "structural_findings": [item.__dict__ for item in structural],
        "image_findings": [item.__dict__ for item in image_findings],
        "image_summary": image_summary,
    }
    return result


def main() -> None:
    result = inject_and_validate()
    output = OUTPUT_ROOT / "detector_validation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    issue_types = [item["issue_type"] for item in result["structural_findings"] + result["image_findings"]]
    print("detected_issue_types:", sorted(set(issue_types)))
    print("finding_count:", len(issue_types))
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
