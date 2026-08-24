from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_real_parquet(path: Path) -> bool:
    return path.is_file() and not path.name.startswith("._") and "__MACOSX" not in str(path)


def find_meta_root(root: Path) -> Path:
    for candidate in root.rglob("info.json"):
        if "__MACOSX" not in str(candidate):
            return candidate.parent
    raise FileNotFoundError("info.json not found")


def decode_image(cell) -> Image.Image:
    with Image.open(io.BytesIO(cell["bytes"])) as image:
        return image.convert("RGB")


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def build_board(episode: dict, out_dir: Path, sample_count: int) -> dict:
    episode_index = int(episode["episode_index"])
    parquet_path = Path(episode["parquet_path"])
    df = pq.read_table(parquet_path).to_pandas()
    sample_indices = np.unique(np.linspace(0, len(df) - 1, sample_count, dtype=int))

    fonts = {"title": load_font(18), "small": load_font(14)}
    tile_w = 120
    tile_h = 120
    left_w = 148
    top_h = 68
    row_h = tile_h + 22
    board = Image.new(
        "RGB",
        (left_w + len(sample_indices) * tile_w + 24, top_h + 3 * row_h + 18),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(board)
    title = f"Episode {episode_index} | {episode['task_label']} | {len(df)} frames"
    draw.text((16, 16), title, fill=(30, 30, 30), font=fonts["title"])
    draw.text((16, 42), "sampled observations across the trajectory", fill=(100, 100, 100), font=fonts["small"])

    row_names = [("image", "main"), ("left_wrist_image", "left wrist"), ("right_wrist_image", "right wrist")]
    samples = []
    prev_main_gray = None

    for j, frame_index in enumerate(sample_indices):
        row = df.iloc[int(frame_index)]
        x = left_w + j * tile_w
        draw.text((x + 42, 45), str(int(row["frame_index"])), fill=(50, 50, 50), font=fonts["small"])
        draw.line((x + tile_w // 2, 63, x + tile_w // 2, 75), fill=(190, 190, 190), width=1)

    for r, (key, label) in enumerate(row_names):
        y = top_h + r * row_h
        draw.text((16, y + 46), label, fill=(90, 90, 90), font=fonts["title"])
        for j, frame_index in enumerate(sample_indices):
            row = df.iloc[int(frame_index)]
            x = left_w + j * tile_w
            img = decode_image(row[key])
            thumb = img.copy()
            thumb.thumbnail((tile_w - 4, tile_h - 4))
            bx = x + (tile_w - thumb.width) // 2
            by = y + 2
            board.paste(thumb, (bx, by))
            draw.rectangle([bx, by, bx + thumb.width - 1, by + thumb.height - 1], outline=(200, 200, 200))
            draw.text((x + 34, y + tile_h + 4), f"f{int(row['frame_index'])}", fill=(90, 90, 90), font=fonts["small"])

        if key == "main":
            # collect coarse motion metrics from the sampled main images
            for frame_index in sample_indices:
                row = df.iloc[int(frame_index)]
                main = decode_image(row["image"])
                gray = np.asarray(main.convert("L"), dtype=np.float32)
                if prev_main_gray is None:
                    main_change = 0.0
                else:
                    main_change = float(np.abs(gray - prev_main_gray).mean())
                prev_main_gray = gray
                state = np.asarray(row["state"], dtype=np.float32)
                actions = np.asarray(row["actions"], dtype=np.float32)
                samples.append(
                    {
                        "sample_index": int(len(samples)),
                        "frame_index": int(row["frame_index"]),
                        "timestamp": float(row["timestamp"]),
                        "state_norm": float(np.linalg.norm(state)),
                        "action_norm": float(np.linalg.norm(actions)),
                        "state_delta": float(np.linalg.norm(state - np.asarray(df.iloc[max(int(frame_index) - 1, 0)]["state"], dtype=np.float32))) if frame_index > 0 else 0.0,
                        "action_delta": float(np.linalg.norm(actions - np.asarray(df.iloc[max(int(frame_index) - 1, 0)]["actions"], dtype=np.float32))) if frame_index > 0 else 0.0,
                        "main_change": main_change,
                    }
                )

    board_dir = out_dir / f"episode-{episode_index:02d}"
    board_dir.mkdir(parents=True, exist_ok=True)
    board_path = board_dir / "board.png"
    board.save(board_path, format="PNG", optimize=True)

    return {
        "episode_index": episode_index,
        "task_index": int(episode["task_index"]),
        "task_label": episode["task_label"],
        "length": int(len(df)),
        "board_path": str(board_path),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build trajectory observation boards")
    parser.add_argument("--sample-count", type=int, default=8, help="Number of sampled frames per trajectory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=find_repo_root() / "outputs" / "trajectory_boards",
        help="Where to write the boards and manifest",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    data_root = repo_root / "初赛数据"
    meta_root = find_meta_root(data_root)
    episodes_meta = {
        int(row["episode_index"]): row
        for row in (
            json.loads(line)
            for line in (meta_root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    parquet_files = sorted(
        p for p in data_root.rglob("*.parquet") if is_real_parquet(p)
    )
    episode_rows = []
    for parquet_path in parquet_files:
        episode_index = int(parquet_path.stem.split("_")[-1])
        meta = episodes_meta.get(episode_index, {})
        if not meta:
            continue
        task_label = meta.get("tasks", ["unknown"])[0]
        episode_rows.append(
            {
                "episode_index": episode_index,
                "task_label": "Material retrieval" if "Material Retrieval" in task_label else "Assembly",
                "task_index": 0 if "Material Retrieval" in task_label else 1,
                "parquet_path": str(parquet_path),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    built = [build_board(row, args.output_dir, args.sample_count) for row in episode_rows]
    manifest = {
        "sample_count": args.sample_count,
        "episodes": built,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"built {len(built)} trajectory boards in {args.output_dir}")
    print(f"manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
