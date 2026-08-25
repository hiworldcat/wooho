from __future__ import annotations

import hashlib
import io
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FINDINGS_PATH = ROOT / "outputs" / "v2" / "diagnostics" / "findings_v2.json"
BASELINES_PATH = ROOT / "outputs" / "v2" / "diagnostics" / "reference_baselines_v2.json"
OUT_DIR = ROOT / "outputs" / "v2" / "vision_state_static_clips"
VIEWS = ["image", "left_wrist_image", "right_wrist_image"]
FPS_FALLBACK = 10
PRE_SECONDS = 3.0
POST_SECONDS = 3.0


def is_real_path(path: Path) -> bool:
    return path.is_file() and not path.name.startswith("._") and "__MACOSX" not in str(path)


def find_meta_root() -> Path:
    for info_path in ROOT.rglob("info.json"):
        if is_real_path(info_path):
            return info_path.parent
    raise FileNotFoundError("Could not find info.json")


def episode_path(meta_root: Path, episode: int) -> Path:
    data_root = meta_root.parent / "data"
    matches = [
        path
        for path in data_root.rglob(f"episode_{episode:06d}.parquet")
        if is_real_path(path)
    ]
    if not matches:
        raise FileNotFoundError(f"Could not find episode {episode}")
    return matches[0]


def decode_image(value: Any) -> Image.Image:
    if not isinstance(value, dict) or not value.get("bytes"):
        raise ValueError("image cell does not contain embedded bytes")
    with Image.open(io.BytesIO(value["bytes"])) as image:
        return image.convert("RGB")


def downsample_gray(image: Image.Image, stride: int = 8) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    return arr.mean(axis=2)[::stride, ::stride]


def to_matrix(values: Any) -> np.ndarray:
    rows = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
    return np.stack(rows) if rows else np.empty((0, 0), dtype=np.float64)


def visual_motion_by_view(df: Any) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for view in VIEWS:
        previous: np.ndarray | None = None
        values: list[float] = []
        for value in df[view]:
            image = decode_image(value)
            gray = downsample_gray(image)
            if previous is not None:
                values.append(float(np.abs(gray - previous).mean()))
            previous = gray
        out[view] = np.asarray(values, dtype=np.float64)
    return out


def robust_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"p05": 0.0, "p95": 0.0, "median": 0.0}
    return {
        "p05": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
        "median": float(np.median(arr)),
    }


def baseline_lookup(baselines: dict[str, Any], section: str, task_index: int | None, *parts: str) -> dict[str, Any]:
    keys = []
    if task_index is not None:
        keys.append("|".join([f"task:{task_index}", *parts]))
    keys.append("|".join(["global", *parts]))
    for key in keys:
        stats = baselines.get(section, {}).get(key)
        if stats and stats.get("count", 0) >= 5:
            return stats
    return {}


def stat_value(stats: dict[str, Any], key: str, default: float) -> float:
    value = stats.get(key)
    if value is None:
        return default
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def safe_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(str(path), size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT_SMALL = safe_font(13)
FONT = safe_font(16)
FONT_BOLD = safe_font(18, bold=True)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill=(255, 255, 255)) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=FONT)
    draw.rectangle((bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2), fill=(0, 0, 0))
    draw.text((x, y), text, font=FONT, fill=fill)


def normalized(value: float, low: float, high: float) -> float:
    if not math.isfinite(value) or high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def plot_series(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    series: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    clip_start: int,
    clip_end: int,
    current_frame: int,
    issue_start: int,
    issue_end: int,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(248, 248, 248), outline=(40, 40, 40))
    w = x1 - x0
    h = y1 - y0
    for i in range(5):
        y = y0 + int(i * h / 4)
        draw.line((x0, y, x1, y), fill=(220, 220, 220))
    frames = np.arange(clip_start, clip_end + 1)
    issue_x0 = x0 + int((max(issue_start, clip_start) - clip_start) / max(clip_end - clip_start, 1) * w)
    issue_x1 = x0 + int((min(issue_end, clip_end) - clip_start) / max(clip_end - clip_start, 1) * w)
    draw.rectangle((issue_x0, y0, issue_x1, y1), fill=(255, 226, 226))
    all_values: list[float] = []
    for _, values, _ in series:
        all_values.extend(values[clip_start : clip_end + 1].tolist())
    max_y = max(all_values) if all_values else 1.0
    max_y = max(max_y, 1e-8)
    for series_index, (name, values, color) in enumerate(series):
        points: list[tuple[int, int]] = []
        for frame in frames:
            value = float(values[frame]) if frame < len(values) else 0.0
            x = x0 + int((frame - clip_start) / max(clip_end - clip_start, 1) * w)
            y = y1 - int(normalized(value, 0.0, max_y) * (h - 20)) - 10
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        lx = x0 + 10 + 170 * series_index
        draw.line((lx, y0 + 15, lx + 24, y0 + 15), fill=color, width=3)
        draw.text((lx + 30, y0 + 6), name, font=FONT_SMALL, fill=(20, 20, 20))
    current_x = x0 + int((current_frame - clip_start) / max(clip_end - clip_start, 1) * w)
    draw.line((current_x, y0, current_x, y1), fill=(0, 0, 0), width=2)
    draw.text((x0 + 8, y1 - 20), f"max y={max_y:.4g}", font=FONT_SMALL, fill=(60, 60, 60))


def render_frame(
    df: Any,
    episode: int,
    finding: dict[str, Any],
    frame: int,
    clip_start: int,
    clip_end: int,
    visual_combined: np.ndarray,
    state_delta_padded: np.ndarray,
    actions_delta_padded: np.ndarray,
    state_low_threshold: float,
    visual_high_threshold: float,
) -> Image.Image:
    canvas = Image.new("RGB", (672, 560), (24, 26, 30))
    draw = ImageDraw.Draw(canvas)
    title = (
        f"{finding['finding_id']} | episode {episode} | frame {frame} | "
        f"issue {finding['frame_start']}-{finding['frame_end']}"
    )
    draw.text((10, 8), title, font=FONT_BOLD, fill=(255, 255, 255))
    y_img = 40
    for idx, view in enumerate(VIEWS):
        x = idx * 224
        image = decode_image(df.iloc[frame][view]).resize((224, 224))
        canvas.paste(image, (x, y_img))
        draw.rectangle((x, y_img, x + 223, y_img + 223), outline=(255, 255, 255), width=1)
        motion_idx = max(0, min(frame - 1, len(visual_combined) - 1))
        draw_label(draw, (x + 6, y_img + 6), view)
        draw_label(draw, (x + 6, y_img + 196), f"vision={visual_combined[motion_idx]:.3f}")
    current_idx = max(0, min(frame - 1, len(visual_combined) - 1))
    state_value = state_delta_padded[frame] if frame < len(state_delta_padded) else 0.0
    action_value = actions_delta_padded[frame] if frame < len(actions_delta_padded) else 0.0
    lines = [
        f"state_delta={state_value:.6g} | actions_delta={action_value:.6g} | visual_motion={visual_combined[current_idx]:.4g}",
        f"static tolerance: low_dim <= {state_low_threshold:.6g}; visual high cutoff in this episode: >= {visual_high_threshold:.4g}",
    ]
    for i, line in enumerate(lines):
        draw.text((12, 278 + i * 22), line, font=FONT, fill=(235, 235, 235))
    plot_series(
        draw,
        (30, 330, 642, 540),
        [
            ("vision_motion", np.r_[visual_combined[0], visual_combined], (220, 64, 58)),
            ("state_delta", state_delta_padded, (54, 132, 230)),
            ("actions_delta", actions_delta_padded, (36, 151, 103)),
        ],
        clip_start,
        clip_end,
        frame,
        int(finding["frame_start"]),
        int(finding["frame_end"]),
    )
    return canvas


class MjpegAviWriter:
    def __init__(self, path: Path, fps: int, size: tuple[int, int]) -> None:
        self.path = path
        self.fps = fps
        self.width, self.height = size
        self.frames: list[bytes] = []

    def add(self, image: Image.Image) -> None:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=88)
        self.frames.append(buf.getvalue())

    @staticmethod
    def chunk(tag: bytes, payload: bytes) -> bytes:
        pad = b"\0" if len(payload) % 2 else b""
        return tag + struct.pack("<I", len(payload)) + payload + pad

    @staticmethod
    def list_chunk(tag: bytes, payload: bytes) -> bytes:
        data = tag + payload
        pad = b"\0" if len(data) % 2 else b""
        return b"LIST" + struct.pack("<I", len(data)) + data + pad

    def write(self) -> None:
        total = len(self.frames)
        max_frame = max((len(frame) for frame in self.frames), default=0)
        avih = struct.pack(
            "<IIIIIIIIIIIIII",
            int(1_000_000 / self.fps),
            max_frame * self.fps,
            0,
            0x10,
            total,
            0,
            1,
            max_frame,
            self.width,
            self.height,
            0,
            0,
            0,
            0,
        )
        strh = struct.pack(
            "<4s4sIHHIIIIIIIIhhhh",
            b"vids",
            b"MJPG",
            0,
            0,
            0,
            0,
            1,
            self.fps,
            0,
            total,
            max_frame,
            0xFFFFFFFF,
            0,
            0,
            0,
            self.width,
            self.height,
        )
        strf = struct.pack(
            "<IiiHH4sIiiII",
            40,
            self.width,
            self.height,
            1,
            24,
            b"MJPG",
            max_frame,
            0,
            0,
            0,
            0,
        )
        hdrl = self.list_chunk(
            b"hdrl",
            self.chunk(b"avih", avih)
            + self.list_chunk(b"strl", self.chunk(b"strh", strh) + self.chunk(b"strf", strf)),
        )
        movi_payload = io.BytesIO()
        index_entries: list[bytes] = []
        offset = 4
        for frame in self.frames:
            payload_size = len(frame)
            movi_payload.write(b"00dc")
            movi_payload.write(struct.pack("<I", payload_size))
            movi_payload.write(frame)
            if payload_size % 2:
                movi_payload.write(b"\0")
            index_entries.append(struct.pack("<4sIII", b"00dc", 0x10, offset, payload_size))
            offset += 8 + payload_size + (payload_size % 2)
        movi = self.list_chunk(b"movi", movi_payload.getvalue())
        idx1 = self.chunk(b"idx1", b"".join(index_entries))
        body = b"AVI " + hdrl + movi + idx1
        self.path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def save_gif(path: Path, frames: list[Image.Image], fps: int) -> None:
    if not frames:
        return
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_root = find_meta_root()
    info = json.loads((meta_root / "info.json").read_text(encoding="utf-8"))
    baselines = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
    fps = int(info.get("fps") or FPS_FALLBACK)
    pre_frames = int(round(PRE_SECONDS * fps))
    post_frames = int(round(POST_SECONDS * fps))
    findings = [
        finding
        for finding in json.loads(FINDINGS_PATH.read_text(encoding="utf-8"))
        if finding.get("issue_type") == "visual_moves_state_static"
    ]
    summaries: list[dict[str, Any]] = []
    for finding in findings:
        episode = int(finding["episode_index"])
        path = episode_path(meta_root, episode)
        table = pq.read_table(path, columns=[*VIEWS, "state", "actions", "frame_index"])
        df = table.to_pandas()
        frames = df["frame_index"].to_numpy(dtype=int)
        state = to_matrix(df["state"])
        actions = to_matrix(df["actions"])
        state_delta = np.linalg.norm(np.diff(state, axis=0), axis=1) if len(state) > 1 else np.array([])
        actions_delta = np.linalg.norm(np.diff(actions, axis=0), axis=1) if len(actions) > 1 else np.array([])
        state_delta_padded = np.r_[0.0, state_delta]
        actions_delta_padded = np.r_[0.0, actions_delta]
        by_view = visual_motion_by_view(df)
        min_len = min(len(values) for values in by_view.values())
        visual_combined = np.mean(np.stack([values[:min_len] for values in by_view.values()]), axis=0)
        visual_stats = robust_stats(visual_combined)
        low_dim = actions_delta if finding.get("column") == "actions" else state_delta
        low_dim_stats = baseline_lookup(baselines, "state_delta", finding.get("task_index"), str(finding.get("column")))
        state_low_threshold = max(1e-8, stat_value(low_dim_stats, "p01", 1e-8) * 0.5)
        visual_high_threshold = visual_stats["p95"]
        issue_start = int(finding["frame_start"])
        issue_end = int(finding["frame_end"])
        clip_start = max(0, issue_start - pre_frames)
        clip_end = min(len(df) - 1, issue_end + post_frames)
        stem = f"{finding['finding_id']}_episode-{episode:02d}_frames-{issue_start:04d}-{issue_end:04d}"
        avi_path = OUT_DIR / f"{stem}.avi"
        gif_path = OUT_DIR / f"{stem}.gif"
        if not avi_path.exists() or not gif_path.exists():
            rendered: list[Image.Image] = []
            for frame in range(clip_start, clip_end + 1):
                rendered.append(
                    render_frame(
                        df,
                        episode,
                        finding,
                        frame,
                        clip_start,
                        clip_end,
                        visual_combined,
                        state_delta_padded,
                        actions_delta_padded,
                        state_low_threshold,
                        visual_high_threshold,
                    )
                )
            writer = MjpegAviWriter(avi_path, fps=fps, size=rendered[0].size)
            for image in rendered:
                writer.add(image)
            writer.write()
            save_gif(gif_path, rendered, fps=fps)
        motion_idx0 = max(0, issue_start - 1)
        motion_idx1 = min(len(visual_combined) - 1, issue_end - 1)
        summaries.append(
            {
                "finding_id": finding["finding_id"],
                "episode_index": episode,
                "task_index": finding.get("task_index"),
                "column": finding.get("column"),
                "issue_frames": [issue_start, issue_end],
                "clip_frames": [clip_start, clip_end],
                "clip_seconds": [round(clip_start / fps, 3), round(clip_end / fps, 3)],
                "fps": fps,
                "avi_path": str(avi_path),
                "gif_path": str(gif_path),
                "state_low_threshold": state_low_threshold,
                "visual_high_threshold": visual_high_threshold,
                "issue_max_visual_motion": float(np.max(visual_combined[motion_idx0 : motion_idx1 + 1])),
                "issue_median_state_delta": float(np.median(state_delta_padded[issue_start : issue_end + 1])),
                "issue_median_actions_delta": float(np.median(actions_delta_padded[issue_start : issue_end + 1])),
                "input_evidence": finding.get("evidence", {}),
            }
        )
    (OUT_DIR / "visual_moves_state_static_clip_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Visual Moves While State/Actions Static Clips",
        "",
        f"- findings: {len(summaries)}",
        f"- fps: {fps}",
        f"- clip window: {PRE_SECONDS:g}s before to {POST_SECONDS:g}s after each issue segment",
        "",
        "| finding | episode | issue frames | clip frames | low-dim tolerance | visual high cutoff | video | gif |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in summaries:
        lines.append(
            "| "
            f"{item['finding_id']} | {item['episode_index']} | "
            f"{item['issue_frames'][0]}-{item['issue_frames'][1]} | "
            f"{item['clip_frames'][0]}-{item['clip_frames'][1]} | "
            f"{item['state_low_threshold']:.6g} | {item['visual_high_threshold']:.4g} | "
            f"`{Path(item['avi_path']).name}` | `{Path(item['gif_path']).name}` |"
        )
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(summaries)} clips to {OUT_DIR}")
    for item in summaries:
        digest = hashlib.sha1(Path(item["avi_path"]).read_bytes()).hexdigest()[:10]
        print(f"{item['finding_id']} ep={item['episode_index']} avi={Path(item['avi_path']).name} sha1={digest}")


if __name__ == "__main__":
    main()
