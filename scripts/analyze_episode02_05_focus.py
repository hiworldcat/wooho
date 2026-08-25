from __future__ import annotations

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
OUT_DIR = ROOT / "outputs" / "v2" / "episode_02_05_focus"
VIEWS = ["image", "left_wrist_image", "right_wrist_image"]
FOCUS_ITEMS = [
    {"finding_id": "V2F-000004", "episode": 2, "task": 0, "issue": (127, 130), "clip": (97, 160)},
    {"finding_id": "V2F-000015", "episode": 5, "task": 0, "issue": (164, 167), "clip": (134, 197)},
]


def is_real_path(path: Path) -> bool:
    return path.is_file() and not path.name.startswith("._") and "__MACOSX" not in str(path)


def find_meta_root() -> Path:
    for info_path in ROOT.rglob("info.json"):
        if is_real_path(info_path):
            return info_path.parent
    raise FileNotFoundError("Could not find info.json")


def episode_path(meta_root: Path, episode: int) -> Path:
    for path in (meta_root.parent / "data").rglob(f"episode_{episode:06d}.parquet"):
        if is_real_path(path):
            return path
    raise FileNotFoundError(f"Could not find episode {episode}")


def decode_image(value: Any) -> Image.Image:
    if not isinstance(value, dict) or not value.get("bytes"):
        raise ValueError("image cell does not contain embedded bytes")
    with Image.open(io.BytesIO(value["bytes"])) as image:
        return image.convert("RGB")


def to_matrix(values: Any) -> np.ndarray:
    rows = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
    return np.stack(rows) if rows else np.empty((0, 0), dtype=np.float64)


def downsample_gray(image: Image.Image, stride: int = 8) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    return arr.mean(axis=2)[::stride, ::stride]


def metrics(df: Any) -> dict[str, Any]:
    grays: dict[str, list[np.ndarray]] = {view: [] for view in VIEWS}
    motion_by_view: dict[str, list[float]] = {view: [] for view in VIEWS}
    diff_by_view: dict[str, list[Image.Image | None]] = {view: [] for view in VIEWS}
    for view in VIEWS:
        for value in df[view]:
            image = decode_image(value)
            gray = downsample_gray(image)
            grays[view].append(gray)
        for idx in range(len(grays[view])):
            if idx + 1 < len(grays[view]):
                diff = np.abs(grays[view][idx + 1] - grays[view][idx])
                motion_by_view[view].append(float(diff.mean()))
                diff_img = Image.fromarray(np.uint8(np.clip(diff / max(diff.max(), 1.0) * 255.0, 0, 255))).resize((224, 224))
                diff_by_view[view].append(diff_img.convert("RGB"))
            else:
                motion_by_view[view].append(0.0)
                diff_by_view[view].append(None)
    min_len = min(len(values) for values in motion_by_view.values())
    combined = np.mean(np.stack([np.asarray(motion_by_view[v][:min_len], dtype=np.float64) for v in VIEWS]), axis=0)
    return {
        "motion_by_view": {k: np.asarray(v, dtype=np.float64) for k, v in motion_by_view.items()},
        "diff_by_view": diff_by_view,
        "combined_motion": combined,
    }


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


FONT_SMALL = safe_font(12)
FONT = safe_font(15)
FONT_BOLD = safe_font(18, bold=True)


def norm(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value) or hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def draw_text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color=(255, 255, 255)) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=FONT)
    draw.rectangle((bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2), fill=(0, 0, 0))
    draw.text((x, y), text, font=FONT, fill=color)


def draw_plot(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    series: list[tuple[str, np.ndarray, tuple[int, int, int], float | None]],
    clip: tuple[int, int],
    issue: tuple[int, int],
    current: int,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(250, 250, 250), outline=(60, 60, 60))
    clip_start, clip_end = clip
    span = max(1, clip_end - clip_start)
    issue_x0 = x0 + int((issue[0] - clip_start) / span * (x1 - x0))
    issue_x1 = x0 + int((issue[1] - clip_start) / span * (x1 - x0))
    draw.rectangle((issue_x0, y0, issue_x1, y1), fill=(255, 225, 225))
    for i in range(5):
        y = y0 + int(i * (y1 - y0) / 4)
        draw.line((x0, y, x1, y), fill=(220, 220, 220))
    values = []
    for _, arr, _, threshold in series:
        values.extend(arr[clip_start : clip_end + 1].tolist())
        if threshold is not None:
            values.append(threshold)
    ymax = max(max(values), 1e-8) if values else 1.0
    for index, (name, arr, color, threshold) in enumerate(series):
        points = []
        for frame in range(clip_start, clip_end + 1):
            value = float(arr[frame]) if frame < len(arr) else 0.0
            x = x0 + int((frame - clip_start) / span * (x1 - x0))
            y = y1 - 10 - int(norm(value, 0.0, ymax) * (y1 - y0 - 22))
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        if threshold is not None:
            ty = y1 - 10 - int(norm(threshold, 0.0, ymax) * (y1 - y0 - 22))
            draw.line((x0, ty, x1, ty), fill=color, width=1)
        lx = x0 + 10 + index * 205
        draw.line((lx, y0 + 18, lx + 28, y0 + 18), fill=color, width=3)
        draw.text((lx + 34, y0 + 9), name, font=FONT_SMALL, fill=(20, 20, 20))
    cx = x0 + int((current - clip_start) / span * (x1 - x0))
    draw.line((cx, y0, cx, y1), fill=(0, 0, 0), width=2)
    draw.text((x0 + 8, y1 - 19), f"clip {clip_start}-{clip_end}, red={issue[0]}-{issue[1]}, y max={ymax:.3g}", font=FONT_SMALL, fill=(40, 40, 40))


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
        return tag + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) % 2 else b"")

    @staticmethod
    def list_chunk(tag: bytes, payload: bytes) -> bytes:
        data = tag + payload
        return b"LIST" + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")

    def write(self) -> None:
        total = len(self.frames)
        max_frame = max(len(frame) for frame in self.frames)
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
        strf = struct.pack("<IiiHH4sIiiII", 40, self.width, self.height, 1, 24, b"MJPG", max_frame, 0, 0, 0, 0)
        hdrl = self.list_chunk(b"hdrl", self.chunk(b"avih", avih) + self.list_chunk(b"strl", self.chunk(b"strh", strh) + self.chunk(b"strf", strf)))
        movi_payload = io.BytesIO()
        index = []
        offset = 4
        for frame in self.frames:
            size = len(frame)
            movi_payload.write(b"00dc")
            movi_payload.write(struct.pack("<I", size))
            movi_payload.write(frame)
            if size % 2:
                movi_payload.write(b"\0")
            index.append(struct.pack("<4sIII", b"00dc", 0x10, offset, size))
            offset += 8 + size + (size % 2)
        movi = self.list_chunk(b"movi", movi_payload.getvalue())
        idx1 = self.chunk(b"idx1", b"".join(index))
        body = b"AVI " + hdrl + movi + idx1
        self.path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def save_gif(path: Path, frames: list[Image.Image], fps: int) -> None:
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=False)


def render_frame(
    df: Any,
    item: dict[str, Any],
    frame: int,
    clip: tuple[int, int],
    issue: tuple[int, int],
    state_delta: np.ndarray,
    action_delta: np.ndarray,
    combined_motion: np.ndarray,
    motion_by_view: dict[str, np.ndarray],
    diff_by_view: dict[str, list[Image.Image | None]],
    action_static_threshold: float,
    visual_high_threshold: float,
) -> Image.Image:
    canvas = Image.new("RGB", (1000, 720), (30, 32, 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), f"Episode {item['episode']} | {item['finding_id']} | frame {frame} | issue {issue[0]}-{issue[1]}", font=FONT_BOLD, fill=(255, 255, 255))
    for i, view in enumerate(VIEWS):
        x = 16 + i * 326
        image = decode_image(df.iloc[frame][view]).resize((180, 180))
        diff = diff_by_view[view][frame]
        if diff is None:
            diff = Image.new("RGB", (180, 180), (0, 0, 0))
        else:
            diff = diff.resize((180, 180))
        canvas.paste(image, (x, 48))
        canvas.paste(diff, (x + 128, 48))
        draw.rectangle((x, 48, x + 179, 227), outline=(255, 255, 255), width=1)
        draw.rectangle((x + 128, 48, x + 307, 227), outline=(255, 255, 255), width=1)
        draw_text_box(draw, (x + 6, 54), view)
        draw_text_box(draw, (x + 134, 202), "diff next")
        draw.text((x, 236), f"{view} motion {frame}->{frame + 1}: {motion_by_view[view][frame]:.3f}", font=FONT, fill=(235, 235, 235))
    current_state = float(state_delta[frame])
    current_action = float(action_delta[frame])
    current_visual = float(combined_motion[frame])
    y0 = 274
    if issue[0] <= frame <= issue[1]:
        draw.rectangle((14, y0 - 6, 986, y0 + 82), fill=(78, 34, 34))
    lines = [
        f"transition {frame}->{frame + 1}: visual_motion={current_visual:.4g} | state_delta={current_state:.6g} | actions_delta={current_action:.6g}",
        f"rule in red span: actions_delta <= {action_static_threshold:.1e} AND combined visual motion >= local p95 ({visual_high_threshold:.3f})",
        "interpretation: command/action row is repeated, while observed robot/object/camera image still changes.",
    ]
    for idx, line in enumerate(lines):
        draw.text((20, y0 + idx * 24), line, font=FONT, fill=(255, 255, 255))
    draw_plot(
        draw,
        (50, 372, 955, 690),
        [
            ("combined vision", combined_motion, (220, 64, 58), visual_high_threshold),
            ("state delta", state_delta, (54, 132, 230), None),
            ("actions delta", action_delta, (36, 151, 103), action_static_threshold),
        ],
        clip,
        issue,
        frame,
    )
    return canvas


def analyze_one(meta_root: Path, fps: int, item: dict[str, Any]) -> dict[str, Any]:
    path = episode_path(meta_root, int(item["episode"]))
    table = pq.read_table(path, columns=[*VIEWS, "state", "actions", "frame_index"])
    df = table.to_pandas()
    state = to_matrix(df["state"])
    actions = to_matrix(df["actions"])
    state_delta = np.r_[np.linalg.norm(np.diff(state, axis=0), axis=1), 0.0]
    action_delta = np.r_[np.linalg.norm(np.diff(actions, axis=0), axis=1), 0.0]
    image = metrics(df)
    combined_motion = image["combined_motion"]
    issue = tuple(item["issue"])
    clip = tuple(item["clip"])
    action_static_threshold = 1e-8
    visual_high_threshold = float(np.quantile(combined_motion[clip[0] : clip[1] + 1], 0.95))
    issue_slice = slice(issue[0], issue[1] + 1)
    per_view_issue = {
        view: {
            "max": float(np.max(values[issue_slice])),
            "median": float(np.median(values[issue_slice])),
        }
        for view, values in image["motion_by_view"].items()
    }
    repeated_actions = bool(np.all(np.linalg.norm(actions[issue[0] : issue[1] + 1] - actions[issue[0]], axis=1) <= action_static_threshold))
    frames: list[Image.Image] = []
    for frame in range(clip[0], clip[1] + 1):
        frames.append(
            render_frame(
                df,
                item,
                frame,
                clip,
                issue,
                state_delta,
                action_delta,
                combined_motion,
                image["motion_by_view"],
                image["diff_by_view"],
                action_static_threshold,
                visual_high_threshold,
            )
        )
    stem = f"episode-{item['episode']:02d}_{item['finding_id']}_focus_frames-{issue[0]:04d}-{issue[1]:04d}"
    avi = OUT_DIR / f"{stem}.avi"
    gif = OUT_DIR / f"{stem}.gif"
    writer = MjpegAviWriter(avi, fps=fps, size=frames[0].size)
    for frame_image in frames:
        writer.add(frame_image)
    writer.write()
    save_gif(gif, frames, fps)
    key_frames = [clip[0], issue[0] - 1, issue[0], issue[1] - 1, issue[1], issue[1] + 1, clip[1]]
    rows = []
    for frame in key_frames:
        if 0 <= frame < len(df):
            rows.append(
                {
                    "frame": int(frame),
                    "time_s": round(frame / fps, 3),
                    "transition": f"{frame}->{frame + 1}",
                    "combined_vision_motion": float(combined_motion[frame]),
                    "state_delta": float(state_delta[frame]),
                    "actions_delta": float(action_delta[frame]),
                    "left_pos": [round(float(x), 5) for x in state[frame, 0:3]],
                    "right_pos": [round(float(x), 5) for x in state[frame, 10:13]],
                }
            )
    return {
        "finding_id": item["finding_id"],
        "episode": int(item["episode"]),
        "issue_frames": list(issue),
        "clip_frames": list(clip),
        "clip_seconds": [round(clip[0] / fps, 3), round(clip[1] / fps, 3)],
        "repeated_actions_in_issue": repeated_actions,
        "action_static_threshold": action_static_threshold,
        "visual_high_threshold": visual_high_threshold,
        "issue_combined_vision_motion_max": float(np.max(combined_motion[issue_slice])),
        "issue_combined_vision_motion_median": float(np.median(combined_motion[issue_slice])),
        "issue_state_delta_max": float(np.max(state_delta[issue_slice])),
        "issue_state_delta_median": float(np.median(state_delta[issue_slice])),
        "issue_actions_delta_max": float(np.max(action_delta[issue_slice])),
        "issue_actions_delta_median": float(np.median(action_delta[issue_slice])),
        "per_view_issue_motion": per_view_issue,
        "key_frames": rows,
        "avi": str(avi),
        "gif": str(gif),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_root = find_meta_root()
    info = json.loads((meta_root / "info.json").read_text(encoding="utf-8"))
    fps = int(info.get("fps") or 10)
    summary = [analyze_one(meta_root, fps, item) for item in FOCUS_ITEMS]
    (OUT_DIR / "episode_02_05_focus_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Episode 02/05 Focus Analysis",
        "",
        "The flagged column is `actions`, not the sensor `state`: the action command row is repeated while the observed state and images continue changing.",
        "",
        "| episode | issue frames | repeated actions | visual p95 cutoff | issue max visual | state median delta | actions median delta | video |",
        "|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['episode']} | {row['issue_frames'][0]}-{row['issue_frames'][1]} | "
            f"{row['repeated_actions_in_issue']} | {row['visual_high_threshold']:.3f} | "
            f"{row['issue_combined_vision_motion_max']:.3f} | {row['issue_state_delta_median']:.6g} | "
            f"{row['issue_actions_delta_median']:.6g} | `{Path(row['avi']).name}` |"
        )
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote focused videos and summary to {OUT_DIR}")
    for row in summary:
        print(
            row["episode"],
            row["issue_frames"],
            "visual_max",
            round(row["issue_combined_vision_motion_max"], 3),
            "state_med",
            round(row["issue_state_delta_median"], 6),
            "actions_med",
            round(row["issue_actions_delta_median"], 6),
        )


if __name__ == "__main__":
    main()
