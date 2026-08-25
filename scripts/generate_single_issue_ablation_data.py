from __future__ import annotations

import argparse
import io
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "ablations" / "single_issue_cases"
IMAGE_COLUMNS = ["image", "left_wrist_image", "right_wrist_image"]
LOW_DIM_COLUMNS = ["state", "actions"]


@dataclass(frozen=True)
class IssueSpec:
    issue_id: str
    axis: str
    hardness: str
    category: str
    target: str
    description: str
    expected_detector_family: str
    apply: Callable[[pd.DataFrame, dict[str, Any], np.random.Generator], dict[str, Any]]


def is_real_path(path: Path) -> bool:
    ignored_parts = {"outputs", "_upload_package", "__MACOSX"}
    return path.is_file() and not any(part in ignored_parts for part in path.parts) and not path.name.startswith("._")


def find_meta_root() -> Path:
    for candidate in ROOT.rglob("info.json"):
        if is_real_path(candidate):
            return candidate.parent
    raise FileNotFoundError("info.json not found under workspace")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_columns_from_info(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    columns = [name for name, spec in features.items() if spec.get("dtype") == "image"]
    return columns or IMAGE_COLUMNS


def find_episode_files(meta_root: Path) -> list[Path]:
    data_root = meta_root.parent
    files = sorted(path for path in data_root.rglob("*.parquet") if is_real_path(path))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_root}")
    return files


def safe_episode_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def choose_source_episode(files: list[Path], min_rows: int = 80) -> Path:
    for path in files:
        if pq.ParquetFile(path).metadata.num_rows >= min_rows:
            return path
    return files[0]


def load_episode(path: Path) -> tuple[pd.DataFrame, pa.Schema]:
    table = pq.read_table(path)
    return table.to_pandas(), table.schema


def write_episode(df: pd.DataFrame, schema: pa.Schema, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path)


def decode_image(cell: Any) -> np.ndarray:
    if not isinstance(cell, dict) or not cell.get("bytes"):
        raise ValueError("image cell does not contain bytes")
    with Image.open(io.BytesIO(cell["bytes"])) as image:
        return np.asarray(image.convert("RGB"))


def encode_image_cell(array: np.ndarray, original: Any) -> dict[str, Any]:
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(buffer, format="PNG")
    path = original.get("path") if isinstance(original, dict) else None
    return {"bytes": buffer.getvalue(), "path": path}


def segment(df: pd.DataFrame, length: int, anchor: float = 0.35) -> tuple[int, int]:
    n = len(df)
    length = max(1, min(length, n - 2))
    start = min(max(1, int(n * anchor)), n - length - 1)
    return start, start + length


def replace_view_with_solid(df: pd.DataFrame, view: str, value: int, length: int) -> dict[str, Any]:
    start, end = segment(df, length)
    original = df.at[start, view]
    shape = decode_image(original).shape
    image = np.full(shape, value, dtype=np.uint8)
    for row in range(start, end):
        df.at[row, view] = encode_image_cell(image, df.at[row, view])
    return {"view": view, "frame_start": start, "frame_end": end - 1, "value": value, "frames": end - start}


def apply_black_screen(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return replace_view_with_solid(df, context["image_columns"][0], 0, 12)


def apply_white_screen(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return replace_view_with_solid(df, context["image_columns"][0], 255, 12)


def apply_flower_screen(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    view = context["image_columns"][0]
    start, end = segment(df, 12)
    shape = decode_image(df.at[start, view]).shape
    block = 8
    coarse_shape = (int(np.ceil(shape[0] / block)), int(np.ceil(shape[1] / block)), shape[2])
    coarse = rng.integers(0, 256, size=coarse_shape, dtype=np.uint8)
    image = np.repeat(np.repeat(coarse, block, axis=0), block, axis=1)[: shape[0], : shape[1], :]
    stripe = np.arange(shape[1], dtype=np.uint8)[None, :, None]
    image = (image.astype(np.uint16) + stripe.astype(np.uint16) * 3) % 256
    image = image.astype(np.uint8)
    for row in range(start, end):
        df.at[row, view] = encode_image_cell(image, df.at[row, view])
    return {"view": view, "frame_start": start, "frame_end": end - 1, "frames": end - start, "block_size": block}


def apply_invalid_image_bytes(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    view = context["image_columns"][0]
    start, end = segment(df, 4)
    for row in range(start, end):
        cell = dict(df.at[row, view])
        cell["bytes"] = b"not-a-valid-image-payload"
        df.at[row, view] = cell
    return {"view": view, "frame_start": start, "frame_end": end - 1, "frames": end - start}


def set_low_dim_value(df: pd.DataFrame, column: str, value: float, length: int = 4, dim: int = 0) -> dict[str, Any]:
    start, end = segment(df, length)
    for row in range(start, end):
        values = list(df.at[row, column])
        values[dim] = value
        df.at[row, column] = values
    return {"column": column, "dimension": dim, "frame_start": start, "frame_end": end - 1, "frames": end - start}


def apply_state_nan(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return set_low_dim_value(df, "state", float("nan"))


def apply_state_inf(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return set_low_dim_value(df, "state", float("inf"))


def apply_action_nan(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return set_low_dim_value(df, "actions", float("nan"))


def apply_state_wrong_dim(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    start, _ = segment(df, 1)
    original_len = len(df.at[start, "state"])
    df.at[start, "state"] = list(df.at[start, "state"])[:-1]
    return {"column": "state", "frame_start": start, "frame_end": start, "original_dim": original_len, "new_dim": original_len - 1}


def apply_timestamp_non_monotonic(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    start, _ = segment(df, 1)
    before = float(df.at[start, "timestamp"])
    df.at[start, "timestamp"] = float(df.at[start - 1, "timestamp"]) - 0.1
    return {"column": "timestamp", "frame_start": start, "frame_end": start, "old_value": before, "new_value": float(df.at[start, "timestamp"])}


def apply_frame_index_duplicate(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    start, _ = segment(df, 1)
    old = int(df.at[start, "frame_index"])
    df.at[start, "frame_index"] = int(df.at[start - 1, "frame_index"])
    return {"column": "frame_index", "frame_start": start, "frame_end": start, "old_value": old, "new_value": int(df.at[start, "frame_index"])}


def apply_frame_index_skip(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    start, _ = segment(df, 1)
    skip = 3
    df.loc[start:, "frame_index"] = df.loc[start:, "frame_index"].astype(np.int64) + skip
    return {"column": "frame_index", "frame_start": start, "frame_end": len(df) - 1, "skip": skip}


def apply_row_reorder(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    start, _ = segment(df, 1)
    temp = df.iloc[start].copy()
    df.iloc[start] = df.iloc[start + 1]
    df.iloc[start + 1] = temp
    return {"frame_start": start, "frame_end": start + 1, "operation": "swap_adjacent_rows"}


def repeat_previous_images(df: pd.DataFrame, views: list[str], length: int) -> dict[str, Any]:
    start, end = segment(df, length)
    for view in views:
        previous = df.at[start - 1, view]
        for row in range(start, end):
            df.at[row, view] = previous
    return {"views": views, "frame_start": start, "frame_end": end - 1, "frames": end - start}


def apply_vision_drop_small(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return repeat_previous_images(df, [context["image_columns"][0]], 4)


def apply_vision_drop_large(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return repeat_previous_images(df, [context["image_columns"][0]], 30)


def jump_images_from_future(df: pd.DataFrame, views: list[str], length: int, offset: int) -> dict[str, Any]:
    n = len(df)
    start, end = segment(df, length, anchor=0.25)
    if end + offset >= n:
        offset = max(1, n - end - 1)
    for view in views:
        for row in range(start, end):
            df.at[row, view] = df.at[row + offset, view]
    return {"views": views, "frame_start": start, "frame_end": end - 1, "frames": end - start, "source_offset_frames": offset}


def apply_vision_jump_small(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return jump_images_from_future(df, [context["image_columns"][0]], 4, 12)


def apply_vision_jump_large(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return jump_images_from_future(df, [context["image_columns"][0]], 30, 24)


def delay_column(df: pd.DataFrame, columns: list[str], delay: int) -> dict[str, Any]:
    for column in columns:
        original = df[column].copy()
        for row in range(delay, len(df)):
            df.at[row, column] = original.iloc[row - delay]
        for row in range(delay):
            df.at[row, column] = original.iloc[0]
    return {"columns": columns, "delay_frames": delay, "delay_seconds": round(delay / 10.0, 3), "frame_start": 0, "frame_end": len(df) - 1}


def apply_state_sensor_delay_small(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return delay_column(df, ["state"], 3)


def apply_state_sensor_delay_large(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return delay_column(df, ["state"], 10)


def apply_video_signal_delay_small(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return delay_column(df, context["image_columns"], 3)


def apply_video_signal_delay_large(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return delay_column(df, context["image_columns"], 10)


def apply_action_response_delay(df: pd.DataFrame, context: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return delay_column(df, ["actions"], 5)


ISSUES: list[IssueSpec] = [
    IssueSpec("black_screen", "non_temporal", "hard", "vision_illegal_or_single", "image", "基座相机的一段图像替换为纯黑帧。", "vision_single", apply_black_screen),
    IssueSpec("white_screen", "non_temporal", "hard", "vision_illegal_or_single", "image", "基座相机的一段图像替换为纯白帧。", "vision_single", apply_white_screen),
    IssueSpec("flower_screen", "non_temporal", "hard", "vision_illegal_or_single", "image", "基座相机的一段图像替换为块状随机彩色花屏。", "vision_single", apply_flower_screen),
    IssueSpec("invalid_image_bytes", "non_temporal", "hard", "vision_illegal", "image", "图像 bytes 替换为不可解码 payload。", "vision_illegal", apply_invalid_image_bytes),
    IssueSpec("state_nan", "non_temporal", "hard", "state_illegal", "state", "state 某一维在短片段内写入 NaN。", "state_illegal", apply_state_nan),
    IssueSpec("state_inf", "non_temporal", "hard", "state_illegal", "state", "state 某一维在短片段内写入 Inf。", "state_illegal", apply_state_inf),
    IssueSpec("action_nan", "non_temporal", "hard", "state_illegal", "actions", "actions 某一维在短片段内写入 NaN。", "state_illegal", apply_action_nan),
    IssueSpec("state_wrong_dim", "non_temporal", "hard", "state_illegal", "state", "单个 state 行被改成非法向量长度。", "state_illegal", apply_state_wrong_dim),
    IssueSpec("timestamp_non_monotonic", "temporal", "hard", "temporal_illegal", "timestamp", "单个 timestamp 被改成向后倒退。", "temporal_illegal", apply_timestamp_non_monotonic),
    IssueSpec("frame_index_duplicate", "temporal", "hard", "temporal_illegal", "frame_index", "单个 frame_index 与前一帧重复。", "temporal_illegal", apply_frame_index_duplicate),
    IssueSpec("frame_index_skip", "temporal", "hard", "temporal_illegal", "frame_index", "frame_index 从局部位置开始整体跳号，但行仍保留。", "temporal_illegal", apply_frame_index_skip),
    IssueSpec("row_reorder", "temporal", "hard", "temporal_illegal", "row", "交换两个相邻行，破坏时间顺序。", "temporal_illegal", apply_row_reorder),
    IssueSpec("vision_drop_small", "temporal", "soft", "vision_temporal", "image", "小面积丢帧：用上一帧图像掩盖短片段缺帧。", "vision_temporal", apply_vision_drop_small),
    IssueSpec("vision_drop_large", "temporal", "soft", "vision_temporal", "image", "大面积丢帧：用上一帧图像掩盖长片段缺帧。", "vision_temporal", apply_vision_drop_large),
    IssueSpec("vision_jump_small", "temporal", "soft", "vision_temporal", "image", "小面积跳帧：短片段使用未来视觉帧，造成局部画面跳变。", "vision_temporal", apply_vision_jump_small),
    IssueSpec("vision_jump_large", "temporal", "soft", "vision_temporal", "image", "大面积跳帧：长片段使用未来视觉帧，造成持续画面跳变。", "vision_temporal", apply_vision_jump_large),
    IssueSpec("state_sensor_delay_small", "temporal", "soft", "state_vision_temporal", "state", "小延迟传感器问题：state 相对图像延迟 3 帧。", "vision_state_temporal", apply_state_sensor_delay_small),
    IssueSpec("state_sensor_delay_large", "temporal", "soft", "state_vision_temporal", "state", "大延迟传感器问题：state 相对图像延迟 10 帧。", "vision_state_temporal", apply_state_sensor_delay_large),
    IssueSpec("video_signal_delay_small", "temporal", "soft", "vision_state_temporal", "image", "小延迟视频信号问题：三路相机相对 state 延迟 3 帧。", "vision_state_temporal", apply_video_signal_delay_small),
    IssueSpec("video_signal_delay_large", "temporal", "soft", "vision_state_temporal", "image", "大延迟视频信号问题：三路相机相对 state 延迟 10 帧。", "vision_state_temporal", apply_video_signal_delay_large),
    IssueSpec("action_response_delay", "temporal", "soft", "state_temporal", "actions", "动作响应延迟：actions 相对观测 state 延迟 5 帧。", "state_temporal", apply_action_response_delay),
]


def issue_specs_by_id() -> dict[str, IssueSpec]:
    return {issue.issue_id: issue for issue in ISSUES}


def build_context(info: dict[str, Any], source_path: Path) -> dict[str, Any]:
    return {
        "fps": int(info.get("fps", 10)),
        "image_columns": image_columns_from_info(info),
        "source_episode_path": str(source_path.relative_to(ROOT)),
        "source_episode_index": safe_episode_index(source_path),
    }


def copy_reference_metadata(meta_root: Path, output_root: Path, issue_count: int) -> None:
    metadata_dir = output_root / "_reference_meta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for name in ["info.json", "tasks.jsonl", "episodes.jsonl"]:
        source = meta_root / name
        if source.exists():
            shutil.copy2(source, metadata_dir / name)
    (metadata_dir / "README.md").write_text(
        "Reference metadata copied from the source dataset. Generated cases are individual parquet files and intentionally do not form a replacement full dataset.\n"
        f"Single-issue case count: {issue_count}\n",
        encoding="utf-8",
    )


def write_design_doc(output_root: Path, manifest: dict[str, Any]) -> None:
    axis_names = {"temporal": "时序", "non_temporal": "非时序"}
    hardness_names = {"hard": "硬性", "soft": "软性"}
    lines = [
        "# 单一问题消融数据设计",
        "",
        "本目录为每一种问题类型生成一个派生 parquet case。每个 case 只注入一种问题，源数据集不做任何修改。",
        "",
        "## 问题类型矩阵",
        "",
        "| issue_id | 轴向 | 性质 | 目标 | 预期检测族 | 说明 |",
        "|---|---|---|---|---|---|",
    ]
    for case in manifest["cases"]:
        lines.append(
            f"| {case['issue_id']} | {axis_names.get(case['axis'], case['axis'])} | {hardness_names.get(case['hardness'], case['hardness'])} | {case['target']} | "
            f"{case['expected_detector_family']} | {case['description']} |"
        )
    lines.extend(
        [
            "",
            "## 消融试验设计",
            "",
            "1. 原始基线：在未修改参考数据集上运行检测器，记录各维度得分和误报情况。",
            "2. 单问题灵敏度：逐个 case 运行检测器，检查是否命中 `expected_detector_family` 对应检测族。",
            "3. 面积/幅度灵敏度：对比 small/large 的丢帧、跳帧、传感器延迟、视频信号延迟样本，观察分数和置信度是否随问题规模单调变化。",
            "4. 硬性/软性区分：硬性非法样本应主要触发 structural legality；软性样本应更多影响 temporal 或 cross-modal，而不应被误判为解码/结构损坏。",
            "5. 视角泛化：本轮图像局部问题默认注入 base camera，下一轮可把同类问题复制到 left/right wrist camera，验证多视角覆盖。",
            "",
            "本轮刻意不生成混合问题，方便后续把检出/漏检准确归因到单一检测能力。",
        ]
    )
    (output_root / "ablation_design.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_cases(output_root: Path, selected_issue_ids: list[str], clean: bool, seed: int) -> dict[str, Any]:
    meta_root = find_meta_root()
    info = json.loads((meta_root / "info.json").read_text(encoding="utf-8"))
    tasks = load_jsonl(meta_root / "tasks.jsonl")
    episodes = load_jsonl(meta_root / "episodes.jsonl")
    source_path = choose_source_episode(find_episode_files(meta_root))
    source_df, source_schema = load_episode(source_path)
    context = build_context(info, source_path)
    specs = issue_specs_by_id()

    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    for offset, issue_id in enumerate(selected_issue_ids):
        spec = specs[issue_id]
        rng = np.random.default_rng(seed + offset)
        df = source_df.copy(deep=True)
        evidence = spec.apply(df, context, rng)
        case_dir = output_root / spec.issue_id
        case_path = case_dir / f"episode_{context['source_episode_index']:06d}_{spec.issue_id}.parquet"
        write_episode(df, source_schema, case_path)
        case = {
            "issue_id": spec.issue_id,
            "axis": spec.axis,
            "hardness": spec.hardness,
            "category": spec.category,
            "target": spec.target,
            "description": spec.description,
            "expected_detector_family": spec.expected_detector_family,
            "source_episode_index": context["source_episode_index"],
            "source_episode_path": context["source_episode_path"],
            "case_path": str(case_path.relative_to(ROOT)),
            "rows": len(df),
            "evidence": evidence,
        }
        cases.append(case)
        (case_dir / "case_manifest.json").write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "generator": str(Path(__file__).relative_to(ROOT)),
        "seed": seed,
        "source_dataset": {
            "codebase_version": info.get("codebase_version"),
            "robot_type": info.get("robot_type"),
            "fps": info.get("fps"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "tasks": tasks,
            "episode_meta_count": len(episodes),
        },
        "policy": {
            "granularity": "one parquet case per issue type",
            "mixed_issues": False,
            "original_dataset_modified": False,
        },
        "cases": cases,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    copy_reference_metadata(meta_root, output_root, len(cases))
    write_design_doc(output_root, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate single-issue ablation parquet cases from the local LeRobot dataset.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--issues", nargs="*", default=[issue.issue_id for issue in ISSUES], choices=[issue.issue_id for issue in ISSUES])
    parser.add_argument("--clean", action="store_true", help="Remove the output root before generation.")
    parser.add_argument("--list-issues", action="store_true", help="Print available issue ids and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_issues:
        for issue in ISSUES:
            print(f"{issue.issue_id}\t{issue.axis}\t{issue.hardness}\t{issue.target}\t{issue.description}")
        return
    manifest = generate_cases(args.output_root, args.issues, args.clean, args.seed)
    print(f"generated_cases: {len(manifest['cases'])}")
    print(f"output_root: {args.output_root}")
    print(f"manifest: {args.output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
