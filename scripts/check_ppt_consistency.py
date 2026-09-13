from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree


FORBIDDEN_TEXT = (
    "待回填",
    "待正式数据复测",
    "参考集版式示例",
    "100/100",
    "29.7/s",
    "0.7047",
)
DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Block final packaging when the PPT still contains placeholders or stale metrics."
    )
    parser.add_argument("--pptx", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/target/v2"))
    return parser.parse_args()


def ppt_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        slide_names = sorted(
            name
            for name in archive.namelist()
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        )
        chunks: list[str] = []
        for name in slide_names:
            root = ElementTree.fromstring(archive.read(name))
            chunks.extend(node.text or "" for node in root.iter(f"{{{DRAWING_NS}}}t"))
    return "\n".join(chunks)


def main() -> None:
    args = parse_args()
    pptx = args.pptx.resolve()
    report_path = args.output_root.resolve() / "reports/dataset_quality_report_v2.json"
    if not pptx.is_file():
        raise SystemExit(f"PPTX not found: {pptx}")
    if not report_path.is_file():
        raise SystemExit(f"frozen report not found: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    text = ppt_text(pptx)
    errors: list[str] = []
    for marker in FORBIDDEN_TEXT:
        if marker in text:
            errors.append(f"PPT contains unfinished or stale text: {marker}")

    episode_count = int(report.get("dataset", {}).get("episodes", -1))
    score = report.get("dataset", {}).get("dataset_quality_score")
    finding_count = int(report.get("summary", {}).get("finding_count", -1))
    if episode_count != 20:
        errors.append(f"frozen report must contain 20 episodes, got {episode_count}")
    if str(score) not in text:
        errors.append(f"PPT does not contain frozen dataset score: {score}")
    if str(finding_count) not in text:
        errors.append(f"PPT does not contain frozen finding count: {finding_count}")

    print("PPT CONSISTENCY: " + ("FAIL" if errors else "PASS"))
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
