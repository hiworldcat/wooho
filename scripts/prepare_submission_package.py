from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "README.md",
    "requirements.txt",
    "run_v2_pipeline.py",
    "scripts/v2_quality_pipeline.py",
    "scripts/geometry_constraints.py",
    "scripts/geometry_config.json",
    "scripts/check_submission_readiness.py",
    "scripts/check_ppt_consistency.py",
    "scripts/export_ppt_metrics.py",
    "scripts/preflight_submission.py",
    "scripts/run_final_submission.py",
    "scripts/run_pipeline_v2.ps1",
)
RESULT_DIRS = ("diagnostics", "reports", "governance", "visuals")


def safe_name(value: str) -> str:
    forbidden = '<>:"/\\|?*'
    cleaned = "".join("_" if char in forbidden else char for char in value).strip()
    if not cleaned or cleaned in {".", ".."}:
        raise argparse.ArgumentTypeError("name must contain visible safe characters")
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a clean, checksummed final-submission ZIP."
    )
    parser.add_argument("--pptx", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--application-form", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/target/v2"))
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument("--contest-name", type=safe_name, default="芜湖杯")
    parser.add_argument("--team-name", type=safe_name, required=True)
    parser.add_argument("--project-name", type=safe_name, required=True)
    parser.add_argument("--destination", type=Path, default=Path("dist"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_required_file(source: Path, destination: Path, label: str) -> None:
    if not source.is_file():
        raise SystemExit(f"missing {label}: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()

    readiness_command = [
        sys.executable,
        str(ROOT / "scripts" / "check_submission_readiness.py"),
        "--output-root",
        str(output_root),
        "--expected-episodes",
        str(args.expected_episodes),
    ]
    readiness = subprocess.run(readiness_command, cwd=ROOT)
    if readiness.returncode != 0:
        raise SystemExit("submission package blocked: target result is not ready")

    ppt_check_command = [
        sys.executable,
        str(ROOT / "scripts" / "check_ppt_consistency.py"),
        "--pptx",
        str(args.pptx.resolve()),
        "--output-root",
        str(output_root),
    ]
    ppt_check = subprocess.run(ppt_check_command, cwd=ROOT)
    if ppt_check.returncode != 0:
        raise SystemExit("submission package blocked: PPT does not match frozen results")

    package_name = f"{args.contest_name}-{args.team_name}-{args.project_name}"
    destination = args.destination.resolve()
    work_root = destination / f".{package_name}.work"
    package_root = work_root / package_name
    zip_path = destination / f"{package_name}.zip"

    if work_root.exists():
        shutil.rmtree(work_root)
    destination.mkdir(parents=True, exist_ok=True)
    package_root.mkdir(parents=True)

    copy_required_file(
        args.application_form.resolve(),
        package_root / "01_参赛表" / args.application_form.name,
        "application form",
    )
    copy_required_file(
        args.pptx.resolve(), package_root / "02_评审材料" / args.pptx.name, "PPTX"
    )
    copy_required_file(
        args.pdf.resolve(), package_root / "02_评审材料" / args.pdf.name, "PDF"
    )

    for relative in SOURCE_FILES:
        copy_required_file(ROOT / relative, package_root / "03_源代码" / relative, relative)

    for directory in RESULT_DIRS:
        source_dir = output_root / directory
        if source_dir.is_dir():
            shutil.copytree(source_dir, package_root / "04_正式运行结果" / directory)

    files = sorted(path for path in package_root.rglob("*") if path.is_file())
    manifest = {
        "package_name": package_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_episodes": args.expected_episodes,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "files": [
            {
                "path": path.relative_to(package_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    (package_root / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    evidence_text = """# 上传留证清单

- [ ] 压缩包文件名、大小与 SHA256 已记录
- [ ] 上传页面显示提交成功
- [ ] 提交时间与队伍信息已截图
- [ ] 平台中的文件名与本地最终文件一致
- [ ] 已重新下载并核对 SHA256
- [ ] GitHub 最终提交哈希已记录
"""
    (package_root / "上传留证清单.md").write_text(evidence_text, encoding="utf-8")

    checksum_files = sorted(path for path in package_root.rglob("*") if path.is_file())
    checksums = "\n".join(
        f"{sha256(path)}  {path.relative_to(package_root).as_posix()}"
        for path in checksum_files
    )
    (package_root / "CHECKSUMS.sha256").write_text(checksums + "\n", encoding="utf-8")

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(work_root))

    with zipfile.ZipFile(zip_path) as archive:
        bad_file = archive.testzip()
        if bad_file:
            raise SystemExit(f"ZIP integrity check failed: {bad_file}")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    if size_mb >= 200:
        raise SystemExit(f"ZIP is too large: {size_mb:.1f} MB")

    print("SUBMISSION PACKAGE: PASS")
    print(f"zip: {zip_path}")
    print(f"size_mb: {size_mb:.2f}")
    print(f"sha256: {sha256(zip_path)}")


if __name__ == "__main__":
    main()
