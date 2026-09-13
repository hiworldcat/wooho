from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="wooho-submission-test-") as directory:
        temp = Path(directory)
        output = temp / "outputs" / "target" / "v2"
        diagnostics = output / "diagnostics"
        reports = output / "reports"
        diagnostics.mkdir(parents=True)
        reports.mkdir(parents=True)

        run = {
            "dataset_role": "target",
            "reference_dataset": "reference",
            "target_dataset": "official-20",
            "same_dataset": False,
            "output_label": "v2",
            "reference_episode_count": 20,
            "target_episode_count": 20,
            "scoring_version": "official_70_20_10_v1",
        }
        finding = {
            "episode_index": 0,
            "issue_type": "blur",
            "illegal": False,
            "severity_score": 60,
            "quality_penalty": 1,
        }
        report = {
            "run": run,
            "scoring": {"version": "official_70_20_10_v1"},
            "dataset": {"episodes": 20, "dataset_quality_score": 88.5},
            "summary": {
                "finding_count": 1,
                "by_issue": {"blur": 1},
                "geometry_status_counts": {"ok": 20},
                "phase_status_counts": {"ok": 20},
            },
            "episodes": [{"episode_index": index} for index in range(20)],
        }
        write_json(diagnostics / "run_manifest_v2.json", run)
        write_json(diagnostics / "findings_v2.json", [finding])
        write_json(diagnostics / "geometry_constraints_v2.json", [])
        write_json(diagnostics / "problem_standards_v2.json", {})
        write_json(diagnostics / "reference_baselines_v2.json", {})
        write_json(reports / "dataset_quality_report_v2.json", report)
        for name in (
            "dataset_quality_report_v2.md",
            "dataset_quality_report_v2_zh.md",
            "episode_quality_report_v2_detail.md",
        ):
            (reports / name).write_text("test\n", encoding="utf-8")
        with (reports / "episode_scores_v2.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["episode_index", "file"])
            writer.writeheader()
            for index in range(20):
                writer.writerow(
                    {"episode_index": index, "file": f"data/chunk-000/episode_{index:06d}.parquet"}
                )

        pptx = temp / "final.pptx"
        slide_xml = """<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
 <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>正式结果 20 88.5 1</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
</p:sld>"""
        with zipfile.ZipFile(pptx, "w") as archive:
            archive.writestr("ppt/slides/slide1.xml", slide_xml)
        pdf = temp / "final.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
        application_form = temp / "application-form.pdf"
        application_form.write_bytes(b"%PDF-1.4\n%%EOF\n")

        metrics_destination = temp / "ppt-metrics"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "export_ppt_metrics.py"),
                "--output-root",
                str(output),
                "--destination",
                str(metrics_destination),
            ],
            cwd=ROOT,
            check=True,
        )
        metrics = json.loads(
            (metrics_destination / "ppt_metrics.json").read_text(encoding="utf-8")
        )
        if metrics["slide_11_overview"]["completed_episodes"] != 20:
            raise AssertionError("PPT metrics episode count is incorrect")

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "prepare_submission_package.py"),
                "--pptx",
                str(pptx),
                "--pdf",
                str(pdf),
                "--application-form",
                str(application_form),
                "--output-root",
                str(output),
                "--team-name",
                "test-team",
                "--project-name",
                "test-project",
                "--destination",
                str(temp / "dist"),
            ],
            cwd=ROOT,
            check=True,
        )
        archive = temp / "dist" / "芜湖杯-test-team-test-project.zip"
        if not archive.is_file():
            raise AssertionError("submission ZIP was not created")
        with zipfile.ZipFile(archive) as handle:
            if handle.testzip() is not None:
                raise AssertionError("submission ZIP is corrupt")
            names = handle.namelist()
            if not any(name.endswith("CHECKSUMS.sha256") for name in names):
                raise AssertionError("checksums file is missing")
            if not any(name.endswith("03_源代码/HANDOFF.md") for name in names):
                raise AssertionError("handoff guide is missing")
            if any("__pycache__" in name or name.endswith(".parquet") for name in names):
                raise AssertionError("excluded data leaked into the ZIP")
        print("submission workflow smoke test passed")


if __name__ == "__main__":
    main()
