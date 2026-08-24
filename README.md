# Woohu Multimodal Data Quality Detection

This repository contains scripts and generated reports for a LeRobot v2.1
multimodal robot data quality detection workflow.

## Contents

- `scripts/`: data inspection, validation, and v2 quality scoring pipeline.
- `outputs/v2/reports/`: human-readable quality reports and episode scores.
- `outputs/v2/diagnostics/`: machine-readable findings, baselines, and standards.
- `赛题.txt`: competition/task description.

## Notes

The raw competition dataset is intentionally excluded from Git. The local
`初赛数据/` directory is ignored because it is large and should not be uploaded to
a normal GitHub repository.

The current v2 pipeline avoids direct Vision-Vision matching. It first estimates
camera pose overlap from robot state and only then uses visual evidence as a weak
cross-check under the Vision-State-Vision category.

## Run

Use the bundled or local Python environment with `numpy`, `Pillow`, and
`pyarrow` installed:

```powershell
python scripts/v2_quality_pipeline.py
```
