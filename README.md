# Woohu Multimodal Data Quality Detection

This repository contains scripts and generated reports for a LeRobot v2.1
multimodal robot data quality detection workflow.

## Contents

- `scripts/`: data inspection, validation, and v2 quality scoring pipeline.
- `scripts/geometry_constraints.py`: standalone P0 spatial geometry detector for Rotation 6D legality, per-arm SE(3), bimanual gates, and weak State-Vision wrist response checks.
- `outputs/v2/reports/`: human-readable quality reports and episode scores.
- `outputs/v2/diagnostics/`: machine-readable findings, baselines, and standards.
- `outputs/legacy/`: legacy report outputs kept only for compatibility.
- `赛题.txt`: competition/task description.

## Notes

The raw competition dataset is intentionally excluded from Git. The local
`初赛数据/` directory is ignored because it is large and should not be uploaded to
a normal GitHub repository.

The current v2 pipeline avoids direct Vision-Vision matching without calibration.
Spatial geometry is handled by a standalone P0 module: unknown calibration-dependent
relations are downgraded, while Rotation 6D, per-arm SE(3), bimanual common-frame
checks, and weak wrist State-Vision responses are reported separately.

## Run

Install the bundled or local Python environment from `requirements.txt`. The v2 pipeline requires explicit reference and target roots:

```powershell
python run_v2_pipeline.py --reference-root "C:\path\to\reference" --target-root "C:\path\to\target"
```

Quick geometry smoke test:

```powershell
python scripts/test_geometry_constraints_smoke.py
```




