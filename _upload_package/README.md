# Woohu Multimodal Data Quality Detection

This package contains the runnable V2 pipeline and generated report formats for
a LeRobot v2.1 multimodal robot data quality detection workflow.

## Contents

- `run_v2_pipeline.py`: formal command-line entry point.
- `requirements.txt`: Python dependencies.
- `scripts/`: v2 quality scoring, geometry checks, and readiness validation.
- `scripts/preflight_submission.py`: dependency, dataset, memory, and disk checks without loading frames.
- `scripts/export_ppt_metrics.py`: frozen-result sheet for PPT updates.
- `scripts/check_ppt_consistency.py`: blocks stale figures and unfinished placeholders.
- `outputs/target/v2/reports/`: destination for formal human-readable reports.
- `outputs/target/v2/diagnostics/`: destination for formal machine-readable findings.
- `赛题.txt`: competition/task description.

## Notes

The raw competition dataset is intentionally excluded from Git. The local
`初赛数据/` directory is ignored because it is large and should not be uploaded to
a normal GitHub repository.

The current v2 pipeline avoids direct Vision-Vision matching. It first estimates
camera pose overlap from robot state and only then uses visual evidence as a weak
cross-check under the Vision-State-Vision category.

The checked-in files under `outputs/` are a historical snapshot and are not a
frozen official target result. A formal run must write to `outputs/target/v2/`.

## Run

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run with separate reference and target datasets:

```powershell
python run_v2_pipeline.py `
  --reference-root "C:\path\to\reference" `
  --target-root "C:\path\to\target" `
  --output-root "outputs\target\v2" `
  --geometry-config "scripts\geometry_config.json"
```

Validate the frozen output before using its numbers in the PPT:

```powershell
python scripts/check_submission_readiness.py `
  --output-root "outputs\target\v2" `
  --expected-episodes 20
```

Run the frozen pipeline and readiness check together:

```powershell
python scripts/run_final_submission.py `
  --reference-root "C:\path\to\reference" `
  --target-root "C:\path\to\target"
```

After the final PPTX, PDF, and application form are ready, use
`scripts/prepare_submission_package.py` to create a clean ZIP with a manifest,
SHA256 checksums, and an upload-evidence checklist.
