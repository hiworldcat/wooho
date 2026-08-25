# V2 Multimodal Robot Data Quality Report

- dataset version: v2.1
- robot type: panda
- episodes: 20
- frames: 4142
- FPS: 10
- dataset quality score: **98.62/100**

## Detection Framework

The v2 pipeline combines hard legality checks, single-view image quality, low-dimensional state/action temporal checks, and state-only cross-modal consistency checks.
Direct Vision-Vision findings are disabled. Multi-view visual agreement is checked only after a state-supported overlap gate; actions are excluded from consistency checks because commands can naturally lead or lag observed state and vision.

| dimension | max_points | current_mean |
|---|---:|---:|
| structural | 25 | 25.0 |
| vision_single | 20 | 19.72 |
| vision_vision | 10 | 10.0 |
| state | 15 | 15.0 |
| temporal | 15 | 14.24 |
| cross_modal | 15 | 14.66 |

## Finding Summary

- merged findings: 0
- critical: 0
- high confidence: 0
- suspicious: 0
- out of distribution: 0

## Top 10 Issue Types

| issue_type | count |
|---|---:|

## Output Files

- `outputs/v2/diagnostics/findings_v2.json`: normalized merged findings
- `outputs/v2/diagnostics/reference_baselines_v2.json`: reference baselines and thresholds
- `outputs/v2/diagnostics/problem_standards_v2.json`: problem definitions and decision rules
- `outputs/v2/reports/episode_scores_v2.csv`: episode-level score table
- `outputs/v2/reports/dataset_quality_report_v2.json`: machine-readable complete report
- `outputs/v2/reports/dataset_quality_report_v2.md`: human-readable summary report
- `outputs/v2/reports/episode_quality_report_v2_detail.md`: episode-level detail report
