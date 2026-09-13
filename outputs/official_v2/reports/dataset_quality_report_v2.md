# V2 Multimodal Robot Data Quality Report

- dataset role: negative_control
- reference dataset: 鍙傝€冮泦
- target dataset: 鍙傝€冮泦
- output label: official_v2
- dataset version: v2.1
- robot type: panda
- episodes: 20
- frames: 4142
- FPS: 10
- dataset quality score: **92.76/100**
- official scoring: `official_70_20_10_v1` = detection quality 69.27/70 + data value 16.75/20 + engineering reliability 6.74/10
- legacy six-dimension mean: 98.95/100 (breakdown only)

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
| cross_modal | 15 | 15.0 |

## Module Status

Status enum: `ok` completed, `warning` core completed with degraded or unavailable submodules, `unavailable` core or submodule could not run, `fail` module execution failed.

| module | ok | warning | unavailable | fail |
|---|---:|---:|---:|---:|
| geometry | 0 | 20 | 0 | 0 |
| arms.left | 20 | 0 | 0 | 0 |
| arms.right | 20 | 0 | 0 | 0 |
| bimanual | 0 | 0 | 20 | 0 |
| state_vision.left | 20 | 0 | 0 | 0 |
| state_vision.right | 20 | 0 | 0 | 0 |

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

- `diagnostics/findings_v2.json`: normalized merged findings
- `diagnostics/reference_baselines_v2.json`: reference baselines and thresholds
- `diagnostics/problem_standards_v2.json`: problem definitions and decision rules
- `diagnostics/run_manifest_v2.json`: run role and version manifest
- `reports/episode_scores_v2.csv`: episode-level score table
- `reports/dataset_quality_report_v2.json`: machine-readable complete report
- `reports/dataset_quality_report_v2.md`: English summary report
- `reports/dataset_quality_report_v2_zh.md`: Chinese summary report
- `reports/episode_quality_report_v2_detail.md`: episode-level detail report
