# V2 Multimodal Robot Data Quality Report

- dataset version: v2.1
- robot type: panda
- episodes: 20
- frames: 4142
- FPS: 10
- dataset quality score: **78.46/100**

## Detection Framework

The v2 pipeline combines hard legality checks, single-view image quality, low-dimensional state checks, temporal checks, and cross-modal consistency checks.
Direct Vision-Vision findings are disabled. Multi-view visual agreement is checked only after a state-supported overlap gate, and those findings are categorized as Vision-State-Vision.

| dimension | max_points | current_mean |
|---|---:|---:|
| structural | 25 | 25.0 |
| vision_single | 20 | 20.0 |
| vision_vision | 10 | 10.0 |
| state | 15 | 15.0 |
| temporal | 15 | 0.75 |
| cross_modal | 15 | 7.71 |

## Finding Summary

- merged findings: 80
- critical: 0
- high confidence: 63
- suspicious: 4
- out of distribution: 13

## Top 10 Issue Types

| issue_type | count |
|---|---:|
| low_dim_freeze_run | 31 |
| low_dim_jitter_or_spike | 14 |
| visual_moves_state_static | 10 |
| state_gated_view_pair_motion_inconsistency | 9 |
| visual_fast_jump | 5 |
| visual_high_frequency_jitter | 4 |
| low_motion_freeze_run | 3 |
| low_cross_modal_correlation | 3 |
| cross_modal_lag_shift | 1 |

## Output Files

- `outputs/v2/diagnostics/findings_v2.json`: normalized merged findings
- `outputs/v2/diagnostics/reference_baselines_v2.json`: reference baselines and thresholds
- `outputs/v2/diagnostics/problem_standards_v2.json`: problem definitions and decision rules
- `outputs/v2/reports/episode_scores_v2.csv`: episode-level score table
- `outputs/v2/reports/dataset_quality_report_v2.json`: machine-readable complete report
- `outputs/v2/reports/dataset_quality_report_v2.md`: human-readable summary report
- `outputs/v2/reports/episode_quality_report_v2_detail.md`: episode-level detail report
