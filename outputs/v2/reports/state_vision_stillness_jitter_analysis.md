# V2 Reference-Set Calibration

## Principle

The official reference set is normal calibration data, not a pool to mine for hard anomalies. Hard findings are emitted only for structural errors or values outside the calibrated reference envelope. Normal in-envelope tail behavior contributes small soft penalties so a clean reference set is close to full score, not exactly full score.

## Adopted Rule

- Actions are excluded from state-vision consistency checks; robot `state` is used for cross-modal geometry and timing.
- Freeze grace window: runs up to 10 motion steps are accepted.
- Freeze severity is piecewise: 10-50 frames rises slowly, 50-100 frames rises faster, 100+ frames is severe.
- Hard fast/jitter findings must exceed the reference envelope, not merely the p99 tail inside the reference set.
- Cross-modal hard findings must fall outside the reference-set observed correlation or lag range.
- In-envelope tail behavior is scored as `soft_penalties` with small per-dimension caps.

## Current Reference Self-Check

| metric | value |
|---|---:|
| episodes_checked | 20 |
| merged_findings | 0 |
| dataset_quality_score | 98.62 |
| episode_score_min | 97.78 |
| episode_score_max | 99.18 |
| soft_penalty_mean | 1.381 |
| soft_penalty_max | 2.219 |

## Soft Penalty Means

| dimension | mean penalty |
|---|---:|
| cross_modal | 0.337 |
| temporal | 0.761 |
| vision_single | 0.284 |
| vision_vision | 0.0 |

## Episode Soft Scores

| episode | total | soft_penalty | soft_penalties |
|---:|---:|---:|---|
| 0 | 99.01 | 0.985 | {"vision_single": 0.229, "temporal": 0.556, "cross_modal": 0.2} |
| 1 | 98.82 | 1.187 | {"vision_single": 0.533, "temporal": 0.654, "cross_modal": 0.0} |
| 2 | 98.93 | 1.067 | {"vision_single": 0.09, "temporal": 0.777, "cross_modal": 0.2} |
| 3 | 97.99 | 2.009 | {"vision_single": 0.565, "temporal": 0.844, "cross_modal": 0.6} |
| 4 | 98.34 | 1.654 | {"vision_single": 0.36, "temporal": 0.827, "cross_modal": 0.467} |
| 5 | 98.83 | 1.169 | {"vision_single": 0.088, "temporal": 0.681, "cross_modal": 0.4} |
| 6 | 98.56 | 1.437 | {"vision_single": 0.499, "temporal": 0.938, "cross_modal": 0.0} |
| 7 | 98.42 | 1.58 | {"vision_single": 0.197, "temporal": 0.983, "cross_modal": 0.4} |
| 8 | 98.46 | 1.54 | {"vision_single": 0.383, "temporal": 0.69, "cross_modal": 0.467} |
| 9 | 98.61 | 1.395 | {"vision_single": 0.261, "temporal": 0.534, "cross_modal": 0.6} |
| 10 | 98.58 | 1.42 | {"vision_single": 0.215, "temporal": 1.205, "cross_modal": 0.0} |
| 11 | 98.41 | 1.588 | {"vision_single": 0.447, "temporal": 0.741, "cross_modal": 0.4} |
| 12 | 97.78 | 2.219 | {"vision_single": 0.344, "temporal": 1.275, "cross_modal": 0.6} |
| 13 | 98.48 | 1.511 | {"vision_single": 0.126, "temporal": 0.785, "cross_modal": 0.6} |
| 14 | 98.76 | 1.248 | {"vision_single": 0.324, "temporal": 0.724, "cross_modal": 0.2} |
| 15 | 99.18 | 0.815 | {"vision_single": 0.207, "temporal": 0.608, "cross_modal": 0.0} |
| 16 | 98.66 | 1.343 | {"vision_single": 0.079, "temporal": 0.664, "cross_modal": 0.6} |
| 17 | 98.98 | 1.018 | {"vision_single": 0.378, "temporal": 0.64, "cross_modal": 0.0} |
| 18 | 98.73 | 1.272 | {"vision_single": 0.153, "temporal": 0.519, "cross_modal": 0.6} |
| 19 | 98.83 | 1.168 | {"vision_single": 0.198, "temporal": 0.57, "cross_modal": 0.4} |

## Outputs

- `outputs/v2/diagnostics/findings_v2.json`
- `outputs/v2/reports/dataset_quality_report_v2.json`
- `outputs/v2/reports/episode_quality_report_v2_detail.md`
