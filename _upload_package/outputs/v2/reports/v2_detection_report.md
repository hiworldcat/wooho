# V2 Detection Report

This report reflects the current implementation in `scripts/v2_quality_pipeline.py`
and the regenerated outputs under `outputs/v2`.

## Current Run

- episodes checked: 20
- merged findings: 80
- dataset quality score: 78.46 / 100
- direct `vision_vision` findings: 0
- geometry-gated multi-view findings: 9

## Key Change

The pipeline no longer performs direct Vision-Vision validation by comparing
camera motion correlations alone.

Multi-camera consistency is now treated as a Vision-State-Vision check:

1. Use known camera semantics from the feature names:
   `image`, `left_wrist_image`, `right_wrist_image`.
2. Use the 20-D state/action layout from the dataset:
   - left end-effector pose: `state[0:3]` position + `state[3:9]` rotation 6D;
   - right end-effector pose: `state[10:13]` position + `state[13:19]` rotation 6D.
3. Reconstruct approximate camera poses:
   - wrist cameras are attached to the corresponding end-effector pose;
   - the base camera is estimated as a fixed look-at camera around the shared workspace;
   - wrist camera optical axes are inferred from state/workspace consistency because
     hand-eye calibration is not provided.
4. Estimate camera frustum overlap against shared workspace targets.
5. Only after the geometric overlap gate is enabled, use visual motion correlation
   as a weak verification signal.

This avoids treating non-overlapping cameras as if they should always agree.

## Active Problem Families

| problem family | implementation role |
|---|---|
| `vision_illegal` | hard image schema/decode/shape checks |
| `state_illegal` | hard low-dimensional schema/finite/shape checks |
| `vision_single` | per-camera single-frame quality checks |
| `vision_vision` | reserved for hard calibration metadata; no direct findings in v2 |
| `vision_state_vision` | geometry-gated multi-view checks |
| `state_vision_state` | state values or transitions inconsistent with visual context |
| `temporal_illegal` | hard timestamp/index/length checks |
| `vision_temporal` | frozen, jumpy, or jittery image sequences |
| `state_temporal` | frozen, fast, or jittery low-dimensional sequences |
| `vision_state_temporal` | lag/correlation consistency between vision and state |

## Current Issue Counts

| issue_type | count | category |
|---|---:|---|
| `low_dim_freeze_run` | 31 | `1.2.2.B` |
| `low_dim_jitter_or_spike` | 14 | `1.2.2.B` |
| `visual_moves_state_static` | 10 | `1.1.2.D` |
| `state_gated_view_pair_motion_inconsistency` | 9 | `1.1.2.C` |
| `visual_fast_jump` | 5 | `1.2.2.A` |
| `visual_high_frequency_jitter` | 4 | `1.2.2.A` |
| `low_motion_freeze_run` | 3 | `1.2.2.A` |
| `low_cross_modal_correlation` | 3 | `1.2.2.C` |
| `cross_modal_lag_shift` | 1 | `1.2.2.C` |

## Evidence Format

Geometry-gated multi-view findings include a `state_gate` object in
`outputs/v2/diagnostics/findings_v2.json`. The evidence records:

- which state column was used (`state` or `actions`);
- which camera maps to which arm;
- estimated workspace center and extent;
- approximate camera position and optical axis for an example frame;
- frustum overlap score;
- the explicit note that real hand-eye calibration was not provided.

The visual correlation appears only inside that gated evidence path.
