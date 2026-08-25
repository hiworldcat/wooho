# Visual And Movement Inconsistency Analysis

## Scope Note

The annotated clips were generated from the v2 findings set that was present before the pipeline was rerun in this session. That set had 53 merged findings and matched `outputs/v2/reports/static_jitter_analysis_summary.json`.

After rerunning the current working copy of `scripts/v2_quality_pipeline.py`, the pipeline emitted 0 findings. The reason is that the current code has extra tolerance guards, including reference-envelope checks, `p99` visual cutoff for cross-modal static checks, and an 11-frame minimum for cross-modal static runs. These guards are stricter than the already generated report set, so the diagnostic outputs are currently not internally consistent.

## Error Families In The Report Set Used For Clips

| issue_type | count | how it happened |
|---|---:|---|
| `low_dim_freeze_run` | 17 | `state` or `actions` delta stayed at or below the low-motion threshold for longer than the freeze grace window. |
| `visual_moves_state_static` | 10 | Combined visual motion was high while the low-dimensional `actions` delta was static. These are the clips generated here. |
| `state_gated_view_pair_motion_inconsistency` | 9 | State-estimated camera frustums overlapped, but the visual motion correlation between the gated camera pair was weak or negative. |
| `low_dim_jitter_or_spike` | 6 | State/action acceleration exceeded the robust acceleration threshold. Single-frame spikes were kept only when severity was at least 95. |
| `visual_fast_jump` | 5 | Visual frame-to-frame motion exceeded the high-motion threshold for that camera/task distribution. |
| `low_cross_modal_correlation` | 3 | Best-lag correlation between visual motion and state/action motion was lower than the calibrated reference distribution. |
| `visual_high_frequency_jitter` | 2 | Visual motion acceleration had clustered spikes above the robust visual-jitter threshold. |
| `cross_modal_lag_shift` | 1 | Best visual-state lag deviated from the reference lag range. |

## Visual Tolerance Rules

- Freeze tolerance: up to 10 motion steps are accepted without reporting.
- Near-zero visual motion threshold: `max(0.2, p01(image_motion) * 0.5)`.
- Fast visual jump threshold in the clip-generating report set: `max(p99(image_motion) * 1.5, median + 8*MAD)`.
- Visual jitter threshold: `max(p99(image_accel) * 2.0, median + 10*MAD)`.
- Geometry gate for multi-view checks: only verify a camera pair when state-estimated overlap score is at least `0.10`.
- Multi-view correlation scoring: lower-tail robust outlier, weak z `3.0`, strong z `6.0`; negative correlation is forced to at least severity 45, correlation below 0.05 to at least severity 35.
- Static low-dimensional tolerance in the generated clips: `actions_delta <= 1e-08`.

## Generated Clips

All clips are 10 FPS and cover 3 seconds before through 3 seconds after the problematic segment. Each frame shows the three camera views, current `state_delta`, `actions_delta`, combined `vision_motion`, the static tolerance, the visual high cutoff, and a timeline plot with the issue span highlighted.

| finding | episode | issue frames | clip frames | visual high cutoff | max visual in issue | actions median in issue | AVI |
|---|---:|---:|---:|---:|---:|---:|---|
| `V2F-000004` | 2 | 127-130 | 97-160 | 14.415 | 19.145 | 0.0 | `outputs/v2/vision_state_static_clips/V2F-000004_episode-02_frames-0127-0130.avi` |
| `V2F-000015` | 5 | 164-167 | 134-197 | 11.399 | 20.833 | 0.0 | `outputs/v2/vision_state_static_clips/V2F-000015_episode-05_frames-0164-0167.avi` |
| `V2F-000020` | 7 | 163-165 | 133-195 | 12.190 | 20.204 | 0.0 | `outputs/v2/vision_state_static_clips/V2F-000020_episode-07_frames-0163-0165.avi` |
| `V2F-000023` | 8 | 107-110 | 77-140 | 14.359 | 21.483 | 0.0 | `outputs/v2/vision_state_static_clips/V2F-000023_episode-08_frames-0107-0110.avi` |
| `V2F-000025` | 9 | 130-133 | 100-163 | 13.719 | 17.075 | 0.0 | `outputs/v2/vision_state_static_clips/V2F-000025_episode-09_frames-0130-0133.avi` |
| `V2F-000033` | 11 | 112-114 | 82-144 | 14.274 | 18.042 | 0.0 | `outputs/v2/vision_state_static_clips/V2F-000033_episode-11_frames-0112-0114.avi` |
| `V2F-000039` | 13 | 119-122 | 89-152 | 12.994 | 17.333 | 0.0 | `outputs/v2/vision_state_static_clips/V2F-000039_episode-13_frames-0119-0122.avi` |
| `V2F-000042` | 14 | 164-167 | 134-197 | 12.870 | 17.298 | 0.0 | `outputs/v2/vision_state_static_clips/V2F-000042_episode-14_frames-0164-0167.avi` |
| `V2F-000053` | 18 | 128-131 | 98-161 | 13.293 | 18.945 | 0.0 | `outputs/v2/vision_state_static_clips/V2F-000053_episode-18_frames-0128-0131.avi` |
| `V2F-000055` | 19 | 92-95 | 62-125 | 13.170 | 18.664 | 0.0 | `outputs/v2/vision_state_static_clips/V2F-000055_episode-19_frames-0092-0095.avi` |
