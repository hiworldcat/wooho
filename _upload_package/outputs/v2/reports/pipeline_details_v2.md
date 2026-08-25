# V2 Pipeline Details

## Pipeline Order

1. Load LeRobot v2.1 metadata and episode parquet files.
2. Infer image views from `info.json`.
3. Build reference baselines for image quality, image motion, state motion, and
   vision-state timing.
4. Inspect each episode:
   - hard temporal/index/schema checks;
   - per-image decode, shape, and quality checks;
   - low-dimensional state/action legality and temporal checks;
   - geometry-gated Vision-State-Vision multi-camera verification;
   - state-only vision-state temporal and motion consistency checks.
5. Merge adjacent findings with the same episode/category/type/modality/view/column.
6. Score each episode by subtracting capped penalties from dimension budgets.

## Geometry-Gated Multi-View Logic

Direct Vision-Vision comparison is disabled for v2 because different cameras do
not necessarily share enough scene overlap. The current implementation first
reconstructs approximate camera poses from the robot state, then estimates
whether two camera frustums should overlap.

Known data parameters:

- `image`: base/global camera
- `left_wrist_image`: left wrist camera
- `right_wrist_image`: right wrist camera
- `state[0:3]`: left end-effector position
- `state[3:9]`: left end-effector rotation 6D
- `state[10:13]`: right end-effector position
- `state[13:19]`: right end-effector rotation 6D
- `actions` has the same 20-D layout, but is not used for consistency checks
  because action commands can naturally lead or lag observed state and vision

Approximate camera model:

| camera | pose source |
|---|---|
| `image` | fixed look-at pose around the observed shared workspace |
| `left_wrist_image` | left end-effector pose, optical axis inferred as local `-Y` |
| `right_wrist_image` | right end-effector pose, optical axis inferred as local `+Y` |

The local wrist-camera optical axes are assumptions because real hand-eye
calibration is not included in the dataset. The assumption was chosen because it
is consistent with the observed dual-arm workspace: left-arm states occupy
positive `y`, right-arm states occupy negative `y`, and local `-Y/+Y` wrist
directions point inward toward the shared manipulation area.

Gate rule:

1. Convert rotation 6D to an orthonormal end-effector rotation matrix.
2. Build approximate camera position and optical axis for each sampled frame.
3. Use the observed workspace center and left/right end-effector midpoint as
   shared target candidates.
4. Count how often at least one shared target lies inside both camera frustums.
5. Enable the visual verifier only when the overlap score is at least `0.10`.

Only enabled pairs are passed to the visual verifier. The geometry gate uses
`state` only; `actions` is intentionally excluded from consistency checks. The
verifier checks motion correlation against the reference distribution, but the
result is treated as a weak signal and is emitted under `vision_state_vision`
(`1.1.2.C`), not under direct `vision_vision` (`1.1.2.B`).

Current geometry-gated issue types:

- `state_gated_view_pair_motion_inconsistency`
- `state_gated_view_motion_scale_mismatch`

## Scoring

Dimension budgets remain:

| dimension | points |
|---|---:|
| structural | 25 |
| vision_single | 20 |
| vision_vision | 10 |
| state | 15 |
| temporal | 15 |
| cross_modal | 15 |

Because geometry-gated multi-view findings are categorized as `1.1.2.C`, their
penalties are applied to `cross_modal`. The `vision_vision` budget is preserved
for future hard camera calibration metadata, but the current data does not
produce direct `vision_vision` findings.

Current regenerated dataset quality score: 78.46 / 100.
