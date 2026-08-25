# Episode 02 And 05 Focus Diagnosis

## Key Finding

The problematic low-dimensional column is `actions`, not the measured `state`.

Both clips show an action-command plateau: the action vector repeats exactly, while wrist-camera images still change and the measured state still has small residual motion. So the symptom is better described as:

`actions` unchanged + wrist-camera vision still moving.

It is not strictly:

`state` unchanged + vision moving.

## Episode 02

- Finding: `V2F-000004`
- Issue span: transitions `127->128` through `129->130`, reported as frames `127-130`
- Clip span: frames `97-160`, i.e. 9.7s to 16.0s at 10 FPS
- Action static run around the issue: transitions `127->134`
- `actions_delta` in issue: median `0.0`
- `state_delta` in issue: median `0.00540267`, max `0.0292711`
- Combined visual motion in issue: max `18.0374`
- Local visual high cutoff in the focused clip: `14.9264`

Per-view visual motion during the issue:

| view | median motion | max motion |
|---|---:|---:|
| `image` | 7.6178 | 13.2496 |
| `left_wrist_image` | 22.6569 | 24.3299 |
| `right_wrist_image` | 13.4311 | 16.5327 |

Interpretation: the strongest evidence comes from the left wrist camera, with additional motion from the right wrist camera. The base camera is comparatively stable. Visually, the wrist-camera views show the end-effectors and floor grid continuing to move after the action vector has stopped changing. This looks like a command-hold / observation-lag segment: action target is held constant, but the real robot state and the wrist-mounted cameras continue settling.

## Episode 05

- Finding: `V2F-000015`
- Issue span: transitions `164->165` through `166->167`, reported as frames `164-167`
- Clip span: frames `134-197`, i.e. 13.4s to 19.7s at 10 FPS
- Action static run around the issue: transitions `164->168`
- `actions_delta` in issue: median `0.0`
- `state_delta` in issue: median `0.00234551`, max `0.104432`
- Combined visual motion in issue: max `20.8333`
- Local visual high cutoff in the focused clip: `10.8127`

Per-view visual motion during the issue:

| view | median motion | max motion |
|---|---:|---:|
| `image` | 6.3676 | 8.5429 |
| `left_wrist_image` | 24.8597 | 30.8406 |
| `right_wrist_image` | 18.6939 | 23.1165 |

Interpretation: episode 05 is stronger than episode 02. The left wrist camera has a very large motion spike, and the right wrist camera also moves a lot, while the base camera stays modest. The first issue transition, `164->165`, still has a large measured `state_delta` (`0.104432`) even though `actions_delta` is already zero. That is the clearest sign of temporal mismatch: commanded action is frozen, but the robot/observation stream is still moving.

## Likely Cause

The most likely mechanism is temporal misalignment or command-hold behavior:

- `actions` records the target command and can remain constant across several frames.
- `state` records measured robot state, which can keep changing after the command is held.
- wrist cameras are attached to the moving end-effectors, so small end-effector motion creates large pixel motion against the floor grid.
- because the detector combines all three visual streams, strong wrist-camera motion can trigger `visual_moves_state_static` even when the base camera is stable.

This does not look like a pure visual corruption event. It looks like a short action-label freeze or lag relative to the visual/state observations.

## Generated Files

- `episode-02_V2F-000004_focus_frames-0127-0130.avi`
- `episode-02_V2F-000004_focus_frames-0127-0130.gif`
- `episode-05_V2F-000015_focus_frames-0164-0167.avi`
- `episode-05_V2F-000015_focus_frames-0164-0167.gif`
- `episode-02_keyframes_contact.png`
- `episode-05_keyframes_contact.png`
- `episode_02_05_focus_summary.json`
