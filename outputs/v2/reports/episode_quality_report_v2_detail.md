# V2 Episode detailed quality report

This report is regenerated from the current merged findings.

## Overview

| episode | task | length | structural | vision_single | vision_vision | state | temporal | cross_modal | total | findings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 222 | 25 | 20 | 10 | 15 | 0 | 15 | 85 | 2 |
| 1 | 1 | 192 | 25 | 20 | 10 | 15 | 0 | 15 | 85 | 2 |
| 2 | 0 | 202 | 25 | 20 | 10 | 15 | 0 | 4.49 | 74.49 | 5 |
| 3 | 1 | 189 | 25 | 20 | 10 | 15 | 0 | 15 | 85 | 2 |
| 4 | 0 | 212 | 25 | 20 | 10 | 15 | 0 | 0 | 70 | 7 |
| 5 | 0 | 247 | 25 | 20 | 10 | 15 | 0 | 0 | 70 | 6 |
| 6 | 1 | 202 | 25 | 20 | 10 | 15 | 0 | 15 | 85 | 3 |
| 7 | 0 | 236 | 25 | 20 | 10 | 15 | 0 | 0 | 70 | 4 |
| 8 | 0 | 191 | 25 | 20 | 10 | 15 | 0 | 0 | 70 | 6 |
| 9 | 0 | 195 | 25 | 20 | 10 | 15 | 0 | 0 | 70 | 4 |
| 10 | 1 | 196 | 25 | 20 | 10 | 15 | 0 | 15 | 85 | 5 |
| 11 | 0 | 202 | 25 | 20 | 10 | 15 | 0 | 5.76 | 75.76 | 4 |
| 12 | 1 | 172 | 25 | 20 | 10 | 15 | 0 | 15 | 85 | 4 |
| 13 | 0 | 304 | 25 | 20 | 10 | 15 | 0 | 4.49 | 74.49 | 3 |
| 14 | 0 | 268 | 25 | 20 | 10 | 15 | 0 | 0 | 70 | 5 |
| 15 | 1 | 181 | 25 | 20 | 10 | 15 | 0 | 15 | 85 | 2 |
| 16 | 1 | 180 | 25 | 20 | 10 | 15 | 0 | 15 | 85 | 7 |
| 17 | 1 | 170 | 25 | 20 | 10 | 15 | 0 | 15 | 85 | 2 |
| 18 | 0 | 217 | 25 | 20 | 10 | 15 | 0 | 0 | 70 | 6 |
| 19 | 0 | 164 | 25 | 20 | 10 | 15 | 15 | 4.49 | 89.49 | 1 |

## Episode Details

### Episode 0

- task: 0
- length: 222
- score_total: 85
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 15
- findings: 2 | critical 0 | high_confidence 2 | suspicious 0 | ood 0
- findings detail:
  - [V2F-000002] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 0-9 | column actions | 高置信异常 | severity 57.5
  - [V2F-000001] low_dim_jitter_or_spike - Low-dimensional state acceleration is an extreme outlier | category 1.2.2.B | state | segment | frame 126 | column state | 高置信异常 | severity 92.68

### Episode 1

- task: 1
- length: 192
- score_total: 85
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 15
- findings: 2 | critical 0 | high_confidence 2 | suspicious 0 | ood 0
- findings detail:
  - [V2F-000003] low_dim_jitter_or_spike - Low-dimensional state acceleration is an extreme outlier | category 1.2.2.B | state | segment | frame 87 | column state | 高置信异常 | severity 88.15
  - [V2F-000004] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 135-191 | column actions | 高置信异常 | severity 90

### Episode 2

- task: 0
- length: 202
- score_total: 74.49
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 4.49
- findings: 5 | critical 0 | high_confidence 4 | suspicious 1 | ood 0
- findings detail:
  - [V2F-000007] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 0-9 | column actions | 高置信异常 | severity 57.5
  - [V2F-000008] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 127-135 | column actions | 高置信异常 | severity 55
  - [V2F-000009] visual_moves_state_static - Visual motion is strong but low-dimensional state is static | category 1.1.2.D | vision-state | segment | frames 127-130 | column actions | 高置信异常 | severity 69
  - [V2F-000006] visual_high_frequency_jitter - Visual motion has high-frequency acceleration spikes | category 1.2.2.A | vision | segment | frames 154-158 | view image | 高置信异常 | severity 65
  - [V2F-000005] visual_fast_jump - Visual sequence has an extreme fast jump | category 1.2.2.A | vision | segment | frames 155-157 | view image | 疑似异常 | severity 41.57

### Episode 3

- task: 1
- length: 189
- score_total: 85
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 15
- findings: 2 | critical 0 | high_confidence 2 | suspicious 0 | ood 0
- findings detail:
  - [V2F-000011] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 140-188 | column actions | 高置信异常 | severity 90
  - [V2F-000010] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 157-184 | column state | 高置信异常 | severity 90

### Episode 4

- task: 0
- length: 212
- score_total: 70
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 0
- findings: 7 | critical 0 | high_confidence 4 | suspicious 0 | ood 3
- findings detail:
  - [V2F-000016] state_gated_view_pair_motion_inconsistency - State-supported overlapping camera pair has weak visual agreement | category 1.1.2.C | vision-state | episode | episode-level | view image|right_wrist_image | 分布外样本 | severity 45
  - [V2F-000017] low_cross_modal_correlation - Vision-State motion correlation is lower than reference | category 1.2.2.C | vision-state | episode | episode-level | view image | column actions | 分布外样本 | severity 45
  - [V2F-000018] cross_modal_lag_shift - Vision-State best lag deviates from reference lag | category 1.2.2.C | vision-state | episode | episode-level | view right_wrist_image | column actions | 分布外样本 | severity 59
  - [V2F-000014] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 0-9 | column state | 高置信异常 | severity 57.5
  - [V2F-000015] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 0-8 | column actions | 高置信异常 | severity 55
  - [V2F-000013] low_motion_freeze_run - Visual motion is near zero for a long window | category 1.2.2.A | vision | segment | frames 5-13 | view right_wrist_image | 高置信异常 | severity 65
  - [V2F-000012] visual_high_frequency_jitter - Visual motion has high-frequency acceleration spikes | category 1.2.2.A | vision | segment | frames 186-190 | view image | 高置信异常 | severity 65

### Episode 5

- task: 0
- length: 247
- score_total: 70
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 0
- findings: 6 | critical 0 | high_confidence 4 | suspicious 0 | ood 2
- findings detail:
  - [V2F-000022] state_gated_view_pair_motion_inconsistency - State-supported overlapping camera pair has weak visual agreement | category 1.1.2.C | vision-state | episode | episode-level | view image|right_wrist_image | 分布外样本 | severity 45
  - [V2F-000023] state_gated_view_pair_motion_inconsistency - State-supported overlapping camera pair has weak visual agreement | category 1.1.2.C | vision-state | episode | episode-level | view left_wrist_image|right_wrist_image | 分布外样本 | severity 45
  - [V2F-000020] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 0-8 | column actions | 高置信异常 | severity 55
  - [V2F-000019] low_dim_jitter_or_spike - Low-dimensional state acceleration is an extreme outlier | category 1.2.2.B | state | segment | frame 159 | column state | 高置信异常 | severity 100
  - [V2F-000021] low_dim_jitter_or_spike - Low-dimensional state acceleration is an extreme outlier | category 1.2.2.B | state | segment | frame 160 | column actions | 高置信异常 | severity 100
  - [V2F-000024] visual_moves_state_static - Visual motion is strong but low-dimensional state is static | category 1.1.2.D | vision-state | segment | frames 164-167 | column actions | 高置信异常 | severity 69

### Episode 6

- task: 1
- length: 202
- score_total: 85
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 15
- findings: 3 | critical 0 | high_confidence 3 | suspicious 0 | ood 0
- findings detail:
  - [V2F-000028] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 156-201 | column actions | 高置信异常 | severity 90
  - [V2F-000025] low_motion_freeze_run - Visual motion is near zero for a long window | category 1.2.2.A | vision | segment | frames 172-191 | view left_wrist_image | 高置信异常 | severity 67.5
  - [V2F-000027] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 172-185 | column state | 高置信异常 | severity 67.5

### Episode 7

- task: 0
- length: 236
- score_total: 70
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 0
- findings: 4 | critical 0 | high_confidence 3 | suspicious 0 | ood 1
- findings detail:
  - [V2F-000031] state_gated_view_pair_motion_inconsistency - State-supported overlapping camera pair has weak visual agreement | category 1.1.2.C | vision-state | episode | episode-level | view left_wrist_image|right_wrist_image | 分布外样本 | severity 45
  - [V2F-000030] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 100-108 | column actions | 高置信异常 | severity 55
  - [V2F-000029] low_dim_jitter_or_spike - Low-dimensional state acceleration is an extreme outlier | category 1.2.2.B | state | segment | frame 158 | column state | 高置信异常 | severity 95.41
  - [V2F-000032] visual_moves_state_static - Visual motion is strong but low-dimensional state is static | category 1.1.2.D | vision-state | segment | frames 163-165 | column actions | 高置信异常 | severity 61

### Episode 8

- task: 0
- length: 191
- score_total: 70
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 0
- findings: 6 | critical 0 | high_confidence 4 | suspicious 1 | ood 1
- findings detail:
  - [V2F-000037] state_gated_view_pair_motion_inconsistency - State-supported overlapping camera pair has weak visual agreement | category 1.1.2.C | vision-state | episode | episode-level | view image|right_wrist_image | 分布外样本 | severity 35
  - [V2F-000035] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 0-8 | column actions | 高置信异常 | severity 55
  - [V2F-000036] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 107-115 | column actions | 高置信异常 | severity 55
  - [V2F-000038] visual_moves_state_static - Visual motion is strong but low-dimensional state is static | category 1.1.2.D | vision-state | segment | frames 107-110 | column actions | 高置信异常 | severity 69
  - [V2F-000034] visual_high_frequency_jitter - Visual motion has high-frequency acceleration spikes | category 1.2.2.A | vision | segment | frames 153-157 | view image | 高置信异常 | severity 65
  - [V2F-000033] visual_fast_jump - Visual sequence has an extreme fast jump | category 1.2.2.A | vision | segment | frames 155-156 | view image | 疑似异常 | severity 35

### Episode 9

- task: 0
- length: 195
- score_total: 70
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 0
- findings: 4 | critical 0 | high_confidence 3 | suspicious 0 | ood 1
- findings detail:
  - [V2F-000041] state_gated_view_pair_motion_inconsistency - State-supported overlapping camera pair has weak visual agreement | category 1.1.2.C | vision-state | episode | episode-level | view left_wrist_image|right_wrist_image | 分布外样本 | severity 45
  - [V2F-000039] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 0-9 | column actions | 高置信异常 | severity 57.5
  - [V2F-000040] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 62-72 | column actions | 高置信异常 | severity 60
  - [V2F-000042] visual_moves_state_static - Visual motion is strong but low-dimensional state is static | category 1.1.2.D | vision-state | segment | frames 130-133 | column actions | 高置信异常 | severity 69

### Episode 10

- task: 1
- length: 196
- score_total: 85
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 15
- findings: 5 | critical 0 | high_confidence 4 | suspicious 1 | ood 0
- findings detail:
  - [V2F-000047] low_dim_jitter_or_spike - Low-dimensional state acceleration is an extreme outlier | category 1.2.2.B | state | segment | frame 98 | column actions | 高置信异常 | severity 85.24
  - [V2F-000048] low_dim_jitter_or_spike - Low-dimensional state acceleration is an extreme outlier | category 1.2.2.B | state | segment | frame 103 | column actions | 高置信异常 | severity 85.17
  - [V2F-000043] visual_fast_jump - Visual sequence has an extreme fast jump | category 1.2.2.A | vision | segment | frames 106-108 | view left_wrist_image | 疑似异常 | severity 35
  - [V2F-000046] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 159-195 | column actions | 高置信异常 | severity 90
  - [V2F-000044] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 168-193 | column state | 高置信异常 | severity 65

### Episode 11

- task: 0
- length: 202
- score_total: 75.76
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 5.76
- findings: 4 | critical 0 | high_confidence 4 | suspicious 0 | ood 0
- findings detail:
  - [V2F-000050] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 0-15 | column actions | 高置信异常 | severity 72.5
  - [V2F-000049] low_dim_jitter_or_spike - Low-dimensional state acceleration is an extreme outlier | category 1.2.2.B | state | segment | frame 112 | column state | 高置信异常 | severity 95.27
  - [V2F-000051] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 112-123 | column actions | 高置信异常 | severity 62.5
  - [V2F-000052] visual_moves_state_static - Visual motion is strong but low-dimensional state is static | category 1.1.2.D | vision-state | segment | frames 112-114 | column actions | 高置信异常 | severity 61

### Episode 12

- task: 1
- length: 172
- score_total: 85
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 15
- findings: 4 | critical 0 | high_confidence 3 | suspicious 1 | ood 0
- findings detail:
  - [V2F-000056] low_dim_jitter_or_spike - Low-dimensional state acceleration is an extreme outlier | category 1.2.2.B | state | segment | frame 84 | column state | 高置信异常 | severity 86.54
  - [V2F-000053] visual_fast_jump - Visual sequence has an extreme fast jump | category 1.2.2.A | vision | segment | frames 87-88 | view right_wrist_image | 疑似异常 | severity 38.1
  - [V2F-000057] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 128-171 | column actions | 高置信异常 | severity 90
  - [V2F-000054] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 140-171 | column state | 高置信异常 | severity 75

### Episode 13

- task: 0
- length: 304
- score_total: 74.49
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 4.49
- findings: 3 | critical 0 | high_confidence 3 | suspicious 0 | ood 0
- findings detail:
  - [V2F-000060] visual_moves_state_static - Visual motion is strong but low-dimensional state is static | category 1.1.2.D | vision-state | segment | frames 119-122 | column actions | 高置信异常 | severity 69
  - [V2F-000059] visual_high_frequency_jitter - Visual motion has high-frequency acceleration spikes | category 1.2.2.A | vision | segment | frames 140-144 | view image | 高置信异常 | severity 65
  - [V2F-000058] visual_fast_jump - Visual sequence has an extreme fast jump | category 1.2.2.A | vision | segment | frames 141-143 | view image | 高置信异常 | severity 53.57

### Episode 14

- task: 0
- length: 268
- score_total: 70
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 0
- findings: 5 | critical 0 | high_confidence 3 | suspicious 0 | ood 2
- findings detail:
  - [V2F-000063] state_gated_view_pair_motion_inconsistency - State-supported overlapping camera pair has weak visual agreement | category 1.1.2.C | vision-state | episode | episode-level | view image|right_wrist_image | 分布外样本 | severity 35
  - [V2F-000065] low_cross_modal_correlation - Vision-State motion correlation is lower than reference | category 1.2.2.C | vision-state | episode | episode-level | view image | column actions | 分布外样本 | severity 45
  - [V2F-000062] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 0-8 | column actions | 高置信异常 | severity 55
  - [V2F-000061] low_dim_jitter_or_spike - Low-dimensional state acceleration is an extreme outlier | category 1.2.2.B | state | segment | frame 142 | column state | 高置信异常 | severity 100
  - [V2F-000064] visual_moves_state_static - Visual motion is strong but low-dimensional state is static | category 1.1.2.D | vision-state | segment | frames 164-167 | column actions | 高置信异常 | severity 69

### Episode 15

- task: 1
- length: 181
- score_total: 85
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 15
- findings: 2 | critical 0 | high_confidence 2 | suspicious 0 | ood 0
- findings detail:
  - [V2F-000067] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 144-180 | column actions | 高置信异常 | severity 90
  - [V2F-000066] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 164-180 | column state | 高置信异常 | severity 75

### Episode 16

- task: 1
- length: 180
- score_total: 85
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 15
- findings: 7 | critical 0 | high_confidence 7 | suspicious 0 | ood 0
- findings detail:
  - [V2F-000073] low_dim_jitter_or_spike - Low-dimensional state acceleration is an extreme outlier | category 1.2.2.B | state | segment | frame 86 | column actions | 高置信异常 | severity 86.32
  - [V2F-000074] low_dim_jitter_or_spike - Low-dimensional state acceleration is an extreme outlier | category 1.2.2.B | state | segment | frame 91 | column actions | 高置信异常 | severity 92.58
  - [V2F-000071] low_dim_jitter_or_spike - Low-dimensional state acceleration is an extreme outlier | category 1.2.2.B | state | segment | frame 92 | column state | 高置信异常 | severity 91.54
  - [V2F-000072] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 140-179 | column actions | 高置信异常 | severity 90
  - [V2F-000069] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 149-160 | column state | 高置信异常 | severity 62.5
  - [V2F-000068] low_motion_freeze_run - Visual motion is near zero for a long window | category 1.2.2.A | vision | segment | frames 168-177 | view image | 高置信异常 | severity 67.5
  - [V2F-000070] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 169-177 | column state | 高置信异常 | severity 55

### Episode 17

- task: 1
- length: 170
- score_total: 85
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 15
- findings: 2 | critical 0 | high_confidence 2 | suspicious 0 | ood 0
- findings detail:
  - [V2F-000077] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 119-169 | column actions | 高置信异常 | severity 90
  - [V2F-000075] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 132-168 | column state | 高置信异常 | severity 90

### Episode 18

- task: 0
- length: 217
- score_total: 70
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 0, cross_modal 0
- findings: 6 | critical 0 | high_confidence 3 | suspicious 0 | ood 3
- findings detail:
  - [V2F-000080] state_gated_view_pair_motion_inconsistency - State-supported overlapping camera pair has weak visual agreement | category 1.1.2.C | vision-state | episode | episode-level | view image|right_wrist_image | 分布外样本 | severity 45
  - [V2F-000081] state_gated_view_pair_motion_inconsistency - State-supported overlapping camera pair has weak visual agreement | category 1.1.2.C | vision-state | episode | episode-level | view left_wrist_image|right_wrist_image | 分布外样本 | severity 35
  - [V2F-000083] low_cross_modal_correlation - Vision-State motion correlation is lower than reference | category 1.2.2.C | vision-state | episode | episode-level | view image | column actions | 分布外样本 | severity 45
  - [V2F-000079] low_dim_freeze_run - Low-dimensional state is nearly unchanged for a long window | category 1.2.2.B | state | segment | frames 0-8 | column actions | 高置信异常 | severity 55
  - [V2F-000078] low_dim_jitter_or_spike - Low-dimensional state acceleration is an extreme outlier | category 1.2.2.B | state | segment | frame 128 | column state | 高置信异常 | severity 96.47
  - [V2F-000082] visual_moves_state_static - Visual motion is strong but low-dimensional state is static | category 1.1.2.D | vision-state | segment | frames 128-131 | column actions | 高置信异常 | severity 69

### Episode 19

- task: 0
- length: 164
- score_total: 89.49
- breakdown: structural 25, vision_single 20, vision_vision 10, state 15, temporal 15, cross_modal 4.49
- findings: 1 | critical 0 | high_confidence 1 | suspicious 0 | ood 0
- findings detail:
  - [V2F-000084] visual_moves_state_static - Visual motion is strong but low-dimensional state is static | category 1.1.2.D | vision-state | segment | frames 92-95 | column actions | 高置信异常 | severity 69

