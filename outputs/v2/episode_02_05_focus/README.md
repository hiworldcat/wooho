# Episode 02/05 Focus Analysis

The flagged column is `actions`, not the sensor `state`: the action command row is repeated while the observed state and images continue changing.

| episode | issue frames | repeated actions | visual p95 cutoff | issue max visual | state median delta | actions median delta | video |
|---:|---:|---|---:|---:|---:|---:|---|
| 2 | 127-130 | True | 14.926 | 18.037 | 0.00540267 | 0 | `episode-02_V2F-000004_focus_frames-0127-0130.avi` |
| 5 | 164-167 | True | 10.813 | 20.833 | 0.00234551 | 0 | `episode-05_V2F-000015_focus_frames-0164-0167.avi` |
