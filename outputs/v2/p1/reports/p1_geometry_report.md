# P1 Spatial Geometry Report

- reference episodes: 20
- target episodes: 20
- findings: 0
- episodes with comparable pairs: 0
- episodes with panel proxy: 0
- episodes with motion coupling: 20

## Finding Types

| issue_type | count |
|---|---:|

## Episode Summary

| episode | task | length | findings | comparable pairs | panel valid | main | panel | overlap | bimanual | coupling | best lag | mismatch sv | mismatch vs |
|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---:|
| 0 | 0 | 222 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.604683 | 3 | 0.04072398190045249 | 0.027149321266968326 |
| 1 | 1 | 192 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.583632 | 4 | 0.0 | 0.0 |
| 2 | 0 | 202 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.754845 | 4 | 0.03482587064676617 | 0.024875621890547265 |
| 3 | 1 | 189 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.582103 | 4 | 0.005319148936170213 | 0.005319148936170213 |
| 4 | 0 | 212 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.512875 | 4 | 0.03317535545023697 | 0.037914691943127965 |
| 5 | 0 | 247 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.585306 | 3 | 0.028455284552845527 | 0.012195121951219513 |
| 6 | 1 | 202 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.580769 | 4 | 0.0 | 0.0 |
| 7 | 0 | 236 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.571658 | 3 | 0.03404255319148936 | 0.02553191489361702 |
| 8 | 0 | 191 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.711433 | 3 | 0.03684210526315789 | 0.010526315789473684 |
| 9 | 0 | 195 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.554724 | 5 | 0.03608247422680412 | 0.020618556701030927 |
| 10 | 1 | 196 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.588335 | 4 | 0.035897435897435895 | 0.005128205128205128 |
| 11 | 0 | 202 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.657666 | 3 | 0.01990049751243781 | 0.0 |
| 12 | 1 | 172 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.586786 | 4 | 0.029239766081871343 | 0.011695906432748537 |
| 13 | 0 | 304 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.725754 | 5 | 0.033003300330033 | 0.0231023102310231 |
| 14 | 0 | 268 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.600911 | 4 | 0.0449438202247191 | 0.04868913857677903 |
| 15 | 1 | 181 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.343667 | 5 | 0.05 | 0.005555555555555556 |
| 16 | 1 | 180 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.527983 | 4 | 0.00558659217877095 | 0.0111731843575419 |
| 17 | 1 | 170 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.580189 | 5 | 0.011834319526627219 | 0.0 |
| 18 | 0 | 217 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.590438 | 5 | 0.018518518518518517 | 0.032407407407407406 |
| 19 | 0 | 164 | 0 | 0 | 0.000 | ok | unavailable | True | unavailable | 0.652333 | 4 | 0.024539877300613498 | 0.03067484662576687 |

## Notes

P1 is intentionally weaker than calibrated 3D geometry. It reports state-visual motion coupling even when cross-view feature overlap is unavailable. Strict reprojection, epipolar, and triangulation checks remain unavailable unless camera calibration is provided.
