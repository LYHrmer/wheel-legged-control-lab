Full observed episodes; each case has equal weight. Unequal terminal durations remain visible.
These averages do not establish successful completion or terrain traversal.

| Seed | Policy | Completed/cases | vx RMSE | Yaw RMSE | Height RMSE | Return | Durations / reason |
|---|---|---:|---:|---:|---:|---:|---|
| 48002 | zero | 2/2 | 0.040473 | 0.019131 | 0.007667 | 36.340411 | dev_flat: 6.00s/time_limit; dev_straight_road: 32.00s/time_limit |
| 48002 | unbounded | 1/2 | 0.110557 | 0.033410 | 0.024279 | 19.146616 | dev_flat: 6.00s/time_limit; dev_straight_road: 21.59s/fall_or_body_contact |
| 48002 | bounded | 2/2 | 0.092074 | 0.038676 | 0.028639 | 24.981425 | dev_flat: 6.00s/time_limit; dev_straight_road: 32.00s/time_limit |
| 48003 | zero | 2/2 | 0.040473 | 0.019131 | 0.007667 | 36.340411 | dev_flat: 6.00s/time_limit; dev_straight_road: 32.00s/time_limit |
| 48003 | unbounded | 1/2 | 0.037777 | 0.031045 | 0.005479 | 32.258822 | dev_flat: 6.00s/time_limit; dev_straight_road: 28.62s/map_boundary |
| 48003 | bounded | 1/2 | 0.059002 | 0.051281 | 0.008461 | 28.231624 | dev_flat: 6.00s/time_limit; dev_straight_road: 26.73s/map_boundary |

Common prefix per seed/case: recorded transitions 1..K, without padding or continuation.
Any termination penalty on transition K remains included; equal time does not mean equal terrain exposure.

| Seed | Case | K / seconds | Policy | vx RMSE | Yaw RMSE | Height RMSE | Return | Signed x progress |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 48002 | dev_flat | 600 / 6.00 | zero | 0.041226 | 0.024120 | 0.004285 | 11.627406 | 1.218529 |
| 48002 | dev_flat | 600 / 6.00 | unbounded | 0.078691 | 0.011563 | 0.019458 | 10.195731 | 1.457785 |
| 48002 | dev_flat | 600 / 6.00 | bounded | 0.061579 | 0.017857 | 0.022719 | 10.128101 | 1.160631 |
| 48003 | dev_flat | 600 / 6.00 | zero | 0.041226 | 0.024120 | 0.004285 | 11.627406 | 1.218529 |
| 48003 | dev_flat | 600 / 6.00 | unbounded | 0.042691 | 0.024305 | 0.005709 | 11.526179 | 1.198690 |
| 48003 | dev_flat | 600 / 6.00 | bounded | 0.055830 | 0.052079 | 0.004799 | 11.073397 | 1.116710 |
| 48002 | dev_straight_road | 2159 / 21.59 | zero | 0.033581 | 0.016800 | 0.009485 | 41.655398 | 4.992790 |
| 48002 | dev_straight_road | 2159 / 21.59 | unbounded | 0.142422 | 0.055257 | 0.029101 | 28.097501 | 5.933743 |
| 48002 | dev_straight_road | 2159 / 21.59 | bounded | 0.056651 | 0.019076 | 0.024761 | 35.718917 | 4.196825 |
| 48003 | dev_straight_road | 2673 / 26.73 | zero | 0.038343 | 0.015386 | 0.010453 | 51.180065 | 6.257837 |
| 48003 | dev_straight_road | 2673 / 26.73 | unbounded | 0.033032 | 0.037593 | 0.005313 | 51.353623 | 5.219683 |
| 48003 | dev_straight_road | 2673 / 26.73 | bounded | 0.062175 | 0.050482 | 0.012122 | 45.389850 | 4.146715 |

Units: vx m/s; yaw rad/s; height and progress m. Zero is reused, not replicated.
JSON preserves all original full summaries, per-case reward terms, prefix termination flags, and input hashes.
