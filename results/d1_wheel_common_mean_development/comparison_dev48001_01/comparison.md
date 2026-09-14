Full observed episodes; each case has equal weight. Unequal terminal durations remain visible.
These averages do not establish successful completion or terrain traversal.

| Seed | Policy | Completed/cases | vx RMSE | Yaw RMSE | Height RMSE | Return | Durations / reason |
|---|---|---:|---:|---:|---:|---:|---|
| 48001 | zero | 2/2 | 0.040473 | 0.019131 | 0.007667 | 36.340411 | dev_flat: 6.00s/time_limit; dev_straight_road: 32.00s/time_limit |
| 48001 | unbounded | 1/2 | 0.048367 | 0.029785 | 0.008958 | 32.405277 | dev_flat: 6.00s/time_limit; dev_straight_road: 29.50s/map_boundary |
| 48001 | bounded | 1/2 | 0.037063 | 0.032741 | 0.006835 | 29.890698 | dev_flat: 6.00s/time_limit; dev_straight_road: 26.30s/map_boundary |

Common prefix per seed/case: recorded transitions 1..K, without padding or continuation.
Any termination penalty on transition K remains included; equal time does not mean equal terrain exposure.

| Seed | Case | K / seconds | Policy | vx RMSE | Yaw RMSE | Height RMSE | Return | Signed x progress |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 48001 | dev_flat | 600 / 6.00 | zero | 0.041226 | 0.024120 | 0.004285 | 11.627406 | 1.218529 |
| 48001 | dev_flat | 600 / 6.00 | unbounded | 0.057385 | 0.034418 | 0.009385 | 11.083195 | 1.562877 |
| 48001 | dev_flat | 600 / 6.00 | bounded | 0.048440 | 0.033901 | 0.007658 | 11.291518 | 1.385975 |
| 48001 | dev_straight_road | 2630 / 26.30 | zero | 0.038314 | 0.015500 | 0.010413 | 50.365408 | 6.136692 |
| 48001 | dev_straight_road | 2630 / 26.30 | unbounded | 0.040433 | 0.025459 | 0.008833 | 49.571424 | 6.986149 |
| 48001 | dev_straight_road | 2630 / 26.30 | bounded | 0.025687 | 0.031580 | 0.006011 | 48.489878 | 6.004032 |

Units: vx m/s; yaw rad/s; height and progress m. Zero is reused, not replicated.
JSON preserves all original full summaries, per-case reward terms, prefix termination flags, and input hashes.
