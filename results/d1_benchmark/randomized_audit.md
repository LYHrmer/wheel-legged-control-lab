# D1 randomized-domain audit

Matched seeds; values are episode mean ± sample standard deviation.

| Controller | Episodes | Failures | Mean reward | Velocity RMSE | Pitch RMSE | Height RMSE |
|---|---:|---:|---:|---:|---:|---:|
| D1 LQR+VMC | 30 | 0 | 3.425 ± 0.308 | 0.326 ± 0.124 | 1.490 ± 0.824 | 9.745 ± 3.491 |
| D1 MPC+VMC | 30 | 1 | 3.545 ± 0.347 | 0.298 ± 0.152 | 2.162 ± 3.216 | 11.429 ± 9.387 |
| D1 LQR+VMC+PPO | 30 | 0 | 3.378 ± 0.311 | 0.327 ± 0.121 | 1.511 ± 0.789 | 9.725 ± 3.262 |

## Paired PPO − LQR differences (95% t confidence interval)

Matched evaluation seeds: 30.

- Mean reward: -0.047 [-0.071, -0.022]
- Velocity RMSE [m/s]: +0.001 [-0.009, +0.011]
- Pitch RMSE [deg]: +0.021 [-0.054, +0.096]
- Height RMSE [mm]: -0.020 [-0.622, +0.582]
