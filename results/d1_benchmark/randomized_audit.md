# D1 randomized-domain audit

Matched seeds; values are episode mean ± sample standard deviation.

| Controller | Episodes | Failures | Velocity RMSE | Pitch RMSE | Height RMSE |
|---|---:|---:|---:|---:|---:|
| D1 LQR+VMC | 30 | 0 | 0.328 ± 0.131 | 1.567 ± 0.956 | 10.080 ± 4.045 |
| D1 MPC+VMC | 30 | 1 | 0.294 ± 0.140 | 1.971 ± 2.231 | 11.024 ± 5.359 |
| D1 LQR+VMC+PPO | 30 | 0 | 0.322 ± 0.118 | 1.494 ± 0.729 | 9.859 ± 2.994 |

## Paired PPO − LQR differences (95% t confidence interval)

- Velocity RMSE [m/s]: -0.006 [-0.017, +0.005]
- Pitch RMSE [deg]: -0.073 [-0.209, +0.064]
- Height RMSE [mm]: -0.221 [-0.993, +0.550]
