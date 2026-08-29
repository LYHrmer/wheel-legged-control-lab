# D1 randomized-domain audit

Matched seeds; values are episode mean ± sample standard deviation.

| Controller | Episodes | Failures | Velocity RMSE | Pitch RMSE | Height RMSE |
|---|---:|---:|---:|---:|---:|
| D1 LQR+VMC | 30 | 0 | 0.291 ± 0.111 | 1.245 ± 0.886 | 10.822 ± 4.345 |
| D1 MPC+VMC | 30 | 1 | 0.291 ± 0.151 | 1.898 ± 2.240 | 11.574 ± 5.289 |
| D1 LQR+VMC+PPO | 30 | 0 | 0.285 ± 0.105 | 1.136 ± 0.642 | 9.902 ± 3.555 |

## Paired PPO − LQR differences (95% t confidence interval)

- Velocity RMSE [m/s]: -0.005 [-0.009, -0.002]
- Pitch RMSE [deg]: -0.109 [-0.206, -0.013]
- Height RMSE [mm]: -0.920 [-1.455, -0.385]
