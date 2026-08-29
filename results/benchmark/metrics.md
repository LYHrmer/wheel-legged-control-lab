# Wheel-legged control benchmark

| Controller | Scenario | Success | Velocity RMSE [m/s] | Pitch RMSE [deg] | Height RMSE [mm] | Effort RMS | Solve P95 [ms] |
|---|---|---:|---:|---:|---:|---:|---:|
| LQR | nominal | 1 | 0.544 | 2.236 | 20.31 | 0.381 | 0.000 |
| MPC | nominal | 1 | 0.540 | 2.169 | 20.32 | 0.380 | 0.417 |
| LQR + PPO | nominal | 1 | 0.436 | 2.309 | 13.74 | 0.383 | 0.000 |
| LQR | push | 1 | 0.672 | 2.680 | 19.75 | 0.385 | 0.000 |
| MPC | push | 1 | 0.668 | 2.615 | 19.77 | 0.384 | 0.434 |
| LQR + PPO | push | 1 | 0.538 | 2.984 | 14.10 | 0.389 | 0.000 |
| LQR | mismatch_delay | 1 | 0.543 | 2.663 | 125.23 | 0.438 | 0.000 |
| MPC | mismatch_delay | 1 | 0.538 | 2.586 | 125.23 | 0.437 | 0.637 |
| LQR + PPO | mismatch_delay | 1 | 0.429 | 2.687 | 33.76 | 0.488 | 0.000 |
