# Randomized-domain audit

Each value is the episode mean ± standard deviation over matched random seeds.

| Controller | Episodes | Failures | Mean reward | Velocity RMSE [m/s] | Pitch RMSE [deg] | Height RMSE [mm] |
|---|---:|---:|---:|---:|---:|---:|
| LQR | 20 | 0 | 0.783 ± 0.050 | 0.565 ± 0.163 | 2.471 ± 0.655 | 99.284 ± 27.200 |
| MPC | 20 | 0 | 0.783 ± 0.050 | 0.563 ± 0.162 | 2.404 ± 0.637 | 99.288 ± 27.186 |
| LQR + PPO | 20 | 0 | 0.881 ± 0.034 | 0.480 ± 0.130 | 2.557 ± 0.716 | 22.384 ± 7.692 |
