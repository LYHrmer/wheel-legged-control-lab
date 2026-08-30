# D1 full-body control benchmark

| Controller | Scenario | Success | Velocity RMSE [m/s] | Pitch RMSE [deg] | Height RMSE [mm] | Torque RMS | 4-wheel contact | Solve P95 [ms] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D1 LQR+VMC | nominal | 1 | 0.263 | 0.930 | 13.73 | 0.214 | 0.987 | 0.000 |
| D1 MPC+VMC | nominal | 1 | 0.240 | 1.268 | 14.32 | 0.213 | 0.972 | 0.391 |
| D1 LQR+VMC+PPO | nominal | 1 | 0.250 | 0.853 | 13.30 | 0.216 | 0.998 | 0.000 |
| D1 LQR+VMC | push | 1 | 0.350 | 1.556 | 16.40 | 0.214 | 0.963 | 0.000 |
| D1 MPC+VMC | push | 1 | 0.344 | 2.055 | 16.28 | 0.216 | 0.918 | 0.523 |
| D1 LQR+VMC+PPO | push | 1 | 0.334 | 1.460 | 15.77 | 0.214 | 0.975 | 0.000 |
| D1 LQR+VMC | mismatch_delay | 1 | 0.252 | 0.905 | 13.17 | 0.222 | 0.990 | 0.000 |
| D1 MPC+VMC | mismatch_delay | 1 | 0.249 | 1.307 | 14.42 | 0.221 | 0.963 | 0.317 |
| D1 LQR+VMC+PPO | mismatch_delay | 1 | 0.252 | 0.847 | 13.17 | 0.223 | 0.997 | 0.000 |
