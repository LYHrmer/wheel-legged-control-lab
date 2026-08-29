# D1 full-body control benchmark

| Controller | Scenario | Success | Velocity RMSE [m/s] | Pitch RMSE [deg] | Height RMSE [mm] | Torque RMS | 4-wheel contact | Solve P95 [ms] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D1 LQR+VMC | nominal | 1 | 0.264 | 0.902 | 13.36 | 0.214 | 0.992 | 0.000 |
| D1 MPC+VMC | nominal | 1 | 0.244 | 1.320 | 14.81 | 0.212 | 0.973 | 0.402 |
| D1 LQR+VMC+PPO | nominal | 1 | 0.254 | 0.874 | 12.19 | 0.216 | 0.998 | 0.000 |
| D1 LQR+VMC | push | 1 | 0.345 | 1.528 | 16.32 | 0.214 | 0.965 | 0.000 |
| D1 MPC+VMC | push | 1 | 0.344 | 1.984 | 16.09 | 0.216 | 0.915 | 0.648 |
| D1 LQR+VMC+PPO | push | 1 | 0.330 | 1.430 | 14.94 | 0.215 | 0.972 | 0.000 |
| D1 LQR+VMC | mismatch_delay | 1 | 0.249 | 0.906 | 13.21 | 0.221 | 0.985 | 0.000 |
| D1 MPC+VMC | mismatch_delay | 1 | 0.252 | 1.321 | 14.38 | 0.221 | 0.972 | 0.408 |
| D1 LQR+VMC+PPO | mismatch_delay | 1 | 0.250 | 0.899 | 12.37 | 0.221 | 0.985 | 0.000 |
