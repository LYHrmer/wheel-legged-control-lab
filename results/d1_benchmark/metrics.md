# D1 full-body control benchmark

| Controller | Scenario | Seed | State | Compensation | Success | Velocity RMSE [m/s] | Pitch RMSE [deg] | Height RMSE [mm] | State age P95 [ms] | Power [W] | Torque saturation | Solve P95 [ms] |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 LQR+VMC | nominal | 21 | oracle | none | 1 | 0.256 | 0.930 | 13.77 | 0.000 | 78.50 | 0.000 | 0.000 |
| D1 MPC+VMC | nominal | 21 | oracle | none | 1 | 0.240 | 1.267 | 14.31 | 0.000 | 70.47 | 0.000 | 0.369 |
| D1 LQR+VMC+PPO | nominal | 21 | oracle | none | 1 | 0.250 | 0.854 | 13.29 | 0.000 | 80.98 | 0.000 | 0.000 |
| D1 LQR+VMC | push | 21 | oracle | none | 1 | 0.349 | 1.586 | 16.48 | 0.000 | 85.78 | 0.000 | 0.000 |
| D1 MPC+VMC | push | 21 | oracle | none | 1 | 0.343 | 2.040 | 16.28 | 0.000 | 84.43 | 0.000 | 0.605 |
| D1 LQR+VMC+PPO | push | 21 | oracle | none | 1 | 0.331 | 1.446 | 15.71 | 0.000 | 87.03 | 0.000 | 0.000 |
| D1 LQR+VMC | mismatch_delay | 21 | oracle | none | 1 | 0.270 | 0.940 | 10.11 | 0.000 | 78.39 | 0.000 | 0.000 |
| D1 MPC+VMC | mismatch_delay | 21 | oracle | none | 1 | 0.252 | 1.340 | 11.31 | 0.000 | 75.01 | 0.000 | 0.340 |
| D1 LQR+VMC+PPO | mismatch_delay | 21 | oracle | none | 1 | 0.249 | 0.887 | 9.87 | 0.000 | 82.88 | 0.000 | 0.000 |
