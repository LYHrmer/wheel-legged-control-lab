# D1 state-delay sensitivity

Fixed delay grid: 0/10/20/30/50 ms. Domain samples, initial state, initial command, estimator seed, and planned push are paired and checked by evaluation seed.

Success and episode duration are the primary robustness outcomes. Every other continuous metric uses only the trajectory prefix observed before truncation or a fall; a smaller error after an early fall is not evidence of better robustness.

Absolute success intervals use Wilson scores. Absolute continuous intervals use mean t intervals. Success differences versus 0 ms use a deterministic paired bootstrap; other differences use paired t intervals. With fewer than two pairs, the interval is unavailable and stored as NaN in CSV.

| Controller | Delay [ms] | Metric | Unit | Estimate [95% CI] | Delta vs 0 ms [95% CI] |
|---|---:|---|---|---:|---:|
| D1 LQR+VMC | 0 | Success | ratio | 1.000 [0.886, 1.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Episode duration | s | 6.000 [6.000, 6.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Mean reward | reward/step | 3.454 [3.353, 3.556] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Velocity RMSE | m/s | 0.307 [0.270, 0.343] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Pitch RMSE | deg | 1.302 [1.114, 1.490] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Maximum absolute pitch | deg | 3.754 [3.114, 4.393] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Height RMSE | mm | 10.177 [9.371, 10.982] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Normalized torque RMS | ratio | 0.235 [0.225, 0.245] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Mean absolute mechanical power | W | 175.957 [128.315, 223.599] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Torque saturation | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Four-wheel contact | ratio | 0.898 [0.848, 0.948] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Undesired-contact steps | steps | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Solve-time P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Measured state age mean | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Measured state age P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Measured state age maximum | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Compensation horizon P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Compensation applied | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Compensation rejected | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Raw position-estimation RMSE | m | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Control position-estimation RMSE | m | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Raw pitch-estimation RMSE | deg | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Control pitch-estimation RMSE | deg | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Raw velocity-estimation RMSE | m/s | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Control velocity-estimation RMSE | m/s | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Residual-action RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Residual-action delta RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Mean residual longitudinal force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 0 | Mean residual vertical force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 10 | Success | ratio | 1.000 [0.886, 1.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 10 | Episode duration | s | 6.000 [6.000, 6.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 10 | Mean reward | reward/step | 3.053 [2.964, 3.141] | -0.402 [-0.479, -0.324] |
| D1 LQR+VMC | 10 | Velocity RMSE | m/s | 0.403 [0.365, 0.441] | +0.096 [+0.069, +0.123] |
| D1 LQR+VMC | 10 | Pitch RMSE | deg | 2.796 [2.553, 3.040] | +1.494 [+1.252, +1.737] |
| D1 LQR+VMC | 10 | Maximum absolute pitch | deg | 9.031 [7.841, 10.221] | +5.277 [+4.085, +6.469] |
| D1 LQR+VMC | 10 | Height RMSE | mm | 12.931 [12.253, 13.609] | +2.754 [+1.922, +3.586] |
| D1 LQR+VMC | 10 | Normalized torque RMS | ratio | 0.307 [0.302, 0.311] | +0.072 [+0.062, +0.082] |
| D1 LQR+VMC | 10 | Mean absolute mechanical power | W | 601.164 [575.329, 626.999] | +425.207 [+379.214, +471.200] |
| D1 LQR+VMC | 10 | Torque saturation | ratio | 0.003 [0.002, 0.003] | +0.003 [+0.002, +0.003] |
| D1 LQR+VMC | 10 | Four-wheel contact | ratio | 0.429 [0.396, 0.462] | -0.469 [-0.506, -0.431] |
| D1 LQR+VMC | 10 | Undesired-contact steps | steps | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 10 | Solve-time P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 10 | Measured state age mean | ms | 10.000 [10.000, 10.000] | +10.000 [+10.000, +10.000] |
| D1 LQR+VMC | 10 | Measured state age P95 | ms | 10.000 [10.000, 10.000] | +10.000 [+10.000, +10.000] |
| D1 LQR+VMC | 10 | Measured state age maximum | ms | 10.000 [10.000, 10.000] | +10.000 [+10.000, +10.000] |
| D1 LQR+VMC | 10 | Compensation horizon P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 10 | Compensation applied | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 10 | Compensation rejected | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 10 | Raw position-estimation RMSE | m | 0.003 [0.002, 0.003] | +0.003 [+0.002, +0.003] |
| D1 LQR+VMC | 10 | Control position-estimation RMSE | m | 0.003 [0.002, 0.003] | +0.003 [+0.002, +0.003] |
| D1 LQR+VMC | 10 | Raw pitch-estimation RMSE | deg | 0.334 [0.316, 0.353] | +0.334 [+0.316, +0.353] |
| D1 LQR+VMC | 10 | Control pitch-estimation RMSE | deg | 0.334 [0.316, 0.353] | +0.334 [+0.316, +0.353] |
| D1 LQR+VMC | 10 | Raw velocity-estimation RMSE | m/s | 0.024 [0.023, 0.025] | +0.024 [+0.023, +0.025] |
| D1 LQR+VMC | 10 | Control velocity-estimation RMSE | m/s | 0.024 [0.023, 0.025] | +0.024 [+0.023, +0.025] |
| D1 LQR+VMC | 10 | Residual-action RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 10 | Residual-action delta RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 10 | Mean residual longitudinal force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 10 | Mean residual vertical force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 20 | Success | ratio | 0.233 [0.118, 0.409] | -0.767 [-0.900, -0.600] |
| D1 LQR+VMC | 20 | Episode duration | s | 4.283 [3.687, 4.878] | -1.717 [-2.313, -1.122] |
| D1 LQR+VMC | 20 | Mean reward | reward/step | 1.943 [1.801, 2.085] | -1.512 [-1.605, -1.418] |
| D1 LQR+VMC | 20 | Velocity RMSE | m/s | 0.441 [0.377, 0.505] | +0.135 [+0.085, +0.185] |
| D1 LQR+VMC | 20 | Pitch RMSE | deg | 11.204 [10.499, 11.909] | +9.902 [+9.235, +10.569] |
| D1 LQR+VMC | 20 | Maximum absolute pitch | deg | 33.306 [30.039, 36.574] | +29.553 [+26.188, +32.918] |
| D1 LQR+VMC | 20 | Height RMSE | mm | 40.708 [36.315, 45.101] | +30.531 [+26.153, +34.909] |
| D1 LQR+VMC | 20 | Normalized torque RMS | ratio | 0.425 [0.417, 0.433] | +0.190 [+0.179, +0.201] |
| D1 LQR+VMC | 20 | Mean absolute mechanical power | W | 3253.004 [3031.646, 3474.362] | +3077.047 [+2863.807, +3290.286] |
| D1 LQR+VMC | 20 | Torque saturation | ratio | 0.048 [0.043, 0.052] | +0.048 [+0.043, +0.052] |
| D1 LQR+VMC | 20 | Four-wheel contact | ratio | 0.105 [0.079, 0.132] | -0.792 [-0.842, -0.743] |
| D1 LQR+VMC | 20 | Undesired-contact steps | steps | 0.467 [-0.050, 0.983] | +0.467 [-0.050, +0.983] |
| D1 LQR+VMC | 20 | Solve-time P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 20 | Measured state age mean | ms | 19.971 [19.965, 19.977] | +19.971 [+19.965, +19.977] |
| D1 LQR+VMC | 20 | Measured state age P95 | ms | 20.000 [20.000, 20.000] | +20.000 [+20.000, +20.000] |
| D1 LQR+VMC | 20 | Measured state age maximum | ms | 20.000 [20.000, 20.000] | +20.000 [+20.000, +20.000] |
| D1 LQR+VMC | 20 | Compensation horizon P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 20 | Compensation applied | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 20 | Compensation rejected | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 20 | Raw position-estimation RMSE | m | 0.007 [0.006, 0.007] | +0.007 [+0.006, +0.007] |
| D1 LQR+VMC | 20 | Control position-estimation RMSE | m | 0.007 [0.006, 0.007] | +0.007 [+0.006, +0.007] |
| D1 LQR+VMC | 20 | Raw pitch-estimation RMSE | deg | 2.486 [2.364, 2.608] | +2.486 [+2.364, +2.608] |
| D1 LQR+VMC | 20 | Control pitch-estimation RMSE | deg | 2.486 [2.364, 2.608] | +2.486 [+2.364, +2.608] |
| D1 LQR+VMC | 20 | Raw velocity-estimation RMSE | m/s | 0.113 [0.106, 0.120] | +0.113 [+0.106, +0.120] |
| D1 LQR+VMC | 20 | Control velocity-estimation RMSE | m/s | 0.113 [0.106, 0.120] | +0.113 [+0.106, +0.120] |
| D1 LQR+VMC | 20 | Residual-action RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 20 | Residual-action delta RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 20 | Mean residual longitudinal force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 20 | Mean residual vertical force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 30 | Success | ratio | 0.000 [0.000, 0.114] | -1.000 [-1.000, -1.000] |
| D1 LQR+VMC | 30 | Episode duration | s | 1.066 [0.873, 1.259] | -4.934 [-5.127, -4.741] |
| D1 LQR+VMC | 30 | Mean reward | reward/step | 1.944 [1.753, 2.134] | -1.511 [-1.687, -1.335] |
| D1 LQR+VMC | 30 | Velocity RMSE | m/s | 0.432 [0.372, 0.493] | +0.126 [+0.069, +0.182] |
| D1 LQR+VMC | 30 | Pitch RMSE | deg | 11.494 [9.702, 13.285] | +10.192 [+8.361, +12.022] |
| D1 LQR+VMC | 30 | Maximum absolute pitch | deg | 29.193 [24.352, 34.033] | +25.439 [+20.579, +30.299] |
| D1 LQR+VMC | 30 | Height RMSE | mm | 36.697 [30.138, 43.257] | +26.521 [+19.800, +33.241] |
| D1 LQR+VMC | 30 | Normalized torque RMS | ratio | 0.445 [0.430, 0.460] | +0.210 [+0.191, +0.229] |
| D1 LQR+VMC | 30 | Mean absolute mechanical power | W | 3654.881 [3269.459, 4040.302] | +3478.923 [+3087.700, +3870.147] |
| D1 LQR+VMC | 30 | Torque saturation | ratio | 0.072 [0.064, 0.081] | +0.072 [+0.064, +0.081] |
| D1 LQR+VMC | 30 | Four-wheel contact | ratio | 0.206 [0.173, 0.239] | -0.692 [-0.754, -0.629] |
| D1 LQR+VMC | 30 | Undesired-contact steps | steps | 1.667 [0.670, 2.663] | +1.667 [+0.670, +2.663] |
| D1 LQR+VMC | 30 | Solve-time P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 30 | Measured state age mean | ms | 29.662 [29.611, 29.712] | +29.662 [+29.611, +29.712] |
| D1 LQR+VMC | 30 | Measured state age P95 | ms | 30.000 [30.000, 30.000] | +30.000 [+30.000, +30.000] |
| D1 LQR+VMC | 30 | Measured state age maximum | ms | 30.000 [30.000, 30.000] | +30.000 [+30.000, +30.000] |
| D1 LQR+VMC | 30 | Compensation horizon P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 30 | Compensation applied | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 30 | Compensation rejected | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 30 | Raw position-estimation RMSE | m | 0.009 [0.008, 0.010] | +0.009 [+0.008, +0.010] |
| D1 LQR+VMC | 30 | Control position-estimation RMSE | m | 0.009 [0.008, 0.010] | +0.009 [+0.008, +0.010] |
| D1 LQR+VMC | 30 | Raw pitch-estimation RMSE | deg | 3.864 [3.393, 4.336] | +3.864 [+3.393, +4.336] |
| D1 LQR+VMC | 30 | Control pitch-estimation RMSE | deg | 3.864 [3.393, 4.336] | +3.864 [+3.393, +4.336] |
| D1 LQR+VMC | 30 | Raw velocity-estimation RMSE | m/s | 0.188 [0.172, 0.204] | +0.188 [+0.172, +0.204] |
| D1 LQR+VMC | 30 | Control velocity-estimation RMSE | m/s | 0.188 [0.172, 0.204] | +0.188 [+0.172, +0.204] |
| D1 LQR+VMC | 30 | Residual-action RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 30 | Residual-action delta RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 30 | Mean residual longitudinal force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 30 | Mean residual vertical force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 50 | Success | ratio | 0.000 [0.000, 0.114] | -1.000 [-1.000, -1.000] |
| D1 LQR+VMC | 50 | Episode duration | s | 0.800 [0.701, 0.898] | -5.200 [-5.299, -5.102] |
| D1 LQR+VMC | 50 | Mean reward | reward/step | 1.669 [1.450, 1.887] | -1.786 [-2.012, -1.559] |
| D1 LQR+VMC | 50 | Velocity RMSE | m/s | 0.495 [0.428, 0.562] | +0.188 [+0.115, +0.262] |
| D1 LQR+VMC | 50 | Pitch RMSE | deg | 15.105 [12.840, 17.371] | +13.803 [+11.520, +16.086] |
| D1 LQR+VMC | 50 | Maximum absolute pitch | deg | 37.017 [31.647, 42.387] | +33.263 [+27.916, +38.611] |
| D1 LQR+VMC | 50 | Height RMSE | mm | 45.894 [37.400, 54.388] | +35.717 [+27.073, +44.360] |
| D1 LQR+VMC | 50 | Normalized torque RMS | ratio | 0.446 [0.425, 0.467] | +0.211 [+0.186, +0.236] |
| D1 LQR+VMC | 50 | Mean absolute mechanical power | W | 4200.213 [3722.884, 4677.543] | +4024.256 [+3537.815, +4510.697] |
| D1 LQR+VMC | 50 | Torque saturation | ratio | 0.078 [0.066, 0.090] | +0.078 [+0.066, +0.090] |
| D1 LQR+VMC | 50 | Four-wheel contact | ratio | 0.217 [0.178, 0.257] | -0.681 [-0.741, -0.620] |
| D1 LQR+VMC | 50 | Undesired-contact steps | steps | 2.633 [0.948, 4.319] | +2.633 [+0.948, +4.319] |
| D1 LQR+VMC | 50 | Solve-time P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 50 | Measured state age mean | ms | 48.623 [48.470, 48.776] | +48.623 [+48.470, +48.776] |
| D1 LQR+VMC | 50 | Measured state age P95 | ms | 50.000 [50.000, 50.000] | +50.000 [+50.000, +50.000] |
| D1 LQR+VMC | 50 | Measured state age maximum | ms | 50.000 [50.000, 50.000] | +50.000 [+50.000, +50.000] |
| D1 LQR+VMC | 50 | Compensation horizon P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 50 | Compensation applied | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 50 | Compensation rejected | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 50 | Raw position-estimation RMSE | m | 0.019 [0.016, 0.021] | +0.019 [+0.016, +0.021] |
| D1 LQR+VMC | 50 | Control position-estimation RMSE | m | 0.019 [0.016, 0.021] | +0.019 [+0.016, +0.021] |
| D1 LQR+VMC | 50 | Raw pitch-estimation RMSE | deg | 7.709 [6.574, 8.844] | +7.709 [+6.574, +8.844] |
| D1 LQR+VMC | 50 | Control pitch-estimation RMSE | deg | 7.709 [6.574, 8.844] | +7.709 [+6.574, +8.844] |
| D1 LQR+VMC | 50 | Raw velocity-estimation RMSE | m/s | 0.340 [0.306, 0.373] | +0.340 [+0.306, +0.373] |
| D1 LQR+VMC | 50 | Control velocity-estimation RMSE | m/s | 0.340 [0.306, 0.373] | +0.340 [+0.306, +0.373] |
| D1 LQR+VMC | 50 | Residual-action RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 50 | Residual-action delta RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 50 | Mean residual longitudinal force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 50 | Mean residual vertical force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Success | ratio | 1.000 [0.886, 1.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Episode duration | s | 6.000 [6.000, 6.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Mean reward | reward/step | 3.562 [3.480, 3.644] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Velocity RMSE | m/s | 0.292 [0.257, 0.326] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Pitch RMSE | deg | 1.586 [1.389, 1.784] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Maximum absolute pitch | deg | 4.954 [3.970, 5.938] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Height RMSE | mm | 10.572 [9.708, 11.436] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Normalized torque RMS | ratio | 0.237 [0.227, 0.247] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Mean absolute mechanical power | W | 164.340 [117.554, 211.126] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Torque saturation | ratio | 0.000 [-0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Four-wheel contact | ratio | 0.896 [0.851, 0.942] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Undesired-contact steps | steps | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Solve-time P95 | ms | 0.399 [0.377, 0.421] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Measured state age mean | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Measured state age P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Measured state age maximum | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Compensation horizon P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Compensation applied | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Compensation rejected | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Raw position-estimation RMSE | m | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Control position-estimation RMSE | m | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Raw pitch-estimation RMSE | deg | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Control pitch-estimation RMSE | deg | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Raw velocity-estimation RMSE | m/s | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Control velocity-estimation RMSE | m/s | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Residual-action RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Residual-action delta RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Mean residual longitudinal force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 0 | Mean residual vertical force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 10 | Success | ratio | 1.000 [0.886, 1.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 10 | Episode duration | s | 6.000 [6.000, 6.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 10 | Mean reward | reward/step | 3.185 [3.088, 3.283] | -0.376 [-0.440, -0.313] |
| D1 MPC+VMC | 10 | Velocity RMSE | m/s | 0.366 [0.326, 0.405] | +0.074 [+0.052, +0.096] |
| D1 MPC+VMC | 10 | Pitch RMSE | deg | 2.599 [2.335, 2.863] | +1.013 [+0.771, +1.255] |
| D1 MPC+VMC | 10 | Maximum absolute pitch | deg | 7.965 [6.831, 9.098] | +3.011 [+1.912, +4.110] |
| D1 MPC+VMC | 10 | Height RMSE | mm | 12.387 [11.579, 13.195] | +1.815 [+0.873, +2.757] |
| D1 MPC+VMC | 10 | Normalized torque RMS | ratio | 0.306 [0.301, 0.311] | +0.069 [+0.059, +0.080] |
| D1 MPC+VMC | 10 | Mean absolute mechanical power | W | 596.336 [563.723, 628.950] | +431.997 [+383.512, +480.482] |
| D1 MPC+VMC | 10 | Torque saturation | ratio | 0.002 [0.001, 0.003] | +0.002 [+0.001, +0.003] |
| D1 MPC+VMC | 10 | Four-wheel contact | ratio | 0.428 [0.386, 0.469] | -0.469 [-0.507, -0.431] |
| D1 MPC+VMC | 10 | Undesired-contact steps | steps | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 10 | Solve-time P95 | ms | 0.459 [0.419, 0.499] | +0.060 [+0.012, +0.107] |
| D1 MPC+VMC | 10 | Measured state age mean | ms | 10.000 [10.000, 10.000] | +10.000 [+10.000, +10.000] |
| D1 MPC+VMC | 10 | Measured state age P95 | ms | 10.000 [10.000, 10.000] | +10.000 [+10.000, +10.000] |
| D1 MPC+VMC | 10 | Measured state age maximum | ms | 10.000 [10.000, 10.000] | +10.000 [+10.000, +10.000] |
| D1 MPC+VMC | 10 | Compensation horizon P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 10 | Compensation applied | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 10 | Compensation rejected | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 10 | Raw position-estimation RMSE | m | 0.003 [0.002, 0.003] | +0.003 [+0.002, +0.003] |
| D1 MPC+VMC | 10 | Control position-estimation RMSE | m | 0.003 [0.002, 0.003] | +0.003 [+0.002, +0.003] |
| D1 MPC+VMC | 10 | Raw pitch-estimation RMSE | deg | 0.308 [0.286, 0.331] | +0.308 [+0.286, +0.331] |
| D1 MPC+VMC | 10 | Control pitch-estimation RMSE | deg | 0.308 [0.286, 0.331] | +0.308 [+0.286, +0.331] |
| D1 MPC+VMC | 10 | Raw velocity-estimation RMSE | m/s | 0.024 [0.023, 0.025] | +0.024 [+0.023, +0.025] |
| D1 MPC+VMC | 10 | Control velocity-estimation RMSE | m/s | 0.024 [0.023, 0.025] | +0.024 [+0.023, +0.025] |
| D1 MPC+VMC | 10 | Residual-action RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 10 | Residual-action delta RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 10 | Mean residual longitudinal force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 10 | Mean residual vertical force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 20 | Success | ratio | 0.267 [0.142, 0.444] | -0.733 [-0.867, -0.567] |
| D1 MPC+VMC | 20 | Episode duration | s | 3.934 [3.365, 4.503] | -2.066 [-2.635, -1.497] |
| D1 MPC+VMC | 20 | Mean reward | reward/step | 1.968 [1.834, 2.102] | -1.593 [-1.710, -1.477] |
| D1 MPC+VMC | 20 | Velocity RMSE | m/s | 0.427 [0.382, 0.472] | +0.135 [+0.096, +0.174] |
| D1 MPC+VMC | 20 | Pitch RMSE | deg | 11.762 [10.686, 12.838] | +10.176 [+9.131, +11.221] |
| D1 MPC+VMC | 20 | Maximum absolute pitch | deg | 35.625 [31.793, 39.456] | +30.671 [+26.910, +34.431] |
| D1 MPC+VMC | 20 | Height RMSE | mm | 39.124 [35.585, 42.663] | +28.552 [+24.934, +32.170] |
| D1 MPC+VMC | 20 | Normalized torque RMS | ratio | 0.424 [0.417, 0.430] | +0.187 [+0.176, +0.197] |
| D1 MPC+VMC | 20 | Mean absolute mechanical power | W | 3209.181 [3012.601, 3405.761] | +3044.842 [+2858.190, +3231.493] |
| D1 MPC+VMC | 20 | Torque saturation | ratio | 0.047 [0.044, 0.050] | +0.047 [+0.044, +0.050] |
| D1 MPC+VMC | 20 | Four-wheel contact | ratio | 0.115 [0.088, 0.143] | -0.781 [-0.827, -0.734] |
| D1 MPC+VMC | 20 | Undesired-contact steps | steps | 0.600 [0.032, 1.168] | +0.600 [+0.032, +1.168] |
| D1 MPC+VMC | 20 | Solve-time P95 | ms | 0.460 [0.419, 0.500] | +0.060 [+0.012, +0.109] |
| D1 MPC+VMC | 20 | Measured state age mean | ms | 19.970 [19.964, 19.975] | +19.970 [+19.964, +19.975] |
| D1 MPC+VMC | 20 | Measured state age P95 | ms | 20.000 [20.000, 20.000] | +20.000 [+20.000, +20.000] |
| D1 MPC+VMC | 20 | Measured state age maximum | ms | 20.000 [20.000, 20.000] | +20.000 [+20.000, +20.000] |
| D1 MPC+VMC | 20 | Compensation horizon P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 20 | Compensation applied | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 20 | Compensation rejected | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 20 | Raw position-estimation RMSE | m | 0.006 [0.006, 0.007] | +0.006 [+0.006, +0.007] |
| D1 MPC+VMC | 20 | Control position-estimation RMSE | m | 0.006 [0.006, 0.007] | +0.006 [+0.006, +0.007] |
| D1 MPC+VMC | 20 | Raw pitch-estimation RMSE | deg | 2.487 [2.331, 2.643] | +2.487 [+2.331, +2.643] |
| D1 MPC+VMC | 20 | Control pitch-estimation RMSE | deg | 2.487 [2.331, 2.643] | +2.487 [+2.331, +2.643] |
| D1 MPC+VMC | 20 | Raw velocity-estimation RMSE | m/s | 0.115 [0.108, 0.122] | +0.115 [+0.108, +0.122] |
| D1 MPC+VMC | 20 | Control velocity-estimation RMSE | m/s | 0.115 [0.108, 0.122] | +0.115 [+0.108, +0.122] |
| D1 MPC+VMC | 20 | Residual-action RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 20 | Residual-action delta RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 20 | Mean residual longitudinal force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 20 | Mean residual vertical force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 30 | Success | ratio | 0.000 [0.000, 0.114] | -1.000 [-1.000, -1.000] |
| D1 MPC+VMC | 30 | Episode duration | s | 1.180 [0.962, 1.399] | -4.820 [-5.038, -4.601] |
| D1 MPC+VMC | 30 | Mean reward | reward/step | 1.909 [1.753, 2.064] | -1.653 [-1.794, -1.512] |
| D1 MPC+VMC | 30 | Velocity RMSE | m/s | 0.438 [0.383, 0.494] | +0.147 [+0.091, +0.203] |
| D1 MPC+VMC | 30 | Pitch RMSE | deg | 11.629 [9.774, 13.484] | +10.042 [+8.146, +11.939] |
| D1 MPC+VMC | 30 | Maximum absolute pitch | deg | 29.287 [24.189, 34.385] | +24.333 [+18.986, +29.680] |
| D1 MPC+VMC | 30 | Height RMSE | mm | 38.887 [32.292, 45.483] | +28.315 [+21.751, +34.879] |
| D1 MPC+VMC | 30 | Normalized torque RMS | ratio | 0.450 [0.434, 0.466] | +0.213 [+0.194, +0.233] |
| D1 MPC+VMC | 30 | Mean absolute mechanical power | W | 3840.939 [3475.057, 4206.821] | +3676.599 [+3307.175, +4046.023] |
| D1 MPC+VMC | 30 | Torque saturation | ratio | 0.075 [0.065, 0.085] | +0.075 [+0.065, +0.085] |
| D1 MPC+VMC | 30 | Four-wheel contact | ratio | 0.205 [0.167, 0.243] | -0.691 [-0.749, -0.633] |
| D1 MPC+VMC | 30 | Undesired-contact steps | steps | 1.267 [0.429, 2.104] | +1.267 [+0.429, +2.104] |
| D1 MPC+VMC | 30 | Solve-time P95 | ms | 0.576 [0.408, 0.744] | +0.177 [+0.004, +0.349] |
| D1 MPC+VMC | 30 | Measured state age mean | ms | 29.681 [29.625, 29.737] | +29.681 [+29.625, +29.737] |
| D1 MPC+VMC | 30 | Measured state age P95 | ms | 30.000 [30.000, 30.000] | +30.000 [+30.000, +30.000] |
| D1 MPC+VMC | 30 | Measured state age maximum | ms | 30.000 [30.000, 30.000] | +30.000 [+30.000, +30.000] |
| D1 MPC+VMC | 30 | Compensation horizon P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 30 | Compensation applied | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 30 | Compensation rejected | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 30 | Raw position-estimation RMSE | m | 0.010 [0.009, 0.011] | +0.010 [+0.009, +0.011] |
| D1 MPC+VMC | 30 | Control position-estimation RMSE | m | 0.010 [0.009, 0.011] | +0.010 [+0.009, +0.011] |
| D1 MPC+VMC | 30 | Raw pitch-estimation RMSE | deg | 3.758 [3.315, 4.202] | +3.758 [+3.315, +4.202] |
| D1 MPC+VMC | 30 | Control pitch-estimation RMSE | deg | 3.758 [3.315, 4.202] | +3.758 [+3.315, +4.202] |
| D1 MPC+VMC | 30 | Raw velocity-estimation RMSE | m/s | 0.205 [0.185, 0.225] | +0.205 [+0.185, +0.225] |
| D1 MPC+VMC | 30 | Control velocity-estimation RMSE | m/s | 0.205 [0.185, 0.225] | +0.205 [+0.185, +0.225] |
| D1 MPC+VMC | 30 | Residual-action RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 30 | Residual-action delta RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 30 | Mean residual longitudinal force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 30 | Mean residual vertical force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 50 | Success | ratio | 0.000 [0.000, 0.114] | -1.000 [-1.000, -1.000] |
| D1 MPC+VMC | 50 | Episode duration | s | 0.796 [0.698, 0.894] | -5.204 [-5.302, -5.106] |
| D1 MPC+VMC | 50 | Mean reward | reward/step | 1.642 [1.464, 1.820] | -1.919 [-2.077, -1.762] |
| D1 MPC+VMC | 50 | Velocity RMSE | m/s | 0.478 [0.424, 0.531] | +0.186 [+0.137, +0.235] |
| D1 MPC+VMC | 50 | Pitch RMSE | deg | 15.932 [13.757, 18.107] | +14.345 [+12.146, +16.545] |
| D1 MPC+VMC | 50 | Maximum absolute pitch | deg | 37.464 [32.239, 42.690] | +32.511 [+27.280, +37.742] |
| D1 MPC+VMC | 50 | Height RMSE | mm | 46.149 [39.089, 53.209] | +35.577 [+28.575, +42.580] |
| D1 MPC+VMC | 50 | Normalized torque RMS | ratio | 0.447 [0.427, 0.468] | +0.210 [+0.187, +0.234] |
| D1 MPC+VMC | 50 | Mean absolute mechanical power | W | 4260.742 [3784.500, 4736.984] | +4096.402 [+3620.101, +4572.704] |
| D1 MPC+VMC | 50 | Torque saturation | ratio | 0.078 [0.066, 0.090] | +0.078 [+0.066, +0.090] |
| D1 MPC+VMC | 50 | Four-wheel contact | ratio | 0.234 [0.186, 0.281] | -0.663 [-0.720, -0.606] |
| D1 MPC+VMC | 50 | Undesired-contact steps | steps | 1.067 [0.168, 1.965] | +1.067 [+0.168, +1.965] |
| D1 MPC+VMC | 50 | Solve-time P95 | ms | 0.497 [0.452, 0.543] | +0.098 [+0.044, +0.152] |
| D1 MPC+VMC | 50 | Measured state age mean | ms | 48.623 [48.470, 48.775] | +48.623 [+48.470, +48.775] |
| D1 MPC+VMC | 50 | Measured state age P95 | ms | 50.000 [50.000, 50.000] | +50.000 [+50.000, +50.000] |
| D1 MPC+VMC | 50 | Measured state age maximum | ms | 50.000 [50.000, 50.000] | +50.000 [+50.000, +50.000] |
| D1 MPC+VMC | 50 | Compensation horizon P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 50 | Compensation applied | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 50 | Compensation rejected | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 50 | Raw position-estimation RMSE | m | 0.019 [0.017, 0.021] | +0.019 [+0.017, +0.021] |
| D1 MPC+VMC | 50 | Control position-estimation RMSE | m | 0.019 [0.017, 0.021] | +0.019 [+0.017, +0.021] |
| D1 MPC+VMC | 50 | Raw pitch-estimation RMSE | deg | 7.290 [6.207, 8.373] | +7.290 [+6.207, +8.373] |
| D1 MPC+VMC | 50 | Control pitch-estimation RMSE | deg | 7.290 [6.207, 8.373] | +7.290 [+6.207, +8.373] |
| D1 MPC+VMC | 50 | Raw velocity-estimation RMSE | m/s | 0.343 [0.312, 0.374] | +0.343 [+0.312, +0.374] |
| D1 MPC+VMC | 50 | Control velocity-estimation RMSE | m/s | 0.343 [0.312, 0.374] | +0.343 [+0.312, +0.374] |
| D1 MPC+VMC | 50 | Residual-action RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 50 | Residual-action delta RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 50 | Mean residual longitudinal force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 50 | Mean residual vertical force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
