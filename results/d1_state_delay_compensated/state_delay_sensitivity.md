# D1 state-delay sensitivity

Latency compensation: `constant_velocity`.

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
| D1 LQR+VMC | 10 | Mean reward | reward/step | 3.139 [3.040, 3.239] | -0.315 [-0.399, -0.231] |
| D1 LQR+VMC | 10 | Velocity RMSE | m/s | 0.383 [0.338, 0.428] | +0.076 [+0.046, +0.107] |
| D1 LQR+VMC | 10 | Pitch RMSE | deg | 2.270 [2.016, 2.525] | +0.968 [+0.739, +1.197] |
| D1 LQR+VMC | 10 | Maximum absolute pitch | deg | 6.904 [5.693, 8.115] | +3.150 [+2.246, +4.054] |
| D1 LQR+VMC | 10 | Height RMSE | mm | 12.133 [11.253, 13.014] | +1.956 [+0.924, +2.989] |
| D1 LQR+VMC | 10 | Normalized torque RMS | ratio | 0.296 [0.289, 0.304] | +0.062 [+0.052, +0.072] |
| D1 LQR+VMC | 10 | Mean absolute mechanical power | W | 519.110 [477.791, 560.429] | +343.153 [+298.272, +388.033] |
| D1 LQR+VMC | 10 | Torque saturation | ratio | 0.002 [0.001, 0.002] | +0.002 [+0.001, +0.002] |
| D1 LQR+VMC | 10 | Four-wheel contact | ratio | 0.511 [0.456, 0.566] | -0.387 [-0.430, -0.343] |
| D1 LQR+VMC | 10 | Undesired-contact steps | steps | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 10 | Solve-time P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 10 | Measured state age mean | ms | 10.000 [10.000, 10.000] | +10.000 [+10.000, +10.000] |
| D1 LQR+VMC | 10 | Measured state age P95 | ms | 10.000 [10.000, 10.000] | +10.000 [+10.000, +10.000] |
| D1 LQR+VMC | 10 | Measured state age maximum | ms | 10.000 [10.000, 10.000] | +10.000 [+10.000, +10.000] |
| D1 LQR+VMC | 10 | Compensation horizon P95 | ms | 10.000 [10.000, 10.000] | +10.000 [+10.000, +10.000] |
| D1 LQR+VMC | 10 | Compensation applied | ratio | 0.999 [0.998, 1.000] | +0.999 [+0.998, +1.000] |
| D1 LQR+VMC | 10 | Compensation rejected | ratio | 0.001 [0.000, 0.002] | +0.001 [+0.000, +0.002] |
| D1 LQR+VMC | 10 | Raw position-estimation RMSE | m | 0.002 [0.002, 0.003] | +0.002 [+0.002, +0.003] |
| D1 LQR+VMC | 10 | Control position-estimation RMSE | m | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 10 | Raw pitch-estimation RMSE | deg | 0.255 [0.234, 0.275] | +0.255 [+0.234, +0.275] |
| D1 LQR+VMC | 10 | Control pitch-estimation RMSE | deg | 0.033 [0.029, 0.037] | +0.033 [+0.029, +0.037] |
| D1 LQR+VMC | 10 | Raw velocity-estimation RMSE | m/s | 0.022 [0.020, 0.023] | +0.022 [+0.020, +0.023] |
| D1 LQR+VMC | 10 | Control velocity-estimation RMSE | m/s | 0.022 [0.021, 0.023] | +0.022 [+0.021, +0.023] |
| D1 LQR+VMC | 10 | Residual-action RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 10 | Residual-action delta RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 10 | Mean residual longitudinal force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 10 | Mean residual vertical force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 20 | Success | ratio | 0.367 [0.219, 0.545] | -0.633 [-0.800, -0.467] |
| D1 LQR+VMC | 20 | Episode duration | s | 3.931 [3.230, 4.632] | -2.069 [-2.770, -1.368] |
| D1 LQR+VMC | 20 | Mean reward | reward/step | 1.936 [1.786, 2.086] | -1.518 [-1.655, -1.381] |
| D1 LQR+VMC | 20 | Velocity RMSE | m/s | 0.443 [0.386, 0.500] | +0.137 [+0.090, +0.183] |
| D1 LQR+VMC | 20 | Pitch RMSE | deg | 12.388 [11.380, 13.395] | +11.085 [+10.102, +12.069] |
| D1 LQR+VMC | 20 | Maximum absolute pitch | deg | 37.164 [33.776, 40.552] | +33.410 [+29.940, +36.880] |
| D1 LQR+VMC | 20 | Height RMSE | mm | 38.992 [33.613, 44.371] | +28.816 [+23.461, +34.170] |
| D1 LQR+VMC | 20 | Normalized torque RMS | ratio | 0.435 [0.428, 0.442] | +0.200 [+0.189, +0.211] |
| D1 LQR+VMC | 20 | Mean absolute mechanical power | W | 3503.662 [3296.759, 3710.566] | +3327.705 [+3127.811, +3527.600] |
| D1 LQR+VMC | 20 | Torque saturation | ratio | 0.053 [0.049, 0.056] | +0.053 [+0.049, +0.056] |
| D1 LQR+VMC | 20 | Four-wheel contact | ratio | 0.090 [0.065, 0.115] | -0.808 [-0.861, -0.755] |
| D1 LQR+VMC | 20 | Undesired-contact steps | steps | 0.600 [-0.009, 1.209] | +0.600 [-0.009, +1.209] |
| D1 LQR+VMC | 20 | Solve-time P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 20 | Measured state age mean | ms | 19.966 [19.959, 19.973] | +19.966 [+19.959, +19.973] |
| D1 LQR+VMC | 20 | Measured state age P95 | ms | 20.000 [20.000, 20.000] | +20.000 [+20.000, +20.000] |
| D1 LQR+VMC | 20 | Measured state age maximum | ms | 20.000 [20.000, 20.000] | +20.000 [+20.000, +20.000] |
| D1 LQR+VMC | 20 | Compensation horizon P95 | ms | 20.000 [20.000, 20.000] | +20.000 [+20.000, +20.000] |
| D1 LQR+VMC | 20 | Compensation applied | ratio | 0.329 [0.296, 0.362] | +0.329 [+0.296, +0.362] |
| D1 LQR+VMC | 20 | Compensation rejected | ratio | 0.671 [0.638, 0.704] | +0.671 [+0.638, +0.704] |
| D1 LQR+VMC | 20 | Raw position-estimation RMSE | m | 0.007 [0.006, 0.007] | +0.007 [+0.006, +0.007] |
| D1 LQR+VMC | 20 | Control position-estimation RMSE | m | 0.006 [0.005, 0.007] | +0.006 [+0.005, +0.007] |
| D1 LQR+VMC | 20 | Raw pitch-estimation RMSE | deg | 2.583 [2.463, 2.704] | +2.583 [+2.463, +2.704] |
| D1 LQR+VMC | 20 | Control pitch-estimation RMSE | deg | 2.306 [2.174, 2.438] | +2.306 [+2.174, +2.438] |
| D1 LQR+VMC | 20 | Raw velocity-estimation RMSE | m/s | 0.113 [0.106, 0.121] | +0.113 [+0.106, +0.121] |
| D1 LQR+VMC | 20 | Control velocity-estimation RMSE | m/s | 0.114 [0.106, 0.121] | +0.114 [+0.106, +0.121] |
| D1 LQR+VMC | 20 | Residual-action RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 20 | Residual-action delta RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 20 | Mean residual longitudinal force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 20 | Mean residual vertical force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 30 | Success | ratio | 0.000 [0.000, 0.114] | -1.000 [-1.000, -1.000] |
| D1 LQR+VMC | 30 | Episode duration | s | 0.810 [0.684, 0.936] | -5.190 [-5.316, -5.064] |
| D1 LQR+VMC | 30 | Mean reward | reward/step | 2.012 [1.803, 2.221] | -1.442 [-1.648, -1.236] |
| D1 LQR+VMC | 30 | Velocity RMSE | m/s | 0.408 [0.351, 0.465] | +0.101 [+0.040, +0.163] |
| D1 LQR+VMC | 30 | Pitch RMSE | deg | 10.719 [8.467, 12.970] | +9.417 [+7.121, +11.713] |
| D1 LQR+VMC | 30 | Maximum absolute pitch | deg | 26.799 [21.419, 32.179] | +23.045 [+17.569, +28.522] |
| D1 LQR+VMC | 30 | Height RMSE | mm | 34.195 [26.476, 41.914] | +24.018 [+16.138, +31.898] |
| D1 LQR+VMC | 30 | Normalized torque RMS | ratio | 0.439 [0.426, 0.452] | +0.204 [+0.186, +0.222] |
| D1 LQR+VMC | 30 | Mean absolute mechanical power | W | 3468.337 [3094.845, 3841.829] | +3292.380 [+2911.104, +3673.656] |
| D1 LQR+VMC | 30 | Torque saturation | ratio | 0.073 [0.066, 0.079] | +0.073 [+0.066, +0.079] |
| D1 LQR+VMC | 30 | Four-wheel contact | ratio | 0.218 [0.186, 0.250] | -0.679 [-0.741, -0.618] |
| D1 LQR+VMC | 30 | Undesired-contact steps | steps | 1.400 [0.355, 2.445] | +1.400 [+0.355, +2.445] |
| D1 LQR+VMC | 30 | Solve-time P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 30 | Measured state age mean | ms | 29.578 [29.528, 29.629] | +29.578 [+29.528, +29.629] |
| D1 LQR+VMC | 30 | Measured state age P95 | ms | 30.000 [30.000, 30.000] | +30.000 [+30.000, +30.000] |
| D1 LQR+VMC | 30 | Measured state age maximum | ms | 30.000 [30.000, 30.000] | +30.000 [+30.000, +30.000] |
| D1 LQR+VMC | 30 | Compensation horizon P95 | ms | 30.000 [30.000, 30.000] | +30.000 [+30.000, +30.000] |
| D1 LQR+VMC | 30 | Compensation applied | ratio | 0.418 [0.372, 0.464] | +0.418 [+0.372, +0.464] |
| D1 LQR+VMC | 30 | Compensation rejected | ratio | 0.582 [0.536, 0.628] | +0.582 [+0.536, +0.628] |
| D1 LQR+VMC | 30 | Raw position-estimation RMSE | m | 0.010 [0.008, 0.011] | +0.010 [+0.008, +0.011] |
| D1 LQR+VMC | 30 | Control position-estimation RMSE | m | 0.009 [0.008, 0.011] | +0.009 [+0.008, +0.011] |
| D1 LQR+VMC | 30 | Raw pitch-estimation RMSE | deg | 3.331 [2.838, 3.823] | +3.331 [+2.838, +3.823] |
| D1 LQR+VMC | 30 | Control pitch-estimation RMSE | deg | 3.261 [2.759, 3.763] | +3.261 [+2.759, +3.763] |
| D1 LQR+VMC | 30 | Raw velocity-estimation RMSE | m/s | 0.192 [0.171, 0.213] | +0.192 [+0.171, +0.213] |
| D1 LQR+VMC | 30 | Control velocity-estimation RMSE | m/s | 0.192 [0.171, 0.213] | +0.192 [+0.171, +0.213] |
| D1 LQR+VMC | 30 | Residual-action RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 30 | Residual-action delta RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 30 | Mean residual longitudinal force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 30 | Mean residual vertical force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 50 | Success | ratio | 0.000 [0.000, 0.114] | -1.000 [-1.000, -1.000] |
| D1 LQR+VMC | 50 | Episode duration | s | 0.723 [0.607, 0.838] | -5.277 [-5.393, -5.162] |
| D1 LQR+VMC | 50 | Mean reward | reward/step | 1.727 [1.556, 1.897] | -1.728 [-1.890, -1.566] |
| D1 LQR+VMC | 50 | Velocity RMSE | m/s | 0.449 [0.397, 0.500] | +0.142 [+0.093, +0.191] |
| D1 LQR+VMC | 50 | Pitch RMSE | deg | 13.518 [11.643, 15.393] | +12.216 [+10.376, +14.055] |
| D1 LQR+VMC | 50 | Maximum absolute pitch | deg | 34.909 [30.387, 39.431] | +31.155 [+26.676, +35.635] |
| D1 LQR+VMC | 50 | Height RMSE | mm | 41.448 [32.385, 50.511] | +31.271 [+22.192, +40.349] |
| D1 LQR+VMC | 50 | Normalized torque RMS | ratio | 0.453 [0.434, 0.472] | +0.218 [+0.197, +0.239] |
| D1 LQR+VMC | 50 | Mean absolute mechanical power | W | 4205.566 [3757.832, 4653.301] | +4029.609 [+3589.204, +4470.014] |
| D1 LQR+VMC | 50 | Torque saturation | ratio | 0.085 [0.074, 0.096] | +0.085 [+0.074, +0.096] |
| D1 LQR+VMC | 50 | Four-wheel contact | ratio | 0.191 [0.156, 0.226] | -0.707 [-0.758, -0.656] |
| D1 LQR+VMC | 50 | Undesired-contact steps | steps | 2.400 [1.246, 3.554] | +2.400 [+1.246, +3.554] |
| D1 LQR+VMC | 50 | Solve-time P95 | ms | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 LQR+VMC | 50 | Measured state age mean | ms | 48.450 [48.271, 48.628] | +48.450 [+48.271, +48.628] |
| D1 LQR+VMC | 50 | Measured state age P95 | ms | 50.000 [50.000, 50.000] | +50.000 [+50.000, +50.000] |
| D1 LQR+VMC | 50 | Measured state age maximum | ms | 50.000 [50.000, 50.000] | +50.000 [+50.000, +50.000] |
| D1 LQR+VMC | 50 | Compensation horizon P95 | ms | 50.000 [50.000, 50.000] | +50.000 [+50.000, +50.000] |
| D1 LQR+VMC | 50 | Compensation applied | ratio | 0.390 [0.346, 0.435] | +0.390 [+0.346, +0.435] |
| D1 LQR+VMC | 50 | Compensation rejected | ratio | 0.610 [0.565, 0.654] | +0.610 [+0.565, +0.654] |
| D1 LQR+VMC | 50 | Raw position-estimation RMSE | m | 0.017 [0.015, 0.019] | +0.017 [+0.015, +0.019] |
| D1 LQR+VMC | 50 | Control position-estimation RMSE | m | 0.017 [0.014, 0.019] | +0.017 [+0.014, +0.019] |
| D1 LQR+VMC | 50 | Raw pitch-estimation RMSE | deg | 6.688 [5.954, 7.423] | +6.688 [+5.954, +7.423] |
| D1 LQR+VMC | 50 | Control pitch-estimation RMSE | deg | 6.595 [5.863, 7.328] | +6.595 [+5.863, +7.328] |
| D1 LQR+VMC | 50 | Raw velocity-estimation RMSE | m/s | 0.324 [0.291, 0.357] | +0.324 [+0.291, +0.357] |
| D1 LQR+VMC | 50 | Control velocity-estimation RMSE | m/s | 0.324 [0.291, 0.357] | +0.324 [+0.291, +0.357] |
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
| D1 MPC+VMC | 0 | Solve-time P95 | ms | 0.424 [0.390, 0.459] | +0.000 [+0.000, +0.000] |
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
| D1 MPC+VMC | 10 | Mean reward | reward/step | 3.286 [3.191, 3.381] | -0.275 [-0.337, -0.213] |
| D1 MPC+VMC | 10 | Velocity RMSE | m/s | 0.338 [0.300, 0.376] | +0.046 [+0.026, +0.067] |
| D1 MPC+VMC | 10 | Pitch RMSE | deg | 2.224 [1.955, 2.493] | +0.638 [+0.402, +0.873] |
| D1 MPC+VMC | 10 | Maximum absolute pitch | deg | 7.046 [5.715, 8.378] | +2.093 [+0.810, +3.376] |
| D1 MPC+VMC | 10 | Height RMSE | mm | 11.931 [11.013, 12.849] | +1.359 [+0.206, +2.512] |
| D1 MPC+VMC | 10 | Normalized torque RMS | ratio | 0.298 [0.290, 0.306] | +0.061 [+0.050, +0.072] |
| D1 MPC+VMC | 10 | Mean absolute mechanical power | W | 519.104 [474.439, 563.769] | +354.764 [+307.062, +402.466] |
| D1 MPC+VMC | 10 | Torque saturation | ratio | 0.001 [0.001, 0.002] | +0.001 [+0.001, +0.002] |
| D1 MPC+VMC | 10 | Four-wheel contact | ratio | 0.514 [0.457, 0.571] | -0.382 [-0.429, -0.336] |
| D1 MPC+VMC | 10 | Undesired-contact steps | steps | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 10 | Solve-time P95 | ms | 0.427 [0.406, 0.448] | +0.002 [-0.041, +0.046] |
| D1 MPC+VMC | 10 | Measured state age mean | ms | 10.000 [10.000, 10.000] | +10.000 [+10.000, +10.000] |
| D1 MPC+VMC | 10 | Measured state age P95 | ms | 10.000 [10.000, 10.000] | +10.000 [+10.000, +10.000] |
| D1 MPC+VMC | 10 | Measured state age maximum | ms | 10.000 [10.000, 10.000] | +10.000 [+10.000, +10.000] |
| D1 MPC+VMC | 10 | Compensation horizon P95 | ms | 10.000 [10.000, 10.000] | +10.000 [+10.000, +10.000] |
| D1 MPC+VMC | 10 | Compensation applied | ratio | 0.999 [0.999, 1.000] | +0.999 [+0.999, +1.000] |
| D1 MPC+VMC | 10 | Compensation rejected | ratio | 0.001 [-0.000, 0.001] | +0.001 [-0.000, +0.001] |
| D1 MPC+VMC | 10 | Raw position-estimation RMSE | m | 0.003 [0.002, 0.003] | +0.003 [+0.002, +0.003] |
| D1 MPC+VMC | 10 | Control position-estimation RMSE | m | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 10 | Raw pitch-estimation RMSE | deg | 0.241 [0.222, 0.260] | +0.241 [+0.222, +0.260] |
| D1 MPC+VMC | 10 | Control pitch-estimation RMSE | deg | 0.032 [0.029, 0.035] | +0.032 [+0.029, +0.035] |
| D1 MPC+VMC | 10 | Raw velocity-estimation RMSE | m/s | 0.021 [0.020, 0.022] | +0.021 [+0.020, +0.022] |
| D1 MPC+VMC | 10 | Control velocity-estimation RMSE | m/s | 0.021 [0.020, 0.023] | +0.021 [+0.020, +0.023] |
| D1 MPC+VMC | 10 | Residual-action RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 10 | Residual-action delta RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 10 | Mean residual longitudinal force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 10 | Mean residual vertical force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 20 | Success | ratio | 0.367 [0.219, 0.545] | -0.633 [-0.800, -0.467] |
| D1 MPC+VMC | 20 | Episode duration | s | 4.414 [3.770, 5.059] | -1.586 [-2.230, -0.941] |
| D1 MPC+VMC | 20 | Mean reward | reward/step | 1.951 [1.806, 2.096] | -1.610 [-1.736, -1.485] |
| D1 MPC+VMC | 20 | Velocity RMSE | m/s | 0.443 [0.396, 0.491] | +0.151 [+0.107, +0.196] |
| D1 MPC+VMC | 20 | Pitch RMSE | deg | 11.403 [10.372, 12.433] | +9.817 [+8.849, +10.785] |
| D1 MPC+VMC | 20 | Maximum absolute pitch | deg | 32.436 [28.860, 36.013] | +27.483 [+23.980, +30.985] |
| D1 MPC+VMC | 20 | Height RMSE | mm | 36.568 [33.354, 39.782] | +25.996 [+23.109, +28.883] |
| D1 MPC+VMC | 20 | Normalized torque RMS | ratio | 0.433 [0.429, 0.437] | +0.196 [+0.185, +0.207] |
| D1 MPC+VMC | 20 | Mean absolute mechanical power | W | 3394.564 [3253.880, 3535.249] | +3230.225 [+3094.127, +3366.322] |
| D1 MPC+VMC | 20 | Torque saturation | ratio | 0.051 [0.047, 0.054] | +0.051 [+0.047, +0.054] |
| D1 MPC+VMC | 20 | Four-wheel contact | ratio | 0.070 [0.056, 0.084] | -0.826 [-0.874, -0.779] |
| D1 MPC+VMC | 20 | Undesired-contact steps | steps | 0.267 [-0.099, 0.633] | +0.267 [-0.099, +0.633] |
| D1 MPC+VMC | 20 | Solve-time P95 | ms | 0.455 [0.423, 0.487] | +0.031 [-0.012, +0.073] |
| D1 MPC+VMC | 20 | Measured state age mean | ms | 19.971 [19.965, 19.978] | +19.971 [+19.965, +19.978] |
| D1 MPC+VMC | 20 | Measured state age P95 | ms | 20.000 [20.000, 20.000] | +20.000 [+20.000, +20.000] |
| D1 MPC+VMC | 20 | Measured state age maximum | ms | 20.000 [20.000, 20.000] | +20.000 [+20.000, +20.000] |
| D1 MPC+VMC | 20 | Compensation horizon P95 | ms | 20.000 [20.000, 20.000] | +20.000 [+20.000, +20.000] |
| D1 MPC+VMC | 20 | Compensation applied | ratio | 0.314 [0.287, 0.340] | +0.314 [+0.287, +0.340] |
| D1 MPC+VMC | 20 | Compensation rejected | ratio | 0.686 [0.660, 0.713] | +0.686 [+0.660, +0.713] |
| D1 MPC+VMC | 20 | Raw position-estimation RMSE | m | 0.006 [0.006, 0.007] | +0.006 [+0.006, +0.007] |
| D1 MPC+VMC | 20 | Control position-estimation RMSE | m | 0.005 [0.005, 0.006] | +0.005 [+0.005, +0.006] |
| D1 MPC+VMC | 20 | Raw pitch-estimation RMSE | deg | 2.435 [2.285, 2.584] | +2.435 [+2.285, +2.584] |
| D1 MPC+VMC | 20 | Control pitch-estimation RMSE | deg | 2.208 [2.049, 2.366] | +2.208 [+2.049, +2.366] |
| D1 MPC+VMC | 20 | Raw velocity-estimation RMSE | m/s | 0.114 [0.110, 0.119] | +0.114 [+0.110, +0.119] |
| D1 MPC+VMC | 20 | Control velocity-estimation RMSE | m/s | 0.115 [0.110, 0.119] | +0.115 [+0.110, +0.119] |
| D1 MPC+VMC | 20 | Residual-action RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 20 | Residual-action delta RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 20 | Mean residual longitudinal force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 20 | Mean residual vertical force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 30 | Success | ratio | 0.000 [0.000, 0.114] | -1.000 [-1.000, -1.000] |
| D1 MPC+VMC | 30 | Episode duration | s | 1.036 [0.869, 1.203] | -4.964 [-5.131, -4.797] |
| D1 MPC+VMC | 30 | Mean reward | reward/step | 1.915 [1.753, 2.077] | -1.647 [-1.805, -1.488] |
| D1 MPC+VMC | 30 | Velocity RMSE | m/s | 0.412 [0.365, 0.459] | +0.120 [+0.069, +0.171] |
| D1 MPC+VMC | 30 | Pitch RMSE | deg | 11.835 [9.766, 13.904] | +10.248 [+8.139, +12.358] |
| D1 MPC+VMC | 30 | Maximum absolute pitch | deg | 30.102 [24.757, 35.448] | +25.149 [+19.869, +30.428] |
| D1 MPC+VMC | 30 | Height RMSE | mm | 46.390 [36.544, 56.236] | +35.818 [+25.655, +45.981] |
| D1 MPC+VMC | 30 | Normalized torque RMS | ratio | 0.459 [0.443, 0.474] | +0.222 [+0.203, +0.240] |
| D1 MPC+VMC | 30 | Mean absolute mechanical power | W | 3883.304 [3449.542, 4317.066] | +3718.964 [+3291.465, +4146.464] |
| D1 MPC+VMC | 30 | Torque saturation | ratio | 0.082 [0.073, 0.092] | +0.082 [+0.073, +0.092] |
| D1 MPC+VMC | 30 | Four-wheel contact | ratio | 0.187 [0.151, 0.222] | -0.710 [-0.761, -0.658] |
| D1 MPC+VMC | 30 | Undesired-contact steps | steps | 2.500 [0.917, 4.083] | +2.500 [+0.917, +4.083] |
| D1 MPC+VMC | 30 | Solve-time P95 | ms | 0.527 [0.484, 0.570] | +0.102 [+0.052, +0.153] |
| D1 MPC+VMC | 30 | Measured state age mean | ms | 29.651 [29.595, 29.707] | +29.651 [+29.595, +29.707] |
| D1 MPC+VMC | 30 | Measured state age P95 | ms | 30.000 [30.000, 30.000] | +30.000 [+30.000, +30.000] |
| D1 MPC+VMC | 30 | Measured state age maximum | ms | 30.000 [30.000, 30.000] | +30.000 [+30.000, +30.000] |
| D1 MPC+VMC | 30 | Compensation horizon P95 | ms | 30.000 [30.000, 30.000] | +30.000 [+30.000, +30.000] |
| D1 MPC+VMC | 30 | Compensation applied | ratio | 0.351 [0.294, 0.408] | +0.351 [+0.294, +0.408] |
| D1 MPC+VMC | 30 | Compensation rejected | ratio | 0.649 [0.592, 0.706] | +0.649 [+0.592, +0.706] |
| D1 MPC+VMC | 30 | Raw position-estimation RMSE | m | 0.011 [0.010, 0.013] | +0.011 [+0.010, +0.013] |
| D1 MPC+VMC | 30 | Control position-estimation RMSE | m | 0.011 [0.009, 0.013] | +0.011 [+0.009, +0.013] |
| D1 MPC+VMC | 30 | Raw pitch-estimation RMSE | deg | 3.651 [3.196, 4.106] | +3.651 [+3.196, +4.106] |
| D1 MPC+VMC | 30 | Control pitch-estimation RMSE | deg | 3.593 [3.128, 4.058] | +3.593 [+3.128, +4.058] |
| D1 MPC+VMC | 30 | Raw velocity-estimation RMSE | m/s | 0.216 [0.192, 0.240] | +0.216 [+0.192, +0.240] |
| D1 MPC+VMC | 30 | Control velocity-estimation RMSE | m/s | 0.216 [0.191, 0.240] | +0.216 [+0.191, +0.240] |
| D1 MPC+VMC | 30 | Residual-action RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 30 | Residual-action delta RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 30 | Mean residual longitudinal force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 30 | Mean residual vertical force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 50 | Success | ratio | 0.000 [0.000, 0.114] | -1.000 [-1.000, -1.000] |
| D1 MPC+VMC | 50 | Episode duration | s | 0.778 [0.688, 0.869] | -5.222 [-5.312, -5.131] |
| D1 MPC+VMC | 50 | Mean reward | reward/step | 1.713 [1.532, 1.894] | -1.849 [-2.014, -1.683] |
| D1 MPC+VMC | 50 | Velocity RMSE | m/s | 0.435 [0.386, 0.484] | +0.143 [+0.094, +0.193] |
| D1 MPC+VMC | 50 | Pitch RMSE | deg | 14.986 [13.244, 16.727] | +13.399 [+11.745, +15.053] |
| D1 MPC+VMC | 50 | Maximum absolute pitch | deg | 35.441 [30.961, 39.921] | +30.487 [+26.132, +34.842] |
| D1 MPC+VMC | 50 | Height RMSE | mm | 45.022 [35.763, 54.282] | +34.450 [+25.250, +43.651] |
| D1 MPC+VMC | 50 | Normalized torque RMS | ratio | 0.465 [0.448, 0.481] | +0.228 [+0.208, +0.248] |
| D1 MPC+VMC | 50 | Mean absolute mechanical power | W | 4575.846 [4105.250, 5046.442] | +4411.506 [+3940.017, +4882.995] |
| D1 MPC+VMC | 50 | Torque saturation | ratio | 0.090 [0.080, 0.101] | +0.090 [+0.080, +0.101] |
| D1 MPC+VMC | 50 | Four-wheel contact | ratio | 0.179 [0.150, 0.209] | -0.717 [-0.765, -0.670] |
| D1 MPC+VMC | 50 | Undesired-contact steps | steps | 3.067 [1.596, 4.537] | +3.067 [+1.596, +4.537] |
| D1 MPC+VMC | 50 | Solve-time P95 | ms | 0.502 [0.442, 0.562] | +0.077 [+0.006, +0.148] |
| D1 MPC+VMC | 50 | Measured state age mean | ms | 48.596 [48.440, 48.752] | +48.596 [+48.440, +48.752] |
| D1 MPC+VMC | 50 | Measured state age P95 | ms | 50.000 [50.000, 50.000] | +50.000 [+50.000, +50.000] |
| D1 MPC+VMC | 50 | Measured state age maximum | ms | 50.000 [50.000, 50.000] | +50.000 [+50.000, +50.000] |
| D1 MPC+VMC | 50 | Compensation horizon P95 | ms | 50.000 [50.000, 50.000] | +50.000 [+50.000, +50.000] |
| D1 MPC+VMC | 50 | Compensation applied | ratio | 0.360 [0.319, 0.401] | +0.360 [+0.319, +0.401] |
| D1 MPC+VMC | 50 | Compensation rejected | ratio | 0.640 [0.599, 0.681] | +0.640 [+0.599, +0.681] |
| D1 MPC+VMC | 50 | Raw position-estimation RMSE | m | 0.019 [0.016, 0.021] | +0.019 [+0.016, +0.021] |
| D1 MPC+VMC | 50 | Control position-estimation RMSE | m | 0.018 [0.016, 0.021] | +0.018 [+0.016, +0.021] |
| D1 MPC+VMC | 50 | Raw pitch-estimation RMSE | deg | 6.829 [6.001, 7.656] | +6.829 [+6.001, +7.656] |
| D1 MPC+VMC | 50 | Control pitch-estimation RMSE | deg | 6.714 [5.889, 7.539] | +6.714 [+5.889, +7.539] |
| D1 MPC+VMC | 50 | Raw velocity-estimation RMSE | m/s | 0.347 [0.316, 0.378] | +0.347 [+0.316, +0.378] |
| D1 MPC+VMC | 50 | Control velocity-estimation RMSE | m/s | 0.347 [0.316, 0.378] | +0.347 [+0.316, +0.378] |
| D1 MPC+VMC | 50 | Residual-action RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 50 | Residual-action delta RMS | ratio | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 50 | Mean residual longitudinal force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| D1 MPC+VMC | 50 | Mean residual vertical force | N | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
