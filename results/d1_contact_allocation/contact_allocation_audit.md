# D1 contact-allocation matched-seed audit

Formal — the preregistered simulation promotion gate did not pass.

Each seed replays the same initial state, command, randomized domain, and push schedule. The treatment is the full mode-specific control path, not the allocator in isolation: the contact allocator, the allocator-matched identified sagittal linear model used by the outer-loop controller, and the mode-specific fixed outer-loop LQR weight R are all changed together between legacy and constrained. This is not an allocator-only causal ablation — differences between the two modes reflect this paired allocator/model/R control-path change, not the allocator alone.

The physics-side errors use the MuJoCo solver contact wrench, averaged over physics substeps. Force and moment are reported separately in N and Nm; they are never combined into a dimensionally invalid score.

Solver outcome and wrench tracking are independent diagnostics. Feasible nonconverged candidates remain visible and count toward the solver-degraded gate. Wrench tracking uses tolerances of 1 N + 0.5% of requested force norm and 0.5 Nm + 0.5% of requested moment norm.

| Metric | Unit | Legacy mean | Constrained mean | Paired delta | Delta 95% t CI |
|---|---:|---:|---:|---:|---|
| Episode success | ratio | 1 | 1 | +0 | [+0, +0] |
| Velocity RMSE | m/s | 0.316342 | 0.239872 | -0.07647 | [-0.1038, -0.04915] |
| Pitch RMSE | deg | 1.43023 | 2.42907 | +0.99884 | [+0.7473, +1.25] |
| Maximum absolute pitch | deg | 4.00301 | 5.03961 | +1.0366 | [+0.3212, +1.752] |
| Height RMSE | mm | 9.96729 | 12.1117 | +2.14439 | [+0.7505, +3.538] |
| Normalized torque RMS | ratio | 0.245144 | 0.235851 | -0.00929329 | [-0.02273, +0.004147] |
| Torque saturation | ratio | 0.000121528 | 0.00355556 | +0.00343403 | [+0.001391, +0.005477] |
| Four-wheel contact | ratio | 0.867 | 0.673889 | -0.193111 | [-0.2889, -0.09736] |
| Partial wheel contact | ratio | 0.132778 | 0.324556 | +0.191778 | [+0.09619, +0.2874] |
| Undesired-contact steps | steps | 0 | 0 | +0 | [+0, +0] |
| Mean absolute mechanical power | W | 223.058 | 163.173 | -59.8847 | [-107.6, -12.19] |
| Allocation solve-time P99 | ms | 0.29837 | 3.91855 | +3.62018 | [+3.305, +3.936] |
| Allocation maximum constraint violation | ratio | 0.0228492 | 9.87724e-11 | -0.0228492 | [-0.05362, +0.00792] |
| Requested-force residual RMS | N | 1.02473 | 62.4232 | +61.3985 | [+50.55, +72.24] |
| Requested-moment residual RMS | Nm | 65.9397 | 8.14799 | -57.7918 | [-59.48, -56.11] |
| Allocated-to-physics force discrepancy RMS | N | 137.949 | 62.4536 | -75.4956 | [-81.43, -69.56] |
| Allocated-to-physics moment discrepancy RMS | Nm | 41.1658 | 15.3064 | -25.8594 | [-27.94, -23.78] |
| Requested-to-physics force error RMS | N | 138.001 | 88.6351 | -49.3654 | [-59.29, -39.44] |
| Requested-to-physics moment error RMS | Nm | 30.4619 | 16.4518 | -14.0101 | [-16.21, -11.81] |
| Allocation solver converged | ratio | 0 | 0.991722 | +0.991722 | [+0.9885, +0.995] |
| Allocation feasible but nonconverged | ratio | 0 | 0.00616667 | +0.00616667 | [+0.0031, +0.009234] |
| Allocation fallback | ratio | 0 | 0 | +0 | [+0, +0] |
| Allocation wrench limited | ratio | 0.9995 | 0.235167 | -0.764333 | [-0.8345, -0.6941] |
| Allocation no-contact | ratio | 0 | 0.00211111 | +0.00211111 | [+0.001173, +0.003049] |

## Promotion checks

| Check | Result | Criterion |
|---|---:|---|
| preregistered seed pool | pass | evaluation_seeds must equal the preregistered held-out seeds 121..150 |
| formal sample size | pass | >= 30 matched episodes |
| success noninferiority | pass | constrained minus legacy success >= 0 |
| force tracking reduction | pass | >= 10% reduction in physics-side force error |
| moment tracking reduction | pass | >= 10% reduction in physics-side moment error |
| constraint feasibility | pass | <= 1e-06 |
| fallback rate | pass | <= 0.01 |
| solver degraded rate | fail | per-episode feasible-nonconverged plus fallback ratio <= 0.01 |
| real time budget | pass | episode P99 allocation time <= one control period |
| physics wrench available | pass | exactly 5 MuJoCo physics contact-wrench samples per control step |

Passing this gate supports promoting the constrained allocator as the default simulation baseline. It is not hardware validation and does not establish a full whole-body controller.
