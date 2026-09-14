# G1 command-domain probe: no complete driving policy passed

The fixed 24-episode matrix completed all 22,400 control transitions (112,000 real MuJoCo substeps), with no early termination. Zero residual and two trained policies pass only the two forward yaw-impulse cases. No branch passes the complete stop or turn case gates. These results do not support selecting a PPO policy as the default for full keyboard driving.

| Fixed case | Zero | Seed 49001, 65k | Seed 49002, 65k | Seed 49003, 65k |
|---|---|---|---|---|
| flat_forward_stop | FAIL | FAIL | FAIL | FAIL |
| flat_reverse_stop | FAIL | FAIL | FAIL | FAIL |
| stationary_turn_left_hold | FAIL | FAIL | FAIL | FAIL |
| stationary_turn_right_hold | FAIL | FAIL | FAIL | FAIL |
| forward_positive_yaw_impulse | PASS | PASS | FAIL | PASS |
| forward_negative_yaw_impulse | PASS | PASS | FAIL | PASS |

The three final checkpoints remain exactly the declared 65,536-step models. There was no training, action transformation, threshold change, policy selection, reset after termination, parameter scan or repeat of the old 32 s road evaluation. All six cases are flat development probes; reverse, stopping and nonzero user-yaw command phases were absent from the training command distribution.

## Detailed measured gates

| Case | Model | Forward-speed RMSE (m/s) | Heading peak (rad) | Height RMSE (m) | Failed gates |
|---|---|---:|---:|---:|---|
| flat_forward_stop | zero | 0.059544 | 0.001597 | 0.005760 | late_stop_speed, late_stop_planar_path, velocity_rmse |
| flat_forward_stop | seed49001_step65536 | 0.061997 | 0.026510 | 0.011447 | late_stop_speed, late_stop_planar_path, velocity_rmse |
| flat_forward_stop | seed49002_step65536 | 0.114367 | 0.142311 | 0.006799 | heading_peak, late_stop_speed, late_stop_planar_path, velocity_rmse |
| flat_forward_stop | seed49003_step65536 | 0.059449 | 0.039473 | 0.014511 | late_stop_speed, late_stop_planar_path, velocity_rmse |
| flat_reverse_stop | zero | 0.059917 | 0.002411 | 0.007338 | late_stop_speed, late_stop_planar_path, velocity_rmse |
| flat_reverse_stop | seed49001_step65536 | 0.060914 | 0.026510 | 0.011700 | late_stop_speed, late_stop_planar_path, velocity_rmse |
| flat_reverse_stop | seed49002_step65536 | 0.083254 | 0.025254 | 0.006932 | late_stop_speed, late_stop_planar_path, velocity_rmse |
| flat_reverse_stop | seed49003_step65536 | 0.058109 | 0.039473 | 0.015095 | height_rmse, late_stop_speed, late_stop_planar_path, velocity_rmse |
| stationary_turn_left_hold | zero | 0.009607 | 0.252353 | 0.007343 | heading_peak |
| stationary_turn_left_hold | seed49001_step65536 | 0.033373 | 0.238865 | 0.007199 | heading_peak, turn_late_heading, turn_late_speed, turn_planar_displacement |
| stationary_turn_left_hold | seed49002_step65536 | 0.052725 | 0.154502 | 0.008876 | heading_peak, turn_late_speed, turn_planar_displacement |
| stationary_turn_left_hold | seed49003_step65536 | 0.030555 | 0.286317 | 0.010410 | heading_peak, turn_late_heading, turn_late_speed, turn_planar_displacement |
| stationary_turn_right_hold | zero | 0.009906 | 0.252351 | 0.007499 | heading_peak |
| stationary_turn_right_hold | seed49001_step65536 | 0.032540 | 0.200340 | 0.012088 | heading_peak, turn_late_speed, turn_planar_displacement |
| stationary_turn_right_hold | seed49002_step65536 | 0.048598 | 0.182040 | 0.007626 | heading_peak, turn_late_speed, turn_planar_displacement |
| stationary_turn_right_hold | seed49003_step65536 | 0.035017 | 0.190588 | 0.015015 | height_rmse, heading_peak, turn_late_speed, turn_planar_displacement |
| forward_positive_yaw_impulse | zero | 0.030374 | 0.001924 | 0.004369 | none |
| forward_positive_yaw_impulse | seed49001_step65536 | 0.039716 | 0.026510 | 0.011541 | none |
| forward_positive_yaw_impulse | seed49002_step65536 | 0.051668 | 0.025254 | 0.009057 | velocity_rmse |
| forward_positive_yaw_impulse | seed49003_step65536 | 0.038601 | 0.039473 | 0.012521 | none |
| forward_negative_yaw_impulse | zero | 0.030274 | 0.002029 | 0.004362 | none |
| forward_negative_yaw_impulse | seed49001_step65536 | 0.039615 | 0.026510 | 0.011564 | none |
| forward_negative_yaw_impulse | seed49002_step65536 | 0.051554 | 0.025254 | 0.009058 | velocity_rmse |
| forward_negative_yaw_impulse | seed49003_step65536 | 0.038551 | 0.039473 | 0.012525 | none |

Forward stopping: even zero exceeds the late-speed limit (0.060869 versus 0.03 m/s) and late cumulative planar-path limit (0.056694 versus 0.05 m). The three PPO branches have late peaks 0.084691, 0.242648 and 0.069368 m/s. Reverse stopping also fails the complete gate for every branch. Some individual reverse metrics improve for learned policies, but none establishes reliable stopping.

Stationary Q/E: zero reaches late heading error below 0.03 rad, late body-forward speed below 0.018 m/s and planar displacement below 0.071 m for both directions, but its transient heading error peaks around 0.25235 rad against the frozen 0.08727 rad limit. Every PPO turning case additionally fails late velocity and planar displacement; some also fail late heading or height. A good late holding value must not erase the transient failure.

Forward yaw impulses: zero passes both signs with speed RMSE about 0.0303 m/s. Seeds49001/49003 also pass but have speed RMSE about 0.0397/0.0386 m/s and larger heading errors. Seed49002 exceeds the 0.05 m/s speed RMSE limit for both signs. This small prescribed impulse is bounded evidence, not broad robustness.

## Actual disturbance and timing audit

Each of the eight formal impulse episodes records exactly 20 perturbed control intervals × 5 native physics substeps. The mj_step-entry observer confirms world-z torque ±0.5 N·m and actual dt≈0.002 s, integrating to ±0.09999999999998899 N·m·s. All other bodies receive zero external wrench; the original plant method, original mj_step and zero wrench state are restored on exit.

The first protocol draft incorrectly proposed setting xfrc_applied before env.step. Source inspection showed that the unchanged plant clears that array each substep. The pre-execution revision uses the existing push_torque_world_nm API through an instance method bridge; the original draft is retained as heading_g1_protocol.pre_injection_api_fix.json. Revision2 SHA256 is cf5dbd51042bfafa163116f4a7fb2a990726d47c5a6000558f28aa6866a664fc. No formal case used the ineffective injection path.

Speed/heading gate windows use exact state endpoint ticks [start,end). The late stop path uses positions500..700 inclusive, summing200 actual planar intervals; it does not use net displacement. Seven focused tests guard timing, signs, missing-window rejection and oscillatory path length. Ruff passed for the new runner/tests. Astra independently reviewed the command, perturbation and endpoint gate code.

A separate two-tick interface smoke initially completed its physics but failed report creation with a KeyError on the reset metadata wrapper. The saved trace/states and execution_failure.json remain in heading_g1_interface_smoke_01. Reading env.episode_metadata fixed it; heading_g1_interface_smoke_02 completed two ticks, ten physics substeps and +0.01 N·m·s. These four total interface-check control transitions are excluded from the 22,400 formal transitions.

The independent read-only auditor verifies every manifest file; 85D observations, all eight requested/applied actions, same initial state/observation hashes, quaternion attitude, heading-reference integration, reward decomposition, every actual substep wrench, late-window values and all18 strict policy-versus-zero common prefixes. It performed zero new physics steps. See heading_g1_audit_01/report.json and audit_heading_g1_records.py.

## Files and reproduction

- Actual runner: scripts/probe_d1_heading_g1.py; tests: tests/test_d1_heading_g1_probe.py.
- Frozen protocol and revision history: heading_g1_protocol.json and heading_g1_protocol.pre_injection_api_fix.json.
- Full trace/state/summary/manifest tree: heading_g1_01/.
- Independent recomputation: heading_g1_audit_01/report.json.

The CLI supports --models-root to map the recorded labels into another training directory while retaining the protocol and model/sidecar SHA checks. The root must contain seed49001/checkpoints/step65536/model.zip and matching metadata, likewise49002 and49003. The original absolute work paths need not exist when this option is provided.

```bash
rtk proxy env PYTHONPATH=.local-deps:src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B -m scripts.probe_d1_heading_g1 --protocol PROTOCOL_JSON --output NEW_DIRECTORY --models-root TRAINING_ROOT
```

This is a task-domain development milestone. GUI parity, broader terrains and a human drive remain separate evidence. No passed subset is presented as a complete G1 pass.
