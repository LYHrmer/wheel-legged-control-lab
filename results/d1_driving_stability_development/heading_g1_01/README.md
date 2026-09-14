# G1 command-domain checks: parking remains unresolved

**Zero residual also fails the parking gates. No tested branch passes complete keyboard-driving G1.** All 24 formal episodes completed: 22,400 control transitions and 112,000 real physics substeps. There was no new training, model selection, gain change, parameter scan, hidden reset or padding. The results do not justify making a PPO policy the default for stopping, reverse driving or turning.

| Fixed development case | Zero | Seed 49001 / 65k | Seed 49002 / 65k | Seed 49003 / 65k |
|---|---|---|---|---|
| Forward, then stop | FAIL | FAIL | FAIL | FAIL |
| Reverse, then stop | FAIL | FAIL | FAIL | FAIL |
| Stationary left turn, then hold | FAIL | FAIL | FAIL | FAIL |
| Stationary right turn, then hold | FAIL | FAIL | FAIL | FAIL |
| Forward with positive yaw impulse | PASS | PASS | FAIL | PASS |
| Forward with negative yaw impulse | PASS | PASS | FAIL | PASS |

All failures and every per-case gate are retained in the [full report](detailed_report.md) and [complete summary](episodes/summary.json). The three checkpoints are the declared **65,536-step** policies. Reverse, stopping and nonzero user-yaw commands were absent from the training command distribution. These six cases isolate command behavior on flat terrain; the prior 32 s road evaluation remains separate.

For forward stopping, zero's late peak speed is **0.060869 m/s** against a 0.03 m/s limit; its late cumulative planar path is **0.056694 m** against a 0.05 m limit. The PPO branches do not solve that problem. Zero's left/right turn reaches the late holding limits but fails the transient heading peak (about **0.25235 rad** against **0.08727 rad**). PPO turning also adds late velocity and displacement failures. Passing a small yaw impulse is not a substitute for reliable parking.

## Actual disturbance, timing and source verification

Each of the eight formal impulse episodes records **20 control intervals × 5 actual MuJoCo substeps**, with world-z torque ±0.5 N·m for approximately 0.002 s per substep. The recorded integral is **±0.09999999999998899 N·m·s**. The observer reads the actual wrench at `mj_step` entry; it does not infer force application merely from a requested value. Other bodies have zero external wrench. The original step methods are restored and applied wrench is cleared on exit.

The [original protocol draft](validation/protocol_before_native_api_fix.json) proposed pre-writing `xfrc_applied`; source inspection showed that the unchanged plant erases it every substep. The [reviewed revision 2](evaluation_protocol.json) therefore uses the existing native `push_torque_world_nm` argument through a probe-local instance wrapper. Only that interface description changed before the formal run. Revision 2 SHA256 is `cf5dbd51042bfafa163116f4a7fb2a990726d47c5a6000558f28aa6866a664fc`.

The [independent read-only audit](validation/independent_record_audit.json) verifies all 123 manifest entries, 85-dimensional observation records, all eight applied action channels, shared initial state/observation hashes, quaternion-derived actual attitude, heading-reference integration, rewards, all substep wrenches and 18 strict learned-versus-zero common prefixes. No physics was rerun for this audit. Gate windows use state endpoints `[start,end)`; late parking path includes positions 500 through 700 and sums 200 displacement intervals. It never substitutes net displacement for path length.

Seven focused [tests](validation/test_d1_heading_g1_probe.py) and Ruff passed. The first isolated two-tick interface smoke completed physics but hit a reporting `KeyError`; its [failure history](validation/interface_smoke_01/positive_torque_interface_smoke_2ticks/zero/execution_failure.json), raw states and trace remain unchanged. The corrected [smoke](validation/interface_smoke_02/summary.json) verifies ten substeps and +0.01 N·m·s. These **four total smoke control transitions** are separate from the 22,400 formal transitions. The [original failed-smoke runner](validation/probe_d1_heading_g1.failed_smoke_original.py) was reconstructed from the two later edits and accepted only after matching the exact original pre-run source hash; its [verification](validation/failed_smoke_source_reconstruction.json) makes that provenance explicit.

All [77 originally frozen paths](validation/frozen77_reference_protocol.json) still match their original hashes. The [crosscheck](validation/frozen77_crosscheck.json) distinguishes 69 paths captured before and after the formal run from eight additional post-run checks (an unused budget runner, an asset license and packaging metadata). The raw protocol is unchanged; the additional checks are not retroactively presented as pre-run measurements.

## Complete records and reproduction

`episodes/` is an exact copy of all 24 original episode directories and root records, including full compressed JSONL traces and lossless NPZ states. [Copied-artifact hashes](copied_artifacts.json) and [publication verification](validation/publication_verification.json) bind every visible copy to its original bytes. [Source snapshot](source_snapshot.tar.gz) and its [manifest](source_snapshot_manifest.json) retain all bound repository inputs, the 77 frozen paths and the new test file.

The three models are reused from the existing sibling [heading training evidence](../heading_learning_01/README.md); [model references](model_references.json) bind their exact hashes. No checkpoint is duplicated here. The probe supports `--models-root`, so original absolute work paths need not exist. From the matching repository/dependency environment, use a **new output directory**:

```bash
rtk proxy env PYTHONPATH=.local-deps:src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B -m scripts.probe_d1_heading_g1 --protocol results/d1_driving_stability_development/heading_g1_01/evaluation_protocol.json --models-root results/d1_driving_stability_development/heading_learning_01/training --output /absolute/path/to/new-g1-run
```

To independently recompute the archived records without running physics:

```bash
rtk proxy env PYTHONPATH=.local-deps:src:. python3 -B results/d1_driving_stability_development/heading_g1_01/helpers/audit_heading_g1_records.py --input results/d1_driving_stability_development/heading_g1_01/episodes --protocol results/d1_driving_stability_development/heading_g1_01/evaluation_protocol.json --output /absolute/path/to/new-g1-audit
```

Historical reproduction requires the bound sources. GUI parity, human operation, wider terrain coverage and a parking repair remain separate work. No further PPO budget was added after these failures.
