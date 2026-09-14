# Dynamic brake → hold: rejected work-only candidate

The fixed reference-only candidate is **not suitable for product integration**. Keep `legacy` as the GUI default. Twelve complete MuJoCo runs (six pairs, 26,400 control transitions) and seven complete eight-second release comparisons expose additional uphill rollback and repeated-stop creep. No repository source was edited, no GUI opened, no RL trained, no parameter search performed.

Read the [full comparison table](audit_01/comparison.md) and [independent arithmetic/integrity audit](audit_01/report.json). The immutable [implementation contract](contract.md) was written before calling Opus or running physics. It fixes a=0.5 m/s², a symmetric ±v²/(2a) reference-error cap while braking, and 30 consecutive ticks at |body-forward velocity|≤0.03 m/s before anchoring hold. All velocity requests during release remain zero.

The decisive failures are:

- **Uphill release (rough spawn, gear 3):** peak backward retreat grows 0.028450 → 0.416399 m; maximum reverse speed grows 0.006598 → 0.072320 m/s; final-two-second mean absolute speed grows 0.004435 → 0.049293 m/s. The robot is already on the uphill ramp at release. The candidate immediately cuts reference error 0.550000 → 0.081306 m. Its low-speed streak reaches only 26 of the required 30 ticks, so it never enters hold. Final-two-second mean retained error is about 0.002964 m versus legacy 0.444730 m. This supports the diagnosis that the cap suppresses the position term needed to resist continuing drift; it does not measure an independently isolated required traction force.
- **Repeated flat stop, second release:** forward excursion grows 0.269312 → 0.465354 m; final-two-second mean absolute speed grows 0.001591 → 0.033307 m/s. It never even begins the low-speed dwell in this release. The candidate stays in brake with a tiny reference error and continuing forward creep. Second-drive trajectories are expected to differ because their first stop already differed; each strategy receives the same timed second command.
- **Ramp spawn, gear 2:** peak retreat improves 0.308819 → 0.193097 m, but final-two-second mean absolute speed worsens 0.006622 → 0.026786 m/s. It enters hold at 14.73 s, so entering hold alone does not establish stable stopping.

Stairs gear 3 improves substantially (peak forward excursion 0.506551 → 0.236218 m and final-two-second mean absolute speed 0.004046 → 0.001845 m/s). Flat straight and reverse cases have mixed results. Only the stairs release passes all four frozen comparative/end-window diagnostic flags. Smaller signed net displacement is never used as a substitute for forward excursion, total backward travel, peak retreat and sustained speed.

All six first-release prefixes preserve qpos, qvel, time and applied torque bitwise. The four legacy cases also reproduce the entire corresponding historical record prefix bitwise. All states are finite, every torque is within the existing actuator bound, and there are zero recovery ticks. These are valid counterexamples to this particular stopping algorithm, not a simulation crash or changed-input comparison.

## Actual Opus contribution and review

The standard `run_opus_task.py` wrapper made one successful, bounded code-only call: requested `opus`, actual model **claude-opus-5**, session **78f77dbd-24e1-4d50-88a5-cc00bb444385**, cost **$0.16119125**, exit 0, about 65 seconds. The cap was $0.40 and timeout 900 s. The call was not retried.

Keep the [receipt](opus_01/receipt.json), [prompt](opus_01/request.txt), [original response](opus_01/response.json) and [unaltered Python](opus_01/dynamic_brake.original.py). Opus wrote the actual helper. Local review found two implementation errors and preserved their [failing mock checks](original_core_red.json): overflow consumed the release edge/counters, and hold diagnostics reported a hypothetical clipped error even though hold did not write it. The [reviewed implementation](dynamic_brake.py) validates arithmetic before committing and reports the actual hold error. Initial idle zero avoids unnecessary memory reads; hold freezes the low-speed counter; wording and formatting were corrected. The algorithm and fixed physical parameters were not tuned. Six [pure-mock checks](core_checked.json) passed after the fixes; Ruff passed for the implementation, runner, checks and independent auditor.

The independent reviewer confirmed the velocity convention and one-step timing. The cap uses **pre-compute** controller memory; legacy then advances distance by body_vx×0.01 and applies its original ±0.55 m reference clamp. Hold entry therefore does not promise an exactly zero error during the subsequent compute, and the old low-level clamp can still move the retained reference after large later drift. Every pre/post memory value and actual accepted command is retained and independently checked.

## Records and reproduction

`runs_01/` contains all 12 runs, each with lossless `states.npz`, telemetry, per-tick `control_memory_trace.jsonl`, injected key events, protocol, compiled model, original repository source archive, work-source archive and analysis. `runs_01/manifest.json` covers 158 files totaling 1,091,516,308 bytes. Every hash, all 50 repository source files inside every archive, all work-source copies, every per-tick reference update and every reported numerical release metric were verified. Source hashes before/after physics match. The manifest excludes itself by design; the root artifact manifest binds it.

The in-process `WorkDrive`/`Bridge` override is explicitly identified in each protocol before the file is written; dynamic telemetry identifies `work_dynamic_brake_hold`. The repository source archive remains the unchanged real implementation. Reproducing a candidate requires both that archive and `work_source.tar.gz`, as stated in the protocol.

From the recorded repository checkout and dependency environment, with **new output directories**:

```bash
rtk proxy env PYTHONPATH=.local-deps:src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B WORK/brake_dynamic_01/run_paired.py --repo REPO --output NEW_RUN_DIRECTORY --historical WORK/gui_baseline
rtk proxy env PYTHONPATH=.local-deps:src:. python3 -B WORK/brake_dynamic_01/audit_records.py --input RUN_DIRECTORY --output NEW_AUDIT_DIRECTORY
```

The first command runs physics; the second only reads records and independently recomputes the evidence. Do not run the historical experiment against later product source changes and call it the same experiment. No automatic integration or further sweep follows this rejection.
