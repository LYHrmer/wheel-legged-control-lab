# RL gradient-clipping comparison: stages 15–16

Two matched 16,384-control PPO continuations and one complete six-task, three-actor evaluation are closed. Grouped actor/critic clipping improved tracking relative to global clipping, but both policies passed only four of six tasks and failed the overall RL contribution gate against the zero-residual controller. **Neither checkpoint is qualified as the default GUI policy.**

See the [Chinese execution report](../../docs/main_route_execution_20260930_15_16.md), [Astra final review](stage16/final_review_15_16.md), [compact results](stage16/result_summary_15_16.json), and [task CSV](stage16/task_results_15_16.csv).

| Result | Zero residual | Global continuation | Grouped continuation |
|---|---:|---:|---:|
| Fixed tasks passed | 5/6 | 4/6 | 4/6 |
| Flat 1.6 m/s hold COM mean | 1.610807 | 1.629706 | 1.610590 |
| Flat 1.6 m/s hold speed RMS | 0.012401 | 0.030832 | 0.014046 |
| Overall RL contribution gate | Baseline | Fail | Fail |

All 18 full-task records and both 600-control floors are valid. All safety and final-stop gates passed. Both learned policies failed the yaw tracking gate; all three actors traversed the ramp geometry but failed its speed-tracking task. Grouped/global pooled drive SSE over the four common passing tasks was 0.726445, yet the required all-six-task condition failed. Grouped/zero pooled SSE was 1.039400; its isolated flat-1.6 drive SSE improvement does not establish overall superiority or better steady-state tracking than zero.

## Models and mechanism evidence

Both models retain the same parent and Adam state before training. Same first 1,024 controls and saved Gaussian arrays matched byte for byte. The first pre-clip gradients matched exactly; independent clipping increased actual actor Adam deltas, while critic deltas remained similar. The grouped first-rollout KL of 0.63234 is retained as a policy-update risk, not asserted as the cause of a later task failure.

| Checkpoint | SHA256 |
|---|---|
| [Global final](stage15/global_01/final_checkpoint/final_model.zip) | `db324c8ba2c7a4821352f1d564b47d4fcffb97784d751e9ea20b23d525674d2b` |
| [Grouped final](stage15/grouped_01/final_checkpoint/final_model.zip) | `1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb` |

Training sources, fixed gradient samples, first paired raw blocks, model metadata/probes and [independent training readback](stage15/training_readback_15.json) are included. This is one paired training seed, with one fixed evaluation seed per task, 99D simulation-oracle observations and a 16D residual policy. There is no hardware or statistical reliability qualification.

## Failure and accounting

The original stage-15 evaluation failed before model load or evaluation controls because a live physics process imported a deliberately isolated offline verifier. Its exception and receipts are preserved. Stage 16 used a separate contract, fixed clean-process bridge and one new reservation; no retraining or seed selection occurred. The two targeted regression tests and original 22 necessary stage-15 checks are preserved separately.

[Actual stages 15–16 totals](stage16/execution_summary_15_16.json): 63,368 controls, 316,840 normal native steps and 8 compiler native steps; 32,768 training controls and 512 optimizer steps. Reservations total 93,968 controls and 469,840 normal native steps plus 8 compiler steps because the failed evaluation reservation was not refunded. Stage 14's offline diagnostic is separately published and excluded from these totals. Sources stayed unchanged and owned processes were reaped.

## What this package can reproduce

`publication_manifest.json` identifies every copied payload by original path, SHA256 and byte count. Selected state arrays, schedules, receipts and numeric summaries permit inspection of reported metrics; the model artifacts and sampled full gradients permit further work under a new bounded contract.

This is a selected evidence package, approximately 97 MB. The [local archive inventory](stage16/local_archive_inventory_15_16.json) identifies 1,538 original files totaling 1,065,187,100 bytes. **Full control/native gzip payloads and most training blocks are retained locally, not uploaded.** Their manifests do not substitute for their contents. Full independent physics-chain readback requires those originals and the recorded frozen dependency/source layout. Some workbench scripts embed that layout and are not portable one-command launchers.

No new GUI, keyboard, single-step, lateral, jumping or self-righting qualification is provided. The stage-13 limited GUI result for the old final keeps its original scope; simulation reset remains distinct from physical recovery. Test and CI success establish software checks, not robot qualification. The final review records one proposed next direction; no further training budget is reserved here.
