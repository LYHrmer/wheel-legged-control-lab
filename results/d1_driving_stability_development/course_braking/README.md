# Course braking development evidence

**The GUI keeps `legacy` braking by default.** Full reference reanchoring remains an explicit experiment because it increases uphill rollback in one measured case. Half reanchoring was also tested and was not selected. These are development checks of the course LQR/VMC controller, not RL training, a hardware test, or a human GUI trial.

The optional mode is `--brake-reference-mode release_reanchor_experimental` on `scripts/run_d1_course_drive.py`. It reanchors the retained distance reference once when the previous **actually accepted** legacy forward command was nonzero and the next command is zero. Side stepping, jumping, recovery and controller reset clear the edge context. The default `legacy` mode retains the original reference. Protocol v4 and each telemetry row record the actual mode and profile. See the [operation guide](../../../docs/interactive_course.md).

## Measured result and decision

Each value below is signed world-x displacement during the four seconds after releasing W. Alpha is the fraction of the old reference error removed: 0 is original, 0.5 is a work-only probe, and 1 is full reanchoring. Negative displacement means ending behind the release position.

| Case | Original α=0 (m) | Half α=0.5 (m) | Full α=1 (m) |
|---|---:|---:|---:|
| Flat, gear 3 | +0.274233 | +0.210168 | +0.148027 |
| Ramp spawn, gear 2 | −0.140819 | −0.095081 | −0.056776 |
| Rough spawn, gear 3 | +0.117574 | −0.000796 | −0.124600 |
| Stairs spawn, gear 3 | +0.497145 | +0.253129 | +0.117512 |

**“Rough spawn, gear 3” reaches the 8° uphill ramp before release** (world x ≈ 3.806 m, pitch ≈ −0.142 rad). Its backward retreat from the furthest forward position grows from **0.009138 m** to **0.087601 m** with half reanchoring and **0.199769 m** with full reanchoring. Near-zero net displacement in the half case hides this backward motion; it is not evidence of a good stop. The full and half candidates therefore remain unselected as defaults. [All alpha metrics and bitwise checks](summary/braking_fraction_analysis.json), [release states for all 12 forward cases](summary/release_prestate_table.json).

On flat ground the original three gears track 0.3/0.4/0.5 m/s with measured steady world-x speeds 0.310764/0.412514/0.498512 m/s. Full reanchoring shortens their time below 0.05 m/s for a continuous 0.2 s from 0.77/0.83/1.01 s to 0.68/0.72/0.76 s. Rough blocks and low-gear stairs still stall; ramp gear 1 still overspeeds before release. The candidate does not establish reliable obstacle traversal. The [initial full-candidate comparison](summary/braking_comparison.md) is historical candidate evidence, not the final default decision.

For A/D, releasing after one second interrupts the support-shift phase and returns near the origin. Completing the measured cycle moves about 0.037 m sideways and retains about 98.3% of it four seconds after handoff. This test does not support blaming the completed-step handoff for the reported short-key rollback. All four side-step trajectories are bitwise identical with and without the braking candidate, with zero braking events. [Baseline diagnosis](summary/baseline_diagnosis.md), [paired comparison](summary/braking_comparison.json).

## Scope and validation

The archive contains **49 physical runs and 81,800 control transitions**. Control period is 0.01 s and physics period is 0.002 s. Each run has its initial reset only. Held keys are injected into the real keyboard-command path headlessly; no GUI window was opened. Integrated qpos, qvel and applied torques are retained. Reported world-x velocity is finite-differenced integrated world position, not relabeled body velocity.

| Original directory | Runs | Purpose |
|---|---:|---|
| `forward_01` | 1 | First completed physical run; report serialization failed afterward |
| `forward_02` | 12 | Original W and release baseline |
| `side_01` | 4 | Original A/D short and complete cycles |
| `release_reference_01` | 4 | Single-variable full-reanchor mechanism probe |
| `forward_braking_01` | 12 | Initial integrated full candidate |
| `side_braking_01` | 4 | Integrated candidate side-step regression |
| `braking_fraction_01` | 12 | Four cases × original/half/full |

All 12 original/full forward pairs preserve every pre-release integrated state and applied torque. Each full-candidate forward run records one release event. The alpha probes preserve pre-release states and torques; alpha 0 and alpha 1 replay the respective original/full endpoints and entire recorded physics arrays bitwise. The first report failure, original incomplete JSON, recovered analysis, unsuccessful tests and network failure are retained.

Final validation: **44 targeted tests passed**, including 19 braking tests; Ruff and diff checks passed. Tests cover default versus direct-legacy full-trajectory equality, explicit experimental behavior, actual accepted commands, one-shot edges, ownership gaps, jump/recovery exclusions, invalid-state atomicity, X/focus loss and actual CLI protocol/telemetry. Test physics is additional to the 49 archived runs. All **77 frozen formal source files** remained unchanged. [Final verification](summary/final_verification.json).

## Actual Opus contribution

Claude CLI requested `opus`; the returned model was **`claude-opus-5`**, session `5465a063-4174-4139-8820-42e2c14f2f92`, cost **$0.095835**. It wrote the original braking core. Local review corrected atomic edge consumption, numeric validation and documentation, then integrated and tested it. [Receipt](provenance/opus_braking_02/receipt.json), [original response](provenance/opus_braking_02/response.json), [unaltered original Python](provenance/opus_braking_02/d1_course_braking.original.py), [request](provenance/opus_braking_02/request.txt).

The earlier call failed with `FailedToOpenSocket`, exit 1, `is_error=true`, zero tokens and zero cost; it contributed no code. Its wrapper's historical `status: returned` only means the process returned, not successful generation. Both the [failed receipt](provenance/opus_braking_01/receipt.json) and [error response](provenance/opus_braking_01/response.json) remain unedited.

## Verify or restore every original byte

The [index](evidence_index.json) maps **576 original files (4,373,282,075 bytes)** to 314 deduplicated blobs totaling **96,260,090 bytes**. Every original run includes its `states.npz`, telemetry, protocol, summary, manifests, compiled model and source archive. Repeated models are stored once. Only the four gzip mtime-header bytes in each `source.tar.gz` are normalized for deduplication; the exact original four bytes remain in the index and are restored before checking the original SHA256. Other payloads are unchanged or losslessly XZ compressed. The largest blob is 26,519,708 bytes.

From the repository root, the standard-library-only verifier checks every stored blob and every reconstructed original SHA256 without writing the 4.1 GiB duplicate tree:

```bash
rtk proxy python3 results/d1_driving_stability_development/course_braking/restore_evidence.py --verify-only
```

To restore all original files into a **new** directory:

```bash
rtk proxy python3 results/d1_driving_stability_development/course_braking/restore_evidence.py --output /absolute/path/to/new-evidence-directory
```

[Bundle verification](bundle_verification.json) records the successful byte-for-byte check. `summary/`, `provenance/` and `helpers/` provide small readable copies; their origins and hashes are listed in [readable copies](readable_copies.json). The restored tree also contains the original builder, probes, analyses, failed outputs and both Opus calls.

For historical physics replay, use each run's archived `source.tar.gz` in an isolated checkout with the matching helper and dependencies. The initial candidate archive predates the explicit-mode switch: its default was the full candidate. Running a historical candidate probe against today's default-legacy source would change the experiment. The work-only alpha override must likewise use its recorded source revision. Read-only analysis of restored records requires no rerun; `helpers/analyze_braking_fraction.py --input RESTORED_ROOT --output NEW_JSON` recomputes the alpha metrics and bitwise comparisons.
