# Dynamic brake → hold: failed development candidate

**The candidate was rejected; the GUI default remains `legacy`.** No product controller change, RL training or human GUI trial is claimed. The fixed reference-only rule was tested in 12 full MuJoCo runs: six legacy/candidate pairs, 26,400 control transitions and seven complete eight-second release comparisons. Every failure is retained.

The rule caps the signed old reference error symmetrically at ±v²/(2a), with a=0.5 m/s², then enters hold after 30 consecutive ticks at |body-forward speed|≤0.03 m/s. Releasing W/S still requests exactly zero forward velocity. Parameters and cases were [fixed before execution](contract.md); no terrain-specific branch or parameter sweep followed the failures.

| Case | Legacy | Dynamic candidate | Result |
|---|---:|---:|---|
| Uphill release: peak backward retreat | 0.028450 m | 0.416399 m | Worse; never enters hold |
| Repeated flat stop, second release: peak forward excursion | 0.269312 m | 0.465354 m | Worse; never enters hold |
| Ramp release: final-two-second mean absolute speed | 0.006622 m/s | 0.026786 m/s | Worse despite entering hold |
| Stairs release: peak forward excursion | 0.506551 m | 0.236218 m | Improves in this case |

The uphill case is named for its **rough-terrain spawn**, but it has reached the uphill ramp by release. Its maximum low-speed streak is 26 of the required 30 ticks. The speed-scaled cap leaves little position error while rollback continues. Repeated stopping also exposes persistent forward creep. This is evidence against this particular algorithm, not evidence that every dynamic braking design fails. [Complete comparison, including total backward travel and sustained motion](summary/comparison.md), [full interpretation](summary/work_report.md).

All six first-release state/torque prefixes and all four corresponding historical legacy prefixes match bitwise. All states are finite; torques stay within unchanged bounds; there are zero recovery ticks. An [independent read-only audit](summary/audit_report.json) verifies all original hashes, all 50 source files inside each run's archive, every pre/post-compute distance/reference update and every reported release metric. The raw evidence distinguishes pre-compute reference limits from the original controller's subsequent distance integration and ±0.55 m clamp.

## Actual Opus implementation

Claude CLI requested `opus` and returned **claude-opus-5**, session `78f77dbd-24e1-4d50-88a5-cc00bb444385`, cost **$0.16119125** under a $0.40 cap and 900 s timeout. One call completed successfully; no retry was made. [Receipt](provenance/receipt.json), [response](provenance/response.json), [unaltered original code](provenance/dynamic_brake.original.py), [prompt](provenance/request.txt).

Review found and corrected two errors before physics: arithmetic overflow consumed helper state, and hold diagnostics described an unexecuted clipping operation. The original [failing checks](summary/original_core_red.json), reviewed [six passing mock checks](summary/core_checked.json), [reviewed core](helpers/dynamic_brake.py) and all original files remain available. Physics parameters were not tuned. Ruff passed for the core, checks, probe and arithmetic auditor.

## Lossless records with shared model blobs

The [index](evidence_index.json) preserves **177 original files, 1,091,656,124 bytes**, including all 12 `states.npz` files, telemetry, full per-tick memory traces, injected commands, protocols, analyses, compiled models, source archives and Opus provenance. Only Python caches are excluded. The 158 files inside the original run manifest comprise 1,091,516,308 bytes; the larger bundle total also includes the root contract, helpers, receipts and reports.

**All twelve model files reuse the two existing model blobs in [`../course_braking/`](../course_braking/README.md).** Ten existing blobs totaling 52,962,068 stored bytes are referenced without copying or changing them. The new local blobs total 21,587,889 bytes; the largest is 6,731,475 bytes. The sibling dependency is bound to its exact index SHA256 `109ada8a690dd994c66bd06a770beed05f8ed99d38727c00c5c8d68539fc9faf`. Restoration verifies that binding, matching blob metadata, compressed hashes and every reconstructed original hash.

Only gzip mtime bytes 4:8 in source archives are normalized for deduplication; their exact original bytes are restored from the index before checking the original SHA. All other compression is lossless. Keep both sibling result directories together.

From the repository root, verify every original byte without writing the duplicate 1.02 GiB tree:

```bash
rtk proxy python3 results/d1_driving_stability_development/brake_dynamic_candidate/restore_evidence.py --verify-only
```

Restore all originals into a **new** directory:

```bash
rtk proxy python3 results/d1_driving_stability_development/brake_dynamic_candidate/restore_evidence.py --output /absolute/path/to/new-evidence-directory
```

[Bundle verification](bundle_verification.json) records the successful check. [Readable copies](readable_copies.json) bind the convenient visible summaries, provenance and helper files to their original indexed SHA values. For a historical physical rerun, use the matching archived repository sources together with each run's `work_source.tar.gz`; later product source changes define a different experiment. The [auditor](helpers/audit_records.py) can recompute recorded evidence without running physics.
