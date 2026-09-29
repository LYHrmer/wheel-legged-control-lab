# D1 world-upright RL: completed training, incomplete evaluation

The single 11-S run completed **65,536 training controls** and saved one final
99-observation/16-action PPO checkpoint. Its worker recorded 64 rollout/train
calls, 256 epochs, 1,024 optimizer steps, one save and two strict reloads.
These are training and checkpoint facts, not a heldout qualification.

The 1,800-second host watchdog interrupted the tenth heldout case while a
gzip native block was being written. Nine earlier cases have complete worker
receipts. The rough-terrain final-policy case has 1,023 returned controls and
one additional five-native-step interval whose outer control did not return;
its case receipt and endpoint archive are absent. Neither ramp actor ran.
The host recorded exit `-15`, zero source-hash changes and no orphaned child.
The worker's later `FileExistsError` arose while trying to seal the already
created interrupted block. The full 12-case qualification and matched-pair RL
contribution remain **unestablished**. The checkpoint is preserved for analysis,
but this package does not authorize it as the default GUI policy.

The [independent partial readback](evidence/rl11/partial_readback_20260929_05.json)
verified the completed training, final checkpoint bytes and strict reload
evidence, all nine complete cases' saved native/contact chains and four
bitwise-identical zero/policy initial-state pairs. It rescored saved records
without replaying physics or loading a model. **All nine complete cases passed
their preregistered task gates.** The four complete comparisons are:

| Heldout task | Zero / final-policy task | Final-policy drive SSE vs zero | Torque-squared proxy ratio |
|---|---|---:|---:|
| Flat 0.6 m/s | pass / pass | +53.054% | 1.0593 |
| Flat 1.6 m/s | pass / pass | +132.413% | 1.1326 |
| Flat 1.2 m/s with yaw | pass / pass | +9.494% | 1.0808 |
| Bumps 0.4 m/s | pass / pass | +15.578% | 1.0429 |

Rough 0.35 m/s zero also passed. SSE is the preregistered drive-window
tracking error; the torque-squared quantity is a normalized proxy, not
measured energy. All four observed SSE and proxy costs worsened with the
policy, so these pairs do not establish RL gain. The rough-policy and both
ramp results remain unknown, and a complete six-task contribution decision
cannot be certified from nine cases.

In the single-seed flat 1.6 m/s script, actual COM forward speed averaged
1.610807 m/s with zero residual and 1.632405 m/s with the final policy;
their RMS errors were 0.012401 and 0.032885 m/s. The policy exceeded
1.5 m/s on all 400 hold ticks and settled 3.37 s after release with 2.642 m
maximum forward excursion, passing this case's stop gate. This is demonstrated
scripted flat-ground speed capability, not cross-terrain 1.6 m/s reliability,
15 mm step crossing at that speed, or GUI performance.

The [closed raw-file manifest](evidence/rl11/partial_closed_manifest_20260929_05.json)
records the local source records' identities, including the interrupted
tenth case; it does not repair or qualify that case. The compact package
contains original source snapshots, the checkpoint, preregistered contract,
GO and small evidence. The source snapshots retain their original
working-directory paths in receipts; this package is neither a standalone
training rerun nor a new GUI distribution. The
`publication_manifest.json` records this package and its related documents;
this read-only check verifies their
file bytes without importing a policy or replaying physics:

```bash
rtk proxy python3 results/d1_rl16_progress_20260929/verify_package.py
```

Package integrity does not replace the full raw-record readback. See
[reader revisions and preserved failures](READBACK_NOTES.md) for the fixes
made to the offline verifier, without changing the original experiment.

Full raw training/native/heldout files are retained locally under the frozen
11-S run. They are **not uploaded** in this update; no raw archive asset URL or digest
is claimed here. The 11-R zero-policy world-upright ramp baseline and 11-M
synthetic model preflight are separate passing prerequisites, not evidence of
RL gain. The Q compact GUI missed its declared performance threshold; the new
GUI12 source is still an unexecuted candidate. Sideways A/D control and Space
jump are not implemented in the qualified entry.

The [2026-09-29 main plan](../../docs/main_plan_20260929.md) and
[handoff](../../docs/codex_handoff_20260929.md) distinguish completed work
from conditional next steps. This package excludes
private driving sessions, Xauthority/cookies and raw Claude conversations.
