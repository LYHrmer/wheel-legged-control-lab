# Isolated CCD patch and rolling shared-leg RL trial

The 65,536-control PPO training and its single final checkpoint are preserved.
Three original evaluation cases completed in run_02. That process then
disappeared during the fourth case for an unestablished reason. Its final
native C counters and exit status are unavailable; the entire original
reservation is closed. The partial fourth case is diagnosis only.
A separately budgeted run_03 evaluated precisely the five remaining cases,
restarting the fourth from the prescribed initial state with the same checkpoint.

- Original run_02 study qualified: **False**; final C counts: **unknown**.
- New five-case execution valid: **True**; independent readback: **True**.
- Four final-policy task and speed gates across the fixed eight cases: **False**.
- Uninterrupted global accounting: **False**. Each run has its own closed budget.
- Prior run_01 failed during environment bootstrap before model construction
  and recorded zero training controls. It has no final native C receipt,
  so its final C counters are not asserted here.
- The first drive intervention (04) failed before transfer because `_steps`
  does not exist on a never-reset environment. It had five compiler steps
  and zero control, normal CCD and training steps; it contributes no case.
- Corrected drive intervention (05) execution valid: **True**; independent readback: **True**.
- Corrected drive candidate qualified: **False**.

## Eight fixed cases

| Speed | Terrain | Actor | Process | Record valid | Task passed | Mean body vx (m/s) | Speed RMS error (m/s) | Failure reason |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- |
| 0.2 | plane | zero | interrupted_run_02_complete_record | True | True | 0.198000525648156 | 0.012770819321421992 | none |
| 0.2 | plane | final_policy | interrupted_run_02_complete_record | True | True | 0.1978277730519005 | 0.013667015406210757 | none |
| 0.2 | box | zero | interrupted_run_02_complete_record | True | False | 0.1956113121670765 | 0.0908778667297846 | speed_window_failed |
| 0.2 | box | final_policy | new_run_03_continuation | True | False | 0.1953428679268952 | 0.0931686283196967 | speed_window_failed |
| 0.25 | plane | zero | new_run_03_continuation | True | True | 0.2473878407054043 | 0.017566236862648705 | none |
| 0.25 | plane | final_policy | new_run_03_continuation | True | True | 0.24721932956026318 | 0.018478997079012873 | none |
| 0.25 | box | zero | new_run_03_continuation | True | False | 0.24533999546182467 | 0.10256116442713259 | speed_window_failed |
| 0.25 | box | final_policy | new_run_03_continuation | True | False | 0.24515866526669589 | 0.10511264946156519 | speed_window_failed |

## Four explicit drive-stage interventions

The one frozen final PPO checkpoint was loaded strictly under its original
controller before an explicit controller-law transfer. There was no retraining.
These four cases are a separate process and do not repair the original
run_02 interruption or replace the eight fixed baseline scores.

| Speed | Terrain | Task passed | Mean body vx (m/s) | Speed RMS error (m/s) | Failure reason |
| ---: | --- | --- | ---: | ---: | --- |
| 0.2 | plane | True | 0.2003480214064334 | 0.0008402388756141311 | none |
| 0.2 | box | False | 0.19838407028694713 | 0.06395771003172954 | speed_window_failed |
| 0.25 | plane | True | 0.25044110740222153 | 0.001089674760653355 | none |
| 0.25 | box | False | 0.24866105863733656 | 0.05394424863681878 | speed_window_failed |

Policy mean-speed deltas (.25 minus .20 m/s command): `{'box': 0.04981579733980068, 'plane': 0.04939155650836269}`.
A zero-policy task failure remains a failure; it is not recast as a policy result.
In the original eight-case comparison, the frozen policy did not establish
an independent advantage over zero residual and was slightly worse on
the saved speed-error comparison. The later change is an explicit drive
controller intervention, not a new learned policy. No zero-residual cases
were run under that new law, so its independent RL contribution is untested.
The corrected intervention improves the recorded box speed errors but
both box cases still fail the unchanged speed RMS gate. The two plane
cases pass; the four-case candidate is therefore not qualified.
These eight deterministic simulator episodes do not establish statistical
reliability or hardware performance.

## Reproduce and audit

`engine_patch/BUILD.md` records the exact isolated DSO build and pinned ABI.
`engine_patch/provenance/engine_binding.py` is the executed fixed-offset
binding bridge; its absolute paths are specific to the original environment.
The installed MuJoCo library was never overwritten; the optional DSO artifact
is for hash comparison only and is not automatically loaded or installed.
The compact package includes source, checkpoint, receipts, scores, source
hashes, decimated trajectories and plots. It omits the full native/endpoint/
trace/state streams. Full paired-initial, torque and contact-record checks
require the raw run directories and pinned independent readbacks.
The complete release archive contains raw states and native records from
the original, continuation and corrected drive processes, including the
04 failed startup and interrupted prefix kept outside the twelve
completed-case result: https://github.com/LYHrmer/wheel-legged-control-lab/releases/download/rolling-rl-damping-20260924/wheel-legged-rolling-rl-speed-20260924-raw.tar.gz
Raw asset SHA-256: `8ca21fcd615b74f4960e992f107c7c1eb39ae2fb8abdd523f82e55fad6e50ff5`; bytes: `256694687`.
`evidence/rl/case_source_map.json` binds each compact case to its raw archive
directory and file hashes; `evidence/drive/case_source_map.json` binds the
four corrected intervention cases. The compact package alone cannot independently
recompute every contact-geometry or force gate.
No original score or frozen input was rewritten.

Original completed-case readback passed: `True`.
New five-case readback passed: `True`.
Corrected drive independent readback passed: `True`.
