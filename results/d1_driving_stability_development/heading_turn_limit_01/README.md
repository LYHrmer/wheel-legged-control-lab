# Fixed inner yaw limit 0.6 → 1.0: rejected turning candidate

2026-09-20. Both candidate turns still fail the unchanged 5° peak-heading gate. Other original turn gates pass. No default, GUI, raw command, reference integration or PPO policy changed. The full stable-driving/obstacle/speed objective remains incomplete.

GPT-6-astra / ultra proposed and reviewed the fixed single-variable causal test. Actual Claude response reports `claude-opus-5`, `firstParty`, exit 0, $0.12714125; its read-only contact module was integrated verbatim. Main agent wrote and tested the runner and performed physics.

| Turn | Inner cap (rad/s) | Heading peak (deg) | Body yaw pulse mean at decision (rad/s) | Wheel-fit yaw pulse mean (rad/s) | Contact tangent RMS at endpoint (m/s) |
|---|---:|---:|---:|---:|---:|
| stationary_turn_left_hold | .6 | 14.45879 | 0.09337540 | 0.44847002 | 0.02192265 |
| stationary_turn_left_hold | 1.0 | 11.47131 | 0.19503178 | 0.76761067 | 0.02951711 |
| stationary_turn_right_hold | .6 | 14.45866 | -0.09408850 | -0.47345146 | 0.02240499 |
| stationary_turn_right_hold | 1.0 | 11.08856 | -0.20897779 | -0.79155607 | 0.02992409 |

The larger cap increases wheel differential, body yaw response, and contact tangent speed; it reduces but does not resolve the heading deficit. This supports a partial cap bottleneck and does not identify the sole remaining cause. No larger cap or parameter sweep was tested.

## Original task and evidence boundaries

- Raw yaw remains ±0.6 rad/s for ticks200..249; the original reference reaches ±0.3 rad at2.5s. All original G1 turn gates are used.
- Two directions × two conditions, exactly3,200 control transitions/16,000 physics substeps. No training, resets for padding or learned policies.
- Baselines bitwise reproduce old G1 zero. State through tick200 and executed interval/observation prefixes before tick200 match across conditions.
- Direct requested/applied torques, inner/outer yaw, wheel geometry, full contact Jacobian velocity decomposition and contact wrench are recorded.
- Contact observations use the synchronized endpoint measurement cache. Wheel-fit/body means shown above use decision states; the contact means are one state later. They are not a continuous friction-work integral.
- `normal_world` keeps the raw contact-frame geom1→geom2 direction; force and torque fields act on the robot. The tangent projection is invariant to normal sign. Missing wheel contacts do not become zero-slip samples.
- All contact-point samples enter the reported RMS equally; multiple contacts and weak/side-facing contacts make this a descriptive statistic, not an independently identified traction parameter.
- Independent audit reconstructs raw yaw integration, heading score, contact-vector decomposition and summed wrench. 13 pure tests pass, including the archive failure regression. These tests add no physics steps.

## Preserved archive failure and budget accounting

Attempt01 completed the left 0.6 baseline (800 control steps), then hit FileExistsError when it tried to recreate the old runner’s exclusive manifest. All raw state/diagnostic data survived. Its first manifest may describe the diagnostic gzip before close; it is preserved as failed-attempt evidence and is not authoritative for that added stream.

Attempt02 carried that verified baseline forward without physics and ran only the remaining2,400 control steps. The current episode tree contains the four complete cases; the top-level manifest covers every preserved artifact. No completed baseline was rerun. The runner now uses a distinct complete_manifest and unique pair receipts; partial-control accounting uses physical substeps plus integer quotient/remainder.

## Independent audit

```bash
rtk proxy env PYTHONPATH=.local-deps:src:. python3 results/d1_driving_stability_development/heading_turn_limit_01/helpers/audit_turn_records.py --input results/d1_driving_stability_development/heading_turn_limit_01/episodes --output /tmp/turn_audit_new.json
```
