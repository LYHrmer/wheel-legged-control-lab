# Fixed 0.5 m/s² release reference: rejected candidate

2026-09-20. **Both original stopping cases still fail.** The candidate remains a zero-policy diagnostic module; no GUI or default controller behavior changed. Stable driving, reliable obstacles, speed, and physical self-righting remain incomplete.

GPT-6-astra / ultra planned and reviewed the bounded experiment. Actual Claude response reports `claude-opus-5`, provider `firstParty`, exit 0, $0.129685. The first sandbox request failed with `FailedToOpenSocket`, zero tokens/$0. Original response, source and root integration diff are retained; root changes to Opus source were type annotation style and export ordering only. Main agent wrote the adapter/probe, integrated, and performed MuJoCo validation.

## Original goal scores

User forward intent becomes exactly zero at tick 400. The release reference is recorded separately; environment rewards refer to that served reference and are diagnostic only. No reward or softened reference replaces the original goal score.

| Case | Mode | Raw vx RMSE (m/s) | Late speed peak (m/s) | Late planar path (m) | Peak excursion after release (m) | Result |
|---|---|---:|---:|---:|---:|---|
| flat_forward_stop | bypass | 0.05954354 | 0.06086945 | 0.05669442 | 0.05407383 | FAIL |
| flat_forward_stop | release_0p5 | 0.06826293 | 0.10602366 | 0.06425530 | 0.10523034 | FAIL |
| flat_reverse_stop | bypass | 0.05991671 | 0.08634778 | 0.07616839 | 0.06779416 | FAIL |
| flat_reverse_stop | release_0p5 | 0.07022989 | 0.09318912 | 0.06759948 | 0.12376301 | FAIL |

Unchanged limits: raw velocity RMSE ≤0.05 m/s; state endpoint ticks 500–799 speed ≤0.03 m/s; accumulated planar path across positions 500–700 ≤0.05 m. All four stop episodes fail these three gates. Other original stop gates pass. Both signed yaw impulse cases pass and remain bitwise unchanged between conditions.

## What the intervention changed

| Direction | First 0.5 s body−wheel RMS, bypass → release (m/s) | First release wheel mean requested torque, bypass → release (Nm/wheel) |
|---|---:|---:|
| flat_forward_stop | 0.12428973 → 0.06302429 | -5.84022239 → 0.43966267 |
| flat_reverse_stop | 0.13340451 → 0.07845400 | 5.67379187 → -0.60609319 |

The tail reduced early wheel/body separation and the abrupt first-tick braking torque, but did not meet the original stop gates. Input smoothing alone is insufficient in this experiment. The observations do not identify PI memory, leg motion, or contact slip as the unique cause. No alternate deceleration, gain sweep or new PPO training was tried.

First served stop speed is ±0.245 m/s at tick 400; first zero is tick 449. The elapsed slew from last nonzero raw reference at tick 399 is 0.5 s. The actual nonzero served intervals after raw release last 0.49 s. These are different timing quantities.

## Verification and scope

- Exactly 8 episodes, 8,000 control transitions, 40,000 physics substeps, zero training steps; no termination/reset padding.
- All four bypass runs bitwise reproduce old G1 zero state/observation/action arrays.
- Stop qpos/qvel/position states through tick 400 match. Observations and executed controls before tick 400 match; observation 400 legitimately contains a different served reference.
- Impulse qpos/qvel/observations/actions, raw/served/servo commands, actual torques and applied wrenches are bitwise identical across conditions. Actual impulses remain ±0.1 N·m·s.
- Direct controller requests, per-substep applied actuator torques, wheel targets and PI memory are saved.
- T transitions versus T+1 states/source records checked; terminal prepared record is not scored as an executed interval.
- Independent read-only audit recomputes original user errors, late windows, path, motion tradeoffs and pair equality. 27 unit/boundary tests pass. These checks do not establish complete stable driving.
- 77 frozen inputs match their original SHA256 values. Old studies, policies, sidecars and results remain untouched.

## Reproduce the audit

```bash
rtk proxy env PYTHONPATH=.local-deps:src:. python3 results/d1_driving_stability_development/heading_release_01/helpers/audit_release_records.py --input results/d1_driving_stability_development/heading_release_01/episodes --output /tmp/heading_release_audit_new.json
```

The output path must be new. The physical runner is `scripts/probe_d1_heading_release.py`; the completed eight-case matrix should not be repeated as a next step. Continue source-grounded damping/contact diagnosis and the separate turning diagnosis before proposing another fixed control-law experiment.
