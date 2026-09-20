# Fixed wheel-center turn compensation: original peak-heading gate still fails

The fixed coefficient-1 candidate improved the two original turn peaks but failed the original 5-degree limit. It is not adopted. The default controller, original raw references and gates are unchanged.

| Case | Plane-zero peak | Candidate peak | Original failed gates |
|---|---:|---:|---|
| Left turn | .234134 rad | .179587 rad (10.290 degrees) | heading_peak |
| Right turn | .234213 rad | .179692 rad (10.296 degrees) | heading_peak |
| Forward stop, no-op | original trajectory | byte-identical | original late speed, late path, raw velocity RMS |
| Reverse stop, no-op | original trajectory | byte-identical | original late speed, late path, raw velocity RMS |

These no-op stop cases do not contain the separately qualified stop-leg damper. Their failures remain explicit; the passing damper records are in `../heading_plane_stop_damping_01`.

An actual Claude Opus call returned the bounded controller core (`claude-opus-5`, provider receipt and original source preserved). Root integrated it and wrote the thin raw-callback environment and fixed runner. gpt-6-astra / ultra supplied the hypothesis, fixed contract and independent reviews. The initial local prompt-assembly attempt failed before any provider call; it is preserved. No new GUI code was generated.

During the original 50-interval pure-turn pulse only, the candidate adds measured leg-center longitudinal speed / .087 to each unclipped wheel target, using the horizontal heading frame. The coefficient is exactly one, with no latch, filter, gain sweep, mean removal, contact weighting, target shaping, PI reset or stop damping. The effective body-yaw request remains limited to .6 rad/s; the compensated wheel differential can imply a larger geometric yaw. Its peak was 1.756 rad/s, and the peak actual wheel target was 4.436 rad/s. Neither is measured body yaw. No target clip or torque protection activated.

Exactly four new cases completed: 3200 control intervals / 16000 native substeps. The four plane-zero baselines were reused, never rerun. Turn execution intervals 0..199 and physical states 0..200 are byte-identical; observations are compared through index199 because observation200 previews the changed target. Each no-op stop matches all 800 intervals and 801 states/observations, complete native records, torque/PI/action/raw/servo values and the original summary except its model label. The raw turning reference remains +/-.300 rad after the unchanged pulse.

Before physics, 26 tests prohibited every integration entry point. Root independently reproduced the original-turn diagnosis byte-for-byte (SHA256 25cfa109082efe578e3ff7ab8c3a9d10e3f667e9412f43b2f9656de558ad1450) and checked 100 independent saved-state target/PI calculations. Maximum target and wheel-request differences were 4.45e-16 rad/s and 1.78e-15 N m. The first saved-state preflight failed on a tuple/NumPy indexing mistake before candidate compute or integration; both source and failure are retained. This algebra check did not predict performance: the candidate's actual target excursions exceeded those calculated independently on old states.

Root separately reconstructed the original raw-reference scores and ran the full independent saved-state/native evidence audit (3,032,135 checks, zero new physics). A supplementary audit matched aggregate pairing claims to the per-case evidence, compared trace positions bitwise with saved physical states, and independently reconstructed cross-track scores. Physics representation, execution evidence and isolation checks are separate from the failed turn task gates. Native force logs and synchronized endpoint velocities retain their phases; no mixed-phase work or passivity claim is made. The audit reconstructs the wheel PI and all final torque protections, but does not independently derive the original leg PD/support request.

The center correction is a rolling approximation. Original data attribute roughly 76.66% of the wheel/body velocity gap to its center-translation term, 20.89% to leg-driven carrier rotation and 2.44% to slip in the pulse window. These are terms in a kinematic identity, not causal fractions or predicted improvement. Later lateral friction and slip remain relevant. This failed candidate is frozen; any next mechanism needs a separate evidence-based contract, rather than a gain/cap sweep or unvalidated combination with the stop damper.

The full straight-driving, obstacle and higher-speed objective remains incomplete. Manual driving, meaningful jump clearance/landing, steps/slopes/rubble and speed progression remain unverified. R is still simulation reset; physical self-righting is not implemented. The previously timed-out RL GUI still has no generated implementation. Frozen77, all older experiments and the three 65k trainings/full24 G1 are unchanged.
