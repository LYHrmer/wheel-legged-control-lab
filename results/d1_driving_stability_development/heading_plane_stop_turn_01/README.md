# Fixed stop/turn composition passes eight prespecified cases

The combined zero-residual native-plane controller passed the six original G1 cases and two specified drive → stop → turn handoffs, with all required whole/prefix comparisons. This qualifies only this finite, deterministic, oracle-state development domain; default entry points remain unchanged.

| New combined case | Physical comparison | Original numerical gates |
|---|---|---|
| Forward and reverse stop | Each complete800/801 trajectory equals the qualified stop damper | All pass |
| Left and right stationary turn | Each complete800/801 trajectory equals qualified turn authority | All pass; peaks3.663°/3.665° |
| Positive and negative forward yaw impulse | Each complete1200/1201 trajectory equals qualified no-op data | All pass |
| Forward → stop → left turn | First800 intervals/state800 match forward stop; obs800 excluded | Global, stop and turn views pass; peak3.068° |
| Reverse → stop → right turn | Same fixed prefix against reverse stop | Global, stop and turn views pass; peak3.063° |

One batch completed8400 control intervals /42000 actual native substeps without retries. Stored standalone baselines were read, not rerun. No training, full24G1 replay, gain sweep or frozen77/old-record edit occurred. Original raw command/reference scoring and all original numerical thresholds were preserved.

An actual Claude Opus call (provider `claude-opus-5`, $0.251285) wrote only the bounded core; root read and integrated its source byte-identically, SHA `bacdf478c3fa4787e0747f45e3fe6187291949128b93023e1396575f8d7d4749`. GPT-6-astra / ultra supplied the fixed contract and independent reviews. Root authored the thin environment/runner/scorer and performed physics and archive reconstruction.

The single inheritance chain is composition → turn authority → original wheel/leg controller. Original wheel PI/antiwindup updates once. Pure nominal preview restores the original uncapped yaw formula only on raw pure turning. After an executed nonzero-forward→zero transition, the original fixed stop increment `−126.4374005337902 J_bodyx.T (J_bodyx qdot_leg)` is added to the unlimited leg request, followed by original protection. Wheel increment remains exactly zero. The stop latch stays active through a subsequent raw turn; it is not suppressed to obtain a pass. Both mechanisms overlap for precisely50 intervals in each handoff.

Original-stop damping is active400 intervals; original turns never activate it; impulses activate neither. Handoffs keep it active1000 intervals and the authority gate active50. Peak added leg torque in the handoffs was23.6273Nm. Wheel targets never clipped; each turn/handoff had8 protected joint-intervals, with actual wheel torque limited to12Nm. Sampled algebraic and protected same-state incremental power were nonpositive; no global passivity or held-interval-work claim is made.

The handoff stop score uses only physical rows0..799 and states0..800, so later stillness cannot dilute stop RMSE. Segment coverage is distinguished from actual episode truncation. The turn score uses rows600..1399/states600..1400; only scoring indices are shifted by600, while the continuous raw reference remains unchanged. Late turn heading/speed use physical endpoints1050..1399 and planar displacement uses state600 as origin. Global1400-step completion and original all-case gates apply separately. Final late turn speeds were.001377/.001341m/s.

Before physics39 nonintegrating checks passed:15 composition/env,16 scoring-window boundaries and8 incompatible-checkpoint rejection. Pure mode sequences verify latch closing/reopening; they do not qualify dynamic restart. Independent source and runner reviews found no remaining blocker. The six stored baseline formats were read without controller calls or physics; this was a format check, not a self-pair proof.

Root reconstructed all8408 saved states, target formulas, original wheel PI, stop increments, protections, native contact records, synchronized endpoint kinematics and original raw scores. The independent archive audit passed8,206,129 checks with zero new integration/contact solves/controller calls. The original base leg PD/support request is archived, not independently reimplemented. Old stop logs lack explicit full-precision target fields; their equivalence uses equal states/raw-servo/PI/requests and the unchanged inactive formula, rather than claiming missing target records were directly compared. Turn target/effective/error fields are directly compared bitwise.

Handoff prefix comparison excludes obs800 and the fixed next-preview/horizon fields at execution799; state800 and all executed physics remain equal. Phase names are display labels and are excluded from handoff trace digests, retained in raw logs. Native contacts keep the actual solved-cache phase; endpoint geometry is separate. The native pulse-moment audit uses each case's actual pulse window, including800..849 for handoffs, rather than a hardcoded old200..249 interval.

This result does not qualify moving turns, all four sign pairings, immediate stop-and-turn onset, dynamic restart, noise/latency, real hardware, GUI/manual acceptance, jumping, steps, other obstacles or speed. R remains simulation reset; physical self-righting is absent. The earlier RL GUI request timed out without code. The next fixed proposal measures actual airborne contact history, oriented collision-wheel clearance and settled landing under the existing PD/PI height channel, before designing a new jump mechanism or claiming obstacle traversal.
