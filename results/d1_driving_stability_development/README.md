# D1 driving stability development

This is an open development record. It does not change the frozen formal studies or establish completion of the driving optimization goal. GUI currently uses classic LQR/VMC; these PPO results use the synchronized learning control chain.

- [Yaw ablation](yaw_ablation/README.md): 19 closed-loop episodes / 56,844 transitions, all failures retained. Actual Opus implemented the runner.
- [Heading hold](heading_hold/summary.json): 14 episodes with unchanged old weights; complete policy episodes increased from 1/6 to 4/6 on the opened road. The remaining failures prevent a complete stability claim. NPZ and lossless gzip traces preserve all samples; `independent_audit.json` records 311 checks (the audit was executed inline, not supplied as a standalone script).
- [Braking candidates](course_braking/README.md): reference interventions and their terrain-dependent regressions. Original behavior remains the default; release reanchoring is experimental.
- [Existing jump clearance](jump_baseline/README.md): 600 real transitions, 60.86 mm body rise but only 5.61 mm simultaneous wheel-bottom clearance.
- [New heading task implementation](heading_task_implementation/astra_task_contract.md): actual Opus implementation with explicit 85-dimensional observation and heading reward. Original response, actual provider receipt and root's modifications are recorded separately.
- [New heading training and development evaluation](heading_learning_01/README.md): three fresh seeds, 196,608 training transitions. All three 65k checkpoints complete the 32-second opened road; zero residual remains competitive. Full states, intermediate checkpoints and action saturation measurements are retained.
- [Stop, turn and yaw impulse probes](heading_g1_01/README.md): 24 episodes / 22,400 transitions with unchanged weights. All episodes finish, but all four controllers fail both parking cases. Zero, 49001 and 49003 pass the two yaw impulse cases; none passes the complete six-case gate.
- [Committed side-step input](side_input_commit/README.md): the course GUI now completes the current A/D cycle after release and repeats only while held. Ten physical cases / 21,900 transitions cover taps, repeats, reversals, cancellation and mode changes. This changes input handling; the measured gait still takes about 10.6 seconds per step.
- [Dynamic braking candidate](brake_dynamic_candidate/README.md): 12 runs / 26,400 transitions. The candidate regresses uphill rollback and repeated stops, so it was not integrated. Lossless evidence reuses the existing course-braking model blobs.
- [Wheel-PI stopping diagnosis and next contract](heading_stop_diagnosis_01/heading_zero_stop_readonly.md): two independent reads of existing G1 traces identify abrupt wheel arrest followed by body reversal. The fixed release-reference candidate is proposed and untested; there is no new physics or accepted braking improvement in this report.
- [Second milestone integration checks](integration_verification_02/summary.json): 3,314 tests pass, Ruff passes, 77 frozen inputs match; new packaged evidence is checked by SHA. These checks do not establish driving-task completion.

The [Chinese project note](../../docs/driving_optimization.md) explains current findings and the order of further work: stable driving, reliable obstacles, then higher speed. Reliable parking and turning, true jump clearance, obstacle traversal and GUI integration of a verified learned policy remain outstanding. Automated input tests do not establish human GUI acceptance.

The [Codex handoff](../../docs/codex_handoff_20260914.md) records the active objective, current entry points, constraints and the next bounded control experiment.
