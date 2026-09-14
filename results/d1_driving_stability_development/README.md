# D1 driving stability development

This is an open development record. It does not change the frozen formal studies or establish completion of the driving optimization goal. GUI currently uses classic LQR/VMC; these PPO results use the synchronized learning control chain.

- [Yaw ablation](yaw_ablation/README.md): 19 closed-loop episodes / 56,844 transitions, all failures retained. Actual Opus implemented the runner.
- [Heading hold](heading_hold/summary.json): 14 episodes with unchanged old weights; complete policy episodes increased from 1/6 to 4/6 on the opened road. The remaining failures prevent a complete stability claim. NPZ and lossless gzip traces preserve all samples; `independent_audit.json` records 311 checks (the audit was executed inline, not supplied as a standalone script).
- [Braking candidates](course_braking/README.md): reference interventions and their terrain-dependent regressions. Original behavior remains the default; release reanchoring is experimental.
- [Existing jump clearance](jump_baseline/README.md): 600 real transitions, 60.86 mm body rise but only 5.61 mm simultaneous wheel-bottom clearance.
- [New heading task implementation](heading_task_implementation/astra_task_contract.md): actual Opus implementation with explicit 85-dimensional observation and heading reward. Original response, actual provider receipt and root's modifications are recorded separately.

The [Chinese project note](../../docs/driving_optimization.md) explains current findings and the order of further work: stable driving, reliable obstacles, then higher speed. A/D tap behavior, true jump clearance and GUI integration of a verified learned policy remain outstanding.
