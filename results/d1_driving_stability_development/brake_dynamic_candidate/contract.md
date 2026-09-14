# Dynamic brake → hold: frozen work-only contract

The prior full-reference candidate improved some flat release metrics but increased uphill backward retreat. Legacy remains the product default. This development experiment compares legacy with one dynamic reference candidate, using unchanged real MuJoCo physics and course gains. It neither trains RL nor edits repository files.

## Fixed algorithm

- Only a previous actually accepted eligible nonzero legacy forward command followed by a requested zero starts braking. Exact command-zero tolerance is 1e-12 m/s.
- Requested velocity remains exactly zero during braking/holding. No change to command, torque, actuator limit, world state, controller reset or gains is permitted.
- At each eligible zero-command braking tick, read controller memory distance and reference before compute, and current measured body-forward velocity from teleop._state.base_linear_velocity_body[0]. This is the same velocity convention as legacy distance integration.
- With a fixed nominal deceleration a=0.5 m/s², calculate b=v*v/(2*a) and replace reference by distance+clip(old_reference-distance, -b, +b). This is a symmetric bound on the existing error; it does not force an error sign or retain the original release location as an anchor.
- Count consecutive pre-compute ticks with |v|<=0.03 m/s, control_dt=0.01 s. After 30 such ticks (0.30 s), set reference to current controller memory distance exactly once and enter hold. Any higher-speed tick resets the count. Hold leaves that reference unchanged until the next nonzero request or loss of eligibility; it does not reacquire merely because speed increases.
- New nonzero request cancels braking/hold before compute, allowing normal legacy integration that tick. Loss of ownership, pending/active jump, recovery, low height and reset clear edge/phase/dwell without creating a later false release. observe_applied receives actual status after legacy compute and clears braking if an actual nonzero command or ineligibility occurs.
- The only controller assignment is _distance_reference_m. A non-None _prepared_control is an error; do not clear or modify it. Validate finite real values and differences and boolean eligibility before any state consumption or controller assignment.
- The limit is explicitly on pre-compute controller memory. Legacy compute then advances distance by v*0.01 and clips the retained reference within its existing ±0.55 m range. Its actual compute error is clip(e_pre-v*0.01, -0.55, +0.55); at hold entry it is clip(-v*0.01, -0.55, +0.55), not necessarily zero. Record both pre- and post-compute memory without changing this legacy integration.
- Hold means this helper stops writing. The unchanged legacy ±0.55 m clamp can still move the actual retained reference after sufficiently large drift; do not equate helper hold_reference_m with an unconditional immutable low-level reference.
- reset clears all helper state/counters. prepare consumes its prior-command edge after successful validation. The helper records phase, release count, clamp/write flags, hold-entry flag, old/new error, bound, speed, dwell count and hold reference every prepare call. It has no physical-state access beyond caller-supplied measured velocity.

## Fixed cases and timing

All cases settle 2 s, use CourseKeyboardCommands held-key sets and actual existing Shift gears, preserve heading feedback and all controller guards, use oracle state and LQR legacy allocation, no GUI and no midrun reset.

| Case | Arena/zone | Gear/key | Held intervals (s) | Duration (s) |
|---|---|---|---|---:|
| flat3 | flat/start | 3/W (+0.5 m/s) | [2,10) | 18 |
| ramp2 | course/ramp | 2/W (+0.4 m/s) | [2,14) | 22 |
| rough3 | course/rough | 3/W (+0.5 m/s) | [2,14) | 22 |
| stairs3 | course/stairs | 3/W (+0.5 m/s) | [2,14) | 22 |
| flat_reverse3 | flat/start | 3/S (-0.5 m/s) | [2,10) | 18 |
| flat_stop_go3 | flat/start | 3/W (+0.5 m/s) | [2,8), [16,22) | 30 |

Both strategies run every case. Every release interval lasts exactly 8 s. No parameter scan or case/terrain conditional algorithm. Stop-go has two release windows and two genuine release events. The second drive differs physically as a consequence of the first stop; only the first release prefix is expected bitwise identical between strategies.

## Evidence and predeclared metrics

Save the complete states/torques/telemetry/source/model, actual command/diagnostic trace, initial hashes, source hashes before and after, complete step counts and failures. Compare initial and all first-release-prefix qpos/qvel/torques bitwise. For the four historical cases also compare legacy first 4 s release against the prior raw baseline to detect harness drift.

For each 8 s release, express longitudinal displacement along the last commanded direction (reverse uses -world_x), and additionally retain signed world_x quantities. Report maximum forward excursion, total backward travel, largest retreat from a prior running peak, maximum reverse velocity, terminal displacement, final-2 s mean signed velocity, mean absolute velocity, peak absolute velocity, signed drift, total travel, and continuous |v|<0.05 m/s time. Report yaw/lateral/roll/pitch, saturation, safety and traction. The controller's low-speed window uses body-forward velocity; independent result metrics use finite differences of integrated world_x and must not conflate the two.

Comparative diagnostic flags: candidate reduces both peak forward excursion and backward retreat (allow only 1e-6 numeric tolerance), final-2 s mean |world_vx|<0.02 m/s, final-2 s total travel<0.04 m, no recovery, finite states and actuator compliance. They are diagnostics, not task certification. Preserve all failed flags and cases. Do not select or deploy based only on smaller signed net displacement.

## Ranked falsifiable mechanisms

1. Retaining a fixed release reference pulls a moving robot back after passing that point: a velocity-scaled moving bound should reduce backward retreat versus full reanchor without changing the drive prefix.
2. On the uphill release, retained position error supplies traction: reducing it even dynamically may cause rollback. If that occurs, this reference-only change is insufficient for slope stopping.
3. Body velocity near zero can be transient while contact/attitude states still evolve: entering hold after the fixed low-speed window may be premature, visible as renewed motion after hold entry.
4. Legacy velocity/attitude force limits dominate some failures: if bounded reference removes error but final sustained speed remains poor, reference manipulation alone cannot meet the stopping requirement.

Opus writes the small core under a .4 USD, 900 s code-only budget using the existing run_opus_task.py wrapper. Retain its unedited response, original source and receipt, then document any review changes before physics.
