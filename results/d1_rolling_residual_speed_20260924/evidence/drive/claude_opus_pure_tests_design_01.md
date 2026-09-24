# Claude pure arithmetic test design 01 (drive damping candidate)

Target: `drive_damping_math_04.py` only, imported by module name (root adds its directory to
`sys.path`). Standard library + NumPy + pytest. No engine/model/controller import, no mocks.

Eight pytest functions in `test_drive_damping_math_opus_01.py`:
1. `test_signed_zero_command_is_inactive_identity_despite_motion` — `+0.0` and `-0.0` are inactive
   identity (zero delta, zero raw power) while the four relative speeds stay nonzero.
2. `test_gate_is_direction_and_magnitude_independent` — `0.4`, `-0.4`, `2.5`, `-1e-9` all activate
   bitwise-identical damping; the gain matches the frozen constant.
3. `test_rotation_projects_first_column_not_first_row` — cyclic permutation rotation (orthogonal,
   det `+1`); a diagonal block makes column-0 projection `(0,2,0)` differ from row-0 `(0,0,4)`.
4. `test_four_leg_delta_power_identity_and_wheel_independence` — hand `jx`/`u` per leg give the fixed
   delta and, independently, sample raw power `-b*sum(u^2) = -30b` (explicitly NOT global passivity);
   two wheel-velocity sets leave leg delta and power bit-identical, wheel delta exactly `0.0`.
5. `test_stationary_legs_with_spinning_wheels_give_zero_damping` — active gate, zero damping.
6. `test_rated_clip_then_outward_position_suppression` — frozen limit arrays, default interior finite
   positions, rated clip, then exact-boundary outward zeroing with inward torque allowed.
7. `test_exact_speed_boundary_suppresses_only_outward_torque` — exact `±20`/`±30` boundaries, both
   signs, wheels included, zero-torque case not suppressed, `limited` indices checked.
8. `test_malformed_inputs_are_rejected` — one loop over 15 shape/dtype/bool/nonfinite cases.

Formula tolerance is `rtol=1e-12, atol=1e-10`; exactness assertions use `array_equal`. Expected
values are hand-derived, never obtained from the functions under test.

I have NOT executed these tests, ruff or any shell command; root alone runs them. Unresolved: stage
order, MRO, record schema, protected-increment recording and global/discrete passivity are outside
this pure module and remain untested here; the `velocity == limit` with zero torque case encodes my
reading of `safe * velocity > 0` as non-suppressing — confirm that is intended.
