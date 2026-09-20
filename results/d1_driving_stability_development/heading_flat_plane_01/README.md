# Native-plane zero-residual development baseline

This new plant identity uses the frozen builder's native horizontal plane and a finite ground-reference/task domain. Robot mechanics, friction, contact material, solver, nominal controller, raw commands and original G1 gates are unchanged. Old heading policies are rejected before deserialization. There is no release reference shaping, extra damping, yaw-limit increase, training or default change.

Six original command profiles completed: 5600 control transitions / 28000 native substeps. The first invocation completed both 800-step stop cases, then rejected the reverse initial observation because three exact zero fields had different sign bits. The complete interrupted record is retained. The next invocation reused those two cases byte-for-byte and executed only the remaining 4000 transitions / 20000 substeps. `episodes/execution_carry.json` and per-case origins in the aggregate summary separate reuse from new execution. Only cross-direction initial observations canonicalize exact signed zero; physical states, nonzero observation values, same-command prefixes, raw traces and scoring remain unchanged.

| Case | Original gates | Relevant result |
|---|---|---|
| Forward stop | Fail | raw velocity RMS .05904 m/s; late peak .08436 m/s; late path .06299 m |
| Reverse stop | Fail | raw velocity RMS .05878 m/s; late peak .08424 m/s; late path .06282 m |
| Left stationary turn | Fail | peak heading error .234134 rad (13.415 degrees), above original 5 degree limit |
| Right stationary turn | Fail | peak heading error .234213 rad (13.419 degrees), above original 5 degree limit |
| Positive yaw impulse | Pass | actual +.1 N m s; original full task gates |
| Negative yaw impulse | Pass | actual -.1 N m s; original full task gates |

All recorded native wheel-floor contact normals have zero horizontal component. Both stop failures retain all three original failed gates; each turn fails only peak heading error. Geometry validity and execution evidence do not imply successful driving. The new plane removes the observed flat-hfield normal anomaly but does not fix the controller's stop/turn deficits.

The nonphysical preflight covers 60 tests, 188 compiled-model invariants and 37 saved poses with 207 active wheel-floor constraints. The root file-only audit recomputes native timing, applied control, external impulse, contact force decomposition/wrench sums, T/T+1 state correspondence and original raw stopping metrics. Active constraints need not carry positive force. Native force logs read the existing solver cache; synchronized endpoint contact kinematics are separately labelled. Missing contact is counted, not reported as zero slip.

Each leaf `complete_manifest.json` was written after files were closed; the original G1 leaf manifest predates the extra close receipts and is not the complete instrumented inventory. The top manifest covers all delivered evidence, including interrupted attempts. Frozen77 and all old experiment records remain unchanged. Jumping, steps, speed increases, manual driving and physical self-righting remain unverified; R remains simulator reset and the timed-out new RL GUI has no generated implementation.
