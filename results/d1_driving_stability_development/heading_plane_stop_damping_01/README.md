# Fixed stop leg damping on the native plane

All four candidate cases pass the original raw-command G1 gates. This is a finite zero-residual development result, not full driving or real-robot qualification. The default controller is unchanged.

The sole controller change is the fixed longitudinal leg damper, b=126.4374005337902 N s/m, implemented by an actual Claude Opus call and integrated by root following gpt-6-astra / ultra planning. It activates only on an executed nonzero-to-zero forward-command transition. There is no release ramp, wheel PI reset, yaw cap increase, new training or gain sweep. Root composed the separately reviewed native-plane environment and stop adapter before first reset.

| Original raw stop metric | Forward baseline | Forward candidate | Reverse baseline | Reverse candidate | Original limit |
|---|---:|---:|---:|---:|---:|
| Full raw velocity RMS, m/s | .059043 | .044766 | .058776 | .044838 | .05 |
| Late peak speed, m/s | .084356 | .005823 | .084236 | .006336 | .03 |
| Late cumulative planar path, m | .062993 | .002591 | .062819 | .002736 | .05 |

The remaining original stop gates also pass. No torque, position or speed protection activated. The maximum added leg torque is 7.640 / 7.622 N m. Recorded pre-protection and same-state protected incremental powers are nonpositive at the control sample; this is not an integrated-work or discrete closed-loop passivity proof.

Only four candidates were executed: two stops of 800 control intervals and two yaw impulses of 1200, totaling 4000 intervals / 20000 native substeps. Their four baselines were read from `../heading_flat_plane_01/episodes`, not rerun. Stop execution intervals 0..399 and states/observations 0..400 are byte-identical to the baseline. Both no-stop impulse trajectories are byte-identical for all 1200 intervals and 1201 states/observations, including raw/servo commands, torques, PI memory and full native entries. Actual external yaw impulses are +.1 and -.1 N m s. No same-case signed-zero exception is used.

Root's independent audit reconstructed the fixed damper from 4000 saved poses without integration, with maximum torque-equation discrepancy 1.33e-15 N m. It independently recomputed raw stopping metrics, compared complete native records, checked timing, applied control, force/wrench decompositions, torque protection and T/T+1 archives. Plane contact normals remained vertical. All 77 frozen source hashes and baseline files are unchanged.

The original fixed-gain derivation remains in `../heading_stop_kinematics_01`. The new plane-specific diagnosis was independently reproduced byte-for-byte by root (SHA256 0ccbdfdc4cc7234ae1395585785b28b343667d0b52da614f7959bb916d30edc7); its helper/report are preserved here. Ten additional nonintegrating tests cover the plane/damper composition, old checkpoint rejection, invalid actions and close failure. The preflight receipt explains the import-order-only difference between the live plane test file and its preserved original.

The previous .5 m/s² release reference failed and remains rejected. Original plane turn cases still fail peak heading error; turning, jumping, step traversal, higher speed, manual driving and physical self-righting are not established by this result. R remains simulator reset. The timed-out RL GUI generation produced no code. The three old 65k trainings and full old 24-case G1 suite were not repeated.
