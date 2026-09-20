# Restored original yaw feedback passes the two original turn cases

The independent native-plane, oracle-state, zero8 candidate passed every original G1 gate for both stationary turns. This is a finite development result. Stop/turn composition and robust driving are not qualified; default entry points remain unchanged.

| Case | Original plane heading peak | Candidate peak | Original gate result |
|---|---:|---:|---|
| Left stationary turn | .234134 rad | .063933 rad (3.663 deg) | all pass |
| Right stationary turn | .234213 rad | .063963 rad (3.665 deg) | all pass |
| Forward stop | original trajectory | complete bitwise no-op | three original failures retained |
| Reverse stop | original trajectory | complete bitwise no-op | three original failures retained |

One new four-case batch completed3200 control intervals /16000 native substeps. Corresponding `flat_plane_02` baselines were read, never rerun. Original raw commands, heading reference and all original gates, including the5-degree peak limit, were retained. A first launch used a nonexistent protocol path and failed during input hashing, before output creation, environment construction or physics; the zero-physics failure and corrected original path are recorded.

The sole intervention is to restore the original `servo_yaw + 4*(servo_yaw-body_yaw_rate)` during raw forward==0 and raw yaw!=0, without the inner ±.6 clamp. There is no new numeric cap, dynamic headroom limiter, leg damper, wheel-center compensation or latch. The original final wheel target ±30rad/s, wheel PI2.2/3, integral±4Nm, antiwindup, rated/outward protection, original leg PD/support and outer heading law remain. Original frozen77 files are unchanged.

The original baseline's inner .6 clamp masked the existing body-rate feedback during all50 pulse intervals. The restored formula changes the feedback response, but successful finite turns do not establish that this was the sole failure mechanism or remove contact/geometry limitations. The old1.0-cap experiment failed on the old hfield plant and kept clipping; it is preserved separately.

Actual wheel targets peaked at11.85298rad/s; no target clipping occurred. Unlimited wheel request peaked at23.09777Nm, with actual12Nm protection. Each turn had8 protected joint-intervals. Actual torque saturation is distinguished from target clipping and from removed inner-cap occupancy. Native mean pulse yaw moments were +3.17557/−3.17612Nm. Individual contact friction utilization reaches1; aggregate loading is not a guarantee against local sliding.

gpt-6-astra / ultra supplied the fixed method,208-pose qualification helper and independent source review. Root ran the helper and104 independent saved-state core calls, whose eight compared quantities had zero error; these restart original PI memory and are not a new trajectory.29 nonintegrating tests passed, followed only by an import-order correction and repository Ruff. Sampled rigid-leg damping proxies and headroom intervals were documented as diagnostics, not loaded closed-loop stability proofs.

An actual bounded Claude Opus call (provider `claude-opus-5`, $0.34730375) wrote the core. Original request/response/source/receipt and root integration diff are preserved. Root corrected documentation, diagnostic tolerance and export order; the control formula was unchanged. Root authored the environment/runner/tests, executed physics and reconstructed all3204 saved states, original/restored target formula, PI, protections, native and synchronized endpoint evidence, and original raw scores. The independent audit passed3,046,809 checks with no new integration, contact solve or controller calls. The archived base leg PD/support request is not independently reimplemented.

Before the turn pulse, execution0..199, physical states0..200, observations0..199 and native entries0..999 are bitwise equal to baseline. Observation200 is excluded because its preview target changes. Both no-op stops preserve every original800/801 record and all original summary fields except model, including all three failed gates. They do not contain the independently passing post-stop damper. Four initial native substeps without wheel support lie within the unchanged settling prefix in each episode.

The next task is a separately specified composition of the two qualified mechanisms, with transition cases and original scoring. This package does not establish GUI/manual driving, reliable jumping, obstacles, speed or physical self-righting. The earlier new-GUI generation timed out with no code; R remains simulation reset. No training or full24G1 replay was performed; older experiments were not edited.
