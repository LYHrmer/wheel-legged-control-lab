# Frozen next contract 06: one common wheel proportional feedback correction

Authority: the user requests completion of RL and speed development under actual Astra-ultra direction. This is the next bounded engineering action after valid stage 05; no additional user approval is required for ordinary implementation/verification already authorized. Root alone executes engines/tests. Astra reviews; actual Sol or actual Opus may implement with provider identity recorded. This contract does not activate itself: final code/input hash review and root's exclusive budget activation precede the one process.

## Inputs and closed history

R=/home/lyh/wheel-legged-control-lab
B=/home/lyh/wheel-legged-control-lab-work/recovery-20260912
L=B/stability_20260923_rl01
D1=B/stability_20260924_drive_damping01
D2=B/stability_20260924_drive_damping02
New W=B/stability_20260924_body_speed01. All new implementation/results are exclusive new files in W. No edits to old frozen files, source, checkpoint, evidence or contracts.

05 closed with exact 4800 control / 24000 normal / 5 compiler, valid records and only two box speed-RMS task failures. Inherit and hash-verify every input of D2/root_execution_freeze_01.json (SHA 3de560e0a46e456b72b30661ec497ec1a1cc497abf49fcf33ab46553a8bc12b9; 662 inputs) and all 77 files in D2/run_01_archive_manifest_01.json (SHA dc547afa31d65b099a6de93d50d796eb1f7956a2f9ac1dc26806c70d4712912c). Bind D2/run_01_boundary_closure_01.json (61a52581b043dc32f804fcb67a215427084a85664c6ec0b16d2ba4673f779a4f), root readback and Astra result/mechanism reports. Old reservations remain closed; no old process or batch restarts.

Scientific predecessor D1 numerical modules and L/next_drive_damping_speed_plan_04/next_contract.md (f2d2d02b1ad525b7a16e9a42efaa29cfa4267574b407e83ea7876944c1bae348) remain unchanged. Reuse the qualified fresh-lifecycle helper D2/drive_fresh_lifecycle_05.py (2a80b2f1c6a50127639447bfb011c436f4a7f4876285cf50f58487fd21ef5191); do not reintroduce a read of absent _steps before reset. No source-wide dynamic glob may invalidate already fixed historical manifests.

## Exactly one numerical change

Keep the 05 driving leg damper at 126.4374005337902 N*s/m, old stop/yaw composition, support/attitude law, original wheel PI gains 2.2/3.0, integral bounds +/-4 Nm, original antiwindup, action map, targets, radius and protections. Original wheel PI computes once per actual control. No new integral, altered raw/servo/nominal target, fitted gain, action authority, timing, state override, terrain cue, future sample, or controller reset at command boundaries.

Use the actually consumed raw binding and same synchronized state. Let omega = state.joint_velocity[[3,7,11,15]] in rad/s and vx = state.base_linear_velocity_body[0], the base inertial COM velocity projected on visible base_link x. It is NOT whole-system COM velocity or visible-origin qvel alone. Let Kp=2.2 and r=0.087 exactly, asserted equal to the frozen controller's original constants. Compute, in this order:

    active = raw_forward_mps != 0.0
    common_error_rad_s = mean(omega) - vx / r
    scalar_delta_nm = Kp * common_error_rad_s if active else 0.0
    delta16 = zeros(16); delta16[[3,7,11,15]] = scalar_delta_nm

The same common delta is applied to four wheel channels, leg channels exactly zero. Both signed zeros of raw forward give inactive identity; neither yaw alone nor terrain controls this gate. No smoothing, separate gain or clamp of the new term. Add to the preceding stage's UNPROTECTED request and wheel_nm, then apply original rated clipping and original outward joint position/speed suppression in the original order. Preserve nominal wheel targets, original parent memory_before/after and leg/support components. Raw and protected increments are distinct records. Original integral/antiwindup continue using original wheel error and original PI request; do not describe them as body-error integration or silently alter them to account for added torque. Actual total protection may occur and must be logged.

Without target/torque protection, the mean proportional contribution becomes Kp*(mean(original wheel target)-vx/r), while individual differential wheel P and the old I remain. That identity is an explanation, not a physical-pass proof. No isolated slip/traction or passivity claim.

## Truthful integration and records

Use a new explicit controller schema d1-rolling-shared-leg-drive-jx-body-common-p-stop-turn-v1 and task schema d1-rolling-residual-drive-jx-body-common-p-transfer-v1. Record schema d1-drive-body-common-p-record-v1. Keep observation/action/control-loop schema meanings unchanged; distinguish the controller law from loop schema. Historical checkpoint training definition is archived separately from the actual new task definition.

Minimal cooperative chain: NewRolling -> RollingResidual -> StopTurn -> NewBodyPStage -> frozen DriveDampingStage -> frozen TurnYaw -> original D1WheelLegController. NewBodyPStage subclasses DriveDampingStage; NewRolling composes RollingResidual and NewBodyPStage. Assert real MRO. Each parent compute/reset runs once. The new stage reads the real DriveWheelLegResult, produces a result subclass retaining drive_damping plus body_common_p, updates only wheel_nm/request/protected/limited fields, and returns through real StopTurn. StopTurn dataclasses.replace must preserve new fields. Pre-parent validate action, raw binding, finite same-time provider values before any PI mutation. Pure nominal preview must remain inherited and must not mutate diagnostics or PI.

Keep authority_record exactly what original TurnYaw computed; it is true INTERMEDIATE output, not final wheel torque. Keep drive_damping exact original intermediate record. Do not change either to make old validator pass. Add a NEW pure validator explicitly linking:

1. original authority wheel requests/torques and original PI memory -> drive base request/protection;
2. frozen drive formula and protection -> actual body-P base request/protection;
3. exact common delta, wheel-only sum and protection -> actual stop base request/protection;
4. actual stop damping/protection -> controller result, trace executed torque and native ctrl.

The 04 validator's final authority.wheel_torque==final-wheel assertion is intentionally inapplicable; do not call it on fabricated/projected rows. Its pure drive/precontrol algebra may be reused directly only where assumptions still hold, or copied narrowly into a new validator with the changed chain explicit. Retain algebra tolerance atol1e-10/rtol1e-12, all actual identity checks exact where copied arrays share a source. Frozen rolling scorer lines 91-124 checks final controller/stop/trace and does not impose authority-final wheel equality; reuse it unchanged on actual records. No new physical scorer/threshold is needed.

New record includes active/raw/servo/time/sequence/age; exact provider body COM velocity vector, rotation, joint q/qdot; measured wheel omega/mean, vx/r, original Kp/r, common error and delta16; actual parent request/protected/wheel_nm, actual sum/protected/wheel_nm, actual protected increment/limited flags, original PI before/after and unchanged nominal/final wheel targets. Preserve diagnostics sufficient to recompute common-P identity using true parent wheel error and integral. No counterfactual field may be labelled actual execution.

Extend simultaneous precontrol records to bind provider body velocity to saved raw qpos/qvel and compiled base_body_ipos_local_m: v_body = R.T @ qvel[:3] + cross(qvel[3:6], body_ipos). Check full vector finite and algebraically consistent, vx matches same-tick endpoint body_vx; do not use whole-system COM endpoint fields. Joint addresses/rotation/state phase checks from 05 remain. Observe before step using the prepared heading_decision.decision.context.state; preserve post-return association and partial failure records.

## Checkpoint transfer, cases, fixed acceptance

Use the SAME final checkpoint L/run_02/training/final_checkpoint/model.zip SHA 4f59d795afbdec256ec17bb7256ea05a3ddd04561e9ba52beffb1de375196b2e and its original strict metadata/probe. Zero learning, optimizer updates or checkpoint writes. Construct original box then plane environments once under construction phase (3+2 hidden compiler steps). Load original checkpoint once while box still has original controller/schema, validate two saved probes without reset/step; then explicit new-law transfer on BOTH never-reset environments using the 05 helper, preserving model/data/address/time. New controllers use warmed nominal geometry only; expected cache miss1/hit3 before transfer -> miss1/hit5 after transfer. No preliminary model smoke or engine probe.

Four deterministic final-policy episodes in order: speed200_plane_final_policy, speed200_box_final_policy, speed250_plane_final_policy, speed250_box_final_policy. Each seed77351, onset175, stop975, terminal1200, raw worldz0.455, yaw0, same plane or15mm box, provider/engine/solver and policy scalar expansion. No new zero-policy episodes or seed changes. Before physics compare reset qpos/qvel/ctrl/qacc_warmstart/observation exactly with BOTH the corresponding saved 05 and historical final-policy initial arrays; keep all sources immutable. The policy may respond differently after intervention; this is not a replay of action arrays.

Per case: unchanged complete raw/native/endpoint/trace recorder and rolling physical scorer; complete 1200/6000 required for task passage. Exact fixed speed window endpoints275..875 inclusive601, mean >=0.9*command and RMS <=0.05 m/s. Retain all old all-time attitude/yaw/lateral, nonwheel contact, every native raw/geometry, positive box wheel contact, crossing and final stable gates without widening: roll/pitch10deg, yaw5deg, lateral0.1m; no nonwheel terrain contact; robot all collision geometry beyond far edge with old margin; endpoint1101..1200 final |bodyvx|<=.03, |whole COM vz|<=.03, zRMSE(.455)<=.015; native5500..5999 wheel positive fractions>=.95. Original box positive-contact gate requires >=one positively loaded wheel and records actual per-wheel indices, not a fabricated four-wheel load gate. Any true record/geometry invalidity or warning aborts; ordinary valid task failure continues remaining fixed cases. Partial valid episodes never count as complete passage.

Candidate qualified only if all4 complete original tasks and new stage chains pass, and actual mean speed(.25)-mean speed(.2)>=.03 m/s on each terrain. Preserve separate execution_valid/task/aggregate fields; no truncation, filtered speed window or rounding to .05. Report paired effects against same-policy05 and historical zero/policy baselines. No independent RL benefit attribution without an appropriate same-law zero comparator; the intervention may qualify a controller carrying the fixed learned policy.

## Fresh finite execution and verification budget

Exactly one new engine process, wall limit600s; 0 training controls, max4800 evaluation controls, max24000 normal native returns (5 per completed control, max1200/6000 per case, no borrowing), max5 compiler hidden native calls. Total allowed step calls<=24005. No other model construction, mj_step, mj_forward, collision replay, engine diagnostic run or engine test outside that unique worker. Existing constructor/reset/recording non-integration kinematics/forward/setConst operations only as the frozen 05 workflow requires; record their actual counters, do not invent a separate allowance for experiments. Existing patched DSO unchanged; four actual GOT bindings, caller origins, hard C and Python budgets, synchronized target phase and protections unchanged. C CCD calls within the valid trajectory are counted but not separate experiments.

Root launcher durably reserves the entire new maximum before Popen, binds exact argv/output/environment/source hashes and child PID, timeout/no retry. On process loss reserve fully consumed; per-case/initial receipts establish available actual lower bounds, never reconstruct missing final counters. Preserve failed/partial artifacts exclusively. Root controls all invocation; reviewers never invoke engine.

Use site-first Python3.10, unchanged LD_PRELOAD/LD_BIND_NOW and exact successful cv2 environment-restore helper, CPU threading and fixed libs. Record durable actual binding proof and initial zero C/Python state before construction. Each case before/after persists actual C/Python/monitor/proof hash, compiler5, monotonic CCD, target/phase, controls and native5T; source hashes before/after. Final C attempts/returns and Python/monitor must agree with actual counts, failed/forbidden/warnings0, phase0/target0. Report actual completed counts independently of scored-case list. Full successful execution reaches4800/24000/5 but actual valid shorter task failures are not falsified as full.

At most two rounds of <=12 new pure tests, root only, engine/gym/torch/scripts imports forbidden except safe source extraction/stubs. Use one round if it passes. Tests must exercise ACTUAL new pure math/validator and actual new-stage method through a pure fake-parent/source seam, not only copied formulas: common/differential identity including unequal wheel targets, real sign/units, active signedzero identity, NaN/shape rejection, leg zero and four equal wheel deltas, original order outward/rated protection, original PI memory not advanced twice, real intermediate chain mutations rejected, same-time COM full-vector raw binding, source MRO and never-reset helper reuse. Existing frozen tests are not rerun in this local bounded batch. Pure AST/Ruff/link-free checks and offline archived analysis are allowed; no tests may instantiate a real environment.

New modules should be small: pure common-P math, cooperative controller/record, explicit new-chain validator, narrow05 transfer/worker derivatives. Root handles launcher/freezer/readback; use existing files as imports unchanged where valid. Final Astra GO binds exact implementation, tests/receipts and dependencies before execution. No provisional GO based solely on AST or pure math.

## Exit and interpretation

On qualification, report actual deterministic simulated straight/obstacle drive and stop at0.2/0.25 with preserved gates, successful learned-checkpoint execution, and attribution limits. Freeze complete records and portable offline verification for publication; no claim of hardware, GUI, self-righting or statistical robustness. No extra rollout is implied by success.

On failure, preserve this single candidate's complete actual result and diagnose the specific remaining source from true torque/protection/contact records. No gain/seed sweep, same-type blind retraining, gate relaxation or silent retry. A further numerical candidate needs a separate finite evidence-based decision; user goal remains active, not satisfied by calling a failure training success. Implementation defects before any control also consume this process and require a scoped new reservation, as05 did. No old budget is inherited.
