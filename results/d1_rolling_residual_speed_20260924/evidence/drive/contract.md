# Fixed driving leg damping and transferred final policy: contract 04

2026-09-24. Actual GPT-6-astra ultra directs and independently reviews, actual GPT-6-sol implements the bounded new control/evaluation modules, and root alone owns engine execution, freezes, ledgers and publication. Actual Claude Opus may implement pure arithmetic tests or offline diagnostics under root coordination. The user has authorized finishing RL and speed validation; this experiment proceeds after a concrete source review without another permission round. New work directory: `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260924_drive_damping01` (D). Existing frozen files and outputs are never modified.

## What is known and the single hypothesis to test

The final shared-heave PPO checkpoint completed65536 training controls,512 returned optimizer calls and2048 epochs. Eight fixed evaluation records are complete across interrupted run_02 and fully closed run_03. All four plane cases pass; all four box cases pass crossing/stability/contact/stop checks but fail only the unchanged speed-error RMS gate. At0.2m/s box zero/policy RMS is0.09087787/0.09316863m/s; at0.25 it is0.10256116/0.10511265m/s. Learned weights changed, but these comparisons do not establish a useful RL effect. Do not retrain this shared-heave setup or call the original study qualified.

Archived body COM forward velocity minus mean relative wheel angular velocity times0.087m has RMS0.0776815 and0.0902712m/s in the0.2/0.25 policy box windows, versus0.0130462/0.0177546 on plane. All eight records have zero torque-protection-limited controls; the saved PI increment equation agrees to1.11e-16Nm. This is a kinematic proxy, not independent slip measurement. In the0.2 box zero record the greatest initial stall at tick364 precedes the first oracle ground-height transition at449; ground-height smoothing alone cannot explain that stall. Body-relative wheel-center motion also changes sign between stall/rebound, but finite differences are descriptive, not an exact sampled controller Jacobian or a proven cause.

Ranked explanations are (1) longitudinal leg/body compliance and pitch coupling contribute to body-speed oscillation despite regulated relative wheel speed, (2) edge-contact rolling/slip and body-speed feedback limitations, (3) later oracle height transitions. Saturation is unsupported by these records. Test only explanation1 now: extend the already fixed longitudinal joint-relative leg damping to the driving phase. The gain was previously qualified during stopping; its effectiveness or safety during driving has NOT yet been demonstrated. Prediction: box body-speed RMS decreases enough to meet the same gate while plane tracking, crossing and stopping remain qualified. Failure disproves adequacy of this single candidate; it does not authorize tuning or more heave training.

Pinned evidence under L=`stability_20260923_rl01`:
- `complete_cases_map_03.json`:55946f0d4c593db5e328b84021b5cbc8f299ce9b56c27e82ced71ae64e7b3dff.
- `run_03/continuation_receipt.json`:18d4b8417e6d7f13796b0bfb853ebed82b9dc82222e9e17fca29599bb9f42257.
- `root_run_03_readback_01.json`:53953c9a06f4a473a95f3bb720f53c0bb0c0761e2feb023c0b2ad5bbe48aaf09.
- `run_03_archive_manifest_01.json`:a029bba59746654a7f28e040fb0e0d39b969f2e8b918b9341aeef4edc9c198a3.
- `root_body_wheel_coupling_01.json`:e0912f42f74e5556aa17afc3fbf0c2dc9359088ecc9df4e89e5b393182b1864e.
- `astra_velocity_mechanism_crosscheck_01.json`:7999fb6ad0862b45f460e8c3792fed31d3116431d55730b31108fd92811487bf.
The old interrupted process has no final C counts; new experiment accounting never repairs or invents them.

## Exactly one numerical change

Import the unchanged constant `STOP_LEG_DAMPING_NSPM = 126.4374005337902`. No new gain, cap, filter, blending coefficient, terrain trigger, body-speed threshold, onset adjustment or parameter search. When the bound actually consumed raw forward command is nonzero, compute exactly the existing longitudinal damping arithmetic:

```
forward_axis = state.base_rotation[:, 0]
for leg in range(4):
    jx = forward_axis @ state.foot_jacobian[leg, :, :3]
    u[leg] = jx @ state.joint_velocity[4*leg:4*leg+3]
    delta[4*leg:4*leg+3] = -STOP_LEG_DAMPING_NSPM * jx * u[leg]
delta[WHEEL_INDICES] = 0.0
```

Raw-forward exact zero, including signed zero, makes this new stage numerically identity. Preserve the original binding/time/servo-forward checks. The original drive-to-stop latch and stop damping remain unchanged: the driving and post-stop increments never act together. Initial hold receives neither damping mechanism. Raw yaw/turn authority stays unchanged. All nominal targets, parent leg PD/support/attitude terms, wheel PI/antiwindup, residual action expansion and original limits remain frozen.

Add delta to the inherited UNCLIPPED requested torque. Repeat the original rated clip, then the original outward position/speed suppression in the same order and with the same constants/comparisons; never add torque after protection. Fold delta into `leg_pd_nm` so requested=leg feedback+support+wheel remains true. Wheel delta is exactlyzero, and the actual inherited wheel request/torque/memory are unchanged for the same sampled state/action. Invoke the original controller and PI update exactlyonce per control, with no shadow controller evaluation.

The new sampled raw increment power is `delta @ qdot = -b*sum(u*u)` up to floating arithmetic. It is not a claim about global/discrete passivity, native-substep work, or the protected torque increment. Record raw and actual protected increments separately.

## Real stage order and honest records

Use new cooperative classes without editing frozen parents:

```
class DriveDampingStageController(TurnYawAuthorityController): ...
class DriveRollingController(RollingResidualCompositionController,
                             DriveDampingStageController): ...
```

Require actual MRO `DriveRollingController -> RollingResidualCompositionController -> StopTurnCompositionController -> DriveDampingStageController -> TurnYawAuthorityController -> D1WheelLegController`. Consequently the frozen StopTurn `super().compute` really executes the new drive stage before constructing its own record. During drive the original inactive stop record naturally describes the already damped torque; during stopping the drive stage is identity and the original stop mechanism supplies its torque. Do not rewrite `last_damping`, project away executed torque, relabel a post-stop correction as old stop output, monkeypatch any frozen method, or evaluate a second parent law. Cooperative reset/initialization runs once and leaves no stale diagnostic.

An explicit new frozen result subclass may add a nested `drive_damping` record; inherited `dataclasses.replace` must preserve it. Give the new controller and task distinct schemas. The record contains active/raw gate, constant gain, actual consumed state rotation/joint position/joint velocity and projected four3-entry Jacobians, four relative speeds,16 raw increments, inherited unclipped request/leg feedback/protected torque, total request/protected torque, raw sampled power and actual protected increment. Values come from the same passed controller state; do not perform any new engine kinematics query for telemetry. Inactive diagnostics must identify the identity stage. Link recorded state inputs to the consumed provider state and available saved pre-control joint position/velocity/rotation under their actual semantics; do not substitute endpoint-after values.

Reuse the unchanged physical scorer and `_case` where possible. Add a new pure validator for the drive stage and composition chain. It must check the raw gate, exact wheel-zero increment, fixed formula, proper protection order, raw power identity, and that drive output is the actual stop input. At all ticks `stop_record.safe_torque_nm == controller_result.torque_nm == executed torque` remains the original honest check. During an active stop, the drive-stage intermediate safe torque need not equal final torque; validate the explicit stage chain instead. Missing/malformed/nonfinite drive records are record-invalid and fatal. Formula comparisons may use documented numerical-only roundoff tolerance (at most1e-10 absolute plus1e-12 relative for algebraic recomputation); no physical threshold changes. Preserve all old contact/raw-native integrity checks.

## Explicit frozen-checkpoint transfer

The model and its training metadata are unchanged:
- `L/run_02/training/final_checkpoint/model.zip`:4f59d795afbdec256ec17bb7256ea05a3ddd04561e9ba52beffb1de375196b2e.
- `model.metadata.json`:44b199ea13989412d274955a9fd85d6e97e58ad708a0c7e3ab7a4085ae3daaf3.
- `reload_probe.npz`:2cb674f54c8c46d5555e5d3580810a003be8d50fa5a7c1a5526dd0f027bb6660.
- Original `L/run_02/protocol.json`:494eb82ca80a842c190bbbff9bc75fa10c2e7832789a2d9bce0785acc3684733.

Construct the original box then plane rolling environments once, before any reset/control. Load this final checkpoint exactlyonce against the still-original box environment with the original strict loader, DSO SHA and training protocol. All original schema/action mapping,85-observation/1-action,65536/512/2048, parameter hash and two probe checks remain true. No learn/train/optimizer step or replacement checkpoint.

Then explicitly replace only the controller adapters on the two inactive, unprepared, never-reset environments with new DriveRollingController instances using their original control config. Do not replace/swap model,data,plant or compiled caches. Verify time0, no loop/decision/active episode, unchanged model/data identity and zero C/native delta across checkpoint load and transfer. This is a declared controller-law transfer, not original-law checkpoint compatibility. An explicit new wrapper publishes new task/controller schemas in reset info and episode metadata; retained training schema/config must be separately named as historical training context. Never feed a fake old-schema view of the new controller to the old loader. Record transfer before/after schemas, exact checkpoint hash, actual MRO, original gains and the single new gate. Observation/action meanings and order stay identical; policy predictions subsequently respond naturally to the new trajectory.

Two original constructors consume3 then2 hidden compiler steps; cold nominal cache becomes miss1/hit1 then miss1/hit3. Installing two new controllers calls only the already-warm nominal cache, ending at miss1/hit5. Enforce these exact counts and no additional model/step/forward from installation. No cache warming, clearing, dummy controller, smoke environment or engine checker.

## Four fixed paired intervention episodes

Run exactly this order in one new exclusive D/run_01:
1. speed200_plane_final_policy.
2. speed200_box_final_policy.
3. speed250_plane_final_policy.
4. speed250_box_final_policy.

All seed77351, onset175, stop975, terminal1200, worldz0.455, yaw0, default oracle, original plane/15mm box and original engine settings. Same deterministic checkpoint, scalar action gate/mapping, eight physical actions with four wheel residuals exactlyzero. Compare reset qpos,qvel,ctrl,qacc_warmstart,observation exactly to states[0] of the corresponding OLD FINAL_POLICY case in complete_cases_map_03. These are declared controlled intervention comparisons, not retries chosen to erase old failures. Preserve/report every result, including worse performance. No new zero-action or seed/case variants under this budget.

Reuse unchanged RollingEngineRuntime, NativeGeometryGuard, StreamingNativeObserver and the complete endpoint/trace/native/state artifacts. Each case reserves its own1200control/6000normal segment; unspent allowance is not transferable. Freeze the exact numerical modules and any new validator before execution. Valid ordinary task failure continues the remaining predetermined cases; warning, geometry/record failure, budget violation or exception stops, with partial evidence preserved and no retry. The new validator runs after each completed archived case before proceeding; live every-native geometry monitoring remains unchanged.

## One process, hard counts and durable boundaries

Fresh budget: exactly1 process,0 training, at most4800 evaluation controls and24000 normal native steps plus5 hidden compiler steps,600seconds. All old allocations are closed; none carries over. Root prepays the full reservation durably before Popen, records child PID/argv, and terminates at deadline without retry. Crash/disappearance consumes the full reservation, never recreating missing terminal counters.

Reuse successful site-first Python3.10 import/bootstrap sequence, exact six OpenCV source hashes and exact expected cv2 LD_LIBRARY_PATH mutation/restoration. Keep absent startup LD_LIBRARY_PATH, exact LD_PRELOAD and LD_BIND_NOW, original DSO3ec7ec9a6a130b9e1fa153c800aab97b59d4692c24971027f02de42957f8fa9c and four actual GOT binding checks. Do not relax binding/helper environment validation. CPU/torch/math threads1. Archive actual binding proof and initial zero C/Python snapshot BEFORE construction, warning callback before construction, and restore it in finally. Do not invoke old study/continuation main.

Persist/fsync each case BEFORE and AFTER receipt with actual C/Python/native-monitor snapshots, spec/checkpoint/baseline initial-state references, actual deltas and exception. At final return require construction attempts=returns5, actual normal attempts=returns=5*actual completed controls, Python attempts/returns/clock aligned, monitor checked=passed=normal returns, actual CCD/compiler callers verified, no warnings/failed/forbidden/violations, phase0/target0, exact four reserved segment limits and source hashes unchanged. Distinguish complete task qualification from a valid ordinary early failure. No assumed terminal C receipt on interruption.

Freeze all prior527 run_03 inputs, run_02 original55-file archive, run_03 full archive via its manifest, complete map and selected baseline states/scores, fixed model/metadata/probe/protocol, this contract, new modules/tests/launcher/freezer and Astra GO input snapshot. Check files individually before/after; newly added repository files do not invalidate an old dynamic directory listing. Never modify an old manifest or source. Pure AST/Ruff is permitted. Root may run at most two bounded rounds of at most12 new pure NumPy/JSON/AST tests each, no MuJoCo import or engine, focused on this formula/protection/stage order/record validation. They are separate from the one actual physical process and from subsequent normal CI. No old physical batch, training or broad local test suite is repeated.

## Acceptance, attribution and exit

Use all unchanged original physical/record/geometry gates from scientific contract SHA500a00e22f86600b986e7d537e8d891c09888fb33e5f11a8a2a774deae7f10e0 and its clarification. In particular all-time posture/yaw/lateral bounds, authentic full-contact geometry and wheel load, no nonwheel contact, full-robot box far-edge clearance, exact final endpoints1101..1200 and native5500..5999, and truthful torque/state/action chain. Keep speed window endpoints275..875 inclusive, mean body COM vx>=0.9command, RMS(vx-command)<=0.05m/s. Both terrain policy mean(0.25)-mean(0.2)>=0.03m/s. No window/threshold/schema spoofing to pass.

Success requires four complete record-valid task-passing episodes, new drive-stage chain valid and exact accounting. Report speed mean/RMS, maximum posture excursion, wheel loading, clearance and final-stop metrics side by side with old same-speed final-policy cases. If the candidate passes, the supported claim is fixed driving damping plus the existing transferred policy passes these four deterministic simulation tasks up to0.25m/s. The learned policy's independent benefit remains unproved, since a zero-policy comparison under the NEW damping law was not run. Improved tracking relative to the old same-policy law is attributed to the declared controller intervention, not to additional learning. Neither four seeds nor hardware/statistical reliability is claimed.

If any unchanged task gate fails, keep it failed. Root and Astra use this complete finite intervention evidence to identify what was/was not improved; no automatic same-gain tuning, extra seed, extra training or second physical run is authorized here. This one candidate is the next concrete engineering action toward the user's goal, not a claim that its hypothesis is already verified. Root proceeds directly to implementation/freeze/one execution after the final source GO, rather than ending at this plan.
