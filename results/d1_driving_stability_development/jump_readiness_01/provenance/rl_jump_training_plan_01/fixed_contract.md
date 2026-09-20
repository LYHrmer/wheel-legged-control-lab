# Fresh PPO jump residual task: fixed training and independent evaluation

2026-09-20. Planner: explicitly requested GPT-6-astra / ultra. Actual Claude Opus implements/tests/static-checks the bounded files in `claude_execution_spec.md`; root reviews, integrates and runs the fixed experiment. This package is planning only: no model/controller construction, physics, training or old rollout replay.

## Decision: RL is the next main experiment

After root accepts the stop/turn composition's evidence and completes the already-proposed1200/6000 height probe, proceed to this fresh PPO task when the **instrumentation and physical records are valid**. The zero-action height probe need not pass jump clearance or landing gates. A failed zero jump is a baseline for learning, not a demand for another serial classical thrust/gain intervention. This decision supersedes the post-probe classical-mechanism branch in the immutable `jump_obstacle_next_plan_01/next_contract.md`; that old file is not edited.

Use existing SB3 PPO and the original eight physical residual channels. Do not rerun the old three65,536-step heading runs or the24-case G1 matrix. Do not load their policy/optimizer weights, broaden their checkpoint metadata, or describe the new task as reproducing those experiments. They trained forward motion with zero user yaw, not requested jumps; G1 failures were out-of-training-domain commands on a different ground representation. They are useful negative scope evidence, not a test of PPO learning this jump task.

The first RL objective is **a requested stationary20 mm net-clearance hop followed by a stable return**, including no-jump holds. It is not yet a learned obstacle-crossing or GUI policy. Existing composed zero-residual stop/turn control remains its separately qualified driving baseline. This task has no moving-forward/yaw request, no new terrain, no450 N injection and no learned self-righting.

## Physical action and independent environment

Create `D1JumpPPOEnv(D1HeadingTrackingEnv)` as a new task; install the existing `D1FlatPlanePlant` before reset, retaining the original `D1WheelLegControllerAdapter(D1WheelLegController(...defaults...))`. Do **not** inherit or loosen `D1FlatPlaneHeadingEnv`, `StopLegDampingController`, `TurnYawAuthorityController` or `StopTurnCompositionController`, which deliberately reject policy actions in their qualified interfaces. Do not duplicate their zero-only compute paths. On this stationary task raw forward/yaw are always0, so the stop latch never arms and authority is inactive; the original residual-capable parent is exactly the appropriate inactive low-level law. A zero-action physical comparison against the existing height probe verifies the thin new task seam once in preflight/evaluation, not via repeated old qualification.

Physical action remains float32 Box[-1,1]^8 in the original order: four independent leg-extension residuals scaled0.04 m and four independent wheel-speed residuals scaled4 rad/s. Preserve original IK table[-.08,.08] m, joint-target rate limits, wheel target±30 rad/s, wheel PI/antiwindup, rated torque80 N m on legs/12 N m on wheels, and original position/speed protections. The joint/controller/action schemas accurately retain these physical semantics; the new task/observation/reward schemas reject old policies.

These actions can change leg geometry throughout crouch, extension, airborne retraction and landing, while wheels can correct horizontal drift. Their4 cm per-leg target range is mechanically relevant to a2 cm clearance objective. The scaffold and actions can reach/clamp some extremes of the original IK table (e.g.−.050−.040<−.080); record actual rate/IK saturation. This is a feasible control interface, not a proof of jump reachability or a prediction that PPO will pass. No actuator authority is increased if it fails.

The raw height scaffold is fixed to the existing schedule relative to request tick s: baseline.455; [s,s+25) .405; [s+25,s+40) .500; [s+40,s+75) .455; [s+75,s+120) .435; thereafter.455. Forward/yaw remain0. The policy learns physical residuals on this scaffold, so report it as scaffolded residual jump learning, not an end-to-end discovered jump from a neutral command. No old LQR vertical forces or force-controller constructor is used. All600-step episodes end at6 s; no reset inside an episode.

## Observation95 with explicit task information

Keep the existing heading observation85 as an exact prefix. Append ten float32 entries in fixed order, clip to[-5,5] only as declared:

| Index | Added value | Source/meaning |
|---:|---|---|
|85|request clock|−1 before a request and throughout no-jump hold; after request clip((k−s)/120,0,1)|
|86|requested net wheel clearance/.04|0 before request/hold; .005/.04, .010/.04 or .020/.04 afterward|
|87..90|four contact booleans|current published oracle state, original leg order|
|91|valid clearance progress p in[0,1]|persistent achieved task progress, defined below;0 before request|
|92|airborne progress|before first valid flight: consecutive fully-unloaded native count/10 clipped[0,1]; thereafter1|
|93..94|world planar offset from reset origin/.10|current published base origin minus recorded published reset origin|

Do not leak the future request time or goal to the actor during settling. The policy only receives the jump request when its prepared raw command changes. Earlier case IDs/seeds/episode metadata are not observation features. Include current contact state and task progress so an MLP need not infer a hidden request or reward latch from identical85-value standing observations.

Call the schema `d1-jump-oracle-task95-v1`, not purely proprioceptive85. Contact/position fields use the published oracle state; achieved-progress features explicitly depend on simulator geometry/contact task accounting. This is an oracle simulation policy, not a sensor-ready robot policy. Reward/evaluation may use actual geometry/forces without pretending those were previously available in the old85 observation.

Normal next observations append updated progress to the parent's already-prepared next decision at endpoint k+1; executed reward/schedule uses pre-step tick k. Preserve the existing cached-terminal-observation semantics and do not bootstrap a terminal state whose next reference was rejected. Reset all progress/origin/task state once before parent's reset prepares tick0.

## Measured task progress and fixed reward

Reuse `scripts/d1_jump_readiness.py` directly: root integrates the actual Opus module SHA256 `54296d2e59029163ddb48f594f0dbf402deeae9ab4587627bc9f8aec7c0eedb1`. Bind using `bind_wheel_plane_geometry(model,wheel_body_ids,plane_geom_id=plant.floor_geom_id)` and sample synchronized arrays using `sample_wheel_clearance(binding,geom_xpos,geom_xmat)`. Consume its existing `wheel_bottom_gap_m`, `simultaneous_minimum_gap_m` and `contact_margin_m` keys; no extra geometry implementation/interface layer. Shift `fixed_height_command` by feeding validated physical tick k through `clip(k-s+200,0,600)` for profile requests; holds use its hold condition. Let g be the simultaneous minimum of four wheel-cylinder plane gaps at a synchronized endpoint; m is the compiled effective contact margin, normally.001 m; c=max(0,g−m) is **net useful gap**. Never use base rise or the independent maxima of four wheels as c.

For training the existing `D1Plant.step(..., measure_contact_wrench=True)` can read all five native solved-contact samples without additional stepping or contact solving. A narrow instance wrapper changes only this diagnostic keyword and preserves all torque/wrench/other arguments; restore it on close. Require `last_control_interval_contact_wrench.physics_sample_count==5`. A full control interval is unloaded only when all four `active_sample_fraction_by_wheel` and measured wheel-force vectors are zero and endpoint g>m. Two consecutive fully-unloaded control intervals conservatively certify10 recorded native evaluations/20 ms. This stricter quantization is fixed; do not retrospectively bridge loaded samples. Confirm via actual native records on evaluation.

Airborne progress counts consecutive unloaded native samples in increments5, resetting to0 on a loaded interval until a valid flight is established. Require a positive whole-robot COM vertical velocity when a prospective unloaded run begins; store that condition for the run. No-jump hold and pre-request activity never earn progress. The bonus window is [s,s+120); after that progress is frozen. The new environment computes true mass-weighted robot COM separately from base-body COM/origin (the approved wheel-geometry helper does not provide COM); it is used for qualification, not replaced by `qvel[2]`.

After a valid flight, set persistent flight_seen. During qualified unloaded intervals in the bonus window, update `p=max(p,clip(c/h,0,1))`, h being the episode's requested net clearance. On the first interval completing flight certification, allow the maximum net gap observed within that just-certified two-interval run; store that run maximum explicitly for deterministic accounting. Subsequent loaded/rebound intervals cannot increase p. Report progress and contact history in info. New episodes reset p/flight/run state.

Preserve and log all original parent reward contributions, but define the new PPO scalar explicitly:

1. Retain original velocity/yaw tracking, attitude, mechanical-power, action-change, heading-goal and termination contributions with their original coefficients and actual dt. Keep all original reward terms in `parent_reward_terms` so the original return is separately available.
2. Use the original height contribution except set its effective contribution to0 for a requested jump on execution ticks[s+25,s+120). This avoids directly penalizing the intended airborne rise against a support-height target. For hold and all other ticks, keep the original height term. The mask is command-time based and predeclared, not chosen after measured flight.
3. Add `clearance_progress = 2*(p_after−p_before)`, a bounded incremental event contribution, not multiplied by dt. Across one episode it totals at most2. Add `flight_once=1` only on the first transition to flight_seen. No reward for repeatedly jumping after the progress cap.
4. Add `stationary_offset = −actual_dt*.5*min((norm(base_xy−initial_xy)/.10)^2,4)` throughout. This supplies the missing incentive for the stationary displacement objective; the actor receives that relative position.
5. Add `landing_once=2` only on the genuine final600th transition if not terminated, flight_seen and p==1, all wheels currently have positive native normal load, abs(base-height−.455)≤.015 m, abs(body_vx)≤.03 m/s, abs(COM_vz)≤.03 m/s, abs roll/pitch≤10 degrees, raw heading error≤5 degrees and planar displacement≤.10 m. No terminal bonus on early termination, mere truncation of an incomplete trace, hold, missing flight, or partial progress. Evaluation requires the stricter whole late window, not just this final training reward condition.

Total positive jump-event contribution is bounded by5 per episode. Progress never increases while supported and time-driven phase labels do not count as takeoff/landing. Rewards are fixed before training; no weight search or best-return selection. Preserve original failure termination: clearance<.22 m, |actual roll/pitch|>.85 rad, nonwheel body contact, map/query/envelope exit. A finite environment failure is a terminated sample with the original penalty; a numerical, source or accounting exception stops the run and is not fabricated as a learnable transition. Normal time limit600 is truncated, not terminated; keep stock PPO time-limit bootstrapping semantics. Hard readiness gates stay stricter than the training fall boundary.

## Training: one fresh bounded run

Reuse stock `MlpPolicy`, `PPO_SETTINGS` and the existing `AfterUpdatePPO` observer pattern. Formal settings: learning_rate3e−4, n_steps128, batch_size128, n_epochs4, gamma=exp(−.01/2), gae_lambda.95, clip_range.2, ent_coef0, net_arch[64,64], log_std_init−2, CPU, one environment, one Torch thread. No algorithm fork, normalization wrapper, exploration retuning, restart from an earlier checkpoint, or policy pretraining/distillation.

First run one pipeline smoke with fresh seed62999,640 physical training transitions, n_steps128/batch128/n_epochs1; fixed first episode requested h=.005,s=200, followed by one hold episode. If the first episode completes, this covers600 steps plus40 after stock auto-reset. Early task termination remains a recorded failure and consumes actual transitions; do not pad or restart the smoke budget. Discard its policy and optimizer. Its performance is not selection evidence.

Then start formal fresh seed62001 once, **131072 actual training transitions**, exactly1024 PPO train calls /4096 optimization epochs if every rollout/update completes. Save declared checkpoints32768,65536 and131072 only after the complete PPO.train returns. Final131072 is the only evaluated learned checkpoint; intermediate files are progress/provenance, not a model-selection pool. Initial parameter hash, fresh optimizer/counters and all actual hyperparameters are recorded. One uninterrupted learn call, no reset_num_timesteps reuse and no old65k deserialization.

Curriculum uses **actual successful training transitions at episode reset**; an active episode retains its chosen target:

- total<32768: h=.005 m net gap;
- 32768≤total<65536: h=.010 m;
- total≥65536: h=.020 m.

Four requested-jump episodes followed by one no-jump hold, repeated by completed/reset episode index. Jump request ticks cycle150,200,250 by jump-episode index; no holdout175/225 in training. All training friction scale1.0, original masses/actuation/noise/delay. Episode RNG/reset seeds are `620010000+episode_index` for formal training and `629990000+episode_index` for smoke (zero-based); they are recorded even when the physical configuration is identical. A fall ends that episode early and consumes its actual transitions; it does not yield padding or a longer total training budget. Stage changes are not triggered by success rate, reward or holdout results. No unannounced terrain/randomization curriculum.

Record each PPO update, checkpoint hash/counters, compact per-control reward/action/progress/saturation/timing/contact summary, episode termination and exposure. Full state/action/reward snapshots at each episode boundary/checkpoint and each failure support audit; final fixed evaluations carry the full native/endpoint archive. Do not claim that compact training logs contain full native replay. Counter conservation uses actual control/native clock even on partial failed intervals; failure ends the batch rather than silently retrying consumed state.

Checkpoint reload verification is **nonintegrating**: compare sidecar/config/hash, parameter hash and deterministic action for saved95-dimensional observations against captured in-memory policy states. Do not call the old runner's reload helper, which adds one physical step per checkpoint. No hidden LQR constructor identification or extra reload/eval step is allowed outside the declared counters.

## Independent final evaluation, frozen before training

Five cases, each600 transitions, evaluated once for zero8 and once for the final131072 policy, maximum6000 controls/30000 native substeps. Request h=.020 net gap in jump cases; hold has no request. All evaluation seeds are distinct from training/reset seeds and fixed in `evaluation_protocol.json`:

| Case | Request tick | Friction scale | Purpose |
|---|---:|---:|---|
|no_jump_hold|none|1.00|Reject unsolicited hopping/drift|
|jump_late_a|175|1.00|Unseen request timing|
|jump_late_b|225|1.00|Second unseen timing|
|jump_lower_friction|175|.95|Declared small synthetic dynamics perturbation|
|jump_higher_friction|225|1.05|Opposite perturbation|

Friction scale multiplies the existing original friction through the existing randomization path; masses/solver/contact margins/actuators stay fixed. Metadata records the compiled actual friction. Checkpoint static semantics remain identical; episode settings can differ as the existing loader allows. These are bounded simulation conditions, not calibrated hardware uncertainty or a statistical robustness claim. With deterministic oracle physics, merely changing an unused RNG seed is not treated as independent physical evidence.

Evaluate all five zero baselines and all five learned cases; no early success-based stop or replacement run. On normal task failure retain first termination, no padding/reset. Root may stop the remaining matrix on integrity/numerical failures. No development evaluation before/following every checkpoint, no holdout-driven reward edit and no winner chosen by return. The fixed final model succeeds only if all four requested-jump cases pass all declared gates and the hold remains stable without unsolicited flight. Otherwise report each measured failure and stop this budget. One training seed does not establish cross-training-seed reliability.

Use the physical gates from the calibrated height probe with exact windows adapted to the fixed600-step episodes: no nonwheel contact/fall/domain exit; heading peak≤5 degrees, roll/pitch≤10 degrees, planar displacement≤.10 m; true native no-load flight for≥20 ms with positive COM velocity at onset; simultaneous net gap≥.020 m; landed before endpoint400; late endpoints400≤j<600 height RMSE≤.015 m, max abs body vx≤.03 m/s, max abs COM vz≤.03 m/s; late path positions S400..S600≤.05 m; per-wheel positive native normal load in≥95% of final1000 native evaluations. Hold meets global/late stability and has no certified flight after settling. Report all action boundary occupancy, IK/rate/torque protections, actual impact forces and false/extra flights. No protected-torque value may exceed original limits; protection activation alone is not task failure.

Raw request/reference and actual geometry determine evaluation, never shaped reward or scaffold phase labels. Archive T actions/commands/torques/PI/reward records, T+1 state/95obs snapshots,5T actual native wrench/contact/timing entries, final and partial receipts. Same-case zero/policy initial95obs/qpos/qvel must match bytes. Once policy actions differ, compare metrics on actual completed durations/common observable prefixes; do not require later learned/zero states to match or pad a failed episode. The95obs task-progress features are logged with their source; no retroactive relabeling as old85 observations.

## Budget and scope

| Work | Max new controls | Max native substeps |
|---|---:|---:|
|One smoke training run|640|3200|
|One formal fresh PPO run|131072|655360|
|Five zero + five final-policy evaluations|6000|30000|
|This RL contract total|**137712**|**688560**|

The previously planned1200/6000 measurement probe is separate and not rerun here. Combined with that prior probe, maximum138912/694560; do not double-count reused files. Nonintegrating tests/reload checks add0. No new experiment beyond these budgets begins automatically.

New task/observation/reward/plant identities and a strict new loader must reject old82/85 models **before** SB3 deserialization. Do not modify77 frozen inputs or old snapshots/checkpoints/manifests. Verify source/test/evaluation protocol hashes before/after; retain real Opus source/receipt and worker verification stdout, output-scope audit and frozen77 checks. An execution worker may write only the declared new files and run only the fixed nonintegrating verification command. Root starts the physical smoke/train/evaluation after reading the frozen contract and reviewing concrete artifacts.

After this experiment, use its learned success/failure evidence to choose the next RL task boundary. Successful stationary hops still require an explicit bounded handoff/repetition and genuine plane+box crossing task before GUI exposure; failed hops get one evidence-based task/action diagnosis, not a concealed classical gain sequence or an automatic second training budget. The original driving→obstacles→speed goal remains open.
