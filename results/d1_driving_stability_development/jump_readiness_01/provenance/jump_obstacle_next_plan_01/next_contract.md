# One next experiment: PD/PI height-profile hop readiness

Status: proposed fixed development diagnostic, executable only after root finishes and accepts the complete stop/turn composition matrix and its independent audit. This planning task ran zero new physics and does not start the diagnostic. If composition fails, retain this package without bypassing that gate.

## Hypothesis and one bounded change

Test whether the newly qualified zero-residual PD/PI controller's existing IK/leg-PD height channel can produce a physically measured stationary hop and recover, using one fixed existing height profile. The only experimental difference is raw commanded clearance. This is not a proposed jump fix and not a port of LQR/VMC thrust: no vertical-force feedforward, extra torque, controller/gain/cap change, residual action, policy, contact-triggered phase or timing search is allowed.

The original legacy trace is insufficient to answer this question for the new controller. Conversely, designing an additional thrust or retraction mechanism before collecting this simple channel response would conflate different controllers. A failed readiness trial is an explicit result; it is not permission to increase a force or rerun another profile.

Use the accepted combined controller and unchanged native plane, oracle synchronized publication, seed55101, dt0.01, five native0.002 s substeps. Raw forward and yaw remain exactly0 for all ticks, so both the stop latch and authority gate must remain inactive. This experiment qualifies the height channel from stationary reset, not hopping after drive/stop or turning. It leaves all original stop/turn thresholds and their records unchanged.

## Single paired batch:1200 control intervals /6000 native substeps

Run exactly two600-interval episodes, `stationary_height_hold` then `stationary_height_profile`. Do not rerun the old LQR jump or a standalone old baseline. No additional seeds, repeats or trials. The600-step hold is a new same-controller, same-plant matched control, not relabeled old post-turn data.

| Execution ticks | Hold clearance | Profile clearance | Profile label |
|---|---:|---:|---|
|0..199|.455|.455|settle|
|200..224|.455|.405|crouch_request|
|225..239|.455|.500|extension_request|
|240..274|.455|.455|return_request|
|275..319|.455|.435|lower_request|
|320..599|.455|.455|hold|

Heights and durations are the unchanged legacy schedule values, used only as a deterministic raw height stimulus. Avoid labeling `return_request` as measured flight or `lower_request` as measured landing. The clearance values lie within the existing D1MotionCommand .38..53 m envelope; commanded zero attitude gives nominal extensions−.050/+.045/0/−.020 m, all inside the original IK table−.08..+.08 m. This source-level range check does not predict actuator rate limiting or achieved motion. No vertical velocity is sent. Every residual action is zero8. No use of `D1InteractiveSimulation`, LQR/MPC, online model identification, old feedforward−80/450/0/60 N, `step()` padding or state teleportation.

No automatic reset/retry if a phase gate cannot be met. Execute each fixed schedule once unless existing termination or an integrity failure ends it. Task readiness failure can be recorded after the fixed observation interval; nonfinite state/invalid record/source mismatch ends the batch. Preserve every actual native call, completed control interval and partial interval. Maximum1200/6000 is a hard cap including all newly propagated plants: no constructor integration is allowed. Model compile/reset/geometry calculations do not count as integration and must be separately labeled.

## Bounded Opus module for the next root request

Request only a new `scripts/d1_jump_readiness.py`; no controller implementation or plant rebuild. It should contain two small independent pure facilities:

1. A stateless fixed schedule function accepting a finite integer prepared tick and explicit `hold`/`profile` condition, returning raw `D1MotionCommand(0.,0.,clearance)` and a phase label. Only the table above is permitted. Reject invalid condition/tick/bool inputs; handle prepared tick600 with the final held command without executing it. No internal advancing clock, latch, controller access or event queue. Root calls this once through the existing raw callback and logs the executed label before parent step prepares the next tick.
2. A read-only wheel-cylinder clearance/airborne summary facility operating on supplied geometry and existing native contact records. For plane normal n, wheel collision center c, unit cylinder axis a, radius R and half-length L, use `gap=n·c-plane_offset-[R*sqrt(max(0,1-(n·a)^2))+L*abs(n·a)]`. Read/validate four collision-cylinder IDs, sizes, local transforms and original joint/body association once from the compiled model; do not infer clearance from base height or visual mesh. Reject unexpected cylinder mapping rather than repair it. A sample function can consume already synchronized geometry arrays; it must never call mj_step, reset, mj_forward, collision solving, controller compute or mutate the plant/cache. The pure reducer reports sampled extrema/runs with explicit timestamps and does not claim continuous-time flight.

Root owns the thin environment, raw callback, source identity, T/T+1 archive, native observer, scoring and experiment execution. No changes to the accepted combined core, imported frozen sources, GUI or default entry point. Use a distinct jump-readiness task/probe schema even though action8/observation85 remain; all policy/checkpoint loads remain rejected before deserialization. Include the exact combined implementation and accepted audit identities in the new preflight, rather than a moving file path.

## Pairing, sampling and mandatory evidence

The paired hold/profile must be identical through execution ticks0..199, physical states S0..200, observations0..199 and first1000 native entries, including signed-zero bytes, raw/servo commands, PI, torque, action, wrench and native contact cache. Obs200 is excluded because preview sees the changed height target; S200 must still match. No baseline match is demanded after the first changed command. Both mechanisms remain inactive for all600 executed intervals and their diagnostics must confirm that. One parent PI update per interval and601 callbacks per completed episode remain required.

Archive600 execution records,601 state/obs snapshots, all3000 actual native ctrl/wrench/contact entries per episode and native returned qpos/qvel/time snapshots for the short takeoff/landing analysis. Root may collect native snapshots by extending its diagnostic observer only, preserving integrator data and original step count. Geometry reconstruction on those saved poses may run afterward using kinematics on scratch data; it must not recompute historical contact loads. Store initial state and final/partial receipts exactly. Each600-step episode includes its single reset and no later reset.

Separate three kinds of measurements:

- The control's planned height/phase, IK joint target, rate limiting, support request, PD request, total unlimited request and final protected torque. State actual joint velocities/limits and PI are also recorded. A command phase never certifies contact mode.
- Synchronized endpoint collision geometry: all four bottom gaps, base-origin height, full robot mass-weighted COM height and vertical velocity, attitude, horizontal displacement and joint state. COM uses original model masses, not just base body or qpos[2]. Native2 ms geometry snapshots retain their own times. Validate geometric queries on synthetic cylinder orientations including horizontal and vertical axes without integration.
- Native actual solved contacts and forces, retaining `native_step_solved_cache_not_synchronized_endpoint` sampling identity. Record geometric/active contact count and normal force separately, per wheel. A solver contact with zero force is not the same as no geometric contact. The old sampler's allocated-support request must never be substituted for actual normal load. Do not multiply endpoint velocities by another phase's native forces to claim energy. Contact impulse and COM momentum change may be reported with sampling residuals, not forced into an exact balance certificate.

## Frozen screening gates, not reliability certification

These are prospective hop-readiness gates for a new command domain; they do not redefine any prior G1 pass/fail. Report the following separately rather than collapse a large base rise into success:

1. **Execution/plant valid:** exact schedule and counts, finite state, original torque/target/joint protection intact, no unexpected wrench/nonwheel contact/fall/domain exit, true native plane normals. Both episodes complete6 s. Full-episode absolute actual roll/pitch≤10 degrees, original-reference heading error≤5 degrees and planar-origin displacement≤0.10 m. Protection occupancy is reported and is not alone a failure; exceeding actual rated applied torque or position/velocity envelope invalidates the readiness claim.
2. **Physical lift observed, profile only:** after request200, require at least10 consecutive actual native intervals (20 ms of recorded native intervals) with zero active wheel contacts and zero actual wheel normal load, and corresponding endpoint minimum wheel-plane gap above the compiled contact margin. The interval labels certify the saved native evaluations/geometry, not an unsampled mathematical continuous-time guarantee. At the measured transition into this interval, whole-robot COM vertical velocity must be positive. Report peak COM rise separately. This rejects merely moving the legs while falling from the reset, resting within margin, and a one-sample contact glitch.
3. **Useful geometric clearance, profile only:** maximum simultaneous minimum wheel gap must exceed the first existing course hurdle height0.020 m plus the compiled wheel/plane contact margin (normally0.001 m, so0.021 m). Report all four individual maxima but never replace the simultaneous minimum with their maxima at different times. The chosen20 mm comes from the first actual course hurdle, not outcome fitting. This is a stationary clearance screen; it does not establish traversal of that hurdle.
4. **Landing/settled return:** after a measured airborne interval, contacts must return before endpoint400. In endpoints400≤j<600, require raw held-height RMSE≤0.015 m, max absolute body-forward speed≤0.03 m/s, max absolute COM vertical speed≤0.03 m/s; sum planar path over S400..S600≤0.05 m. Require each wheel to have positive actual normal load in at least95% of the final1000 native intervals. Report nonwheel contact, maximum impact normal force, pitch/roll, joint/torque protection and any rebound. The20 ms no-load,0.03 m/s vertical-speed and95% load-occupancy values are explicit conservative development screens, not claimed hardware guarantees or thresholds extracted from successful new results. The hold episode must meet the same global/late settled-state checks but is not required to become airborne.

If pure geometry or original physical plant invariants invalidate any proposed threshold interpretation, root resolves that **before** physics and records the fixed contract revision; do not repair the interpretation after observing outcomes. Do not compare whole-episode height RMSE to a constant0.455 during the deliberately changing command and claim a jump failure/success from that mislabeled score.

## Single decision after this diagnostic

If the new PD/PI chain does not achieve true flight/useful clearance/settled landing, freeze the result. Use its native impulse, joint motion, target-rate limitation and phase-specific torque evidence to select exactly one later thrust/retraction/landing mechanism with actuator/geometry derivation. This contract does not prescribe such an unmeasured mechanism, authorize a gain sweep or append more physics.

If all screens pass, the conclusion is only one deterministic stationary height-profile hop with measured flight and landing. Reliability still needs a separate small repeated/handoff contract; there is no immediate GUI Space deployment. The next obstacle work must retain real plane+box geometry. Use the original course's first15 mm stair or20 mm hurdle with its exact collision geometry and an explicitly versioned synchronized ground-query/contact-identity adapter; existing full course geometry remains declared. It cannot be represented as flat-plane metadata or borrowed old hfield normals.

Before any actual crossing, require offline geom-ID/material/robot invariant checks and saved/synthetic pose distance tests at box top/front/edge. Real horizontal front-face normals are expected. A future crossing scorer must require the relevant complete robot collision support bounds to pass the far face, actual contact-mode history and stable landing/stop; base-x progress or timed jump completion is insufficient. Height/approach speed/timing are fixed only in that later contract. No obstacle integrations, higher stairs, ramps, rubble, speed increases, RL or GUI backend migration are included in this1200-step diagnostic proposal.
