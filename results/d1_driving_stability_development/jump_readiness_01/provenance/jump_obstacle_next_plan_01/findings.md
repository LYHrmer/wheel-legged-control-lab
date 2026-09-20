# What the existing jump and obstacle records establish

Read-only planning by GPT-6-astra / ultra, 2026-09-20. This package contains no new model/controller construction, integration, contact solve, training, GUI code or control implementation. `read_legacy_jump.py` imports only the Python standard library, checks the original decompressed trace hash, and computes statistics from saved JSON.

## Exact origin of 60.86 mm versus 5.61 mm

The original evidence is `results/d1_driving_stability_development/jump_baseline/{jump_probe.py,protocol.json,summary.json,trace.json.gz}`. It uses `D1InteractiveSimulation(arena='flat')`, default LQR/VMC and legacy mixed measurements, with Space requested before execution tick200. It records600 main-episode transitions. The native builder's flat arena is a real MuJoCo plane, not the zero heightfield diagnosed in the separate heading experiments. Thus the jump deficit cannot be dismissed as that heightfield contact-normal problem.

The audit reproduces the archived summary exactly:

| Quantity | Original saved result | What it means |
|---|---:|---|
| Maximum base-origin rise |0.060861934055593425 m at endpoint243 /2.43 s|`qpos[2]` minus its2.00 s value; not whole-robot COM ballistic rise|
| Maximum simultaneous minimum wheel bottom |0.005609603011612507 m at endpoint244 /2.44 s|Minimum of all four actual oriented wheel-cylinder bottom heights at one endpoint|
| Four wheel bottoms at clearance peak |11.01765,11.00465,5.61343,5.60960 mm|Rear pair limits simultaneous clearance; base pitch about−0.040245 rad|
| Clearance above1 mm |7 saved endpoints,2.41..2.47 s|60 ms between endpoints;70 ms nominal sample bins, neither proves continuous no-load flight|
| Clearance above2 mm |6 saved endpoints,2.42..2.47 s|50 ms between endpoints|
| Clearance above5 mm |2 saved endpoints,2.44..2.45 s|Only10 ms between endpoints|
| Clearance above15 or20 mm |No saved endpoint|Does not reach even the first course stair/hurdle height simultaneously|

At2.00 s, before the request, base origin is0.4649456678 m and all wheel bottoms are already positive, approximately0.404..0.502 mm. The robot geoms use1 mm margin. Consequently `bottom > 0` is not a valid airborne detector: loaded resting contacts can have positive geometric separation. The old trace does not contain actual wheel contact loads/native samples needed to classify them. Its `upward_support_n` is the controller's allocated support **request**, not measured contact force. The peak request reaches973.860 N during thrust; zero allocated support at the height peak is not proof of zero actual support.

The geometry formula in `scripts/d1_side_step.py:111` is `extent_z = .087*sqrt(1-a_z²)+.020*abs(a_z)`, using each wheel's world axis. The URDF four foot collisions each have radius0.087 m, length0.040 m, coincident foot origin and joint cylinder axis; this supports the old clearance definition for its saved poses. New instrumentation should obtain dimensions and transform from compiled **collision geoms**, and validate the mapping instead of assuming visual mesh bounds or axis alignment. This planning audit trusts the original saved wheel-bottom values and verifies their statistical reduction; it did not perform fresh forward kinematics.

The logged16-torque peak is47.1148 N m on legs and4.23831 N m on wheels, with no logged sample at the80/12 N m rated bounds. This does not prove the unprotected law never saturated: the old trace lacks an unlimited request and per-native actuator receipts, and teleop also applies a slew bound. Final recorded roll/pitch is small, but the final-second base origin remains0.48005..0.48207 m, with vertical speed peak0.03515 m/s. Do not infer settled landing from the final attitude alone.

## Space, R and GUI identity

`run_d1_course_drive.py:95..133` routes a PRESS of Space/J to a guarded jump request; side-step activity can block it. `interactive.py:281..302` accepts a request with≥3 wheel contacts, absolute roll/pitch<0.18 rad, and no fall. It does not test obstacle height or forward approach speed. The old time-driven schedule is:

| Execution ticks after request200 | Phase name | Commanded height | Vertical feedforward |
|---|---|---:|---:|
|200..224|crouch|0.405 m|−80 N|
|225..239|thrust|0.500 m|450 N|
|240..274|flight|0.455 m|0 N|
|275..319|landing|0.435 m|60 N|
|320 onward|ready|0.455 m|0 N|

`completed_jumps` increments when this schedule ends. The phase called flight lasts through2.75 s even though the largest sampled clearance occurs at2.44 s and geometry is again negative at some later samples. It is a schedule label, not a measured contact mode.

The GUI uses `D1LQRVMCController`/`D1MPCVMCController` with `D1VMCController`, terrain-attitude feedback and an additional torque slew limiter. Its public baseline choices are lqr/mpc. The new heading path uses the separate `D1WheelLegController` joint PD + nominal support + wheel PI, plus heading feedback and qualified stop/turn extensions. That controller's support total is nominal `mass*9.81`; its height channel changes IK joint targets. It has no `vertical_feedforward_force_n` parameter and explicitly rejects nonzero vertical-velocity commands. Copying Space or the four heights is not a validated transfer of the old jump mechanism.

R in the course runner calls `simulation.reset(current_zone)` and resets plant/controller/estimator/command state, opening a new recording segment. It is a simulation reset, not torque-based self-righting. The earlier attempts to obtain a new RL heading GUI produced no GUI code; no such GUI is created by this package. The old two-action residual gate also rejects jump phases, so its presence in the legacy GUI does not establish learned jumping. New zero8 stop/turn evidence is not a learned-policy result.

## Hidden construction budget

Do not instantiate the old interactive backend merely to inspect its UI or reuse its schedule during zero-physics planning. `hierarchical.py:62` calls `identify_sagittal_model` unless an explicit model is supplied. `linear_model.py:50..111` has only a process-local LRU cache; on a miss it performs500 settling control steps plus11 finite-difference/affine steps on a private plant. At default dt this is511 additional control intervals /2555 native integrations, before the saved jump loop begins. The archived jump trace proves its600 main-episode transitions; its protocol is not evidence of a600-total-integration fresh process. A fresh execution of the provided old CLI would incur this constructor path. No old budget or record was edited to conceal the distinction. The new PD/PI readiness probe must avoid the old LQR/MPC construction path entirely.

## What old course obstacle 'passes' mean

`results/d1_interactive/course_metrics.md` reports LQR stairs progress4.07 m and a jump-zone 'pass' with progress2.98 m, one completed jump and two hurdles. MPC reports jump-zone progress2.36 m/one hurdle, but stairs fails its progress gate. These are real earlier course metrics under different control/measurement semantics, not qualification of the new heading controller.

Their jump acceptance in `interactive.py:632..642` counts a hurdle once the **base x** exceeds its center+0.30 m, combined with completion of the timed jump schedule and minimum progress. It does not require every wheel/body collision shape to cross the obstacle, measure simultaneous clearance, distinguish rolling/climbing from jumping, or prove a no-load flight followed by stable landing. Preserve these original results under their original metric names; do not translate 'hurdles' into certified ballistic hurdle clearing.

## Native plane and box boundary

The current `D1FlatPlanePlant` deliberately accepts only a world plane at z0, no hfield, and terrain IDs exactly `{floor}`. Its analytic query always returns zero height and zero slopes in a finite domain. It is unsuitable for pretending to test a step by changing metadata or raising a visual marker. Its global vertical-normal invariant is correct on a plane but wrong on a real box: a box's front face legitimately produces horizontal normals.

The unchanged course builder already creates real world-fixed boxes on a native plane. `terrain.py` defines first stairs as15 mm high,360 mm long and1.24 m wide; jump lane boxes are20/40/60 mm high,160 mm long and1.40 m wide. A direct `D1Plant(arena='course', sampling_mode='synchronized')` therefore uses original robot geometry and genuine obstacles without any new robot builder, but includes **all** course boxes. It must not be labeled an isolated one-box model.

`build_d1_model` has no custom-spec or obstacle callback parameter; it creates the URDF MjSpec, adds terrain, then compiles immediately. `D1Plant.__init__` immediately caches IDs and nominal arrays. A post-construction `plant.model = another_model` would invalidate those caches and is not an acceptable obstacle integration seam. An isolated single box requires its own reviewed construction/identity contract, or the existing course may be used with an explicitly bounded known corridor. Changing frozen files, silently monkeypatching the shared builder, or disabling unseen boxes after initialization is not proposed here.

For future box trials, classify contacts by the actual terrain geom ID and normal in that box's local frame, measure robot-vs-box distances/crossings using collision geometry, and keep plane contacts separate. The installed MuJoCo header declares `mj_geomDistance` for read-only signed geom-distance queries; any future use must validate supported pairs and transform/domain semantics locally before relying on it. At a vertical step face a scalar heightfield-like ground query is discontinuous and cannot represent contact normals; ground-reference behavior requires an explicit independent definition. Old display ray/height agreement does not establish loaded collision-normal behavior.

A21 mm stationary wheel-clearance peak would still not prove crossing a160 mm-wide hurdle at0.25 m/s. Horizontal travel during the actual airborne interval, front/rear wheel timing, contact on the obstacle, and all robot geometry must be checked. Stationary hop readiness, rolling onto a15 mm step, and jumping across a20 mm hurdle are separate claims.
