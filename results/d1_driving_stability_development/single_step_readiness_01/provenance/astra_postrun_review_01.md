# Astra post-run independent review 01

2026-09-22 · Reviewer: actual GPT-6-astra ultra · Scope: read-only source, sealed JSON/gzip evidence and pure NumPy. This reviewer performed no MuJoCo calls, environment construction, integration, tests or source edits. The only writes are this new report and the requested new next-terminal contract.

## Conclusion

Keep the original box verdict **invalid / not qualified**. The failing strict normal check is justified for the recorded candidate; this is not the earlier small multiccd tangent perturbation issue. Root's subsequent four-pose static reconstruction establishes that the anomalous contact is emitted by MuJoCo 3.12.0 collision generation at the saved pre-step pose. It is not introduced by this recorder's phase selection or terrain-normal sign conversion. The specific internal numerical branch is not established.

There is a separate diagnostic design weakness: the scorer converts a precise geometry failure into `record_invalid:unhandled_exception:ValueError` and suppresses all observations. Correct that information loss prospectively in a supplementary diagnostic layer; do not reinterpret the frozen result as a successful readiness trial. The next contract written alongside this review freezes zero additional control/native/forward and zero additional static MuJoCo calls. RL remains closed.

## Evidence and failure localization

The original contract SHA256 is `1ea04dc310486ed7025a73ed1e61a7977fdaafe0545f3de859905835d1b5c238`. Trial manifest SHA256 is `80b7a829b6be1332bf050d68b2774903457961a5400c5efa8b59da094e366f46`. I independently checked all 24 archive entries, including bytes and SHA256, and all current protocol source hashes: matched.

Both runs contain 6000 native rows. Pure independent raw checks covered all 45,430 plane contact rows and 46,387 box-trial contact rows, finding no inconsistencies in geom/body identity, frame finite/orthonormal/right-handed properties, active/local-force fields, geom-order normal/world-force conversion, wheel and box load aggregates, nonwheel/geometric aggregates, or qpos chaining. This confirms those audited raw relationships; it is not a declaration that geometry or every possible archive property is valid.

Across 11,053 wheel-box contact rows, exactly one is geometrically invalid. All 11,032 positive-load box rows are valid under the same strict feature rules; their maximum normal-cone residual is 0.0010027708200854321. The single invalid row is:

| Field | Archived value |
| --- | --- |
| Native / contact index | 3486 / 5, both zero-based |
| Start time | 6.971999999999454 s |
| Geom pair | 59 `RL_foot#geom1` → 1 `terrain_single_15mm_box` |
| Position | [-3.0296555548441106, 0.2213752018031465, 0.015656105537109645] m |
| Dist / inclusion margin | +0.000676533074470901 / 0.001 m |
| Terrain→robot normal | [0.004160111414019363, 0.0000019856892281108387, -0.9999913466971002] |
| Active / efc | true / 36 |
| Local force and torque | all six components exactly 0 |
| Supporting faces / residual | none / 1.0 |

The contact lies above the box top, well inside its XY footprint, yet points almost directly downward from box to wheel. The preserved tolerance is 0.001676633074470901 m. The bottom-face distance is 0.015656105537109645 m, so no negative-z support face is admissible. The residual is 1.0, far beyond 0.0021. Increasing the perturbation allowance would conceal a wrong sign, not resolve numerical roundoff.

The rejection is at `scripts/d1_single_step_integrity.py:82–93`, specifically line 91. The same position/normal support criterion is in `scripts/d1_single_step_geometry.py:255–275`. Recording at `scripts/d1_single_step_records.py:166–220` preserves the raw frame, corrects geom-order sign, and retains the zero-load candidate. It does not silently filter it.

## Static reconstruction and independent geometric check

Reviewed `scripts/audit_d1_single_step_contact.py:63–106` and `static_contact_forensic_01.json` (SHA256 `0c46a59f8905e1310b7a73d4c6c749440264161ddb27a4d067ac0ce005d30bc2`). Root's four static reconstructions use scratch data, saved pose/velocity/time, then only kinematics, comPos, collision and geomDistance. No force recomputation or control path is invoked. The receipt reports zero native/forward/setConst/control and unchanged integrator state.

All geom IDs, positions, distances and complete contact frames exactly match the saved rows for 3485 before, 3486 before and 3487 before. Using 3486 returned yields the next pose and does not match 3486's solved-cache contact geometry, as expected. Therefore this anomaly's phase provenance is resolved: the saved before pose reproduces it directly in collision output.

I separately computed a cylinder's minimum world-z support point using only the saved rotation/position and size [radius 0.087, half-length 0.02]. With axis a=R[:,2], the minimum point is c−h sign(a_z)a−r(e_z−a_z a)/sqrt(1−a_z²). For 3486 before:

- Minimum point = [-3.032189854174804, 0.23354291236819996, 0.01567653249319506] m, with XY inside the box footprint.
- Analytic gap above top = 0.0006765324931950617 m, so the entire uninflated wheel cylinder is above the box top.
- Saved geomDistance = 0.0006765329763443506 m, differing by approximately 4.83e-10 m; its saved witness segment runs from wheel z≈0.01567653 to box z≈0.015.

This independently supports upward box-to-wheel separation. The downward native candidate is an actual candidate-geometry anomaly. Its positive distance does not make it inactive: efc>=0 records an active constraint. Its solved six-vector is exactly zero, which establishes zero direct wrench for that candidate only. It does not prove that deleting/replacing it would leave other constraints, solver iterations or the trajectory unchanged. At that same sample the two neighboring RL contacts carry approximately 50.27 N and 51.85 N.

## Mechanism: supported entry point, limited causal claim

Official documentation describes repeated narrowphase evaluations with small tangential rotations for multiple contacts, including the positive-margin path. [MuJoCo multiple contacts](https://mujoco.readthedocs.io/en/latest/computation/#multiple-contacts).

In the fixed 3.12.0 source, `mjc_Convex` uses 1e-3 perturbations and can assign an added contact the primary contact's distance. `mjc_penetration` derives normals from witness differences. This explains why equal distances and a later contact index are compatible with an anomalous extra candidate. It does not identify which perturbation or numerical branch failed here; the archive has no internal witness/iteration provenance. Treat a multiccd narrowphase defect as the leading mechanism hypothesis, with native collision-output anomaly already confirmed. [Pinned 3.12.0 collision source](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_collision_convex.c#L81-L124), [multicontact path](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_collision_convex.c#L823-L895).

## Saved task observations and permissible conclusions

Supplementary `observations_01/observations.json` preserves `qualification_granted=false` and `original_score_overridden=false`. The all-native/endpoint roll, pitch and heading envelopes are approximately 0.002689°, 3.891925° and 0.005393°; maximum lateral displacement is 0.00004785 m. Last 100 synchronized endpoints give max |body vx|=0.004056213 m/s, max |whole COM vz|=0.000425196 m/s and world-z RMSE=0.006607143 m. Last 500 native samples have positive wheel-load occupancy [1,1,1,1]. Recorded nonwheel terrain contacts are zero. Final minimum clearance across all physical collision geometry, beyond far edge and each geom margin, is +0.308175620 m.

All four wheels have positive box-load records; first positive contact is native 1779. Positive box feature counts are front-top edge 1016, top 9557, back-top edge 459. These observations support a mechanically promising saved trajectory, but they do not override the failed all-candidate geometry qualification. Do not label the box trial passed or use it to unlock RL.

I reviewed `scripts/summarize_d1_single_step_observations.py:15–52` and plot captions at lines 74–89. The exact late windows are correct; its extrema include native before/returned states, explaining the slight difference from endpoint-only maxima. Its explicit rejection caption and supplemental flags are appropriate. Its hardcoded far-edge calculation and anomaly index are acceptable for this named frozen case; the script is not yet a general validator. Likewise, the static audit's exact-match field refers to the listed geometry fields, not solver-force equivalence. No additional substantive overclaim was found.

## Next bounded work

The exclusive new contract is `next_terminal_contact_plan_01/next_contract.md`. It requires a real failure fixture, an append-only pure diagnostic layer that reports raw fidelity separately from strict candidate geometry, a cylinder support proof, and a pinned-source mechanism audit. It reuses the four static reconstructions and permits no further MuJoCo calls, no physics, no engine/parameter changes and no training. It retains 0.0021 and every original physical gate. Successful completion means better diagnosis with the original box still unqualified; a real engine correction or renewed readiness attempt requires a separately frozen future contract.

The diagnosis skill's repair-completion phase is intentionally not claimed: the native anomaly remains reproducible and no engine fix was attempted. The available frozen fixture is a deterministic failure signal for future pure diagnostic regressions. Remaining uncertainty is the internal numerical cause, not whether the recorded candidate is geometrically inconsistent.

Contract SHA256: `157f0a33b386cea0ef99556a0d693370126ae57479db0460e4b211dd7b27e7d9`.

## Artifact anchors

- `original_contract`: `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/next_terminal_handoff_plan_01/next_contract.md`
  SHA256 `1ea04dc310486ed7025a73ed1e61a7977fdaafe0545f3de859905835d1b5c238`
- `trial_manifest`: `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260922/single_step_readiness_01/manifest.json`
  SHA256 `80b7a829b6be1332bf050d68b2774903457961a5400c5efa8b59da094e366f46`
- `static_forensic`: `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260922/static_contact_forensic_01.json`
  SHA256 `0c46a59f8905e1310b7a73d4c6c749440264161ddb27a4d067ac0ce005d30bc2`
- `observations`: `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260922/observations_01/observations.json`
  SHA256 `b50121afd6dd1a519a02ab1a4b8adc4d495772da2cca172e671fe2af8e3232ea`
- `contact_inventory`: `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260922/contact_failure_inventory_01.json`
  SHA256 `48e1f39fbeb52727f647933cd6d6172049fe86124f4f47479c603f92b25c728b`
- `frozen_scorer`: `/home/lyh/wheel-legged-control-lab/scripts/d1_single_step_scoring.py`
  SHA256 `d9c13912cafaa92a52d53671b8016a85328116bc2cc4e6f267eb5007f33fefed`
- `frozen_integrity`: `/home/lyh/wheel-legged-control-lab/scripts/d1_single_step_integrity.py`
  SHA256 `fa1d1ca12de8bed2051b374720ab8df774a913308ef6d413a46c49411c5bda8e`
- `frozen_geometry`: `/home/lyh/wheel-legged-control-lab/scripts/d1_single_step_geometry.py`
  SHA256 `7c6a1c7450286286f3e732e214b6d0a08247eab2f073dcfae671453529de18ed`
- `static_audit_script`: `/home/lyh/wheel-legged-control-lab/scripts/audit_d1_single_step_contact.py`
  SHA256 `5b39d2f0deda605bbfeaddc5425f169be130b5bebfd94751ae8e12ce76ecbdc7`
- `observation_script`: `/home/lyh/wheel-legged-control-lab/scripts/summarize_d1_single_step_observations.py`
  SHA256 `98ab13e3fa4863882b36a2543474292f427fd5e53d2ad7b5cdb7e5d2e5cfe52e`
