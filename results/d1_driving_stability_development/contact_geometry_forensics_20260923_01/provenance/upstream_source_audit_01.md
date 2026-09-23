# MuJoCo 3.12.0 upstream source audit 01

2026-09-23 · Actual GPT-6-astra ultra. Scope: frozen contract C. This audit read the complete 2026-09-22 next-terminal contract, selected immutable metadata/fixture/source files, and official fixed-version source text. It performed **zero MuJoCo imports/calls, model/data constructions, tests, full-trial scans or old-file modifications**. A bounded independent sub-review checked dispatch/sign semantics; conclusions below were checked against the downloaded raw source's actual line numbers.

## Result and evidence boundary

The default 3.12.0 source path can be narrowed to **cylinder–box generic convex collision → repeated ±0.001 rotations → native GJK/EPA witness recovery → position-only admission of extra candidates**. Driver/frame construction does not selectively reverse one same-pair candidate. Equal `dist` on the bad and good contacts is not independent evidence that their recovered normals are consistent: the extra candidate's distance is overwritten with the initial distance.

The old static evidence already establishes a genuine native collision-output anomaly, reproduced at saved native 3486 before with no dynamics. This source audit does **not** establish which perturbation, EPA face, affine coefficient, termination condition or arithmetic operation caused it. Internal numerical root cause remains unlocalized. No engine fix, geometry qualification or readiness pass is claimed. The user priority remains stable straight motion → reliable obstacle crossing → higher speed; this unresolved geometry qualification belongs before speed increases or RL.

## Provenance and applicable flags

Official tag URLs use `google-deepmind/mujoco/3.12.0`; byte-exact fetched files and SHA256/URL/version manifests are under `sources/`. GitHub's tag-reference API returned HTTP 403 rate limiting, so no tag-to-commit mapping is asserted. The version-tag URL plus preserved bytes and SHA256 identify the audited content. A reported runtime version of 3.12.0 does not, alone, prove binary/source-build equivalence; future instrumented reproduction must establish that equivalence by baseline agreement before causal claims.

Frozen robot geom 59 is cylinder, radius 0.087 m, half-length 0.02 m, margin 0.001 m; box geom 1 has half-size [0.18,0.62,0.0075] m, margin 0. The archived bad contact's inclusion margin is 0.001 m. The hashed builder `scripts/d1_single_step_plant.py:35–74` sets the robot margin and does not override ccd flags/tolerance/iterations; the URDF only specifies compiler discardvisual=false. The old qualification compared options including disableflags/enableflags/ccd fields but did not save their numeric values in the geometry manifest, so the default flags below are a source-derived inference, not a fresh runtime read.

`engine_init.c:51–100` initializes ccd_tolerance=1e-6, ccd_iterations=35 and both flag sets=0. `mjtype.h:53–75` places nativeccd and multiccd among disable bits, hence enabled for the zero default. [Fixed defaults](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_init.c#L51-L100), [flag definitions](https://github.com/google-deepmind/mujoco/blob/3.12.0/include/mujoco/mjtype.h#L53-L75).

## Dispatch and sign chain: verified source behavior

`engine_collision_driver.c:530–542` orders geometry by type; cylinder(5) precedes box(6). The cylinder/box dispatch entry at line 52 is `mjc_Convex`. Generic pair margins are summed, rather than taking max; explicit pair/override handling is separate (lines 160–175). The generator receives margin+gap (1919–1941). Later, the precontact normal is copied directly into the same ordered pair's contact frame (2061–2077). This sum/max distinction does not change this case: box margin=0 and wheel margin=0.001. It also does not justify changing the frozen analyzer tolerance. [Driver](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_collision_driver.c#L530-L542), [dispatch](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_collision_driver.c#L43-L55), [margin](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_collision_driver.c#L160-L175), [copy-out](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_collision_driver.c#L2061-L2077).

The public contact definition states geom[0]→geom[1] (`mjdata.h:37–44`). `mju_makeFrame` normalizes the existing nonzero first axis, builds perpendicular tangents and their cross product; it does not choose orientation from either shape's center/surface (`engine_util_spatial.c:512–538`). Thus a single anomalous same-pair sign is not explained by an ordinary downstream pair-order conversion. [Contact definition](https://github.com/google-deepmind/mujoco/blob/3.12.0/include/mujoco/mjdata.h#L37-L44), [frame completion](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_util_spatial.c#L512-L538).

## Convex path: verified source behavior

At `engine_collision_convex.c:855–876`, positive margins force max_contacts=1; cylinder–box also falls through to one without margin. Lines 881–961 first generate a contact, then, if multiccd is enabled, try both signs around two tangent axes. Each trial rotates the two shapes oppositely about the initial point and restores their poses afterward. Accepted extras are judged by position separation (822–829); no extra normal-direction consistency predicate appears there. Their distances are replaced by the primary value (946–950). `mjc_fixNormal` is called only for the disabled-nativeccd/libccd path.

In the native path (87–132), `mjc_penetration` accepts negative `mjc_ccd` return values and forms midpoint position and normalized x1−x2 normal from the witness pair. A finite, orthonormal downstream frame therefore does not establish the physical correctness of that witness direction. [Penetration wrapper](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_collision_convex.c#L87-L132), [multicontact gates and loop](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_collision_convex.c#L855-L961), [distinct-position predicate](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_collision_convex.c#L822-L829).

## GJK/EPA seam: verified source, conditional deduction

`engine_collision_gjk.c:340–367` expands both support objects by half their object margin. The cylinder–box path skips sphere/capsule special treatment. `mjc_ccd` runs GJK and conditionally builds an EPA polytope (2323–2446). With max_contacts=1, polygon manifold clipping is not used. EPA selects/expands a face, with convergence and numerical exit conditions (1363–1509). A retained face yields affine witness combinations (1342–1357); the signed depth is negative face-distance magnitude. `triAffineCoord` computes the coefficients (1027–1068).

Conditional on this native fixed-source path actually executing, a cylinder–box contact accepted through the wrapper's negative-distance gate requires negative penetration recovery here; ordinary positive GJK separation alone cannot emit that contact. This narrows the relevant instrumented seam to EPA face/witness output and its consumption, but does not identify a faulty face or coefficient. The fields available for a future trace include simplex, separated, nx, distances, witnesses, GJK/EPA iteration counts and EPA status (`engine_collision_gjk.h:82–103`). [GJK/EPA entry](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_collision_gjk.c#L2323-L2446), [witness recovery](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_collision_gjk.c#L1342-L1509), [status fields](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_collision_gjk.h#L82-L103).

## What this evidence supports

- **Observed previously, not rerun:** the saved pre-step pose exactly reproduces the anomaly; analytical uninflated cylinder support and saved distance independently place the wheel above the top. The candidate's actual solved six-vector is zero. All original gates stay unchanged.
- **Verified in source:** the pair/sign chain, perturbation scheme, distance replacement and lack of an additional normal-direction admission predicate above.
- **Strong inference:** this is an extra perturbed candidate from the generic convex path. Same-pair ordering and equal copied distance fit that path, but the archive lacks axis_id/angle_id and raw internal witnesses.
- **Not established:** a specific arithmetic bug, an exact EPA termination branch, source/binary bit equivalence, or that deleting an unloaded constraint would leave the coupled solve unchanged. An `epa_status` success code alone would not prove full EPA convergence, because status assignment occurs before the EPA expansion loop.

Three discriminating hypotheses remain for a future instrumented study: (1) returned EPA witness direction becomes inconsistent with its selected face or the known perturbed support geometry; (2) a prior support/pose or initial-polytope calculation supplies an already inconsistent candidate; (3) the instrumented source/build does not reproduce the packaged binary's case, preventing source attribution. Observe the first differing invariant before proposing a patch. Do not substitute normal flipping, abs(normal), zero-load filtering, a larger cone threshold, different margin or different collision flags.

## Proposed next minimal engine study — not authorized/executed by this audit

A new separately frozen contract could allow an isolated native narrowphase harness, with **zero control/native integration/forward/inverse/constraint solving/training**. Keep installed MuJoCo and all lab sources/evidence unchanged; build only a separately located fixed-version instrumented engine if specifically authorized. Pin build inputs, scalar precision/compiler flags and binary hashes. Copy the original pair's exact saved world centers/rotation matrices, shape sizes, pair margin/gap, options and geom order; avoid quaternion reconstruction that changes input rounding.

Bound the first study to **three baseline pair-generator calls**, one each at native 3485,3486,3487 before, with no repeat-until-reproduces loop. Each call internally has at most initial+four perturbation evaluations; record these separately. Log at most those 15 evaluations: axis/angle, rotated poses, unmodified support parameters, return depth, witness pair, selected EPA face/projection, affine coefficients, termination reason and residuals. Compare the emitted pair geometry with the already saved baseline. If exact baseline agreement fails, stop at a reproduction-gap report; do not infer that the anomaly is fixed.

Only after a concrete violated invariant and causal branch are established should the new contract permit **one source-local numerical correction and at most three corresponding patched calls** (six pair-generator calls total, at most 30 internal penetration evaluations). If no defensible correction emerges, finish with a minimal reproducer and unresolved root cause; no flag/threshold/margin bypass. The corrected output must satisfy independent support geometry and retain legitimate comparison candidates; count disappearance alone is not success. Preserve before/after raw outputs and the reason for any rejected candidate.

These static checks can qualify an engine hypothesis/patch for further review only. They cannot qualify the original trial, certify the full trajectory under changed collision output, enable RL or authorize higher speed. Any subsequent prospective readiness run needs its own physics budget and unchanged task gates.

## Saved source inventory

Line references above use the byte-exact downloaded raw files. Browser-extracted line numbering drops some blank lines; do not reuse its numerical labels as GitHub source line anchors.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| engine_collision_convex.c | 51740 | `c616a3a43195cc2bb7ad52546c0a3443a1b514cf9dbcd79363247b1803a993d8` |
| engine_collision_gjk.c | 79605 | `0f3ff5b77893132dcc8412d4e96d80f9d625e99cb3009bb7a4505eba16320855` |
| engine_collision_gjk.h | 4124 | `756622585c9b5d1c5f824eb8e7d69c8c8393addd74733cfddbb01321ff3a219e` |
| engine_collision_driver.c | 88036 | `6981c6f7b82363fb4702be7c873a73980cbeb59c827cf68ceac485368cc4fd12` |
| engine_io.c | 73829 | `61f2993a625070a1e10b25cc6b9bb2d18310bbefc4d11c38aebe38edf3dcc00d` |
| mjtype.h | 29096 | `3fbda7518b6f97568c95d82cdae71289650c8eba2724ae18ab37284d5db3e804` |
| mjdata.h | 25555 | `93aaee9927e47fce4ac2f0f07dc90aad6769b8a9a401e262bd87135c29c90ec5` |
| engine_util_spatial.c | 16775 | `2752e10f311306df219dd3c78ea3fc2d34def413708cd98076bb46ff392ea4d7` |
| engine_init.c | 8035 | `a328cd8f96e11922050d8db8d233cebaac40ae670aef79cc9ec67c6a443d8139` |

`source_manifest.json` and `source_manifest_addendum.json` record each official URL, fixed tag, local path, SHA256, bytes and line count. Source copyright/license headers were preserved. No tag commit hash or installed-binary hash was invented.

Contract anchor: `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260922/next_terminal_contact_plan_01/next_contract.md`, SHA256 `157f0a33b386cea0ef99556a0d693370126ae57479db0460e4b211dd7b27e7d9`. Historical static evidence anchor: SHA256 `0c46a59f8905e1310b7a73d4c6c749440264161ddb27a4d067ac0ce005d30bc2`. All new execution counters for this audit remain zero, including complete two-trial offline scans.
