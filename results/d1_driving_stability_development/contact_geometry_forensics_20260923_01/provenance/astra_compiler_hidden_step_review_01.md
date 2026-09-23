# Compiler hidden-step review and blocked plan 01

Read-only review by actual GPT-6-astra ultra, 2026-09-23. No model creation, MuJoCo import, native call, build, test or integration was performed by this review. The prepared plan-01 harness was sealed before execution.

## Confirmed blocker

Fixed official MuJoCo 3.12.0 `sources/user_model.cc:5575-5614` computes `asleep_init` and, unless it is true, executes `mj_step(m, d)` while validating a newly compiled model. A world-only model with no trees cannot satisfy the sleeping-tree escape condition. Consequently `nq=nv=0` does not make XML compilation satisfy a literal zero-step contract. This call uses temporary compiler-owned data; a subsequently allocated data object's zero time would not establish that no hidden step occurred. `engine_forward.c` implements the step/forward/integrator path and contains no no-DOF exemption that removes the call itself.

`next_engine_kernel_plan_01/next_contract.md` remains frozen at SHA256 `0a442f05ba09693d9e97bdc171cee05fef9e1c835225a6bd414209c4c9e374b9` and is **blocked, never activated**. Do not interpret its permission for static compilation as overriding its explicit prohibition on hidden integration. Do not introduce sleep flags, fabricate a successful call ledger, or exempt zero-DOF steps. Sol removed `mj_loadXML` from the prepared C source and made `main` return BLOCKED before any engine call; no kernel budget was spent.

## Narrower source-backed alternative

The installed pinned shared library exports `mjc_ccd`, `mjc_ccdSize`, and `mjc_initCCDObj`. These are non-public internal interfaces, despite symbol export. A replacement must say **native GJK/EPA with a locally copied perturbation wrapper**, not direct `mjc_Convex`, and must use complete official typed headers rather than guessed ctypes layouts.

For `g>=0` and cylinder/box types only, `engine_collision_convex.c:726-783` reads exactly `mjModel.geom_type`, `geom_size`, `mjData.geom_xpos`, and `geom_xmat`; it copies dimensions/poses into CCD descriptors and sets native support function pointers. A zero-initialized typed carrier providing only those arrays is a restricted read adapter, not an allocated/compiled/valid simulation model. It may be passed only to that verified initializer, then discarded. Cylinder/box support uses descriptor fields only. `mjc_ccd` receives no model/data at all; `max_contacts=1` makes mesh multicontact branches unreachable. `mjc_ccdSize(0,0,35)` has an explicit source-defined size, including minimum box buffer dimensions; use aligned malloc memory and preserve warning output.

This removes the need for all model compilation/allocation/reset calls and data arenas. The complete wrapper, initialization read set, buffer bounds, full ABI headers and strict attempt ledger still require implementation review before any query. Differences caused by local wrapper arithmetic must remain visible; successful reproduction is evidence of the native narrowphase defect, not a repaired engine or qualified robot run.

## Fixed new official source evidence

The exclusive `sources/source_manifest_kernel_addendum.json` records URLs, sizes and SHA256 for the following official tag-3.12.0 files:

- `engine_collision_convex.h`: `3df5241f900dd9c01565c857191eeb740104b09f75a11af56500b73fbaefddfe`
- `user_model.cc`: `4f4a3e85668a19df2c77b7df449f6a11babdb885f5e6c56563b85263b1b015f3`
- `engine_forward.c`: `c81f57dfa0249b07bfd92ac14ac145e43581a1695be607c5ae80fab2417c6d52`
- `engine_setconst.c`: `72d8fab9a3e13ea103bbc5a80e126799bde1b94132b8820e1a388df93a4d533f`

Raw URLs are `https://raw.githubusercontent.com/google-deepmind/mujoco/3.12.0/` plus `src/engine/<filename>` or `src/user/user_model.cc`. No full-engine source compilation or execution occurred. Old readiness scores, original 2400/12000 exhaustion, and closed qualification/RL gates are unchanged.
