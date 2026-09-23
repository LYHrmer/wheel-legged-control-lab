# Contract 02 execution review

**GO for the first and only bounded native-CCD process.** Actual GPT-6-astra ultra reviewed the local wrapper/gates and final telemetry delta. The final C SHA256 is `27bca448180d267c2954f127279c70939f85420d5d1b5da4fe991d63da8cf888`. No remaining substantive execution blocker was found. Root may compile/link this exact source, freeze the binary and dependencies, then execute via its exclusive one-shot launcher. This is permission to execute the bounded diagnostic, not evidence that a query has already run or reproduced the fault.

The earlier message GO bound to a092c400... is superseded, not silently reused: root had not compiled or executed it. Sol's already-started final edits added safe raw-status telemetry, explicit case/slot names and reference differences before candidate-decision logging. They did not alter native entrypoints, geometry, perturbation order, query limits or fourth-case gates. Edits have stopped.

## Review evidence

- Typed carriers provide only geom_type/size and geom_xpos/xmat to the source-audited cylinder/box initializer. They are not engine simulation objects and are never passed to model APIs. Full official typed headers define the ABI. Support callbacks copy/use descriptors without retaining carrier pointers. Internal GJK/EPA has no hidden model/step/forward path on this fixed max_contacts1 primitive branch.
- Attempt records are fsync'd before each native CCD call. Cap16 is enforced. Init cap8, buffer-size call1, version functions each1, explicit counted pure-math cap512. The workspace has sufficient aligned allocation and is freed through its original malloc pointer. No model compile/load/reset, mj_step, forward, broadphase, or constraint solve appears in the harness.
- Pair margin .001 is passed intact to both descriptors, then native support adds half internally. Primary and axis0/axis1 with -0.001,+0.001 perturbations match source order. Only pos/mat is restored, preserving mutable primitive support cache. Rbound follows upstream ordered multiply/add/sqrt with no margin. Initial tangent-frame and paired inverse rotations have been checked against source.
- Strict negative returned distance gates witness conversion. The native status, simplex, witness, raw query distance, original normal/tangent, distinctness and overwritten primary distance remain separate. EPA status0 is not asserted to prove convergence.
- The original .0021 cone residual and distance-derived support tolerance remain. Diagnostic correspondence cannot grant geometry or readiness. Fourth case requires both healthy neighbors and all three target candidates corresponding to the old pairs, a valid primary, and exactly the old contact5 downward anomaly. It only skips the local extra queries; no model flags or dynamics are changed.
- Root's outer launcher was reviewed: exclusive fsync'd process budget before Popen; one attempt, 45-second timeout, no retry; all-source hashes before/after; all events retained. Counts must be reconciled from attempted/returned sequences, not only summary. Its frozen argv/output and event basename must agree with the C interface.

The independent source_dispatch reviewer confirmed the initializer read set, complete ABI, workspace, call footprint and durable attempt bounds. It also independently identified the two initial blockers: helper name finite conflicted with glibc, and nonfinite outputs could damage NDJSON. Both were repaired: finite_vec, JSON null plus raw hex, and finite checks for all valid simplex/witness/descriptor data followed by immediate stop after persisting a failing return. Its prior GO covered a092c400; the final telemetry-only delta has been sent for immediate independent hash rebinding. Root should attach that message before running; no further implementation work is requested.

Sol reports pure generator Ruff and C -fsyntax-only -Wall -Wextra -Werror passed. Neither reviewer compiled, tested or entered the engine. The generated inputs contain no XML and no quaternion conversion.

## Binding and interface

`astra_native_ccd_go_source_snapshot_02.json` records exact byte sizes and SHA256 of source, the old pure-generator import dependency, fixture, generated input header/JSON, staged official internal headers, installed public headers, libccd declaration dependencies, pinned shared library, contract and launcher. Root additionally freezes the binary and actual compile command. Linking must use the exact installed libmujoco.so.3.12.0 path and its fixed rpath; do not use fast-math or mjUSESINGLE. Suitable ordinary flags are -std=c11 -O2 -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off, include paths for the pinned MuJoCo headers and kernel_preparation_02, and -lm -ldl. Compilation does not run the entrypoint.

Executable arguments: `[BINARY, EXISTING_OUTPUT_DIRECTORY]`; event file: `ccd_events.ndjson`. The root launcher creates the output directory first and uses frozen argv beginning `rtk proxy`. No XML argument exists.

Generator SHA256 `2e00cdfab9a1ec97dba58074ae9258eca9806dd4afe264459dac5f434a79f5f8`; input header `d0a43bae3bd60242cd8bf11e01487aee7e84841bae5799316453a7f792011ba5`; input JSON `79190a820af11e36b71eb8a25c769a668b00def7d74d2f22c89b3270fc679dff`; contract `0869d5836add5dc12a295d5ac4618bff984244c9703516202f9485a29fdf4768`; library `bd3f702ace8a31e1046f746880387858a981d11b01772a55ebef48ffd55ea5b8`.

## Result limits

This is a non-public native GJK/EPA entry with a source-equivalent local perturbation wrapper, not direct mjc_Convex. Compilation arithmetic may differ, so bitwise reproduction must be observed. Success narrows the defect, not repairs the engine; failure does not negate the old full-scene anomaly. Separately inspect the fourth primary-only result before saying it preserves the healthy primary. No new force is calculated.

The zero-call archive phase is closed at two identical full passes and 12 pure cases in one local round. Contract01 remains blocked because the model compiler internally calls mj_step. Contract02 has its own native-query ledger and zero integration. Original scores, exhausted 2400/12000 batch, closed qualification/RL gates and the goal order stable straight → reliable crossing → speed remain unchanged.
