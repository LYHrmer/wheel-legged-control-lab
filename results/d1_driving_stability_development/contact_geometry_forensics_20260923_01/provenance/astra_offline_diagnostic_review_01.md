# Astra offline diagnostic review 01

2026-09-23 · Actual GPT-6-astra ultra · Read-only review of frozen new implementation, real fixture and first-pass outputs; no diagnostic rerun, tests or MuJoCo execution by this reviewer.

## Decision

No remaining substantive blocker was found for root's one permitted second complete offline verification. The first-pass outputs preserve the original failed box qualification while correctly distinguishing faithful raw recording from an invalid native geometric candidate. Keep the four implementation/fixture files frozen. No second pure-test round is necessary unless code changes.

The two input-binding defects found during static review were repaired before execution: CLI now verifies the frozen old and current contracts, trial manifest and real-fixture SHA anchors before charging a pass, and verifies the fixture's four exact source mappings, hashes/bytes and embedded static/geometry documents. The existing 0..2 integer pass journal is validated; no alternate log/path should reset that budget. Root coordinates all execution.

## Implementation and preserved semantics

Reviewed `scripts/d1_single_step_contact_diagnostics.py` and `scripts/diagnose_d1_single_step_archive.py`, plus the 12-case test file and fixture. All four current SHA256/byte counts match `diagnostic_source_freeze_01.json`.

The contact core independently checks raw IDs/body names, orthonormal frame, active/force fields, geom-order normal and world-force links, then applies the original position/normal cone formula. The bad contact is retained with its original frame, normal, distance, inclusion margin and six zero load components. Because its stored feature accurately says invalid, its geometric failure creates a geometry issue without falsely declaring a raw-record mismatch. No load epsilon or zero-load exemption was introduced.

The archive layer continues through every contact after that geometric failure; it checks native/control/endpoint counts and timing, finite states, qpos/qvel chains, held torque, foreign force arrays, identities and summaries. All geometric/loaded flags fail closed on raw errors. The old scorer is imported only for pure parsing helpers; the combined old scorer/validator is not executed or patched. The frozen task score is copied unchanged, and qualification/RL flags stay false.

## First-pass result verified

| Output | plane_only | single_15mm_box |
| --- | ---: | ---: |
| Native / endpoints / controls | 6000 / 1201 / 1200 | 6000 / 1201 / 1200 |
| Contact rows | 45430 | 46387 |
| Raw record links valid | true | true |
| Raw issues | 0 | 0 |
| All-candidate geometry valid | true | false |
| Box contact rows / positive box rows | 0 / 0 | 11053 / 11032 |
| Geometry issues | 0 | 1 |

The sole box issue is native 3486/contact 5, `box_normal_outside_support_cone`, no candidate faces, residual 1.0, unchanged limit 0.0021 and position tolerance 0.001676633074470901 m. Its raw contact dictionary is exactly equal to the authenticated fixture's original bad contact. All 11032 positive-load box rows remain geometrically valid as a separate observation; this does not replace the all-candidate gate.

The plane's `positive_load_candidate_geometry_valid=false` reflects that it has zero box contacts; the counts and preserved original score make that scope clear. It does not reverse the plane's original success. The box's copied original score remains record_valid=false and task_passed=false.

The saved pure support proof has minimum point z=0.015676532493195075 m and positive gap=0.0006765324931950756 m, with XY inside the original box. The analytic-minus-saved engine distance is −4.831492750366831e−10 m. The final few binary64 digits differ from the previous standalone arithmetic arrangement by about 1.4e−17 m; no physical threshold or outcome depends on this harmless expression-order difference.

The first test XML records 12 tests and zero failures/errors. Test receipt and full-pass receipt report no forbidden import attempts, no loaded MuJoCo module and zero new control/native/static MuJoCo calls. Current journals record one of two allowed test rounds and one of two allowed complete offline passes.

## Exact second-pass acceptance and transition

Root should execute the same frozen source once more using the SAME full-pass budget file and a fresh exclusive output directory. `plane_only.json`, `single_15mm_box.json` and `support_proof.json` must be byte-identical to pass 1. The receipt's full_pass_number must change from 1 to 2; all other substantive receipt fields should match. Compare receipt structure with that one intentional counter difference rather than demanding impossible identical receipt bytes. Recheck four source hashes before/after. No third full pass is authorized.

If these conditions hold, the current zero-call contract is complete. Root can close it and activate `next_engine_kernel_plan_01/next_contract.md` under the already-authorized routine validation scope. Its separate four-call static kernel budget does not resurrect the exhausted old 2400/12000 dynamic budget. Sol may prepare the new C harness and exact inputs in parallel, but no kernel/model call occurs before the phase transition and static implementation review.

If results differ unexpectedly, preserve both outputs and diagnose the finite difference offline; do not spend an unrecorded third pass or amend old evidence. The next engine phase remains unactivated until the difference is understood.

## Reviewed first-pass file hashes

- `plane_only.json`: SHA256 `0252b2c5cb97b11dfed138091a5d2b9b67e3d35eaab8e855de2559131c2e3a87`
- `single_15mm_box.json`: SHA256 `aefa243b69497f3749f033fbbff44d9f4e9704b2237d475523083d758cd5c033`
- `support_proof.json`: SHA256 `6673f14915b280d5f2ba06478d1d7a1d451a4080088b99ea31e50efc23e61103`
- `receipt.json`: SHA256 `a5a24a8fa24b543a502026b351d85ff71a47a49c682f657a498ae150e6a2ceea`

Reviewer JSON comparison: bad-contact exact fixture match = true; test XML count = 12; failures+errors = 0. These are reads/comparisons of produced artifacts, not execution of the diagnostic or test suite.
