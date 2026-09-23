# Publication wording review, 2026-09-23

Reviewer: actual GPT-6-sol. Scope: read-only comparison of `docs/codex_handoff_20260923.md` and the public `contact_geometry_forensics_20260923_01/README.md` against the existing reconciliation, budget closures, original scores, and staged evidence paths. This review did not run a test, scan an archive, import or call MuJoCo, or start the native CCD executable.

## Publication conditions (staging, not published defects)

1. The handoff instructs a successor to read `W/final_publication_receipt_01.json`, which was absent at review time. Root confirms that this is a staging item. Create a pending receipt before publication and fill actual commit/push/CI outcomes only after those actions; do not prefill success. Check the handoff link before final publication.
2. The handoff says to follow a newly added next source contract, and the README says the next step follows an independent frozen contract. At review time, the work directory and public package contain only kernel plans 01 and 02; plan 01 is blocked and plan 02 is exhausted. Root confirms Astra is drafting the next contract. Before publication, name its actual path and SHA256 in both texts after it exists and is frozen. If it is not ready, say it is pending and that no new physical/native call is authorized under the old budgets. Do not let these sentences imply that contract 02 can be rerun.

## Consistency checked

- `zero_call_contract_closure_01.json` and its budget confirm 12 pure cases in one local round, two byte-identical complete offline passes, receipts differing only by pass number, and 2/2 full-pass budget exhausted. The local zero-call phase and the later native CCD phase are separated correctly in both texts.
- `native_ccd_budget_closure_02.json`, `native_ccd_reconciliation_02.json`, and the launcher receipt confirm one process attempt, 16 durable CCD attempts/16 returns, zero remaining queries, descriptor init 8, CCD size 1, explicit pure math 171, version and versionString once each, and zero new model compile/allocation, control, native integration, or training. The initial budget's attempted state is a startup record, not unused capacity. The warning/retry claims agree with the launcher receipt.
- All nine contacts across native 3485–3487 and the fourth primary-only comparison are marked bitwise equal for position, distance, and full frame. Query 7 is the 3486 axis0-negative bad contact with GJK 6, EPA 4, status 0; the texts correctly avoid claiming status 0 proves convergence or that the numerical root cause is known. The fourth case is a local primary-only diagnostic, not a full-robot multiccd-off validation.
- The original box score still says `record_valid=false` and `task_passed=false`; the plane score says `task_passed=true`. The published result paths, adjacent original physical archive path, frozen fixture path, and public provenance paths checked in this review exist. The prepublication identity audit records no mismatches in the frozen 77, formal 215, physical 193, archived 24, and native input 140 sets.
- The kernel 01 compiler hidden-`mj_step` blocker is explicit and its entry remains prohibited. The README's instruction against query 17 and the handoff's no-third-pass/no-old-batch language correctly protect exhausted budgets.

No other material count mismatch or misleading rerun instruction found in the two texts. This is a wording and evidence-boundary review, not an independent physics execution or a fresh hash audit.

## Execution attribution

My bounded implementation work consisted of Python Ruff/AST checks, C syntax-only checks, and pure input preparation. I performed **zero native CCD calls**, zero model/static/dynamic MuJoCo calls, zero tests, and zero full archive passes. Root alone ran the pure tests, two full offline passes, and the one native process containing 16 CCD queries, then reconciled its event log. Astra directed and independently reviewed the plan and source. The three failed Claude connection attempts produced no code.
