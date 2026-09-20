# Fixed RL jump contract — frozen 2026-09-20

RL is the next main experiment after valid readiness records; zero baseline jump failure does not block learning. This package creates no controller, model, physics, training, or old rollout replay.

Read `fixed_contract.md`, then `claude_execution_spec.md`. Actual Claude worker A implements task/env and their two tests; worker B implements runner/test against the reviewed first-stage API. Reuse `scripts/d1_jump_readiness.py` directly. Root reviews the concrete artifacts and launches execution.

`training_protocol.json` fixes 640 smoke + 131072 fresh PPO transitions. `evaluation_protocol.json` freezes five zero/final-policy pairs, maximum 6000 controls. Total maximum 137712 controls / 688560 native substeps; the prior 1200-control readiness probe is separate. Evaluation seeds are 77101..77105 in listed case order; new training reset seed namespaces are disjoint. No old 65k policy load/replay, no 24-case G1 replay, no 77 frozen-source changes.

The helper API and original SHA are pinned. This package contains plans and JSON only. Runtime source hashes, valid readiness/native archive, composition audit and worker test/output-scope evidence remain required in root's execution preflight; their absence here is not a fabricated completed preflight. A positive learned result would qualify only the declared stationary oracle hop task, not GUI, obstacle crossing or hardware robustness.

`checksums.sha256` covers every other file in this directory. JSON was parsed, arithmetic/case/seed separation checked, and the actual helper SHA verified using standard-library file operations only (0 integration).
