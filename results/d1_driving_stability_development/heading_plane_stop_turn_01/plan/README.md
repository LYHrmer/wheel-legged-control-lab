# Stop/turn composition planning package

This is a fixed, zero-physics planning handoff, prepared by the explicitly requested GPT-6-astra / ultra agent. It contains no controller implementation or execution result. The qualified standalone stop and turn evidence is pinned by `input_sha256.json`; the successful left turn was not extrapolated to the right: both completed authority results and the independent audit were available before this package was frozen.

- `opus_core_spec.md`: bounded specification for the real Claude Opus request, writing only the new single-inheritance core. Root owns integration and validation.
- `fixed_contract.md`: exact one-PI composition, two independent gates, plane environment seam, metadata, observations, scoring, phase-separated contact evidence, baseline pairing and failure accounting.
- `cases.json`: original six cases unchanged plus two mirrored settled-stop -> turn handoffs; exactly 8400 candidate control intervals / 42000 native substeps maximum, zero training. Unique stored baseline data reused:5600 intervals; additional1600 prefix comparisons reuse those same data again and are not physical reruns.
- `input_sha256.json`:40 exact existing input identities. Case complete manifests transitively name the underlying trajectory files. The new implementation/runner/test identities must be added to root's later execution preflight.
- `planning_checks.json`: file-only consistency checks, not controller qualification or physics validation.
- `checksums.sha256`: hashes of this planning package, excluding the checksum file itself.

The control mechanism is unchanged fixed stop damping after an actually executed nonzero-to-zero forward transition, combined with the qualified original inner yaw-feedback formula only during raw pure turn. The only new physical interaction tested is the stop latch remaining active through a later turn pulse. The contract uses one original PI update and does not merge the failed center-compensation or turn-only-damping candidates.

The independent handoff review by `/root/astra_stop_after_release` confirmed the fixed8-case budget and the critical slice boundaries: no1400-row dilution of original stop RMSE; no forged truncation at800; observations800 and next-prepared fields at execution799 excluded from handoff prefix equality; turn score shifted600 ticks with position origin S600 and the original continuous reference. Nonzero-forward unlatching/rearming is a pure sequence-test prerequisite, not a claim that dynamic restart has been physically validated.

No source, frozen input, prior record or default controller was changed to produce this package. No controller was instantiated/called, no MuJoCo operation ran, no model qualification was repeated, and no physical state was propagated. The root's later tests and single fixed run must retain their own receipts, source hashes and actual counts. Success would qualify only the listed zero-residual flat-plane cases and two stated handoffs; obstacle/jump/speed objectives remain subsequent work.
