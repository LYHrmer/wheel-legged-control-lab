# GUI13 revision02 final incremental source review

Decision: **SOURCE GO** for the exact sources below, run through the reviewed outer host. The two blocking findings in `astra_revision02_go.md` are resolved. That earlier record remains unchanged. This is a bounded source review, not a physical result; this reviewer ran no tests, model calls, engine imports, or controls.

## Final source identity

Paths are relative to continuation13.

| File | SHA-256 |
|---|---|
| gui13_revision02/launch_gui13.py | `1cd33d139ee55e532178c5eadbd47da0214a9ec5397f7e52995af02721b4fe7b` |
| gui13_revision02/run_gui13.py | `341ded83608b52034bd623c2ff47e7521d78578552558d4d0286c0d341b091c7` |
| gui13_revision02/gui13_contract.py | `c4f88161e65d31b4bbd9267c2c05e1a5a13941376c4ef856517d443561b59cdb` |
| gui13_revision02/prepare_gui13_plan.py | `fb3f25a90be0f1b35a65da7cce48729ff3daee184ad666b774d3f7df7e758900` |
| gui13_revision02/phase_contract.md | `5a7615a6759aa8f547bec8f38c525139a42c0e1c126855b5bf97efa30838e042` |
| run_stage13_root_host.py | `89201e0bb751cb919954f7ff7d9bb19062f8d77dd6bc87fe0f06cf760f5722bd` |

## Standards

No blocking standards finding in the two-fix delta. The initial streaming-hash and single-source-table conclusions remain valid. The original control, archive, bridge, and independent reader algorithms were not reopened.

## Spec

Both earlier blockers are closed by the actual source:

- The outer host sets the monotonic preflight origin before launching and passes it through the environment. The GUI outer launcher preserves that origin. The independent outer loop enforces the 240-second no-readiness deadline while the GUI outer launcher is in xauth/Xvfb startup as well as during inner/worker preflight. Outer and inner readiness checks bind an owned live PID, session SHA, source count, completed-source-check flags, and a timestamp between the shared origin and its 240-second boundary. The worker retains its alarm through the exclusive readiness write, flush, and fsync; a preflight exception therefore does not proceed into `run()` or its engine/model imports. A missing readiness cannot yield a successful outer-host return.
- The outer host successfully enabling Linux `PR_SET_CHILD_SUBREAPER` is now a prerequisite to `Popen`. `discover()` adds its direct children, including reparented descendants, then follows remaining live descendants with matching PID/start ticks. Thus a setsid child whose intermediate parent exits between scans is recoverably owned by this host. TERM/KILL cleanup retains the reviewed ownership checks and adopted-child reaping, with no broad process-group sweep. Failure cleanup is bounded by the remaining outer window and the documented four-second grace.

The final phase contract is now 240 seconds of pure preflight, execution soft/close/hard at 95/115/120 seconds from worker readiness, a separate 180-second full-SHA postcheck after worker cleanup, and a 600-second outer total including isolated display cleanup. Hashing remains part of the reported total wall. These are source-enforced bounds; the actual receipts must still show the phase durations, completed posthash, clean exits, and no owned survivors before an arm is accepted.

After the final plan is rebound to these sources and root's pure `--check` passes, root may execute the already approved **single headless seed-88813, 600-control development arm** using this exact outer host. Only successful independent saved-record/numerical readback and phase/cleanup receipt checks permit the **single matched GUI arm**. The per-arm 3000 normal native plus two cold calls, load one, probe one batch, 600 regular predictions, zero learning/saving, no physical retry, and all prior physical/numerical/performance gates remain unchanged. This approval does not qualify the default GUI or expand the original task scope.

Standards: 0 blocking findings. Spec: 0 remaining blocking findings in the reviewed delta.
