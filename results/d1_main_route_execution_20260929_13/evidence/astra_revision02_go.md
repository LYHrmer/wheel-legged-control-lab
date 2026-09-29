# GUI13 revision02 bounded incremental review

Decision at the source snapshot below: **NO-GO pending two specific phase/ownership fixes**. This review performs no engine import, model call, physical control, test execution, or replay. The prior numerical and interface SOURCE GO remains the baseline; only the revision02 diff and the new outer host were reviewed.

## Reviewed source identity

Paths are relative to continuation13.

| File | SHA-256 |
|---|---|
| gui13_revision02/launch_gui13.py | `e864f0fd2010336e73548f34368f51ddb07f96a00ae91adb433e6db0d542474b` |
| gui13_revision02/run_gui13.py | `0c7c812ce8351345da4f6916684f2de70b58d9d727a2fcc25a436315371e0d8a` |
| gui13_revision02/gui13_contract.py | `c4f88161e65d31b4bbd9267c2c05e1a5a13941376c4ef856517d443561b59cdb` |
| gui13_revision02/prepare_gui13_plan.py | `fb3f25a90be0f1b35a65da7cce48729ff3daee184ad666b774d3f7df7e758900` |
| gui13_revision02/phase_contract.md | `e558e4548ec376acfe65610db7cb9d69ddfeccaced204e9d533062e55eae9ff9` |
| run_stage13_root_host.py | `bc292a3e00a7df0c66ad1f53ea14b414f659bbd045b9501bfb11d4793acc5bdf` |

## Standards

No separate blocking standards issue found in this bounded diff. `gui13_bridge.py` and `test_gui13_bridge.py` are byte-identical to the previously reviewed versions. Streaming digest counts the actual bytes read and preserves the digest shape. `sources()` resolves one closure and compares the training identities against that same computed table, removing the duplicated full dependency read without dropping the independent worker SHA check.

## Spec

1. **The shared 240-second pure-preflight bound does not yet cover the entire GUI path.** `phase_contract.md` explicitly includes private Xvfb startup. In `launch_gui13.py:299-300`, the outer SIGALRM context ends immediately after `preflight()`. The subsequent `isolated_x11.run()` performs xauth and Xvfb startup before starting the inner launcher; those operations may continue after the shared 240-second deadline. The existing outer host only intervenes at 595 seconds. For example, an outer SHA pass finishing just before 240 seconds permits another xauth/Xvfb startup window before the inner launcher rejects the expired deadline. In addition, the worker cancels its preflight timer before writing readiness (`run_gui13.py:660-676`), and the launcher's readiness check (`launch_gui13.py:219`) has no upper bound of `started + 240`. Keep one independent outer preflight watchdog through validated worker readiness, share its monotonic origin across all launchers, require the readiness timestamp to be within 240 seconds, and retain the worker preflight timer through readiness persistence. This does not require changing the physics or numerical contract.

2. **Polling PPIDs alone can miss an owned orphan and produce a false cleanup success.** `run_stage13_root_host.py:66-79` discovers only descendants whose parent remains in the current process snapshot and in `tracked`; scans occur every 0.2 seconds. If a previously unseen parent forks a child in a new session and exits before the next scan, the live child is reparented outside the known tree and never enters `tracked`. The final `no_orphans` check therefore cannot prove the requested cleanup on the parent-crash path. Enable a Linux child subreaper before spawning, discover adopted direct children as well as tracked descendants, and keep the birth-identity checks; an equivalent ownership mechanism is acceptable. Do not expand cleanup to unrelated process groups.

The requested seed, 600 controls, 3000 normal native plus two cold calls per arm, model-entry limits, 95/115/120-second execution phases, independent 180-second posthash, no-retry rule, and headless-readback-before-GUI order are otherwise unchanged by the reviewed diff. No old tests need to be rerun solely for this review. Re-review only the two fixes and bind their final source hashes before the one-shot physical run.

Standards: 0 blocking findings. Spec: 2 blocking findings, covering preflight timing and descendant ownership.
