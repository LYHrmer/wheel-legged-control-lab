# C37 interfaces: three methods changed, everything else frozen

Authoritative design: `astra_plan/contract37.json`. This file states the seams.

## The complete behavioural diff from frozen C35

| file | method | change |
|---|---|---|
| `continuation36/sol36/core36.py` | `Core36._all_loaded` | phase-aware support requirement: inside the pre-unload window it delegates to the frozen `Core35._stance_loaded`, outside it to the frozen `Core35._all_loaded` |
| `continuation36/sol36/adapter36.py` | `PairController36._reference_record35` | declares `support_mode='pair'` with the stance pair in that same window |
| `continuation37/root37/runtime37.py` | `SideAccess37._checked_query` | short-circuits the legacy `fullM` destination branch |

Nothing else. The native guard, the record schema, the paired IK, the coupled inertia
arithmetic, the force allocation, the wheel brake, the torque clipping, the archive writer
and `record35.record_case35`'s 200 preparation / side session / 400 retention accounting are
the frozen C35 code, reused byte for byte.

## Pre-unload regime (inherited from C36, physically confirmed)

`runtime35.NativeWindow35.sink` picks its regime from the consumed reference alone, so
declaring `'pair'` makes the **unmodified** guard enforce the stance-pair gate: a true active
terrain contact and an independently summed positive normal load on each stance wheel, plus a
stance-pair normal sum of at least `0.5 m g`, at every one of the five native substeps, with
no loss grace. `horizontal_active` stays `False` through transfer, so the 12 mm clearance
requirement is neither added nor removed there.

The boundary is the design's own ramp: `1 - smooth35(elapsed/0.06)` passes 0.5 at exactly
half the ramp, so the window opens at `elapsed = 0.03 s`.

C36 attempt 1 confirmed this against physics. At native 1020 to 1024 — the exact substep and
state where C35 raised — the regime is `pair`, the swing wheel carries 0.0 N as the design
intends, the stance pair carries 458.6 N against the 236.2 N requirement, and the gate
passes. `native_guard.failure` was `None` for the whole run.

## fullM destination check (new in C37)

The frozen checker read:

```python
modern = args[1] is self.scratch
packed = isinstance(args[2], np.ndarray) and np.shares_memory(args[2], self.scratch.qM)
if not (modern or packed):
    self.reject('C35 fullM must use registered scratch')
```

MuJoCo 3.12.0 exposes the inertia as `MjData.M` and has no `qM`. The repository's
`d1_fast_side_step._dense_inertia` already handles both bindings and correctly takes the
modern `mj_fullM(model, data, matrix)` branch, so `modern` is already `True` — but Python
evaluates `packed` anyway and dereferences the missing attribute. C35 never reached a
coupled-inertia computation (its ledger records `fullM: 0`), so the defect was never
exercised.

C37 lifts the predicate into a pure function, `registered_destination37(modern, third,
scratch)`, and short-circuits it: `packed` is evaluated only when `modern` is false and only
when the scratch actually exposes `qM`. Acceptance stays exactly `modern or packed`. Seven
pure tests cover it, including one asserting the frozen expression genuinely raises on the
real input and one asserting C37 agrees with the frozen predicate on every combination where
the frozen predicate could run at all.

## Worker seam

`root37/worker37.py` rebinds four globals of the sealed `worker35` in this process before
`execute35` runs, and declares all four here and in the GO:

| rebound | to | why |
|---|---|---|
| `CONTRACT` | the C37 identity | so the worker receipt is honest |
| `bootstrap35` | `bootstrap37` | swaps `PairController35` and `SideAccess35` in the symbol table `execute35` already reads them from, asserting it received the frozen classes first |
| `origins35` | `origins37` | also records `core36`, `adapter36` and `runtime37`, still requiring every module inside the freeze |
| `preflight35` | `preflight37` | checks the C37 contract and budget, and requires the C35, C36 and C37 specification files all frozen |

No byte of any C35 or C36 file is modified. The runtime fences `execute35` hardcodes, 9000
controls and 45000 native substeps, are unchanged and declared as fences; planned
consumption is 3600 controls and 18000 native substeps across two cases.

`continuation35/root35/host35.py` is reused unchanged as the host, and
`read35/read35.py` with `launch_saved_reader35.py` as the independent reader.

## What is deliberately left wrong

The other six corrections the review identified are **not** applied: the nominal crouch stays
at thigh 0.800 rather than the aligned 0.731, the landing lead stays 0.975 s, the swing
profile stays quintic at a 0.16 m/s peak, the lift stays 35 mm, there is no load/transfer
overlap and there is no swing-reaction feedforward. That is the point. With the crouch
unchanged the whole-body COM sits 21.7 mm forward of the contact rectangle centre and
two-support is predicted statically infeasible; this campaign exists to test that prediction
against physics, not to go fast. No speed target is set and none may be claimed.

## Signing

The source GO is signed by root, not by gpt-6-astra ultra, and records
`astra_reviewed: false` with the signing basis in its own fields.
