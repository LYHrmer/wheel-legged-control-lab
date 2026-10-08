# C36 interfaces: what the pre-unload correction touches, and what it does not

Authoritative design: `astra_plan/contract36.json`. This file states the seams so the
independent reader and any later review can locate the change exactly.

## The entire behavioural diff

Two methods, in two new files. Nothing else in the control chain is new.

`sol36/core36.py` defines `Core36(Core35)` and overrides one method, `_all_loaded`. Outside
the pre-unload window it delegates to the frozen `Core35._all_loaded` staticmethod so the
original arithmetic is preserved byte for byte. Inside the window it delegates to the frozen
`Core35._stance_loaded`, called with the swing pair exactly as the frozen transfer branch
already calls it, so the stance pair is derived by the original complement logic rather than
by new code. `preunload36()` is the window predicate: `transfer` with
`phase_elapsed_s >= 0.03`, or `transfer_restore` at any elapsed.

`sol36/adapter36.py` defines `PairController36(PairController35)`, installs a `Core36` at
`__init__`, `reset` and `start` (the frozen class builds a fresh `Core35` in the latter two,
and the swap happens before any step consumes it), and overrides `_reference_record35` to
declare `support_mode='pair'` with the stance pair inside the same window. It also records
`preunload36` and `commanded_swing_weight36` so the regime switch is visible in the saved
record rather than implicit.

## Why the native guard needs no change

`runtime35.NativeWindow35.sink` selects its regime from the consumed reference alone:

```python
expected_count = 4 if proof['support_mode'] == 'all4' else 2
```

So declaring `'pair'` makes the **unmodified** frozen guard enforce the stance-pair gate: a
true active terrain contact and an independently summed positive normal load on each stance
wheel, plus a stance-pair normal sum of at least `0.5 m g`, at every one of the five native
substeps, with no loss grace. `horizontal_active` stays `False` through transfer, so the
12 mm clearance requirement is neither added nor removed there. The guard file, the record
schema and the force evidence are all reused as frozen.

## Why this is a correction and not a relaxation

`clarifications35_01.json` placed the whole `transfer` phase in the all-four regime and
required strictly positive measured load on every wheel at every substep with no grace.
`clarifications35_03.json` required the same phase to ramp the intended swing pair's
allocator weight to zero within 0.06 s. The zero crossing of the swing-pair load therefore
always falls inside transfer, and C35 stopped on exactly that at `native 1020`, with wheel 3
at precisely 0 N and an 0.87 mm real gap. The replacement is the stance-pair gate the frozen
air regime already applies one phase later; strictness is unchanged and only the set of
required legs moves, and only where the design has already committed to unloading them.

The boundary is the design's own ramp. `1 - smooth35(elapsed/0.06)` passes 0.5 at exactly
half the ramp, so the window opens at `elapsed = 0.03 s`; a pure test asserts that identity
to 1e-15 and that the weight function reproduces the frozen ramp at every tenth.

## Worker seam

`root36/worker36.py` rebinds four globals of the sealed `worker35` in this process before
`execute35` runs, and declares all four in its docstring and in the GO:

| rebound | to | why |
|---|---|---|
| `CONTRACT` | the C36 identity | so the worker receipt is honest, not labelled C35 |
| `bootstrap35` | `bootstrap36` | swaps `PairController35` in the symbol table `execute35` already reads the class from |
| `origins35` | `origins36` | records the two C36 modules alongside the C35 ones, and still requires every module to be inside the freeze |
| `preflight35` | `preflight36` | checks the C36 contract identity and budget instead of C35's, and additionally requires both the C35 and C36 specification files to be frozen |

No byte of any C35 file is modified. `bootstrap36` keeps the original under
`worker35.bootstrap35_frozen` and asserts it received the frozen controller class before
replacing it. The runtime fences `execute35` hardcodes, 9000 controls and 45000 native
substeps, are unchanged and are declared as fences; the planned consumption is 3600 controls
and 18000 native substeps across two cases.

`root35/host35.py` is reused unchanged as the host, and `read35/read35.py` with
`launch_saved_reader35.py` as the independent reader, since neither contains C35-specific
behaviour that the correction affects. The per-case 200 preparation, side session and 400
retention accounting in `record35.record_case35` is likewise reused as frozen.

## What is deliberately left wrong

The campaign does **not** apply the other six corrections the review identified: the nominal
crouch stays at thigh 0.800 rather than the aligned 0.731, the landing lead stays at
0.975 s, the swing profile stays quintic at a 0.16 m/s peak, the lift stays 35 mm, there is
no load/transfer overlap and there is no swing-reaction feedforward. That is the point: with
the crouch unchanged the whole-body COM sits 21.7 mm forward of the contact rectangle centre
and two-support is predicted to be statically infeasible. This campaign exists to test that
prediction against physics, not to go fast.

## Signing

The source GO is signed by root, not by gpt-6-astra ultra, and records
`astra_reviewed: false` with the signing basis in its own fields. No Astra review is
claimed anywhere.
