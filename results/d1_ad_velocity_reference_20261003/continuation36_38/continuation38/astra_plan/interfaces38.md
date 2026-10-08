# C38 interfaces: a record-semantics fix, nothing physical

Authoritative design: `astra_plan/contract38.json`.

## The complete diff from C37

One method: `continuation38/sol38/adapter38.py`'s `PairController38._reference_record35`
starts from the **frozen** `PairController35._reference_record35` (not C36's), then declares
`support_mode='pair'` and `stance_legs` for the pre-unload window, leaving `swing_legs` at
whatever the frozen record already computed from `reference.phase`.

`core36.Core36._all_loaded` and `runtime37.SideAccess37._checked_query` are inherited
unchanged from C36/C37.

## Why this cannot move the trajectory

The native guard's regime selection reads only `support_mode` and `stance_legs`. The
controller's own `self.swing_legs` — which gates the coupled-inertia (`fullM`) computation
in `PairController35.compute` and the swing-leg PD gains — is set independently by the frozen
`compute` from `reference.phase in AIR_PHASES35`. The declared record and the controller's
internal state are two different things; C36 conflated them by writing a value into the
record that didn't match the controller's own gating. C38 only touches the record.

So C38 is expected to reproduce C37's 299 controls and `touchdown_dwell_timeout` bit for bit
on every physical quantity. Only the saved record, and therefore the independent audit
outcome, should differ. This is the determinism check: if any native row differs, the record
change had a physical side effect that was not supposed to exist and must be investigated
before anything else proceeds.

## Worker seam

`root38/worker38.py` rebinds the same four `worker35` globals as C36/C37, declared in the
GO. `PairController35` and `SideAccess35` are swapped for `PairController38` and
`SideAccess37`.

## What C38 is explicitly not

Not a fix for the touchdown rocking C37 measured. That requires changing the entry crouch
pose, which the C35/C36/C37/C38 controller never chooses — it only inherits whatever pose
B22 handed off at control 200. Correcting it needs a new sub-phase in the state machine
(a controlled pose transition before the first `transfer`), which does not exist yet and is
scoped as a separate, later contract with its own pure tests and its own GO. C38 does not
attempt it.
