# Fixed next contract: stationary-hop shared leg action qualification

Status: **0-integration implementation/qualification only**. This is the sole proposed next task/action mechanism. It is not a claim that hopping is dynamically achievable and does not launch a second training run. Root may implement and review it without another user confirmation; any subsequent finite training/evaluation contract must be explicitly recorded separately.

## Hypothesis and immutable boundaries

For this stationary jump task, expose one bounded common-leg policy coordinate instead of eight independently learned residual coordinates. Apply it only during the already defined 120-execution jump request window. This single time-dependent action embedding jointly constrains relative leg shape and wheel residuals; an eventual result cannot separately attribute improvement to either restriction without another experiment. The falsifiable hypothesis is that coherent extension/retraction exploration produces useful airborne-clearance experience while original attitude feedback handles symmetry errors.

Preserve native plane, robot geometry, nominal support/PD/PI, control/native dt (10/2 ms), actual torque/velocity/position limits, height profile, 95 observation meanings, reward coefficients and event conditions, original net-gap/flight/landing/stability gates, seeds, and source provenance. Do not add thrust, change gains, increase the 40 mm residual scale, relax the 80 mm extension domain, change the requested height profile, densify reward in this candidate, or choose another checkpoint from the failed run. The old failed independent-eight-action class, schema, checkpoint, and records remain immutable.

## Bounded Claude output and exact pure API

Write only a new `scripts/d1_jump_heave_action.py` and `tests/test_d1_jump_heave_action.py`. No environment, controller, training, or evaluation execution; no edits to existing files. The module imports only NumPy and the standard library and implements:

```python
def map_heave_action(action, *, executed_tick: int, request_tick: int | None) -> np.ndarray:
    ...  # fresh float64 array, shape (8,)

def heave_action_metadata() -> dict:
    ...  # fresh JSON-compatible identity/config copy
```

`action` is a numeric finite array of exact shape (1,); reject scalars, wrong shapes, booleans and nonfinite values before applying the gate. Convert to float64 and clip its sole value to [-1,1], matching the parent physical controller's finite-action clipping convention. Tick arguments are non-Boolean nonnegative integers; request_tick=None means no request. Do not infer time from model.time or mutate a latch. For a request at s, active is exactly s <= executed_tick < s+120. No float time tolerance is involved.

Let c=clip(action[0],-1,1). When active, return **[c,c,c,c,0,0,0,0]**. Otherwise return eight positive floating-point zeros. Canonical original channel order is **[FL extension, FR extension, RL extension, RR extension, FL wheel speed, FR wheel speed, RL wheel speed, RR wheel speed]**. Positive extension moves the wheel downward in the body frame. The existing controller applies its unchanged 0.04 m scale to the first four channels; its original 4 rad/s scale multiplies four exact zeros. The matrix is the 8-by-1 column [1,1,1,1,0,0,0,0] within the fixed gate and zero outside.

Examples for s=175: action=[0.5] at k=174 -> zero8; k=175 and k=294 -> [.5,.5,.5,.5,0,0,0,0]; k=295 -> zero8. action=[-1] at k=200 adds -0.04 m to each **requested** extension before unchanged IK/domain/joint-target limits. action=[2] maps to [+1,+1,+1,+1,0,0,0,0]. request_tick=None maps every k=0..599 to zero8. Negative/noninteger ticks, malformed/nonfinite actions fail even when inactive.

Metadata must explicitly contain a new action identity `d1-jump-shared-heave-window-v1`, policy dimension 1, physical dimension 8, ordered channels, the exact embedding matrix, 0.04 m extension scale, unchanged wheel scale 4 rad/s with residual forced zero, gate duration 120 and integer execution semantics. Return copies, never a mutable shared global. Tests cover both s=175 and s=225 boundaries, no-request all 600 ticks, array ownership, deterministic repeated calls, malformed inputs, finite clipping and exact wheel zeros. They must not import/instantiate a simulator.

## Integration constraints to qualify, not implement in the pure module

A subsequent separately versioned environment presents Gym action Box(-1,1,(1,)) to PPO and passes the expanded physical eight-vector to the existing parent step **exactly once**. The task/observation/checkpoint identity must state the new embedding; reject every old eight-action policy before deserializing it. No dimension-based force loading or projection of the final failed policy. Observation length remains 95; the original 85 prefix and action-change reward use the actually applied **eight physical actions**, preserving their original units and meanings. Save both policy_action_1 and applied_physical_action_8. The changed event gate is indexed by the current executed interval, not the next prepared context.

Do not clear controller PI memory, previous physical action, target history, or physical state at gate boundaries. In particular k=s+120 has applied action zero but its previous-action record can still contain the nonzero action from k=s+119. The existing action-change penalty sees that transition. Joint target increment protection is relative to current measured joint position; returning residual to zero does not instantly return qpos, qvel, leg targets or PI memory to a zero-baseline trajectory. Preview must not consume the gate or update memory. A whole no-request 600-step episode has all-zero physical actions from reset and can require baseline physical equivalence in a future actual regression. A post-jump interval cannot require such equivalence.

## Fixed zero-integration numerical qualification

Use only the already saved **zero-condition** four jump evaluations under W/rl_jump_evaluation_01 and the saved zero hold. Preserve their hashes. Fixed set: 480 prepared states (four cases times k=s..s+119), plus k=s-1 and k=s+120 from each case (eight boundary states), plus zero-hold k=200: **489 saved states**. For each active-window state use the three scalar input-space vertices {-1,0,+1}; each of the nine outside/hold states needs only one nonzero policy input to prove physical zero. Total **1449 independent same-state algebra evaluations, 0 transitions**. These are domain-endpoint checks, not a gain grid, rollout, policy selection or an opportunity to tune any value.

A separate scratch helper in a fresh WORK directory may reconstruct saved state kinematics and the original before-compute controller memory. Install fail-on-call guards on mj_step, mj_step1 and mj_step2 before any construction. Restore each saved state/memory independently; never march shadow memory between rows. The zero scalar must reproduce the archived zero-action controller result within explicitly recorded machine-arithmetic tolerance; report maximum discrepancies, do not refit. Check finite requested extensions, original extension-domain clips, true 12-leg target-increment clips, unprotected/safe torque, wheel PI/antiwindup and rated-limit protection. Report saturation and dead regions as findings; the input extremes can legitimately meet the existing +/-0.08 m extension bounds. They must not be widened to make qualification pass.

At both boundaries verify raw and servo commands unchanged, previous physical action retained, PI-before supplied unchanged and PI-after updated only by the original single compute, no preview mutation, correct execute/next-prepare distinction, and zero mapping outside the request. Boundary algebra does not model the unknown candidate's future state. Observe and report the original requested-support/control behavior; do not assert airborne physical work from unsynchronized force/velocity samples.

Qualification passes only if the embedding, metadata/loader rejection, no-hidden-integration guards, zero-action reconstruction and existing finite/rated safety law are correct. Failure of those properties stops integration for a concrete implementation fix. Lack of a global stability proof is not itself a blocker. The result qualifies an action interface, **not** clearance reachability or future learned performance. A later independently frozen learned evaluation must still require native unloaded >=20 ms, net gap >=20 mm, true rising COM, all-wheel load return and the original drift/stability gates; invalid records or merely larger body rise cannot pass. There is no additional physics or training budget in this contract, no automatic continuation, and no rerun of any earlier batch.
