# Saved-record reader revisions

The successful run used `verify_short_rl16_partial_11_04.py`, pinned to
`verify_short_rl16_training_11_04.py`. Its [host receipt](evidence/publication_20260929/offline_attempt_05/receipt.json)
records exit 0 and 190.878 seconds. It imported neither the engine nor a policy,
and replayed no physics. It checked saved training blocks, checkpoint identity,
nine complete native/contact chains and four bitwise-matched initial states.
The [readback](evidence/rl11/partial_readback_20260929_05.json) separates verified
individual task scores from the unfinished six-task paired qualification.

Earlier readers failed on three reader assumptions. Their source snapshots and
failure receipts remain in this package; no original experiment file changed:

1. `num_timesteps` does not advance inside PPO `train()`; it advances during
   rollout collection. The reader now checks the recorded zero within-train
   delta and the separate cumulative 1,024-step rollout counts.
2. A reader-local array dictionary shadowed its integer control count, causing
   a `TypeError`. The new revision uses separate variable names.
3. The inherited controller checker equated a policy prediction with the
   controller's input. The experiment's frozen environment clips the prediction,
   gates it to zero at stationary commands, then sends that physical vector to
   the controller while retaining the prediction in the record. The new checker
   verifies each stage separately and recomputes the controller from its actual
   physical input. It does not alter records or weaken torque/contact checks.

An earlier reader03 tool session was lost during a daemon restart. After
confirming that no process or output remained, the durable
[attempt 04](evidence/publication_20260929/offline_attempt_04/receipt.json)
captured the gate-assumption failure. This was an offline reader failure, not
evidence of a physical failure or corrupted completed trajectory.

The interrupted tenth experiment case is different: its incomplete block and
missing endpoint/case receipts remain incomplete. The
[raw-file manifest](evidence/rl11/partial_closed_manifest_20260929_05.json)
records those bytes without treating them as a completed case. Raw trajectories
remain local; their hashes do not imply that they are available on GitHub.
