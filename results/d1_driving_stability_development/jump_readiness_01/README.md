# Stationary height-profile probe and residual PPO task preparation

One fixed hold/profile pair completed 1200 control intervals / 6000 native steps. Both episode records, actual torque protections, source identities and the common physical/observation prefix pass. Frozen77 inputs and old experiments remain unchanged.

The profile does **not** qualify as a useful jump: its unloaded interval lasted 12 ms (requires20 ms); maximum simultaneous four-wheel cylinder gap was1.4913875 mm, or0.4913875 mm net of the1 mm contact margin (requires>21 mm raw for this readiness probe). Individual front-wheel peaks of36.459/36.447 mm do not replace the simultaneous minimum. The hold and profile both meet their fixed late settling gates. This is a valid zero-action baseline for new RL.

Actual Claude Opus supplied the measurement, readiness, PPO task/env drafts and part of the tests. Root integrated real APIs, completed the runnable archive, and fixed two task-reward defects: pre-request flight carryover and post-landing rebound credit. One original test incorrectly allowed rebound progress; its failed run and corrected contract tests are retained. Root ran261 nonintegrating checks and the unique physical pair. Later direct Claude repair/test/runner/scorer calls timed out with no generated code; timeout receipts are retained. Do not label those calls successful.

The new task retains original eight residual actions, adds ten explicit oracle task features to the original85, and preserves default controller/actuator limits. New PPO smoke/formal/evaluation results are **not** in this package. Their frozen plan is included. No old65k or24G1 rerun, new GUI, obstacle traversal, higher-speed qualification or physical self-righting is claimed.

Raw native solved-contact forces and returned-pose geometry have different recorded phases. Geometry/whole-robot COM reconstruction uses scratch kinematics/Jacobians only; no new integration or contact solve. Source snapshots and actual Claude/root diffs are under provenance/.
