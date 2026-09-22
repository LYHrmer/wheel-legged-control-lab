# Progress

2026-09-22: Began new exclusive work directory. New control/native count 0/0. No old simulation or training rerun.

Started two actual Claude code-only bounded requests (geometry, env), each <=$2/600s; pending receipts. Root creates exact builder/cache extension and owns integration. Initial broad process listing hit sandbox wrapper; corrected filtered /proc audit found no relevant processes. An initial targeted source path was absent (mujoco_plant.py); corrected to model.py.

Geometry preflight 01: 0 native/0 control, 2589 explicit mj_forward and 2 mj_setConst; robot/solver/initial-state equivalence passed. Failed exact vertical ray at x-face boundary due floating subtraction and contact feature classification due secondary ~0.001 components in multi-contact native normals. No physical run authorized. Fix static qualification against compiled closed-bound distance and actual normal cone residual; preserve failed preflight.

47 pure tests pass with actual native/control0/0, Ruff pass, geometry preflight03 all11 checks pass; root signed execution_preflight01. Constructor/static gates used explicit forward calls, never dynamic load evidence. New fixed physical batch starting once, plane then box.

2026-09-22 post-run: both cases completed 1200 controls/6000 native each. Total 2400/12000 exhausted, no RL. Plane passed. Box original score invalid due one unloaded reversed native candidate normal at index3486/contact5. Four static pose reconstructions used 0 step/forward and reproduced before-step contacts exactly. Supplementary observations do not override qualification. Frozen77/formal215/physical193 and original archive all unchanged. Root ran 47 prephysics + 7 new forensic pure tests, all passed; actual Claude Opus generated latter tests.
