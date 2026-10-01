# C31 saved-only post-evaluation diagnosis

Formal branch: **gain_gate_failed**. This document quotes the independent v2 gates; it does not replace them.

| Condition/direction | Formal passed | Learned/zero cycle | Learned/fixed cycle |
|---|---:|---:|---:|
| nominal_left | False | 0.917536 | 0.998968 |
| nominal_right | False | 0.917536 | 0.996910 |
| heldout_left | False | 0.917536 | 0.998968 |
| heldout_right | False | 0.916588 | 0.995881 |

Controls descriptively read: 20761; genuine latches including training: 306. The 64 training cycles contributed macro/cycle/batch records only.

Phase durations, action residuals, actual overlap, whole-wheel/prior-5T eligibility and torque-square cost are in `diagnostic.json`. Same-phase post-nominal counts are observed waiting, not a single speed cause. Torque-square is not energy, and the 400-control retention is a qualification window. No new model/physics/native payload was run or rescanned.
