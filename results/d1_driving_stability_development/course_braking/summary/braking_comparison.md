# Course braking implementation check

| Case | Old release Δx (m) | New release Δx (m) | Old final abs(vx) | New final abs(vx) | New diagnostic failures |
|---|---:|---:|---:|---:|---|
| flat_start_gear1 | +0.148469 | +0.105492 | 0.001999 | 0.005552 | none |
| flat_start_gear2 | +0.185677 | +0.129076 | 0.001013 | 0.009447 | none |
| flat_start_gear3 | +0.274233 | +0.148027 | 0.010862 | 0.013388 | none |
| course_rough_gear1 | -0.018693 | -0.025345 | 0.000500 | 0.004832 | steady_vx_within_25_percent |
| course_rough_gear2 | +0.239715 | -0.042942 | 0.037051 | 0.011555 | steady_vx_within_25_percent |
| course_rough_gear3 | +0.117574 | -0.124600 | 0.005883 | 0.043673 | steady_vx_within_25_percent |
| course_ramp_gear1 | +0.173508 | +0.177553 | 0.001184 | 0.003782 | steady_vx_within_25_percent |
| course_ramp_gear2 | -0.140819 | -0.056776 | 0.058507 | 0.034806 | none |
| course_ramp_gear3 | +0.211057 | +0.094723 | 0.002991 | 0.020128 | none |
| course_stairs_gear1 | -0.020317 | -0.033480 | 0.002974 | 0.005805 | steady_vx_within_25_percent |
| course_stairs_gear2 | -0.078382 | -0.068859 | 0.053875 | 0.027851 | steady_vx_within_25_percent |
| course_stairs_gear3 | +0.497145 | +0.117512 | 0.132945 | 0.025388 | none |

All 12 pairs preserve every integrated state and applied torque before release. One release event per run. All four side-step plans preserve the entire physical trajectory and produce no release-reference events. All 77 frozen source hashes match.

These checks validate retained-reference correction. Rough blocks and low-gear stairs still stall; ramp gear 1 still overspeeds before release. Smaller post-release displacement is not terrain traversal or robust hardware performance.
