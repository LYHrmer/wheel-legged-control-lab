# Dynamic braking: six paired development checks

**Candidate rejected for product integration.** Legacy remains the default.

Each row covers the complete eight seconds after release. Values are legacy → dynamic candidate; distances use the last commanded direction (S reverses the world-x sign).

| Case / release | Peak forward (m) | Total backward (m) | Peak retreat (m) | Max reverse speed (m/s) | Last2 mean absolute speed (m/s) | Last2 signed world drift (m) |
|---|---:|---:|---:|---:|---:|---:|
| flat3 / 1 | 0.304756 → 0.316382 | 0.000000 → 0.000000 | 0.000000 → 0.000000 | 0.000000 → 0.000000 | 0.006697 → 0.008126 | 0.013394 → 0.016253 |
| ramp2 / 1 | 0.048758 → 0.052513 | 0.309969 → 0.193097 | 0.308819 → 0.193097 | 0.058967 → 0.029683 | 0.006622 → 0.026786 | -0.002609 → -0.053572 |
| rough3 / 1 | 0.126712 → 0.077782 | 0.028450 → 0.416399 | 0.028450 → 0.416399 | 0.006598 → 0.072320 | 0.004435 → 0.049293 | -0.008869 → -0.098585 |
| stairs3 / 1 | 0.506551 → 0.236218 | 0.008136 → 0.000000 | 0.008136 → 0.000000 | 0.013477 → 0.000000 | 0.004046 → 0.001845 | 0.008091 → 0.003690 |
| flat_reverse3 / 1 | 0.230652 → 0.234219 | 0.078640 → 0.049251 | 0.078640 → 0.049251 | 0.015138 → 0.009767 | 0.011507 → 0.008182 | 0.023013 → 0.016364 |
| flat_stop_go3 / 1 | 0.290272 → 0.299009 | 0.000000 → 0.000000 | 0.000000 → 0.000000 | 0.000000 → 0.000000 | 0.006178 → 0.007183 | 0.012355 → 0.014366 |
| flat_stop_go3 / 2 | 0.269312 → 0.465354 | 0.000000 → 0.000000 | 0.000000 → 0.000000 | 0.000000 → 0.000000 | 0.001591 → 0.033307 | 0.003183 → 0.066615 |

12 completed runs, 26,400 control transitions, seven complete paired release windows. All six first-release prefixes and four historical legacy prefixes are bitwise identical. All states are finite and torques remain within bounds; zero recovery ticks. Every per-tick reference update and every reported release metric was independently recomputed from retained records.

The rough-spawn case is uphill by release. Its retained error drops from 0.550000 m to 0.081306 m immediately; it never reaches the 30 consecutive low-speed ticks needed to enter hold. The second stop-go release also fails to enter hold. The speed-based bound permits persistent creep while preventing the reference error from building the prior restoring action. This falsifies this particular reference-only stopping hypothesis; it does not establish that every dynamic braking design fails.

Ramping into hold itself is not a stop guarantee: ramp2 enters hold at 14.73 s but ends with a higher final-two-second speed than legacy. The unchanged underlying ±0.55 m reference clamp and v*dt integration are explicitly retained. No terrain branch or parameter search followed these failures.
