# 06 common body-speed feedback: deterministic simulation record

Execution valid: **True**; candidate qualified: **True**. The same frozen learned checkpoint ran under the new controller law. No independent RL benefit is established without a same-law zero-residual comparator.

## Four fixed cases

| Command | Terrain | Task passed | Mean body vx (m/s) | Speed RMS (m/s) |
| ---: | --- | --- | ---: | ---: |
| 0.20 | plane | True | 0.201299644881 | 0.001546767046 |
| 0.20 | box | True | 0.200093709931 | 0.036520709438 |
| 0.25 | plane | True | 0.251664512870 | 0.002004290905 |
| 0.25 | box | True | 0.250394443929 | 0.035807931463 |

The unchanged speed gate uses endpoints 275..875 (601 samples), mean body vx at least 90% of command and RMS error at most **0.05 m/s**. Policy mean-speed increase from 0.20 to 0.25 m/s: plane 0.050364867989 m/s; box 0.050300733997 m/s (required at least 0.03 m/s on both terrains). All original physical task gates remain unchanged. The independent root readback rescored the full saved native chain and physical gates. The portable verifier checks identities, record links, and speed arithmetic; it does not itself repeat physical contact-geometry scoring. These deterministic cases do not establish hardware or statistical reliability.

Full raw asset: https://github.com/LYHrmer/wheel-legged-control-lab/releases/download/body-speed-qualified-20260924/wheel-legged-body-speed-20260924-raw.tar.gz
SHA-256: `3f11d67c35568e74e2b3f927247ec77a90fbf4ed3b90df0fc572227632bc735a`; bytes: `73568857`.

Run `provenance/offline_verify_06.py --package PACKAGE --archive-root EXTRACTED_RAW` after extracting the asset. Historical 05 raw data remains in its earlier release.
