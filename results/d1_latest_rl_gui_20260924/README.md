# D1 latest RL interactive GUI — 07 evidence

On the validated machine, `rtk proxy d1-rl` opens a fresh default policy/box window;
from the repository root, `rtk proxy python3 -B scripts/play_d1_latest_rl.py`
is the source entry. Append `--check` for a read-only source preflight.

This package records two bounded validation sessions of the published final RL checkpoint
and the 06 body-speed controller. It documents the actual saved results; it does not
establish an independent RL benefit, hardware behavior, or broad terrain reliability.

The GUI box run used the saved final policy; the headless plane run used exact zero
policy residual with the same baseline controller. Different terrains make the two
profiles an interface check, not a causal policy comparison.

| Profile | Controls | Normal native | Compiler | Readback |
|---|---:|---:|---:|---|
| gui_policy_box | 1200 | 6000 | 3 | PASS |
| headless_zero_plane | 1200 | 6000 | 3 | PASS |

The GUI box run's switched-command 0.25 m/s interval had mean observed body speed 0.25558668 m/s over 400 controls.
This interval includes transients and is descriptive, not the 06 fixed-speed
qualification window. The 12-second GUI simulation took 130.25 seconds of wall
time in validation; real-time playback and human keyboard usability are unproven.

The full raw asset is intended for the `latest-rl-gui-20260924` release:
https://github.com/LYHrmer/wheel-legged-control-lab/releases/tag/latest-rl-gui-20260924
Confirm the actual published URL, asset digest and CI in the final publication receipt.

`summary.json` links the compact evidence to the separate full raw asset;
`publication_manifest.json` lists every compact file hash. The archive manifest
lists the complete closed A/B run files, including native and trace records.

Earlier frozen packages remain available at the existing releases:
https://github.com/LYHrmer/wheel-legged-control-lab/releases/tag/rolling-rl-damping-20260924
https://github.com/LYHrmer/wheel-legged-control-lab/releases/tag/body-speed-qualified-20260924
