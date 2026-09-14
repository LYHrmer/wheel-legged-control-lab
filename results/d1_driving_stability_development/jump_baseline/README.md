# Existing flat-course jump diagnostic

600 actual control transitions: settle 2 s, request Space, observe 4 s. The unchanged LQR/VMC jump raises the body by 60.862 mm, but the maximum simultaneous minimum wheel-bottom clearance is only 5.610 mm. These are different geometric measures. No improved jump is claimed.

`trace.json.gz` decompresses to the complete original JSON trace: every qpos/qvel, torque, support force, wheel-bottom height and phase. `protocol.json` records source identities and the uncompressed SHA256. Geometry is evaluated with `mj_forward` on a copied `MjData`; the live integrator is untouched. The peak search uses samples in (2, 4] s; the full trace extends to 6 s.

Reproduce from repository root with `PYTHONPATH=.local-deps:src:. python3 results/d1_driving_stability_development/jump_baseline/jump_probe.py --output /tmp/d1-jump-new`. Output must not already exist. This diagnostic does not establish hurdle traversal, flight duration or repeated landing reliability.
