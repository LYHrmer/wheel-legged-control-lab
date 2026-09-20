# Jump/obstacle next-step planning, zero new physics

This package is the requested GPT-6-astra / ultra read-only audit and one bounded next-step proposal. It does not implement control, construct a MuJoCo model/controller, run an old experiment, train a policy, or create a GUI. The pending stop/turn composition must pass before root considers its proposed new batch.

`findings.md` traces the original60.86 mm base-origin rise versus5.61 mm simultaneous wheel clearance to the immutable600-row LQR/VMC record, explains phase/contact/geometry limits and the old constructor's possible511-step identification overhead, and distinguishes course plane+boxes from the heading heightfield and native-plane-only environments.

`next_contract.md` specifies exactly one next experiment: new PD/PI hold versus fixed existing raw height-profile stimulus,600 steps each, maximum1200 control intervals /6000 native substeps. No extra force, gain/cap change or jump controller is proposed. It separately checks measured lift, useful simultaneous clearance and settled landing, and keeps all obstacle crossing/reliability claims outside the diagnostic. The only bounded module requested from real Opus later would be pure schedule/geometry instrumentation; root owns execution.

`read_legacy_jump.py` is a reproducible standard-library-only helper. It decompresses existing JSON and reproduces the old metrics without importing the robot, controller, MuJoCo or NumPy. `legacy_jump_audit.json` records the result and immutable input hashes. Its `passed` flag means the saved trace reduction matches the archived summary; it does not mean the old jump cleared a useful obstacle.

Reproduce into a **new** output file, never overwrite the packaged result:

```bash
rtk proxy python3 -B /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/jump_obstacle_next_plan_01/read_legacy_jump.py --input /home/lyh/wheel-legged-control-lab/results/d1_driving_stability_development/jump_baseline --output /tmp/legacy_jump_readonly_new.json
```

`proposed_cases.json` fixes the two schedules and proposed budget. `input_sha256.json` pins this planning snapshot; root later adds the accepted combined-controller/audit and actual new-module/test/runner identities to its preflight. `checksums.sha256` freezes the package itself.

Independent bounded source checks by `/root/astra_stop_after_release` and its reviewer confirmed Space/R behavior, LQR/VMC identity, construction-time identification, the existing course box geometry and lack of a public isolated-box builder hook. Neither reviewer instantiated or ran the backend.

R remains a simulation reset. No new RL GUI exists. No flat-plane/stop/turn pass in this package is claimed as jump, obstacle, speed or dynamic restart capability. All historical files and frozen77 remain unchanged.
