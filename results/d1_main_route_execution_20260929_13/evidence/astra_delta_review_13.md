# GUI13 final interface delta review

Decision: **SOURCE GO** for the four exact files below, subject to the already approved one-shot plan and root's pure launcher preflight. This is a source and saved-receipt review, not a physical execution result. The previously frozen `astra_plan_13.md` is unchanged.

| File under continuation13 | SHA-256 |
|---|---|
| gui13/run_gui13.py | `3dd6f14bbe7a1aa58ff95090a6fb50aa04a97edc015db1e6c3e54aa475c119a2` |
| gui13/launch_gui13.py | `1778bc305956d6eb819e7f7f7d626c14dc06cd6e88d6f32902c8db978ddafa0d` |
| sol_verify13/verify_gui13.py | `a6c30d9dcfef7251a1087babaf8f226c1e1adcd2c45f595f309d1efefa897b7c` |
| sol_verify13/test_verify_gui13.py | `5d7e05057bff46340fc548b2c56eb8feee2f211b9bbe9ded57297953897bbe35` |

Reviewed against the actual producer interfaces, independently of the reader fixtures:

- `course_native_guard_08.py` emits `mode=train`, `failure`, native attempted/returned/checked, contact and attitude fields. The compact segment uses `record_valid`, `archive_failure`, and `full_contact_qualification_recorded`. Reader mappings match these actual fields and retain the compact-only limitation.
- `d1_rolling_engine_runtime.py` closes its allowed native phase with `phase=0` and zero `target_model`/`target_data`. The reader checks these actual C-state keys alongside the distinct Python control/native ledger counts.
- `ActuatorTrace` contains exactly `requested_nm`, `limited_nm`, `delayed_nm`, `applied_nm`, and `joint_velocity_rps`. The worker now archives the existing five traces per control and existing compiled `actuator_ids`; these additions perform no model or engine calls and change no control calculation. The reader binds requested torque to controller safe torque and applied torque to both native control arrays through the saved actuator ordering. It also checks continuity across every adjacent native state and all 600 control endpoints. Applied actions require finite 16D vectors and a nonzero value somewhere in the arm.
- `async_course_renderer_12.py` performs a render for both `rendered` and `reused` statuses. The reader divides their active-window count by actual active wall seconds, checks monotonic poll intervals and snapshot ages, and keeps GUI performance separate from numeric equivalence. Screenshot records use `png_sha256` and a pre-flip OpenGL `pixels_sha256`; decoding the PNG and flipping its rows back matches the producer. Initial, drive, and final captures require the corresponding control indices, valid bytes, nonzero pixels, and contrast.
- The launcher freezes only the two `sol_verify13` reader sources, together with the pure-test receipts. `run_arm` additionally freezes the GO plan itself immediately after preflight; the previous concern about a missing plan hash does not apply to the current source. Pair reading requires identical source closures, plan identity, evidence, traces, states, and native records.

Root's actual `reader_green_receipt_01.json` binds the reader and test hashes above and reports five pure tests passing in 0.004 s with zero physics controls and zero model calls. The performance fixture includes an explicit low-FPS failure. Those fixtures are a limited pure check; producer-field inspection supplies the interface evidence described above.

Execution scope is unchanged: root alone may run one new seed-88813 headless arm of at most 600 controls, read its saved records, and proceed to the single matched GUI arm of at most 600 controls only after the headless execution/numeric gates pass. Each fresh process retains at most 3000 normal native calls plus two cold compiler calls; the prior model-call, 95/115/120-second, no-retry, ownership, and failure-stop limits remain in force. Root separately records isolated-X11 outer elapsed time and cleanup. No old training or qualification case is authorized for replay. This review does not qualify the default GUI, hardware keyboard latency, arbitrary manual operation, a 1600-control case, or full contact/force behavior.
