"""Independent saved-record-only readback of the unique 10 terminal probe.

No model, simulator, policy or learner is imported or executed. This qualifies
only a task-failure terminal-observation return and an eight-control reset.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W.parent))
import numpy as np
from verify_course_e_08_03 import (
    _body_com_velocity,
    _roll_pitch_deg,
    check_binding,
    close,
    identity,
    require,
)
from verify_rl16_training_08 import (
    _ground_hit,
    _terrain_reference,
    _yaw_from_qpos,
)

SCHEMA = "d1-terminal-probe-10-v1"
CONTRACT_SHA = "8da4297b163c3f03ce776e34bd0f27307aec394ec7b2e33833d061bf3b38cf3f"
NATIVE_KEYS = (
    "qpos_before", "qvel_before", "ctrl_before", "qacc_warmstart_before",
    "qpos_after", "qvel_after", "ctrl_after", "qacc_warmstart_after",
    "qacc_after", "actuator_force_after", "qfrc_actuator_after",
    "start_time_s", "end_time_s",
)
STATE_KEYS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation", "time")


def _json(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), "saved JSON absent: " + str(path))
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "saved JSON is not an object: " + str(path))
    return value


def _npz(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file() and not path.is_symlink(), "saved NPZ absent: " + str(path))
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _same(left, right) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return (a.shape == b.shape and a.dtype == b.dtype
            and a.tobytes(order="C") == b.tobytes(order="C"))


def _numeric_list_matches(left, right) -> bool:
    """JSON loses dtype, so compare numeric values with the saved float64 type."""
    return _same(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))


def _verify_source(run: Path) -> tuple[dict, dict]:
    session = _json(run / "session.json")
    launcher = _json(run / "launcher_receipt.json")
    worker = _json(run / "worker_receipt.json")
    require(session["schema"] == launcher["schema"] == worker["schema"] == SCHEMA
            and session["phase"] == "terminal_probe"
            and session["output_directory"] == str(run)
            and session["contract_sha256"] == CONTRACT_SHA
            and session["control_limit"] == 720
            and session["normal_native_limit"] == 3600
            and session["compiler_native_limit"] == 2
            and session["wallclock_limit_s"] == 180
            and session["segments"] == [697, 8]
            and session["seed"] == 1322591124
            and session["retry_permitted"] is False,
            "terminal probe session/source/budget identity differs")
    frozen = session["source_hashes"]
    require(isinstance(frozen, dict) and len(frozen) >= 100,
            "terminal probe source closure absent")
    for name, expected in frozen.items():
        require(Path(name).is_absolute() and identity(Path(name)) == expected,
                "terminal probe frozen source changed: " + name)
    for path in (W / "run_terminal_probe_10.py", W / "terminal_safe_course_10.py",
                 W / "next_contract_10.md", W / "fixture_01/manifest.json",
                 W / "astra_terminal_probe_go_10.json"):
        require(str(path) in frozen, "critical terminal10 input not frozen: " + str(path))
    go = _json(W / "astra_terminal_probe_go_10.json")
    require(identity(W / "astra_terminal_probe_go_10.json")["sha256"]
            == session["go_sha256"] and go["decision"] == "GO"
            and go["contract_sha256"] == CONTRACT_SHA,
            "terminal probe executed without actual signed GO")
    for name, expected in go["inputs"].items():
        require(frozen.get(name) == expected and identity(Path(name)) == expected,
                "terminal probe GO input differs: " + name)
    require(launcher["exit_code"] == 0 and launcher["failure"] is None
            and launcher["source_hash_mismatches"] == []
            and launcher["worker_exited"] is True
            and launcher["owned_worker_cleanup"]["no_orphans"] is True
            and launcher["fully_reserved_budget_closed"] is True
            and launcher["retry_permitted"] is False
            and worker["failure"] is None and worker["warnings"] == []
            and worker["execution_complete"] is True
            and worker["record_valid"] is True
            and worker["source_hash_mismatches"] == []
            and worker["policy_loaded"] is False
            and worker["checkpoint_loaded"] is False
            and worker["learning_or_training_performed"] is False
            and worker["retry_permitted"] is False,
            "terminal probe host/worker was not a fully closed no-policy run")
    fixture = Path(session["fixture_directory"])
    manifest_path = fixture / "manifest.json"
    require(identity(manifest_path)["sha256"] == session["fixture_manifest_sha256"],
            "terminal10 fixture manifest drifted")
    manifest = _json(manifest_path)
    require(manifest["schema"] == "d1-terminal-saved-fixture-10-v1"
            and manifest["completed_control_rows"] == 696
            and manifest["native_indices"] == [1136000, 1139485]
            and manifest["control_indices"] == [227200, 227896]
            and manifest["last_failed_tick_original_gaussian_unavailable"] is True,
            "terminal10 fixture is not the original episode142 saved prefix")
    for name, expected in manifest["outputs"].items():
        require(identity(fixture / name) == expected,
                "terminal10 fixture output drifted: " + name)
    for name, expected in manifest["sources"].items():
        require(identity(Path(name)) == expected and frozen.get(name) == expected,
                "terminal10 original closed source drifted: " + name)
    return session, {"frozen_source_count": len(frozen),
                     "go_sha256": session["go_sha256"],
                     "fixture_manifest_sha256": session["fixture_manifest_sha256"]}


def _counter_delta(before: dict, after: dict, controls: int,
                   *, compiler: int = 0) -> None:
    cb, ca = before["C_state"], after["C_state"]
    pb, pa = before["python"], after["python"]
    require(ca["construction_attempts"] - cb["construction_attempts"] == compiler
            and ca["construction_returns"] - cb["construction_returns"] == compiler
            and ca["control_attempts"] - cb["control_attempts"] == controls * 5
            and ca["control_returns"] - cb["control_returns"] == controls * 5
            and pa["control_attempted"] - pb["control_attempted"] == controls
            and pa["control_completed"] - pb["control_completed"] == controls
            and pa["native_attempted"] - pb["native_attempted"] == controls * 5
            and pa["native_returned"] - pb["native_returned"] == controls * 5
            and pa["clock_advanced_substeps"] - pb["clock_advanced_substeps"]
            == controls * 5,
            "terminal10 C/Python segment counts are not exact 5T")


def _native_segment(folder: Path, receipt: dict, controls: int,
                    expected: dict | None, start_index: int) -> tuple[dict, list[dict]]:
    segment = receipt["native_segment"]
    require(segment["mode"] == "train"
            and segment["native_returned"] == segment["native_attempted"] == 5 * controls
            and segment["partial_native_interval"] is False
            and segment["failure"] is None
            and len(segment["native_files"]) == len(segment["train_array_files"]) == 1,
            "terminal10 native segment lacks complete compact train arrays")
    array = _npz(folder / segment["train_array_files"][0])
    require(set(array) == set(NATIVE_KEYS)
            and all(values.shape[0] == controls * 5
                    and np.isfinite(values).all() for values in array.values()),
            "terminal10 actual 13 native arrays missing/invalid")
    if expected is not None:
        require(set(expected) == set(NATIVE_KEYS)
                and all(_same(array[key], expected[key]) for key in NATIVE_KEYS),
                "terminal10 replay does not bit-match all 3485 original native rows")
    with gzip.open(folder / segment["native_files"][0], "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    require(len(rows) == controls * 5, "terminal10 native summary row count differs")
    for index, row in enumerate(rows):
        require(row["native_index"] == start_index + index
                and close(row["start_time_s"], array["start_time_s"][index], atol=1e-12)
                and close(row["end_time_s"], array["end_time_s"][index], atol=1e-12)
                and close(row["end_time_s"] - row["start_time_s"], .002, atol=1e-12)
                and row["nonwheel_contact_count"] == 0,
                "terminal10 native clock/contact/index differs")
        roll, pitch = _roll_pitch_deg(array["qpos_after"][index])
        require(close(row["roll_deg"], roll, atol=1e-8)
                and close(row["pitch_deg"], pitch, atol=1e-8)
                and max(abs(roll), abs(pitch)) <= 45,
                "terminal10 saved native posture differs from qpos")
    return array, rows


def _state_links(folder: Path, native: dict, controls: int) -> dict:
    states = _npz(folder / "states.npz")
    require(set(states) == set(STATE_KEYS)
            and all(value.shape[0] == controls + 1 and np.isfinite(value).all()
                    for value in states.values())
            and states["observation"].shape == (controls + 1, 99),
            "terminal10 control states/99D observations missing")
    for tick in range(controls):
        for key in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
            # The controller writes ctrl after the saved pre-control snapshot
            # and before the first native step; its requested/applied chain is
            # verified in _controls, not equated to that earlier snapshot.
            require((key == "ctrl" or _same(states[key][tick],
                                            native[key + "_before"][5 * tick]))
                    and _same(states[key][tick + 1],
                              native[key + "_after"][5 * tick + 4]),
                    "terminal10 100Hz state/native link differs: " + key)
        require(close(states["time"][tick], native["start_time_s"][5 * tick], atol=1e-12)
                and close(states["time"][tick + 1],
                          native["end_time_s"][5 * tick + 4], atol=1e-12),
                "terminal10 state/native clock link differs")
    return states


def _controls(folder: Path, native: dict, states: dict, controls: int,
              fixture: dict | None) -> list[dict]:
    rows = json.loads((folder / "control_records.json").read_text())
    require(isinstance(rows, list) and len(rows) == controls,
            "terminal10 control records absent or wrong length")
    for tick, row in enumerate(rows):
        info = row["info"]
        require(row["tick"] == tick and np.isfinite(row["reward"])
                and len(row["native_actuator_traces"]) == 5
                and info["completed_control_intervals"] == tick + 1
                and info["native_interval_summary"]["native_returns"] == 5
                and info["native_interval_summary"]["nonwheel_contact_count"] == 0
                and _numeric_list_matches(row["policy_input_action"],
                                          info["policy_input_action"])
                and _numeric_list_matches(info["policy_input_action"],
                                          info["policy_clipped_action"]),
                "terminal10 control/native/action link differs")
        for substep, trace in enumerate(row["native_actuator_traces"]):
            index = 5 * tick + substep
            require(_numeric_list_matches(trace["applied_nm"],
                                          native["ctrl_after"][index])
                    and _numeric_list_matches(trace["delayed_nm"],
                                              trace["applied_nm"]),
                    "terminal10 actual five-substep actuator trace differs")
        if fixture is not None and tick < 696:
            require(_same(states["observation"][tick],
                          fixture["input_observation99"][tick])
                    and _numeric_list_matches(info["policy_input_action"],
                                              fixture["policy_env_input16"][tick])
                    and _numeric_list_matches(info["applied_action"],
                                              fixture["effective_action16"][tick])
                    and _numeric_list_matches(info["controller_record"]["calculation"]
                                              ["safe_torque_nm"],
                                              fixture["stage_safe_torque_nm"][tick])
                    and close(row["reward"], fixture["reward"][tick], atol=0),
                    "terminal10 first696 compact obs/action/torque/reward differs")
        elif fixture is not None:
            require(tick == 696 and row["terminated"] is True
                    and row["truncated"] is False
                    and info["terminal_reason"] == "fall_or_low_clearance"
                    and _numeric_list_matches(row["policy_input_action"],
                                              np.zeros(16, dtype=np.float64))
                    and _numeric_list_matches(info["applied_action"],
                                              np.zeros(16, dtype=np.float64))
                    and info["terminal_observation_provenance"]["fallback_used"] is True,
                    "terminal10 failed original tick did not return task-failure fallback")
        else:
            require(row["terminated"] is False and row["truncated"] is False
                    and _numeric_list_matches(row["policy_input_action"],
                                              np.zeros(16, dtype=np.float64)),
                    "terminal10 post-reset zero-only eight controls differed")
    return rows


def _terminal_truth(run: Path, rows: list[dict], states: dict, native: dict,
                    binding: dict, geometry: dict) -> dict:
    terminal = _json(run / "episode142_replay697/actual_terminal_fallback.json")
    info = rows[-1]["info"]
    provenance = info["terminal_observation_provenance"]
    require(terminal["provenance"] == provenance
            and provenance["fallback_used"] is True
            and provenance["observation_only_not_executable"] is True
            and provenance["nominal_source"]
            == "consumed_precontrol_baseline_for_nonexecutable_task_terminal"
            and provenance["source_tick"] == 696
            and provenance["terminal_tick"] == 697
            and provenance["task_failure_reason"] == "fall_or_low_clearance"
            and "nominal_roll_or_pitch_above_0p3_rad"
            in provenance["envelope_violations"]
            and provenance["additional_prepare_or_control_calls"] == 0
            and terminal["actual_task_failure_reason"] == "fall_or_low_clearance"
            and terminal["actual_post_metrics"] == info["metrics"]
            and _numeric_list_matches(terminal["post_observation99"],
                                      states["observation"][-1])
            and _numeric_list_matches(terminal["effective_action16"],
                                      info["applied_action"]),
            "terminal10 fallback provenance/post observation differs")
    calc = info["controller_record"]["calculation"]
    baseline = terminal["consumed_precontrol_baseline"]
    require(_numeric_list_matches(baseline["nominal_joint_target_rad"],
                                  calc["consumed_nominal_joint_target_rad"])
            and _numeric_list_matches(baseline["nominal_wheel_speed_rad_s"],
                                      calc["consumed_nominal_wheel_speed_rad_s"])
            and _numeric_list_matches(terminal["post_committed_wheel_integral_nm"],
                                      calc["wheel_integral_after_nm"]),
            "terminal10 consumed baseline/post-committed I lacks actual chain")
    qpos, qvel = native["qpos_after"][-1], native["qvel_after"][-1]
    com = _body_com_velocity(qpos, qvel, binding)
    height, normal, geom_id = _ground_hit(
        geometry, float(qpos[0]), float(qpos[1]),
    )
    yaw = _yaw_from_qpos(qpos)
    reference_roll, reference_pitch = _terrain_reference(normal, yaw)
    roll, pitch = _roll_pitch_deg(qpos)
    metrics = info["metrics"]
    face_degrees = math.degrees(math.acos(float(np.clip(normal[2], -1, 1))))
    require(geom_id == metrics["ground_geom_id"] == 64
            and 81.0 <= face_degrees <= 83.0
            and close(height, metrics["ground_height_m"], atol=1e-8)
            and close(float(qpos[2] - height), metrics["clearance_m"], atol=1e-8)
            and close(com[0], metrics["body_com_vx_mps"], atol=1e-8)
            and close(com[1], metrics["body_com_vy_mps"], atol=1e-8)
            and close(float(qvel[5]), metrics["body_yaw_rate_rps"], atol=1e-8)
            and close(math.radians(roll) - reference_roll,
                      metrics["relative_roll_rad"], atol=1e-8)
            and close(math.radians(pitch) - reference_pitch,
                      metrics["relative_pitch_rad"], atol=1e-8)
            and abs(metrics["relative_pitch_rad"]) > math.radians(20)
            and max(abs(provenance["original_nominal_roll_rad"]),
                    abs(provenance["original_nominal_pitch_rad"])) > .3,
            "terminal10 true COM/endcap/relative-pitch failure was not reproduced")
    return {"actual_post_COM_vx_mps": float(com[0]),
            "actual_post_COM_vy_mps": float(com[1]),
            "actual_ground_geom_id": geom_id,
            "actual_ground_normal_world": normal.tolist(),
            "endcap_normal_tilt_deg": face_degrees,
            "actual_relative_pitch_deg": math.degrees(metrics["relative_pitch_rad"]),
            "post_observation99_finite": True,
            "consumed_baseline_postI_chain_exact": True,
            "fallback_observation_only": True}


def _closed_manifest(run: Path) -> dict:
    files = {}
    for path in sorted(run.rglob("*")):
        require(not path.is_symlink(), "terminal10 run contains symlink")
        if path.is_file():
            files[path.relative_to(run).as_posix()] = identity(path)
    return {"schema": "d1-terminal-probe-closed-file-manifest-v1",
            "run": str(run), "file_count": len(files),
            "total_bytes": sum(row["bytes"] for row in files.values()),
            "files": files}


def verify(run: Path) -> tuple[dict, dict]:
    require(run.is_absolute() and run.is_dir(), "terminal10 --run must be absolute")
    session, source = _verify_source(run)
    worker = _json(run / "worker_receipt.json")
    initial = _json(run / "runtime_initial.json")
    construction = _json(run / "construction_receipt.json")
    final = worker["C_Python_final"]
    c, p, guard = final["C_state"], final["python"], worker["native_guard"]
    require(initial["proof"]["passed"] is True
            and initial["proof"]["dso_path"] == session["library"]
            and initial["proof"]["dso_sha256"]
            == session["source_hashes"][session["library"]]["sha256"]
            and len(initial["proof"]["jump_slots"]) == 4
            and all(slot["passed"] is True for slot in initial["proof"]["jump_slots"])
            and construction["C_state"]["construction_attempts"]
            == construction["C_state"]["construction_returns"] == 2
            and construction["nominal_cache"]["misses"] == 1,
            "terminal10 actual DSO/GOT/cold2 construction proof differs")
    # The probe records its compiled geometry but deliberately does not copy
    # the training run's model binding. The sealed diagnosis identifies that
    # original construction receipt; prove its hash and identical geometry
    # before using its compiled body/geom address map for offline recompute.
    diagnosis_path = W.parent / "astra_t_terminal_saved_state_diagnosis_09.json"
    diagnosis = _json(diagnosis_path)
    original_construction_path = W.parent / "rl16_training_run_01/construction_receipt.json"
    original_construction = _json(original_construction_path)
    require(str(diagnosis_path) in session["source_hashes"]
            and identity(original_construction_path)["sha256"]
            == diagnosis["source_hashes"][str(original_construction_path)]
            and original_construction["compiled_geometry"]
            == construction["compiled_geometry"],
            "terminal10 original compiled binding not linked to current course")
    binding = check_binding(original_construction["actual_geometry_binding"])
    mapping = {}
    rows = construction["compiled_geometry"]["world_collision_geoms"]
    require(len(rows) == 92 and construction["compiled_geometry"]["control_dt_s"] == .01
            and construction["compiled_geometry"]["native_dt_s"] == .002,
            "terminal10 true compiled92 course geometry differs")
    for row in rows:
        gid = row["geom_id"]
        require(gid not in mapping and row["body_id"] == 0
                and binding["geom_bodyid"][gid] == 0
                and close(binding["geom_size"][gid], row["size_m"]),
                "terminal10 ground geom does not bind actual compiled model")
        mapping[gid] = row
    require(c["construction_attempts"] == c["construction_returns"] == 2
            and c["control_attempts"] == c["control_returns"] == 3525
            and c["control_limit"] == 3600
            and c["phase"] == c["target_model"] == c["target_data"] == 0
            and c["violations"] == 0
            and c["native_construction_caller_verified"] is True
            and c["ccd_attempts"] == c["ccd_returns"]
            and (c["ccd_attempts"] == 0 or c["native_ccd_caller_verified"] is True)
            and c["first_control_step_caller_dladdr"]["offset"] == 777003
            and p["control_attempted"] == p["control_completed"] == 705
            and p["native_attempted"] == p["native_returned"] == 3525
            and p["clock_advanced_substeps"] == 3525
            and p["native_failed"] == p["forbidden_entries"] == 0
            and p["fatal_latched"] is False
            and guard["native_attempted"] == guard["native_returned"]
            == guard["native_checked"] == 3525
            and guard["failure"] is None
            and guard["nonwheel_ground_contacts"] == 0,
            "terminal10 final actual C/Python/guard budgets or native proof differ")
    fixture = Path(session["fixture_directory"])
    expected = _npz(fixture / "expected_native.npz")
    saved_controls = _npz(fixture / "completed_controls.npz")
    require(set(expected) == set(NATIVE_KEYS)
            and all(value.shape[0] == 3485 for value in expected.values())
            and all(value.shape[0] == 696 for value in saved_controls.values()),
            "terminal10 original fixture shape differs")
    folders = (run / "episode142_replay697", run / "post_terminal_reset8")
    receipts = [_json(folder / "segment_receipt.json") for folder in folders]
    require(len(worker["segments"]) == 2
            and [row["name"] for row in worker["segments"]]
            == ["episode142_replay697", "post_terminal_reset8"]
            and [row["completed_controls"] for row in worker["segments"]] == [697, 8]
            and all(receipt["failure"] is None for receipt in receipts),
            "terminal10 actual segment closure differs")
    before0 = _json(folders[0] / "boundary_before.json")
    after0 = receipts[0]["boundary_after"]
    before1 = _json(folders[1] / "boundary_before.json")
    after1 = receipts[1]["boundary_after"]
    _counter_delta(before0, after0, 697)
    _counter_delta(before1, after1, 8)
    # start_segment arms the C guard for the next episode, so phase and
    # target pointers legitimately differ at this boundary. All completed
    # control/native/compiler counters must remain unchanged.
    _counter_delta(after0, before1, 0)
    require(after1 == final
            and receipts[0]["bitwise_saved_native_rows_compared"] == 3485
            and receipts[1]["bitwise_saved_native_rows_compared"] == 0,
            "terminal10 source replay/reset boundary or native comparison differs")
    native0, _ = _native_segment(folders[0], receipts[0], 697, expected, 0)
    native1, _ = _native_segment(folders[1], receipts[1], 8, None, 3485)
    states0, states1 = (_state_links(folder, native, n)
                        for folder, native, n in ((folders[0], native0, 697),
                                                  (folders[1], native1, 8)))
    records0 = _controls(folders[0], native0, states0, 697, saved_controls)
    records1 = _controls(folders[1], native1, states1, 8, None)
    reset0 = _json(folders[0] / "reset_receipt.json")
    reset1 = _json(folders[1] / "reset_receipt.json")
    require(reset0["model_address"] == reset1["model_address"]
            == construction["model_address"]
            and reset0["data_address"] == reset1["data_address"]
            == construction["data_address"]
            and reset0["seed"] == reset1["seed"] == 1322591124
            and reset0["episode_metadata"]["episode_index"] == 0
            and reset1["episode_metadata"]["episode_index"] == 1
            and reset0["episode_metadata"]["max_steps"]
            == reset1["episode_metadata"]["max_steps"] == 1600,
            "terminal10 reset changed model/data, seed or original episode horizon")
    for reset in (reset0, reset1):
        _counter_delta(reset["before"], reset["after"], 0)
    require(reset0["before"] == before0 and reset1["before"] == before1
            and _same(states0["qpos"][0], expected["qpos_before"][0]),
            "terminal10 reset source or first native initial state differs")
    truth = _terminal_truth(run, records0, states0, native0, binding, mapping)
    require(len(records1) == 8 and not (run / "final_checkpoint").exists()
            and not (run / "heldout").exists(),
            "terminal10 invented checkpoint/heldout or missed reset eight controls")
    result = {
        "schema": "d1-terminal-probe-independent-saved-readback-v1",
        "run": str(run), "source_closure": source,
        "offline_reader_source": {"path": str(Path(__file__).resolve()),
                                  **identity(Path(__file__).resolve())},
        "execution_complete": True,
        "main_replay_controls": 697,
        "main_native_all_13_arrays_bitwise_equal_to_original": True,
        "main_native_rows_bitwise_compared": 3485,
        "first_696_saved_observation_action_torque_reward_equal": True,
        "failed_tick_returned_task_failure_terminal": True,
        "terminal_truth": truth,
        "reset_same_model_data_and_seed": True,
        "post_terminal_zero_controls": 8,
        "post_terminal_native_returns": 40,
        "normal_native_returned": 3525,
        "compiler_native_returned": 2,
        "new_training_or_policy_execution": False,
        "baseline_ramp_endcap_semantics_fixed": False,
        "ramp_task_qualified": False,
        "final_RL_model_available": False,
        "engine_or_policy_imported": False,
        "physics_replayed_by_verifier": False,
    }
    return result, _closed_manifest(run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    require(args.run.is_absolute(), "terminal10 --run must be absolute")
    result, manifest = verify(args.run.resolve(strict=True))
    for target in (args.output, args.manifest_output):
        if target is not None:
            require(target.is_absolute() and not target.exists(),
                    "terminal10 offline output must be new and exclusive")
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    if args.manifest_output is not None:
        with args.manifest_output.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
