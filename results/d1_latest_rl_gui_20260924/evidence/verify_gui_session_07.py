"""Independent saved-record readback for one bounded 07 validation run.

Only stdlib, NumPy, and the published pure validators are imported. No engine,
policy, scene creation, or physical rescore is performed. Missing evidence fails.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.abc
import json
import math
import os
import sys
from pathlib import Path


class NoPhysics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in {"mujoco", "torch", "stable_baselines3", "gymnasium"}:
            raise RuntimeError("forbidden readback import: " + fullname)


if any(name.partition(".")[0] in {"mujoco", "torch", "stable_baselines3", "gymnasium"}
       for name in sys.modules):
    raise RuntimeError("physics/policy module already imported")
sys.meta_path.insert(0, NoPhysics())
sys.dont_write_bytecode = True
import numpy as np

REPO = Path(__file__).resolve().parents[3] / "wheel-legged-control-lab"
BODY = REPO / "results/d1_body_speed_feedback_20260924/source"
sys.path[:0] = [str(BODY), str(REPO)]
from body_speed_validator_06 import (
    validate_body_precontrol_archive,
    validate_body_trace,
)
from scripts.d1_single_step_integrity import validate_record_links

MODEL_SHA = "4f59d795afbdec256ec17bb7256ea05a3ddd04561e9ba52beffb1de375196b2e"
METADATA_SHA = "44b199ea13989412d274955a9fd85d6e97e58ad708a0c7e3ab7a4085ae3daaf3"
PROBE_SHA = "2cb674f54c8c46d5555e5d3580810a003be8d50fa5a7c1a5526dd0f027bb6660"
PAIR = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")


def require(ok: bool, label: str) -> None:
    if not ok:
        raise ValueError(label)


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def digest(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


def same(a, b) -> bool:
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def check_sources(session: dict) -> int:
    frozen = session["source_hashes"]
    require(isinstance(frozen, dict) and len(frozen) >= 100, "source closure absent")
    for name, expected in frozen.items():
        path = Path(name)
        require(path.is_absolute() and path.is_file() and digest(path) == expected,
                f"source hash differs: {name}")
    origins = read(Path(session["output_directory"]) / "module_origins.json")
    actual = origins["published_numerical_and_binding_sources"]
    require(len(actual) >= 9 and all(path in frozen for path in actual.values()),
            "actual imported module origins missing from frozen source closure")
    return len(frozen)


def check_segment(run: Path, number: int, session: dict) -> dict:
    folder = run / f"segment_{number:02d}"
    limit = (1000, 200)[number]
    before, after = read(folder / "boundary_before.json"), read(folder / "boundary_after.json")
    reset = read(folder / "reset_pair_receipt.json")
    episode = read(folder / "episode_metadata.json")
    manifest = read(folder / "geometry_manifest.json")
    require(episode["task_schema"] == "d1-latest-rl-straight-interactive-v1"
            and episode["interactive_actor"] == session["actor"]
            and episode["interactive_input_origin"] == "scripted_acceptance"
            and episode["rolling_task_definition"]["physical_terrain"] == session["terrain"],
            f"segment {number}: interactive task metadata differs")
    trace, endpoints, native, entries, polls = (rows(folder / name) for name in
        ("trace.jsonl.gz", "endpoints.jsonl.gz", "native.jsonl.gz",
         "native_entry_events.jsonl.gz", "keyboard_polls.jsonl.gz"))
    precontrol = json.loads((folder / "pre_control_states.json").read_text(encoding="utf-8"))
    commands = json.loads((folder / "command_records.json").read_text(encoding="utf-8"))
    require(len(trace) == limit and len(endpoints) == limit + 1
            and len(precontrol) == limit and len(native) == 5 * limit
            and len(entries) == len(native) and len(polls) == limit + 1,
            f"segment {number}: full records absent")
    require([row["tick"] for row in polls] == list(range(limit + 1))
            and all(row["origin"] == "scripted_acceptance" for row in polls),
            f"segment {number}: scripted keyboard polling evidence differs")
    require(before["segment"] == after["segment"] == f"segment_{number:02d}"
            and after["completed_controls"] == limit
            and after["recorded_control_rows"] == limit
            and after["error"] is None, f"segment {number}: incomplete boundary")
    expected_stop = ("operator_simulation_reset", "keyboard_escape")[number]
    require(after["stop_reason"] == expected_stop, f"segment {number}: stop reason differs")
    require(reset["exact_initial_pair"] is True and reset["segment"] == f"segment_{number:02d}"
            and reset["seed"] == 77351
            and reset["control_native_compiler_budget_unchanged"] is True,
            f"segment {number}: initial/reset evidence differs")
    for key in ("construction_attempts", "construction_returns", "control_attempts",
                "control_returns"):
        require(reset["C_before_reset"][key] == reset["C_after_reset"][key],
                f"segment {number}: reset changed C budget {key}")
    for key in ("native_attempted", "native_returned", "control_attempted",
                "control_completed"):
        require(reset["python_before_reset"][key] == reset["python_after_reset"][key],
                f"segment {number}: reset changed Python budget {key}")
    schedule = [0.0] * limit
    if number == 0:
        schedule[175:401] = [.20] * 226
        schedule[401:801] = [.25] * 400
    else:
        schedule[175:200] = [.20] * 25
    for tick, row in enumerate(trace):
        raw = row["raw_command"]
        action = np.asarray(row["applied_action"], dtype=float)
        scalar = float(row["clipped_policy_scalar"])
        policy = np.asarray(row["policy_action"], dtype=float)
        expected_leg = .25 * scalar if schedule[tick] else 0.0
        require(row["tick"] == tick and row["endpoint_tick"] == tick + 1
                and row["checkpoint_sha256"] == MODEL_SHA
                and raw["tick"] == tick and raw["forward_velocity_mps"] == schedule[tick]
                and row["forward_gate_active"] is bool(schedule[tick])
                and policy.shape == (1,) and np.isfinite(policy).all()
                and -1 <= scalar <= 1 and scalar == float(np.clip(policy[0], -1, 1))
                and action.shape == (8,) and np.isfinite(action).all()
                and np.array_equal(action[:4], np.full(4, expected_leg))
                and np.array_equal(action[4:], np.zeros(4))
                and same(row["action"], action)
                and same(row["controller_result"]["clipped_action"], action)
                and same(row["torque_nm"], row["controller_result"]["torque_nm"]),
                f"segment {number}: raw/policy/applied action mismatch at {tick}")
        if session["actor"] == "zero":
            require(row["policy_predict_called"] is False and scalar == 0.0
                    and same(row["policy_action"], [0.0]),
                    f"segment {number}: zero actor produced a policy action at {tick}")
        else:
            require(row["policy_predict_called"] is True,
                    f"segment {number}: saved policy not called at {tick}")
    require([row["tick"] for row in endpoints] == list(range(limit + 1)),
            f"segment {number}: endpoint ticks differ")
    require(len(commands) == limit + 1 and
            [row["tick"] for row in commands] == list(range(limit + 1)),
            f"segment {number}: prepared callback count differs")
    for index, row in enumerate(native):
        require(row["index"] == index and row["returned"] is True and row["error"] is None
                and entries[index]["attempt"] == index
                and entries[index]["time_s"] == row["start_time_s"]
                and math.isclose(row["start_time_s"], index * .002, abs_tol=1e-12)
                and math.isclose(row["end_time_s"], (index + 1) * .002, abs_tol=1e-12)
                and math.isclose(row["actual_dt_s"], .002, abs_tol=1e-12)
                and same(row["ctrl_nm"], trace[index // 5]["torque_nm"]),
                f"segment {number}: native entry/return differs at {index}")
        if index:
            require(same(native[index - 1]["qpos_returned"], row["qpos_before"])
                    and same(native[index - 1]["qvel_returned"], row["qvel_before"]),
                    f"segment {number}: native chain broken at {index}")
    with np.load(folder / "states.npz", allow_pickle=False) as saved, np.load(
            folder / "initial_state.npz", allow_pickle=False) as initial:
        require(all(key in saved and key in initial for key in PAIR),
                f"segment {number}: initial/state arrays absent")
        require(all(saved[key].shape[0] == limit + 1 and same(saved[key][0], initial[key])
                    and np.isfinite(saved[key]).all() for key in PAIR),
                f"segment {number}: state/initial pair differs")
        require(all(hashlib.sha256(initial[key].tobytes()).hexdigest()
                    == reset["array_sha256"][key] for key in PAIR),
                f"segment {number}: initial raw-array receipt hash differs")
        require(saved["observation"].shape == (limit + 1, 85)
                and saved["observation"].dtype == np.float32,
                f"segment {number}: 85D float32 observation differs")
        for tick in range(limit):
            first, last = native[5 * tick], native[5 * tick + 4]
            require(same(first["qpos_before"], saved["qpos"][tick])
                    and same(first["qvel_before"], saved["qvel"][tick])
                    and same(last["qpos_returned"], saved["qpos"][tick + 1])
                    and same(last["qvel_returned"], saved["qvel"][tick + 1])
                    and same(saved["ctrl"][tick + 1], trace[tick]["torque_nm"]),
                    f"segment {number}: native/NPZ state link differs at {tick}")
            require(hashlib.sha256(np.asarray(saved["observation"][tick],
                                                dtype=np.float32).tobytes()).hexdigest()
                    == trace[tick]["observation_before_sha256"],
                    f"segment {number}: actor observation link differs at {tick}")
        pre = validate_body_precontrol_archive(
            precontrol, saved_qpos=saved["qpos"], saved_qvel=saved["qvel"],
            joint_qpos_addresses=manifest["actual_compiled_joint_qpos_addresses"],
            joint_dof_addresses=manifest["actual_compiled_joint_dof_addresses"],
            base_body_ipos_local_m=manifest["base_body_ipos_local_m"], endpoints=endpoints)
        initial_values = {key: np.asarray(initial[key]).tolist() for key in PAIR}
    links = validate_record_links({"geometry_manifest": manifest, "trace": trace,
                                   "endpoints": endpoints, "native": native})
    stages = validate_body_trace(trace, expected_raw_schedule=schedule,
                                 pre_control_states=precontrol, native_rows=native)
    require(stages["record_valid"] is True and pre["record_valid"] is True,
            f"segment {number}: pure stage/precontrol validation failed")
    del links
    for ep in endpoints:
        roll, pitch, yaw = np.rad2deg(ep["rpy_rad"])
        initial_yaw = float(np.rad2deg(endpoints[0]["rpy_rad"][2]))
        require(max(abs(roll), abs(pitch)) <= 10 and abs(yaw - initial_yaw) <= 5
                and abs(ep["base_position_m"][1] - endpoints[0]["base_position_m"][1]) <= .1
                and ep["nonwheel_terrain_contacts"] == 0,
                f"segment {number}: posture, drift, or nonwheel contact gate failed")
    box_load = sum(float(np.sum(row["contacts"]["wheel_box_positive_normal_load_n"]))
                   for row in native)
    require(all(row["contacts"]["nonwheel_terrain_contacts"] == 0 for row in native),
            f"segment {number}: native nonwheel contact")
    if number == 0 and session["terrain"] == "box":
        require(box_load > 0, "GUI box run lacks positive wheel-box load")
    interval = {}
    for target in (.20, .25):
        measured = np.asarray([endpoints[tick]["body_vx_mps"] for tick, raw in enumerate(schedule)
                               if raw == target], dtype=float)
        if len(measured):
            interval[str(target)] = {"controls": len(measured),
                "mean_body_vx_mps": float(measured.mean()),
                "rms_error_mps": float(np.sqrt(np.mean((measured - target) ** 2)))}
    return {"controls": limit, "native_returns": len(native), "box_positive_load_sum_n": box_load,
            "body_stage": stages, "precontrol": pre, "interval_descriptive_only": interval,
            "boundary_before": before, "boundary_after": after,
            "initial": initial_values,
            "reset": reset, "nonzero_leg_actions": sum(
                bool(np.any(np.asarray(row["applied_action"])[:4])) for row in trace)}


def check_render(run: Path, session: dict, segments: list[dict]) -> dict:
    if session["validation_profile"] != "gui_policy_box":
        require(not (run / "render_receipt.json").exists(), "headless run has GUI receipt")
        return {"headless": True}
    render = read(run / "render_receipt.json")
    frames, screenshots = render["frames"], render["screenshots"]
    require(render["rendered_frames"] == len(frames) and len(frames) > 0
            and render["no_render_step_forward_collision"] is True,
            "GUI render receipt differs")
    for frame in frames:
        require(frame["native_counter_delta"] == frame["python_counter_delta"] == 0
                and frame["live_integrator_unchanged"] is True
                and frame["measurement_unchanged"] is True,
                "render changed physical state/counters")
    by_id = {row["frame_id"]: row for row in screenshots}
    require({"drive_200", "drive_650", "after_reset"} <= set(by_id),
            "required distinct GUI screenshots absent")
    require(len({by_id[key]["source_tick"] for key in by_id}) >= 2,
            "screenshots do not show distinct simulation ticks")
    for name in ("drive_200", "drive_650", "after_reset"):
        row = by_id[name]
        path = Path(row["path"])
        require(path.is_file() and path.resolve().is_relative_to(run)
                and digest(path) == {"sha256": row["png_sha256"], "bytes": row["bytes"]}
                and row["bytes"] > 0 and any(
                    frame["source_tick"] == row["source_tick"]
                    and frame["source_time_s"] == row["source_time_s"] for frame in frames),
                f"GUI screenshot/frame mismatch: {name}")
    return {"rendered_frames": len(frames), "screenshots": len(screenshots)}


def verify(run: Path) -> dict:
    require(run.is_absolute() and run.is_dir(), "--run must be an absolute existing directory")
    session, worker, launcher = (read(run / name) for name in
        ("session.json", "worker_receipt.json", "launcher_receipt.json"))
    require(session["mode"] == "validation" and session["output_directory"] == str(run)
            and session["segments"] == [1000, 200]
            and (session["validation_profile"], session["actor"], session["terrain"])
            in {("gui_policy_box", "final_policy", "box"),
                ("headless_zero_plane", "zero", "plane")},
            "not a frozen 07 validation profile")
    require(launcher["exit_code"] == 0 and launcher["reason"] is None
            and launcher["source_hash_mismatches"] == []
            and worker["failure"] is None and worker["execution_complete"] is True
            and worker["warnings"] == [] and worker["completed_controls"] == 1200
            and worker["no_training_or_policy_update"] is True
            and worker["not_a_replay_of_fixed_06_qualification"] is True,
            "launcher/worker execution incomplete")
    driver = read(run / "validation_driver_receipt.json")
    require(driver["profile"] == session["validation_profile"]
            and driver["complete"] is True and driver["passed"] is True
            and driver["human_usability_assessed"] is False
            and driver["transport"] == (
                "owned-window XSendEvent" if session["validation_profile"] == "gui_policy_box"
                else "pure input callbacks"),
            "scripted acceptance driver receipt differs")
    reload = read(run / "strict_original_checkpoint_reload.json")
    transfer = read(run / "controller_law_transfer.json")
    require(reload["model_sha256"] == MODEL_SHA
            and reload["metadata_sha256"] == METADATA_SHA
            and reload["probe_sha256"] == PROBE_SHA
            and reload["old_schema_checked_before_transfer"] is True
            and reload["two_saved_probe_predictions_checked"] is True
            and reload["live_policy_predict_calls"] == 0
            and reload["native_return_delta"] == 0
            and transfer["same_C_state_after_loader_and_transfer"] is True
            and transfer["interactive_callback_installed_after_strict_loader"] is True,
            "strict original checkpoint/controller transfer evidence differs")
    initial_runtime = read(run / "runtime_initial.json")
    binding = initial_runtime["binding_proof"]
    initial_c = initial_runtime["initial_C_state"]
    require(binding["passed"] is True and len(binding["jump_slots"]) == 4
            and all(slot["passed"] is True for slot in binding["jump_slots"])
            and all(initial_c[key] == 0 for key in (
                "construction_attempts", "construction_returns", "control_attempts",
                "control_returns", "ccd_attempts", "ccd_returns", "violations"))
            and initial_runtime["before_model_construction"] is True,
            "initial four-slot GOT or zero C counters absent")
    source_count = check_sources(session)
    parts = [check_segment(run, i, session) for i in range(2)]
    first, second = parts
    require(all(same(first["initial"][key], second["initial"][key]) for key in PAIR)
            and first["reset"]["model_address"] == second["reset"]["model_address"]
            and first["reset"]["data_address"] == second["reset"]["data_address"]
            and first["boundary_after"]["same_model_address"] == second["boundary_before"]["same_model_address"]
            and first["boundary_after"]["same_data_address"] == second["boundary_before"]["same_data_address"],
            "simulation reset reconstructed the model/data or changed exact initial pair")
    require(worker["live_policy_predict_calls"] == (1200 if session["actor"] == "final_policy" else 0)
            and worker["live_policy_predict_returns"] == worker["live_policy_predict_calls"]
            and (sum(part["nonzero_leg_actions"] for part in parts) > 0
                 if session["actor"] == "final_policy" else
                 sum(part["nonzero_leg_actions"] for part in parts) == 0),
            "actual RL/zero action evidence differs")
    cstate, ledger, monitor = (worker[key] for key in
        ("C_final", "python_ledger_final", "native_monitor"))
    require(cstate["construction_attempts"] == cstate["construction_returns"] == 3
            and cstate["control_attempts"] == cstate["control_returns"] == 6000
            and cstate["ccd_attempts"] == cstate["ccd_returns"] >= 0
            and (cstate["native_ccd_caller_verified"] is True
                 if cstate["ccd_attempts"] else
                 cstate["first_ccd_caller"] == 0
                 and cstate["native_ccd_caller_verified"] is False)
            and (cstate["ccd_attempts"] > 0 if session["terrain"] == "box" else True)
            and cstate["native_construction_caller_verified"] is True
            and cstate["violations"] == 0
            and ledger["control_attempted"] == ledger["control_completed"] == 1200
            and ledger["native_attempted"] == ledger["native_returned"] == 6000
            and ledger["native_failed"] == ledger["forbidden_entries"] == 0
            and ledger["clock_advanced_substeps"] == 6000
            and monitor["checked_native_returns"] == monitor["passed_native_returns"] == 6000
            and monitor["failure"] is None
            and monitor["segments"]["segment_00"]["passed"] == 5000
            and monitor["segments"]["segment_01"]["passed"] == 1000,
            "final C/Python/native monitor counters differ")
    for index, part in enumerate(parts):
        before, after = part["boundary_before"], part["boundary_after"]
        native_limit = 5 * (1000, 200)[index]
        require(after["C_state"]["control_returns"] - before["C_state"]["control_returns"]
                == native_limit
                and after["C_state"]["control_attempts"] - before["C_state"]["control_attempts"]
                == native_limit
                and after["python_ledger"]["native_returned"]
                - before["python_ledger"]["native_returned"] == native_limit
                and after["python_ledger"]["control_completed"]
                - before["python_ledger"]["control_completed"] == (1000, 200)[index],
                f"segment {index}: cumulative native/control delta differs")
    render = check_render(run, session, parts)
    return {"schema": "d1-latest-rl-independent-readback-v1", "passed": True,
            "profile": session["validation_profile"], "actor": session["actor"],
            "terrain": session["terrain"], "controls": 1200, "normal_native": 6000,
            "compiler_native": 3, "frozen_source_files": source_count,
            "segments": [{key: part[key] for key in
                          ("controls", "native_returns", "box_positive_load_sum_n",
                           "interval_descriptive_only", "nonzero_leg_actions")}
                         for part in parts],
            "render": render, "old_constant_speed_window_rescored": False,
            "different_terrain_profiles_are_not_rl_causal_comparison": True,
            "engine_or_policy_imported": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require(args.run.is_absolute(), "--run must be absolute")
    result = verify(args.run.resolve(strict=True))
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        output = args.output
        require(output.is_absolute() and not output.exists(), "--output must be a new absolute file")
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
