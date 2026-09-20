"""Fixed two-episode runner for the D1 stationary height-profile hop readiness
diagnostic: hold then profile, 600 executed ticks each, seed 55101.

The runner owns source identity, per-tick command verification against the fixed
pure schedule, the native observer, geometry reconstruction, scoring, the paired
prefix proof and all archiving.  It never calls the G1 episode entries and never
weakens their command verification: the expected command is re-derived here from
the frozen schedule and compared field by field against what actually reached the
controller on that tick.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from scripts.d1_jump_readiness import (
    TERMINAL_PREPARED_TICK,
    bind_wheel_plane_geometry,
    fixed_height_command,
    sample_wheel_clearance,
)
from scripts.d1_jump_readiness_env import READINESS_PROBE_SCHEMA, JumpReadinessEnv
from scripts.d1_jump_readiness_records import (
    READINESS_SCORE_SCHEMA,
    NativeStepObserver,
    ScratchKinematics,
    per_wheel_contact_detail,
    score_readiness,
)
from scripts.d1_native_contact_diagnostics import sample_native_contacts
from wheel_legged_control.d1.model import (
    JOINT_POSITION_HIGH,
    JOINT_POSITION_LOW,
    JOINT_TORQUE_LIMIT,
    JOINT_VELOCITY_LIMIT,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 55101
CONTROL_DT_S = 0.01
NATIVE_DT_S = 0.002
SUBSTEPS_PER_TICK = 5
TICKS = 600
MAX_CONTROL_INTERVALS = 1200
MAX_NATIVE_SUBSTEPS = 6000
CONDITIONS = (("stationary_height_hold", "hold"), ("stationary_height_profile", "profile"))
PAIRED_SLICES = {"execution": 200, "states": 201, "observations": 200, "native": 1000}
EXCLUDED_PAIR_FIELDS = ("condition", "display_phase", "next_prepared_clearance_m",
                        "next_prepared_phase")
_TOL_M = 1.0e-12
_TOL_S = 1.0e-9

SOURCE_FILES = (
    "scripts/d1_jump_readiness.py",
    "scripts/d1_jump_readiness_env.py",
    "scripts/d1_jump_readiness_records.py",
    "scripts/probe_d1_jump_readiness.py",
    "tests/test_d1_jump_readiness_env.py",
    "tests/test_d1_jump_readiness_records.py",
    "tests/test_d1_jump_readiness_runner.py",
    "jump_obstacle_next_plan_01/next_contract.md",
    "jump_obstacle_next_plan_01/proposed_cases.json",
    "opus_jump_readiness_01/original_module.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(root: Path = ROOT) -> dict[str, Any]:
    found, missing = {}, []
    for name in SOURCE_FILES:
        target = root / name
        if target.is_file():
            found[name] = sha256(target)
        else:
            missing.append(name)
    return {"files": found, "missing": missing}


def write_json(path: Path, value: Any) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def tagged(value: Any) -> Any:
    """Encode non-finite numbers with an explicit tag instead of dropping them."""
    if isinstance(value, dict):
        return {k: tagged(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tagged(v) for v in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            return {"nonfinite": "nan"}
        if number in (float("inf"), float("-inf")):
            return {"nonfinite": "inf" if number > 0 else "-inf"}
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return tagged(value.tolist())
    return value


def expected_schedule(condition: str) -> list[dict[str, Any]]:
    """Pure per-tick expectation, including the never-executed prepared tick 600."""
    rows = []
    for tick in range(TERMINAL_PREPARED_TICK + 1):
        command, label = fixed_height_command(tick, condition)
        rows.append({"tick": tick, "clearance_m": float(command.clearance_m),
                     "display_phase": label, "executed": tick < TICKS,
                     "forward_mps": float(command.forward_velocity_mps),
                     "yaw_rps": float(command.yaw_rate_rps)})
    return rows


def verify_tick_command(condition: str, tick: int, row: dict[str, Any]) -> dict[str, Any]:
    """Field-by-field check of one executed tick against the fixed schedule."""
    command, label = fixed_height_command(tick, condition)
    height = float(command.clearance_m)
    action = np.asarray(row.get("action", []), dtype=np.float64)
    world = float(row.get("world_command_base_height_m", float("nan")))
    ground = float(row.get("ground_height_m", float("nan")))
    checks = {
        "raw_clearance": abs(float(row.get("raw_clearance_m", float("nan"))) - height) <= _TOL_M,
        "servo_clearance": abs(
            float(row.get("servo_clearance_m", float("nan"))) - height) <= _TOL_M,
        "world_height_over_ground": abs((world - ground) - height) <= _TOL_M,
        "raw_forward_zero": float(row.get("raw_forward_mps", 1.0)) == 0.0,
        "raw_yaw_zero": float(row.get("raw_yaw_rps", 1.0)) == 0.0,
        "display_phase": row.get("display_phase") == label,
        "control_time": abs(
            float(row.get("control_time_s", float("nan"))) - tick * CONTROL_DT_S) <= _TOL_S,
        "zero_action": action.shape == (8,) and bool(np.all(action == 0.0)),
        "known_wrench": row.get("unknown_wrench") is False,
        "attitude_matches_truth": bool(
            abs(float(row.get("quaternion_attitude_error_rad", float("nan")))) <= 1.0e-6),
        "stop_latch_inactive": row.get("stop_latch_active") is False,
        "authority_gate_inactive": row.get("authority_gate_active") is False,
    }
    return {"tick": tick, "condition": condition, "checks": checks,
            "passed": all(bool(v) for v in checks.values())}


def bitwise_equal(first: Any, second: Any) -> bool:
    left = np.asarray(first)
    right = np.asarray(second)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype.hasobject:
        raise TypeError("object-array pointer bytes are not record identity")
    return left.tobytes() == right.tobytes()


def paired_prefix(hold: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Bitwise prefix comparison of the two episodes' shared initial segment."""
    checks: dict[str, Any] = {}
    for group, count in PAIRED_SLICES.items():
        group_checks: dict[str, bool] = {}
        left, right = hold.get(group) or {}, profile.get(group) or {}
        keys = sorted(set(left) | set(right))
        for key in keys:
            if key in EXCLUDED_PAIR_FIELDS:
                continue
            first, second = left.get(key), right.get(key)
            if first is None or second is None:
                group_checks[key] = False
                continue
            group_checks[key] = (len(first) >= count and len(second) >= count
                                 and bitwise_equal(first[:count], second[:count]))
        checks[group] = {"count": count, "fields": group_checks,
                         "passed": bool(group_checks) and all(group_checks.values())}
    return {"schema": READINESS_PROBE_SCHEMA,
            "excluded_fields": list(EXCLUDED_PAIR_FIELDS),
            "excluded_observation_index": 200,
            "self_pairing_is_not_accepted": True,
            "groups": checks,
            "passed": all(entry["passed"] for entry in checks.values())}


def resolve_steps(*, analyze_only: bool) -> int:
    """Analysis mode forces zero steps; there is no hidden rollout."""
    return 0 if analyze_only else TICKS * len(CONDITIONS)


def execute_episode(env: Any, *, condition: str, ticks: int = TICKS,
                    action: Any = None) -> dict[str, Any]:
    """Run one fixed episode with full consumption accounting in ``finally``."""
    if action is None:
        action = np.zeros(8, dtype=np.float32)
    accounting: dict[str, Any] = {
        "condition": condition, "requested_ticks": int(ticks),
        "completed_control_intervals": 0, "verifications": [], "error": None,
        "terminated": False, "truncated": False,
    }
    try:
        env.reset(seed=SEED)
        for tick in range(int(ticks)):
            result = env.step(action)
            accounting["completed_control_intervals"] = tick + 1
            row = dict(getattr(env, "last_readiness_row", {}) or {})
            row.setdefault("action", np.asarray(action, dtype=np.float64).tolist())
            verified = verify_tick_command(condition, tick, row)
            accounting["verifications"].append(verified)
            if not verified["passed"]:
                raise RuntimeError(
                    f"tick {tick} command verification failed: {verified['checks']}")
            accounting["terminated"] = bool(result[2])
            accounting["truncated"] = bool(result[3])
            if result[2] or result[3]:
                break
    except BaseException as error:  # accounting must survive any failure
        accounting["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        accounting["height_callback_count"] = int(getattr(env, "height_callback_count", 0))
        accounting["parent_step_calls"] = int(getattr(env, "parent_step_calls", 0))
        accounting["executed_ticks"] = list(getattr(env, "executed_ticks", []))
        accounting["terminal_prepared_tick_executed"] = bool(
            TERMINAL_PREPARED_TICK in accounting["executed_ticks"])
    return accounting


def budget_receipt(episodes: list[dict[str, Any]], native_calls: int) -> dict[str, Any]:
    intervals = sum(int(e.get("completed_control_intervals", 0)) for e in episodes)
    return {
        "schema": READINESS_PROBE_SCHEMA,
        "completed_control_intervals": intervals,
        "actual_native_calls": int(native_calls),
        "maximum_new_control_intervals": MAX_CONTROL_INTERVALS,
        "maximum_new_native_substeps": MAX_NATIVE_SUBSTEPS,
        "within_budget": intervals <= MAX_CONTROL_INTERVALS
        and int(native_calls) <= MAX_NATIVE_SUBSTEPS,
        "partial_episode_consumption_preserved": True,
    }


def write_manifest(directory: Path) -> Path:
    entries = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            entries[str(path.relative_to(directory))] = {
                "bytes": path.stat().st_size, "sha256": sha256(path)}
    target = Path(directory) / "manifest.json"
    write_json(target, {"schema": READINESS_PROBE_SCHEMA, "files": entries})
    return target


def analyze_stored_records(folder: Path) -> dict[str, Any]:
    """Recompute the score from stored records only; zero steps are executed."""
    payload = json.loads((Path(folder) / "readiness_records.json").read_text())
    score = score_readiness(payload)
    return {"schema": READINESS_SCORE_SCHEMA, "steps": resolve_steps(analyze_only=True),
            "score": score}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--cases", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.analyze_only:
        result = analyze_stored_records(args.output)
        write_json(args.output / "readiness_score_reanalysis.json", tagged(result))
        return 0
    if not all((args.preflight, args.contract, args.cases)):
        raise ValueError("physical execution requires preflight, contract and cases")
    preflight = validate_preflight(args.preflight, args.contract, args.cases)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    write_json(args.output / "preflight.json", preflight)
    outputs = {}
    for name, condition in CONDITIONS:
        folder = args.output / name
        folder.mkdir()
        env = JumpReadinessEnv(diagnostic_output=folder, condition=condition)
        outputs[name] = collect_episode(env, folder, seed=SEED, condition=condition)
        if outputs[name]["error"] is not None:
            break
    payload = {"episodes": {name: data["episode"] for name, data in outputs.items()}}
    write_json(args.output / "readiness_records.json", tagged(payload))
    score = score_readiness(payload)
    write_json(args.output / "score.json", score)
    pair = archive_prefix_check(outputs)
    write_json(args.output / "paired_prefix.json", pair)
    receipt = budget_receipt(
        [data["episode"] for data in outputs.values()],
        sum(data["native_receipt"]["attempted_native_calls"] for data in outputs.values()))
    receipt["returned_native_calls"] = sum(
        data["native_receipt"]["returned_native_calls"] for data in outputs.values())
    receipt["source_unchanged"] = check_hashes(preflight["input_sha256"])
    receipt["frozen77_unchanged"] = check_frozen()
    receipt["errors"] = {name: data["error"] for name, data in outputs.items()}
    receipt["records_valid"] = (
        len(outputs) == 2 and all(data["records_valid"] for data in outputs.values())
        and pair["passed"] and receipt["source_unchanged"] and receipt["frozen77_unchanged"])
    write_json(args.output / "receipt.json", receipt)
    write_manifest(args.output)
    print(json.dumps({"receipt": receipt, "score": score}, indent=2))
    return 0 if receipt["records_valid"] else 1


def check_hashes(expected):
    return all(Path(name).is_file() and sha256(Path(name)) == digest
               for name, digest in expected.items())


def check_frozen():
    expected = json.loads((ROOT / "results/d1_budget_study/protocol.json").read_text())["source_sha256"]
    return len(expected) == 77 and check_hashes({str(ROOT / name): digest
                                               for name, digest in expected.items()})


def validate_preflight(path, contract, cases):
    data = json.loads(path.read_text())
    for field in ("passed", "root_execution_authorized", "frozen77_unchanged"):
        if data.get(field) is not True:
            raise ValueError(f"preflight {field} must be true")
    if data.get("new_budget_control") != 1200 or data.get("new_budget_native_substeps") != 6000:
        raise ValueError("readiness budget must be 1200/6000")
    required = [Path(__file__).resolve(), contract.resolve(), cases.resolve()]
    required += [ROOT / "scripts" / name for name in (
        "d1_jump_readiness.py", "d1_jump_readiness_records.py", "d1_jump_readiness_env.py")]
    hashes = data.get("input_sha256", {})
    if not all(str(file) in hashes for file in required):
        raise ValueError("preflight omits an executable input")
    if not check_hashes(hashes) or not check_frozen():
        raise ValueError("preflight inputs or frozen77 changed")
    return data


def write_rows(path, rows):
    with gzip.open(path, "xt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(tagged(row), allow_nan=False, separators=(",", ":")) + "\n")


def quaternion_rpy(q):
    w, x, y, z = np.asarray(q)
    return np.array([np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                     np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
                     np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))])


def collect_episode(env, folder, *, seed, condition=None, action_function=None):
    """Archive a single six-second episode; exceptions retain consumed work, never retry."""
    plant = env.plant
    binding = bind_wheel_plane_geometry(plant.model, tuple(plant.wheel_body_ids_by_leg),
                                       plane_geom_id=plant.floor_geom_id)
    body_ids = list(range(1, plant.model.nbody))
    scratch = ScratchKinematics(plant.model, body_ids=body_ids,
                               masses=plant.model.body_mass[body_ids],
                               base_body_id=plant.base_body_id)
    def contacts(actual):
        return {"detail": per_wheel_contact_detail(actual.model, actual.data, binding,
                                                  force_reader=mujoco.mj_contactForce),
                "native": sample_native_contacts(actual)}
    observer = NativeStepObserver(mujoco, plant.model, plant.data,
                                  contact_reader=contacts, plant=plant)
    arrays = {key: [] for key in ("qpos", "qvel", "qacc_warmstart", "ctrl", "time", "observation")}
    trace, endpoints, intervals = [], [], []
    origin, initial_yaw = None, None
    error, terminated, truncated = None, False, False
    flags = {key: False for key in ("unknown_wrench", "nonwheel_contact", "fall", "domain_exit",
                                    "nonfinite_state", "rated_torque_exceeded",
                                    "position_envelope_exceeded", "velocity_envelope_exceeded")}
    flags.update(reset_count=0, plane_normals_valid=True, protections_intact=True)

    def snapshot(observation, truth, tick):
        nonlocal origin, initial_yaw
        for key in ("qpos", "qvel", "qacc_warmstart", "ctrl"):
            arrays[key].append(np.array(getattr(plant.data, key), copy=True))
        arrays["time"].append(float(plant.data.time))
        arrays["observation"].append(np.array(observation, copy=True))
        state = scratch.reconstruct(plant.data.qpos, plant.data.qvel,
                                    label="returned_control_endpoint", time_s=plant.data.time)
        geometry = sample_wheel_clearance(binding, state["geom_xpos"], state["geom_xmat"])
        rpy = quaternion_rpy(plant.data.qpos[3:7])
        if not np.allclose(rpy, truth.base_rpy, rtol=0., atol=1e-12):
            raise RuntimeError("independent endpoint quaternion disagrees with truth")
        if origin is None:
            origin = np.array(plant.data.qpos[:2], copy=True)
            initial_yaw = float(rpy[2])
        context = env.heading_decision
        endpoints.append({"tick": tick, "time_s": float(plant.data.time),
            "roll_rad": float(rpy[0]), "pitch_rad": float(rpy[1]),
            "heading_error_rad": float(np.arctan2(np.sin(rpy[2]-initial_yaw), np.cos(rpy[2]-initial_yaw))),
            "planar_displacement_m": float(np.linalg.norm(plant.data.qpos[:2]-origin)),
            "planar_position_m": plant.data.qpos[:2].tolist(),
            "achieved_clearance_m": float(plant.data.qpos[2]),
            "commanded_clearance_m": float(context.user_command.clearance_m),
            "body_vx_mps": float(truth.base_linear_velocity_body[0]),
            "com_vz_mps": state["com_velocity_mps"][2],
            "com_position_m": state["com_position_m"],
            "com_velocity_mps": state["com_velocity_mps"],
            "min_gap_m": geometry["simultaneous_minimum_gap_m"],
            "wheel_gap_m": geometry["wheel_bottom_gap_m"],
            "contact_margin_m": geometry["contact_margin_m"]})

    try:
        with observer:
            obs, reset_info = env.reset(seed=seed)
            flags["reset_count"] += 1
            write_json(folder / "episode_metadata.json", tagged(reset_info))
            snapshot(obs, env.heading_decision.decision.context.state, 0)
            for tick in range(TICKS):
                before = env.heading_decision
                action = (np.zeros(8, dtype=np.float32) if action_function is None
                          else np.asarray(action_function(obs), dtype=np.float32))
                obs, reward, terminated, truncated, info = env.step(action)
                record = env._controller.controller.last_result
                row = {"tick": tick, "action": action, "reward": reward,
                       "terminated": terminated, "truncated": truncated, "info": info,
                       "raw_command": asdict(before.user_command),
                       "servo_command": asdict(before.servo_command),
                       "world_command": asdict(before.decision.world_command),
                       "controller": asdict(record)}
                trace.append(row)
                if condition is not None:
                    verification = verify_tick_command(condition, tick, env.last_readiness_row)
                    row["readiness"] = dict(env.last_readiness_row)
                    if not verification["passed"]:
                        raise RuntimeError(f"raw schedule verification failed at tick {tick}")
                actual_position = np.asarray(record.joint_target_rad)
                mask = np.isfinite(JOINT_POSITION_LOW)
                flags["position_envelope_exceeded"] |= bool(
                    np.any(actual_position[mask] < JOINT_POSITION_LOW[mask]-1e-12)
                    or np.any(actual_position[mask] > JOINT_POSITION_HIGH[mask]+1e-12))
                delta = actual_position - before.decision.context.state.joint_position
                flags["velocity_envelope_exceeded"] |= bool(np.any(
                    np.abs(delta) > JOINT_VELOCITY_LIMIT * CONTROL_DT_S + 1e-12)
                    or np.any(np.abs(record.wheel_speed_target_rad_s) > 30.+1e-12))
                flags["rated_torque_exceeded"] |= bool(np.any(
                    np.abs(record.torque_nm) > JOINT_TORQUE_LIMIT + 1e-12))
                flags["fall"] |= info.get("terminal_reason") == "fall_or_body_contact"
                flags["domain_exit"] |= info.get("terminal_reason") in (
                    "map_boundary", "ground_query_outside_map", "control_reference_out_of_envelope")
                snapshot(obs, env.last_transition.truth, tick+1)
                if terminated or truncated:
                    break
    except BaseException as exc:  # noqa: BLE001 -- persist partial physical work; caller stops the batch
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        env.close()
        # The raw native archive is committed before reconstruction, preserving even a failed call.
        write_rows(folder / "native.jsonl.gz", observer.entries)
        write_rows(folder / "trace.jsonl.gz", trace)
        np.savez_compressed(folder / "states.npz", **{key: np.asarray(value) for key, value in arrays.items()})

    try:
        for native in observer.entries:
            if native["returned"] is not True:
                continue
            detail = native["contacts"]["detail"]
            cache = native["contacts"]["native"]
            start = scratch.reconstruct(native["qpos_before"], native["qvel_before"],
                                         label="native_input", time_s=native["start_time_s"])
            returned = scratch.reconstruct(native["qpos_returned"], native["qvel_returned"],
                                            label="native_returned", time_s=native["end_time_s"])
            gap = sample_wheel_clearance(binding, returned["geom_xpos"], returned["geom_xmat"])
            intervals.append({key: native[key] for key in (
                "index", "returned", "start_time_s", "end_time_s", "actual_dt_s")})
            intervals[-1].update(active_wheel_contacts=detail["active_contact_count"],
                geometric_wheel_contacts=detail["geometric_contact_count"],
                wheel_normal_load_n=detail["wheel_normal_load_n"],
                invalid_load=detail["invalid_load"], invalid_normal_values=detail["invalid_normal_values"],
                endpoint_min_gap_m=gap["simultaneous_minimum_gap_m"],
                endpoint_wheel_gap_m=gap["wheel_bottom_gap_m"], contact_margin_m=gap["contact_margin_m"],
                start_com_vz_mps=start["com_velocity_mps"][2],
                returned_com_velocity_mps=returned["com_velocity_mps"],
                reconstruction_start_sha256=start["input_sha256"],
                reconstruction_returned_sha256=returned["input_sha256"])
            flags["unknown_wrench"] |= bool(np.any(native["xfrc_applied"])
                                             or np.any(native["qfrc_applied"]))
            flags["nonwheel_contact"] |= cache["active_nonwheel_contacts"] > 0
            flags["plane_normals_valid"] &= (cache["max_horizontal_normal"] <= 1e-12
                                              and cache["max_vertical_normal_error"] <= 1e-12)
            flags["rated_torque_exceeded"] |= bool(np.any(np.abs(native["ctrl_nm"]) > JOINT_TORQUE_LIMIT+1e-12))
            flags["nonfinite_state"] |= not all(np.isfinite(native[key]).all() for key in (
                "qpos_before", "qvel_before", "qpos_returned", "qvel_returned", "ctrl_nm"))
    except BaseException as exc:  # noqa: BLE001 -- report failed reconstruction without retrying physics
        error = error or {"type": type(exc).__name__, "message": str(exc), "phase": "reconstruction"}
    completed = int(env._steps)
    episode = {"completed_control_intervals": completed, "execution": flags,
               "endpoints": endpoints, "intervals": intervals}
    native_receipt = observer.receipt()
    records_valid = (error is None and completed == 600 and len(trace) == 600
                     and len(endpoints) == 601 and len(intervals) == 3000
                     and native_receipt["attempted_native_calls"] == 3000
                     and not flags["nonfinite_state"] and not flags["unknown_wrench"]
                     and flags["plane_normals_valid"]
                     and all(not row["invalid_load"] for row in intervals))
    receipt = {"error": error, "records_valid": records_valid, "completed_control_intervals": completed,
               "native_receipt": native_receipt, "reconstruction": scratch.receipt(),
               "terminated": bool(terminated), "truncated": bool(truncated),
               "height_callback_count": getattr(env, "height_callback_count", None),
               "binding": binding.as_receipt()}
    write_json(folder / "receipt.json", tagged(receipt))
    write_json(folder / "records.json", tagged(episode))
    return {**receipt, "episode": episode, "arrays": arrays, "trace": trace,
            "native": observer.entries}


def archive_prefix_check(outputs):
    if len(outputs) != 2:
        return {"passed": False, "reason": "two actual episode archives required"}
    hold, profile = [outputs[name] for name, _ in CONDITIONS]
    checks = {}
    for key in hold["arrays"]:
        count = 200 if key == "observation" else 201
        checks[key] = bitwise_equal(hold["arrays"][key][:count], profile["arrays"][key][:count])
    def digest_trace(row):
        row = tagged(row)
        if "readiness" in row:
            row["readiness"].pop("condition", None)
            row["readiness"].pop("display_phase", None)
        return json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":"))
    checks["executed_R0_to_R199"] = all(
        digest_trace(a) == digest_trace(b) for a, b in zip(hold["trace"][:200], profile["trace"][:200], strict=True))
    checks["native_first1000"] = all(
        json.dumps(tagged(a), sort_keys=True) == json.dumps(tagged(b), sort_keys=True)
        for a, b in zip(hold["native"][:1000], profile["native"][:1000], strict=True))
    return {"passed": all(checks.values()), "checks": checks,
            "observation_200_excluded": "profile request first visible in this prepared decision",
            "excluded_fields": ["trace.readiness.condition", "trace.readiness.display_phase"]}


if __name__ == "__main__":  # pragma: no cover - root-owned execution entry
    raise SystemExit(main())
