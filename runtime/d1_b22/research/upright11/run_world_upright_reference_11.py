"""One bounded, zero-action world-upright reference run on the real course.

The launcher owns the process and the only physical execution. Importing this
module is stdlib-only; engine imports occur after the frozen-session preflight.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts import run_d1_latest_rl as common

HERE = Path(__file__).resolve().parent
SCHEMA = "d1-world-upright-reference-11-v1"
CONTRACT_SHA = "2f891044eac2c881f7240dc0da22a79ced22458c75f1df7ac6b103b7a01ae375"
CONTROLS = 1800
SPAWN = (2.75, 0.0, 0.455)
SEED = 88611


def _identity(path: Path) -> dict[str, Any]:
    return {"sha256": common._sha256(path), "bytes": path.stat().st_size}


def _boundary(runtime: Any) -> dict[str, Any]:
    return common._jsonable({"C_state": runtime.state(), "python": runtime.ledger_receipt()})


def _preflight(session: dict[str, Any], path: Path) -> Path:
    expected = {
        "schema": SCHEMA, "phase": "reference_baseline", "control_limit": CONTROLS,
        "normal_native_limit": CONTROLS * 5, "compiler_native_limit": 2,
        "wallclock_limit_s": 180, "segments": [CONTROLS], "seed": SEED,
        "contract_sha256": CONTRACT_SHA, "retry_permitted": False,
    }
    if any(session.get(key) != value for key, value in expected.items()):
        raise RuntimeError("world-upright reference identity or prepaid cap differs")
    if (common._absolute(session, "session_path") != path
            or session.get("argv") != sys.argv
            or sys.argv[1:] != ["--session", str(path)]
            or session.get("worker_argv", [])[-len(sys.argv):] != sys.argv
            or session.get("run_id") != "world_upright_reference_11_01"
            or type(session.get("wall_clock_utc")) is not str):
        raise RuntimeError("world-upright session/argv identity differs")
    output = common._absolute(session, "output_directory")
    if not output.is_dir():
        raise RuntimeError("world-upright output reservation absent")
    sources = session.get("source_hashes")
    if not isinstance(sources, dict) or len(sources) < 100:
        raise RuntimeError("world-upright frozen source closure absent")
    for name, expected_hash in sources.items():
        source = Path(name)
        if (not source.is_absolute() or not source.is_file() or source.is_symlink()
                or _identity(source) != expected_hash):
            raise RuntimeError("world-upright frozen input differs: " + name)
    required = (
        Path(__file__).resolve(), HERE / "world_upright_course_11.py",
        HERE / "next_reference_contract_11.md", HERE / "astra_reference_go_11.json",
        common._absolute(session, "library"),
        common._absolute(session, "binding_module"),
        common._absolute(session, "continuation_import_repair"),
    )
    if any(str(item) not in sources for item in required):
        raise RuntimeError("world-upright critical runtime source absent from freeze")
    if _identity(HERE / "next_reference_contract_11.md")["sha256"] != CONTRACT_SHA:
        raise RuntimeError("world-upright contract differs")
    go_path = HERE / "astra_reference_go_11.json"
    go = json.loads(go_path.read_text())
    if (_identity(go_path)["sha256"] != session["go_sha256"]
            or go.get("decision") != "GO"
            or go.get("contract_sha256") != CONTRACT_SHA):
        raise RuntimeError("world-upright independent SOURCE GO absent")
    for name, record in go["inputs"].items():
        if sources.get(name) != record:
            raise RuntimeError("world-upright GO/source freeze differs: " + name)
    if common._sha256(Path(session["library"])) != common.DSO_SHA256:
        raise RuntimeError("world-upright engine DSO differs")
    for key, value in session["runtime_environment"].items():
        if os.environ.get(key) != value:
            raise RuntimeError("world-upright child runtime environment differs: " + key)
    if ("LD_LIBRARY_PATH" in os.environ or os.environ.get("LD_BIND_NOW") != "1"
            or os.environ.get("LD_PRELOAD") != session["library"]):
        raise RuntimeError("world-upright engine environment not isolated")
    return output


def _schedule() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[float]]]:
    """Pure, fixed 1800-tick request/servo/path before any model construction."""
    from full_drive_command_08 import FullDriveCommand, QualifiedCommandCaps
    from full_drive_servo_08 import FullDriveCommandServo

    caps = QualifiedCommandCaps(1.6, 0.0, 0.0, 0.3, False)
    servo = FullDriveCommandServo(caps)
    raw, applied = [], []
    x, y = SPAWN[:2]
    path = [[x, y]]
    for tick in range(CONTROLS):
        vx = .4 if 175 <= tick < 1355 else 0.0
        request = FullDriveCommand(vx, 0.0, 0.0, .455, False)
        receipt = servo.advance(tick, request)
        raw.append(asdict(request))
        applied.append(asdict(receipt.applied))
        x += receipt.applied.forward_velocity_mps * .01
        path.append([x, y])
    if (len(raw) != CONTROLS or len(path) != CONTROLS + 1
            or not 7.0 < x < 8.0
            or any(item["lateral_velocity_mps"] or item["yaw_rate_rps"]
                   or item["jump_requested"] for item in raw)):
        raise RuntimeError("world-upright preregistered nominal schedule differs")
    return raw, applied, path


def _binding(plant: Any) -> dict[str, Any]:
    model = plant.model
    return {
        "base_body_id": int(plant.base_body_id),
        "base_body_ipos": model.body_ipos[plant.base_body_id].copy(),
        "wheel_index_by_body_id": dict(plant._wheel_index_by_body_id),
        "joint_ids": plant.joint_ids.copy(),
        "joint_qpos_addresses": plant.qpos_addresses.copy(),
        "joint_dof_addresses": plant.dof_addresses.copy(),
        "actuator_ids": plant.actuator_ids.copy(),
        "actuator_trntype": model.actuator_trntype.copy(),
        "actuator_trnid": model.actuator_trnid.copy(),
        "actuator_gear": model.actuator_gear.copy(),
        "body_mass": model.body_mass.copy(),
        "nominal_total_mass_kg": plant.nominal_total_mass_kg,
        "geom_bodyid": model.geom_bodyid.copy(),
        "geom_type": model.geom_type.copy(),
        "geom_size": model.geom_size.copy(),
        "geom_contype": model.geom_contype.copy(),
        "geom_conaffinity": model.geom_conaffinity.copy(),
    }


def _x_projection(plant: Any) -> dict[str, Any]:
    """Exact box and cylinder world-X support of compiled collision geoms."""
    import numpy as np

    model, data = plant.model, plant.measurement_data
    integration = plant.data
    if (not np.array_equal(data.qpos, integration.qpos)
            or not np.array_equal(data.qvel, integration.qvel)
            or float(data.time) != float(integration.time)):
        raise RuntimeError("post-measurement cache is not at the final integrated state")
    ramp = {}
    for row in plant.collision_terrain_metadata["world_collision_geoms"]:
        if row["name"] not in ("terrain_ramp_up", "terrain_ramp_deck",
                                "terrain_ramp_down"):
            continue
        gid = int(row["geom_id"])
        rotation = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
        half = np.asarray(model.geom_size[gid])
        extent = float(np.abs(rotation[0]) @ half)
        ramp[row["name"]] = {"geom_id": gid,
                              "world_x_max_m": float(data.geom_xpos[gid, 0] + extent),
                              "center_world_m": np.asarray(data.geom_xpos[gid]).tolist(),
                              "rotation_world_from_local": rotation.tolist(),
                              "compiled_half_size_m": half.tolist()}
    wheel = {}
    for gid in range(model.ngeom):
        body = int(model.geom_bodyid[gid])
        index = plant._wheel_index_by_body_id.get(body)
        if index is None or not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        # The four actual wheel collision geoms are cylinders. Their local
        # axis is Z; radial X support is radius times the world-X planar norm.
        if int(model.geom_type[gid]) != 5:
            raise RuntimeError("wheel collision primitive is not the compiled cylinder")
        rotation = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
        radius, half_width = map(float, model.geom_size[gid, :2])
        extent = (radius * math.hypot(float(rotation[0, 0]), float(rotation[0, 1]))
                  + half_width * abs(float(rotation[0, 2])))
        if index in wheel:
            raise RuntimeError("multiple wheel collision geoms on one wheel body")
        wheel[index] = {"geom_id": gid,
                        "world_x_min_m": float(data.geom_xpos[gid, 0] - extent),
                        "center_world_m": np.asarray(data.geom_xpos[gid]).tolist(),
                        "rotation_world_from_local": rotation.tolist(),
                        "compiled_radius_m": radius,
                        "compiled_half_width_m": half_width}
    if len(ramp) != 3 or set(wheel) != set(range(4)):
        raise RuntimeError("actual ramp/wheel compiled projection identities differ")
    far_edge = max(row["world_x_max_m"] for row in ramp.values())
    return {"cache_source": "plant.measurement_data_post_refresh_measurements",
            "measurement_and_integration_qpos_qvel_time_equal": True,
            "measurement_qpos": np.asarray(data.qpos).tolist(),
            "measurement_qvel": np.asarray(data.qvel).tolist(),
            "measurement_time_s": float(data.time),
            "ramp": ramp, "wheel": wheel,
            "all_wheel_collision_shapes_past_ramp":
            all(row["world_x_min_m"] > far_edge for row in wheel.values())}


def _module_origins(session: dict[str, Any]) -> dict[str, str]:
    origins = {}
    allowed = {HERE, HERE.parent / "course_impl08"}
    for name, module in tuple(sys.modules.items()):
        if not name.endswith(("_08", "_11")) or not getattr(module, "__file__", None):
            continue
        source = Path(module.__file__).resolve(strict=True)
        if source.parent not in allowed or str(source) not in session["source_hashes"]:
            raise RuntimeError("world-upright module origin not frozen: " + name)
        origins[name] = str(source)
    return origins


def _run_segment(runtime: Any, guard: Any, env: Any, output: Path,
                 raw: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    folder = output / "ramp_zero_reference"
    folder.mkdir(exist_ok=False)
    runtime.start_segment("ramp_zero_reference", CONTROLS)
    runtime.bind(env)
    guard.start_segment(env.plant, folder, "ramp_zero_reference", CONTROLS, "heldout")
    common._save(folder / "boundary_before.json", _boundary(runtime))
    common._save(folder / "geometry_manifest.json", env.plant.collision_terrain_metadata)
    records: list[dict[str, Any]] = []
    states = {name: [] for name in (
        "qpos", "qvel", "ctrl", "qacc_warmstart", "observation", "time",
    )}
    failure = None
    stop_reason = "control_limit"

    def snapshot(obs: Any) -> None:
        for name in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
            states[name].append(np.asarray(getattr(env.plant.data, name)).copy())
        states["observation"].append(np.asarray(obs).copy())
        states["time"].append(float(env.plant.data.time))

    try:
        before = _boundary(runtime)
        obs, reset_info = env.reset(seed=SEED)
        after = _boundary(runtime)
        for key in ("construction_attempts", "construction_returns",
                    "control_attempts", "control_returns"):
            if before["C_state"][key] != after["C_state"][key]:
                raise RuntimeError("world-upright reset refunded or advanced C budget")
        common._save(folder / "reset_receipt.json", {
            "before": before, "after": after, "seed": SEED,
            "model_address": int(env.plant.model._address),
            "data_address": int(env.plant.data._address),
            "episode_metadata": reset_info["episode_metadata"],
        })
        snapshot(obs)
        for tick in range(CONTROLS):
            action = np.zeros(16, dtype=np.float64)
            obs, reward, terminated, truncated, info = runtime.control_step(env, action)
            traces = env.plant.last_control_interval_actuator_traces
            consumed = env.last_transition.decision
            world = asdict(consumed.world_command)
            ground = asdict(consumed.ground)
            if (len(traces) != 5 or info["completed_control_intervals"] != tick + 1
                    or not np.array_equal(info["policy_input_action"], action)
                    or not np.array_equal(info["applied_action"], action)
                    or info["raw_operator_command"] != raw[tick]
                    or info["reference_roll_rad"] != 0.0
                    or info["reference_pitch_rad"] != 0.0
                    or world["roll_rad"] != 0.0 or world["pitch_rad"] != 0.0
                    or world["base_height_m"]
                    != ground["height_m"] + info["consumed_command"]["clearance_m"]
                    or info["controller_record"]["nominal_support"]
                    ["consumed_commanded_roll_rad"] != 0.0
                    or info["controller_record"]["nominal_support"]
                    ["consumed_commanded_pitch_rad"] != 0.0
                    or info["metrics"]["task_roll_error_rad"]
                    != info["metrics"]["absolute_roll_rad"]
                    or info["metrics"]["task_pitch_error_rad"]
                    != info["metrics"]["absolute_pitch_rad"]):
                raise RuntimeError("world-upright saved zero/control/reference chain differs")
            records.append({
                "tick": tick, "policy_input_action": action,
                "reward": reward, "terminated": terminated,
                "truncated": truncated, "info": info,
                "consumed_world_command": world,
                "consumed_actual_ground_reference": ground,
                "native_actuator_traces": [asdict(trace) for trace in traces],
            })
            snapshot(obs)
            if terminated:
                stop_reason = info["terminal_reason"]
                break
            if truncated:
                if tick != CONTROLS - 1:
                    raise RuntimeError("world-upright early TimeLimit truncation")
                stop_reason = "time_limit_1800"
                break
            if (tick + 1) % 200 == 0:
                print("upright11", tick + 1, "actual_COM",
                      info["metrics"]["body_com_vx_mps"], flush=True)
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        stop_reason = "exception"
        raise
    finally:
        runtime.unbind()
        native_segment = guard.finish_segment()
        common._save(folder / "control_records.json", records)
        if states["time"]:
            with (folder / "states.npz").open("xb") as stream:
                np.savez_compressed(stream, **{
                    key: np.asarray(value) for key, value in states.items()
                })
                stream.flush()
                os.fsync(stream.fileno())
        common._save(folder / "segment_receipt.json", {
            "name": "ramp_zero_reference", "limit": CONTROLS,
            "completed_controls": len(records), "failure": failure,
            "stop_reason": stop_reason, "native_segment": native_segment,
            "boundary_after": _boundary(runtime),
        })
    return {"completed_controls": len(records), "stop_reason": stop_reason,
            "native_segment": native_segment}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    session_path = args.session.resolve(strict=True)
    session = json.loads(session_path.read_text())
    output = _preflight(session, session_path)
    common._save(output / "worker_preflight.json", {
        "before_engine_import": True, "schema": SCHEMA,
        "frozen_source_count": len(session["source_hashes"]),
    })
    common._load_import_repair(session, output)
    import engine_binding
    import mujoco
    from course_native_guard_08 import CourseNativeGuard
    from full_drive_command_08 import QualifiedCommandCaps
    from scripts.d1_rolling_engine_runtime import RollingEngineRuntime
    from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry
    from world_upright_course_11 import WorldUprightCourseEnv

    if Path(engine_binding.__file__).resolve() != common._absolute(session, "binding_module"):
        raise RuntimeError("world-upright engine binding origin differs")
    if (_nominal_geometry.cache_info().misses != 0
            or _nominal_geometry.cache_info().currsize != 0):
        raise RuntimeError("world-upright nominal geometry cache was not cold")
    raw, applied, path = _schedule()
    common._save(output / "schedule.json", {
        "schema": "d1-world-upright-reference-schedule-11-v1",
        "seed": SEED, "spawn_position_m": SPAWN,
        "raw_commands": raw, "pure_servo_commands": applied,
        "pure_nominal_xy_m": path,
        "pure_nominal_only_not_actual_progress": True,
    })

    def command_source(tick: int, time_s: float):
        from full_drive_command_08 import FullDriveCommand

        if (type(tick) is not int or not 0 <= tick < CONTROLS
                or not math.isclose(time_s, tick * .01, rel_tol=0, abs_tol=1e-10)):
            raise RuntimeError("world-upright requested tick/clock differs")
        return FullDriveCommand(**raw[tick])

    warnings: list[str] = []

    def warning(message: str) -> None:
        warnings.append(str(message))
        raise RuntimeError("MuJoCo warning: " + str(message))

    def terminated(_sig: int, _frame: Any) -> None:
        raise common.TerminationRequested("world-upright reference interrupted")

    old_warning = mujoco.get_mju_user_warning()
    old_term = signal.signal(signal.SIGTERM, terminated)
    runtime = guard = None
    result = None
    failure = None
    started = time.monotonic()
    try:
        mujoco.set_mju_user_warning(warning)
        with RollingEngineRuntime(Path(session["library"]), control_limit=CONTROLS,
                                  construction_limit=2) as runtime:
            common._save(output / "runtime_initial.json", {
                "proof": runtime.proof, **_boundary(runtime),
            })
            with CourseNativeGuard(runtime) as guard:
                caps = QualifiedCommandCaps(1.6, 0.0, 0.0, 0.3, False)
                env = runtime.construct(lambda: WorldUprightCourseEnv(
                    caps=caps, command_source=command_source,
                    spawn_position_m=SPAWN, max_steps=CONTROLS,
                    mode="test_actor",
                ))
                state = runtime.state()
                cache = _nominal_geometry.cache_info()._asdict()
                if (state["construction_attempts"] != 2
                        or state["construction_returns"] != 2
                        or cache["misses"] != 1 or cache["currsize"] != 1
                        or env.pure_test_components):
                    raise RuntimeError("world-upright constructor was not exact cold2")
                common._save(output / "module_origins.json", _module_origins(session))
                common._save(output / "construction_receipt.json", {
                    "C_state": state, "nominal_cache": cache,
                    "model_address": int(env.plant.model._address),
                    "data_address": int(env.plant.data._address),
                    "compiled_geometry": env.plant.collision_terrain_metadata,
                    "actual_geometry_binding": _binding(env.plant),
                })
                env.set_native_interval_reader(
                    guard.interval_summary, begin_interval=guard.begin_interval,
                )
                result = _run_segment(runtime, guard, env, output, raw)
                common._save(output / "final_compiled_x_projection.json",
                             _x_projection(env.plant))
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        signal.signal(signal.SIGTERM, old_term)
        mujoco.set_mju_user_warning(old_warning)
        if runtime is not None:
            runtime.unbind()
        source_mismatches = [name for name, value in session["source_hashes"].items()
                             if not Path(name).is_file()
                             or _identity(Path(name)) != value]
        complete = (failure is None and result is not None and not source_mismatches
                    and result["completed_controls"] == CONTROLS
                    and result["stop_reason"] == "time_limit_1800")
        common._save(output / "worker_receipt.json", {
            "schema": SCHEMA, "failure": failure, "warnings": warnings,
            "execution_complete": complete,
            "record_valid": failure is None and result is not None
            and not source_mismatches,
            "baseline_qualification_pending_independent_readback": True,
            "result": result, "C_Python_final": None if runtime is None else _boundary(runtime),
            "native_guard": None if guard is None else guard.report(),
            "policy_loaded": False, "policy_predictions": 0,
            "training_or_learning_performed": False,
            "elapsed_wall_s": time.monotonic() - started,
            "retry_permitted": False,
            "source_hash_mismatches": source_mismatches,
        })
    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
