"""Single counted E0 authority / E1 baseline speed diagnostic, never training."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

from scripts import run_d1_latest_rl as common

SCHEMA = "d1-course-e-rl16-authority-speed-08-v1"
CONTRACT_SHA = "475273fd4517263606af7abd07008cdd6811f9992e6affacf6eb758a9ccbddd7"


def preflight(session, path):
    expected = {
        "schema": SCHEMA,
        "phase": "E",
        "control_limit": 2400,
        "normal_native_limit": 12000,
        "compiler_native_limit": 2,
        "wallclock_limit_s": 600,
        "segments": [800, 1600],
        "seed": 88401,
        "contract_sha256": CONTRACT_SHA,
        "retry_permitted": False,
    }
    if any(session.get(k) != v for k, v in expected.items()):
        raise RuntimeError("E identity/cap differs from reviewed source")
    if session["argv"] != sys.argv or sys.argv[1:] != ["--session", str(path)]:
        raise RuntimeError("E argv mismatch")
    for name, record in session["source_hashes"].items():
        source = Path(name)
        if (
            not source.is_file()
            or source.is_symlink()
            or source.stat().st_size != record["bytes"]
            or common._sha256(source) != record["sha256"]
        ):
            raise RuntimeError("frozen E input differs: " + name)
    if str(Path(__file__).resolve()) not in session["source_hashes"]:
        raise RuntimeError("worker not frozen")
    if common._sha256(Path(session["library"])) != common.DSO_SHA256:
        raise RuntimeError("unreviewed engine DSO")
    for key, value in session["runtime_environment"].items():
        if os.environ.get(key) != value:
            raise RuntimeError("E environment differs: " + key)
    if (
        "LD_LIBRARY_PATH" in os.environ
        or os.environ.get("LD_BIND_NOW") != "1"
        or os.environ.get("LD_PRELOAD") != session["library"]
    ):
        raise RuntimeError("E isolated engine environment invalid")


def boundary(runtime):
    return common._jsonable(
        {"C_state": runtime.state(), "python": runtime.ledger_receipt()}
    )


def save_arrays(path, **values):
    import numpy as np

    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())


def measure_compiled_scan(runtime, env, output):
    """Finite pure CPU microbenchmark of the actual 92-primitive map."""
    import numpy as np
    from full_drive_observation_08 import scan_compiled_course_ground

    positions = [
        (-8.0, -4.7),
        (0.35, -2.2),
        (0.35, 0.0),
        (2.75, 0.0),
        (3.5, 0.0),
        (4.0, -2.2),
        (5.0, 0.0),
        (0.0, 2.2),
    ]
    before = boundary(runtime)
    durations = []
    for index in range(512):
        x, y = positions[index % len(positions)]
        start = time.perf_counter_ns()
        scan_compiled_course_ground(
            env.ground_map, x_m=x, y_m=y, yaw_rad=(index % 5) * 0.15
        )
        durations.append(time.perf_counter_ns() - start)
    after = boundary(runtime)
    if before != after:
        raise RuntimeError("pure terrain query advanced engine/runtime state")
    common._save(
        output / "ground_scan_benchmark.json",
        {
            "pure_cpu_only": True,
            "calls": 512,
            "positions": positions,
            "durations_ns": durations,
            "median_ns": float(np.median(durations)),
            "p95_ns": float(np.percentile(durations, 95)),
            "max_ns": max(durations),
            "terrain_geoms": len(env.plant.terrain_geom_ids),
            "native_counter_delta": 0,
            "before": before,
            "after": after,
        },
    )


def check_compiled_flat_corridor(runtime, env, output):
    """Conservative nominal swept footprint from the actual reset geometry."""
    import math
    import numpy as np
    from course_corridor_08 import assert_nominal_corridor_clearance
    from full_drive_schedule_08 import e1_ladder_schedule
    from full_drive_servo_08 import FullDriveCommandServo

    plant = env.plant
    before = boundary(runtime)
    manifest = {
        "base_world_position_m": plant.data.qpos[:3].copy().tolist(),
        "model_identity": {
            "model_address": int(plant.model._address),
            "data_address": int(plant.data._address),
            "source": "actual_compiled_course_reset_geometry_v1",
            "model_ngeom": int(plant.model.ngeom),
        },
        "robot_collision_geoms": [
            {
                "geom_id": gid,
                "body_id": int(plant.model.geom_bodyid[gid]),
                "world_center_m": plant.data.geom_xpos[gid].copy().tolist(),
                "rbound_m": float(plant.model.geom_rbound[gid]),
                "collision": True,
            }
            for gid in range(plant.model.ngeom)
            if plant.model.geom_bodyid[gid] != 0
            and (plant.model.geom_contype[gid] or plant.model.geom_conaffinity[gid])
        ],
        "terrain_world_geoms": plant.collision_terrain_metadata[
            "world_collision_geoms"
        ],
    }
    schedule = e1_ladder_schedule()
    servo = FullDriveCommandServo(env.caps)
    x, y = schedule.spawn_position_m[:2]
    yaw = 0.0
    points = [(x, y)]
    for tick, raw in enumerate(schedule.raw_commands):
        command = servo.advance(tick, raw).applied
        x += math.cos(yaw) * command.forward_velocity_mps * 0.01
        y += math.sin(yaw) * command.forward_velocity_mps * 0.01
        yaw += command.yaw_rate_rps * 0.01
        points.append((x, y))
    report = assert_nominal_corridor_clearance(manifest, np.asarray(points))
    after = boundary(runtime)
    if before != after:
        raise RuntimeError("compiled footprint precheck changed engine/runtime state")
    common._save(output / "flat_corridor_geometry.json", manifest)
    save_arrays(output / "flat_corridor_nominal_path.npz", xy_m=np.asarray(points))
    common._save(
        output / "flat_corridor_precheck.json",
        {
            "nominal_only_not_measured_trajectory": True,
            "report": report,
            "before": before,
            "after": after,
            "native_counter_delta": 0,
        },
    )


def segment(runtime, guard, env, output, *, index, schedule, seed):
    import numpy as np
    from full_drive_schedule_08 import e0_stationary_probe_action

    label = "e0_action_authority" if index == 0 else "e1_zero_residual_speed"
    limit = 800 if index == 0 else 1600
    folder = output / label
    folder.mkdir(exist_ok=False)
    runtime.start_segment(label, limit)
    runtime.bind(env)
    guard.start_segment(
        env.plant,
        folder,
        label,
        limit,
        "authority_probe" if index == 0 else "baseline_ladder",
    )
    common._save(folder / "boundary_before.json", boundary(runtime))
    common._save(
        folder / "geometry_manifest.json", env.plant.collision_terrain_metadata
    )
    env.max_steps = limit
    env.controller.enable_stationary_action_probe = index == 0
    env.command_source = schedule
    records = []
    states = {
        name: []
        for name in ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation", "time")
    }
    failure = None
    stop_reason = "control_limit"
    started = time.monotonic()

    def snapshot(observation):
        for name in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
            states[name].append(getattr(env.plant.data, name).copy())
        states["observation"].append(np.array(observation, copy=True))
        states["time"].append(float(env.plant.data.time))

    try:
        before_reset = boundary(runtime)
        obs, _ = env.reset(seed=seed)
        after_reset = boundary(runtime)
        for key in (
            "control_attempts",
            "control_returns",
            "construction_attempts",
            "construction_returns",
        ):
            if after_reset["C_state"][key] != before_reset["C_state"][key]:
                raise RuntimeError("reset changed integration/compiler count")
        common._save(
            folder / "reset_receipt.json",
            {
                "before": before_reset,
                "after": after_reset,
                "seed": seed,
                "model_address": int(env.plant.model._address),
                "data_address": int(env.plant.data._address),
                "stationary_probe_enabled": env.controller.enable_stationary_action_probe,
                "episode_metadata": env.episode_metadata,
            },
        )
        if index == 0:
            check_compiled_flat_corridor(runtime, env, output)
        snapshot(obs)
        for tick in range(limit):
            action = e0_stationary_probe_action(tick) if index == 0 else np.zeros(16)
            before = time.perf_counter_ns()
            obs, reward, terminated, truncated, info = runtime.control_step(env, action)
            elapsed = time.perf_counter_ns() - before
            traces = env.plant.last_control_interval_actuator_traces
            row = {
                "tick": tick,
                "policy_input_action": action.copy(),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "wall_ns": elapsed,
                "info": info,
                "native_actuator_traces": traces,
            }
            # Retain actual immutable values until one segment-end write. The
            # guard independently stores each real native state/contact cache.
            records.append(row)
            snapshot(obs)
            if len(traces) != 5:
                raise RuntimeError("E actuator trace did not cover five native steps")
            calc = info["controller_record"]["calculation"]
            if not np.array_equal(
                calc["safe_torque_nm"], env.last_transition.requested_torque_nm
            ):
                raise RuntimeError("E controller-to-loop torque authority differs")
            if not np.array_equal(info["applied_action"], action):
                raise RuntimeError(
                    "E requested action was not the actual applied residual"
                )
            interval = info["native_interval_summary"]
            if (
                interval["nonwheel_contact_count"] != 0
                or max(interval["max_abs_roll_deg"], interval["max_abs_pitch_deg"]) > 10
            ):
                stop_reason = "E_native_safety_qualification_failed"
                break
            if terminated:
                stop_reason = info["terminal_reason"]
                break
            if truncated and tick != limit - 1:
                raise RuntimeError("E task truncated before prepaid segment limit")
            if (tick + 1) % 200 == 0:
                print(
                    label,
                    tick + 1,
                    "actual_COM",
                    info["metrics"]["body_com_vx_mps"],
                    flush=True,
                )
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        stop_reason = "exception"
        raise
    finally:
        runtime.unbind()
        guard.finish_segment()
        # ActuatorTrace is a dataclass; JSON conversion is only at closure.
        for row in records:
            row["native_actuator_traces"] = [
                asdict(trace) for trace in row["native_actuator_traces"]
            ]
        common._save(folder / "control_records.json", records)
        if states["time"]:
            save_arrays(
                folder / "states.npz", **{k: np.asarray(v) for k, v in states.items()}
            )
        receipt = {
            "name": label,
            "limit": limit,
            "completed": len(records),
            "failure": failure,
            "stop_reason": stop_reason,
            "active_and_recording_wall_s": time.monotonic() - started,
            "qualification_passed": failure is None
            and stop_reason == "control_limit"
            and len(records) == limit,
            "boundary_after": boundary(runtime),
        }
        common._save(folder / "segment_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    path = args.session.resolve(strict=True)
    session = json.loads(path.read_text())
    preflight(session, path)
    output = common._absolute(session, "output_directory")
    common._save(
        output / "worker_preflight.json",
        {
            "before_engine_import": True,
            "source_count": len(session["source_hashes"]),
            "schema": SCHEMA,
        },
    )
    common._load_import_repair(session, output)
    import mujoco
    import course_corridor_08
    from course_native_guard_08 import CourseNativeGuard
    from full_drive_command_08 import QualifiedCommandCaps
    from full_drive_env_08 import FullDriveCourseEnv
    from full_drive_schedule_08 import (
        FLAT_SPAWN_M,
        e0_stationary_probe_command,
        e1_ladder_schedule,
        precheck_nominal_path,
    )
    from scripts.d1_rolling_engine_runtime import RollingEngineRuntime
    from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry

    expected_directory = Path(__file__).resolve().parent / "course_impl08"
    origins = {"course_corridor_08": str(Path(course_corridor_08.__file__).resolve())}
    for name, value in tuple(sys.modules.items()):
        if name.endswith("_08") and getattr(value, "__file__", None):
            origin = Path(value.__file__).resolve()
            if (
                origin.parent != expected_directory
                or str(origin) not in session["source_hashes"]
            ):
                raise RuntimeError(
                    "new control module not frozen at expected origin: " + name
                )
            origins[name] = str(origin)
    common._save(output / "module_origins.json", origins)
    if (
        _nominal_geometry.cache_info().misses != 0
        or _nominal_geometry.cache_info().currsize != 0
    ):
        raise RuntimeError("E nominal geometry cache is not cold")
    caps = QualifiedCommandCaps(1.6, 0.0, 0.0, 0.3, False)
    schedule = e1_ladder_schedule()
    nominal_path = precheck_nominal_path(schedule, caps)
    common._save(output / "E1_nominal_path.json", asdict(nominal_path))
    if not nominal_path.passed:
        raise RuntimeError("E1 pure command path outside lane")
    warnings = []

    def warning(message):
        warnings.append(str(message))
        raise RuntimeError("MuJoCo warning: " + str(message))

    def terminated(_sig, _frame):
        raise common.TerminationRequested("E interrupted or wall limit reached")

    old_warning = mujoco.get_mju_user_warning()
    old_term = signal.signal(signal.SIGTERM, terminated)
    runtime = guard = env = None
    failure = None
    receipts = []
    started = time.monotonic()
    try:
        mujoco.set_mju_user_warning(warning)
        with RollingEngineRuntime(
            Path(session["library"]), control_limit=2400, construction_limit=2
        ) as runtime:
            common._save(
                output / "runtime_initial.json",
                {"proof": runtime.proof, **boundary(runtime)},
            )
            with CourseNativeGuard(runtime) as guard:
                env = runtime.construct(
                    lambda: FullDriveCourseEnv(
                        caps=caps,
                        command_source=e0_stationary_probe_command,
                        spawn_position_m=FLAT_SPAWN_M,
                        max_steps=800,
                        mode="test_actor",
                        enable_stationary_action_probe=True,
                    )
                )
                state = runtime.state()
                cache = _nominal_geometry.cache_info()._asdict()
                if (
                    state["construction_attempts"] != 2
                    or state["construction_returns"] != 2
                    or cache["misses"] != 1
                    or cache["currsize"] != 1
                    or env.pure_test_components
                ):
                    raise RuntimeError(
                        "E construction not one course plus one nominal scratch"
                    )
                common._save(
                    output / "construction_receipt.json",
                    {
                        "C_state": state,
                        "nominal_cache": cache,
                        "actual_geometry_binding": {
                            "base_body_id": int(env.plant.base_body_id),
                            "base_body_ipos": env.plant.model.body_ipos[
                                env.plant.base_body_id
                            ].copy(),
                            "wheel_index_by_body_id": dict(
                                env.plant._wheel_index_by_body_id
                            ),
                            "joint_ids": env.plant.joint_ids.copy(),
                            "joint_qpos_addresses": env.plant.qpos_addresses.copy(),
                            "joint_dof_addresses": env.plant.dof_addresses.copy(),
                            "actuator_ids": env.plant.actuator_ids.copy(),
                            "actuator_trntype": env.plant.model.actuator_trntype.copy(),
                            "actuator_trnid": env.plant.model.actuator_trnid.copy(),
                            "actuator_gear": env.plant.model.actuator_gear.copy(),
                            "body_mass": env.plant.model.body_mass.copy(),
                            "nominal_total_mass_kg": env.plant.nominal_total_mass_kg,
                            "geom_bodyid": env.plant.model.geom_bodyid.copy(),
                            "geom_type": env.plant.model.geom_type.copy(),
                            "geom_size": env.plant.model.geom_size.copy(),
                            "geom_contype": env.plant.model.geom_contype.copy(),
                            "geom_conaffinity": env.plant.model.geom_conaffinity.copy(),
                        },
                    },
                )
                measure_compiled_scan(runtime, env, output)
                env.set_native_interval_reader(
                    guard.interval_summary, begin_interval=guard.begin_interval
                )
                for index, source in enumerate(
                    (e0_stationary_probe_command, schedule.command_source)
                ):
                    row = segment(
                        runtime,
                        guard,
                        env,
                        output,
                        index=index,
                        schedule=source,
                        seed=88401,
                    )
                    receipts.append(row)
                    if not row["qualification_passed"]:
                        break
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        signal.signal(signal.SIGTERM, old_term)
        mujoco.set_mju_user_warning(old_warning)
        if runtime is not None:
            runtime.unbind()
        complete = (
            failure is None
            and len(receipts) == 2
            and all(row["qualification_passed"] for row in receipts)
        )
        common._save(
            output / "worker_receipt.json",
            {
                "schema": SCHEMA,
                "failure": failure,
                "warnings": warnings,
                "execution_complete": complete,
                "segments": receipts,
                "training_steps": 0,
                "C_final": None if runtime is None else runtime.state(),
                "python": None if runtime is None else runtime.ledger_receipt(),
                "native_guard": None if guard is None else guard.report(),
                "elapsed_wall_s": time.monotonic() - started,
                "retry_permitted": False,
                "independent_readback_pending": True,
                "measured_speed_qualification_pending": True,
            },
        )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
