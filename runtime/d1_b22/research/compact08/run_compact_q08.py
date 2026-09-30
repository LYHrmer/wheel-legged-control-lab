"""One fresh 08-Q compact equivalence/GUI performance run, root execution only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import scripts.run_d1_latest_rl as seven

CONTRACT_SHA = "604332e9ffd262bbf292dd247670d114e4054f4841a7acf0af7f2c8e8572eaf2"
SCHEMA = "d1-compact-equivalence-q08-v1"
PAIR = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")


def stats(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "total_ns": sum(values),
        "median_ns": ordered[len(ordered) // 2],
        "p95_ns": ordered[round((len(ordered) - 1) * 0.95)],
        "max_ns": ordered[-1],
    }


def preflight(session, path):
    if (
        session.get("schema") != SCHEMA
        or session.get("contract_sha256") != CONTRACT_SHA
        or session.get("control_limit") != 800
        or session.get("normal_native_limit") != 4000
        or session.get("compiler_native_limit") != 3
        or session.get("segments") != [400, 400]
        or session.get("seed") != 77351
        or session.get("actor") != "final_policy"
        or session.get("terrain") != "box"
        or session.get("retry_permitted") is not False
        or session.get("argv") != sys.argv
        or sys.argv[1:] != ["--session", str(path)]
    ):
        raise RuntimeError("Q identity, argv or bounded reservation differs")
    records = session["source_hashes"]
    for name, expected in records.items():
        source = Path(name)
        if (
            not source.is_absolute()
            or not source.is_file()
            or source.is_symlink()
            or source.stat().st_size != expected["bytes"]
            or seven._sha256(source) != expected["sha256"]
        ):
            raise RuntimeError("source hash mismatch: " + name)
    fixed = {
        "library": seven.DSO_SHA256,
        "checkpoint_model": seven.MODEL_SHA256,
        "checkpoint_metadata": seven.METADATA_SHA256,
        "checkpoint_probe": seven.PROBE_SHA256,
        "training_protocol": seven.TRAINING_PROTOCOL_SHA256,
        "contract_path": CONTRACT_SHA,
    }
    for key, digest in fixed.items():
        source = seven._absolute(session, key)
        if str(source) not in records or seven._sha256(source) != digest:
            raise RuntimeError("fixed Q dependency differs: " + key)
    required = [
        Path(__file__).resolve(),
        *(
            Path(__file__).resolve().parent / name
            for name in (
                "compact_native_q08.py",
                "compact_viewer_q08.py",
                "profile_driver_q08.py",
            )
        ),
    ]
    if any(str(item) not in records for item in required) or len(records) < 100:
        raise RuntimeError("Q source closure absent")
    for key, value in session["runtime_environment"].items():
        if os.environ.get(key) != value:
            raise RuntimeError("Q runtime environment differs: " + key)
    if (
        "LD_LIBRARY_PATH" in os.environ
        or os.environ.get("LD_BIND_NOW") != "1"
        or os.environ.get("LD_PRELOAD") != session["library"]
    ):
        raise RuntimeError("isolated engine environment differs")


def run_segment(
    session,
    runtime,
    env,
    actor,
    commands,
    driver,
    viewer,
    monitor,
    output,
    index,
    reference,
    predictions,
):
    import numpy as np
    from dataclasses import asdict
    from scripts.d1_single_step_records import compiled_geometry_manifest
    from wheel_legged_control.d1.control_loop import D1MotionCommand

    folder = output / f"segment_{index:02d}"
    folder.mkdir(exist_ok=False)
    name = folder.name
    runtime.start_segment(name, 400)
    runtime.bind(env)
    inner, plant = env.unwrapped, env.unwrapped.plant
    monitor.start_segment(plant, folder, name, 400)
    before_c, before_python = runtime.state(), runtime.ledger_receipt()
    # Snapshot mutable ledger lists so boundaries describe the boundary itself.
    before_python = json.loads(json.dumps(seven._jsonable(before_python)))
    seven._save(
        folder / "boundary_before.json", {"C_state": before_c, "python": before_python}
    )
    manifest = compiled_geometry_manifest(plant)
    manifest["joint_qpos_addresses"] = plant.qpos_addresses.tolist()
    manifest["joint_dof_addresses"] = plant.dof_addresses.tolist()
    seven._save(folder / "geometry_manifest.json", manifest)
    arrays = {key: [] for key in (*PAIR, "time")}
    traces = []
    timings = []
    polls = []
    completed = 0
    failure = None
    stopped = "error"
    reset_ns = None
    first_poll = None
    active_start = None
    initial_draw_ns = 0
    end_poll = None
    setup_start = time.perf_counter_ns()
    policy_before = predictions["returned"]
    frame_before = 0 if viewer is None else len(viewer.frame_receipts)
    captures_before = 0 if viewer is None else len(viewer.screenshot_receipts)

    def snapshot(observation):
        for key in PAIR[:-1]:
            arrays[key].append(getattr(plant.data, key).copy())
        arrays["observation"].append(np.asarray(observation, dtype=np.float32).copy())
        arrays["time"].append(float(plant.data.time))

    def display(tick):
        if viewer is None:
            return
        # Keep status current; expensive copy/render occurs only every ten ticks.
        raw = inner.command_records[-1]
        prepared = float(raw["forward_velocity_mps"])
        actual = float(
            inner.heading_decision.decision.context.state.base_linear_velocity_body[0]
        )
        viewer.set_status(D1MotionCommand(prepared, 0.0, 0.455), tick * 0.01)
        viewer.controls_help = (
            "W | 1/2 | X | R | Esc",
            "Forward | gear | stop | reset | save/exit",
        )
        viewer.extra_help = (
            "RL residual + body speed control | "
            + ("SETTLING" if tick < 175 else "ACTIVE")
            + "\n"
            f"Selected {commands.selected_speed_mps:.2f} | request {commands.raw_forward_mps:.2f} | "
            f"prepared {prepared:.2f} | actual COM {actual:+.3f} m/s"
        )
        if tick in (0, 200, 400):
            viewer.request_capture(f"compact_{tick}", folder / f"compact_{tick}.png")
        viewer.copy_and_sync(plant, runtime, source_tick=tick)

    try:
        reset_start = time.perf_counter_ns()
        observation, info = env.reset(seed=session["seed"])
        reset_ns = time.perf_counter_ns() - reset_start
        if np.asarray(observation).shape != (85,):
            raise RuntimeError("Q observation is not85D")
        initial = {
            key: (
                np.asarray(observation, dtype=np.float32).copy()
                if key == "observation"
                else getattr(plant.data, key).copy()
            )
            for key in PAIR
        }
        address = (int(plant.model._address), int(plant.data._address))
        if not reference:
            reference.update(arrays=initial, addresses=address)
        exact = address == reference["addresses"] and all(
            initial[key].dtype == reference["arrays"][key].dtype
            and initial[key].tobytes() == reference["arrays"][key].tobytes()
            for key in PAIR
        )
        reset_c = runtime.state()
        reset_python = runtime.ledger_receipt()
        unchanged = all(
            before_c[key] == reset_c[key]
            for key in (
                "construction_attempts",
                "construction_returns",
                "control_attempts",
                "control_returns",
            )
        )
        unchanged = unchanged and all(
            before_python[key] == reset_python[key]
            for key in (
                "native_attempted",
                "native_returned",
                "control_attempted",
                "control_completed",
            )
        )
        seven._save(
            folder / "reset_receipt.json",
            {
                "same_initial": exact,
                "budget_unchanged": unchanged,
                "C_before": before_c,
                "C_after": reset_c,
                "model_address": address[0],
                "data_address": address[1],
                "initial_sha256": {
                    k: hashlib.sha256(v.tobytes()).hexdigest()
                    for k, v in initial.items()
                },
            },
        )
        if not exact or not unchanged:
            raise RuntimeError("reset changed initial state or refunded budget")
        seven._save(folder / "episode_metadata.json", info)
        with (folder / "initial_state.npz").open("xb") as stream:
            np.savez_compressed(stream, **initial)
        snapshot(observation)
        active_start = time.perf_counter_ns()
        display(0)
        initial_draw_ns = (
            time.perf_counter_ns() - active_start if viewer is not None else 0
        )
        while True:
            tick = completed
            cycle_start = time.perf_counter_ns()
            if first_poll is None:
                first_poll = cycle_start
            driver.before_poll(index, tick)
            poll_start = time.perf_counter_ns()
            if viewer is not None:
                viewer.poll(tick * 0.01)
            poll_end = time.perf_counter_ns()
            driver.after_poll(index, tick)
            polls.append(
                {
                    "tick": tick,
                    "start_ns": poll_start,
                    "end_ns": poll_end,
                    "keyboard": commands.snapshot(),
                }
            )
            if commands.stopped:
                stopped = "keyboard_escape"
                end_poll = cycle_start
                break
            if commands.consume_reset_request():
                if index != 0:
                    raise RuntimeError("Q reset reservation exhausted")
                stopped = "operator_simulation_reset"
                end_poll = cycle_start
                break
            if completed >= 400:
                raise RuntimeError("Q driver failed to exit at boundary")
            raw = dict(inner.command_records[-1])
            decision = inner.heading_decision
            if decision.tick != tick or raw["tick"] != tick:
                raise RuntimeError("Q prepared timing differs")
            predict_start = time.perf_counter_ns()
            predictions["attempted"] += 1
            action = np.asarray(
                actor.predict(observation, deterministic=True)[0], dtype=np.float32
            )
            predictions["returned"] += 1
            predict_end = time.perf_counter_ns()
            if action.shape != (1,) or not np.isfinite(action).all():
                raise RuntimeError("Q actor scalar invalid")
            control_start = time.perf_counter_ns()
            observation, reward, terminated, truncated, info = runtime.control_step(
                env, action
            )
            control_end = time.perf_counter_ns()
            completed += 1
            mapping = info["rolling_action"]
            traces.append(
                {
                    "tick": tick,
                    "policy_action": action.copy(),
                    "raw_command": raw,
                    "servo_command": asdict(decision.servo_command),
                    "world_command": asdict(decision.decision.world_command),
                    "clipped_policy_scalar": mapping["clipped_scalar"],
                    "forward_gate_active": mapping["forward_gate_active"],
                    "physical_action": np.asarray(info["physical_action"]).copy(),
                    "applied_action": np.asarray(info["applied_action"]).copy(),
                    "torque_nm": inner.last_transition.requested_torque_nm.copy(),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            snapshot(observation)
            work_end = time.perf_counter_ns()
            display(completed)
            timings.append(
                {
                    "tick": tick,
                    "cycle_start_ns": cycle_start,
                    "poll_start_ns": poll_start,
                    "poll_end_ns": poll_end,
                    "predict_start_ns": predict_start,
                    "predict_end_ns": predict_end,
                    "control_start_ns": control_start,
                    "control_end_ns": control_end,
                    "work_end_ns": work_end,
                    "cycle_end_ns": time.perf_counter_ns(),
                    "capture_after": viewer is not None and completed in (200, 400),
                }
            )
            if terminated or truncated:
                raise RuntimeError("Q task terminated before reserved boundary")
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        closure_start = time.perf_counter_ns()
        runtime.unbind()
        native = monitor.finish_segment()
        with (folder / "states.npz").open("xb") as stream:
            np.savez_compressed(stream, **{k: np.asarray(v) for k, v in arrays.items()})
        seven._save(folder / "trace.json", traces)
        seven._save(folder / "command_records.json", inner.command_records)
        seven._save(folder / "pre_control_states.json", env.pre_control_states)
        seven._save(folder / "keyboard_polls.json", polls)
        if viewer is not None:
            seven._save(
                folder / "render_frames.json", viewer.frame_receipts[frame_before:]
            )
            seven._save(
                folder / "screenshots.json",
                viewer.screenshot_receipts[captures_before:],
            )
        boundary = {
            "segment": name,
            "completed_controls": completed,
            "stop_reason": stopped,
            "failure": failure,
            "C_state": runtime.state(),
            "python": runtime.ledger_receipt(),
            "native": native,
            "live_policy_calls": predictions["returned"] - policy_before,
            "record_mode": "compact_equivalence_probe",
        }
        boundary = json.loads(json.dumps(seven._jsonable(boundary)))
        seven._save(folder / "boundary_after.json", boundary)
        closure_end = time.perf_counter_ns()
        active_end = end_poll if end_poll is not None else closure_start
        intervals = [b["start_ns"] - a["start_ns"] for a, b in zip(polls, polls[1:])]
        noncapture = [
            interval
            for row, interval in zip(timings, intervals)
            if not row["capture_after"]
        ]
        active_ns = None if active_start is None else active_end - active_start
        timing = {
            "calls": timings,
            "polls": polls,
            "reset_ns": reset_ns,
            "setup_ns": None if active_start is None else active_start - setup_start,
            "initial_draw_ns": initial_draw_ns,
            "initial_capture_in_active_wall": viewer is not None,
            "first_poll_ns": first_poll,
            "active_start_ns": active_start,
            "active_end_ns": active_end,
            "active_wall_ns": active_ns,
            "closure_ns": closure_end - closure_start,
            "rtf": None if not active_ns else (completed * 0.01) / (active_ns / 1e9),
            "work_excluding_draw": stats(
                [x["work_end_ns"] - x["cycle_start_ns"] for x in timings]
            ),
            "control": stats(
                [x["control_end_ns"] - x["control_start_ns"] for x in timings]
            ),
            "inference": stats(
                [x["predict_end_ns"] - x["predict_start_ns"] for x in timings]
            ),
            "poll_interval_all": stats(intervals),
            "poll_interval_noncapture": stats(noncapture),
        }
        seven._save(folder / "timing.json", timing)
    return boundary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    path = args.session.resolve(strict=True)
    session = seven._load(path)
    preflight(session, path)
    output = seven._absolute(session, "output_directory")
    seven._save(
        output / "worker_preflight.json",
        {
            "schema": SCHEMA,
            "before_engine_import": True,
            "contract_sha256": CONTRACT_SHA,
            "source_count": len(session["source_hashes"]),
        },
    )
    seven._load_import_repair(session, output)
    import mujoco
    from body_speed_transfer_env_06 import (
        BodyCommonPTransferredEnv,
        install_body_controller,
    )
    from scripts.d1_rolling_engine_runtime import RollingEngineRuntime
    from scripts.d1_rolling_residual_checkpoint import load_final_checkpoint
    from scripts.d1_rolling_residual_env import D1RollingResidualEnv
    from scripts.d1_rolling_residual_task import RollingEpisodeSpec
    from scripts.d1_latest_rl_controls import LatestRLCommands
    from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry
    from compact_native_q08 import CompactNativeGuard
    from compact_viewer_q08 import CompactViewer
    from profile_driver_q08 import LogicalProfileQDriver

    origins = seven._check_origins(session)
    seven._save(output / "module_origins.json", {"origins": origins})
    if _nominal_geometry.cache_info()._asdict() != {
        "hits": 0,
        "misses": 0,
        "maxsize": 1,
        "currsize": 0,
    }:
        raise RuntimeError("nominal cache not cold")

    class InteractiveEnv(BodyCommonPTransferredEnv):
        @property
        def episode_metadata(self):
            value = dict(super().episode_metadata)
            value["historical_definition"] = value.pop("rolling_task_definition")
            value["task_schema"] = "d1-latest-rl-straight-interactive-v1"
            value["rolling_task_definition"] = {
                "schema": value["task_schema"],
                "command_source": seven.COMMAND_SCHEMA,
                "physical_terrain": "box",
                "settle_ticks": 175,
                "terminal_prepared_tick": 1200,
                "command_set_mps": [0.0, 0.20, 0.25],
                "requested_world_height_m": 0.455,
                "yaw_request_rps": 0.0,
            }
            value["record_mode"] = "compact_equivalence_probe"
            value["instrumentation"] = (
                "08-Q; original85D scalar policy and06 control unchanged"
            )
            return value

    warnings = []

    def warning(message):
        warnings.append(str(message))
        raise RuntimeError("MuJoCo warning: " + str(message))

    def term(_sig, _frame):
        raise seven.TerminationRequested("Q timeout/interruption")

    previous_warning = mujoco.get_mju_user_warning()
    previous_sigterm = signal.signal(signal.SIGTERM, term)
    runtime = monitor = viewer = actor = original = None
    failure = None
    segments = []
    predictions = {"attempted": 0, "returned": 0}
    reference = {}
    started = time.perf_counter_ns()
    phase = {}
    try:
        mujoco.set_mju_user_warning(warning)
        with RollingEngineRuntime(
            seven._absolute(session, "library"), control_limit=800, construction_limit=3
        ) as runtime:
            seven._save(
                output / "runtime_initial.json",
                {
                    "proof": runtime.proof,
                    "C_state": runtime.state(),
                    "python": runtime.ledger_receipt(),
                    "before_model": True,
                },
            )
            with CompactNativeGuard(runtime, output) as monitor:
                now = time.perf_counter_ns()
                original = runtime.construct(
                    lambda: D1RollingResidualEnv(
                        RollingEpisodeSpec(0.20, 175, 975, True)
                    )
                )
                phase["construction_ns"] = time.perf_counter_ns() - now
                constructed = runtime.state()
                cache = _nominal_geometry.cache_info()._asdict()
                if (
                    constructed["construction_returns"] != 3
                    or constructed["construction_attempts"] != 3
                    or cache != {"hits": 1, "misses": 1, "maxsize": 1, "currsize": 1}
                ):
                    raise RuntimeError("Q cold construction not exactly3")
                seven._save(
                    output / "construction_receipt.json",
                    {"C_state": constructed, "cache": cache},
                )
                now = time.perf_counter_ns()
                actor = load_final_checkpoint(
                    seven._absolute(session, "checkpoint_model"),
                    seven._absolute(session, "checkpoint_metadata"),
                    original,
                    expected_dso_sha256=seven.DSO_SHA256,
                    expected_training_protocol_sha256=seven.TRAINING_PROTOCOL_SHA256,
                )
                phase["strict_load_ns"] = time.perf_counter_ns() - now
                if runtime.state() != constructed:
                    raise RuntimeError("strict loader advanced physics")
                receipt = install_body_controller(original)
                if (
                    runtime.state() != constructed
                    or _nominal_geometry.cache_info().misses != 1
                ):
                    raise RuntimeError("transfer changed physical/cache identity")
                seven._save(
                    output / "strict_load_and_transfer.json",
                    {
                        "checkpoint_sha256": seven.MODEL_SHA256,
                        "two_saved_probe_observations_one_batch": True,
                        "native_delta": 0,
                        "transfer": receipt,
                    },
                )
                env = InteractiveEnv(original, receipt)
                commands = LatestRLCommands()
                env.unwrapped.command_source = seven.InteractiveCommandSource(
                    env.unwrapped, commands
                )
                driver = LogicalProfileQDriver(commands)
                try:
                    for index in (0, 1):
                        if index:
                            commands.reset_for_new_segment()
                            driver.reset_for_new_segment()
                            before = runtime.state()
                            now = time.perf_counter_ns()
                            viewer = CompactViewer(
                                env.unwrapped.plant.model,
                                mujoco.MjData(env.unwrapped.plant.model),
                                commands,
                                terrain_label="15 mm box",
                            )
                            phase["gui_construction_ns"] = time.perf_counter_ns() - now
                            if runtime.state() != before:
                                raise RuntimeError(
                                    "viewer creation changed native counts"
                                )
                        row = run_segment(
                            session,
                            runtime,
                            env,
                            actor,
                            commands,
                            driver,
                            viewer,
                            monitor,
                            output,
                            index,
                            reference,
                            predictions,
                        )
                        segments.append(row)
                        print(
                            "08-Q",
                            index,
                            row["completed_controls"],
                            row["stop_reason"],
                            flush=True,
                        )
                        if row["stop_reason"] != "operator_simulation_reset":
                            break
                finally:
                    seven._save(output / "driver_receipt.json", driver.report())
                    if viewer is not None:
                        seven._save(
                            output / "render_receipt.json",
                            {
                                "frames": viewer.frame_receipts,
                                "screenshots": viewer.screenshot_receipts,
                                "rendered_frames": viewer.rendered_frames,
                                "skipped_frames": viewer.skipped_frames,
                                "draw_calls": viewer.draw_calls,
                            },
                        )
                        viewer.close()
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        mujoco.set_mju_user_warning(previous_warning)
        if runtime is not None:
            runtime.unbind()
        completed = None if runtime is None else runtime.control_completed
        complete = (
            failure is None
            and completed == 800
            and predictions == {"attempted": 800, "returned": 800}
            and len(segments) == 2
            and [s["completed_controls"] for s in segments] == [400, 400]
            and [s["stop_reason"] for s in segments]
            == ["operator_simulation_reset", "keyboard_escape"]
        )
        seven._save(
            output / "worker_receipt.json",
            {
                "schema": SCHEMA,
                "contract_sha256": CONTRACT_SHA,
                "failure": failure,
                "execution_complete": complete,
                "warnings": warnings,
                "segments": segments,
                "control_completed": completed,
                "control_attempted": None
                if runtime is None
                else runtime.control_attempted,
                "predictions": predictions,
                "strict_loader_probe_observations": 2 if actor is not None else 0,
                "strict_loader_probe_batches": 1 if actor is not None else 0,
                "C_final": None if runtime is None else runtime.state(),
                "python": None if runtime is None else runtime.ledger_receipt(),
                "monitor": None if monitor is None else monitor.report(),
                "viewer_closed": viewer is None or viewer.window is None,
                "elapsed_wall_ns": time.perf_counter_ns() - started,
                "phases": phase,
                "performance_qualification_pending": True,
                "independent_readback_pending": True,
                "old_budget_reused": False,
                "retry_permitted": False,
                "training_steps": 0,
            },
        )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
