"""One bounded 99D/16D flat-0.6 development arm, headless or GUI.

The owner thread alone loads the policy and owns physics. The main thread may
read a leased MjData snapshot, and in GUI mode renders that detached data.
This 600-control script is new development evidence, not a replay of 11-S.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gui13_contract import (
    CHECKPOINT,
    HERE,
    HORIZON,
    MANIFEST,
    NORMAL_NATIVE,
    PAIR,
    SCHEMA,
    SEED,
    W,
    command_speed,
    digest,
    read_json,
    verify_evidence,
)


def preflight(session: dict, path: Path) -> Path:
    """Run without importing an engine or policy package."""
    from scripts import run_d1_latest_rl as common

    if (session.get("schema") != SCHEMA or session.get("arm") not in ("headless", "gui")
            or session.get("actor") != "final_policy" or session.get("terrain") != "flat"
            or session.get("seed") != SEED or session.get("control_limit") != HORIZON
            or session.get("normal_native_limit") != NORMAL_NATIVE
            or session.get("compiler_native_limit") != 2
            or session.get("wallclock_soft_s") != 95
            or session.get("wallclock_hard_s") != 120
            or session.get("retry_permitted") is not False
            or session.get("evidence") != verify_evidence()
            or session.get("argv") != sys.argv
            or sys.argv[1:] != ["--session", str(path)]):
        raise RuntimeError("GUI13 worker session differs from reviewed 600-tick profile")
    output = Path(session["output_directory"])
    if output.parent != HERE or not output.is_dir() or path != output / "session.json":
        raise RuntimeError("GUI13 requires the launcher-created exclusive output")
    sources = session.get("source_hashes")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("GUI13 source closure missing")
    plan_path = Path(session["plan_path"])
    if (not plan_path.is_absolute() or plan_path.is_symlink()
            or sources.get(str(plan_path)) != digest(plan_path)
            or session.get("plan_sha256") != digest(plan_path)["sha256"]):
        raise RuntimeError("GUI13 reviewed plan identity differs")
    required = [Path(__file__), HERE / "gui13_contract.py", HERE / "gui13_bridge.py",
                HERE / "launch_gui13.py", W / "gui12/run_world_upright_gui_12.py",
                W / "gui12/latest_frame_mailbox_12.py", W / "gui12/async_course_renderer_12.py",
                W / "rl11/rl16_learning_11.py", W / "upright11/world_upright_course_11.py",
                W / "course_impl08/course_native_guard_08.py",
                W / "continuation13/archive13/atomic_archive_13.py", MANIFEST]
    if any(str(item.resolve()) not in sources for item in required):
        raise RuntimeError("GUI13 critical source missing from closure")
    for name, expected in sources.items():
        source = Path(name)
        if (not source.is_absolute() or not source.is_file() or source.is_symlink()
                or digest(source) != expected):
            raise RuntimeError("GUI13 frozen source changed: " + name)
    manifest = read_json(MANIFEST)
    for name in manifest["files"]:
        if str(CHECKPOINT / name) not in sources:
            raise RuntimeError("GUI13 final payload missing from closure")
    if str(CHECKPOINT / "final_metadata.json") not in sources:
        raise RuntimeError("GUI13 final metadata missing from closure")
    if (common._sha256(Path(session["library"])) != common.DSO_SHA256
            or os.environ.get("LD_PRELOAD") != session["library"]
            or os.environ.get("LD_BIND_NOW") != "1"
            or "LD_LIBRARY_PATH" in os.environ):
        raise RuntimeError("GUI13 isolated fixed DSO differs")
    for key, value in session["runtime_environment"].items():
        if os.environ.get(key) != value:
            raise RuntimeError("GUI13 isolated runtime environment differs: " + key)
    return output


def make_atomic_guard(runtime: Any, writer: Any) -> Any:
    """Install writer at the native guard's actual gzip/array flush sites."""
    from archive13.atomic_archive_13 import AtomicCourseNativeGuard

    return AtomicCourseNativeGuard(runtime, writer)


def run(session: dict, output: Path, *, execution_started: float,
        writer_factory: Callable[[], Any] | None = None) -> dict:
    """Execute one arm. `writer_factory` is the injectable archive transaction seam."""
    from scripts import run_d1_latest_rl as common

    if writer_factory is None:
        from archive13.atomic_archive_13 import ArchiveWriter

        writer_factory = ArchiveWriter
    writer = writer_factory()
    owner_started = execution_started
    stop = threading.Event()
    prior_term = signal.signal(signal.SIGTERM, lambda _number, _frame: stop.set())
    common._load_import_repair(session, output)
    import mujoco
    import numpy as np
    from async_course_renderer_12 import AsyncCourseRenderer
    from full_drive_command_08 import QualifiedCommandCaps
    from gui13_bridge import (
        LogicalScript,
        fence_copy_data,
        mailbox_type,
        owner_runtime_type,
        pause_snapshot_satisfied,
    )
    from rl16_learning_11 import load_and_verify_final
    from run_rl16_training_08 import (
        _boundary,
        _capture_state,
        _numeric_actor_action,
        _require_unchanged_reset,
    )
    from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry
    from world_upright_course_11 import WorldUprightCourseEnv

    ready = threading.Event()
    start = threading.Event()
    done = threading.Event()
    initial_frame = threading.Event()
    initial_ack = threading.Event()
    final_frame = threading.Event()
    final_ack = threading.Event()
    final_pending = threading.Event()
    input_request = threading.Event()
    input_request_done = threading.Event()
    input_ack = threading.Event()
    input_ack_done = threading.Event()
    shared: dict[str, Any] = {}
    warning_messages: list[str] = []

    schedule = LogicalScript(threading.get_ident())

    def owner() -> None:
        runtime = guard = env = actor = None
        segment = None
        failure = None
        completed = predictions = 0
        records: list[dict] = []
        states: dict[str, list] = {key: [] for key in (*PAIR, "time")}
        final_info = None
        original_copy = None
        copy_edges = None
        warning_old = None
        active_start_ns = active_end_ns = None
        pause_proofs: list[dict] = []
        model_calls = {"load_attempted": 0, "load_returned": 0,
                       "predict_attempted": 0, "predict_returned": 0,
                       "learn_attempted": 0, "save_attempted": 0}
        model_hooks = None
        prefix_saved = False
        try:
            runtime_class = owner_runtime_type(stop)
            warning_old = mujoco.get_mju_user_warning()

            def on_warning(message):
                warning_messages.append(str(message))
                stop.set()
                raise RuntimeError("MuJoCo warning: " + str(message))

            mujoco.set_mju_user_warning(on_warning)
            with runtime_class(Path(session["library"]), control_limit=HORIZON,  # noqa: SIM117 - guard needs entered runtime
                               construction_limit=2) as runtime:
                with make_atomic_guard(runtime, writer) as guard:
                    caps = QualifiedCommandCaps(0.6, 0.0, 0.0, 0.0, False)
                    env = runtime.construct(lambda: WorldUprightCourseEnv(
                        caps=caps, command_source=schedule,
                        spawn_position_m=(-8.0, -4.7, 0.455),
                        max_steps=HORIZON, mode="eval"))
                    cold = _nominal_geometry.cache_info()._asdict()
                    state = runtime.state()
                    if (state["construction_attempts"] != 2
                            or state["construction_returns"] != 2
                            or cold["misses"] != 1 or cold["currsize"] != 1):
                        raise RuntimeError("GUI13 did not construct exactly cold2")
                    writer.commit_json(output / "construction.json", common._jsonable({
                        "boundary": _boundary(runtime), "nominal_cache": cold,
                        "geometry": env.plant.collision_terrain_metadata,
                        "actuator_ids": env.plant.actuator_ids.tolist(),
                        "model_address": int(env.plant.model._address),
                        "data_address": int(env.plant.data._address)}))
                    env.set_native_interval_reader(guard.interval_summary,
                                                   begin_interval=guard.begin_interval)
                    from stable_baselines3 import PPO
                    from stable_baselines3.common.base_class import BaseAlgorithm

                    load_descriptor = PPO.__dict__.get("load")
                    old_load, old_predict = PPO.load, BaseAlgorithm.predict
                    old_learn, old_save = PPO.learn, BaseAlgorithm.save

                    def counted_load(cls, *args, **kwargs):
                        model_calls["load_attempted"] += 1
                        loaded = old_load(*args, **kwargs)
                        model_calls["load_returned"] += 1
                        return loaded

                    def counted_predict(model, *args, **kwargs):
                        model_calls["predict_attempted"] += 1
                        result = old_predict(model, *args, **kwargs)
                        model_calls["predict_returned"] += 1
                        return result

                    def reject_learn(model, *args, **kwargs):
                        model_calls["learn_attempted"] += 1
                        raise RuntimeError("GUI13 learning forbidden")

                    def reject_save(model, *args, **kwargs):
                        model_calls["save_attempted"] += 1
                        raise RuntimeError("GUI13 policy save forbidden")

                    PPO.load = classmethod(counted_load)
                    BaseAlgorithm.predict = counted_predict
                    PPO.learn = reject_learn
                    BaseAlgorithm.save = reject_save
                    model_hooks = (PPO, BaseAlgorithm, load_descriptor, old_predict,
                                   old_learn, old_save)
                    before_load = _boundary(runtime)
                    actor, load_report = load_and_verify_final(
                        str(CHECKPOINT), read_json(MANIFEST), return_model=True)
                    if _boundary(runtime) != before_load or load_report.get(
                            "probe_actions_byte_exact") is not True or model_calls != {
                                "load_attempted": 1, "load_returned": 1,
                                "predict_attempted": 1, "predict_returned": 1,
                                "learn_attempted": 0, "save_attempted": 0}:
                        raise RuntimeError("GUI13 strict final load changed physics/probe")
                    writer.commit_json(output / "strict_load.json", common._jsonable(load_report))
                    model = env.plant.model
                    slots = tuple(mujoco.MjData(model) for _ in range(3))
                    display = mujoco.MjData(model)
                    mailbox = mailbox_type()(slots)
                    original_copy, copy_edges = fence_copy_data(
                        mujoco, plant=env.plant, slots=slots, display=display,
                        mailbox=mailbox, owner_ident=threading.get_ident(),
                        main_ident=shared["main_ident"], stop_event=stop)
                    shared.update(model=model, display=display, mailbox=mailbox,
                                  original_copy=original_copy)
                    ready.set()
                    if not start.wait(timeout=20):
                        raise TimeoutError("GUI13 renderer did not start")
                    folder = output / "case"
                    folder.mkdir(exist_ok=False)
                    runtime.start_segment("gui13_flat0p6", HORIZON)
                    runtime.bind(env)
                    guard.start_segment(env.plant, folder, "gui13_flat0p6", HORIZON,
                                        "train")
                    before_reset = _boundary(runtime)
                    observation, reset_info = env.reset(
                        seed=SEED, options={"command_source": schedule,
                                            "spawn_position_m": (-8.0, -4.7, 0.455)})
                    after_reset = _boundary(runtime)
                    _require_unchanged_reset(before_reset, after_reset)
                    writer.commit_json(folder / "reset.json", common._jsonable({
                        "before": before_reset, "after": after_reset,
                        "episode_metadata": reset_info["episode_metadata"]}))

                    def snapshot(obs):
                        row = _capture_state(env, obs)
                        for key, values in states.items():
                            values.append(row[key])

                    def stable_render_pause(stage, request, acknowledge):
                        before = _boundary(runtime)
                        # Keep each buffer distinct in the actual comparison.
                        live = tuple(np.asarray(getattr(env.plant.data, name)).copy()
                                     for name in ("qpos", "qvel", "ctrl", "qacc_warmstart"))
                        measured = tuple(np.asarray(getattr(env.plant.measurement_data, name)).copy()
                                         for name in ("qpos", "qvel", "ctrl", "qacc_warmstart"))
                        request.set()
                        if not acknowledge.wait(timeout=5):
                            raise TimeoutError("GUI13 render pause acknowledgement missing")
                        unchanged = (before == _boundary(runtime)
                                     and all(np.array_equal(a, getattr(env.plant.data, name))
                                             for a, name in zip(live, ("qpos", "qvel", "ctrl", "qacc_warmstart")))
                                     and all(np.array_equal(a, getattr(env.plant.measurement_data, name))
                                             for a, name in zip(measured, ("qpos", "qvel", "ctrl", "qacc_warmstart"))))
                        pause_proofs.append({"stage": stage, "unchanged": unchanged,
                                             "C_and_python_boundary": before})
                        if not unchanged:
                            raise RuntimeError("GUI13 render changed numeric state/counts")

                    def publish(index, info=None):
                        metrics = {} if info is None else info["metrics"]
                        request = 0.0 if index == 0 else command_speed(index - 1)
                        applied = (0.0 if info is None else
                                   float(info["consumed_command"]["forward_velocity_mps"]))
                        metadata = {
                            "episode_id": 0, "control_index": index,
                            "sim_time_s": float(env.plant.data.time),
                            "source_wall_ns": time.monotonic_ns(),
                            "requested_com_speed_mps": request,
                            "applied_com_speed_mps": applied,
                            "actual_com_speed_mps": metrics.get("body_com_vx_mps"),
                            "policy_mode": "final_policy",
                            "terrain_state": "flat",
                            "stop_state": "soft_stop" if stop.is_set() else "running",
                        }
                        mailbox.publish(
                            lambda slot: mujoco.mj_copyData(
                                slot, model, env.plant.measurement_data), metadata)

                    snapshot(observation)
                    publish(0)
                    stable_render_pause("initial", initial_frame, initial_ack)
                    active_start_ns = time.perf_counter_ns()
                    active_start_monotonic_ns = time.monotonic_ns()
                    for tick in range(HORIZON):
                        if time.monotonic() - owner_started >= 95:
                            writer.request_stop()
                        if stop.is_set() or writer.stop_requested:
                            if runtime.thread_violations:
                                runtime._fatal = True
                            stop.set()
                            break
                        if tick in (174, 424, 429):
                            shared["requested_prepare_tick"] = tick + 1
                            input_request_done.clear()
                            input_request.set()
                            if not input_request_done.wait(timeout=5):
                                raise TimeoutError("GUI13 main logical request missing")
                            input_request.clear()
                        input_observation = np.asarray(observation, dtype=np.float32).copy()
                        action, predicted = _numeric_actor_action(
                            "final_policy", actor, observation)
                        if not predicted:
                            raise RuntimeError("GUI13 policy prediction missing")
                        predictions += 1
                        observation, reward, terminated, truncated, info = runtime.control_step(
                            env, action)
                        completed += 1
                        final_info = info
                        if tick in (174, 424, 429):
                            input_ack_done.clear()
                            input_ack.set()
                            if not input_ack_done.wait(timeout=5):
                                raise TimeoutError("GUI13 main logical acknowledgement missing")
                            input_ack.clear()
                        traces = env.plant.last_control_interval_actuator_traces
                        records.append({
                            "tick": tick, "input_observation99": input_observation,
                            "policy_input_action": action, "policy_predict_called": True,
                            "native_actuator_traces": [asdict(trace) for trace in traces],
                            "reward": float(reward), "terminated": terminated,
                            "truncated": truncated, "info": info})
                        snapshot(observation)
                        if completed % 5 == 0 or terminated or truncated:
                            publish(completed, info)
                        if (len(traces) != 5 or
                                info["native_interval_summary"]["native_returns"] != 5):
                            raise RuntimeError("GUI13 controller/native chain incomplete")
                        if (guard.max_abs_roll_deg > 10.0 or guard.max_abs_pitch_deg > 10.0
                                or info["native_interval_summary"]["nonwheel_contact_count"] != 0
                                or info["metrics"]["clearance_m"] < 0.28):
                            raise RuntimeError("GUI13 native posture/contact/clearance gate failed")
                        if terminated or truncated:
                            break
                    active_end_ns = time.perf_counter_ns()
                    active_end_monotonic_ns = time.monotonic_ns()
                    shared["owner_completed"] = completed
                    final_pending.set()
                    publish(completed, final_info)
                    stable_render_pause("final", final_frame, final_ack)
                    # The guard is compact train-record mode, never full-force qualification.
                    segment = guard.finish_segment()
                    runtime.unbind()
                    budget = _boundary(runtime)
                    c_state, python = budget["C_state"], budget["python"]
                    if (segment["record_valid"] is not True
                            or segment["native_returned"] != 5 * completed
                            or segment["native_attempted"] != 5 * completed
                            or guard.checked != 5 * completed
                            or runtime.control_completed != completed
                            or runtime.control_attempted != completed
                            or runtime.returned != 5 * completed
                            or runtime.attempted != runtime.returned
                            or c_state["construction_attempts"] != 2
                            or c_state["construction_returns"] != 2
                            or c_state["control_attempts"] != 5 * completed
                            or c_state["control_returns"] != 5 * completed
                            or c_state["violations"] != 0
                            or python["forbidden_entries"] != 0
                            or model_calls["load_attempted"] != 1
                            or model_calls["load_returned"] != 1
                            or model_calls["predict_attempted"] != predictions + 1
                            or model_calls["predict_returned"] != predictions + 1
                            or model_calls["learn_attempted"] != 0
                            or model_calls["save_attempted"] != 0):
                        raise RuntimeError("GUI13 C/Python/native/model budget differs")
                    writer.commit_gzip_rows(folder / "controls.jsonl.gz", [
                        common._jsonable(row) for row in records])
                    writer.commit_npz(folder / "states.npz", {
                        key: np.asarray(values) for key, values in states.items()})
                    prefix_saved = True
                    writer.commit_json(folder / "case_receipt.json", common._jsonable({
                        "completed_controls": completed, "policy_predictions": predictions,
                        "termination": None if final_info is None else final_info["terminal_reason"],
                        "logical_script_acks": schedule.acks,
                        "native_segment": segment, "boundary": _boundary(runtime),
                        "compact_native_only": True,
                        "full_contact_force_qualified": False}))
        except BaseException as error:  # noqa: BLE001 - preserve first physical/owner failure
            failure = {"type": type(error).__name__, "message": str(error)}
            stop.set()
        finally:
            if not prefix_saved and not writer.failed and states["time"]:
                folder = output / "case"
                if folder.is_dir():
                    try:
                        if not (folder / "controls.jsonl.gz").exists():
                            writer.commit_gzip_rows(folder / "controls.jsonl.gz", [
                                common._jsonable(row) for row in records])
                        if not (folder / "states.npz").exists():
                            writer.commit_npz(folder / "states.npz", {
                                key: np.asarray(values) for key, values in states.items()})
                        prefix_saved = True
                    except BaseException as archive_error:  # noqa: BLE001 - retain first failure
                        shared["secondary_prefix_error"] = repr(archive_error)
            if runtime is not None and runtime.thread_violations and failure is None:
                failure = {"type": "ThreadOwnershipError",
                           "message": "foreign thread called owner runtime"}
            if guard is not None and getattr(guard, "_segment", None) is not None:
                try:
                    segment = guard.finish_segment()
                except BaseException as archive_error:  # noqa: BLE001 - seal without masking first failure
                    if failure is None:
                        failure = {"type": type(archive_error).__name__,
                                   "message": str(archive_error)}
                    shared["secondary_archive_error"] = repr(archive_error)
            if runtime is not None and getattr(runtime, "active_plant", None) is not None:
                try:
                    runtime.unbind()
                except BaseException as unbind_error:  # noqa: BLE001 - preserve cleanup evidence
                    if failure is None:
                        failure = {"type": type(unbind_error).__name__,
                                   "message": str(unbind_error)}
            if not ready.is_set():
                ready.set()
            if model_hooks is not None:
                PPO, BaseAlgorithm, load_descriptor, old_predict, old_learn, old_save = model_hooks
                if load_descriptor is None:
                    delattr(PPO, "load")
                else:
                    PPO.load = load_descriptor
                BaseAlgorithm.predict = old_predict
                PPO.learn = old_learn
                BaseAlgorithm.save = old_save
            if warning_old is not None:
                mujoco.set_mju_user_warning(warning_old)
            try:
                result = {
                    "schema": SCHEMA, "arm": session["arm"], "failure": failure,
                    "completed_controls": completed, "policy_predictions": predictions,
                    "model_calls": model_calls,
                    "strict_load_probe_rows": None if actor is None else load_report.get("probe_rows"),
                    "pause_render_proofs": pause_proofs,
                    "active_start_perf_ns": active_start_ns,
                    "active_end_perf_ns": active_end_ns,
                    "active_start_monotonic_ns": locals().get("active_start_monotonic_ns"),
                    "active_end_monotonic_ns": locals().get("active_end_monotonic_ns"),
                    "saved_numeric_prefix": prefix_saved,
                    "active_wall_s": None if active_start_ns is None or active_end_ns is None
                    else (active_end_ns - active_start_ns) / 1e9,
                    "active_rtf": None if active_start_ns is None or active_end_ns is None
                    else completed * 0.01 / ((active_end_ns - active_start_ns) / 1e9),
                    "native_segment": segment,
                    "guard": None if guard is None else guard.report(),
                    "boundary": None if runtime is None else _boundary(runtime),
                    "thread_violations": [] if runtime is None else runtime.thread_violations,
                    "copy_edges": copy_edges, "warnings": warning_messages,
                    "writer_committed": list(writer.committed),
                    "writer_partial": list(writer.partial),
                    "original_1600_task_replayed": False,
                    "qualified_for_default_GUI": False,
                    "elapsed_wall_s": time.monotonic() - owner_started,
                }
                shared["owner_result"] = result
            except BaseException as error:  # noqa: BLE001 - report closure failure from owner
                shared["owner_error"] = repr(error)
            done.set()

    shared["main_ident"] = threading.get_ident()
    thread = threading.Thread(target=owner, name="gui13-numeric-owner", daemon=False)
    thread.start()
    if not ready.wait(timeout=30):
        stop.set()
        thread.join(timeout=5)
        raise TimeoutError("GUI13 owner did not initialize")
    viewer = None
    frames: list[dict] = []
    polls: list[int] = []
    logical_acks: list[dict] = []
    captured_mid = False
    last_display_control_index = -1
    main_error = None
    try:
        if "model" not in shared:
            raise RuntimeError("GUI13 owner failed before model setup")
        if session["arm"] == "gui":
            viewer = AsyncCourseRenderer(
                shared["model"], shared["display"], shared["mailbox"],
                key_codes=(), width=800, height=500, max_fps=15,
                title="D1 RL flat 0.6 | no proven RL benefit")
        start.set()
        while not done.wait(timeout=0.015):
            if input_request.is_set() and not input_request_done.is_set():
                prepared_tick = shared["requested_prepare_tick"]
                events = {175: ("W_PRESS",), 425: ("W_RELEASE", "S_PRESS"),
                          430: ("S_RELEASE",)}[prepared_tick]
                schedule.submit(prepared_tick, events)
                input_request_done.set()
            if input_ack.is_set() and not input_ack_done.is_set():
                ack = schedule.latest_ack()
                if ack is None or ack["prepared_tick"] != shared["requested_prepare_tick"]:
                    raise RuntimeError("GUI13 main saw another logical event ack")
                logical_acks.append(ack)
                input_ack_done.set()
            if initial_frame.is_set() and not initial_ack.is_set():
                if viewer is not None:
                    frame = viewer.render_latest(force=True,
                        capture_path=output / "frame_initial.png")
                    frames.append(frame)
                    last_display_control_index = frame["control_index"]
                else:
                    lease = shared["mailbox"].acquire_latest()
                    if lease is None:
                        raise RuntimeError("GUI13 initial snapshot missing")
                    try:
                        mujoco.mj_copyData(shared["display"], shared["model"], lease.data)
                        last_display_control_index = lease.metadata["control_index"]
                    finally:
                        lease.release()
                initial_ack.set()
            if final_frame.is_set() and not final_ack.is_set():
                if viewer is not None:
                    frame = viewer.render_latest(force=True,
                        capture_path=output / "frame_final.png")
                    frames.append(frame)
                    last_display_control_index = frame["control_index"]
                else:
                    lease = shared["mailbox"].acquire_latest()
                    if lease is None and pause_snapshot_satisfied(
                            shared.get("owner_completed"), last_display_control_index, lease):
                        final_ack.set()
                    elif lease is None:
                        raise RuntimeError("GUI13 final snapshot missing")
                    else:
                        try:
                            mujoco.mj_copyData(shared["display"], shared["model"], lease.data)
                            last_display_control_index = lease.metadata["control_index"]
                        finally:
                            lease.release()
                final_ack.set()
            if viewer is not None:
                event = viewer.poll()
                polls.append(event["wall_ns"])
                if event["window_close"] or not event["focused"]:
                    stop.set()
                capture = None
                if (not captured_mid and initial_ack.is_set()
                        and not final_pending.is_set()
                        and shared["mailbox"].snapshot_statistics()["published"] >= 51):
                    capture = output / "frame_drive.png"
                    captured_mid = True
                frame = viewer.render_latest(force=capture is not None,
                                             capture_path=capture)
                if frame["status"] != "skipped":
                    frames.append(frame)
                    last_display_control_index = frame["control_index"]
            elif initial_ack.is_set() and not final_pending.is_set():
                polls.append(time.monotonic_ns())
                lease = shared["mailbox"].acquire_latest()
                if lease is not None:
                    try:
                        mujoco.mj_copyData(shared["display"], shared["model"], lease.data)
                        last_display_control_index = lease.metadata["control_index"]
                    finally:
                        lease.release()
    except BaseException as error:  # noqa: BLE001 - stop owner on any renderer failure
        main_error = {"type": type(error).__name__, "message": str(error)}
        stop.set()
    finally:
        stop.set()
        start.set()
        thread.join(timeout=5)
        if not thread.is_alive() and shared.get("original_copy") is not None:
            mujoco.mj_copyData = shared["original_copy"]
        if viewer is not None:
            viewer.close()
        signal.signal(signal.SIGTERM, prior_term)
    if thread.is_alive():
        raise TimeoutError("GUI13 owner exceeded cleanup grace")
    if "owner_error" in shared:
        raise RuntimeError("GUI13 owner closure failed: " + shared["owner_error"])
    result = shared["owner_result"]
    if main_error is not None and result["failure"] is None:
        result["failure"] = main_error
    if not writer.failed:
        try:
            writer.commit_json(output / "render_receipt.json", common._jsonable({
                "arm": session["arm"], "frames": frames,
                "poll_timestamps_ns": polls, "logical_acks": logical_acks,
                "mailbox": shared.get("mailbox").snapshot_statistics()
                if shared.get("mailbox") is not None else None}))
        except BaseException as error:  # noqa: BLE001 - issue failure receipt for archive error
            if result["failure"] is None:
                result["failure"] = {"type": type(error).__name__, "message": str(error)}
    result["writer_committed"] = list(writer.committed)
    result["writer_partial"] = list(writer.partial)
    if writer.failed:
        if result["failure"] is None:
            result["failure"] = {"type": "ArchiveFailure", "message": "archive writer failed"}
        writer.commit_failure_receipt(output / "worker_failure_receipt.json",
                                      common._jsonable(result))
    else:
        try:
            writer.commit_json(output / "worker_receipt.json", common._jsonable(result))
        except BaseException as error:  # noqa: BLE001 - final receipt failure must be recorded
            if result["failure"] is None:
                result["failure"] = {"type": type(error).__name__,
                                     "message": str(error)}
            writer.commit_failure_receipt(output / "worker_failure_receipt.json",
                                          common._jsonable(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    path = args.session.resolve(strict=True)
    session = read_json(path)
    def preflight_timeout(_number, _frame):
        raise TimeoutError("GUI13 pure preflight exceeded 240 s")

    remaining = session["preflight_started_monotonic"] + 240 - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("GUI13 pure preflight deadline already expired")
    previous = signal.signal(signal.SIGALRM, preflight_timeout)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        output = preflight(session, path)
        # This barrier precedes import repair and every engine/model call.
        ready_ns = time.monotonic_ns()
        ready = {"pid": os.getpid(), "session_sha256": digest(path)["sha256"],
                 "monotonic_ns": ready_ns, "source_count": len(session["source_hashes"]),
                 "full_source_sha256_checked": True, "engine_imported": False,
                 "model_loaded": False}
        import json

        with (output / "preflight_ready.json").open("x", encoding="utf-8") as stream:
            json.dump(ready, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    result = run(session, output, execution_started=ready_ns / 1e9)
    return 0 if (result["failure"] is None and result["completed_controls"] == HORIZON
                 and result["policy_predictions"] == HORIZON
                 and result["native_segment"]["native_returned"] == NORMAL_NATIVE
                 and result["native_segment"]["native_attempted"] == NORMAL_NATIVE
                 and result["native_segment"]["failure"] is None
                 and result["native_segment"]["record_valid"] is True
                 and result["model_calls"] == {
                     "load_attempted": 1, "load_returned": 1,
                     "predict_attempted": 601, "predict_returned": 601,
                     "learn_attempted": 0, "save_attempted": 0}) else 1


if __name__ == "__main__":
    raise SystemExit(main())
