"""One prepaid latest-RL GUI or scripted validation session.

This entry is launched only by the stdlib-only exclusive parent. Imports that
can load MuJoCo, create a model or deserialize PPO are delayed until after the
session/source preflight and the reviewed OpenCV import repair. This is a new
interactive task; it does not claim to rerun the fixed 06 qualification cases.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

CONTRACT_SHA256 = "d90970745b938ede81341b308c1ba0dc5792e951cb84b54067d39a3509b3fc1f"
MODEL_SHA256 = "4f59d795afbdec256ec17bb7256ea05a3ddd04561e9ba52beffb1de375196b2e"
METADATA_SHA256 = "44b199ea13989412d274955a9fd85d6e97e58ad708a0c7e3ab7a4085ae3daaf3"
PROBE_SHA256 = "2cb674f54c8c46d5555e5d3580810a003be8d50fa5a7c1a5526dd0f027bb6660"
DSO_SHA256 = "3ec7ec9a6a130b9e1fa153c800aab97b59d4692c24971027f02de42957f8fa9c"
TRAINING_PROTOCOL_SHA256 = "494eb82ca80a842c190bbbff9bc75fa10c2e7832789a2d9bce0785acc3684733"
BODY_MODULE_SHA256 = {
    "body_speed_controller_06.py": "7b0a21302ef459d0a4d25a6db442d05d0d0305acc5f27e1ac48d67d5bd090d31",
    "body_speed_math_06.py": "adde9d1336bff274c0007174d7654968c1fd030b2dc44f2f17b339c708524aa6",
    "body_speed_transfer_env_06.py": "9af939ddcd6f3b2eb9715219b10326bdd9ffd0e061c2ad9de3a16f35bc82bfa1",
    "body_speed_validator_06.py": "5d9e6be948d0b3849c4fd9cd4edfa1e5e08203db8ca6acf5762b421f85409b98",
}
SESSION_SCHEMA = "d1-latest-rl-gui-session-v1"
COMMAND_SCHEMA = "d1-latest-rl-interactive-raw-command-v1"
SEGMENT_SCHEMA = "d1-latest-rl-interactive-segment-v1"
PROFILE_GUI = "gui_policy_box"
PROFILE_ZERO = "headless_zero_plane"
PAIR_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")


class TerminationRequested(RuntimeError):
    """Turn a launcher timeout into a durable partial-session boundary."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _save(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_row(stream: Any, value: Any) -> None:
    stream.write(json.dumps(_jsonable(value), sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def _jsonable(value: Any) -> Any:
    """Serialize NumPy-like records without importing an engine at preflight."""
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    return value


def _absolute(session: dict[str, Any], key: str) -> Path:
    value = session.get(key)
    if not isinstance(value, str):
        raise TypeError(f"session {key} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or str(path.resolve()) != value:
        raise ValueError(f"session {key} is not canonical and absolute")
    return path


def _preflight(session: dict[str, Any], session_path: Path) -> dict[str, Any]:
    if session.get("schema") != SESSION_SCHEMA or session.get("retry_permitted") is not False:
        raise ValueError("unknown or retryable GUI session")
    if _absolute(session, "session_path") != session_path.resolve():
        raise ValueError("session path differs from actual argv")
    output = _absolute(session, "output_directory")
    if not output.is_dir():
        raise ValueError("launcher must precreate the exclusive output directory")
    mode, actor, terrain = (session.get(key) for key in ("mode", "actor", "terrain"))
    if mode not in ("manual", "validation") or actor not in ("final_policy", "zero"):
        raise ValueError("unsupported GUI mode or actor")
    if terrain not in ("box", "plane"):
        raise ValueError("unsupported physical terrain")
    profile = session.get("validation_profile")
    if mode == "manual":
        if profile is not None or session.get("segments") != [1200, 1200]:
            raise ValueError("manual profile or segment reservation differs")
        expected = (2400, 12000, 3)
        if session.get("freeze_path") is not None or session.get("freeze_sha256") is not None:
            raise ValueError("ordinary manual invocation must not depend on old W archive")
    else:
        if (profile, actor, terrain) not in ((PROFILE_GUI, "final_policy", "box"),
                                             (PROFILE_ZERO, "zero", "plane")):
            raise ValueError("fixed validation profile identity differs")
        if session.get("segments") != [1000, 200]:
            raise ValueError("validation segment reservation differs")
        expected = (1200, 6000, 3)
    limits = tuple(session.get(key) for key in
                   ("control_limit", "normal_native_limit", "compiler_native_limit"))
    if limits != expected or any(type(value) is not int for value in limits):
        raise ValueError("physical budget differs from reviewed profile")
    if session.get("seed") != 77351 or session.get("render_quality") not in ("normal", "low"):
        raise ValueError("seed or rendering configuration differs")
    if session.get("contract_sha256") != CONTRACT_SHA256:
        raise ValueError("GUI contract identity differs")
    fixed = {
        "contract_path": CONTRACT_SHA256,
        "library": DSO_SHA256,
        "checkpoint_model": MODEL_SHA256,
        "checkpoint_metadata": METADATA_SHA256,
        "checkpoint_probe": PROBE_SHA256,
    }
    for key, digest in fixed.items():
        if _sha256(_absolute(session, key)) != digest:
            raise ValueError(f"frozen {key} hash differs")
    sources = session.get("source_hashes")
    if not isinstance(sources, dict) or len(sources) < 100:
        raise ValueError("published source closure is missing")
    for name, row in sources.items():
        path = Path(name)
        if (not path.is_absolute() or not isinstance(row, dict)
                or not path.is_file() or path.stat().st_size != row.get("bytes")
                or _sha256(path) != row.get("sha256")):
            raise ValueError(f"frozen source identity differs: {name}")
    required = (Path(__file__).resolve(), _absolute(session, "contract_path"),
                _absolute(session, "library"), _absolute(session, "checkpoint_model"),
                _absolute(session, "checkpoint_metadata"), _absolute(session, "checkpoint_probe"),
                _absolute(session, "binding_module"),
                _absolute(session, "continuation_import_repair"),
                Path(__file__).with_name("d1_latest_rl_controls.py"),
                Path(__file__).with_name("d1_latest_rl_viewer.py"))
    if any(str(path) not in sources for path in required):
        raise ValueError("critical runtime input absent from source freeze")
    published = _absolute(session, "published_body_source")
    for name, digest in BODY_MODULE_SHA256.items():
        path = published / name
        if sources.get(str(path), {}).get("sha256") != digest:
            raise ValueError(f"published 06 numerical module differs: {name}")
    freeze_path = session.get("freeze_path")
    if mode == "validation":
        path = _absolute(session, "freeze_path")
        if (_sha256(path) != session.get("freeze_sha256")
                or len(_load(path).get("files", {})) != 767):
            raise ValueError("historical 06 freeze differs")
        old_files = _load(path)["files"]
        if any(sources.get(name) != row for name, row in old_files.items()):
            raise ValueError("07 validation lacks an old 06 frozen input")
    elif freeze_path is not None:
        raise ValueError("manual historical freeze must be absent")
    environment = session.get("runtime_environment")
    if not isinstance(environment, dict):
        raise TypeError("runtime environment record is missing")
    for key, value in environment.items():
        actual = os.environ.get(key)
        if actual != value:
            raise ValueError(f"runtime environment differs: {key}")
    if "LD_LIBRARY_PATH" in os.environ or os.environ.get("LD_PRELOAD") != str(
        _absolute(session, "library")) or os.environ.get("LD_BIND_NOW") != "1":
        raise ValueError("isolated engine environment is not clean")
    return {"mode": mode, "actor": actor, "terrain": terrain, "profile": profile,
            "frozen_files": len(sources), "session_sha256": _sha256(session_path)}


class InteractiveCommandSource:
    """Replace only the command callback after strict load and 06 transfer."""

    def __init__(self, inner: Any, commands: Any):
        self.inner = inner
        self.commands = commands

    def __call__(self, time_s: float) -> Any:
        from scripts.d1_rolling_residual_task import CONTROL_DT_S, RAW_WORLD_HEIGHT_M
        from wheel_legged_control.d1.control_loop import D1MotionCommand

        tick = round(float(time_s) / CONTROL_DT_S)
        if (abs(float(time_s) - tick * CONTROL_DT_S) > 1e-10
                or tick != int(self.inner._steps) or not 0 <= tick <= 1200):
            raise RuntimeError("interactive raw callback tick/time mismatch")
        requested = self.commands.raw_forward_mps
        applied = (0.0 if tick < 175 or tick == 1200 else requested)
        ground_height = float(self.inner.loop.provider.ground_reference().height_m)
        clearance = RAW_WORLD_HEIGHT_M - ground_height
        self.inner._controller.controller.bind_raw_command(
            forward_velocity_mps=applied, yaw_rate_rps=0.0,
            control_time_s=float(time_s),
        )
        self.inner.raw_callback_count += 1
        self.inner.command_records.append({
            "schema": COMMAND_SCHEMA, "tick": tick, "time_s": float(time_s),
            "selected_speed_mps": self.commands.selected_speed_mps,
            "requested_forward_velocity_mps": requested,
            "forward_velocity_mps": applied, "yaw_rate_rps": 0.0,
            "raw_world_height_m": RAW_WORLD_HEIGHT_M,
            "ground_height_m": ground_height, "motion_clearance_m": clearance,
            "prepared_world_height_m": None,
            "settle_gate_active": tick < 175,
            "terminal_prepared_only": tick == 1200,
            "keyboard": self.commands.snapshot(),
        })
        return D1MotionCommand(applied, 0.0, clearance)


def _load_import_repair(session: dict[str, Any], output: Path) -> None:
    source = _absolute(session, "continuation_import_repair")
    spec = importlib.util.spec_from_file_location("published_07_import_repair", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("reviewed import-repair source is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arguments = argparse.Namespace(library=_absolute(session, "library"), output=output)
    module._check_and_restore_import_environment(arguments,
                                                 {"files": session["source_hashes"]})


def _check_origins(session: dict[str, Any]) -> dict[str, str]:
    import body_speed_controller_06
    import body_speed_math_06
    import body_speed_transfer_env_06
    import body_speed_validator_06
    import drive_damping_controller_04
    import drive_damping_math_04
    import drive_damping_validator_04
    import drive_fresh_lifecycle_05
    import engine_binding

    source = _absolute(session, "published_body_source")
    module_names = (body_speed_controller_06, body_speed_math_06,
                    body_speed_transfer_env_06, body_speed_validator_06,
                    drive_damping_controller_04, drive_damping_math_04,
                    drive_damping_validator_04, drive_fresh_lifecycle_05)
    expected = {module.__name__: source / f"{module.__name__}.py" for module in module_names}
    expected["engine_binding"] = _absolute(session, "binding_module")
    if Path(engine_binding.__file__).resolve() != expected["engine_binding"]:
        raise RuntimeError("published binding module origin differs")
    if any(Path(sys.modules[name].__file__).resolve() != path for name, path in expected.items()):
        raise RuntimeError("published numerical/binding module origin differs")
    return {name: str(path) for name, path in expected.items()}


def _segment(
    session: dict[str, Any], runtime: Any, env: Any, actor: Any,
    commands: Any, viewer: Any, driver: Any, output: Path,
    *, index: int, limit: int, monitor: Any,
    reset_reference: dict[str, Any], predict_stats: dict[str, int],
) -> dict[str, Any]:
    """One real reset and at most ``limit`` actual control transitions."""
    import numpy as np

    from scripts.d1_jump_readiness_records import ScratchKinematics
    from scripts.d1_single_step_records import StreamingNativeObserver, compiled_geometry_manifest
    from scripts.probe_d1_single_step import endpoint_record

    name = f"segment_{index:02d}"
    folder = output / name
    folder.mkdir(exist_ok=False)
    runtime.start_segment(name, limit)
    runtime.bind(env)
    before_c = runtime.state()
    before_ledger = runtime.ledger_receipt()
    inner = env.unwrapped
    plant = inner.plant
    _save(folder / "boundary_before.json", {
        "schema": SEGMENT_SCHEMA, "segment": name, "limit": limit,
        "C_state": before_c, "python_ledger": before_ledger,
        "native_monitor_checked_before": monitor.checked,
        "same_model_address": int(plant.model._address),
        "same_data_address": int(plant.data._address),
    })
    geometry = compiled_geometry_manifest(plant)
    geometry["joint_qpos_addresses"] = plant.qpos_addresses.tolist()
    geometry["joint_dof_addresses"] = plant.dof_addresses.tolist()
    geometry["actual_compiled_joint_qpos_addresses"] = plant.qpos_addresses.tolist()
    geometry["actual_compiled_joint_dof_addresses"] = plant.dof_addresses.tolist()
    geometry["compiled_model_address"] = int(plant.model._address)
    geometry["live_data_address"] = int(plant.data._address)
    _save(folder / "geometry_manifest.json", geometry)
    body_ids = list(range(1, plant.model.nbody))
    scratch = ScratchKinematics(
        plant.model, body_ids=body_ids, masses=plant.model.body_mass[body_ids],
        base_body_id=plant.base_body_id,
    )
    observer = StreamingNativeObserver(plant, folder)
    arrays: dict[str, list[Any]] = {key: [] for key in (*PAIR_FIELDS, "time")}
    recorded = 0
    completed = 0
    policy_calls_before = predict_stats["attempted"]
    stopped = "limit"
    terminated = truncated = False
    error = None
    viewer_events_before = 0 if viewer is None else len(viewer.events)
    frame_before = 0 if viewer is None else len(viewer.frame_receipts)
    screenshots_before = 0 if viewer is None else len(viewer.screenshot_receipts)

    def snapshot(observation: Any, tick: int, stream: Any) -> None:
        for key in PAIR_FIELDS[:-1]:
            arrays[key].append(getattr(plant.data, key).copy())
        arrays["observation"].append(np.asarray(observation, dtype=np.float32).copy())
        arrays["time"].append(float(plant.data.time))
        _write_row(stream, endpoint_record(inner, scratch, tick))

    def display(observation: Any, tick: int, raw: dict[str, Any] | None) -> None:
        if viewer is None:
            return
        from wheel_legged_control.d1.control_loop import D1MotionCommand

        actual = float(inner.heading_decision.decision.context.state.base_linear_velocity_body[0])
        prepared = 0.0 if raw is None else float(raw["forward_velocity_mps"])
        selected = commands.selected_speed_mps
        scalar = (None if env.env.last_rolling_action is None else
                  env.env.last_rolling_action["policy_scalar"])
        physical = (None if env.env.last_rolling_action is None else
                    env.env.last_rolling_action["physical_action"][0])
        viewer.set_status(D1MotionCommand(prepared, 0.0, 0.455), tick * 0.01)
        viewer.extra_help = (
            f"ORACLE | {'LEARNED RESIDUAL ON' if session['actor'] == 'final_policy' else 'ZERO RESIDUAL'}"
            " | 06 BODY SPEED CONTROLLER ON\n"
            f"Selected {selected:.2f} | requested {commands.raw_forward_mps:.2f} | "
            f"prepared {prepared:.2f} | observed COM {actual:+.3f} | "
            f"error {actual-prepared:+.3f} m/s\n"
            f"policy scalar {scalar!r} | applied leg residual {physical!r} | "
            f"checkpoint {MODEL_SHA256[:12]} | engine bound {runtime.proof['passed']}\n"
            f"{session['terrain']} | episode {index+1} tick {tick} | "
            f"total control {runtime.control_completed}/{session['control_limit']} | "
            f"native {runtime.returned}/{session['normal_native_limit']} | "
            f"compiler {runtime.state()['construction_returns']}/3 | "
            f"remaining {session['control_limit']-runtime.control_completed}\n"
            f"{'SETTLING' if tick < 175 else 'PAUSED' if stopped != 'limit' else 'ACTIVE'}"
            f" | output {output} | R simulation reset, Esc save/exit"
        )
        viewer.copy_and_sync(plant, runtime, source_tick=tick)

    try:
        with (gzip.open(folder / "trace.jsonl.gz", "xt") as trace_stream,
              gzip.open(folder / "endpoints.jsonl.gz", "xt") as endpoint_stream,
              gzip.open(folder / "keyboard_polls.jsonl.gz", "xt") as poll_stream,
              observer):
            observation, reset_info = env.reset(seed=session["seed"])
            _save(folder / "episode_metadata.json", reset_info)
            if tuple(np.asarray(observation).shape) != (85,):
                raise RuntimeError("interactive reset did not produce 85D observation")
            reset_c = runtime.state()
            reset_ledger = runtime.ledger_receipt()
            frozen_c = ("construction_attempts", "construction_returns",
                        "control_attempts", "control_returns")
            frozen_python = ("native_attempted", "native_returned",
                             "control_attempted", "control_completed")
            reset_budget_unchanged = (
                all(reset_c[key] == before_c[key] for key in frozen_c)
                and all(reset_ledger[key] == before_ledger[key] for key in frozen_python)
                and all(reset_c[key] >= before_c[key] for key in reset_c
                        if isinstance(reset_c[key], int) and key in before_c)
            )
            initial = {
                "qpos": plant.data.qpos.copy(), "qvel": plant.data.qvel.copy(),
                "ctrl": plant.data.ctrl.copy(),
                "qacc_warmstart": plant.data.qacc_warmstart.copy(),
                "observation": np.asarray(observation, dtype=np.float32).copy(),
            }
            address_pair = (int(plant.model._address), int(plant.data._address))
            first = not reset_reference
            if first:
                reset_reference.update({"arrays": initial, "addresses": address_pair})
            exact_pair = (address_pair == reset_reference["addresses"] and all(
                value.dtype == reset_reference["arrays"][key].dtype
                and value.shape == reset_reference["arrays"][key].shape
                and value.tobytes() == reset_reference["arrays"][key].tobytes()
                for key, value in initial.items()
            ))
            np.savez_compressed(folder / "initial_state.npz", **initial)
            _save(folder / "reset_pair_receipt.json", {
                "schema": SEGMENT_SCHEMA, "segment": name, "seed": session["seed"],
                "first_segment_reference": first, "exact_initial_pair": exact_pair,
                "model_address": address_pair[0], "data_address": address_pair[1],
                "array_sha256": {
                    key: hashlib.sha256(value.tobytes()).hexdigest()
                    for key, value in initial.items()
                },
                "observation_shape": list(initial["observation"].shape),
                "C_before_reset": before_c, "C_after_reset": reset_c,
                "python_before_reset": before_ledger,
                "python_after_reset": reset_ledger,
                "control_native_compiler_budget_unchanged": reset_budget_unchanged,
            })
            if not exact_pair or not reset_budget_unchanged:
                raise RuntimeError("interactive reset changed initial pair or spent control budget")
            snapshot(observation, 0, endpoint_stream)
            if viewer is not None and index == 1:
                viewer._last_render = -math.inf
                viewer.request_capture("after_reset", folder / "after_reset.png")
            display(observation, 0, dict(inner.command_records[-1]))

            while True:
                tick = completed
                wall_start = time.monotonic()
                if driver is not None:
                    driver.before_poll(index, tick)
                if viewer is not None:
                    viewer.poll(tick * 0.01)
                if driver is not None:
                    driver.after_poll(index, tick)
                _write_row(poll_stream, {
                    "event": "keyboard_poll", "tick": tick,
                    "keyboard": commands.snapshot(),
                    "origin": "scripted_acceptance" if driver else "human_window",
                })
                if commands.stopped or (viewer is not None and not viewer.is_running()):
                    stopped = "keyboard_escape" if commands.stopped else "viewer_closed"
                    break
                if commands.consume_reset_request():
                    if index == 0:
                        stopped = "operator_simulation_reset"
                    else:
                        stopped = "reset_reservation_exhausted"
                    break
                if completed >= limit or terminated or truncated:
                    if session["mode"] == "validation":
                        raise RuntimeError("scripted profile did not exit at segment boundary")
                    if index == 1 and completed >= limit:
                        stopped = "global_budget_exhausted"
                        break
                    stopped = "terminal_pause" if (terminated or truncated) else "budget_pause"
                    display(observation, completed, dict(inner.command_records[-1]))
                    time.sleep(0.01)
                    continue
                raw = dict(inner.command_records[-1])
                decision = inner.heading_decision
                if decision is None or decision.tick != tick or raw["tick"] != tick:
                    raise RuntimeError("prepared interactive command is not bound to this tick")
                observation_before_sha256 = hashlib.sha256(
                    np.asarray(observation, dtype=np.float32).tobytes()
                ).hexdigest()
                if session["actor"] == "zero":
                    action = np.zeros(1, dtype=np.float32)
                else:
                    predict_stats["attempted"] += 1
                    action = np.asarray(actor.predict(observation, deterministic=True)[0],
                                        dtype=np.float32)
                    predict_stats["returned"] += 1
                if action.shape != (1,) or not np.isfinite(action).all():
                    raise RuntimeError("live deterministic scalar policy output is invalid")
                observation, reward, terminated, truncated, info = runtime.control_step(env, action)
                completed += 1
                control = inner._controller.controller
                receipt = info["rolling_action"]
                row = {
                    "schema": SEGMENT_SCHEMA, "tick": tick, "endpoint_tick": completed,
                    "actor": session["actor"], "policy_action": info["policy_action"],
                    "policy_predict_called": session["actor"] == "final_policy",
                    "observation_before_sha256": observation_before_sha256,
                    "clipped_policy_scalar": receipt["clipped_scalar"],
                    "forward_gate_active": receipt["forward_gate_active"],
                    "action": info["physical_action"],
                    "applied_action": info["applied_action"],
                    "raw_command": raw,
                    "servo_command": asdict(decision.servo_command),
                    "world_command": asdict(decision.decision.world_command),
                    "stop_active": control.last_damping.active,
                    "turn_active": control.last_authority.active,
                    "torque_nm": inner.last_transition.requested_torque_nm,
                    "callback_count_after_prepare": inner.raw_callback_count,
                    "stop_record": asdict(control.last_damping),
                    "authority_record": asdict(control.last_authority),
                    "controller_result": asdict(control.last_result),
                    "reward": reward, "reward_terms": info.get("reward_terms"),
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "terminal_reason": info.get("terminal_reason"),
                    "metrics": info.get("metrics"), "checkpoint_sha256": MODEL_SHA256,
                    "keyboard_after_poll": commands.snapshot(),
                    "request_to_execution_timing": "poll(k) changes prepared command at k+1",
                }
                _write_row(trace_stream, row)
                recorded += 1
                snapshot(observation, completed, endpoint_stream)
                if viewer is not None and index == 0 and completed in (200, 650):
                    viewer.request_capture(f"drive_{completed}", folder / f"drive_{completed}.png")
                display(observation, completed, dict(inner.command_records[-1]))
                if (terminated or truncated) and session["mode"] == "validation":
                    raise RuntimeError("scripted acceptance terminated before its fixed boundary")
                if session["mode"] == "manual":
                    time.sleep(max(0.0, 0.01 - (time.monotonic() - wall_start)))
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        stopped = "error"
        raise
    finally:
        runtime.unbind()
        np.savez_compressed(folder / "states.npz", **{
            key: np.asarray(value) for key, value in arrays.items()
        })
        np.savez_compressed(
            folder / "terminal_integrator.npz",
            qpos=plant.data.qpos, qvel=plant.data.qvel, ctrl=plant.data.ctrl,
            qacc_warmstart=plant.data.qacc_warmstart, time=np.asarray(plant.data.time),
        )
        _save(folder / "pre_control_states.json", env.pre_control_states)
        _save(folder / "command_records.json", inner.command_records)
        if viewer is not None:
            _save(folder / "keyboard_events.json", viewer.events[viewer_events_before:])
            _save(folder / "render_frames.json", viewer.frame_receipts[frame_before:])
            _save(folder / "screenshots.json", viewer.screenshot_receipts[screenshots_before:])
        after_c = runtime.state()
        after_ledger = runtime.ledger_receipt()
        _save(folder / "boundary_after.json", {
            "schema": SEGMENT_SCHEMA, "segment": name, "stop_reason": stopped,
            "error": error, "completed_controls": completed,
            "recorded_control_rows": recorded, "terminated": bool(terminated),
            "truncated": bool(truncated),
            "live_policy_predict_calls": predict_stats["attempted"] - policy_calls_before,
            "strict_loader_probe_observations_session_total": 2,
            "strict_loader_predict_batch_calls_session_total": 1,
            "C_state": after_c,
            "python_ledger": after_ledger, "native_observer": observer.receipt(),
            "native_monitor_checked_after": monitor.checked,
            "same_model_address": int(plant.model._address),
            "same_data_address": int(plant.data._address),
            "scratch_kinematics": scratch.receipt(),
            "no_retry_or_padding": True,
        })
    return {"segment": name, "completed_controls": completed,
            "stop_reason": stopped, "terminated": bool(terminated),
            "truncated": bool(truncated), "native_attempted": observer.attempted_calls,
            "native_returned": observer.returned_calls,
            "recorded_control_rows": recorded,
            "live_policy_predict_calls": predict_stats["attempted"] - policy_calls_before,
            "error": error}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args(argv)
    session_path = args.session.resolve(strict=True)
    session = _load(session_path)
    preflight = _preflight(session, session_path)
    output = _absolute(session, "output_directory")
    _save(output / "worker_preflight.json", {
        "schema": SESSION_SCHEMA, **preflight,
        "before_engine_or_model_import": True,
        "contract_sha256": CONTRACT_SHA256,
    })

    # Add only the already hash-checked, published modules. Site-packages must
    # remain first in PYTHONPATH for the frozen MuJoCo/extension GOT proof.
    for path in (_absolute(session, "published_body_source"),
                 _absolute(session, "binding_module").parent):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    _load_import_repair(session, output)

    # Engine-capable imports occur only after pure preflight/import repair.
    import mujoco
    from body_speed_transfer_env_06 import BodyCommonPTransferredEnv, install_body_controller

    from scripts.d1_latest_rl_controls import LatestRLCommands
    from scripts.d1_latest_rl_viewer import LatestRLViewer
    from scripts.d1_rolling_engine_runtime import RollingEngineRuntime
    from scripts.d1_rolling_native_monitor import NativeGeometryGuard
    from scripts.d1_rolling_residual_checkpoint import load_final_checkpoint
    from scripts.d1_rolling_residual_env import D1RollingResidualEnv
    from scripts.d1_rolling_residual_task import RollingEpisodeSpec
    from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry

    origins = _check_origins(session)
    _save(output / "module_origins.json", {
        "schema": SESSION_SCHEMA, "published_numerical_and_binding_sources": origins,
        "body_numerics_identical_to_qualified_06": True,
    })
    if _nominal_geometry.cache_info()._asdict() != {
            "hits": 0, "misses": 0, "maxsize": 1, "currsize": 0}:
        raise RuntimeError("nominal geometry cache was not cold before model construction")

    class InteractiveTransferredEnv(BodyCommonPTransferredEnv):
        task_schema = "d1-latest-rl-straight-interactive-v1"

        @property
        def episode_metadata(self) -> dict[str, Any]:
            metadata = dict(super().episode_metadata)
            metadata["historical_fixed_raw_schedule"] = metadata.pop("raw_schedule", None)
            metadata["historical_fixed_episode_spec"] = metadata.pop("rolling_episode_spec", None)
            metadata["historical_06_task_definition"] = metadata.pop("rolling_task_definition", None)
            metadata["task_schema"] = self.task_schema
            metadata["rolling_task_definition"] = {
                "schema": self.task_schema,
                "command_source": COMMAND_SCHEMA,
                "command_set_mps": [0.0, 0.20, 0.25],
                "settle_ticks": 175, "terminal_prepared_tick": 1200,
                "requested_world_height_m": 0.455, "yaw_request_rps": 0.0,
                "input_timing": "poll before k; changed intent prepared after control k for k+1",
                "actor": session["actor"], "controller_schema": metadata["controller_schema"],
                "action_schema": metadata["action_schema"],
                "observation_schema": metadata["observation_schema"],
                "physical_terrain": session["terrain"],
                "simulation_reset_only": True,
            }
            metadata["interactive_actor"] = session["actor"]
            metadata["interactive_input_origin"] = (
                "scripted_acceptance" if session["mode"] == "validation" else "human_window"
            )
            return metadata

    warning_messages: list[str] = []

    def warning(message: str) -> None:
        warning_messages.append(str(message))
        raise RuntimeError("MuJoCo warning: " + str(message))

    previous_warning = mujoco.get_mju_user_warning()
    runtime = monitor = viewer = driver = None
    segments: list[dict[str, Any]] = []
    strict_loader_probe_native_delta = None
    cache_construct = cache_transfer = None
    original = transferred = loaded = None
    failure = None
    C_final = ledger_final = monitor_final = None
    reset_reference: dict[str, Any] = {}
    predict_stats = {"attempted": 0, "returned": 0}
    started = time.monotonic()
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def on_sigterm(_number: int, _frame: Any) -> None:
        raise TerminationRequested("launcher sent SIGTERM; saving partial session")

    signal.signal(signal.SIGTERM, on_sigterm)
    try:
        mujoco.set_mju_user_warning(warning)
        with RollingEngineRuntime(
            _absolute(session, "library"), control_limit=session["control_limit"],
            construction_limit=session["compiler_native_limit"],
        ) as runtime:
            _save(output / "runtime_initial.json", {
                "schema": SESSION_SCHEMA, "binding_proof": runtime.proof,
                "initial_C_state": runtime.state(),
                "initial_python_ledger": runtime.ledger_receipt(),
                "before_model_construction": True,
            })
            with NativeGeometryGuard(runtime, output) as monitor:
                initial_spec = RollingEpisodeSpec(
                    0.20, 175, 975, session["terrain"] == "box"
                )
                original = runtime.construct(lambda: D1RollingResidualEnv(initial_spec))
                cache_construct = _nominal_geometry.cache_info()._asdict()
                after_construction = runtime.state()
                if (after_construction["construction_attempts"] != 3
                        or after_construction["construction_returns"] != 3
                        or cache_construct != {"hits": 1, "misses": 1,
                                               "maxsize": 1, "currsize": 1}):
                    raise RuntimeError("one cold original environment did not consume 3 compiler steps")
                _save(output / "construction_receipt.json", {
                    "schema": SESSION_SCHEMA, "C_state": after_construction,
                    "python_ledger": runtime.ledger_receipt(),
                    "cache": cache_construct, "terrain": session["terrain"],
                    "one_selected_environment_and_active_plant": True,
                    "compiler_hidden_native_steps": 3,
                })
                prior_loader = runtime.ledger_receipt()["native_returned"]
                loaded = load_final_checkpoint(
                    _absolute(session, "checkpoint_model"),
                    _absolute(session, "checkpoint_metadata"), original,
                    expected_dso_sha256=DSO_SHA256,
                    expected_training_protocol_sha256=TRAINING_PROTOCOL_SHA256,
                )
                strict_loader_probe_native_delta = (
                    runtime.ledger_receipt()["native_returned"] - prior_loader
                )
                if strict_loader_probe_native_delta != 0 or runtime.state() != after_construction:
                    raise RuntimeError("strict checkpoint load changed physical state/counts")
                _save(output / "strict_original_checkpoint_reload.json", {
                    "schema": SESSION_SCHEMA, "model_sha256": MODEL_SHA256,
                    "metadata_sha256": METADATA_SHA256, "probe_sha256": PROBE_SHA256,
                    "old_schema_checked_before_transfer": True,
                    "two_saved_probe_observations_checked_in_one_batch": True,
                    "live_policy_predict_calls": 0,
                    "native_return_delta": strict_loader_probe_native_delta,
                })
                transfer_receipt = install_body_controller(original)
                cache_transfer = _nominal_geometry.cache_info()._asdict()
                if (runtime.state() != after_construction
                        or runtime.ledger_receipt()["native_returned"] != prior_loader
                        or cache_transfer["misses"] != 1):
                    raise RuntimeError("06 controller transfer changed engine/cache identity")
                transferred = InteractiveTransferredEnv(original, transfer_receipt)
                commands = LatestRLCommands()
                commands.select_initial_speed(float(session.get("speed_mps", 0.20)))
                inner = transferred.unwrapped
                inner.command_source = InteractiveCommandSource(inner, commands)
                _save(output / "controller_law_transfer.json", {
                    "schema": SESSION_SCHEMA, "receipt": transfer_receipt,
                    "cache_after_original_construct": cache_construct,
                    "cache_after_transfer": cache_transfer,
                    "same_C_state_after_loader_and_transfer": True,
                    "interactive_callback_installed_after_strict_loader": True,
                })

                gui = session["mode"] == "manual" or session.get("validation_profile") == PROFILE_GUI
                if gui:
                    display_data = mujoco.MjData(inner.plant.model)
                    viewer = LatestRLViewer(
                        inner.plant.model, display_data, commands,
                        terrain_label="15 mm box" if session["terrain"] == "box" else "plane",
                        render_quality=session["render_quality"],
                    )
                if session["mode"] == "validation":
                    from scripts.d1_latest_rl_validation import ValidationDriver

                    driver = ValidationDriver(session["validation_profile"], commands,
                                              viewer=viewer)
                try:
                    with (viewer if viewer is not None else nullcontext()):
                        for index, limit in enumerate(session["segments"]):
                            if index == 1:
                                if not segments or segments[0]["stop_reason"] != "operator_simulation_reset":
                                    break
                                commands.reset_for_new_segment()
                            row = _segment(
                                session, runtime, transferred, loaded, commands, viewer, driver,
                                output, index=index, limit=limit, monitor=monitor,
                                reset_reference=reset_reference, predict_stats=predict_stats,
                            )
                            segments.append(row)
                            print(f"07 {row['segment']}: {row['completed_controls']}/{limit} "
                                  f"controls, {row['stop_reason']}", flush=True)
                            if row["stop_reason"] != "operator_simulation_reset":
                                break
                finally:
                    if driver is not None:
                        _save(output / "validation_driver_receipt.json", driver.report())
                        driver.close()
                    if viewer is not None:
                        _save(output / "render_receipt.json", {
                            "schema": SESSION_SCHEMA,
                            "rendered_frames": viewer.rendered_frames,
                            "skipped_frames": viewer.skipped_frames,
                            "frames": viewer.frame_receipts,
                            "screenshots": viewer.screenshot_receipts,
                            "no_render_step_forward_collision": True,
                        })
                runtime.unbind()
                C_final = runtime.state()
                ledger_final = runtime.ledger_receipt()
                monitor_final = monitor.report()
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        mujoco.set_mju_user_warning(previous_warning)
        if runtime is not None:
            try:
                runtime.unbind()
                C_final = runtime.state()
                ledger_final = runtime.ledger_receipt()
            except BaseException as exc:  # noqa: BLE001 - preserve original failure
                if failure is None:
                    failure = {"type": type(exc).__name__, "message": str(exc)}
        if monitor is not None:
            monitor_final = monitor.report()
        scored_controls = sum(row["completed_controls"] for row in segments)
        completed = None if runtime is None else runtime.control_completed
        attempted = None if runtime is None else runtime.control_attempted
        execution_complete = (failure is None and all(row["error"] is None for row in segments)
                              and completed == scored_controls
                              and (session["mode"] == "manual" or (
                                  len(segments) == 2
                                  and [row["completed_controls"] for row in segments] == [1000, 200]
                                  and [row["stop_reason"] for row in segments]
                                  == ["operator_simulation_reset", "keyboard_escape"])))
        _save(output / "worker_receipt.json", {
            "schema": SESSION_SCHEMA, "mode": session["mode"],
            "actor": session["actor"], "terrain": session["terrain"],
            "input_origin": "scripted_acceptance" if driver else "human_window",
            "segments": segments, "completed_controls": completed,
            "attempted_controls": attempted,
            "closed_segment_control_rows": scored_controls,
            "live_policy_predict_calls": predict_stats["attempted"],
            "live_policy_predict_returns": predict_stats["returned"],
            "strict_loader_probe_observations_session_total": 2 if loaded is not None else None,
            "strict_loader_predict_batch_calls_session_total": 1 if loaded is not None else None,
            "strict_loader_native_delta": strict_loader_probe_native_delta,
            "cache_after_construct": cache_construct, "cache_after_transfer": cache_transfer,
            "reset_pair_receipts_saved": sum(
                (output / f"segment_{index:02d}" / "reset_pair_receipt.json").exists()
                for index in range(2)
            ),
            "reset_pair_reference_available": bool(reset_reference),
            "C_final": C_final, "python_ledger_final": ledger_final,
            "native_monitor": monitor_final, "warnings": warning_messages,
            "execution_complete": execution_complete,
            "independent_readback_pending": True,
            "failure": failure, "elapsed_wall_s": time.monotonic() - started,
            "no_training_or_policy_update": True,
            "not_a_replay_of_fixed_06_qualification": True,
        })
    return 0 if execution_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
