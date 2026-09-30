"""Stage-15 heldout recorder with archive13 transactions and unique actor labels.

The root owns the runtime, model, environment, native guard, budgets and calls.
This module starts none of them on import. The six-task scorer remains pure.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts import run_d1_latest_rl as common
from short_heldout_11 import HELDOUT_SCHEMA, WorldUprightHeldoutSchedule
from geometry21 import qualify_reset21, compare_reset_to_template21
from geometry_runtime22 import (capture_actual_reset_geometry22,
                                load_sealed_templates22)
from recipes22 import SPEC as SPEC22

RECORD_SCHEMA = "d1-world-upright-stage20-heldout-record-v1"
PAIR_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")
ACTORS = ("zero", "old", "A", "B")


def record_case(
    runtime: Any, guard: Any, env: Any, policy: Any,
    schedule: WorldUprightHeldoutSchedule, actor: str, folder: Path, *,
    writer: Any, pair_initial: dict[str, Any] | None,
    checkpoint_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Record one bounded physical case, preserving its first execution error.

    The caller passes the same ArchiveWriter already installed in
    AtomicCourseNativeGuard. A failed transaction latches that writer; no
    normal block is retried. A separate failure receipt is best effort only.
    """
    import numpy as np
    from archive13.atomic_archive_13 import ArchiveWriter, AtomicCourseNativeGuard
    from run_rl16_training_08 import (
        _boundary, _capture_state, _numeric_actor_action, _require_unchanged_reset,
    )
    from run_world_upright_reference_11 import _x_projection
    from world_upright_course_11 import (
        WORLD_UPRIGHT_REFERENCE_SCHEMA, WORLD_UPRIGHT_REWARD_SCHEMA,
        WORLD_UPRIGHT_TASK_SCHEMA, WorldUprightCourseEnv,
    )

    if (not isinstance(schedule, WorldUprightHeldoutSchedule)
            or actor not in ACTORS
            or type(env) is not WorldUprightCourseEnv
            or env.mode != "eval" or env.controller.mode != "eval"
            or not isinstance(writer, ArchiveWriter)
            or not isinstance(guard, AtomicCourseNativeGuard)
            or guard.writer is not writer or writer.failed):
        raise ValueError("stage-15 case requires the reviewed eval owner and archive")
    if actor == "zero":
        if policy is not None or checkpoint_sha256 is not None:
            raise ValueError("zero case cannot carry a policy checkpoint")
    elif (policy is None or type(checkpoint_sha256) is not str
          or len(checkpoint_sha256) != 64
          or any(c not in "0123456789abcdef" for c in checkpoint_sha256)):
        raise ValueError("continuation actor requires its frozen checkpoint SHA256")
    if not isinstance(folder, Path):
        raise TypeError("heldout output must be a Path")
    folder.mkdir(exist_ok=False)
    name = f"heldout_{schedule.case_id}_{actor}"
    cap = schedule.control_cap
    env.max_steps = cap
    score_actor = "zero" if actor == "zero" else "final_policy"
    states: dict[str, list[Any]] = {key: [] for key in (*PAIR_FIELDS, "time")}
    rows: list[dict[str, Any]] = []
    row_files: list[dict[str, Any]] = []
    completed = prediction_attempts = predictions = 0
    done = terminated = truncated = False
    final_info: dict[str, Any] | None = None
    first_error: BaseException | None = None
    stop_reason = "control_cap"
    initial: dict[str, Any] | None = None
    guard_segment: dict[str, Any] | None = None
    bound = guard_started = False
    started = time.monotonic()

    def remember(error: BaseException) -> None:
        nonlocal first_error, stop_reason
        if first_error is None:
            first_error = error
            stop_reason = "exception"

    def save_json(name: str, value: Any) -> None:
        writer.commit_json(folder / name, common._jsonable(value))

    def snapshot(observation: Any) -> None:
        item = _capture_state(env, observation)
        for field, values in states.items():
            values.append(item[field])

    def flush_rows() -> None:
        if not rows:
            return
        index = len(row_files)
        path = folder / f"control_records_{index:04d}.jsonl.gz"
        receipt = writer.commit_gzip_rows(
            path, [common._jsonable(row) for row in rows],
        )
        row_files.append({"file": path.name, "rows": len(rows),
                          "archive_manifest": receipt["manifest"],
                          **receipt["payloads"][0]})
        rows.clear()

    try:
        runtime.start_segment(name, cap)
        runtime.bind(env)
        bound = True
        guard.start_segment(env.plant, folder, name, cap, "heldout")
        guard_started = True
        save_json("boundary_before.json", _boundary(runtime))
        save_json("geometry_manifest.json", env.plant.collision_terrain_metadata)
        save_json("schedule.json", {
            "schema": schedule.schema, "record_schema": RECORD_SCHEMA,
            "case_id": schedule.case_id, "actor": actor,
            "scoring_actor": score_actor, "checkpoint_sha256": checkpoint_sha256,
            "seed": schedule.seed, "terrain": schedule.terrain,
            "spawn_position_m": schedule.spawn_position_m, "control_cap": cap,
            "raw_command_sha256": schedule.command_sha256,
            "raw_commands": [asdict(row) for row in schedule.raw_commands],
        })
        before_reset = _boundary(runtime)
        observation, reset_info = env.reset(
            seed=schedule.seed,
            options={"command_source": schedule.command_source,
                     "spawn_position_m": schedule.spawn_position_m},
        )
        after_reset = _boundary(runtime)
        _require_unchanged_reset(before_reset, after_reset)
        episode = reset_info["episode_metadata"]
        if (episode.get("task_schema") != WORLD_UPRIGHT_TASK_SCHEMA
                or episode.get("reward_schema") != WORLD_UPRIGHT_REWARD_SCHEMA
                or episode.get("reference_schema") != WORLD_UPRIGHT_REFERENCE_SCHEMA
                or episode.get("max_steps") != cap):
            raise RuntimeError("heldout reset has another task/reference/horizon")
        initial = _capture_state(env, observation)
        from state21 import control_state20
        initial["control_state20"] = control_state20(env)
        if pair_initial is not None:
            if any(
                initial[field].shape != pair_initial[field].shape
                or initial[field].dtype != pair_initial[field].dtype
                or initial[field].tobytes(order="C")
                != pair_initial[field].tobytes(order="C")
                for field in PAIR_FIELDS
            ):
                raise RuntimeError("matched initial states differ bitwise")
            if (int(env.plant.model._address), int(env.plant.data._address)) != pair_initial["identity"]:
                raise RuntimeError("matched case changed model/data identity")
        if pair_initial is not None and initial["control_state20"] != pair_initial["control_state20"]:
            raise RuntimeError("matched initial full controller states differ")
        initial["identity"] = (int(env.plant.model._address), int(env.plant.data._address))
        initial["actor"] = actor
        save_json("reset_receipt.json", {
            "before": before_reset, "after": after_reset, "seed": schedule.seed,
            "episode_metadata": episode,
            "control_reset_state": initial["control_state20"],
            "model_address": initial["identity"][0],
            "data_address": initial["identity"][1],
            "exact_initial_pair": pair_initial is not None,
            "paired_with": None if pair_initial is None else pair_initial["actor"],
        })
        writer.commit_npz(folder / "initial_state.npz",
                          {key: initial[key] for key in PAIR_FIELDS})
        before_geometry = _boundary(runtime)
        actual_geometry = capture_actual_reset_geometry22(env)
        save_json('geometry21_reset_manifest.json',actual_geometry)
        actual_initial = {key:initial[key] for key in (*PAIR_FIELDS,'time')}
        qualification,nominal_path = qualify_reset21(
            schedule,actual_geometry,actual_initial,kind='fixed_eval',
            case_id=schedule.case_id)
        templates,reference_controls,_ = load_sealed_templates22(SPEC22)
        expected_geometry,expected_initial = templates[schedule.terrain]
        bridge = compare_reset_to_template21(
            expected_geometry,actual_geometry,expected_initial,actual_initial,
            reference_controls[schedule.terrain],initial['control_state20'])
        save_json('geometry21_reset_homotopy.json',bridge)
        if not bridge['passed'] or _boundary(runtime)!=before_geometry:
            raise RuntimeError('actual heldout reset differs from finite source-derived template or counters')
        save_json('geometry21_receipt.json',qualification)
        writer.commit_npz(folder/'geometry21_nominal_path.npz',{'path_xy':nominal_path})
        snapshot(observation)
        for tick in range(cap):
            if writer.stop_requested:
                stop_reason = "soft_stop"
                break
            input_observation = np.asarray(observation, dtype=np.float32).copy()
            prediction_attempts += int(actor != "zero")
            action, predicted = _numeric_actor_action(score_actor, policy, observation)
            predictions += int(predicted)
            started_control = time.perf_counter_ns()
            observation, reward, terminated, truncated, info = runtime.control_step(env, action)
            elapsed_ns = time.perf_counter_ns() - started_control
            completed += 1
            final_info = info
            if terminated and truncated:
                raise RuntimeError("heldout cannot be both task-terminal and time-truncated")
            traces = env.plant.last_control_interval_actuator_traces
            if (len(traces) != 5 or info["native_interval_summary"]["native_returns"] != 5
                    or info.get("task_schema") != WORLD_UPRIGHT_TASK_SCHEMA
                    or info.get("reward_schema") != WORLD_UPRIGHT_REWARD_SCHEMA
                    or info.get("reference_schema") != WORLD_UPRIGHT_REFERENCE_SCHEMA):
                raise RuntimeError("heldout control lacks native or world-task evidence")
            calc = info["controller_record"]["calculation"]
            if (not np.array_equal(calc["safe_torque_nm"], env.last_transition.requested_torque_nm)
                    or not np.array_equal(info["applied_action"], calc["applied_action"])):
                raise RuntimeError("heldout controller-to-physical torque/action differs")
            rows.append({
                "tick": tick, "actor": actor, "scoring_actor": score_actor,
                "checkpoint_sha256": checkpoint_sha256,
                "policy_predict_called": predicted,
                "policy_input_action": action,
                "input_observation99": input_observation,
                "reward": float(reward), "terminated": terminated,
                "truncated": truncated, "control_wall_ns": elapsed_ns,
                "info": info,
                "native_actuator_traces": [asdict(trace) for trace in traces],
            })
            snapshot(observation)
            if len(rows) == 200:
                flush_rows()
            if terminated or truncated:
                done = True
                stop_reason = info["terminal_reason"]
                break
        if not done and completed == cap:
            raise RuntimeError("heldout cap exhausted without a genuine terminal transition")
        if schedule.terrain == "ramp" and done:
            save_json("final_compiled_x_projection.json", _x_projection(env.plant))
    except BaseException as error:
        remember(error)
    finally:
        if bound:
            try:
                runtime.unbind()
            except BaseException as error:
                remember(error)
        if guard_started:
            # A recorder-side transaction can fail before the guard's next
            # block flush. Latch that failure at the guard seam so its close
            # cannot attempt the same native block through a failed writer.
            if writer.failed and guard._archive_error is None:
                guard._archive_error = writer._first_error
            try:
                guard_segment = guard.finish_segment()
            except BaseException as error:
                remember(error)
        if not writer.failed:
            try:
                flush_rows()
                if states["time"]:
                    writer.commit_npz(folder / "states.npz",
                                      {key: np.asarray(values)
                                       for key, values in states.items()})
            except BaseException as error:
                remember(error)

        try:
            boundary_after = _boundary(runtime)
        except BaseException as error:
            remember(error)
            boundary_after = None
        failure = None if first_error is None else {
            "type": type(first_error).__name__, "message": str(first_error),
        }
        receipt = {
            "schema": RECORD_SCHEMA, "schedule_schema": HELDOUT_SCHEMA,
            "case_id": schedule.case_id, "actor": actor,
            "scoring_actor": score_actor, "checkpoint_sha256": checkpoint_sha256,
            "seed": schedule.seed, "terrain": schedule.terrain,
            "raw_command_sha256": schedule.command_sha256,
            "control_cap": cap, "completed_controls": completed,
            "policy_prediction_attempts": prediction_attempts,
            "policy_predictions": predictions, "termination_recorded": done,
            "terminated": terminated, "truncated": truncated,
            "stop_reason": stop_reason, "failure": failure,
            "record_valid": failure is None and not writer.failed and done
            and len(states["time"]) == completed + 1
            and guard_segment is not None and guard_segment["record_valid"]
            and guard_segment["native_attempted"] == 5 * completed
            and guard_segment["native_returned"] == 5 * completed
            and prediction_attempts == predictions == (completed if actor != "zero" else 0),
            "full_horizon_without_task_termination": (
                failure is None and not writer.failed and done
                and not terminated and truncated and completed == cap
            ),
            "final_compiled_x_projection": (
                "final_compiled_x_projection.json" if schedule.terrain == "ramp"
                and failure is None and done else None
            ),
            "initial_state_pair_exact": None if pair_initial is None else failure is None,
            "controller_record_blocks": row_files,
            "native_segment": guard_segment,
            "last_info_metrics": None if final_info is None else final_info.get("metrics"),
            "boundary_after": boundary_after,
            "elapsed_wall_s": time.monotonic() - started,
            "full_physical_qualification_pending_independent_scoring": True,
        }
        if not writer.failed:
            try:
                save_json("case_receipt.json", receipt)
            except BaseException as error:
                remember(error)
        if writer.failed:
            if first_error is None:
                remember(RuntimeError("heldout archive writer failed"))
            try:
                writer.commit_failure_receipt(folder / "archive_failure_receipt.json", {
                    "schema": RECORD_SCHEMA, "case_id": schedule.case_id,
                    "actor": actor, "completed_controls": completed,
                    "failure": {"type": type(first_error).__name__,
                                "message": str(first_error)},
                    "archive_error": repr(writer._first_error),
                    "run_failed": True,
                })
            except BaseException:
                pass  # The first error stays authoritative; the writer is latched.
    if first_error is not None:
        raise first_error
    if initial is None:
        raise RuntimeError("heldout reset did not save an initial state")
    return receipt, initial


__all__ = ("RECORD_SCHEMA", "PAIR_FIELDS", "ACTORS", "record_case")
