"""Typed world-upright heldout schedule and one bounded physical recorder.

The case table, seeds, speeds and yaw requests must come from a later frozen
contract. Nothing in this module selects a case or starts an engine process.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts import run_d1_latest_rl as common

HELDOUT_SCHEMA = "d1-world-upright-short-rl16-heldout-v1"
PAIR_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "observation")


@dataclass(frozen=True, slots=True)
class WorldUprightHeldoutSchedule:
    """One exact, preregistered 1600- or 1800-control raw command sequence."""

    case_id: str
    terrain: str
    seed: int
    spawn_position_m: tuple[float, float, float]
    raw_commands: tuple[Any, ...]
    schema: str = HELDOUT_SCHEMA

    def __post_init__(self) -> None:
        from full_drive_command_08 import FullDriveCommand
        from full_drive_schedule_08 import COURSE_SPAWNS_M

        if (type(self.case_id) is not str or not self.case_id
                or self.terrain not in COURSE_SPAWNS_M
                or tuple(self.spawn_position_m) != COURSE_SPAWNS_M[self.terrain]
                or type(self.seed) is not int or not 0 <= self.seed < 2**32
                or type(self.raw_commands) is not tuple
                or len(self.raw_commands) not in (1600, 1800)
                or any(not isinstance(row, FullDriveCommand) for row in self.raw_commands)
                or self.schema != HELDOUT_SCHEMA):
            raise ValueError("heldout schedule differs from its reviewed course identity")

    @property
    def control_cap(self) -> int:
        return len(self.raw_commands)

    @property
    def command_sha256(self) -> str:
        rows = [asdict(row) for row in self.raw_commands]
        return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()

    def command_source(self, tick: int, time_s: float) -> Any:
        import math

        if (type(tick) is not int or not 0 <= tick < self.control_cap
                or type(time_s) not in (int, float)
                or not math.isclose(time_s, tick * 0.01, rel_tol=0.0, abs_tol=1e-10)):
            raise ValueError("heldout requested a tick outside its frozen sequence")
        return self.raw_commands[tick]


def record_heldout_case(
    runtime: Any, guard: Any, env: Any, policy: Any,
    schedule: WorldUprightHeldoutSchedule, actor: str, folder: Path, *,
    pair_initial: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Save every actual endpoint and controller/native chain for one case."""
    import numpy as np
    from run_rl16_training_08 import (
        _boundary,
        _capture_state,
        _numeric_actor_action,
        _require_unchanged_reset,
        _save_arrays,
        _sha_bytes,
        _write_gzip_rows,
    )
    from run_world_upright_reference_11 import _x_projection
    from world_upright_course_11 import (
        WORLD_UPRIGHT_REFERENCE_SCHEMA,
        WORLD_UPRIGHT_REWARD_SCHEMA,
        WORLD_UPRIGHT_TASK_SCHEMA,
        WorldUprightCourseEnv,
    )

    if (not isinstance(schedule, WorldUprightHeldoutSchedule)
            or actor not in ("zero", "final_policy")
            or type(env) is not WorldUprightCourseEnv
            or env.mode != "eval" or env.controller.mode != "eval"):
        raise ValueError("heldout requires the reviewed world-upright env and matched actor")
    if not isinstance(folder, Path):
        raise TypeError("heldout output must be a Path")
    folder.mkdir(exist_ok=False)
    name = f"heldout_{schedule.case_id}_{actor}"
    cap = schedule.control_cap
    # Only a reset follows this assignment. The same compiled model/data pair
    # is used for both actors and for all requested horizons.
    env.max_steps = cap
    runtime.start_segment(name, cap)
    runtime.bind(env)
    guard.start_segment(env.plant, folder, name, cap, "heldout")
    common._save(folder / "boundary_before.json", _boundary(runtime))
    common._save(folder / "geometry_manifest.json", env.plant.collision_terrain_metadata)
    common._save(folder / "schedule.json", {
        "schema": schedule.schema, "case_id": schedule.case_id, "actor": actor,
        "seed": schedule.seed, "terrain": schedule.terrain,
        "spawn_position_m": schedule.spawn_position_m, "control_cap": cap,
        "raw_command_sha256": schedule.command_sha256,
        "raw_commands": [asdict(row) for row in schedule.raw_commands],
    })
    states: dict[str, list[Any]] = {key: [] for key in (*PAIR_FIELDS, "time")}
    rows: list[dict[str, Any]] = []
    row_files: list[dict[str, Any]] = []
    completed = prediction_attempts = predictions = 0
    done = terminated = truncated = False
    final_info: dict[str, Any] | None = None
    failure = None
    stop_reason = "control_cap"
    initial: dict[str, Any] | None = None
    started = time.monotonic()

    def snapshot(observation: Any) -> None:
        item = _capture_state(env, observation)
        for field, values in states.items():
            values.append(item[field])

    def flush_rows() -> None:
        if not rows:
            return
        index = len(row_files)
        path = folder / f"control_records_{index:04d}.jsonl.gz"
        _write_gzip_rows(path, rows)
        row_files.append({"file": path.name, "rows": len(rows), **_sha_bytes(path)})
        rows.clear()

    try:
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
        if pair_initial is not None:
            if any(
                initial[field].shape != pair_initial[field].shape
                or initial[field].dtype != pair_initial[field].dtype
                or initial[field].tobytes(order="C")
                != pair_initial[field].tobytes(order="C")
                for field in PAIR_FIELDS
            ):
                raise RuntimeError("zero/final initial states differ bitwise")
            if (int(env.plant.model._address), int(env.plant.data._address)) != pair_initial["identity"]:
                raise RuntimeError("matched pair changed model/data identity")
        initial["identity"] = (int(env.plant.model._address), int(env.plant.data._address))
        common._save(folder / "reset_receipt.json", {
            "before": before_reset, "after": after_reset,
            "seed": schedule.seed, "episode_metadata": episode,
            "model_address": initial["identity"][0],
            "data_address": initial["identity"][1],
            "exact_initial_pair": pair_initial is not None,
            "paired_with": None if pair_initial is None else "zero",
        })
        _save_arrays(folder / "initial_state.npz",
                     **{key: initial[key] for key in PAIR_FIELDS})
        snapshot(observation)
        for tick in range(cap):
            input_observation = np.asarray(observation, dtype=np.float32).copy()
            prediction_attempts += int(actor == "final_policy")
            action, predicted = _numeric_actor_action(actor, policy, observation)
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
                "tick": tick, "actor": actor,
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
        if schedule.terrain == "ramp":
            common._save(folder / "final_compiled_x_projection.json", _x_projection(env.plant))
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        stop_reason = "exception"
        raise
    finally:
        runtime.unbind()
        guard_segment = guard.finish_segment()
        flush_rows()
        if states["time"]:
            _save_arrays(folder / "states.npz",
                         **{key: np.asarray(values) for key, values in states.items()})
        receipt = {
            "schema": HELDOUT_SCHEMA,
            "case_id": schedule.case_id, "actor": actor,
            "seed": schedule.seed, "terrain": schedule.terrain,
            "raw_command_sha256": schedule.command_sha256,
            "control_cap": cap, "completed_controls": completed,
            "policy_prediction_attempts": prediction_attempts,
            "policy_predictions": predictions, "termination_recorded": done,
            "terminated": terminated, "truncated": truncated,
            "stop_reason": stop_reason, "failure": failure,
            "record_valid": failure is None and done and len(states["time"]) == completed + 1
            and guard_segment["native_attempted"] == 5 * completed
            and guard_segment["native_returned"] == 5 * completed
            and prediction_attempts == predictions == (completed if actor == "final_policy" else 0),
            "full_horizon_without_task_termination": (
                failure is None and done and not terminated and truncated and completed == cap
            ),
            "final_compiled_x_projection": (
                "final_compiled_x_projection.json" if schedule.terrain == "ramp"
                and failure is None else None
            ),
            "initial_state_pair_exact": None if pair_initial is None else failure is None,
            "controller_record_blocks": row_files,
            "native_segment": guard_segment,
            "last_info_metrics": None if final_info is None else final_info["metrics"],
            "boundary_after": _boundary(runtime),
            "elapsed_wall_s": time.monotonic() - started,
            "full_physical_qualification_pending_independent_scoring": True,
        }
        common._save(folder / "case_receipt.json", receipt)
    if initial is None:
        raise RuntimeError("heldout reset did not save an initial state")
    return receipt, initial


__all__ = ("HELDOUT_SCHEMA", "PAIR_FIELDS", "WorldUprightHeldoutSchedule", "record_heldout_case")
