"""Pure stage-15 floor gate and new-seed six-task scoring adapter.

The caller independently verifies archive manifests, source/checkpoint/ELF
identity, controller/native/force chains and paired resets before using these
numeric functions. Neither function imports a model or replays physics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

import score15
from recipes15 import FLOOR_SEED, ORDER, floor_schedule

FLOOR_SCORE_SCHEMA = "d1-world-upright-stage15-floor-pure-score-v1"
EVALUATION_SCORE_SCHEMA = "d1-world-upright-stage15-three-actor-pure-score-v1"
EXPERIMENT_ACTORS = ("zero", "global_continue", "grouped_continue")


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return result


def _pose(qpos: np.ndarray) -> tuple[float, float]:
    if qpos.ndim != 1 or qpos.size < 7:
        raise ValueError("floor qpos lacks a base quaternion")
    w, x, y, z = map(float, qpos[3:7])
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if abs(norm - 1.0) > 1e-6:
        raise ValueError("floor base quaternion is not unit length")
    roll = math.degrees(math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2*(w*y - z*x)))))
    return roll, pitch


def score_floor_saved(
    receipt: Mapping[str, Any], saved_schedule: Mapping[str, Any],
    states: Mapping[str, Any], control_rows: Sequence[Mapping[str, Any]],
    native_rows: Sequence[Mapping[str, Any]], geometry: Mapping[str, Any], *,
    warning_count: int, geometry_invalid_count: int,
) -> dict[str, Any]:
    """Evaluate the 600-control floor gate from independently read artifacts.

    This is numeric screening only. A true result still needs independent
    archive, source, checkpoint, controller and native force verification.
    """
    schedule = floor_schedule()
    actor = receipt.get("actor")
    if actor not in EXPERIMENT_ACTORS[1:]:
        raise ValueError("floor requires one named continuation actor")
    if (saved_schedule.get("case_id") != schedule.case_id
            or saved_schedule.get("actor") != actor
            or saved_schedule.get("seed") != FLOOR_SEED
            or saved_schedule.get("terrain") != "flat"
            or saved_schedule.get("control_cap") != 600
            or saved_schedule.get("raw_command_sha256") != schedule.command_sha256
            or saved_schedule.get("raw_commands")
            != [asdict(row) for row in schedule.raw_commands]):
        raise ValueError("floor saved command schedule differs")
    checkpoint_sha256 = receipt.get("checkpoint_sha256")
    if (type(checkpoint_sha256) is not str or len(checkpoint_sha256) != 64
            or any(c not in "0123456789abcdef" for c in checkpoint_sha256)):
        raise ValueError("floor checkpoint identity is absent")
    if (receipt.get("case_id") != schedule.case_id
            or receipt.get("seed") != FLOOR_SEED
            or receipt.get("control_cap") != 600
            or receipt.get("raw_command_sha256") != schedule.command_sha256
            or receipt.get("checkpoint_sha256") != saved_schedule.get("checkpoint_sha256")
            or receipt.get("scoring_actor") != "final_policy"):
        raise ValueError("floor receipt identity differs")
    if (type(warning_count) is not int or warning_count < 0
            or type(geometry_invalid_count) is not int or geometry_invalid_count < 0):
        raise ValueError("floor warning and geometry counts must be nonnegative integers")
    n = receipt.get("completed_controls")
    if (type(n) is not int or not 1 <= n <= 600
            or len(control_rows) != n or len(native_rows) != 5*n):
        raise ValueError("floor saved control/native prefix length differs")
    qpos = np.asarray(states.get("qpos"), dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[0] != n+1 or qpos.shape[1] < 7 or not np.isfinite(qpos).all():
        raise ValueError("floor lacks finite qpos endpoints for its saved prefix")
    observation = _array(states.get("observation"), (n+1, 99), "floor observation")
    times = _array(states.get("time"), (n+1,), "floor time")
    if not np.allclose(times, np.arange(n+1) * .01, rtol=0.0, atol=1e-10):
        raise ValueError("floor endpoint time sequence differs")
    for key in ("qvel", "ctrl", "qacc_warmstart"):
        values = np.asarray(states.get(key), dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != n+1 or not np.isfinite(values).all():
            raise ValueError(f"floor {key} endpoint sequence differs")

    world = geometry.get("world_collision_geoms")
    if not isinstance(world, Sequence):
        raise ValueError("floor has no saved compiled terrain geometry")
    floors = [row for row in world if isinstance(row, Mapping) and row.get("name") == "floor"]
    if len(floors) != 1:
        raise ValueError("floor has no unique compiled plane")
    plane = floors[0]
    if plane.get("type") != "plane":
        raise ValueError("floor terrain is not the compiled plane")
    floor_z = float(_array(plane.get("position_m"), (3,), "floor position")[2])

    vx = np.empty(n, dtype=np.float64)
    yaw = np.empty(n, dtype=np.float64)
    clearance = np.empty(n, dtype=np.float64)
    map_escape = fall = 0
    for tick, row in enumerate(control_rows):
        if (row.get("tick") != tick or row.get("actor") != actor
                or row.get("policy_predict_called") is not True):
            raise ValueError("floor control row index/actor/predict link differs")
        if not np.array_equal(_array(row.get("input_observation99"), (99,),
                                     "floor input observation"), observation[tick]):
            raise ValueError("floor input observation breaks endpoint chain")
        _array(row.get("policy_input_action"), (16,), "floor policy action")
        info = row.get("info")
        if not isinstance(info, Mapping):
            raise ValueError("floor control has no actual info")
        metrics = info.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("floor control has no actual metrics")
        vx[tick] = float(_array(metrics.get("body_com_vx_mps"), (), "COM vx"))
        yaw[tick] = float(_array(metrics.get("body_yaw_rate_rps"), (), "body yaw"))
        clearance[tick] = float(_array(metrics.get("clearance_m"), (), "clearance"))
        if info.get("terminal_reason") == "course_map_boundary":
            map_escape += 1
        if info.get("terminal_reason") == "fall_or_low_clearance":
            fall += 1
    roll_pitch = [_pose(qpos[0])]
    native_clearance = [float(qpos[0, 2] - floor_z)]
    native_xy = [qpos[0, :2]]
    nonwheel = 0
    first_native_index = native_rows[0].get("native_index")
    if type(first_native_index) is not int:
        raise ValueError("floor native index is absent")
    for index, row in enumerate(native_rows):
        if row.get("native_index") != first_native_index + index:
            raise ValueError("floor native rows skip an integrator return")
        start = float(_array(row.get("start_time_s"), (), "native start time"))
        end = float(_array(row.get("end_time_s"), (), "native end time"))
        if (not math.isclose(start, index*.002, rel_tol=0.0, abs_tol=1e-10)
                or not math.isclose(end, (index+1)*.002,
                                    rel_tol=0.0, abs_tol=1e-10)):
            raise ValueError("floor native clock sequence differs")
        after = row.get("after")
        if not isinstance(after, Mapping):
            raise ValueError("floor native row has no state")
        native_qpos = np.asarray(after.get("qpos"), dtype=np.float64)
        if native_qpos.shape != (qpos.shape[1],) or not np.isfinite(native_qpos).all():
            raise ValueError("floor native qpos is nonfinite or incomplete")
        roll_pitch.append(_pose(native_qpos))
        native_clearance.append(float(native_qpos[2] - floor_z))
        native_xy.append(native_qpos[:2])
        if index % 5 == 4 and not np.array_equal(native_qpos, qpos[index//5+1]):
            raise ValueError("floor native exit differs from endpoint")
        count = row.get("nonwheel_contact_count")
        if type(count) is not int or count < 0:
            raise ValueError("floor native nonwheel count differs")
        nonwheel += count
    roll_pitch_array = np.asarray(roll_pitch)
    native_xy_array = np.asarray(native_xy)
    hold = vx[325:min(n, 425)]
    hold_yaw = yaw[325:min(n, 425)]
    mean_error = abs(float(np.mean(hold)) - .4) if len(hold) == 100 else None
    vx_rms = (float(np.sqrt(np.mean(np.square(hold - .4))))
              if len(hold) == 100 else None)
    yaw_rms = (float(np.sqrt(np.mean(np.square(hold_yaw))))
               if len(hold) == 100 else None)
    max_roll = float(np.max(np.abs(roll_pitch_array[:, 0])))
    max_pitch = float(np.max(np.abs(roll_pitch_array[:, 1])))
    min_clearance = min(float(np.min(clearance)), min(native_clearance))
    max_x = float(np.max(np.abs(native_xy_array[:, 0])))
    max_y = float(np.max(np.abs(native_xy_array[:, 1])))
    guard = receipt.get("native_segment")
    complete = (
        n == 600
        and receipt.get("termination_recorded") is True
        and receipt.get("terminated") is False
        and receipt.get("truncated") is True
        and receipt.get("stop_reason") in ("time_limit", "horizon_cap")
        and receipt.get("failure") is None
        and receipt.get("policy_prediction_attempts") == receipt.get("policy_predictions") == n
        and isinstance(guard, Mapping)
        and guard.get("native_attempted") == guard.get("native_returned") == 5*n
        and guard.get("record_valid") is True
        and guard.get("force_sampling_performed") is True
        and guard.get("full_contact_qualification_recorded") is True
        and guard.get("archive_failure") is None
    )
    reasons = []
    for failed, message in (
        (not complete, "incomplete 600-control/3000-native record"),
        (warning_count != 0, "engine warning"),
        (geometry_invalid_count != 0, "invalid compiled geometry"),
        (max_roll > 10.0, "world roll exceeds 10 degrees"),
        (max_pitch > 10.0, "world pitch exceeds 10 degrees"),
        (min_clearance < .28, "clearance below 0.28 m"),
        (nonwheel != 0, "nonwheel terrain contact"),
        (max_x > 10.5 or max_y > 5.8 or map_escape,
         "course map boundary exceeded"),
        (fall != 0, "task fall"),
        (mean_error is None or mean_error > .2,
         "last 100 drive mean COM vx error exceeds 0.2 m/s"),
        (vx_rms is None or vx_rms > .25,
         "last 100 drive COM vx RMS error exceeds 0.25 m/s"),
        (yaw_rms is None or yaw_rms > .15,
         "last 100 drive yaw RMS exceeds 0.15 rad/s"),
    ):
        if failed:
            reasons.append(message)
    return {
        "schema": FLOOR_SCORE_SCHEMA, "case_id": schedule.case_id,
        "experiment_actor": actor,
        "checkpoint_sha256": checkpoint_sha256,
        "seed": FLOOR_SEED, "numeric_gate_passed": not reasons,
        "reasons": reasons, "completed_controls": n,
        "last_drive_window": [325, 425],
        "last_drive_observed_ticks": len(hold),
        "last_drive_mean_abs_target_error_mps": mean_error,
        "last_drive_vx_rms_error_mps": vx_rms,
        "last_drive_yaw_rms_rps": yaw_rms,
        "max_abs_world_roll_deg": max_roll,
        "max_abs_world_pitch_deg": max_pitch,
        "min_initial_or_native_clearance_m": min_clearance,
        "native_nonwheel_count": nonwheel,
        "max_initial_or_native_abs_x_m": max_x,
        "max_initial_or_native_abs_y_m": max_y,
        "independent_archive_source_controller_force_verification_required": True,
        "complete_stopping_qualification_from_600_controls": False,
    }


def score_canonical_cases(
    canonical_cases: Sequence[Mapping[str, Any]], *,
    checkpoint_sha256_by_actor: Mapping[str, str],
) -> dict[str, Any]:
    """Score complete six-task sets for zero and one or two continuation actors.

    Only the numeric actor alias is sent to the unchanged six-pair arithmetic;
    the experiment actor and checkpoint identity remain attached to each row.
    """
    enabled = tuple(actor for actor in EXPERIMENT_ACTORS[1:]
                    if actor in checkpoint_sha256_by_actor)
    if (not isinstance(canonical_cases, Sequence)
            or not 1 <= len(enabled) <= 2
            or set(checkpoint_sha256_by_actor) != set(enabled)
            or len(canonical_cases) != 6 * (1 + len(enabled))):
        raise ValueError("stage-15 scoring needs zero and each enabled actor on six tasks")
    if any(type(value) is not str or len(value) != 64
           or any(c not in "0123456789abcdef" for c in value)
           for value in checkpoint_sha256_by_actor.values()):
        raise ValueError("stage-15 checkpoint identity must be SHA256")
    table: dict[tuple[str, str], dict[str, Any]] = {}
    for case in canonical_cases:
        actor = case.get("experiment_actor")
        case_id = case.get("case_id")
        if actor not in ("zero", *enabled) or case_id not in ORDER:
            raise ValueError("unknown stage-15 actor/task")
        key = (case_id, actor)
        if key in table:
            raise ValueError("duplicate stage-15 actor/task")
        expected_sha = checkpoint_sha256_by_actor.get(actor)
        if case.get("checkpoint_sha256") != expected_sha:
            raise ValueError("case checkpoint differs from its experiment actor")
        scoring_actor = "zero" if actor == "zero" else "final_policy"
        if case.get("scoring_actor") != scoring_actor:
            raise ValueError("case scoring actor alias differs")
        numeric = dict(case, actor=scoring_actor)
        scored = score15.score_case(numeric)
        scored["experiment_actor"] = actor
        scored["checkpoint_sha256"] = expected_sha
        table[key] = scored
    if set(table) != {(case_id, actor) for case_id in ORDER
                      for actor in ("zero", *enabled)}:
        raise ValueError("stage-15 enabled six-task table is incomplete")
    pair_scores = {}
    for actor in enabled:
        rows = [table[(case_id, member)] for case_id in ORDER
                for member in ("zero", actor)]
        pair_scores[actor] = score15.score_pairs(rows)
    return {
        "schema": EVALUATION_SCORE_SCHEMA,
        "scores": [table[(case_id, actor)] for case_id in ORDER
                   for actor in ("zero", *enabled)],
        "zero_pairs": pair_scores,
        "all_three_actors_present": len(enabled) == 2,
        "independent_saved_record_verification_required": True,
    }


__all__ = (
    "FLOOR_SCORE_SCHEMA", "EVALUATION_SCORE_SCHEMA", "EXPERIMENT_ACTORS",
    "score_floor_saved", "score_canonical_cases",
)
