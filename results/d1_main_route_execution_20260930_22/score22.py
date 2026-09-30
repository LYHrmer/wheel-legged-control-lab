"""Pure C22 heldout and floor scorer with explicit session seeds.

Only a separate saved-record reader establishes source, model, controller,
contact-force and 5T provenance. This module consumes its canonical numeric
arrays; it imports no simulator, policy, model or training worker.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rl16_heldout_score_08 as prior
import score18 as c18

PRIOR_SHA256 = "9ca55bff625125aac63e598c06f2177cb88518a7268febd0abb48bbccfe5bc1f"
if hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest() != PRIOR_SHA256:
    raise RuntimeError("frozen 08 pure scoring arithmetic changed")

SCORE_SCHEMA = "d1-course-world-upright-rl16-heldout-pure-score-21-v1"
PAIR_SCHEMA = "d1-course-world-upright-rl16-heldout-pure-pair-21-v1"
FLOOR_SCORE_SCHEMA = "d1-world-upright-stage22-floor-pure-score-v1"
TASK_SCHEMA = "d1-course-world-upright-rl16-task-v1"
REFERENCE_SCHEMA = "d1-course-world-upright-roll-pitch-zero-v1"
REWARD_SCHEMA = "d1-course-world-upright-bodycom-yaw-terrain-action-torque-v1"
ACTORS = ("zero", "final_policy")
RAMP_GEOMS = ("terrain_ramp_up", "terrain_ramp_deck", "terrain_ramp_down")
SPEC = json.loads((Path(__file__).parent / "spec22.json").read_text())
CASE_SPECS = {row["case_id"]: row for group in ("regression", "development", "final_sealed")
              for row in SPEC["evaluation"][group]}
ORIGINAL_SIX = tuple(row["case_id"] for row in SPEC["evaluation"]["regression"][:6])
HELDOUT_CASES = {name: (row["terrain"], row["speed_mps"],
    row.get("yaw_amplitude_rps", max((abs(s[2]) for s in row.get("yaw_segments", [])), default=0.)),
    row["seed"], row["horizon"]) for name, row in CASE_SPECS.items()}
EVIDENCE_BOUNDARY = (
    "Pure arithmetic only. A separate independent saved-record reader must bind "
    "all source/checkpoint/ELF hashes, actual 5T native integrator/contactForce "
    "and controller-applied-torque chains, full C/Python budgets, paired initial "
    "states, ramp compiled geometry and final wheel shape projection. Supplied "
    "reader_asserted booleans are echoed, never used as qualification evidence."
)


def case_windows(case_id: str) -> dict[str, Any]:
    row = CASE_SPECS[case_id]
    return {"control_cap": row["horizon"], "hold": list(row["hold"]),
            "release_tick_t0": row["release_tick"],
            "drive": [row["drive_start"], row["release_tick"]],
            "final": list(row["final"])}


def expected_raw_commands(case_id: str) -> list[dict]:
    """Independent value construction; never imports the runtime recipe."""
    row = CASE_SPECS[case_id]
    segments = row.get("yaw_segments", [])
    if row.get("yaw_amplitude_rps", 0.):
        start = row["hold"][0]
        amp = row["yaw_amplitude_rps"] * (-1 if row.get("mirror", False) else 1)
        segments = [(start, start+100, amp), (start+100, start+300, -amp),
                    (start+300, row["hold"][1], amp)]
    return [{"forward_velocity_mps": row["speed_mps"]
             if row["drive_start"] <= tick < row["release_tick"] else 0.,
             "lateral_velocity_mps": 0.,
             "yaw_rate_rps": next((y for a, b, y in segments if a <= tick < b), 0.),
             "clearance_m": .455, "jump_requested": False}
            for tick in range(row["horizon"])]


def _servo_commands(case_id: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    vx = yaw = 0.
    forward, turning = [], []
    for row in expected_raw_commands(case_id)[:n]:
        vx += max(-.005, min(.005, row["forward_velocity_mps"]-vx))
        yaw += max(-.006, min(.006, row["yaw_rate_rps"]-yaw))
        forward.append(vx)
        turning.append(yaw)
    return np.asarray(forward), np.asarray(turning)


def _speed(case_id, target, n, start, end, vx, servo):
    if case_id in ORIGINAL_SIX:
        return prior._speed_block(case_id, target, n, start, end, vx, servo)
    hold = vx[start:min(n, end)]
    observed = len(hold)
    mean = float(np.mean(hold)) if observed else None
    rms = float(np.sqrt(np.mean((hold-target)**2))) if observed else None
    reasons = []
    if observed != end-start:
        reasons.append("preregistered hold window incomplete")
    if mean is None or abs(mean-target) > .04:
        reasons.append("hold mean speed outside target +/- 0.04 m/s")
    if rms is None or rms > .05:
        reasons.append("hold speed RMS exceeds 0.05 m/s")
    return {"passed": not reasons, "reasons": reasons, "hold_window": [start, end],
            "hold_observed_ticks": observed, "hold_complete": observed == end-start,
            "mean_com_vx_mps": mean, "rms_com_vx_error_vs_target_mps": rms,
            "rms_com_vx_error_vs_applied_servo_mps": float(np.sqrt(np.mean(
                (hold-servo[start:min(n, end)])**2))) if observed else None,
            "gate": "straight_hold", "mean_tolerance_mps": .04, "max_rms_mps": .05}


def _yaw(amp, n, start, end, rate, servo, final_window):
    if not amp:
        return {"applicable": False, "passed": None, "reasons": []}
    observed = max(0, min(n, end)-start)
    hold, target = rate[start:min(n, end)], servo[start:min(n, end)]
    rms = float(np.sqrt(np.mean((hold-target)**2))) if observed else None
    positive = float(np.sum(hold[target > 0])*.01) if observed else None
    negative = float(np.sum(hold[target < 0])*.01) if observed else None
    tail = rate[final_window[0]:min(n, final_window[1])]
    final_mean = float(np.mean(np.abs(tail))) if len(tail) else None
    reasons = []
    for fail, message in ((observed != end-start, "preregistered yaw hold incomplete"),
        (rms is None or rms > .12, "yaw RMS exceeds 0.12 rad/s"),
        (positive is None or positive <= 0, "positive-servo integrated yaw must be positive"),
        (negative is None or negative >= 0, "negative-servo integrated yaw must be negative"),
        (len(tail) != final_window[1]-final_window[0], "final yaw window incomplete"),
        (final_mean is None or final_mean > .05, "final mean absolute yaw exceeds 0.05 rad/s")):
        if fail:
            reasons.append(message)
    return {"applicable": True, "passed": not reasons, "reasons": reasons,
            "hold_observed_ticks": observed, "rms_yaw_rate_error_vs_applied_servo_rps": rms,
            "positive_servo_integrated_yaw_rad": positive,
            "negative_servo_integrated_yaw_rad": negative,
            "positive_servo_ticks": int(np.sum(target > 0)),
            "negative_servo_ticks": int(np.sum(target < 0)),
            "final_mean_abs_yaw_rate_rps": final_mean, "max_rms_rps": .12}


def _terms(n, start, end, vx, yaw, servo_vx, servo_yaw, torque):
    stop = min(n, end)
    observed = max(0, stop-start)
    span = slice(start, stop) if observed else slice(0, 0)
    ev, ey = (vx[span]-servo_vx[span])/.25, (yaw[span]-servo_yaw[span])/.4
    costs = np.mean((torque[span]/prior.TORQUE_LIMITS_NM[None, :, None])**2, axis=(1, 2))
    sv, sy = float(np.sum(ev**2)), float(np.sum(ey**2))
    return {"drive_window": [start, end], "drive_completed_stop": stop,
            "drive_completed_ticks": observed, "drive_complete": observed == end-start,
            "sse_total": sv+sy, "sse_vx": sv, "sse_yaw": sy,
            "error_mean": (sv+sy)/observed if observed else None,
            "torque_cost_sum": float(np.sum(costs)),
            "torque_cost_mean": float(np.mean(costs)) if observed else None,
            "torque_cost_by_tick": costs.tolist(),
            "scales": {"vx_mps": .25, "yaw_rps": .4},
            "torque_limits_nm": prior.TORQUE_LIMITS_NM.tolist(),
            "note": "Normalized torque-squared proxy, not energy."}


def _record(completed: int, cap: int, terminated: bool,
            truncated: bool, reason: str) -> dict[str, Any]:
    reasons = []
    if terminated and truncated:
        reasons.append("terminated and truncated cannot both be true")
    if not (terminated or truncated):
        reasons.append("no real terminal transition")
    if completed < cap and truncated:
        reasons.append("TimeLimit before preregistered cap")
    if reason == "exception":
        reasons.append("worker exception is not a legitimate task terminal")
    if terminated and reason not in (
        "fall_or_low_clearance", "nonwheel_ground_contact", "course_map_boundary"
    ):
        reasons.append("unknown task-failure terminal reason")
    if truncated and reason not in ("time_limit", "horizon_cap"):
        reasons.append("unknown TimeLimit reason")
    return {
        "record_valid": not reasons, "reasons": reasons,
        "full_horizon": not reasons and completed == cap and not terminated,
        "early_task_termination": not reasons and terminated,
    }


def _safety(
    native_roll: np.ndarray, native_pitch: np.ndarray,
    native_relative_tilt: np.ndarray, native_clearance: np.ndarray,
    native_nonwheel: np.ndarray, native_x: np.ndarray, native_y: np.ndarray,
    initial_roll: float, initial_pitch: float,
    initial_relative_tilt: float, initial_clearance: float,
    initial_x: float, initial_y: float,
    warnings: int, geometry_invalid: int, map_escapes: int, falls: int,
) -> dict[str, Any]:
    roll = max(abs(initial_roll), float(np.max(np.abs(native_roll))))
    pitch = max(abs(initial_pitch), float(np.max(np.abs(native_pitch))))
    relative = max(abs(initial_relative_tilt),
                   float(np.max(np.abs(native_relative_tilt))))
    clearance = min(initial_clearance, float(np.min(native_clearance)))
    max_x = max(abs(initial_x), float(np.max(np.abs(native_x))))
    max_y = max(abs(initial_y), float(np.max(np.abs(native_y))))
    nonwheel = int(np.sum(native_nonwheel))
    reasons = []
    if roll > 10.0:
        reasons.append("world absolute roll exceeds 10 degrees")
    if pitch > 10.0:
        reasons.append("world absolute pitch exceeds 10 degrees")
    if clearance < .28:
        reasons.append("native/initial clearance below 0.28 m")
    if nonwheel:
        reasons.append("nonwheel terrain contact candidate")
    if max_x > 10.5 or max_y > 5.8:
        reasons.append("initial/native base position outside |x|<=10.5, |y|<=5.8")
    for count, name in ((warnings, "warning"), (geometry_invalid, "geometry invalid"),
                        (map_escapes, "map escape"), (falls, "task fall")):
        if count:
            reasons.append(f"{count} {name} event(s)")
    return {
        "passed": not reasons, "reasons": reasons,
        "max_abs_world_roll_deg": roll, "max_abs_world_pitch_deg": pitch,
        "max_geometry_relative_tilt_deg_diagnostic_only": relative,
        "min_initial_or_native_clearance_m": clearance,
        "max_initial_or_native_abs_world_x_m": max_x,
        "max_initial_or_native_abs_world_y_m": max_y,
        "nonwheel_ground_candidate_total": nonwheel,
        "native_samples_checked": len(native_roll),
        "initial_state_included": True,
    }


def _final(completed: int, window: list[int], com_vx: np.ndarray,
           yaw_rate: np.ndarray, clearance: np.ndarray) -> dict[str, Any]:
    start, end = window
    observed = max(0, min(end, completed) - start)
    sl = slice(start, min(end, completed))
    vx = float(np.mean(np.abs(com_vx[sl]))) if observed else None
    yaw = float(np.mean(np.abs(yaw_rate[sl]))) if observed else None
    std = float(np.std(clearance[sl], ddof=0)) if observed else None
    reasons = []
    if observed != 100:
        reasons.append("final 100 actual post samples incomplete")
    if vx is None or vx > .05:
        reasons.append("final mean absolute COM vx exceeds 0.05 m/s")
    if yaw is None or yaw > .05:
        reasons.append("final mean absolute body yaw rate exceeds 0.05 rad/s")
    if std is None or std > .02:
        reasons.append("final clearance population std exceeds 0.02 m")
    return {"passed": not reasons, "reasons": reasons,
            "window": window, "observed_ticks": observed,
            "mean_abs_com_vx_mps": vx, "mean_abs_yaw_rate_rps": yaw,
            "clearance_population_std_m": std}


def _ramp_geometry(case_id: str, case: Mapping[str, object]) -> dict[str, Any]:
    if case_id != "ramp_0p45_complete":
        return {"applicable": False, "passed": None, "reasons": []}
    names = prior._field(case, "ramp_positive_wheel_load_native_counts_by_geom")
    ramp = prior._field(case, "ramp_geom_world_x_max_m_by_name")
    wheel = prior._field(case, "wheel_collision_world_x_min_m_by_index")
    if not isinstance(names, Mapping) or set(names) != set(RAMP_GEOMS):
        raise ValueError("ramp has no three exact compiled geom load counts")
    if not isinstance(ramp, Mapping) or set(ramp) != set(RAMP_GEOMS):
        raise ValueError("ramp has no three compiled world-X extrema")
    if not isinstance(wheel, Mapping) or set(wheel) != set(map(str, range(4))):
        raise ValueError("ramp has no four wheel collision world-X minima")
    counts = {key: prior._count(names, key) for key in RAMP_GEOMS}
    maxima = {key: prior._scalar(ramp, key) for key in RAMP_GEOMS}
    minima = {key: prior._scalar(wheel, key) for key in map(str, range(4))}
    edge = max(maxima.values())
    reasons = []
    if any(value == 0 for value in counts.values()):
        reasons.append("each of the three actual ramp geoms needs positive wheel load")
    if any(value <= edge for value in minima.values()):
        reasons.append("each whole wheel collision cylinder must clear every ramp box")
    return {
        "applicable": True, "passed": not reasons, "reasons": reasons,
        "positive_wheel_load_native_counts_by_geom": counts,
        "ramp_geom_world_x_max_m_by_name": maxima,
        "wheel_collision_world_x_min_m_by_index": minima,
        "all_ramp_world_x_max_m": edge,
        "source": "canonical numbers require independent compiled-binding and contactForce proof",
    }


def score_case(canonical_case: Mapping[str, object], *, expected_seed: int,
               mirror: bool = False) -> dict[str, Any]:
    """Score a canonical saved-data case; no engine provenance is self-certified."""
    case_id = prior._text(canonical_case, "case_id")
    if case_id not in HELDOUT_CASES:
        raise ValueError("unknown preregistered C22 heldout case")
    terrain, target, yaw_amplitude, _old_seed, cap = HELDOUT_CASES[case_id]
    if type(expected_seed) is not int or not 0 <= expected_seed < 2**32:
        raise ValueError("expected_seed must come from the frozen session")
    if (type(mirror) is not bool or mirror != CASE_SPECS[case_id].get("mirror", False)
            or expected_seed != CASE_SPECS[case_id]["seed"]):
        raise ValueError("mirror or seed differs from the frozen C22 case")
    actor = prior._text(canonical_case, "actor")
    if actor not in ACTORS or canonical_case.get("experiment_actor") not in ("zero", "old", "A", "B"):
        raise ValueError("actor is not zero/final_policy")
    if (prior._count(canonical_case, "seed") != expected_seed
            or canonical_case.get("mirror") is not mirror
            or prior._text(canonical_case, "terrain") != terrain
            or prior._text(canonical_case, "task_schema") != TASK_SCHEMA
            or prior._text(canonical_case, "reference_schema") != REFERENCE_SCHEMA
            or prior._text(canonical_case, "reward_schema") != REWARD_SCHEMA):
        raise ValueError("case seed/terrain/world-task identity differs")
    if actor != ("zero" if canonical_case["experiment_actor"] == "zero" else "final_policy"):
        raise ValueError("scoring actor differs from experiment actor")
    n = prior._count(canonical_case, "completed_controls")
    if not 1 <= n <= cap:
        raise ValueError("actual completed controls outside this case's cap")
    prior._indices(canonical_case, "tick_index", n)
    prior._indices(canonical_case, "native_index", 5*n)
    servo_vx = prior._reals(canonical_case, "applied_servo_vx_mps", (n,))
    servo_yaw = prior._reals(canonical_case, "applied_servo_yaw_rps", (n,))
    expected_vx, expected_yaw = _servo_commands(case_id, n)
    if (not np.array_equal(servo_vx, expected_vx)
            or not np.array_equal(servo_yaw, expected_yaw)):
        raise ValueError("applied servo differs from independent frozen command timeline")
    com_vx = prior._reals(canonical_case, "com_vx_mps", (n,))
    yaw_rate = prior._reals(canonical_case, "body_yaw_rate_rps", (n,))
    heading = prior._reals(canonical_case, "heading_rad", (n,))
    clearance = prior._reals(canonical_case, "clearance_m", (n,))
    lateral = prior._reals(canonical_case, "lateral_offset_m", (n,))
    projection = prior._reals(canonical_case, "forward_projection_m", (n,))
    torque = prior._reals(canonical_case, "motor_torque_nm", (n, 16, 5))
    native_roll = prior._reals(canonical_case, "native_roll_deg", (5*n,))
    native_pitch = prior._reals(canonical_case, "native_pitch_deg", (5*n,))
    native_tilt = prior._reals(canonical_case, "native_terrain_relative_tilt_deg", (5*n,))
    native_clearance = prior._reals(canonical_case, "native_clearance_m", (5*n,))
    native_projection = prior._reals(
        canonical_case, "native_forward_projection_m", (5*n,),
    )
    native_nonwheel = prior._nonneg_counts(
        canonical_case, "native_nonwheel_ground_candidate_count", 5*n,
    )
    native_x = prior._reals(canonical_case, "native_base_x_m", (5*n,))
    native_y = prior._reals(canonical_case, "native_base_y_m", (5*n,))
    initial_roll = prior._scalar(canonical_case, "initial_roll_deg")
    initial_pitch = prior._scalar(canonical_case, "initial_pitch_deg")
    initial_tilt = prior._scalar(canonical_case, "initial_terrain_relative_tilt_deg")
    initial_clearance = prior._scalar(canonical_case, "initial_clearance_m")
    initial_x = prior._scalar(canonical_case, "initial_base_x_m")
    initial_y = prior._scalar(canonical_case, "initial_base_y_m")
    load_counts = prior._family_counts(canonical_case)
    terminated = prior._flag(canonical_case, "terminated")
    truncated = prior._flag(canonical_case, "truncated")
    stop_reason = prior._text(canonical_case, "stop_reason")
    warnings = prior._count(canonical_case, "warning_count")
    geometry_invalid = prior._count(canonical_case, "geometry_invalid_count")
    map_escapes = prior._count(canonical_case, "map_escape_count")
    falls = prior._count(canonical_case, "fall_count")
    asserted = canonical_case.get("reader_asserted", {})
    if not isinstance(asserted, Mapping):
        raise TypeError("reader_asserted must be a mapping")

    windows = case_windows(case_id)
    hold_begin, hold_end = windows["hold"]
    release = windows["release_tick_t0"]
    record = _record(n, cap, terminated, truncated, stop_reason)
    # A solver warning or invalid compiled geometry is a bad execution
    # record, never a learnable zero-policy task failure for branch A.
    if warnings or geometry_invalid:
        record["reasons"].extend(
            ["engine warning makes the saved record invalid"] if warnings else []
        )
        record["reasons"].extend(
            ["invalid geometry makes the saved record invalid"]
            if geometry_invalid else []
        )
        record["record_valid"] = False
        record["full_horizon"] = False
        record["early_task_termination"] = False
    safety = _safety(native_roll, native_pitch, native_tilt, native_clearance,
                     native_nonwheel, native_x, native_y, initial_roll, initial_pitch,
                     initial_tilt, initial_clearance, initial_x, initial_y,
                     warnings, geometry_invalid, map_escapes, falls)
    speed = _speed(case_id, target, n, hold_begin, hold_end, com_vx, servo_vx)
    yaw = _yaw(yaw_amplitude, n, hold_begin, hold_end, yaw_rate, servo_yaw, windows["final"])
    final = _final(n, windows["final"], com_vx, yaw_rate, clearance)
    stopping = prior._stop_block(case_id, n, release, com_vx, yaw_rate, servo_vx,
                                 projection, native_projection)
    terrain_channel = prior._terrain_block(
        terrain, n, servo_vx, lateral, heading, projection, load_counts,
    )
    ramp_geometry = _ramp_geometry(case_id, canonical_case)
    terms = _terms(n, windows["drive"][0], release, com_vx, yaw_rate, servo_vx, servo_yaw, torque)
    hold_terms = _terms(n, hold_begin, hold_end, com_vx, yaw_rate, servo_vx, servo_yaw, torque)
    reasons = []
    if not record["record_valid"]:
        reasons.append("record integrity failed")
    if not record["full_horizon"]:
        reasons.append(f"observed {n} of {cap} controls without full task success")
    for name, block in (("safety", safety), ("speed", speed), ("yaw", yaw),
                        ("final_window", final), ("stopping", stopping),
                        ("terrain_channel", terrain_channel),
                        ("ramp_geometry", ramp_geometry)):
        if block["passed"] is False:
            reasons.extend(name + ": " + reason for reason in block["reasons"])
    return {
        "schema": SCORE_SCHEMA, "case_id": case_id, "actor": actor,
        "seed": expected_seed, "mirror": mirror, "terrain": terrain, "target_speed_mps": target,
        "yaw_amplitude_rps": yaw_amplitude, "completed_controls": n,
        "native_returns_expected_from_completed": 5*n,
        "terminated": terminated, "truncated": truncated,
        "stop_reason": stop_reason,
        "windows": {**windows,
                    "hold_observed_ticks": prior._observed(windows["hold"], n),
                    "final_observed_ticks": prior._observed(windows["final"], n),
                    "drive_observed_ticks": prior._observed(windows["drive"], n)},
        "record": record, "safety": safety, "speed": speed, "yaw": yaw,
        "final_window": final, "stopping": stopping,
        "terrain_channel": terrain_channel,
        "ramp_geometry": ramp_geometry,
        "rl_terms": terms, "hold_terms": hold_terms, "task_passed": not reasons, "task_reasons": reasons,
        "reader_asserted_unverified": {str(key): value for key, value in asserted.items()},
        "evidence_boundary": EVIDENCE_BOUNDARY,
    }


def score_pairs(case_scores: Sequence[Mapping[str, object]], *, policy_actor: str) -> dict[str, Any]:
    """Unchanged original-six C18 contribution gate for exactly zero versus A or B."""
    if policy_actor not in ("A", "B"):
        raise ValueError("regression contribution actor must be A or B")
    selected = []
    for row in case_scores:
        if row.get("case_id") in ORIGINAL_SIX and row.get("experiment_actor") in ("zero", policy_actor):
            if row.get("schema") not in (SCORE_SCHEMA, c18.SCORE_SCHEMA):
                raise ValueError("unknown score source schema")
            selected.append(dict(row, schema=c18.SCORE_SCHEMA,
                                 actor="zero" if row["experiment_actor"] == "zero" else "final_policy"))
    return dict(c18.score_pairs(selected), schema=PAIR_SCHEMA, policy_actor=policy_actor)


def score_floor_canonical(case: Mapping[str, object], *, expected_seed: int) -> dict[str, Any]:
    actor = case.get("experiment_actor")
    if actor not in ("A", "B"):
        raise ValueError("C22 floor actor must be A or B")
    adapted = dict(case, actor="final_policy", experiment_actor="grouped_continue")
    result = c18.score_floor_canonical(adapted, expected_seed=expected_seed)
    return dict(result, schema=FLOOR_SCORE_SCHEMA, experiment_actor=actor)


def ratio(numerator: float, denominator: float) -> dict[str, Any]:
    """No epsilon and no substitution of zero denominators."""
    if not np.isfinite([numerator, denominator]).all() or min(numerator, denominator) < 0:
        raise ValueError("ratio operands must be finite nonnegative values")
    return {"numerator": numerator, "denominator": denominator,
            "ratio": numerator/denominator if denominator > 0 else None,
            "defined": denominator > 0}


def equal_case_pool(scores: Sequence[Mapping[str, object]], *, split: str) -> dict[str, Any]:
    """Four preregistered cases have equal weight, separately for all four actors."""
    group = {"development": "development", "final": "final_sealed"}.get(split)
    if group is None:
        raise ValueError("unknown C22 evaluation split")
    expected = {row["case_id"]: row for row in SPEC["evaluation"][group]}
    table = {}
    for row in scores:
        if row.get("case_id") not in expected:
            continue
        key = (row["case_id"], row.get("experiment_actor"))
        if key in table or key[1] not in ("zero", "old", "A", "B") or row.get("schema") != SCORE_SCHEMA:
            raise ValueError("duplicate/unknown C22 case-actor or score schema")
        table[key] = row
    required = {(case, actor) for case in expected for actor in ("zero", "old", "A", "B")}
    missing = sorted(required-set(table))
    incomplete = [list(key) for key, row in table.items() if not (
        row["record"]["record_valid"] and row["record"]["full_horizon"]
        and row["safety"]["passed"] and row["rl_terms"]["drive_complete"]
        and row["hold_terms"]["drive_complete"])]
    if missing or incomplete:
        return {"complete": False, "missing": missing, "invalid_or_incomplete": incomplete,
                "actors": {}, "comparisons": {}, "claim_eligible": False}
    families = {"yaw": [name for name in expected if expected[name]["terrain"] == "flat"],
                "bumps": [name for name in expected if expected[name]["terrain"] == "bumps"]}
    if any(len(names) != 2 for names in families.values()):
        raise ValueError("new evaluation pool must contain two yaw and two bumps cases")
    actors = {}
    for actor in ("zero", "old", "A", "B"):
        rows = [table[(name, actor)] for name in expected]
        actors[actor] = {"equal_case_error_mean": float(np.mean([
            row["rl_terms"]["sse_total"]/row["rl_terms"]["drive_completed_ticks"] for row in rows])),
            "equal_case_torque_cost_mean": float(np.mean([row["rl_terms"]["torque_cost_mean"] for row in rows])),
            "all_tasks_passed": all(row["task_passed"] for row in rows), "families": {}}
        for family, names in families.items():
            actors[actor]["families"][family] = {term: float(np.mean([
                table[(name, actor)][key]["sse_total"]/table[(name, actor)][key]["drive_completed_ticks"]
                for name in names])) for term, key in (("drive_error_mean", "rl_terms"), ("hold_error_mean", "hold_terms"))}
    comparisons = {}
    for denominator in ("A", "old", "zero"):
        comparisons["B/"+denominator] = {
            "error": ratio(actors["B"]["equal_case_error_mean"], actors[denominator]["equal_case_error_mean"]),
            "cost": ratio(actors["B"]["equal_case_torque_cost_mean"], actors[denominator]["equal_case_torque_cost_mean"])}
    family_ratios = {family: {term: ratio(actors["B"]["families"][family][term],
                                              actors["A"]["families"][family][term])
                              for term in ("drive_error_mean", "hold_error_mean")}
                     for family in families}
    return {"complete": True, "missing": [], "invalid_or_incomplete": [],
            "case_weight": .25, "family_case_weight": .5,
            "actors": actors, "comparisons": comparisons, "family_B_over_A": family_ratios,
            "evidence_boundary": "Numeric pooling only; paired hidden state, coverage and sealed phase must be verified independently."}


def original_six_pool(scores: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """Original drive samples pooled as raw sums, never new-case equal weights."""
    table = {}
    for row in scores:
        key = (row.get("case_id"), row.get("experiment_actor"))
        if key[0] not in ORIGINAL_SIX:
            continue
        if key in table or key[1] not in ("zero", "old", "A", "B"):
            raise ValueError("duplicate or unknown original-six actor")
        table[key] = row
    required = {(case, actor) for case in ORIGINAL_SIX for actor in ("zero", "old", "A", "B")}
    missing = sorted(required-set(table))
    invalid = [list(key) for key,row in table.items() if not (
        row["record"]["record_valid"] and row["record"]["full_horizon"]
        and row["safety"]["passed"] and row["rl_terms"]["drive_complete"])]
    if missing or invalid:
        return {"complete": False, "missing": missing, "invalid_or_incomplete": invalid,
                "actors": {}, "comparisons": {}}
    actors = {}
    for actor in ("zero", "old", "A", "B"):
        rows = [table[(name,actor)] for name in ORIGINAL_SIX]
        counts = sum(row["rl_terms"]["drive_completed_ticks"] for row in rows)
        cost = sum(row["rl_terms"]["torque_cost_sum"] for row in rows)
        actors[actor] = {"pooled_sse_total": sum(row["rl_terms"]["sse_total"] for row in rows),
            "pooled_torque_cost_sum": cost, "pooled_torque_cost_mean": cost/counts,
            "drive_completed_ticks": counts, "all_six_tasks_passed": all(row["task_passed"] for row in rows)}
    return {"complete": True, "actors": actors, "case_count": 6,
        "comparisons": {"B/"+actor: {
            "error": ratio(actors["B"]["pooled_sse_total"],actors[actor]["pooled_sse_total"]),
            "cost": ratio(actors["B"]["pooled_torque_cost_mean"],actors[actor]["pooled_torque_cost_mean"])}
            for actor in ("A", "old", "zero")},
        "mirror_excluded": True}
