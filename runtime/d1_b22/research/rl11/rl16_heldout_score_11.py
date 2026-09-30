"""Pure 11-S world-upright, variable-horizon heldout numeric scorer.

Only a separate saved-record reader establishes source, model, controller,
contact-force and 5T provenance. This module consumes its canonical numeric
arrays; it imports no simulator, policy, model or training worker.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rl16_heldout_score_08 as prior

PRIOR_SHA256 = "9ca55bff625125aac63e598c06f2177cb88518a7268febd0abb48bbccfe5bc1f"
if hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest() != PRIOR_SHA256:
    raise RuntimeError("frozen 08 pure scoring arithmetic changed")

SCORE_SCHEMA = "d1-course-world-upright-rl16-heldout-pure-score-11-v1"
PAIR_SCHEMA = "d1-course-world-upright-rl16-heldout-pure-pair-11-v1"
TASK_SCHEMA = "d1-course-world-upright-rl16-task-v1"
REFERENCE_SCHEMA = "d1-course-world-upright-roll-pitch-zero-v1"
REWARD_SCHEMA = "d1-course-world-upright-bodycom-yaw-terrain-action-torque-v1"
ACTORS = ("zero", "final_policy")
RAMP_GEOMS = ("terrain_ramp_up", "terrain_ramp_deck", "terrain_ramp_down")
HELDOUT_CASES: dict[str, tuple[str, float, float, int, int]] = {
    "flat_0p6": ("flat", .6, 0.0, 88701, 1600),
    "flat_1p6": ("flat", 1.6, 0.0, 88702, 1600),
    "flat_1p2_yaw": ("flat", 1.2, .3, 88703, 1600),
    "bumps_0p4": ("bumps", .4, 0.0, 88704, 1600),
    "rough_0p35": ("rough", .35, 0.0, 88705, 1600),
    "ramp_0p45_complete": ("ramp", .45, 0.0, 88706, 1800),
}
EVIDENCE_BOUNDARY = (
    "Pure arithmetic only. A separate independent saved-record reader must bind "
    "all source/checkpoint/ELF hashes, actual 5T native integrator/contactForce "
    "and controller-applied-torque chains, full C/Python budgets, paired initial "
    "states, ramp compiled geometry and final wheel shape projection. Supplied "
    "reader_asserted booleans are echoed, never used as qualification evidence."
)


def case_windows(case_id: str) -> dict[str, Any]:
    if case_id not in HELDOUT_CASES:
        raise ValueError("unknown preregistered 11-S heldout case")
    _, speed, _, _, cap = HELDOUT_CASES[case_id]
    if case_id == "ramp_0p45_complete":
        hold_begin, hold_end = 600, 1000
        release = 1355
    else:
        hold_begin = 175 + prior.ramp_ticks(speed)
        hold_end = hold_begin + 400
        release = hold_end
    return {
        "control_cap": cap,
        "hold": [hold_begin, hold_end],
        "release_tick_t0": release,
        "drive": [175, release],
        "final": [cap - 100, cap],
    }


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


def score_case(canonical_case: Mapping[str, object]) -> dict[str, Any]:
    """Score a canonical saved-data case; no engine provenance is self-certified."""
    case_id = prior._text(canonical_case, "case_id")
    if case_id not in HELDOUT_CASES:
        raise ValueError("unknown 11-S heldout case")
    terrain, target, yaw_amplitude, seed, cap = HELDOUT_CASES[case_id]
    actor = prior._text(canonical_case, "actor")
    if actor not in ACTORS:
        raise ValueError("actor is not zero/final_policy")
    if (prior._count(canonical_case, "seed") != seed
            or prior._text(canonical_case, "terrain") != terrain
            or prior._text(canonical_case, "task_schema") != TASK_SCHEMA
            or prior._text(canonical_case, "reference_schema") != REFERENCE_SCHEMA
            or prior._text(canonical_case, "reward_schema") != REWARD_SCHEMA):
        raise ValueError("case seed/terrain/world-task identity differs")
    n = prior._count(canonical_case, "completed_controls")
    if not 1 <= n <= cap:
        raise ValueError("actual completed controls outside this case's cap")
    prior._indices(canonical_case, "tick_index", n)
    prior._indices(canonical_case, "native_index", 5*n)
    servo_vx = prior._reals(canonical_case, "applied_servo_vx_mps", (n,))
    servo_yaw = prior._reals(canonical_case, "applied_servo_yaw_rps", (n,))
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
    speed = prior._speed_block(case_id, target, n, hold_begin, hold_end, com_vx, servo_vx)
    yaw = prior._yaw_block(yaw_amplitude, n, hold_begin, hold_end, yaw_rate, servo_yaw)
    final = _final(n, windows["final"], com_vx, yaw_rate, clearance)
    stopping = prior._stop_block(case_id, n, release, com_vx, yaw_rate, servo_vx,
                                 projection, native_projection)
    terrain_channel = prior._terrain_block(
        terrain, n, servo_vx, lateral, heading, projection, load_counts,
    )
    ramp_geometry = _ramp_geometry(case_id, canonical_case)
    terms = prior._rl_terms_block(n, release, com_vx, yaw_rate, servo_vx, servo_yaw,
                                  torque)
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
        "seed": seed, "terrain": terrain, "target_speed_mps": target,
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
        "rl_terms": terms, "task_passed": not reasons, "task_reasons": reasons,
        "reader_asserted_unverified": {str(key): value for key, value in asserted.items()},
        "evidence_boundary": EVIDENCE_BOUNDARY,
    }


def score_pairs(case_scores: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """Apply the frozen 05 two-branch SSE/cost arithmetic to six new-task pairs."""
    if not isinstance(case_scores, Sequence) or isinstance(case_scores, (str, bytes)):
        raise TypeError("case_scores must be a sequence")
    if len(case_scores) != 12:
        raise ValueError("exactly six zero/final pairs are required")
    table: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in case_scores:
        if not isinstance(row, Mapping) or row.get("schema") != SCORE_SCHEMA:
            raise ValueError("all cases must come from the 11-S numeric scorer")
        key = (str(row["case_id"]), str(row["actor"]))
        if key[0] not in HELDOUT_CASES or key[1] not in ACTORS or key in table:
            raise ValueError("unknown or duplicate scored case/actor")
        table[key] = row
    if set(table) != {(case, actor) for case in HELDOUT_CASES for actor in ACTORS}:
        raise ValueError("one or more preregistered pair members are absent")
    pairs = [prior._pair_row(case_id, table[(case_id, "zero")],
                             table[(case_id, "final_policy")])
             for case_id in HELDOUT_CASES]
    reasons = []
    for case_id, actor in table:
        if not table[(case_id, actor)]["record"]["record_valid"]:
            reasons.append(f"{case_id}/{actor}: invalid saved record")
    for pair in pairs:
        if pair["capability_regression"]:
            reasons.append(pair["case_id"] + ": zero-pass/policy-fail regression")
        if not pair["policy_full_horizon"]:
            reasons.append(pair["case_id"] + ": policy incomplete")
        if not pair["policy_safety_passed"]:
            reasons.append(pair["case_id"] + ": policy safety gate failed")
    cost = prior._cost_gate(pairs)
    branch_a = prior._branch_a(pairs)
    branch_b = prior._branch_b(pairs)
    contribution = not reasons and cost["passed"] and (
        branch_a["passed"] or branch_b["passed"]
    )
    return {
        "schema": PAIR_SCHEMA, "case_count": 12, "pair_count": 6,
        "pairs": pairs, "prerequisites": {"passed": not reasons, "reasons": reasons,
            "external_independent_reader_required": [
                "byte-exact same initial arrays and strict unique final policy",
                "source/checkpoint/ELF hash and C/Python 5T closure",
                "actual controller/native force and compiled ramp/wheel geometry",
            ]},
        "cost_gate": cost, "branch_a_task_conversion": branch_a,
        "branch_b_sse_improvement": branch_b,
        "RL_contribution_passed": bool(contribution),
        "all_six_policy_task_qualified": all(pair["policy_task_passed"] for pair in pairs),
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "not_established_here": [
            "statistical reliability from one seed per pair", "real hardware",
            "complete course traversal beyond the six scheduled tasks",
            "independent model/source/physics provenance",
        ],
    }
