"""Pure numeric scorer for the twelve preregistered 08-R heldout episodes.

This module is arithmetic only: stdlib + NumPy.  It constructs no model, imports
no engine/policy/worker source, replays no physics and re-derives no geometry.
A separate offline reader (owned by root / publication_handoff) parses the saved
episode folders written by ``run_rl16_training_08._heldout_case``
(``schedule.json``, ``reset_receipt.json``, ``initial_state.npz``, ``states.npz``,
``control_records_*.jsonl.gz``, ``native_block_*.jsonl.gz``, ``case_receipt.json``),
verifies raw source/checkpoint/dependency hashes, the actual five native
substeps per control, the controller->requested->applied torque chain, the
C/Python budgets, reset identity and the byte-exact zero/final_policy initial
arrays, and then hands this module *canonical numeric arrays*.  Everything this
module returns is therefore conditional on that separate reader's receipt; see
``EVIDENCE_BOUNDARY`` and the ``evidence_boundary`` field of every result.

Authoritative definitions: ``next_rl_pilot_contract_08.md`` section 5,
``rl_schedule_geometry_addendum_08_01.md``, ``rl_semantics_and_terrain_addendum_08_02.md``
and ``rl_heldout_scoring_addendum_08_05.md``.  No threshold, window or formula
here may be relaxed after seeing T results.

Public API
----------
``score_case(canonical_case) -> dict``
    Scores one episode (one ``case_id``/``actor``).  Record integrity, physical
    qualification, speed/yaw metrics and the RL contribution terms are kept in
    separate blocks and never substituted for one another.
``score_pairs(case_scores) -> dict``
    Aggregates the twelve ``score_case`` results into the six same-seed
    zero/final_policy pairs and evaluates branch A / branch B and the cost gate.

Indexing and units (canonical contract with the reader)
-------------------------------------------------------
Control tick ``t`` is zero based.  ``N = completed_controls`` (1..1600) is the
number of *actually completed* controls; a valid early task termination has
``N < 1600`` and is scored on its observed prefix only - nothing is padded or
extrapolated.  For every per-control array, index ``k`` holds the value of the
true **post** state of control tick ``k`` (engine time ``(k + 1) * 0.01`` s);
pre-state quantities are never mixed in.  The target for tick ``k`` is the servo
value *already consumed* at tick ``k``.  Native sample index ``5 * t + i``
(``i`` in 0..4) is the ``i``-th actual native return of control tick ``t``; the
native arrays therefore have length ``5 * N``.  Angles are radians unless the
key says ``_deg``; velocities m/s, rates rad/s, lengths m, times s, torques N*m.

Required canonical keys
-----------------------
Identity (immutable, from the saved receipts):
  ``case_id``   one of the six preregistered names (``HELDOUT_CASES``)
  ``actor``     ``"zero"`` or ``"final_policy"``
  ``seed``      int, must equal the preregistered seed of ``case_id``
  ``terrain``   must equal the preregistered lane of ``case_id``
Counts / termination:
  ``completed_controls``            int in [1, 1600]
  ``terminated``, ``truncated``     bool, from the actual gym transition
  ``stop_reason``                   str (``info["terminal_reason"]`` or cap)
  ``warning_count``                 int, engine warning callbacks (must be 0)
  ``geometry_invalid_count``        int, native guard geometry failures
  ``map_escape_count``              int, out-of-course terminations
  ``fall_count``                    int, task-fall events
Per-control arrays, shape ``(N,)`` float64 finite:
  ``tick_index``                    int array, must equal ``arange(N)``
  ``applied_servo_vx_mps``          consumed applied servo forward of tick k
  ``applied_servo_yaw_rps``         consumed applied servo yaw rate of tick k
  ``com_vx_mps``                    actual post body-frame forward component of
                                    the base COM linear velocity (never wheel
                                    r*omega, never a raw command)
  ``body_yaw_rate_rps``             actual post body-frame angular velocity z
  ``heading_rad``                   actual post base heading (yaw), unwrapped ok
  ``clearance_m``                   post base world z minus compiled terrain
                                    height under the same XY
  ``lateral_offset_m``              signed post offset from the spawn forward
                                    line (scorer uses the absolute value)
  ``forward_projection_m``          ``h0 . (p_post - p_spawn)`` with ``h0`` the
                                    world horizontal unit vector of the spawn
                                    heading
Per-control torque, shape ``(N, 16, 5)`` float64 finite:
  ``motor_torque_nm``               actual applied motor torque of every one of
                                    the five native substeps, leg-major with
                                    ``j % 4 == 3`` the wheel (limits
                                    ``[80, 80, 80, 12]`` repeated four times)
Native arrays, shape ``(5 * N,)``:
  ``native_index``                  int array, must equal ``arange(5 * N)``
  ``native_roll_deg``, ``native_pitch_deg``          absolute post-native pose
  ``native_terrain_relative_tilt_deg``               angle between the base up
                                    axis and the local compiled terrain normal
  ``native_clearance_m``            post-native clearance
  ``native_forward_projection_m``   post-native spawn-axis projection
  ``native_nonwheel_ground_candidate_count``         int array, per native
                                    return count of nonwheel ground candidates
Initial (reset) sample, scalars:
  ``initial_roll_deg``, ``initial_pitch_deg``,
  ``initial_terrain_relative_tilt_deg``, ``initial_clearance_m``
Contact family evidence:
  ``positive_wheel_load_native_counts_by_family``    mapping family -> int count
                                    of native returns with at least one
                                    genuinely positive wheel normal load, bound
                                    by ``mj_contactForce`` and compiled geom
                                    identity by the reader
Optional, echoed but never used as evidence:
  ``reader_asserted``               mapping of the reader's own booleans
                                    (``geometry_passed``, ``controller_verified``,
                                    ``native_5T``, ``source_hash_ok``,
                                    ``initial_pair_exact``, ...)

Malformed input (absent key, wrong shape/dtype, NaN/Inf, duplicated or missing
tick/native index, unknown or mismatched case/actor/seed, arrays shorter than a
claimed full-length horizon) raises ``KeyError``/``TypeError``/``ValueError``.
That is input corruption, not a physical result, and is deliberately kept
distinct from ``record.record_valid`` (engine-record integrity) and from
``task_passed`` (a legitimate task failure).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

SCORE_SCHEMA = "d1-course-rl16-heldout-pure-score-08-v1"
PAIR_SCHEMA = "d1-course-rl16-heldout-pure-pair-score-08-v1"
SCORING_ADDENDUM = "rl_heldout_scoring_addendum_08_05.md"

CONTROL_DT_S = 0.01
CONTROL_CAP = 1600
SETTLE_TICKS = 175
HOLD_TICKS = 400
SERVO_RATE_MPS_PER_TICK = 0.005
FINAL_WINDOW = (1500, 1600)
SETTLED_RUN_TICKS = 25

# case_id -> (terrain, target speed m/s, yaw amplitude rad/s, seed)
HELDOUT_CASES: dict[str, tuple[str, float, float, int]] = {
    "flat_0p6": ("flat", 0.6, 0.0, 88501),
    "flat_1p6": ("flat", 1.6, 0.0, 88502),
    "flat_1p2_yaw": ("flat", 1.2, 0.3, 88503),
    "bumps_0p4": ("bumps", 0.4, 0.0, 88504),
    "rough_0p35": ("rough", 0.35, 0.0, 88505),
    "ramp_0p35": ("ramp", 0.35, 0.0, 88506),
}
ACTORS = ("zero", "final_policy")
# Native guard family label of the obstacle lane that each terrain case must load.
TERRAIN_FAMILY = {"bumps": "bump", "rough": "rough", "ramp": "ramp"}

FLAT_MAX_ABS_POSE_DEG = 10.0
TERRAIN_MAX_RELATIVE_TILT_DEG = 10.0
TERRAIN_MAX_ABS_PITCH_DEG = 20.0
TERRAIN_MAX_ABS_ROLL_DEG = 12.0
MIN_CLEARANCE_M = 0.28
FINAL_MAX_MEAN_ABS_COM_VX_MPS = 0.05
FINAL_MAX_MEAN_ABS_YAW_RATE_RPS = 0.05
FINAL_MAX_CLEARANCE_STD_M = 0.02
HIGH_SPEED_MEAN_BAND_MPS = (1.52, 1.68)
HIGH_SPEED_MAX_RMS_MPS = 0.08
HIGH_SPEED_MIN_FRACTION_ABOVE = 0.90
HIGH_SPEED_FRACTION_THRESHOLD_MPS = 1.5
STRAIGHT_MAX_RMS_MPS = 0.05
STRAIGHT_MEAN_TOLERANCE_MPS = 0.04
YAW_MAX_RMS_RPS = 0.12
STOP_SETTLE_MAX_S = 4.2
STOP_DISTANCE_MAX_M = 3.2
STOP_QUALIFY_ABS_COM_VX_MPS = 0.05
STOP_QUALIFY_ABS_YAW_RATE_RPS = 0.05
TERRAIN_MAX_LATERAL_OFFSET_M = 0.25
TERRAIN_MAX_ABS_WRAPPED_HEADING_RAD = 0.15
TERRAIN_MIN_PROJECTION_FRACTION = 0.75

SSE_VX_SCALE_MPS = 0.25
SSE_YAW_SCALE_RPS = 0.4
TORQUE_LIMITS_NM = np.array([80.0, 80.0, 80.0, 12.0] * 4, dtype=np.float64)

BRANCH_B_MIN_PAIRS = 4
BRANCH_B_POOLED_SSE_FRACTION = 0.85
BRANCH_B_PER_PAIR_SSE_TOLERANCE = 1.02
BRANCH_B_MIN_NO_WORSE_PAIRS = 4
COST_GATE_FRACTION = 1.20

EVIDENCE_BOUNDARY = (
    "This module is a pure arithmetic scorer over canonical numeric arrays. It "
    "establishes no provenance: raw source/checkpoint/dependency hashes, the "
    "five actual native integrator/contactForce returns per control, the "
    "controller->requested->applied torque link, the C/Python budget ledgers, "
    "reset identity and the byte-exact paired initial arrays are established "
    "only by the separate offline reader owned by root/publication_handoff, "
    "whose result is bound by SHA256. Any boolean supplied in 'reader_asserted' "
    "(geometry_passed, controller_verified, native_5T, source_hash_ok, "
    "initial_pair_exact, ...) is echoed verbatim and is never used as a gate, "
    "an input to any metric, or independent evidence here."
)
NOT_ESTABLISHED_HERE = (
    "statistical significance (single seed, six pairs)",
    "hardware or cross-terrain robustness",
    "complete ramp/bump/rough course traversal",
    "any RL-specific gain when the zero baseline also passes",
    "mechanical energy or electrical consumption (c is a normalized torque-squared proxy)",
    "record provenance, source/checkpoint identity and native/budget counts",
)


# --------------------------------------------------------------------------- #
# canonical input validation
# --------------------------------------------------------------------------- #
def _field(case: Mapping[str, object], key: str) -> object:
    if not isinstance(case, Mapping):
        raise TypeError("canonical case must be a mapping")
    if key not in case:
        raise KeyError("canonical case lacks required key: " + key)
    return case[key]


def _text(case: Mapping[str, object], key: str) -> str:
    value = _field(case, key)
    if type(value) is not str or not value:
        raise TypeError(key + " must be a nonempty string")
    return value


def _flag(case: Mapping[str, object], key: str) -> bool:
    value = _field(case, key)
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(key + " must be a boolean")
    return bool(value)


def _count(case: Mapping[str, object], key: str) -> int:
    value = _field(case, key)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(key + " must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(key + " must be nonnegative")
    return result


def _scalar(case: Mapping[str, object], key: str) -> float:
    value = _field(case, key)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(key + " must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(key + " must be finite")
    return result


def _reals(case: Mapping[str, object], key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(_field(case, key))
    if value.dtype.kind not in "fiu":
        raise TypeError(key + " must be a real numeric array")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, actual {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(key + " contains NaN or Inf")
    return array


def _indices(case: Mapping[str, object], key: str, size: int) -> np.ndarray:
    value = np.asarray(_field(case, key))
    if value.dtype.kind not in "iu":
        raise TypeError(key + " must be an integer index array")
    if value.shape != (size,):
        raise ValueError(f"{key} must have shape {(size,)}, actual {value.shape}")
    if not np.array_equal(value, np.arange(size)):
        raise ValueError(key + " has duplicated, reordered or missing entries")
    return np.asarray(value, dtype=np.int64)


def _nonneg_counts(case: Mapping[str, object], key: str, size: int) -> np.ndarray:
    value = np.asarray(_field(case, key))
    if value.dtype.kind not in "iu":
        raise TypeError(key + " must be an integer count array")
    if value.shape != (size,):
        raise ValueError(f"{key} must have shape {(size,)}, actual {value.shape}")
    if (value < 0).any():
        raise ValueError(key + " must be nonnegative")
    return np.asarray(value, dtype=np.int64)


def _family_counts(case: Mapping[str, object]) -> dict[str, int]:
    value = _field(case, "positive_wheel_load_native_counts_by_family")
    if not isinstance(value, Mapping):
        raise TypeError("positive_wheel_load_native_counts_by_family must be a mapping")
    counts: dict[str, int] = {}
    for family, count in value.items():
        if type(family) is not str:
            raise TypeError("positive wheel load family keys must be strings")
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
            raise TypeError("positive wheel load counts must be integers")
        if int(count) < 0:
            raise ValueError("positive wheel load counts must be nonnegative")
        counts[family] = int(count)
    return counts


def _number(value: Any) -> float:
    """Return a plain finite float; never a NumPy scalar, NaN or Inf."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("scorer produced a nonfinite value")
    return result


def _mean_or_none(array: np.ndarray) -> float | None:
    return None if array.size == 0 else _number(np.mean(array))


def _wrap(angle: float) -> float:
    return _number((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def ramp_ticks(target_speed_mps: float) -> int:
    """Preregistered servo ramp length, identical to ``full_drive_schedule_08``."""
    return math.ceil(float(target_speed_mps) / SERVO_RATE_MPS_PER_TICK - 1e-12)


def case_windows(case_id: str) -> dict[str, Any]:
    """Exact preregistered windows of one case, independent of any result."""
    if case_id not in HELDOUT_CASES:
        raise ValueError("unknown preregistered heldout case: " + str(case_id))
    _, speed, _, _ = HELDOUT_CASES[case_id]
    ticks = ramp_ticks(speed)
    hold_begin = SETTLE_TICKS + ticks
    hold_end = hold_begin + HOLD_TICKS
    return {
        "ramp_ticks": ticks,
        "hold": [hold_begin, hold_end],
        "release_tick_t0": hold_end,
        "final": [FINAL_WINDOW[0], FINAL_WINDOW[1]],
        "drive": [SETTLE_TICKS, hold_end],
    }


def _observed(window: Sequence[int], completed: int) -> int:
    begin, end = int(window[0]), int(window[1])
    return max(0, min(end, completed) - begin)


# --------------------------------------------------------------------------- #
# single case
# --------------------------------------------------------------------------- #
def score_case(canonical_case: Mapping[str, object]) -> dict[str, Any]:
    """Score one heldout episode from canonical numeric arrays.

    Returns JSON-serializable numbers, booleans, strings and ``None`` only.
    Blocks are deliberately separate: ``record`` (engine-record integrity),
    ``safety``/``speed``/``yaw``/``final_window``/``stopping``/``terrain_channel``
    (physical qualification), and ``rl_terms`` (local RL contribution inputs).
    A valid early termination scores the observed prefix and fails any
    unobserved required hold/stop gate; nothing is padded or extrapolated.
    """
    case_id = _text(canonical_case, "case_id")
    if case_id not in HELDOUT_CASES:
        raise ValueError("unknown preregistered heldout case: " + case_id)
    terrain, target_speed, yaw_amplitude, seed = HELDOUT_CASES[case_id]
    actor = _text(canonical_case, "actor")
    if actor not in ACTORS:
        raise ValueError("actor must be one of the frozen matched pair: " + actor)
    if _count(canonical_case, "seed") != seed:
        raise ValueError("case seed differs from the preregistered pair seed")
    if _text(canonical_case, "terrain") != terrain:
        raise ValueError("case terrain differs from the preregistered course lane")

    completed = _count(canonical_case, "completed_controls")
    if not 1 <= completed <= CONTROL_CAP:
        raise ValueError("completed_controls must be in [1, 1600]")
    natives = 5 * completed

    _indices(canonical_case, "tick_index", completed)
    _indices(canonical_case, "native_index", natives)
    servo_vx = _reals(canonical_case, "applied_servo_vx_mps", (completed,))
    servo_yaw = _reals(canonical_case, "applied_servo_yaw_rps", (completed,))
    com_vx = _reals(canonical_case, "com_vx_mps", (completed,))
    yaw_rate = _reals(canonical_case, "body_yaw_rate_rps", (completed,))
    heading = _reals(canonical_case, "heading_rad", (completed,))
    clearance = _reals(canonical_case, "clearance_m", (completed,))
    lateral = _reals(canonical_case, "lateral_offset_m", (completed,))
    projection = _reals(canonical_case, "forward_projection_m", (completed,))
    torque = _reals(canonical_case, "motor_torque_nm", (completed, 16, 5))
    native_roll = _reals(canonical_case, "native_roll_deg", (natives,))
    native_pitch = _reals(canonical_case, "native_pitch_deg", (natives,))
    native_tilt = _reals(canonical_case, "native_terrain_relative_tilt_deg", (natives,))
    native_clearance = _reals(canonical_case, "native_clearance_m", (natives,))
    native_projection = _reals(canonical_case, "native_forward_projection_m", (natives,))
    native_nonwheel = _nonneg_counts(
        canonical_case, "native_nonwheel_ground_candidate_count", natives
    )
    initial_roll = _scalar(canonical_case, "initial_roll_deg")
    initial_pitch = _scalar(canonical_case, "initial_pitch_deg")
    initial_tilt = _scalar(canonical_case, "initial_terrain_relative_tilt_deg")
    initial_clearance = _scalar(canonical_case, "initial_clearance_m")
    load_counts = _family_counts(canonical_case)

    terminated = _flag(canonical_case, "terminated")
    truncated = _flag(canonical_case, "truncated")
    stop_reason = _text(canonical_case, "stop_reason")
    warnings = _count(canonical_case, "warning_count")
    geometry_invalid = _count(canonical_case, "geometry_invalid_count")
    map_escapes = _count(canonical_case, "map_escape_count")
    falls = _count(canonical_case, "fall_count")
    asserted = _field(canonical_case, "reader_asserted") if "reader_asserted" in canonical_case else {}
    if not isinstance(asserted, Mapping):
        raise TypeError("reader_asserted must be a mapping when present")

    windows = case_windows(case_id)
    hold_begin, hold_end = windows["hold"]
    t0 = windows["release_tick_t0"]
    record = _record_block(completed, terminated, truncated, stop_reason)
    full_horizon = record["full_horizon"]

    safety = _safety_block(
        terrain, native_roll, native_pitch, native_tilt, native_clearance,
        native_nonwheel, initial_roll, initial_pitch, initial_tilt,
        initial_clearance, warnings, geometry_invalid, map_escapes, falls,
    )
    speed = _speed_block(
        case_id, target_speed, completed, hold_begin, hold_end, com_vx, servo_vx,
    )
    yaw = _yaw_block(
        yaw_amplitude, completed, hold_begin, hold_end, yaw_rate, servo_yaw,
    )
    final_window = _final_block(completed, com_vx, yaw_rate, clearance)
    stopping = _stop_block(
        case_id, completed, t0, com_vx, yaw_rate, servo_vx, projection,
        native_projection,
    )
    channel = _terrain_block(
        terrain, completed, servo_vx, lateral, heading, projection, load_counts,
    )
    rl_terms = _rl_terms_block(
        completed, t0, com_vx, yaw_rate, servo_vx, servo_yaw, torque,
    )

    task_reasons: list[str] = []
    if not record["record_valid"]:
        task_reasons.append("record integrity failed")
    if not full_horizon:
        task_reasons.append(
            f"observed {completed} of 1600 controls; a task_passed episode needs the "
            "complete horizon without a task termination"
        )
    for name, block in (
        ("safety", safety), ("speed", speed), ("yaw", yaw),
        ("final_window", final_window), ("stopping", stopping),
        ("terrain_channel", channel),
    ):
        if block["passed"] is False:
            task_reasons.extend(name + ": " + reason for reason in block["reasons"])
    task_passed = not task_reasons

    return {
        "schema": SCORE_SCHEMA,
        "scoring_addendum": SCORING_ADDENDUM,
        "case_id": case_id,
        "actor": actor,
        "seed": seed,
        "terrain": terrain,
        "target_speed_mps": _number(target_speed),
        "yaw_amplitude_rps": _number(yaw_amplitude),
        "completed_controls": completed,
        "native_returns_expected_from_completed": natives,
        "terminated": terminated,
        "truncated": truncated,
        "stop_reason": stop_reason,
        "windows": {
            **windows,
            "hold_observed_ticks": _observed(windows["hold"], completed),
            "final_observed_ticks": _observed(windows["final"], completed),
            "drive_observed_ticks": _observed(windows["drive"], completed),
        },
        "record": record,
        "safety": safety,
        "speed": speed,
        "yaw": yaw,
        "final_window": final_window,
        "stopping": stopping,
        "terrain_channel": channel,
        "task_passed": task_passed,
        "task_reasons": task_reasons,
        "rl_terms": rl_terms,
        "reader_asserted_unverified": {str(key): value for key, value in asserted.items()},
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "not_established_here": list(NOT_ESTABLISHED_HERE),
    }


def _record_block(completed: int, terminated: bool, truncated: bool,
                  stop_reason: str) -> dict[str, Any]:
    """Engine-record integrity only; a genuine task failure is not corruption."""
    reasons: list[str] = []
    if terminated and truncated:
        reasons.append("a terminal transition cannot be both terminated and truncated")
    if not (terminated or truncated):
        reasons.append(
            "episode stopped without a genuine terminal transition "
            f"(completed {completed}, stop_reason {stop_reason!r})"
        )
    if completed < CONTROL_CAP and truncated and not terminated:
        reasons.append(
            f"truncation without termination before the 1600 cap at {completed} controls"
        )
    if stop_reason == "exception":
        reasons.append("worker recorded an exception rather than a physical terminal")
    return {
        "record_valid": not reasons,
        "reasons": reasons,
        "full_horizon": (not reasons) and completed == CONTROL_CAP and not terminated,
        "early_task_termination": (not reasons) and terminated,
        "note": (
            "a valid early task termination keeps record_valid=True and "
            "task_passed=False; it is never padded, extrapolated or re-labelled "
            "as record corruption"
        ),
    }


def _safety_block(
    terrain: str, native_roll: np.ndarray, native_pitch: np.ndarray,
    native_tilt: np.ndarray, native_clearance: np.ndarray,
    native_nonwheel: np.ndarray, initial_roll: float, initial_pitch: float,
    initial_tilt: float, initial_clearance: float, warnings: int,
    geometry_invalid: int, map_escapes: int, falls: int,
) -> dict[str, Any]:
    """All-time gates over the initial state and every actual native return."""
    max_roll = _number(max(float(np.max(np.abs(native_roll))), abs(initial_roll)))
    max_pitch = _number(max(float(np.max(np.abs(native_pitch))), abs(initial_pitch)))
    max_tilt = _number(max(float(np.max(np.abs(native_tilt))), abs(initial_tilt)))
    min_clearance = _number(min(float(np.min(native_clearance)), initial_clearance))
    nonwheel_total = int(np.sum(native_nonwheel))
    reasons: list[str] = []
    if terrain == "flat":
        if max_roll > FLAT_MAX_ABS_POSE_DEG:
            reasons.append(f"flat absolute roll {max_roll} deg exceeds 10 deg")
        if max_pitch > FLAT_MAX_ABS_POSE_DEG:
            reasons.append(f"flat absolute pitch {max_pitch} deg exceeds 10 deg")
    else:
        if max_tilt > TERRAIN_MAX_RELATIVE_TILT_DEG:
            reasons.append(f"terrain relative normal tilt {max_tilt} deg exceeds 10 deg")
        if max_pitch > TERRAIN_MAX_ABS_PITCH_DEG:
            reasons.append(f"terrain absolute pitch {max_pitch} deg exceeds 20 deg")
        if max_roll > TERRAIN_MAX_ABS_ROLL_DEG:
            reasons.append(f"terrain absolute roll {max_roll} deg exceeds 12 deg")
    if nonwheel_total:
        reasons.append(f"{nonwheel_total} nonwheel ground contact candidates")
    if min_clearance < MIN_CLEARANCE_M:
        reasons.append(f"all-time clearance {min_clearance} m below 0.28 m")
    for count, label in ((falls, "task fall"), (warnings, "engine warning"),
                         (geometry_invalid, "invalid geometry"),
                         (map_escapes, "map escape")):
        if count:
            reasons.append(f"{count} {label} event(s)")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "max_abs_roll_deg": max_roll,
        "max_abs_pitch_deg": max_pitch,
        "max_terrain_relative_tilt_deg": max_tilt,
        "min_clearance_m": min_clearance,
        "nonwheel_ground_candidate_total": nonwheel_total,
        "fall_count": falls,
        "warning_count": warnings,
        "geometry_invalid_count": geometry_invalid,
        "map_escape_count": map_escapes,
        "native_samples_checked": int(native_roll.size),
        "initial_state_included": True,
    }


def _speed_block(
    case_id: str, target_speed: float, completed: int, hold_begin: int,
    hold_end: int, com_vx: np.ndarray, servo_vx: np.ndarray,
) -> dict[str, Any]:
    """Longitudinal qualification on the exact 400-tick hold window."""
    stop = min(hold_end, completed)
    observed = max(0, stop - hold_begin)
    hold = com_vx[hold_begin:stop] if observed > 0 else com_vx[:0]
    servo_hold = servo_vx[hold_begin:stop] if observed > 0 else servo_vx[:0]
    mean = _mean_or_none(hold)
    rms = None if observed == 0 else _number(
        math.sqrt(float(np.mean((hold - target_speed) ** 2)))
    )
    servo_rms = None if observed == 0 else _number(
        math.sqrt(float(np.mean((hold - servo_hold) ** 2)))
    )
    above = int(np.sum(hold > HIGH_SPEED_FRACTION_THRESHOLD_MPS))
    fraction = _number(above / HOLD_TICKS)
    reasons: list[str] = []
    if observed < HOLD_TICKS:
        reasons.append(
            f"hold window [{hold_begin},{hold_end}) observed only {observed} of 400 "
            "post samples; the complete hold is required"
        )
    if case_id == "flat_1p6":
        low, high = HIGH_SPEED_MEAN_BAND_MPS
        if mean is None or not low <= mean <= high:
            reasons.append(f"hold mean COM vx {mean} m/s outside [1.52, 1.68]")
        if rms is None or rms > HIGH_SPEED_MAX_RMS_MPS:
            reasons.append(f"hold COM vx RMS error {rms} m/s exceeds 0.08")
        if fraction < HIGH_SPEED_MIN_FRACTION_ABOVE:
            reasons.append(
                f"proportion of hold ticks with COM vx > 1.5 m/s is {fraction} "
                "(count / 400) below 0.90"
            )
    else:
        if rms is None or rms > STRAIGHT_MAX_RMS_MPS:
            reasons.append(f"hold COM vx RMS error {rms} m/s exceeds 0.05")
        if mean is None or abs(mean - target_speed) > STRAIGHT_MEAN_TOLERANCE_MPS:
            reasons.append(
                f"hold mean COM vx {mean} m/s outside target {target_speed} +/- 0.04"
            )
    return {
        "passed": not reasons,
        "reasons": reasons,
        "hold_window": [hold_begin, hold_end],
        "hold_observed_ticks": observed,
        "hold_complete": observed == HOLD_TICKS,
        "mean_com_vx_mps": mean,
        "rms_com_vx_error_vs_target_mps": rms,
        "rms_com_vx_error_vs_applied_servo_mps": servo_rms,
        "ticks_above_1p5_mps": above,
        "fraction_above_1p5_mps_over_400": fraction,
        "gate": "flat_1p6" if case_id == "flat_1p6" else "straight_hold",
        "source": "actual post body COM vx versus the servo already consumed at that tick",
    }


def _yaw_block(
    yaw_amplitude: float, completed: int, hold_begin: int, hold_end: int,
    yaw_rate: np.ndarray, servo_yaw: np.ndarray,
) -> dict[str, Any]:
    """Signed yaw qualification; applicable to flat_1p2_yaw only."""
    if yaw_amplitude == 0.0:
        return {
            "applicable": False,
            "passed": None,
            "reasons": [],
            "note": "no yaw hold is preregistered for this case",
        }
    stop = min(hold_end, completed)
    observed = max(0, stop - hold_begin)
    rate = yaw_rate[hold_begin:stop] if observed > 0 else yaw_rate[:0]
    servo = servo_yaw[hold_begin:stop] if observed > 0 else servo_yaw[:0]
    rms = None if observed == 0 else _number(
        math.sqrt(float(np.mean((rate - servo) ** 2)))
    )
    positive = _number(float(np.sum(rate[servo > 0.0])) * CONTROL_DT_S) if observed else None
    negative = _number(float(np.sum(rate[servo < 0.0])) * CONTROL_DT_S) if observed else None
    final = yaw_rate[FINAL_WINDOW[0]:min(FINAL_WINDOW[1], completed)]
    final_mean = None if final.size == 0 else _number(float(np.mean(np.abs(final))))
    reasons: list[str] = []
    if observed < HOLD_TICKS:
        reasons.append(
            f"yaw hold observed only {observed} of 400 post samples; the complete "
            "100/200/100 signed hold is required"
        )
    if rms is None or rms > YAW_MAX_RMS_RPS:
        reasons.append(f"yaw rate RMS error {rms} rad/s exceeds 0.12")
    if positive is None or positive <= 0.0:
        reasons.append(
            f"integrated yaw over positive-servo ticks is {positive} rad; it must be > 0"
        )
    if negative is None or negative >= 0.0:
        reasons.append(
            f"integrated yaw over negative-servo ticks is {negative} rad; it must be < 0"
        )
    if final_mean is None or final_mean > FINAL_MAX_MEAN_ABS_YAW_RATE_RPS:
        reasons.append(
            f"final-window mean |yaw rate| {final_mean} rad/s exceeds 0.05"
        )
    return {
        "applicable": True,
        "passed": not reasons,
        "reasons": reasons,
        "hold_observed_ticks": observed,
        "rms_yaw_rate_error_vs_applied_servo_rps": rms,
        "positive_servo_integrated_yaw_rad": positive,
        "negative_servo_integrated_yaw_rad": negative,
        "positive_servo_ticks": int(np.sum(servo > 0.0)),
        "negative_servo_ticks": int(np.sum(servo < 0.0)),
        "final_mean_abs_yaw_rate_rps": final_mean,
        "note": "net angle near zero is not accepted in place of both signed integrals",
    }


def _final_block(completed: int, com_vx: np.ndarray, yaw_rate: np.ndarray,
                 clearance: np.ndarray) -> dict[str, Any]:
    """The preregistered final 100 post samples, window F = [1500, 1600)."""
    begin, end = FINAL_WINDOW
    stop = min(end, completed)
    observed = max(0, stop - begin)
    vx = com_vx[begin:stop] if observed > 0 else com_vx[:0]
    rate = yaw_rate[begin:stop] if observed > 0 else yaw_rate[:0]
    height = clearance[begin:stop] if observed > 0 else clearance[:0]
    mean_vx = None if observed == 0 else _number(float(np.mean(np.abs(vx))))
    mean_yaw = None if observed == 0 else _number(float(np.mean(np.abs(rate))))
    std = None if observed == 0 else _number(float(np.std(height, ddof=0)))
    reasons: list[str] = []
    if observed < end - begin:
        reasons.append(
            f"final window [{begin},{end}) observed only {observed} of 100 post samples"
        )
    if mean_vx is None or mean_vx > FINAL_MAX_MEAN_ABS_COM_VX_MPS:
        reasons.append(f"final mean |COM vx| {mean_vx} m/s exceeds 0.05")
    if mean_yaw is None or mean_yaw > FINAL_MAX_MEAN_ABS_YAW_RATE_RPS:
        reasons.append(f"final mean |yaw rate| {mean_yaw} rad/s exceeds 0.05")
    if std is None or std > FINAL_MAX_CLEARANCE_STD_M:
        reasons.append(f"final clearance population std {std} m exceeds 0.02")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "window": [begin, end],
        "observed_ticks": observed,
        "mean_abs_com_vx_mps": mean_vx,
        "mean_abs_yaw_rate_rps": mean_yaw,
        "clearance_population_std_m": std,
        "note": "no vertical-velocity gate exists; height stability is this std plus "
                "the all-time clearance and posture gates",
    }


def _stop_block(
    case_id: str, completed: int, t0: int, com_vx: np.ndarray,
    yaw_rate: np.ndarray, servo_vx: np.ndarray, projection: np.ndarray,
    native_projection: np.ndarray,
) -> dict[str, Any]:
    """Earliest 25 consecutive qualifying post endpoints at or after t0.

    The distance is the maximum positive spawn-axis projection over EVERY saved
    native return from the release pre-state up to the settled endpoint, so an
    overshoot followed by a rollback cannot cancel itself out.
    """
    qualify = (np.abs(com_vx) <= STOP_QUALIFY_ABS_COM_VX_MPS) & (
        np.abs(yaw_rate) <= STOP_QUALIFY_ABS_YAW_RATE_RPS
    )
    settled: int | None = None
    run = 0
    for tick in range(t0, completed):
        run = run + 1 if qualify[tick] else 0
        if run == SETTLED_RUN_TICKS:
            settled = tick
            break
    # The release pre-state at t0 is the post state of tick t0 - 1.
    release = _number(projection[t0 - 1]) if 0 < t0 <= completed else None
    time_to_settle = None if settled is None else _number((settled + 1 - t0) * CONTROL_DT_S)
    excursion = None
    endpoint = None
    if settled is not None and release is not None:
        span = native_projection[5 * t0:5 * settled + 5] - release
        # max(0, ...) including the release pre point itself (projection 0).
        excursion = _number(max(0.0, float(np.max(span)) if span.size else 0.0))
        endpoint = _number(projection[settled] - release)
    net = None if release is None else _number(projection[completed - 1] - release)
    servo_zero = None
    if t0 < completed:
        zeros = np.flatnonzero(servo_vx[t0:completed] == 0.0)
        if zeros.size:
            servo_zero = int(t0 + zeros[0])
    gate_applies = case_id == "flat_1p6"
    reasons: list[str] = []
    if gate_applies:
        if settled is None:
            reasons.append(
                "no 25 consecutive qualifying post endpoints at or after t0; a "
                "truncated last sample is not accepted as a stop"
            )
        else:
            if time_to_settle is None or time_to_settle > STOP_SETTLE_MAX_S:
                reasons.append(f"time to settle {time_to_settle} s exceeds 4.2 s")
            if excursion is None or excursion > STOP_DISTANCE_MAX_M:
                reasons.append(
                    f"maximum native forward excursion {excursion} m exceeds 3.2 m"
                )
    return {
        "passed": not reasons,
        "reasons": reasons,
        "gate_applies": gate_applies,
        "release_tick_t0": t0,
        "release_pre_projection_m": release,
        "found_settled_window": settled is not None,
        "settled_tick": settled,
        "time_to_settle_s": time_to_settle,
        "max_native_forward_excursion_m": excursion,
        "endpoint_projection_m": endpoint,
        "final_net_projection_m": net,
        "applied_servo_zero_tick": servo_zero,
        "notes": [
            ("the 4.2 s / 3.2 m gate is preregistered for flat_1p6 only; the other "
             "five cases report the same quantities without that gate and still "
             "have to pass the final-window stop gate"),
            ("the rate-limited brake after raw release is executed by the baseline "
             "controller with zero residual and is not credited to RL"),
        ],
    }


def _terrain_block(
    terrain: str, completed: int, servo_vx: np.ndarray,
    lateral: np.ndarray, heading: np.ndarray, projection: np.ndarray,
    load_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Directional and load channel gates for bumps/rough/ramp (addendum 02)."""
    if terrain == "flat":
        return {
            "applicable": False,
            "passed": None,
            "reasons": [],
            "note": "the lateral/heading/projection/load channel applies to the "
                    "three terrain cases",
        }
    # The terrain channel covers the whole observed episode. Settling cannot
    # erase a later lateral/heading drift or truncate the servo-distance debt.
    end = completed - 1
    span = slice(0, completed)
    max_lateral = _number(float(np.max(np.abs(lateral[span]))))
    max_heading = _number(max(abs(_wrap(float(value))) for value in heading[span]))
    net = _number(projection[end])
    required = _number(
        TERRAIN_MIN_PROJECTION_FRACTION
        * float(np.sum(np.maximum(servo_vx[span], 0.0))) * CONTROL_DT_S
    )
    family = TERRAIN_FAMILY[terrain]
    loads = int(load_counts.get(family, 0))
    reasons: list[str] = []
    if max_lateral > TERRAIN_MAX_LATERAL_OFFSET_M:
        reasons.append(f"lateral offset {max_lateral} m exceeds 0.25 m")
    if max_heading > TERRAIN_MAX_ABS_WRAPPED_HEADING_RAD:
        reasons.append(f"wrapped heading {max_heading} rad exceeds 0.15 rad")
    if net < required:
        reasons.append(
            f"net forward projection {net} m below 0.75 x integrated positive "
            f"consumed servo distance ({required} m)"
        )
    if loads < 1:
        reasons.append(
            f"no native return with a genuinely positive wheel load on the "
            f"terrain_{family}_ family"
        )
    return {
        "applicable": True,
        "passed": not reasons,
        "reasons": reasons,
        "window": [0, end + 1],
        "observed_ticks": end + 1,
        "max_abs_lateral_offset_m": max_lateral,
        "max_abs_wrapped_heading_rad": max_heading,
        "net_forward_projection_m": net,
        "required_forward_projection_m": required,
        "family": family,
        "positive_wheel_load_native_returns": loads,
        "positive_wheel_load_counts_by_family": {
            str(key): int(value) for key, value in load_counts.items()
        },
        "note": "limited progress on this lane is not a complete ramp/step/rough "
                "course traversal claim",
    }


def _rl_terms_block(
    completed: int, t0: int, com_vx: np.ndarray, yaw_rate: np.ndarray,
    servo_vx: np.ndarray, servo_yaw: np.ndarray, torque: np.ndarray,
) -> dict[str, Any]:
    """Drive-window SSE and normalized torque-squared cost of D = [175, t0).

    ``e[t] = ((post COM vx - servo vx) / .25)^2 + ((post yaw rate - servo yaw) / .4)^2``
    ``c[t] = mean over 16 motors and five native returns of
    (actual applied torque / limit)^2``. Squaring precedes both averages, so
    alternating native torque signs cannot cancel. Both axes count in every case.
    """
    stop = min(t0, completed)
    begin = SETTLE_TICKS
    observed = max(0, stop - begin)
    span = slice(begin, stop) if observed > 0 else slice(0, 0)
    vx_error = (com_vx[span] - servo_vx[span]) / SSE_VX_SCALE_MPS
    yaw_error = (yaw_rate[span] - servo_yaw[span]) / SSE_YAW_SCALE_RPS
    sse_vx = _number(float(np.sum(vx_error ** 2)))
    sse_yaw = _number(float(np.sum(yaw_error ** 2)))
    cost = np.mean((torque[span] / TORQUE_LIMITS_NM[None, :, None]) ** 2,
                   axis=(1, 2))
    return {
        "drive_window": [begin, t0],
        "drive_completed_stop": stop,
        "drive_completed_ticks": observed,
        "drive_complete": stop == t0,
        "sse_total": _number(sse_vx + sse_yaw),
        "sse_vx": sse_vx,
        "sse_yaw": sse_yaw,
        "torque_cost_sum": _number(float(np.sum(cost))),
        "torque_cost_mean": _mean_or_none(cost),
        "torque_cost_by_tick": [_number(value) for value in cost],
        "scales": {"vx_mps": SSE_VX_SCALE_MPS, "yaw_rps": SSE_YAW_SCALE_RPS},
        "torque_limits_nm": [float(value) for value in TORQUE_LIMITS_NM],
        "note": "c is a normalized torque-squared proxy; it is not mechanical "
                "energy or electrical consumption. D excludes the settle phase "
                "and the post-release baseline brake.",
    }


# --------------------------------------------------------------------------- #
# pair aggregation
# --------------------------------------------------------------------------- #
def score_pairs(case_scores: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """Aggregate twelve ``score_case`` results into the six matched pairs.

    All twelve per-case results and all six pairs are retained, failures
    included.  A policy can never claim a benefit by adding a new failure or by
    weakening a safety gate: branch A needs a genuine zero-task-fail ->
    policy-task-pass conversion with no new policy failure, branch B needs the
    pooled SSE improvement on the common task-passing pairs, and both branches
    additionally need the shared prerequisites and the pooled cost gate.
    """
    if not isinstance(case_scores, Sequence) or isinstance(case_scores, (str, bytes)):
        raise TypeError("case_scores must be a sequence of score_case results")
    if len(case_scores) != 12:
        raise ValueError("pair scoring needs exactly the twelve heldout results")
    table: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in case_scores:
        if not isinstance(row, Mapping):
            raise TypeError("each case score must be a mapping")
        if row.get("schema") != SCORE_SCHEMA:
            raise ValueError("case score was not produced by this scorer version")
        key = (str(row["case_id"]), str(row["actor"]))
        if key[0] not in HELDOUT_CASES or key[1] not in ACTORS:
            raise ValueError("case score has an unknown case_id/actor")
        if key in table:
            raise ValueError("duplicate case score: " + repr(key))
        table[key] = row
    missing = [
        f"{case_id}/{actor}" for case_id in HELDOUT_CASES for actor in ACTORS
        if (case_id, actor) not in table
    ]
    if missing:
        raise ValueError("pair scoring lacks results for: " + ", ".join(missing))

    pairs: list[dict[str, Any]] = []
    for case_id in HELDOUT_CASES:
        pairs.append(_pair_row(case_id, table[(case_id, "zero")],
                               table[(case_id, "final_policy")]))

    prerequisite_reasons: list[str] = []
    for (case_id, actor), row in table.items():
        if not row["record"]["record_valid"]:  # type: ignore[index]
            prerequisite_reasons.append(f"{case_id}/{actor}: record integrity failed")
    for pair in pairs:
        if pair["capability_regression"]:
            prerequisite_reasons.append(
                f"{pair['case_id']}: zero task_passed but policy did not (capability regression)"
            )
        if not pair["policy_full_horizon"]:
            prerequisite_reasons.append(
                f"{pair['case_id']}: policy did not complete the 1600-control horizon"
            )
        if not pair["policy_safety_passed"]:
            prerequisite_reasons.append(
                f"{pair['case_id']}: policy failed a safety gate"
            )
    prerequisites_passed = not prerequisite_reasons

    cost = _cost_gate(pairs)
    branch_a = _branch_a(pairs)
    branch_b = _branch_b(pairs)
    contribution = bool(
        prerequisites_passed and cost["passed"] and (branch_a["passed"] or branch_b["passed"])
    )
    policy_qualified = all(pair["policy_task_passed"] for pair in pairs)
    return {
        "schema": PAIR_SCHEMA,
        "scoring_addendum": SCORING_ADDENDUM,
        "case_count": 12,
        "pair_count": len(pairs),
        "pairs": pairs,
        "prerequisites": {
            "passed": prerequisites_passed,
            "reasons": prerequisite_reasons,
            "numeric_scope": (
                "all twelve records internally valid, no zero task_passed pair lost "
                "by the policy, and every policy episode complete with all safety "
                "gates passed"
            ),
            "external_prerequisites_pending_independent_reader": [
                "byte-exact paired initial qpos/qvel/ctrl/qacc_warmstart/observation",
                "the formal actor is the unique strictly loaded final checkpoint",
                "raw source/ELF/dependency hashes and the C/Python budget ledgers",
                "five actual native integrator/contactForce returns per control",
                "controller -> requested -> applied torque chain",
            ],
        },
        "cost_gate": cost,
        "branch_a_task_conversion": branch_a,
        "branch_b_sse_improvement": branch_b,
        "RL_contribution_passed": contribution,
        "RL_contribution_reason": (
            "prerequisites AND cost_gate AND (branch A OR branch B)"
            if contribution else
            "at least one of prerequisites, cost gate, branch A or branch B failed"
        ),
        "all_six_policy_task_qualified": policy_qualified,
        "capability_and_contribution_are_distinct": (
            "all-six policy task qualification (including the formal 1.6 m/s gate) "
            "is a separate claim from local RL contribution; a local SSE "
            "improvement never substitutes for the capability gate, and a passing "
            "policy where the zero baseline also passes credits the system and "
            "controller, not RL alone"
        ),
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "not_established_here": list(NOT_ESTABLISHED_HERE),
    }


def _pair_row(case_id: str, zero: Mapping[str, object],
              policy: Mapping[str, object]) -> dict[str, Any]:
    zero_rl = zero["rl_terms"]  # type: ignore[index]
    policy_rl = policy["rl_terms"]  # type: ignore[index]
    drive_begin, drive_end = zero_rl["drive_window"]  # type: ignore[index]
    common_stop = min(int(zero_rl["drive_completed_stop"]),  # type: ignore[index]
                      int(policy_rl["drive_completed_stop"]))  # type: ignore[index]
    common_ticks = max(0, common_stop - int(drive_begin))
    zero_cost = list(zero_rl["torque_cost_by_tick"])  # type: ignore[index]
    policy_cost = list(policy_rl["torque_cost_by_tick"])  # type: ignore[index]
    zero_common = [_number(value) for value in zero_cost[:common_ticks]]
    policy_common = [_number(value) for value in policy_cost[:common_ticks]]
    zero_task = bool(zero["task_passed"])  # type: ignore[index]
    policy_task = bool(policy["task_passed"])  # type: ignore[index]
    sse_zero = _number(zero_rl["sse_total"])  # type: ignore[index]
    sse_policy = _number(policy_rl["sse_total"])  # type: ignore[index]
    if sse_zero == 0.0:
        ratio: float | None = None
        ratio_reason = (
            "zero SSE is exactly 0; no ratio is defined and no epsilon is added"
        )
        no_worse = sse_policy == 0.0
    else:
        ratio = _number(sse_policy / sse_zero)
        ratio_reason = None
        no_worse = sse_policy <= BRANCH_B_PER_PAIR_SSE_TOLERANCE * sse_zero
    return {
        "case_id": case_id,
        "seed": int(zero["seed"]),  # type: ignore[index]
        "terrain": str(zero["terrain"]),  # type: ignore[index]
        "zero_task_passed": zero_task,
        "policy_task_passed": policy_task,
        "zero_task_reasons": list(zero["task_reasons"]),  # type: ignore[index]
        "policy_task_reasons": list(policy["task_reasons"]),  # type: ignore[index]
        "zero_record_valid": bool(zero["record"]["record_valid"]),  # type: ignore[index]
        "policy_record_valid": bool(policy["record"]["record_valid"]),  # type: ignore[index]
        "zero_early_task_termination": bool(
            zero["record"]["early_task_termination"]  # type: ignore[index]
        ),
        "policy_full_horizon": bool(policy["record"]["full_horizon"]),  # type: ignore[index]
        "policy_safety_passed": bool(policy["safety"]["passed"]),  # type: ignore[index]
        "zero_safety_passed": bool(zero["safety"]["passed"]),  # type: ignore[index]
        "capability_regression": zero_task and not policy_task,
        "task_conversion": (not zero_task) and policy_task,
        "common_task_passing": zero_task and policy_task,
        "drive_window": [int(drive_begin), int(drive_end)],
        "zero_drive_completed_ticks": int(zero_rl["drive_completed_ticks"]),  # type: ignore[index]
        "policy_drive_completed_ticks": int(
            policy_rl["drive_completed_ticks"]  # type: ignore[index]
        ),
        "zero_drive_complete": bool(zero_rl["drive_complete"]),  # type: ignore[index]
        "policy_drive_complete": bool(policy_rl["drive_complete"]),  # type: ignore[index]
        "common_drive_stop": common_stop,
        "common_drive_ticks": common_ticks,
        "sse_zero": sse_zero,
        "sse_policy": sse_policy,
        "sse_zero_vx": _number(zero_rl["sse_vx"]),  # type: ignore[index]
        "sse_zero_yaw": _number(zero_rl["sse_yaw"]),  # type: ignore[index]
        "sse_policy_vx": _number(policy_rl["sse_vx"]),  # type: ignore[index]
        "sse_policy_yaw": _number(policy_rl["sse_yaw"]),  # type: ignore[index]
        "sse_ratio": ratio,
        "sse_ratio_reason": ratio_reason,
        "sse_no_worse_within_1p02": no_worse,
        "common_cost_sum_zero": _number(sum(zero_common)),
        "common_cost_sum_policy": _number(sum(policy_common)),
        "own_completed_drive_cost_sum_zero": _number(
            zero_rl["torque_cost_sum"]  # type: ignore[index]
        ),
        "own_completed_drive_cost_sum_policy": _number(
            policy_rl["torque_cost_sum"]  # type: ignore[index]
        ),
        "policy_only_drive_suffix_ticks": max(0, len(policy_cost) - common_ticks),
        "policy_only_drive_suffix_cost_sum": _number(sum(policy_cost[common_ticks:])),
        "zero_only_drive_suffix_ticks": max(0, len(zero_cost) - common_ticks),
        "zero_only_drive_suffix_cost_sum": _number(sum(zero_cost[common_ticks:])),
        "note": (
            "a valid zero early task failure is a real comparison result, not a bad "
            "record; the cost conclusion covers only the common drive interval and "
            "is not a whole-task energy claim"
        ),
    }


def _cost_gate(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pooled normalized torque-squared cost over the common completed D of all six pairs."""
    reasons: list[str] = []
    empty = [pair["case_id"] for pair in pairs if pair["common_drive_ticks"] == 0]
    total_ticks = sum(int(pair["common_drive_ticks"]) for pair in pairs)
    zero_sum = _number(sum(float(pair["common_cost_sum_zero"]) for pair in pairs))
    policy_sum = _number(sum(float(pair["common_cost_sum_policy"]) for pair in pairs))
    if empty:
        reasons.append(
            "empty common completed drive interval for: " + ", ".join(empty)
            + "; without common cost samples this contract cannot issue an RL "
              "contribution, although a capability conversion is still reported"
        )
    pooled_zero = None if total_ticks == 0 else _number(zero_sum / total_ticks)
    pooled_policy = None if total_ticks == 0 else _number(policy_sum / total_ticks)
    ratio: float | None = None
    ratio_reason: str | None = None
    if pooled_zero is None:
        ratio_reason = "no common completed drive ticks; pooled cost is undefined"
    elif pooled_zero == 0.0:
        ratio_reason = "pooled zero cost is exactly 0; no ratio is defined and no epsilon is added"
        if pooled_policy != 0.0:
            reasons.append("pooled zero cost is 0 while the policy cost is positive")
    else:
        raw_ratio = pooled_policy / pooled_zero
        if math.isfinite(raw_ratio):
            ratio = float(raw_ratio)
        else:
            ratio_reason = (
                "pooled cost ratio overflows finite float; no epsilon is added and "
                "the gate uses the direct pooled-cost comparison"
            )
        if pooled_policy > COST_GATE_FRACTION * pooled_zero:
            reasons.append(
                "pooled normalized torque-squared cost exceeds 1.20 times "
                f"the zero cost (reported ratio {ratio})"
            )
    return {
        "passed": not reasons,
        "reasons": reasons,
        "pooled_common_drive_ticks": total_ticks,
        "pooled_cost_zero": pooled_zero,
        "pooled_cost_policy": pooled_policy,
        "cost_ratio": ratio,
        "cost_ratio_reason": ratio_reason,
        "empty_common_interval_cases": empty,
        "per_pair": [
            {
                "case_id": pair["case_id"],
                "common_drive_ticks": int(pair["common_drive_ticks"]),
                "common_cost_sum_zero": float(pair["common_cost_sum_zero"]),
                "common_cost_sum_policy": float(pair["common_cost_sum_policy"]),
                "policy_only_drive_suffix_ticks": int(pair["policy_only_drive_suffix_ticks"]),
                "policy_only_drive_suffix_cost_sum": float(
                    pair["policy_only_drive_suffix_cost_sum"]
                ),
            }
            for pair in pairs
        ],
        "scope": (
            "the cost comparison is defined only on the same-tick intersection of "
            "the two actually completed drive intervals of each pair"
        ),
    }


def _branch_a(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    conversions = [pair["case_id"] for pair in pairs if pair["task_conversion"]]
    regressions = [pair["case_id"] for pair in pairs if pair["capability_regression"]]
    reasons: list[str] = []
    if not conversions:
        reasons.append("no genuine zero task_fail -> policy task_pass conversion")
    if regressions:
        reasons.append("new policy task failure(s) on: " + ", ".join(regressions))
    return {
        "passed": not reasons,
        "reasons": reasons,
        "conversion_cases": conversions,
        "capability_regression_cases": regressions,
        "requirement": ">=1 conversion, no new policy failure, shared prerequisites "
                       "and the pooled cost gate",
    }


def _branch_b(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    common = [pair for pair in pairs if pair["common_task_passing"]]
    names = [pair["case_id"] for pair in common]
    zero_sum = _number(sum(float(pair["sse_zero"]) for pair in common))
    policy_sum = _number(sum(float(pair["sse_policy"]) for pair in common))
    no_worse = [pair["case_id"] for pair in common if pair["sse_no_worse_within_1p02"]]
    reasons: list[str] = []
    ratio: float | None = None
    ratio_reason: str | None = None
    if len(common) < BRANCH_B_MIN_PAIRS:
        reasons.append(
            f"only {len(common)} common task-passing pairs; branch B needs at least 4"
        )
    if zero_sum == 0.0:
        ratio_reason = (
            "pooled zero SSE is exactly 0; no ratio is defined and no epsilon is added"
        )
        reasons.append(
            "no error change can be shown against a pooled zero SSE of 0"
            if policy_sum == 0.0 else
            "pooled zero SSE is 0 while the pooled policy SSE is positive"
        )
    else:
        ratio = _number(policy_sum / zero_sum)
        if policy_sum > BRANCH_B_POOLED_SSE_FRACTION * zero_sum:
            reasons.append(
                f"pooled policy/zero SSE ratio {ratio} is above the required 0.85"
            )
    if len(no_worse) < BRANCH_B_MIN_NO_WORSE_PAIRS:
        reasons.append(
            f"only {len(no_worse)} pairs satisfy S_policy <= 1.02 * S_zero; four are required"
        )
    return {
        "passed": not reasons,
        "reasons": reasons,
        "common_task_passing_cases": names,
        "common_task_passing_count": len(common),
        "pooled_sse_zero": zero_sum,
        "pooled_sse_policy": policy_sum,
        "pooled_sse_ratio": ratio,
        "pooled_sse_ratio_reason": ratio_reason,
        "pairs_no_worse_within_1p02": no_worse,
        "excluded_cases": [
            pair["case_id"] for pair in pairs if not pair["common_task_passing"]
        ],
        "requirement": "|J| >= 4, pooled policy SSE <= 0.85 x pooled zero SSE, and "
                       ">= 4 pairs with S_policy <= 1.02 x S_zero; excluded pairs "
                       "stay fully published and are never deleted to claim success",
    }
