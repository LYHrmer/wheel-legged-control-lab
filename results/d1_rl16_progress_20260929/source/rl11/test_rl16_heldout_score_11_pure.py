"""Six arithmetic-only boundary cases for the preregistered 11-S scorer.

Synthetic arrays here are not robot evidence. No simulator/model/policy is
imported; actual heldout provenance is owned by the independent raw reader.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "course_impl08"))

import rl16_heldout_score_11 as score


def _case(case_id: str, actor: str, *, completed: int | None = None,
          vx_error: float = 0.0, torque_scale: float = 1.0) -> dict:
    terrain, speed, yaw_amplitude, seed, cap = score.HELDOUT_CASES[case_id]
    window = score.case_windows(case_id)
    n = cap if completed is None else completed
    servo_vx = np.zeros(n)
    servo_yaw = np.zeros(n)
    previous_vx = previous_yaw = 0.0
    for tick in range(n):
        raw_vx = speed if 175 <= tick < window["release_tick_t0"] else 0.0
        previous_vx += max(-.005, min(.005, raw_vx - previous_vx))
        servo_vx[tick] = previous_vx
        raw_yaw = 0.0
        hold_tick = tick - window["hold"][0]
        if yaw_amplitude and 0 <= hold_tick < 400:
            raw_yaw = yaw_amplitude if hold_tick < 100 or hold_tick >= 300 else -yaw_amplitude
        previous_yaw += max(-.006, min(.006, raw_yaw - previous_yaw))
        servo_yaw[tick] = previous_yaw
    actual_vx = servo_vx + vx_error
    projection = np.cumsum(actual_vx) * .01
    spawn_x = -8.0 if terrain == "flat" else 2.75 if terrain == "ramp" else .35
    torque = np.tile(
        (np.array([8.0, 8.0, 8.0, 1.2] * 4) * torque_scale)[None, :, None],
        (n, 1, 5),
    )
    full = n == cap
    row = {
        "case_id": case_id, "actor": actor, "seed": seed, "terrain": terrain,
        "task_schema": score.TASK_SCHEMA,
        "reference_schema": score.REFERENCE_SCHEMA,
        "reward_schema": score.REWARD_SCHEMA,
        "completed_controls": n, "terminated": not full, "truncated": full,
        "stop_reason": "time_limit" if full else "fall_or_low_clearance",
        "warning_count": 0, "geometry_invalid_count": 0,
        "map_escape_count": 0, "fall_count": 0 if full else 1,
        "tick_index": np.arange(n), "native_index": np.arange(5*n),
        "applied_servo_vx_mps": servo_vx,
        "applied_servo_yaw_rps": servo_yaw,
        "com_vx_mps": actual_vx,
        "body_yaw_rate_rps": servo_yaw.copy(),
        "heading_rad": np.cumsum(servo_yaw)*.01,
        "clearance_m": np.full(n, .455),
        "lateral_offset_m": np.zeros(n),
        "forward_projection_m": projection,
        "motor_torque_nm": torque,
        "native_roll_deg": np.zeros(5*n),
        "native_pitch_deg": np.zeros(5*n),
        "native_terrain_relative_tilt_deg": np.zeros(5*n),
        "native_clearance_m": np.full(5*n, .455),
        "native_forward_projection_m": np.repeat(projection, 5),
        "native_nonwheel_ground_candidate_count": np.zeros(5*n, dtype=np.int64),
        "initial_roll_deg": 0.0, "initial_pitch_deg": 0.0,
        "initial_terrain_relative_tilt_deg": 0.0,
        "initial_clearance_m": .455,
        "initial_base_x_m": spawn_x, "initial_base_y_m": 0.0,
        "native_base_x_m": spawn_x + np.repeat(projection, 5),
        "native_base_y_m": np.zeros(5*n),
        "positive_wheel_load_native_counts_by_family": {"floor": 5*n}
        if terrain == "flat" else {"floor": 5*n, score.prior.TERRAIN_FAMILY[terrain]: 5*n},
        "reader_asserted": {"geometry_passed": True},
    }
    if terrain == "ramp":
        row.update({
            "ramp_positive_wheel_load_native_counts_by_geom": {
                name: 1 for name in score.RAMP_GEOMS
            },
            "ramp_geom_world_x_max_m_by_name": {
                name: 6.0 for name in score.RAMP_GEOMS
            },
            "wheel_collision_world_x_min_m_by_index": {
                str(index): 7.0 for index in range(4)
            },
        })
    return row


def _twelve(zero_error: float, policy_error: float,
            *, policy_torque_scale: float = 1.0) -> list[dict]:
    rows = []
    for name in score.HELDOUT_CASES:
        rows.append(score.score_case(_case(name, "zero", vx_error=zero_error)))
        rows.append(score.score_case(_case(name, "final_policy", vx_error=policy_error,
                                           torque_scale=policy_torque_scale)))
    return rows


def test_frozen_1600_and_1800_windows_and_world_identity():
    assert score.case_windows("flat_1p6") == {
        "control_cap": 1600, "hold": [495, 895], "release_tick_t0": 895,
        "drive": [175, 895], "final": [1500, 1600],
    }
    assert score.case_windows("ramp_0p45_complete") == {
        "control_cap": 1800, "hold": [600, 1000], "release_tick_t0": 1355,
        "drive": [175, 1355], "final": [1700, 1800],
    }
    case = _case("ramp_0p45_complete", "final_policy")
    result = score.score_case(case)
    assert result["speed"]["hold_observed_ticks"] == 400
    assert result["final_window"]["window"] == [1700, 1800]
    assert result["rl_terms"]["drive_completed_ticks"] == 1180
    assert result["record"]["full_horizon"] is True
    assert result["task_passed"] is True
    case["reference_schema"] = "d1-course-old-terrain-target-v1"
    try:
        score.score_case(case)
    except ValueError as error:
        assert "identity" in str(error)
    else:
        raise AssertionError("old task reference was accepted")


def test_world_pose_gate_keeps_old_geometry_relative_diagnostic():
    case = _case("ramp_0p45_complete", "zero")
    case["native_terrain_relative_tilt_deg"][:] = 82.0
    case["initial_terrain_relative_tilt_deg"] = 82.0
    result = score.score_case(case)
    assert result["safety"]["max_geometry_relative_tilt_deg_diagnostic_only"] == 82.0
    assert result["safety"]["passed"] is True
    case["native_pitch_deg"][11] = 10.000001
    assert score.score_case(case)["safety"]["passed"] is False
    case["native_pitch_deg"][11] = 0.0
    case["native_base_x_m"][11] = 10.500001
    assert score.score_case(case)["safety"]["passed"] is False


def test_complete_ramp_needs_three_real_loads_and_all_four_shapes():
    good = _case("ramp_0p45_complete", "final_policy")
    assert score.score_case(good)["ramp_geometry"]["passed"] is True
    good["ramp_positive_wheel_load_native_counts_by_geom"]["terrain_ramp_deck"] = 0
    assert score.score_case(good)["ramp_geometry"]["passed"] is False
    good["ramp_positive_wheel_load_native_counts_by_geom"]["terrain_ramp_deck"] = 1
    good["wheel_collision_world_x_min_m_by_index"]["3"] = 6.0
    assert score.score_case(good)["ramp_geometry"]["passed"] is False
    del good["wheel_collision_world_x_min_m_by_index"]["3"]
    try:
        score.score_case(good)
    except ValueError as error:
        assert "four wheel" in str(error)
    else:
        raise AssertionError("missing fourth wheel collision shape was accepted")


def test_valid_partial_is_not_full_or_a_bad_record():
    early = score.score_case(_case("ramp_0p45_complete", "zero", completed=999))
    assert early["record"]["record_valid"] is True
    assert early["record"]["early_task_termination"] is True
    assert early["speed"]["hold_observed_ticks"] == 399
    assert early["speed"]["passed"] is False
    assert early["final_window"]["observed_ticks"] == 0
    assert early["task_passed"] is False
    corrupt = _case("ramp_0p45_complete", "zero", completed=999)
    corrupt["terminated"] = False
    corrupt["truncated"] = False
    assert score.score_case(corrupt)["record"]["record_valid"] is False
    corrupt["terminated"] = True
    corrupt["geometry_invalid_count"] = 1
    assert score.score_case(corrupt)["record"]["record_valid"] is False
    corrupt["geometry_invalid_count"] = 0
    corrupt["warning_count"] = 1
    assert score.score_case(corrupt)["record"]["record_valid"] is False
    corrupt["completed_controls"] = 1801
    try:
        score.score_case(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError("more than the ramp 1800 cap was accepted")


def test_1p6_stop_needs_25_endpoints_and_native_peak():
    case = _case("flat_1p6", "final_policy")
    good = score.score_case(case)
    assert good["stopping"]["found_settled_window"] is True
    assert good["stopping"]["passed"] is True
    release = float(case["forward_projection_m"][894])
    case["native_forward_projection_m"][5*950] = release + 3.5
    assert score.score_case(case)["stopping"]["passed"] is False
    case = _case("flat_1p6", "final_policy")
    case["com_vx_mps"][1228::25] = .2
    stopped = score.score_case(case)["stopping"]
    assert stopped["found_settled_window"] is False
    assert stopped["passed"] is False


def test_paired_sse_cost_boundaries_and_zero_denominators():
    gain = score.score_pairs(_twelve(.02, .01))
    assert gain["branch_b_sse_improvement"]["passed"] is True
    assert gain["cost_gate"]["passed"] is True
    assert gain["RL_contribution_passed"] is True
    costly = score.score_pairs(_twelve(.02, .01, policy_torque_scale=1.1))
    assert costly["cost_gate"]["cost_ratio"] > 1.20
    assert costly["RL_contribution_passed"] is False
    zero = score.score_pairs(_twelve(0.0, 0.0, policy_torque_scale=0.0))
    assert zero["branch_b_sse_improvement"]["pooled_sse_ratio"] is None
    assert zero["branch_b_sse_improvement"]["passed"] is False
    assert zero["RL_contribution_passed"] is False
    inside = score.score_pairs(_twelve(.02, .02*math.sqrt(.84)))
    outside = score.score_pairs(_twelve(.02, .02*math.sqrt(.86)))
    assert inside["branch_b_sse_improvement"]["passed"] is True
    assert outside["branch_b_sse_improvement"]["passed"] is False
