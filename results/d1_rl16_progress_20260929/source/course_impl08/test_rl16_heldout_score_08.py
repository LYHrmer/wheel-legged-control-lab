"""Pure boundary tests for ``rl16_heldout_score_08``.

Stdlib + NumPy + the pure scorer only: no engine, model, worker, simulator mock
or physics claim.  The synthetic arrays below exercise arithmetic, window
indexing and rejection paths; they are not evidence about the real robot, and a
passing test here never substitutes for root's independent raw record / geometry
/ budget readback.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rl16_heldout_score_08 as score


# --------------------------------------------------------------------------- #
# synthetic canonical-record builders (arithmetic only)
# --------------------------------------------------------------------------- #
def _servo_profile(case_id: str):
    """Reproduce the preregistered applied servo request of one case."""
    terrain, speed, amplitude, seed = score.HELDOUT_CASES[case_id]
    ramp = score.ramp_ticks(speed)
    hold_begin = 175 + ramp
    t0 = hold_begin + 400
    vx = np.zeros(1600, dtype=np.float64)
    yaw = np.zeros(1600, dtype=np.float64)
    for tick in range(1600):
        raw_vx = speed if 175 <= tick < t0 else 0.0
        previous = vx[tick - 1] if tick else 0.0
        vx[tick] = previous + max(-0.005, min(0.005, raw_vx - previous))
        raw_yaw = 0.0
        hold_tick = tick - hold_begin
        if amplitude != 0.0 and 0 <= hold_tick < 400:
            raw_yaw = amplitude if hold_tick < 100 or hold_tick >= 300 else -amplitude
        previous_yaw = yaw[tick - 1] if tick else 0.0
        yaw[tick] = previous_yaw + max(-0.006, min(0.006, raw_yaw - previous_yaw))
    return terrain, speed, amplitude, seed, ramp, t0, vx, yaw


def _canonical(
    case_id: str, actor: str, *, completed: int = 1600, vx_offset: float = 0.0,
    torque_scale: float = 1.0, abs_yaw_rate: bool = False,
):
    """A canonical record whose default tracks the applied servo exactly."""
    terrain, _speed, _amplitude, seed, _ramp, _t0, servo_vx, servo_yaw = _servo_profile(case_id)
    n = completed
    servo_vx, servo_yaw = servo_vx[:n], servo_yaw[:n]
    com_vx = servo_vx + vx_offset
    yaw_rate = np.abs(servo_yaw) if abs_yaw_rate else servo_yaw.copy()
    projection = np.cumsum(com_vx) * score.CONTROL_DT_S
    torque = np.tile(
        (np.array([8.0, 8.0, 8.0, 1.2] * 4, dtype=np.float64) * torque_scale)[:, None],
        (1, 5),
    )
    full = n == 1600
    return {
        "case_id": case_id,
        "actor": actor,
        "seed": seed,
        "terrain": terrain,
        "completed_controls": n,
        "terminated": not full,
        "truncated": full,
        "stop_reason": "horizon_cap" if full else "task_failure_clearance",
        "warning_count": 0,
        "geometry_invalid_count": 0,
        "map_escape_count": 0,
        "fall_count": 0,
        "tick_index": np.arange(n),
        "applied_servo_vx_mps": servo_vx,
        "applied_servo_yaw_rps": servo_yaw,
        "com_vx_mps": com_vx,
        "body_yaw_rate_rps": yaw_rate,
        "heading_rad": np.cumsum(yaw_rate) * score.CONTROL_DT_S,
        "clearance_m": np.full(n, 0.455),
        "lateral_offset_m": np.zeros(n),
        "forward_projection_m": projection,
        "motor_torque_nm": np.tile(torque, (n, 1, 1)),
        "native_index": np.arange(5 * n),
        "native_roll_deg": np.zeros(5 * n),
        "native_pitch_deg": np.zeros(5 * n),
        "native_terrain_relative_tilt_deg": np.zeros(5 * n),
        "native_clearance_m": np.full(5 * n, 0.455),
        "native_forward_projection_m": np.repeat(projection, 5),
        "native_nonwheel_ground_candidate_count": np.zeros(5 * n, dtype=np.int64),
        "initial_roll_deg": 0.0,
        "initial_pitch_deg": 0.0,
        "initial_terrain_relative_tilt_deg": 0.0,
        "initial_clearance_m": 0.455,
        "positive_wheel_load_native_counts_by_family": (
            {"floor": 5 * n} if terrain == "flat"
            else {"floor": 5 * n, score.TERRAIN_FAMILY[terrain]: 5 * n}
        ),
        "reader_asserted": {"geometry_passed": True, "initial_pair_exact": True},
    }


def _twelve(zero_offset, policy_offset, *, policy_torque_scale=1.0,
            policy_completed=None, zero_completed=None, torque_scale=1.0):
    """Twelve scored cases; the offsets may be a float or a per-case mapping."""
    rows = []
    for case_id in score.HELDOUT_CASES:
        for actor, offset in (("zero", zero_offset), ("final_policy", policy_offset)):
            value = offset[case_id] if isinstance(offset, dict) else offset
            completed = zero_completed if actor == "zero" else policy_completed
            rows.append(score.score_case(_canonical(
                case_id, actor,
                completed=1600 if completed is None else completed,
                vx_offset=value,
                torque_scale=torque_scale if actor == "zero" else policy_torque_scale,
            )))
    return rows


def _json_pure(value) -> bool:
    """No NumPy scalar/array, no NaN and no Inf may leave the scorer."""
    if isinstance(value, (np.generic, np.ndarray)):
        return False
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_pure(item)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_json_pure(item) for item in value)
    return False


# --------------------------------------------------------------------------- #
# 1. the complete 400-tick high-speed hold
# --------------------------------------------------------------------------- #
def test_flat_1p6_complete_hold_and_stopping_pass():
    result = score.score_case(_canonical("flat_1p6", "final_policy"))
    assert result["windows"]["hold"] == [495, 895]
    assert result["windows"]["release_tick_t0"] == 895
    assert result["windows"]["drive"] == [175, 895]
    assert result["speed"]["hold_observed_ticks"] == 400
    assert result["speed"]["hold_complete"] is True
    assert abs(result["speed"]["mean_com_vx_mps"] - 1.6) < 1e-12
    assert result["speed"]["fraction_above_1p5_mps_over_400"] == 1.0
    assert result["final_window"]["observed_ticks"] == 100
    assert abs(result["final_window"]["clearance_population_std_m"]) <= 1e-12
    assert result["stopping"]["found_settled_window"] is True
    assert result["stopping"]["time_to_settle_s"] <= 4.2
    assert result["stopping"]["max_native_forward_excursion_m"] <= 3.2
    assert result["terrain_channel"]["applicable"] is False
    assert result["record"]["record_valid"] is True
    assert result["task_passed"] is True
    assert result["task_reasons"] == []
    assert result["rl_terms"]["drive_completed_ticks"] == 720
    assert result["rl_terms"]["sse_total"] == 0.0
    assert _json_pure(result)


# --------------------------------------------------------------------------- #
# 2. 399 observed hold ticks: prefix scored, nothing padded
# --------------------------------------------------------------------------- #
def test_early_termination_scores_prefix_without_padding():
    case = _canonical("flat_1p6", "final_policy", completed=894)
    result = score.score_case(case)
    assert result["record"]["record_valid"] is True
    assert result["record"]["early_task_termination"] is True
    assert result["record"]["full_horizon"] is False
    assert result["speed"]["hold_observed_ticks"] == 399
    assert result["speed"]["hold_complete"] is False
    assert abs(result["speed"]["mean_com_vx_mps"] - 1.6) < 1e-12
    assert result["speed"]["passed"] is False
    assert any("399 of 400" in reason for reason in result["speed"]["reasons"])
    assert result["final_window"]["observed_ticks"] == 0
    assert result["final_window"]["mean_abs_com_vx_mps"] is None
    assert result["final_window"]["passed"] is False
    assert result["stopping"]["found_settled_window"] is False
    assert result["stopping"]["max_native_forward_excursion_m"] is None
    assert result["task_passed"] is False
    assert result["rl_terms"]["drive_completed_ticks"] == 719
    assert result["rl_terms"]["drive_complete"] is False
    assert _json_pure(result)


# --------------------------------------------------------------------------- #
# 3. signed yaw hold and the final yaw stop gate
# --------------------------------------------------------------------------- #
def test_signed_yaw_integrals_and_final_yaw_gate():
    good = score.score_case(_canonical("flat_1p2_yaw", "final_policy"))
    yaw = good["yaw"]
    assert yaw["applicable"] is True
    assert yaw["hold_observed_ticks"] == 400
    assert yaw["rms_yaw_rate_error_vs_applied_servo_rps"] == 0.0
    assert yaw["positive_servo_integrated_yaw_rad"] > 0.0
    assert yaw["negative_servo_integrated_yaw_rad"] < 0.0
    assert yaw["final_mean_abs_yaw_rate_rps"] == 0.0
    assert good["speed"]["passed"] is True
    assert good["task_passed"] is True

    unsigned = score.score_case(
        _canonical("flat_1p2_yaw", "final_policy", abs_yaw_rate=True)
    )
    assert unsigned["yaw"]["negative_servo_integrated_yaw_rad"] > 0.0
    assert unsigned["yaw"]["passed"] is False
    assert any("must be < 0" in reason for reason in unsigned["yaw"]["reasons"])
    assert unsigned["task_passed"] is False


# --------------------------------------------------------------------------- #
# 4. terrain lateral / heading / projection / positive wheel load channel
# --------------------------------------------------------------------------- #
def test_terrain_channel_offset_projection_and_load():
    good = score.score_case(_canonical("ramp_0p35", "final_policy"))
    channel = good["terrain_channel"]
    assert channel["applicable"] is True and channel["passed"] is True
    assert channel["family"] == "ramp"
    assert channel["positive_wheel_load_native_returns"] > 0
    assert channel["net_forward_projection_m"] >= channel["required_forward_projection_m"]
    assert good["task_passed"] is True

    drifted = _canonical("ramp_0p35", "final_policy")
    drifted["lateral_offset_m"] = np.full(1600, -0.3)
    result = score.score_case(drifted)
    assert result["terrain_channel"]["passed"] is False
    assert any("lateral offset" in reason for reason in result["terrain_channel"]["reasons"])

    unloaded = _canonical("ramp_0p35", "final_policy")
    unloaded["positive_wheel_load_native_counts_by_family"] = {"floor": 8000}
    result = score.score_case(unloaded)
    assert result["terrain_channel"]["positive_wheel_load_native_returns"] == 0
    assert any("positive wheel load" in reason
               for reason in result["terrain_channel"]["reasons"])

    short = _canonical("ramp_0p35", "final_policy")
    short["forward_projection_m"] = short["forward_projection_m"] * 0.5
    result = score.score_case(short)
    assert any("net forward projection" in reason
               for reason in result["terrain_channel"]["reasons"])
    assert result["task_passed"] is False

    # Synthetic post-settle drift remains part of the whole-course channel.
    late = _canonical("ramp_0p35", "final_policy")
    late["lateral_offset_m"][1300] = 0.3
    late["forward_projection_m"][-1] = 0.0
    result = score.score_case(late)
    assert result["stopping"]["found_settled_window"] is True
    assert result["terrain_channel"]["window"] == [0, 1600]
    assert result["terrain_channel"]["max_abs_lateral_offset_m"] == 0.3
    assert result["terrain_channel"]["net_forward_projection_m"] == 0.0
    assert result["terrain_channel"]["passed"] is False


# --------------------------------------------------------------------------- #
# 5. 25 consecutive settled endpoints and the native maximum excursion
# --------------------------------------------------------------------------- #
def test_stop_requires_25_consecutive_and_uses_native_maximum():
    interrupted = _canonical("flat_1p6", "final_policy")
    com = interrupted["com_vx_mps"].copy()
    com[1228::25] = 0.2  # breaks every candidate 25-tick run after t0
    interrupted["com_vx_mps"] = com
    result = score.score_case(interrupted)
    assert result["stopping"]["found_settled_window"] is False
    assert result["stopping"]["passed"] is False
    assert any("25 consecutive" in reason for reason in result["stopping"]["reasons"])
    assert result["final_window"]["passed"] is True  # the stop gate alone failed
    assert result["task_passed"] is False

    overshoot = _canonical("flat_1p6", "final_policy")
    release = float(overshoot["forward_projection_m"][894])
    native = overshoot["native_forward_projection_m"].copy()
    native[5 * 950] = release + 3.5  # forward spike later rolled back
    overshoot["native_forward_projection_m"] = native
    result = score.score_case(overshoot)
    assert result["stopping"]["endpoint_projection_m"] < 3.2
    assert abs(result["stopping"]["max_native_forward_excursion_m"] - 3.5) < 1e-9
    assert any("3.2 m" in reason for reason in result["stopping"]["reasons"])
    assert result["task_passed"] is False


# --------------------------------------------------------------------------- #
# 6. branch B: pooled 15% improvement boundary and a zero denominator
# --------------------------------------------------------------------------- #
def test_branch_b_pooled_sse_boundary_and_zero_denominator():
    gain = score.score_pairs(_twelve(0.02, 0.01))
    assert gain["pair_count"] == 6
    assert gain["prerequisites"]["passed"] is True
    branch_b = gain["branch_b_sse_improvement"]
    assert branch_b["common_task_passing_count"] == 6
    assert abs(branch_b["pooled_sse_ratio"] - 0.25) < 1e-9
    assert branch_b["passed"] is True
    assert gain["cost_gate"]["passed"] is True
    assert abs(gain["cost_gate"]["cost_ratio"] - 1.0) < 1e-12
    assert gain["RL_contribution_passed"] is True
    assert gain["all_six_policy_task_qualified"] is True
    assert _json_pure(gain)

    inside = score.score_pairs(_twelve(0.02, 0.02 * math.sqrt(0.84)))
    assert inside["branch_b_sse_improvement"]["pooled_sse_ratio"] < 0.85
    assert inside["branch_b_sse_improvement"]["passed"] is True

    outside = score.score_pairs(_twelve(0.02, 0.02 * math.sqrt(0.86)))
    assert outside["branch_b_sse_improvement"]["pooled_sse_ratio"] > 0.85
    assert outside["branch_b_sse_improvement"]["passed"] is False
    assert outside["branch_a_task_conversion"]["passed"] is False
    assert outside["RL_contribution_passed"] is False

    exact = score.score_pairs(_twelve(0.0, 0.0, torque_scale=0.0,
                                      policy_torque_scale=0.0))
    pooled = exact["branch_b_sse_improvement"]
    assert pooled["pooled_sse_zero"] == 0.0 and pooled["pooled_sse_policy"] == 0.0
    assert pooled["pooled_sse_ratio"] is None
    assert "no epsilon" in pooled["pooled_sse_ratio_reason"]
    assert pooled["passed"] is False
    assert exact["pairs"][0]["sse_ratio"] is None
    assert exact["pairs"][0]["sse_no_worse_within_1p02"] is True
    assert exact["cost_gate"]["pooled_cost_zero"] == 0.0
    assert exact["cost_gate"]["cost_ratio"] is None
    assert exact["cost_gate"]["passed"] is True
    assert exact["RL_contribution_passed"] is False
    assert _json_pure(exact)


# --------------------------------------------------------------------------- #
# 7. per-pair 2% rule, 20% cost gate and a genuine task conversion
# --------------------------------------------------------------------------- #
def test_per_pair_two_percent_cost_gate_and_task_conversion():
    names = list(score.HELDOUT_CASES)
    worse = {name: 0.02 * (math.sqrt(1.05) if index < 3 else math.sqrt(0.5))
             for index, name in enumerate(names)}
    mixed = score.score_pairs(_twelve(0.02, worse))
    branch_b = mixed["branch_b_sse_improvement"]
    assert branch_b["common_task_passing_count"] == 6
    assert branch_b["pooled_sse_ratio"] < 0.85          # pooled improvement holds
    assert len(branch_b["pairs_no_worse_within_1p02"]) == 3
    assert branch_b["passed"] is False                  # the 2% per-pair rule fails
    assert mixed["RL_contribution_passed"] is False

    costly = score.score_pairs(_twelve(0.02, 0.01, policy_torque_scale=1.1))
    assert costly["branch_b_sse_improvement"]["passed"] is True
    assert abs(costly["cost_gate"]["cost_ratio"] - 1.21) < 1e-9
    assert costly["cost_gate"]["passed"] is False
    assert costly["RL_contribution_passed"] is False

    boundary = score.score_pairs(_twelve(
        0.02, 0.01, policy_torque_scale=math.sqrt(1.20),
    ))["cost_gate"]
    assert boundary["passed"] is (
        boundary["pooled_cost_policy"]
        <= score.COST_GATE_FRACTION * boundary["pooled_cost_zero"]
    )

    # Opposite native torque signs cannot cancel before torque-squared cost.
    opposing = _canonical("flat_0p6", "zero")
    opposing["motor_torque_nm"][:, :, 1::2] *= -1.0
    opposed = score.score_case(opposing)
    assert abs(opposed["rl_terms"]["torque_cost_mean"] - 0.01) < 1e-12

    # One genuine zero early task failure converted by the policy, no new failure.
    rows = []
    for case_id in score.HELDOUT_CASES:
        early = case_id == "flat_1p6"
        rows.append(score.score_case(_canonical(
            case_id, "zero", completed=700 if early else 1600, vx_offset=0.02,
        )))
        rows.append(score.score_case(_canonical(case_id, "final_policy", vx_offset=0.02)))
    converted = score.score_pairs(rows)
    pair = next(row for row in converted["pairs"] if row["case_id"] == "flat_1p6")
    assert pair["zero_task_passed"] is False and pair["policy_task_passed"] is True
    assert pair["zero_record_valid"] is True          # a real failure, not corruption
    assert pair["zero_early_task_termination"] is True
    assert pair["task_conversion"] is True and pair["capability_regression"] is False
    assert pair["common_drive_ticks"] == 525          # overlap only
    assert pair["policy_only_drive_suffix_ticks"] == 195
    assert pair["policy_only_drive_suffix_cost_sum"] > 0.0
    assert converted["branch_a_task_conversion"]["conversion_cases"] == ["flat_1p6"]
    assert converted["branch_a_task_conversion"]["passed"] is True
    assert converted["branch_b_sse_improvement"]["common_task_passing_count"] == 5
    assert converted["cost_gate"]["passed"] is True
    assert converted["RL_contribution_passed"] is True
    assert converted["all_six_policy_task_qualified"] is True
    assert "only the common drive interval" in pair["note"]


# --------------------------------------------------------------------------- #
# 8. malformed, missing, nonfinite and forged canonical input
# --------------------------------------------------------------------------- #
def test_malformed_canonical_input_is_rejected():
    def expect(error, mutate):
        case = _canonical("flat_0p6", "zero")
        mutate(case)
        try:
            score.score_case(case)
        except error:
            return
        except BaseException as other:
            raise AssertionError(f"expected {error.__name__}, got {other!r}") from other
        raise AssertionError(f"expected {error.__name__}, nothing raised")

    expect(KeyError, lambda case: case.pop("com_vx_mps"))
    expect(KeyError, lambda case: case.pop("native_nonwheel_ground_candidate_count"))
    expect(ValueError, lambda case: case.update(
        com_vx_mps=np.full(1600, np.nan)))
    expect(ValueError, lambda case: case.update(
        native_clearance_m=np.full(8000, np.inf)))
    expect(ValueError, lambda case: case.update(
        tick_index=np.concatenate([np.arange(1599), [1598]])))
    expect(ValueError, lambda case: case.update(native_index=np.arange(7999)))
    expect(ValueError, lambda case: case.update(seed=88502))
    expect(ValueError, lambda case: case.update(actor="tuned_policy"))
    expect(ValueError, lambda case: case.update(case_id="flat_9p9"))
    expect(TypeError, lambda case: case.update(terminated=1))
    contradictory = _canonical("flat_0p6", "zero")
    contradictory["terminated"] = True
    assert score.score_case(contradictory)["record"]["record_valid"] is False
    expect(TypeError, lambda case: case.update(
        motor_torque_nm=np.full((1600, 16, 5), None, dtype=object)))
    # A forged full-length claim: 1600 declared, 900 samples actually present.
    expect(ValueError, lambda case: case.update(
        com_vx_mps=np.zeros(900), tick_index=np.arange(1600)))
    # Six-pair aggregation refuses an incomplete or duplicated pair table.
    rows = _twelve(0.0, 0.0)
    try:
        score.score_pairs(rows[:11])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for eleven case scores")
    try:
        score.score_pairs(rows[:11] + [rows[0]])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a duplicated case score")
