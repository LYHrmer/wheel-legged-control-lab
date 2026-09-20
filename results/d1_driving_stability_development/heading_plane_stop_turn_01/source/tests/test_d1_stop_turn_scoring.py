"""Guard the original physical scoring windows against reindexing mistakes."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts import probe_d1_heading_g1 as g1
from scripts.d1_stop_turn_scoring import score_handoff


@pytest.fixture
def evidence():
    path = Path(__file__).resolve().parents[1]/"results/d1_driving_stability_development/heading_g1_01/evaluation_protocol.json"
    protocol = json.loads(path.read_text())
    case = deepcopy(protocol["cases"][0])
    case.update(name="forward_stop_then_left_turn", max_transitions=1400,
                episode_seconds=14., gate_groups=["all_cases"])
    case["command"]["yaw_pulse"] = {"start_tick": 800, "end_tick_exclusive": 850,
                                     "user_yaw_rate_rps": .6}
    rows, positions, reference = [], [np.zeros(3)], .7
    for k in range(1400):
        command = g1.command_at_tick(case["command"], k)
        after = reference+.01*command.yaw_rate_rps
        rows.append({"tick": k, "endpoint_tick": k+1, "terminated": False,
            "truncated": k == 1399, "terminal_reason": "time_limit" if k == 1399 else None,
            "applied_action": [0.]*8, "user_forward_error_mps": 0.,
            "body_forward_mps": command.forward_velocity_mps, "heading_error_rad": 0.,
            "cross_track_after_m": 0., "reward": 0., "servo_reward": 0.,
            "reward_terms": {"heading_goal": 0.}, "actual_roll_pitch_rad": [0., 0.],
            "metrics": {"height_error_m": 0., "undesired_ground_contacts": 0,
                        "roll_error_rad": 0., "pitch_error_rad": 0.},
            "heading_task": {"actual_dt_s": .01, "user_yaw_rate_error_after": 0.,
                             "reference_heading_before": reference, "reference_heading_after": after}})
        positions.append(positions[-1]+np.array([.01*command.forward_velocity_mps, 0., 0.]))
        reference = after
    return case, rows, np.asarray(positions), protocol["cases"], protocol["proposed_gates"]


def test_fixed_views_keep_reference_and_real_truncation(evidence):
    case, rows, positions, originals, gates = evidence
    before = deepcopy(rows)
    result = score_handoff(case, rows, positions, originals, gates)
    assert result["passed"]
    stop, turn = result["stop_segment"], result["turn_segment"]
    assert not stop["actual_last_row_truncated"] and not stop["original_duration_predicate_on_view"]
    assert stop["segment_coverage"] and turn["actual_last_row_truncated"]
    assert turn["metrics"]["planar_origin_displacement_peak_m"] == 0.
    assert turn["metrics"]["final_reference_delta_rad"] == pytest.approx(.3)
    assert rows == before


@pytest.mark.parametrize("endpoint,included", [(499, False), (500, True), (799, True), (800, False)])
def test_stop_speed_exact_endpoint_window(evidence, endpoint, included):
    case, rows, positions, originals, gates = evidence
    rows[endpoint-1]["body_forward_mps"] = .031
    result = score_handoff(case, rows, positions, originals, gates)
    assert result["stop_segment"]["gates"]["checks"]["late_stop_speed"] is (not included)


@pytest.mark.parametrize("endpoint,included", [(1049, False), (1050, True), (1399, True), (1400, False)])
def test_turn_late_exact_endpoint_window(evidence, endpoint, included):
    case, rows, positions, originals, gates = evidence
    rows[endpoint-1]["heading_error_rad"] = .06  # Below global5deg, above late3deg.
    result = score_handoff(case, rows, positions, originals, gates)
    assert result["global"]["gates"]["checks"]["heading_peak"]
    assert result["turn_segment"]["gates"]["checks"]["turn_late_heading"] is (not included)


def test_stop_RMSE_cannot_be_diluted_by_later_hold(evidence):
    case, rows, positions, originals, gates = evidence
    for row in rows[:800]:
        row["user_forward_error_mps"] = .06
    result = score_handoff(case, rows, positions, originals, gates)
    assert result["global"]["metrics"]["velocity_rmse_mps"] < .05
    assert result["stop_segment"]["metrics"]["velocity_rmse_mps"] == pytest.approx(.06)
    assert not result["stop_segment"]["gates"]["checks"]["velocity_rmse"]


@pytest.mark.parametrize("state_tick,included", [(499, False), (500, True), (700, True), (701, False)])
def test_stop_path_uses_both_endpoint_positions(evidence, state_tick, included):
    case, rows, positions, originals, gates = evidence
    positions[state_tick, 1] = .06
    result = score_handoff(case, rows, positions, originals, gates)
    assert result["stop_segment"]["gates"]["checks"]["late_stop_planar_path"] is (not included)


def test_turn_displacement_uses_state600_and_includes_state1400(evidence):
    case, rows, positions, originals, gates = evidence
    assert positions[600, 0] > .1
    positions[1400, 1] = .11
    result = score_handoff(case, rows, positions, originals, gates)
    assert result["turn_segment"]["metrics"]["planar_origin_displacement_peak_m"] == pytest.approx(.11)
    assert not result["turn_segment"]["gates"]["checks"]["turn_planar_displacement"]


def test_incomplete_actual_trajectory_fails_without_padding(evidence):
    case, rows, positions, originals, gates = evidence
    result = score_handoff(case, rows[:-1], positions[:-1], originals, gates)
    assert not result["passed"] and not result["global"]["gates"]["checks"]["complete_duration"]
    assert not result["turn_segment"]["segment_coverage"]
