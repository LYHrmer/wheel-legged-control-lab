"""Independent evaluation rejects reward-like false flight evidence."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.d1_jump_ppo_scoring import score_jump_episode


@pytest.fixture
def protocol():
    return json.loads((Path(__file__).resolve().parents[1] / "results/d1_driving_stability_development/jump_readiness_01/provenance/rl_jump_training_plan_01/evaluation_protocol.json").read_text())


def records():
    endpoints = [{"tick": k, "time_s": .01*k, "roll_rad": 0., "pitch_rad": 0.,
                  "heading_error_rad": 0., "planar_displacement_m": 0.,
                  "planar_position_m": [0.,0.], "achieved_clearance_m": .455,
                  "commanded_clearance_m": .455, "body_vx_mps": 0., "com_vz_mps": 0.}
                 for k in range(601)]
    intervals = [{"index": k, "returned": True, "start_time_s": .002*k,
                  "end_time_s": .002*(k+1), "actual_dt_s": .002,
                  "active_wheel_contacts": [1]*4, "wheel_normal_load_n": [100.]*4,
                  "invalid_load": False, "invalid_normal_values": [],
                  "endpoint_min_gap_m": -.001, "contact_margin_m": .001,
                  "start_com_vz_mps": 0.} for k in range(3000)]
    flags = {k: False for k in ("unknown_wrench","nonwheel_contact","fall","domain_exit",
                               "nonfinite_state","rated_torque_exceeded",
                               "position_envelope_exceeded","velocity_envelope_exceeded")}
    flags.update(reset_count=1,plane_normals_valid=True,protections_intact=True)
    return {"completed_control_intervals": 600,"execution":flags,"endpoints":endpoints,"intervals":intervals}


def fly(data,start=1100,count=10,gap=.022,vz=.1):
    for row in data["intervals"][start:start+count]:
        row.update(active_wheel_contacts=[0]*4,wheel_normal_load_n=[0.]*4,
                   endpoint_min_gap_m=gap,start_com_vz_mps=vz)


def test_requested_physical_success_and_stable_hold(protocol):
    data=records();original=deepcopy(data)
    assert score_jump_episode(data,protocol["cases"][0],protocol)["passed"]
    fly(data)
    score=score_jump_episode(data,protocol["cases"][1],protocol)
    assert score["passed"] and score["max_qualified_net_gap_m"] == pytest.approx(.021)
    assert not score_jump_episode(data,protocol["cases"][0],protocol)["passed"]
    assert data["endpoints"] == original["endpoints"]


@pytest.mark.parametrize("change",["nine","downward","late","gap","single_high","loaded_wheel","late_return"])
def test_false_jump_evidence_rejected(protocol,change):
    data=records()
    fly(data,count=9 if change=="nine" else 10,vz=-.1 if change=="downward" else .1,
        start=1600 if change=="late" else 1100,gap=.002 if change in ("gap","single_high") else .022)
    if change=="single_high":
        data["intervals"][1150]["endpoint_min_gap_m"] = .1
    if change=="loaded_wheel":
        data["intervals"][1105]["wheel_normal_load_n"][0]=.00001
    if change=="late_return":
        for row in data["intervals"][1110:2000]:
            row["wheel_normal_load_n"][3]=0.
    assert not score_jump_episode(data,protocol["cases"][1],protocol)["passed"]


def test_short_trace_and_altered_protocol_fail(protocol):
    data=records();data["intervals"].pop()
    assert not score_jump_episode(data,protocol["cases"][0],protocol)["passed"]
    protocol["gates"]["jump_simultaneous_net_clearance_min_m"] = .001
    with pytest.raises(ValueError,match="changed"):
        score_jump_episode(data,protocol["cases"][0],protocol)


def test_hold_cannot_hide_descending_flight_that_crosses_settling_boundary(protocol):
    data = records()
    fly(data,start=990,count=30,vz=.1)
    for row in data["intervals"][1000:1020]:
        row["start_com_vz_mps"] = -.1
    result = score_jump_episode(data,protocol["cases"][0],protocol)
    assert len(result["flight_runs"]) == 1
    assert not result["checks"]["no_flight_after_settling"] and not result["passed"]
