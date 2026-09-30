"""Pure C20 evaluation arithmetic tests; never constructs a model or simulator."""
import copy
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
W = HERE.parent
for folder in (W / "course_impl08", W / "continuation18", HERE):
    sys.path.insert(0, str(folder))

import numpy as np
import pytest
import score18
import score20


def canonical(case_id="dev_yaw_left", *, actor="A", speed_error=0., yaw_error=0.):
    row = score20.CASE_SPECS[case_id]
    n = row["horizon"]
    vx, yaw = score20._servo_commands(case_id, n)
    actual_vx, actual_yaw = vx.copy(), yaw.copy()
    start, end = row["hold"]
    actual_vx[start:end] += speed_error
    actual_yaw[start:end] += yaw_error
    projection = np.cumsum(vx)*.01
    return {"case_id": case_id, "actor": "final_policy" if actor != "zero" else "zero",
        "experiment_actor": actor, "terrain": row["terrain"], "seed": row["seed"],
        "mirror": row.get("mirror", False), "task_schema": score20.TASK_SCHEMA,
        "reference_schema": score20.REFERENCE_SCHEMA, "reward_schema": score20.REWARD_SCHEMA,
        "completed_controls": n, "tick_index": np.arange(n), "native_index": np.arange(5*n),
        "applied_servo_vx_mps": vx, "applied_servo_yaw_rps": yaw,
        "com_vx_mps": actual_vx, "body_yaw_rate_rps": actual_yaw,
        "heading_rad": np.zeros(n), "clearance_m": np.full(n, .455),
        "lateral_offset_m": np.zeros(n), "forward_projection_m": projection,
        "motor_torque_nm": np.ones((n,16,5)), "native_roll_deg": np.zeros(5*n),
        "native_pitch_deg": np.zeros(5*n), "native_terrain_relative_tilt_deg": np.zeros(5*n),
        "native_clearance_m": np.full(5*n,.455),
        "native_forward_projection_m": np.repeat(projection,5),
        "native_nonwheel_ground_candidate_count": np.zeros(5*n,dtype=int),
        "native_base_x_m": np.zeros(5*n), "native_base_y_m": np.zeros(5*n),
        "initial_roll_deg": 0., "initial_pitch_deg": 0., "initial_terrain_relative_tilt_deg": 0.,
        "initial_clearance_m": .455, "initial_base_x_m": 0., "initial_base_y_m": 0.,
        "positive_wheel_load_native_counts_by_family": {"bump": 100},
        "terminated": False, "truncated": True, "stop_reason": "horizon_cap",
        "warning_count": 0, "geometry_invalid_count": 0, "map_escape_count": 0, "fall_count": 0}


def score(value):
    return score20.score_case(value, expected_seed=value["seed"], mirror=value["mirror"])


def test_fixed_timelines_and_dynamic_windows():
    a = score20.expected_raw_commands("final_yaw_left")
    assert a[199]["forward_velocity_mps"] == 0 and a[200]["forward_velocity_mps"] == 1.1
    assert a[440]["yaw_rate_rps"] == .28 and a[570]["yaw_rate_rps"] == -.28
    value = canonical("final_yaw_left")
    result = score(value)
    assert result["task_passed"]
    assert result["rl_terms"]["drive_window"] == [200,880]
    assert result["rl_terms"]["drive_completed_ticks"] == 680
    assert result["hold_terms"]["drive_completed_ticks"] == 400
    assert result["yaw"]["max_rms_rps"] == .12


def test_fixed_gates_and_command_tamper():
    assert not score(canonical(speed_error=.051))["speed"]["passed"]
    assert not score(canonical(yaw_error=.121))["yaw"]["passed"]
    value = canonical()
    value["applied_servo_yaw_rps"] = np.zeros(value["completed_controls"])
    with pytest.raises(ValueError, match="servo"):
        score(value)
    value = canonical()
    value["seed"] += 1
    with pytest.raises(ValueError):
        score(value)


def test_original_case_arithmetic_unchanged():
    value = canonical("flat_0p6",speed_error=.012)
    new = score(value)
    old = score18.score_case(value,expected_seed=value["seed"])
    for key in ("speed","safety","final_window","stopping","terrain_channel","ramp_geometry"):
        assert new[key] == old[key]
    for key in ("sse_total","sse_vx","sse_yaw","torque_cost_sum","torque_cost_mean","drive_window"):
        assert new["rl_terms"][key] == old["rl_terms"][key]


def test_equal_case_weighting_and_no_epsilon():
    rows = []
    for index, spec in enumerate(score20.SPEC["evaluation"]["development"]):
        for actor in ("zero","old","A","B"):
            count = 10 if index == 0 else 100
            value = float(index+1)
            rows.append({"schema":score20.SCORE_SCHEMA,"case_id":spec["case_id"],
                "experiment_actor":actor,"record":{"record_valid":True,"full_horizon":True},
                "safety":{"passed":True},"task_passed":True,
                "rl_terms":{"drive_complete":True,"drive_completed_ticks":count,
                            "sse_total":value*count,"torque_cost_mean":value/10},
                "hold_terms":{"drive_complete":True,"drive_completed_ticks":400,
                              "sse_total":value*400}})
    result = score20.equal_case_pool(rows, split="development")
    assert result["complete"]
    assert result["actors"]["B"]["equal_case_error_mean"] == 2.5
    assert result["actors"]["B"]["equal_case_torque_cost_mean"] == .25
    assert result["family_B_over_A"]["yaw"]["drive_error_mean"]["ratio"] == 1.
    assert not score20.ratio(0.,0.)["defined"]
    assert score20.ratio(1.,0.)["ratio"] is None
    assert not score20.equal_case_pool(rows[:-1],split="development")["complete"]
    broken = copy.deepcopy(rows)
    broken[0]["record"]["full_horizon"] = False
    assert not score20.equal_case_pool(broken,split="development")["complete"]
    with pytest.raises(ValueError,match="duplicate"):
        score20.equal_case_pool(rows+[rows[0]],split="development")


def test_original_six_pool_uses_raw_sample_weighting():
    rows=[]
    total_cost=total_count=0
    for index,case in enumerate(score20.ORIGINAL_SIX):
        count=10 if index==0 else 100
        cost=(index+1)*count
        total_cost+=cost
        total_count+=count
        for actor in ('zero','old','A','B'):
            rows.append({'case_id':case,'experiment_actor':actor,
                'record':{'record_valid':True,'full_horizon':True},
                'safety':{'passed':True},'task_passed':True,
                'rl_terms':{'drive_complete':True,'drive_completed_ticks':count,
                            'sse_total':float(cost),'torque_cost_sum':float(cost)}})
    result=score20.original_six_pool(rows)
    assert result['complete']
    assert result['actors']['B']['pooled_torque_cost_mean']==total_cost/total_count
    assert result['comparisons']['B/A']['cost']['ratio']==1.
    assert not score20.original_six_pool(rows[:-1])['complete']
