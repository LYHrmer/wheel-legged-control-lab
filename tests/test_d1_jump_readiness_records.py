"""Adversarial stored-record scoring and observer lifecycle tests; no physics."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.d1_jump_readiness_records import NativeStepObserver, score_readiness


def episode():
    endpoints = [{"tick": k, "time_s": k * .01, "roll_rad": 0., "pitch_rad": 0.,
                  "heading_error_rad": 0., "planar_displacement_m": 0.,
                  "planar_position_m": [0., 0.], "achieved_clearance_m": .455,
                  "commanded_clearance_m": .455, "body_vx_mps": 0., "com_vz_mps": 0.,
                  "min_gap_m": -.001, "wheel_gap_m": [-.001] * 4,
                  "contact_margin_m": .001} for k in range(601)]
    intervals = [{"index": k, "returned": True, "start_time_s": k * .002,
                  "end_time_s": (k + 1) * .002, "actual_dt_s": .002,
                  "active_wheel_contacts": [1] * 4, "geometric_wheel_contacts": [1] * 4,
                  "wheel_normal_load_n": [100.] * 4, "invalid_load": False,
                  "invalid_normal_values": [], "endpoint_min_gap_m": -.001,
                  "endpoint_wheel_gap_m": [-.001] * 4, "contact_margin_m": .001,
                  "start_com_vz_mps": 0.} for k in range(3000)]
    for row in intervals[1125:1135]:
        row.update(active_wheel_contacts=[0]*4, geometric_wheel_contacts=[0]*4,
                   wheel_normal_load_n=[0.]*4, endpoint_min_gap_m=.022,
                   endpoint_wheel_gap_m=[.022]*4, start_com_vz_mps=.1)
    flags = {k: False for k in ("unknown_wrench", "nonwheel_contact", "fall", "domain_exit",
                               "nonfinite_state", "rated_torque_exceeded",
                               "position_envelope_exceeded", "velocity_envelope_exceeded")}
    flags.update(reset_count=1, plane_normals_valid=True, protections_intact=True)
    return {"completed_control_intervals": 600, "execution": flags,
            "endpoints": endpoints, "intervals": intervals}


def scored(value):
    return score_readiness({"episodes": {"stationary_height_hold": value,
                                        "stationary_height_profile": value}})


def profile(value):
    return scored(value)["episodes"]["stationary_height_profile"]


def test_native_peak_between_control_samples_passes_and_input_unchanged():
    value = episode()
    original = deepcopy(value)
    result = profile(value)
    assert result["readiness_passed"]
    assert result["screens"]["useful_clearance"]["detail"]["at_native_index"] == 1125
    assert value == original


@pytest.mark.parametrize("field,bad", [
    ("index", 1124), ("returned", False), ("actual_dt_s", .003),
    ("start_time_s", 8.), ("wheel_normal_load_n", [-1., 0., 0., 0.]),
    ("active_wheel_contacts", [True, 0, 0, 0]), ("invalid_load", True),
    ("wheel_normal_load_n", [float("nan")] * 4),
])
def test_malformed_native_record_never_passes(field, bad):
    value = episode()
    value["intervals"][1125][field] = bad
    assert not profile(value)["execution_valid"]


@pytest.mark.parametrize("change", ["nine", "descending", "active", "load", "margin"])
def test_flight_requires_ten_real_unloaded_intervals_and_upward_com(change):
    value = episode()
    row = value["intervals"][1125]
    if change == "nine":
        row["endpoint_min_gap_m"] = .001
    elif change == "descending":
        row["start_com_vz_mps"] = -.1
    elif change == "active":
        row["active_wheel_contacts"][0] = 1
    elif change == "load":
        row["wheel_normal_load_n"][0] = .00001
    else:
        for item in value["intervals"][1125:1135]:
            item["endpoint_min_gap_m"] = .001
    assert not profile(value)["screens"]["physical_lift"]["passed"]


def test_clearance_threshold_strict_and_late_load_requires_each_wheel():
    value = episode()
    for row in value["intervals"][1125:1135]:
        row["endpoint_min_gap_m"] = .020 + .001
    assert not profile(value)["screens"]["useful_clearance"]["passed"]
    for row in value["intervals"][-1000:]:
        row["wheel_normal_load_n"][3] = 0.
    assert not profile(value)["screens"]["late_settled"]["passed"]


def test_missing_duplicated_and_wrong_endpoint_clock_rejected():
    for key, value in [("tick", 399), ("time_s", 4.1)]:
        data = episode()
        data["endpoints"][400][key] = value
        assert not profile(data)["execution_valid"]
    data = episode()
    data["intervals"].pop()
    assert not profile(data)["execution_valid"]
    assert not score_readiness({})["readiness_passed"]


@pytest.mark.parametrize("failure", [None, "native", "reader", "reader_mutation"])
def test_observer_preserves_binding_records_actual_return_and_does_not_retry(failure):
    data = SimpleNamespace(time=0., qpos=np.zeros(2), qvel=np.zeros(2),
                           qacc_warmstart=np.zeros(2), ctrl=np.zeros(2),
                           xfrc_applied=np.zeros((2, 6)), qfrc_applied=np.zeros(2))
    calls = []
    def fake_native(model, observed):
        calls.append(observed)
        if failure == "native":
            raise ValueError("native failure")
        observed.time += .002
        observed.qpos[0] = 1.
    module = SimpleNamespace(mj_step=fake_native)
    model = object()
    def reader(observed):
        if failure == "reader":
            raise ValueError("reader failure")
        if failure == "reader_mutation":
            observed.qacc_warmstart[0] += 1
        return {"seen": observed.time}
    observer = NativeStepObserver(module, model, data, contact_reader=reader)
    try:
        with observer:
            module.mj_step(model, data)
    except (ValueError, RuntimeError):
        assert failure is not None
    assert module.mj_step is fake_native and len(calls) == 1
    assert observer.attempted_calls == 1
    assert observer.returned_calls == (0 if failure == "native" else 1)
    assert observer.entries[0]["returned"] == (failure != "native")
    assert observer.entries[0]["qfrc_applied"].tolist() == [0., 0.]
