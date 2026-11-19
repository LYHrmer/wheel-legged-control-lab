import copy
import math

import numpy as np
import pytest

from scripts import terrain_preview84_trials as trial


def sample(index, delta=-2e-5, slope=0.04):
    return {
        "step": index,
        "control_s": (index - 1) * trial.DT,
        "time_s": index * trial.DT,
        "preview_valid": True,
        "preview_slope_rad": slope,
        "preview_captured_s": 0.0,
        "applied_delta_pitch_rad": delta,
        "front_contact_x_m": 0.41 if index >= 6 else None,
        "front_track_x_m": 0.81 if index >= 9 else 0.41,
        "pitch_error_rad": 0.04,
        "base_x_m": 0.3,
        "speed_error_mps": 0.03,
        "torque_mean_square_nm2": 4.0,
        "saturated_axes": 0,
        "torque_bound_violations": 0,
        "force_headroom_n": 36.0,
    }


def metrics(rows):
    return trial.case_metrics(rows, "up", 8417, "on", 0.4, 0.0, -90000.0, "fallen")


def test_three_consecutive_signed_ticks_must_precede_wheel_contact():
    rows = [sample(i) for i in range(1, 11)]
    rows[1]["applied_delta_pitch_rad"] = 0
    m = metrics(rows)
    assert m["first_correct_correction_s"] == pytest.approx(0.02)
    assert m["correction_confirmed_s"] == pytest.approx(0.04)
    assert m["signed_correction_before_contact"] and m["pitch_window_steps"] == 4
    assert m["pitch_rmse_deg"] == pytest.approx(math.degrees(0.04))
    rows[4]["applied_delta_pitch_rad"] = 0
    assert not metrics(rows)["signed_correction_before_contact"]


def test_crossing_and_full_horizon_are_different_requirements():
    rows = [sample(i, slope=0.0) for i in range(1, 8)]
    m = metrics(rows)
    assert m["pitch_rmse_deg"] is None
    assert m["first_front_contact_s"] is not None
    assert not m["full_horizon"] and not m["signed_correction_before_contact"]


def test_geometry_blends_signed_slopes_and_stays_flat_before_entry():
    x = np.array([0.0, 0.4, 0.46, 0.52, 0.6, 0.7])
    up = trial.ramp_heights(x, 0.4, 4.0)
    assert np.array_equal(up[:2], [0.0, 0.0])
    np.testing.assert_allclose(trial.ramp_heights(x, 0.4, -4.0), -up)
    assert (up[-1] - up[-2]) / (x[-1] - x[-2]) == pytest.approx(math.tan(math.radians(4)))


def test_gate_demands_all_models_pairs_and_sign_controls():
    cases = [
        {
            "terrain": t,
            "seed": s,
            "arm": a,
            "full_horizon": True,
            "crossed_window_s": None if t == "flat" else 2.0,
            "speed_rmse_mps": 0.1,
            "torque_rms_nm": 1.0,
            "torque_bound_violations": 0,
            "pitch_rmse_deg": {"off": 1.0, "on": 0.8, "inverted": 1.2}[a],
            "signed_correction_before_contact": t != "flat",
        }
        for t in trial.TERRAINS
        for s in trial.SEEDS
        for a in trial.ARMS
    ]
    assert trial.evaluate(cases)["passed"]
    with pytest.raises(ValueError):
        trial.evaluate(cases[:-1])
    changed = copy.deepcopy(cases)
    next(c for c in changed if c["terrain"] == "up" and c["arm"] == "on")["pitch_rmse_deg"] = 1.1
    assert not trial.evaluate(changed)["passed"]
    changed = copy.deepcopy(cases)
    for c in changed:
        if c["terrain"] == "down" and c["arm"] == "inverted":
            c["pitch_rmse_deg"] = 0.7
    assert not trial.evaluate(changed)["passed"]
