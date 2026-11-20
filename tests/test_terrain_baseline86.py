import json
import math
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from scripts import terrain_baseline86 as baseline
from wheel_legged_control.d1 import hierarchical
from wheel_legged_control.d1.controllers import D1Command


def nominal_constants():
    return SimpleNamespace(control_dt=.01, nominal_total_mass_kg=45., base_body_id=0,
                           wheel_radius_m=.087, model=SimpleNamespace(
                               body_ipos=np.zeros((1, 3)), opt=SimpleNamespace(gravity=[0, 0, -9.81])))


def test_published_model_is_required_and_both_control_costs_use_the_same_readonly_calibration(
    tmp_path, monkeypatch,
):
    def unexpected_identification(*args, **kwargs):
        pytest.fail("control setup attempted physical identification")

    monkeypatch.setattr(hierarchical, "identify_sagittal_model", unexpected_identification)
    path = tmp_path / "model.npz"
    with ZipFile(baseline.ARCHIVED_MODEL) as saved:
        path.write_bytes(saved.read(baseline.MODEL_MEMBER))
    model = baseline.load_model()
    np.testing.assert_array_equal(model.a, baseline.load_model(path).a)
    assert not model.a.flags.writeable
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        baseline.load_model(path)

    plant = nominal_constants()
    old, new = [baseline.controller_for(plant, model, normalized_input_cost=arm)
                for arm in (False, True)]
    assert old.linear_model is new.linear_model is model
    np.testing.assert_array_equal(old.q, new.q)
    assert old.r.item() == 1e-8 and new.r.item() == 1 / 180**2
    assert old.gain[0, 1] == pytest.approx(-93618.1946, abs=.01)
    assert new.gain[0, 1] == pytest.approx(-187.02, abs=.01)
    assert type(old.low_level) is type(new.low_level) is baseline.SupportTransferVMC
    assert old.low_level.wheel_velocity_gain == new.low_level.wheel_velocity_gain == 0
    assert old.longitudinal_force_limit_n == new.longitudinal_force_limit_n == 180
    with pytest.raises(ValueError, match="archived"):
        baseline.controller_for(plant, None, normalized_input_cost=True)


def test_force_tap_distinguishes_raw_force_from_limited_output_without_changing_torque():
    controller = baseline.controller_for(nominal_constants(), baseline.load_model(),
                                         normalized_input_cost=True)
    controller.low_level.compute = lambda _cmd, _state, *, longitudinal_force_n, **kw: longitudinal_force_n
    for raw, expected in ((843., 180.), (-10.5, -10.5), (-370., -180.)):
        applied = controller._compose_torque(None, D1Command(), raw, None, 0.)
        assert applied == expected == controller.last_longitudinal_force_n
        assert controller.last_unclipped_baseline_force_n == raw


def observed():
    return {"sequence": 0, "captured_s": 0., "available_s": .04, "query_x_m": .4,
            "reason": "valid", "slope_rad": .02, "corroborated_slope_rad": .02,
            "online_pair_start_beam_by_track": [0, 0], "true_floor_chord_rad_by_track": [.02, .02]}


def test_original_gates_keep_falls_incomplete_windows_and_false_sensor_confirmations_separate():
    rows = []
    for k in range(1, 601):
        rows.append({"control_s": (k - 1) / 100, "time_s": k / 100, "step": k,
                     "mounts": {"82": observed()}, "front_contact_x_m": 1.0 if k >= 8 else .8,
                     "front_track_x_m": 1.4 if k >= 12 else .9, "pitch_error_rad": .02,
                     "torque_bound_violations": 0, "termination_reason": None,
                     "applied_longitudinal_force_n": (180. if k < 4 else -12.),
                     "speed_error_mps": .2, "torque_mean_square_nm2": 25.})
    m, evidence = baseline.score_rows(rows, "up", 1, 1.0, normalized_input_cost=True)
    assert m["passed"] and m["true_chord_passed"] and m["first_sensor_window_corroborated"]
    assert m["first_confirmation_s"] == .06 and evidence["first_registered_window"]["sequence"] == 0
    assert m["crossed_window_s"] == .12 and m["zero_window_pitch_rmse_deg"] == pytest.approx(math.degrees(.02))
    assert m["speed_rmse_mps"] == pytest.approx(.2) and m["torque_rms_nm"] == 5.
    assert m["fx_limited_ticks"] == 3 and m["max_progress_after_toe_m"] == pytest.approx(.4)

    tail = rows[:10]
    tail[-1]["termination_reason"] = "fallen"
    for row in tail:
        row["mounts"]["82"]["corroborated_slope_rad"] = None
    m, _ = baseline.score_rows(tail, "up", 1, 1.0, normalized_input_cost=True)
    assert m["sensor_passed"] and not m["true_chord_passed"] and not m["first_sensor_window_corroborated"]
    assert not m["passed"] and not m["full_horizon"] and not m["zero_window_complete"]
    assert m["termination_reason"] == "fallen" and m["zero_window_pitch_rmse_deg"] is None


def test_one_tick_mock_case_logs_both_timestamps_and_partial_directory_is_exclusive(tmp_path, monkeypatch):
    model = baseline.load_model()
    controller = baseline.controller_for(nominal_constants(), model, normalized_input_cost=True)
    controller.last_unclipped_baseline_force_n = 181.
    controller.last_longitudinal_force_n = 180.
    controller.compute = lambda *args: np.zeros(16)
    plant = SimpleNamespace(
        actuator_torque_limit_nm=np.ones(16) * 12., actuator_ids=list(range(16)),
        data=SimpleNamespace(ctrl=np.zeros(16)), base_rpy=np.array([.1, .2, .3]),
        base_velocity=lambda **kw: (np.array([.3, 0., 0.]), np.zeros(3)), step=lambda _: None,
    )
    state = SimpleNamespace(control_time_s=0., measurement_time_s=0., base_rpy=np.zeros(3),
                            base_position=np.zeros(3), base_linear_velocity_body=np.zeros(3),
                            base_angular_velocity_body=np.zeros(3))
    ep = SimpleNamespace(plant=plant, controller=controller, state=state, source=None, toe=1.)
    monkeypatch.setattr(baseline, "new_episode", lambda *args, **kwargs: ep)
    monkeypatch.setattr(baseline, "range_clock", lambda: SimpleNamespace(observe=lambda *_: {"82": observed()}))
    monkeypatch.setattr(baseline, "baseline_command", lambda _: D1Command())
    monkeypatch.setattr(baseline, "physical_sample", lambda *args: {
        "step": 1, "time_s": .01, "termination_reason": "fallen", "front_contact_x_m": None,
        "front_track_x_m": 0., "pitch_error_rad": .1, "torque_bound_violations": 0,
    })
    directory = tmp_path / "case"
    m = baseline.run_zero_case(directory, "up", 1, model, normalized_input_cost=True)
    assert m == json.loads((directory / "metrics.json").read_text()) and not m["passed"]
    row = json.loads((directory / "ticks.jsonl").read_text())
    assert row["baseline_force_n"] == 181. and row["applied_longitudinal_force_n"] == 180.
    assert row["estimated_pitch_rad"] == 0. and row["pre_truth_pitch_rad"] == .2
    assert row["estimator_time_s"] == row["control_s"] == 0. and row["time_s"] == .01
    assert row["requested_torque_nm"] == row["applied_torque_nm"] == [0.] * 16
    (directory / "metrics.json").unlink()
    with pytest.raises(FileExistsError):
        baseline.run_zero_case(directory, "up", 1, model, normalized_input_cost=True)
    assert not (directory / "metrics.json").exists()
