from types import SimpleNamespace

import numpy as np
import pytest

from scripts import terrain_preview85_action as action


def test_axle_force_transfer_shares_the_same_headroom_across_all_legs():
    nominal = np.array([19.0, 12.0, 8.0, 7.0])
    plus, positive = action.transferred_forces(nominal, 10, 20)
    minus, negative = action.transferred_forces(nominal, -10, 20)
    np.testing.assert_array_equal(plus, [20, 13, 7, 6])
    np.testing.assert_array_equal(minus, [14, 7, 13, 12])
    assert (positive, negative) == (2.0, -10.0)
    assert sum(nominal) == sum(plus) == sum(minus)
    assert action.transferred_forces([20, 12, 8, 7], 10, 20)[1] == 0


def test_force_to_torque_uses_individual_foot_jacobians_and_never_drives_wheels(monkeypatch):
    low = object.__new__(action.SupportTransferVMC)
    low.total_mass_kg = 20.0
    nominal, torque = np.full(4, 25.0), np.arange(16.0)
    monkeypatch.setattr(action.D1VMCController, "_support_torque",
                        lambda *args: (torque, nominal))
    jacobian = np.zeros((4, 3, 4))
    jacobian[:, 2, :] = np.arange(1, 17).reshape(4, 4)
    state = SimpleNamespace(foot_jacobian=jacobian)
    low.requested_transfer_n = 0.0
    baseline, _ = low._support_torque(state, None, 0)
    assert baseline is torque
    low.requested_transfer_n = 2.0
    target, _ = low._support_torque(state, None, 0)
    np.testing.assert_array_equal(target - baseline, [-1, -2, -3, 0, -5, -6, -7, 0,
                                                      9, 10, 11, 0, 13, 14, 15, 0])


def test_loaded_front_precontact_trigger_and_pulse_length():
    assert action._trigger(None, 0.4) == "approach"
    assert action._trigger(0.34, 0.4) == "approach"
    assert action._trigger(0.36, 0.4) == "checkpoint"
    assert action._trigger(0.4, 0.4) == "missed_precontact_band"
    for sign in (-1, 0, 1):
        assert [action.pulse_at(sign, i, 100) for i in range(1, 65)] == [2 * sign] * 16 + [0] * 48


def _branches():
    result = {}
    for name, sign in action.ARMS.items():
        rows = []
        for step in range(1, action.BRANCH_TICKS + 1):
            torque = np.zeros(16)
            torque[0] = 2 * sign
            rows.append({
                "front_contact_x_m": 0.38 if step < 3 + sign else 0.4,
                "pitch_error_rad": 0.05 + sign * 0.01 if step > 3 else 0.8,
                "applied_torque_nm": torque.tolist(),
                "longitudinal_drive_n": 14.0, "front_normal_load_n": 20.0 + sign,
                "rear_normal_load_n": 15.0 - sign, "support_sum_error_n": 0.0,
                "termination_reason": None, "torque_bound_violations": 0,
                "requested_transfer_n": float(sign), "allocated_transfer_n": float(sign),
            })
        result[name] = rows
    return result


def test_paired_rmse_uses_latest_actual_contact_and_common_end():
    rows = _branches()
    result = action.compare_branches(rows, 0.4)
    assert result["passed"]
    assert (result["common_entry_start_tick"], result["common_entry_end_tick"]) == (4, 64)
    assert result["positive_minus_negative_pitch_rmse_deg"] == pytest.approx(np.rad2deg(.02))
    assert result["max_abs_common_loaded_pos_minus_neg_axle_difference_n"] == 4.0
    rows["negative"].pop()
    shortened = action.compare_branches(rows, 0.4)
    assert shortened["common_entry_end_tick"] == 63
    assert not shortened["passed"]


def test_missing_coverage_torque_and_wheel_regressions_stay_visible():
    rows = _branches()
    for row in rows["positive"]:
        row["front_contact_x_m"] = None
        row["rear_normal_load_n"] = 0.0
    rows["negative"][0]["applied_torque_nm"][3] = 1.0
    rows["positive"][-1]["termination_reason"] = "fallen"
    result = action.compare_branches(rows, 0.4)
    assert not result["passed"]
    assert result["positive_minus_negative_pitch_rmse_deg"] is None
    assert result["max_abs_common_loaded_pos_minus_neg_axle_difference_n"] is None
    assert not result["checks"]["first_wheel_and_longitudinal_commands_match"]
    assert not result["checks"]["no_added_fall"]


def test_restoring_branch_state_rewinds_all_native_buffers_rng_delay_and_integrals(tmp_path, monkeypatch):
    class Buffer:
        def __init__(self, model):
            self.q = np.array([0.125, 1.25])

    def copy_data(target, model, source):
        target.q[:] = source.q

    monkeypatch.setattr(action.mujoco, "MjData", Buffer)
    monkeypatch.setattr(action.mujoco, "mj_copyData", copy_data)
    monkeypatch.setattr(action, "_integration_state", lambda model, data: data.q.copy())
    plant = SimpleNamespace(model=object(), data=Buffer(None), measurement_data=Buffer(None),
                            sampling_mode="synchronized", actuator_channel=None,
                            _last_control_interval_contact_wrench=np.array([21.0]),
                            _last_actuator_traces=())
    estimator = SimpleNamespace(model=object(), data=Buffer(None), velocity=np.array([.25]))
    measurements = SimpleNamespace(plant=plant, rng=np.random.default_rng(1234))
    source = SimpleNamespace(estimator=estimator, measurements=measurements, history=[4, 5])
    ep = SimpleNamespace(plant=plant, controller=SimpleNamespace(distance=1.0),
                         source=source, state=np.array([2.0]))
    snapshot = action.BranchCheckpoint(ep, tmp_path)
    identities, draws = [], []
    for _ in action.ARMS:
        identities.append(snapshot.restore(ep))
        draws.append(ep.source.measurements.rng.random())
        assert ep.source.measurements.plant is plant
        assert ep.source.history == [4, 5] and ep.controller.distance == 1.0
        np.testing.assert_array_equal(ep.state, [2.0])
        for buffer in (plant.data, plant.measurement_data, ep.source.estimator.data):
            np.testing.assert_array_equal(buffer.q, [.125, 1.25])
            buffer.q[:] = -3.0
        ep.state[:] = 7
        ep.source.history.clear()
        ep.controller.distance = 100
    assert identities[0] == identities[1] == identities[2]
    assert draws[0] == draws[1] == draws[2]
