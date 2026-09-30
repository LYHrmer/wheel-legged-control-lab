"""Pure C18 filtered-common-integral fixtures; no model construction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "continuation17"))
sys.path.insert(0, str(HERE))

from controller18 import FullDriveController18
from residual17 import compute_residual17
from residual18 import compute_residual18


WHEELS = (3, 7, 11, 15)
Q = np.tile((0.0, .8, -1.5, 0.0), 4)


def inputs(*, servo=.45, body=.5, wheel_omega=4.5,
           nominal_wheels=None, common=True, variant="combined"):
    qdot = np.zeros(16)
    qdot[list(WHEELS)] = wheel_omega
    return {
        "joint_position_rad": Q.copy(),
        "joint_velocity_rad_s": qdot,
        "nominal_joint_target_rad": Q.copy(),
        "nominal_wheel_speed_rad_s": np.full(4, servo/.087)
        if nominal_wheels is None else np.asarray(nominal_wheels, dtype=float),
        "support_torque_nm": np.zeros(16),
        "base_rotation_world_from_body": np.eye(3),
        "foot_jacobian_world": np.zeros((4, 3, 4)),
        "body_com_forward_mps": body,
        "servo_forward_mps": servo,
        "controller_variant": variant,
        "action_enabled": True,
        "leg_longitudinal_damping_active": common,
        "body_common_p_active": common,
    }


def calculate(state, z=0.0, integral=None, action=None):
    return compute_residual18(
        state, np.zeros(16) if action is None else action,
        np.zeros(4) if integral is None else integral, z,
    )


def test_constant_difference_filter_converges_to_c17_dc_reference():
    state = inputs()
    old = compute_residual17(state, np.zeros(16), np.zeros(4))
    z = 0.0
    first = calculate(state, z)
    d = first["wheel_common_reference_difference_rad_s"]
    assert first["wheel_common_reference_alpha"] == pytest.approx(1/21)
    assert first["wheel_common_reference_tau_s"] == pytest.approx(.20)
    assert first["wheel_common_reference_z_after_rad_s"] == pytest.approx(d/21)
    assert first["wheel_integral_error_rad_s"].mean() == pytest.approx(
        np.mean(first["wheel_speed_error_rad_s"])-d/21)
    for _ in range(220):
        result = calculate(state, z)
        z = result["wheel_common_reference_z_after_rad_s"]
    np.testing.assert_allclose(result["wheel_integral_error_rad_s"],
                               old["wheel_integral_error_rad_s"], atol=1e-4)


def test_transient_preserves_differential_and_16d_action_path():
    state = inputs(nominal_wheels=(4.0, 5.0, 6.0, 7.0), wheel_omega=4.0)
    action = np.full(16, .2)
    new = calculate(state, action=action)
    old = compute_residual17(state, action, np.zeros(4))
    np.testing.assert_allclose(new["wheel_integral_error_rad_s"]-
                               np.mean(new["wheel_integral_error_rad_s"]),
                               new["wheel_speed_error_rad_s"]-
                               np.mean(new["wheel_speed_error_rad_s"]), atol=1e-12)
    for name in ("applied_action", "wheel_action_offset_rad_s", "leg_action_offset_rad",
                 "wheel_speed_target_rad_s", "common_wheel_delta_torque_nm",
                 "consumed_nominal_wheel_speed_rad_s", "consumed_support_torque_nm"):
        np.testing.assert_array_equal(new[name], old[name])


def test_inactive_stop_clears_filter_and_keeps_protection():
    state = inputs(servo=0.0, body=0.0, wheel_omega=30.0,
                   nominal_wheels=(30.0,)*4, common=False)
    state["action_enabled"] = False
    result = calculate(state, z=5.0, integral=np.full(4, 4.0), action=np.ones(16))
    assert result["wheel_common_reference_z_before_rad_s"] == 5.0
    assert result["wheel_common_reference_z_after_rad_s"] == 0.0
    assert result["wheel_integral_mode"] == "differential_wheel_feedback"
    np.testing.assert_array_equal(result["wheel_integral_error_rad_s"],
                                  result["wheel_speed_error_rad_s"])
    assert not np.any(result["applied_action"])
    assert np.all(result["speed_outward_mask"][list(WHEELS)])
    assert not np.any(result["safe_torque_nm"][list(WHEELS)])


def test_antiwindup_unwinds_using_filtered_integral_error():
    state = inputs(servo=.45, body=1.0, wheel_omega=0.0,
                   nominal_wheels=(8.0,)*4)
    before = np.full(4, -3.0)
    result = calculate(state, z=15.0, integral=before)
    assert np.all(result["wheel_candidate_request_nm"] > 12.0)
    assert np.all(result["wheel_integral_error_rad_s"] < 0.0)
    assert np.all(result["wheel_integral_commit_mask"])
    assert np.all(result["wheel_integral_after_nm"] < before)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), True, [1.0]))
def test_explicit_filter_state_rejects_invalid_values(bad):
    with pytest.raises((TypeError, ValueError)):
        calculate(inputs(), z=bad)


def test_adapter_reset_clears_both_integrators_and_stop_latch():
    class FakeNominal:
        def reset(self):
            self.called = True

    adapter = object.__new__(FullDriveController18)
    adapter._nominal = FakeNominal()
    adapter._wheel_integral_nm = np.ones(4)
    adapter._wheel_common_reference_z_rad_s = 3.0
    adapter._stop_latched = True
    adapter.reset()
    assert adapter._nominal.called
    assert adapter.wheel_common_reference_z_rad_s == 0.0
    np.testing.assert_array_equal(adapter.wheel_integral_nm, np.zeros(4))
    assert not adapter.stop_latched
    assert adapter._preview is None
