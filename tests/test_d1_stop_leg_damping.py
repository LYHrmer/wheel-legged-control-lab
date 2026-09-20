"""Controller algebra and safety ordering; no physics steps or integrations."""
import json
from dataclasses import replace

import numpy as np
import pytest

from scripts.d1_stop_leg_damping import STOP_LEG_DAMPING_NSPM, StopLegDampingController
from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.model import JOINT_POSITION_HIGH, D1Plant
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource
from wheel_legged_control.d1.wheel_leg_controller import D1WheelLegController


@pytest.fixture(scope="module")
def state():
    plant = D1Plant(sampling_mode="synchronized")
    result = D1MujocoTruthStateSource(plant).reset()
    assert plant.data.time == 0
    return result


@pytest.mark.parametrize("enabled", (False, True))
def test_latch_reset_preview_and_single_parent_call(state, monkeypatch, enabled):
    parent_compute = D1WheelLegController.compute
    called = []

    def tracked(self, *args, **kwargs):
        torque = parent_compute(self, *args, **kwargs)
        called.append((torque, self.last_result))
        return torque

    monkeypatch.setattr(D1WheelLegController, "compute", tracked)
    controller = StopLegDampingController(enabled=enabled)
    for vx, active in ((0., False), (.25, False), (0., True), (0., True), (-.2, False), (0., True)):
        command = D1Command(forward_velocity_mps=vx)
        previous = controller.last_damping
        controller.nominal_targets(command, state)
        controller.nominal_targets(command, state)
        assert controller.last_damping is previous
        count = len(called)
        torque = controller.compute(command, state, np.zeros(8))
        assert len(called) == count + 1
        assert controller.last_damping.active == (enabled and active)
        if not enabled or not active:
            assert torque is called[-1][0]
            assert controller.last_result is called[-1][1]
    controller.reset()
    assert controller.last_damping is None and controller.previous_forward_mps is None
    assert not controller.stop_damping_active
    controller.compute(D1Command(), state, np.zeros(8))
    assert not controller.last_damping.active


@pytest.mark.parametrize("direction", (-1, 1))
def test_dissipation_wheel_identity_and_single_pi_update(state, direction):
    velocity = direction*np.tile([.2, -.4, .1, .3], 4)
    state = replace(state, joint_velocity=velocity)
    controller, baseline = StopLegDampingController(), D1WheelLegController()
    for vx in (direction*.25, 0.):
        command = D1Command(forward_velocity_mps=vx)
        controller.compute(command, state, np.zeros(8))
        baseline.compute(command, state, np.zeros(8))
    record = controller.last_damping
    assert record.active and record.joint_power_w < 0
    assert record.joint_power_w == pytest.approx(-STOP_LEG_DAMPING_NSPM*np.sum(record.leg_relative_forward_mps**2))
    np.testing.assert_array_equal(record.delta_torque_nm[3::4], np.zeros(4))
    np.testing.assert_array_equal(controller.last_result.torque_nm[3::4], baseline.last_result.torque_nm[3::4])
    np.testing.assert_array_equal(controller.control_memory.wheel_integral_nm, baseline.control_memory.wheel_integral_nm)
    with pytest.raises(ValueError):
        record.delta_torque_nm[0] = 100
    assert record.total_requested_torque_nm is not baseline.last_result.requested_torque_nm


def test_increment_combines_before_clipping_and_respects_outward_position_guard(state, monkeypatch):
    baseline = D1WheelLegController()
    baseline.compute(D1Command(), state, np.zeros(8))
    template = baseline.last_result
    request = np.zeros(16)
    request[0] = 100.

    def synthetic_parent(self, *args, **kwargs):
        self.last_result = replace(template, leg_pd_nm=request, support_nm=np.zeros(16),
                                   wheel_nm=np.zeros(16), requested_torque_nm=request,
                                   torque_nm=np.clip(request, -80, 80))
        return self.last_result.torque_nm.copy()

    monkeypatch.setattr(D1WheelLegController, "compute", synthetic_parent)
    jacobian = np.zeros((4, 3, 4))
    jacobian[0, 0, 0] = 1.
    velocity = np.zeros(16)
    velocity[0] = .4
    state = replace(state, base_rotation=np.eye(3), foot_jacobian=jacobian, joint_velocity=velocity)
    controller = StopLegDampingController()
    controller.compute(D1Command(forward_velocity_mps=.25), state, np.zeros(8))
    torque = controller.compute(D1Command(), state, np.zeros(8))
    assert torque[0] == pytest.approx(100.-STOP_LEG_DAMPING_NSPM*.4)
    position = state.joint_position.copy()
    position[0] = JOINT_POSITION_HIGH[0]
    torque = controller.compute(D1Command(), replace(state, joint_position=position), np.zeros(8))
    assert torque[0] == 0
    velocity[0] = 21.
    torque = controller.compute(D1Command(), replace(state, foot_jacobian=np.zeros_like(jacobian),
                                                     joint_velocity=velocity), np.zeros(8))
    assert torque[0] == 0  # Outward at the positive speed bound.


@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("action", (np.ones(8)*.01, np.ones(7), np.ones(8)*np.nan))
def test_invalid_residual_rejected_before_parent_or_latch(state, enabled, action):
    controller = StopLegDampingController(enabled=enabled)
    controller.compute(D1Command(forward_velocity_mps=.25), state, np.zeros(8))
    record, result = controller.last_damping, controller.last_result
    memory = controller.control_memory.wheel_integral_nm.copy()
    with pytest.raises(ValueError):
        controller.compute(D1Command(), state, action)
    assert controller.last_damping is record and controller.last_result is result
    assert controller.previous_forward_mps == .25
    np.testing.assert_array_equal(controller.control_memory.wheel_integral_nm, memory)


@pytest.mark.parametrize("value", (0, 1, None, "true"))
def test_enabled_requires_boolean(value):
    with pytest.raises(TypeError):
        StopLegDampingController(enabled=value)


@pytest.mark.parametrize("enabled", (False, True))
def test_diagnostic_env_rejects_old_heading_checkpoint_before_loading(tmp_path, enabled):
    from scripts.d1_heading_tracking_env import D1HeadingTrackingEnv, load_heading_policy
    from scripts.probe_d1_heading_stop_damping import StopDampingEnv

    original = D1HeadingTrackingEnv(episode_seconds=.02)
    candidate = StopDampingEnv(damping_enabled=enabled, episode_seconds=.02)
    try:
        metadata = tmp_path / "old_heading.metadata.json"
        metadata.write_text(json.dumps({"recorded_episode": {
            "heading_task_config": original.heading_task_config}}))
        with pytest.raises(ValueError, match="incompatible"):
            load_heading_policy(tmp_path / "model_must_not_be_loaded.zip", metadata, candidate)
        with pytest.raises(ValueError, match="exactly zero"):
            candidate.step(np.ones(8)*.1)
        assert candidate.plant.data.time == original.plant.data.time == 0
    finally:
        candidate.close()
        original.close()
