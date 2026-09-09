"""All controller/source choices cross the same real simulation step Interface."""

from dataclasses import replace

import numpy as np
import pytest

from wheel_legged_control.d1.actuator_channel import ActuatorChannel, ActuatorChannelConfig
from wheel_legged_control.d1.control_loop import (
    D1ControlLoop,
    D1ForceControllerAdapter,
    D1MotionCommand,
    D1WheelLegControllerAdapter,
)
from wheel_legged_control.d1.hierarchical import D1LQRVMCController, D1MPCVMCController
from wheel_legged_control.d1.model import JOINT_TORQUE_LIMIT, D1Plant
from wheel_legged_control.d1.state_estimation import D1EstimatorImpairments
from wheel_legged_control.d1.state_provider import D1StateProviderConfig, build_d1_state_provider
from wheel_legged_control.d1.training_terrain import TrainingGroundReference
from wheel_legged_control.d1.wheel_leg_controller import D1WheelLegController


def make_loop(controller="lqr", source="oracle"):
    actuator = ActuatorChannel(ActuatorChannelConfig(torque_limit_nm=tuple(JOINT_TORQUE_LIMIT)))
    plant = D1Plant(sampling_mode="synchronized", actuator_channel=actuator)
    if source == "imu_encoder_fusion":
        config = D1StateProviderConfig(
            source, initial_position_m=(0, 0, 0.455), initial_rpy_rad=(0, 0, 0)
        )
        provider = build_d1_state_provider(plant, config)
    else:
        config = D1StateProviderConfig(
            source,
            impairments=D1EstimatorImpairments(delay_steps=1)
            if source == "truth_impairment"
            else None,
        )
        provider = build_d1_state_provider(
            plant, config, oracle_ground_query=lambda x, y: TrainingGroundReference(0, 0, 0)
        )
    adapter = (
        D1WheelLegControllerAdapter(D1WheelLegController())
        if controller == "wheel_leg"
        else D1ForceControllerAdapter(
            D1LQRVMCController(plant) if controller == "lqr" else D1MPCVMCController(plant)
        )
    )
    return D1ControlLoop(plant, provider, adapter)


@pytest.mark.parametrize("controller", ["lqr", "mpc", "wheel_leg"])
@pytest.mark.parametrize("source", ["oracle", "truth_impairment", "imu_encoder_fusion"])
def test_sources_and_controllers_follow_one_clock_and_receipt_chain(controller, source):
    loop = make_loop(controller, source)
    loop.reset(seed=7)
    command = D1MotionCommand(0.15, 0.08)
    previous = None
    for tick in range(6):
        decision = loop.prepare(command)
        assert loop.prepare(command) is decision
        assert decision.context.proposal.tick == tick
        if previous is None:
            np.testing.assert_array_equal(decision.context.previous_normalized_action, 0)
        else:
            np.testing.assert_array_equal(decision.context.previous_normalized_action, previous)
        action = np.full(loop.controller.action_size, 1.2 if tick % 2 else -0.1)
        transition = loop.step(action)
        previous = np.clip(action, -1, 1)
        np.testing.assert_array_equal(transition.raw_action, action)
        np.testing.assert_array_equal(transition.receipt.normalized_action, previous)
        assert transition.state.sequence == transition.truth.sequence == tick + 1
        assert transition.state.control_time_s == loop.plant.data.time
        np.testing.assert_array_equal(transition.truth.base_position, loop.plant.data.qpos[:3])
        assert len(loop.plant.last_control_interval_actuator_traces) == 5
        for trace in loop.plant.last_control_interval_actuator_traces:
            np.testing.assert_array_equal(trace.requested_nm, transition.requested_torque_nm)


def test_actor_decision_does_not_consume_audit_truth(monkeypatch):
    reference, polluted = (
        make_loop(source="imu_encoder_fusion"),
        make_loop(source="imu_encoder_fusion"),
    )
    for loop in (reference, polluted):
        loop.reset(seed=7)
    original = polluted._truth_source.read

    def contaminated_truth():
        state = original()
        return replace(state, base_position=state.base_position + np.asarray((10, 0, 0.1)))

    monkeypatch.setattr(polluted._truth_source, "read", contaminated_truth)
    for _ in range(8):
        a, b = reference.prepare(D1MotionCommand(0.1)), polluted.prepare(D1MotionCommand(0.1))
        np.testing.assert_array_equal(a.context.state.base_position, b.context.state.base_position)
        assert a.world_command == b.world_command
        assert a.context.proposal.baseline == b.context.proposal.baseline
        left, right = reference.step(np.zeros(2)), polluted.step(np.zeros(2))
        np.testing.assert_array_equal(left.requested_torque_nm, right.requested_torque_nm)
        assert right.truth.base_position[0] - left.truth.base_position[0] == pytest.approx(10)


def test_invalid_action_does_not_consume_decision_or_advance_integrators():
    loop = make_loop()
    loop.reset()
    decision = loop.prepare(D1MotionCommand(0.1))
    for invalid in (np.zeros(8), np.asarray((np.nan, 0))):
        with pytest.raises(ValueError, match="action"):
            loop.step(invalid)
        assert loop.plant.data.time == 0
        assert loop.prepare(D1MotionCommand(0.1)) is decision
    loop.step(np.zeros(2))


def test_decision_cannot_be_silently_replaced_or_consumed_twice():
    loop = make_loop()
    with pytest.raises(RuntimeError, match="reset"):
        loop.prepare(D1MotionCommand())
    loop.reset()
    loop.prepare(D1MotionCommand(0.1))
    with pytest.raises(RuntimeError, match="replace"):
        loop.prepare(D1MotionCommand(0.2))
    loop.step(np.zeros(2))
    with pytest.raises(RuntimeError, match="prepare"):
        loop.step(np.zeros(2))


def test_reset_clears_all_runtime_memories_and_repeats_initial_rollout():
    loop = make_loop(source="imu_encoder_fusion")
    trajectories = []
    for _ in range(2):
        loop.reset(seed=3)
        states = []
        for _ in range(10):
            loop.prepare(D1MotionCommand(0.1, 0.1))
            loop.step(np.ones(2) * 0.1)
            states.append(loop.plant.data.qpos.copy())
        trajectories.append(np.asarray(states))
    np.testing.assert_array_equal(*trajectories)


def test_fusion_placement_prior_is_not_silently_replaced_by_spawn_truth():
    loop = make_loop(source="imu_encoder_fusion")
    with pytest.raises(ValueError, match="prior"):
        loop.reset(base_position=(1, 0, 0.455))


def test_physics_failure_requires_complete_reset(monkeypatch):
    loop = make_loop()
    loop.reset()
    loop.prepare(D1MotionCommand())

    def failure(*args, **kwargs):
        raise ValueError("injected step failure")

    monkeypatch.setattr(loop.plant, "step", failure)
    with pytest.raises(ValueError, match="injected"):
        loop.step(np.zeros(2))
    with pytest.raises(RuntimeError, match="reset"):
        loop.prepare(D1MotionCommand())


@pytest.mark.parametrize(
    "kwargs", [{"forward_velocity_mps": np.nan}, {"yaw_rate_rps": 2}, {"clearance_m": 0.1}]
)
def test_invalid_motion_command_fails_before_control(kwargs):
    with pytest.raises(ValueError):
        D1MotionCommand(**kwargs)


@pytest.mark.parametrize("terrain_feedforward", [False, True])
def test_force_adapter_damping_uses_origin_velocity_consistently(terrain_feedforward):
    loop = make_loop()
    loop.reset()
    state = loop.provider.read()
    linear, angular = np.asarray((0.3, -0.2, 0.1)), np.asarray((0.2, 0.3, -0.1))
    state = replace(
        state,
        base_linear_velocity_body=linear,
        base_linear_velocity_world=linear,
        base_angular_velocity_body=angular,
        base_angular_velocity_world=angular,
    )
    adapter = loop.controller
    adapter.terrain_velocity_feedforward = terrain_feedforward
    ground = TrainingGroundReference(0, -0.04, 0.03)
    from wheel_legged_control.d1.controllers import D1Command

    command = D1Command()
    origin = linear - np.cross(angular, adapter.controller.low_level.nominal_base_com_offset_body_m)
    expected = linear[2] - origin[2]
    if terrain_feedforward:
        expected += -np.tan(ground.pitch_rad) * origin[0] + np.tan(ground.roll_rad) * origin[1]
    expected *= adapter.controller.low_level.height_kd
    proposal = adapter.preview(command, state, ground)
    assert proposal.baseline.vertical_feedforward_force_n == pytest.approx(expected)
    # Computing with an inconsistent preview FF would violate the controller's
    # prepared-decision contract; the same correction must cross both calls.
    torque = adapter.compute(command, state, ground, np.zeros(2))
    assert torque.shape == (16,) and np.isfinite(torque).all()


@pytest.mark.parametrize("value", [np.asarray([0.1]), True, 1j, "0.1"])
def test_motion_command_values_are_real_scalars(value):
    with pytest.raises(ValueError):
        D1MotionCommand(forward_velocity_mps=value)
