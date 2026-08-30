from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from wheel_legged_control.d1.model import JOINT_POSITION_HIGH, D1Plant
from wheel_legged_control.d1.state_estimation import (
    D1EstimatorImpairments,
    D1MujocoTruthStateSource,
    D1NoisyDelayedStateSource,
    compensate_d1_state_constant_velocity,
    make_d1_state_source,
    prepare_d1_control_state,
)


def _yaw_rotation(angle_rad: float) -> np.ndarray:
    return np.asarray(
        (
            (np.cos(angle_rad), -np.sin(angle_rad), 0.0),
            (np.sin(angle_rad), np.cos(angle_rad), 0.0),
            (0.0, 0.0, 1.0),
        )
    )


def test_truth_source_returns_control_ready_immutable_snapshot() -> None:
    plant = D1Plant()
    source = D1MujocoTruthStateSource(plant)

    state = source.reset()

    assert state.sequence == 0
    assert state.control_time_s == pytest.approx(plant.data.time)
    assert state.measurement_time_s == pytest.approx(plant.data.time)
    np.testing.assert_allclose(state.base_position, plant.base_position)
    np.testing.assert_allclose(state.base_rpy, plant.base_rpy)
    np.testing.assert_allclose(state.projected_gravity_body, plant.projected_gravity_body)
    linear_body, angular_body = plant.base_velocity(local=True)
    linear_world, angular_world = plant.base_velocity(local=False)
    np.testing.assert_allclose(state.base_linear_velocity_body, linear_body)
    np.testing.assert_allclose(state.base_angular_velocity_body, angular_body)
    np.testing.assert_allclose(state.base_linear_velocity_world, linear_world)
    np.testing.assert_allclose(state.base_angular_velocity_world, angular_world)
    np.testing.assert_allclose(state.joint_position, plant.joint_position)
    np.testing.assert_allclose(state.joint_velocity, plant.joint_velocity)
    np.testing.assert_allclose(state.reduced_state(), plant.reduced_state())
    assert state.foot_position.shape == (4, 3)
    assert state.foot_jacobian.shape == (4, 3, 4)
    assert state.wheel_contact.shape == (4,)
    assert state.wheel_contact_point.shape == (4, 3)
    assert state.wheel_ground_contacts == plant.wheel_ground_contacts
    assert state.undesired_ground_contacts == plant.undesired_ground_contacts
    assert state.has_fallen() == plant.has_fallen()

    with pytest.raises(ValueError):
        state.joint_position[0] = 1.0
    with pytest.raises(ValueError):
        state.joint_position.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        state.sequence = 4  # type: ignore[misc]


def test_truth_source_requires_reset_and_advances_publication_sequence() -> None:
    plant = D1Plant()
    source = D1MujocoTruthStateSource(plant)

    with pytest.raises(RuntimeError, match="reset"):
        source.read()

    source.reset()
    plant.step(np.zeros(16))
    state = source.read()

    assert state.sequence == 1
    assert state.control_time_s == pytest.approx(plant.control_dt)
    assert state.measurement_time_s == pytest.approx(plant.control_dt)


def test_noisy_delayed_source_has_explicit_measurement_age() -> None:
    plant = D1Plant()
    source = D1NoisyDelayedStateSource(
        plant,
        impairments=D1EstimatorImpairments(delay_steps=2),
        seed=7,
    )

    initial = source.reset()
    states = []
    for _ in range(3):
        plant.step(np.zeros(16))
        states.append(source.read())

    assert initial.measurement_time_s == pytest.approx(0.0)
    assert states[0].measurement_time_s == pytest.approx(0.0)
    assert states[1].measurement_time_s == pytest.approx(0.0)
    assert states[2].measurement_time_s == pytest.approx(plant.control_dt)
    assert states[2].control_time_s == pytest.approx(3.0 * plant.control_dt)
    assert states[2].age_s == pytest.approx(2.0 * plant.control_dt)
    assert [state.sequence for state in states] == [1, 2, 3]


def test_noisy_source_is_reproducible_when_reset_with_the_same_seed() -> None:
    plant = D1Plant()
    impairments = D1EstimatorImpairments(
        base_position_std_m=0.01,
        base_rotation_std_rad=0.01,
        base_linear_velocity_std_mps=0.03,
        base_angular_velocity_std_radps=0.02,
        joint_position_std_rad=0.005,
        joint_velocity_std_radps=0.01,
        foot_position_std_m=0.003,
        foot_jacobian_std=0.002,
        contact_point_std_m=0.002,
    )
    source = D1NoisyDelayedStateSource(plant, impairments=impairments)

    plant.reset()
    first = source.reset(seed=123)
    plant.reset()
    second = source.reset(seed=123)

    for field in (
        "base_position",
        "base_rotation",
        "base_linear_velocity_body",
        "base_angular_velocity_body",
        "base_linear_velocity_world",
        "base_angular_velocity_world",
        "joint_position",
        "joint_velocity",
        "foot_position",
        "foot_jacobian",
        "wheel_contact_point",
    ):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


def test_factory_defaults_to_truth_and_switches_when_impairments_are_supplied() -> None:
    plant = D1Plant()

    assert isinstance(make_d1_state_source(plant), D1MujocoTruthStateSource)
    assert isinstance(
        make_d1_state_source(plant, impairments=D1EstimatorImpairments()),
        D1NoisyDelayedStateSource,
    )


def test_state_rejects_inconsistent_frame_representations() -> None:
    state = D1MujocoTruthStateSource(D1Plant()).reset()

    with pytest.raises(ValueError, match="proper orthonormal"):
        replace(state, base_rotation=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="velocities must agree"):
        replace(state, base_linear_velocity_world=np.ones(3))


def test_latency_compensation_is_identity_without_measurement_age() -> None:
    state = D1MujocoTruthStateSource(D1Plant()).reset()

    result = compensate_d1_state_constant_velocity(state)

    assert result.control_state is state
    assert result.status == "bypassed"
    assert result.applied_horizon_s == 0.0


def test_latency_compensation_extrapolates_twist_and_joint_kinematics() -> None:
    state = D1MujocoTruthStateSource(D1Plant()).reset()
    base_rotation = _yaw_rotation(0.3)
    linear_body = np.asarray((1.0, 0.5, -0.25))
    angular_body = np.asarray((0.0, 0.0, 1.0))
    joint_velocity = np.linspace(-0.2, 0.2, 16)
    delayed = replace(
        state,
        control_time_s=0.02,
        measurement_time_s=0.0,
        base_rotation=base_rotation,
        base_linear_velocity_body=linear_body,
        base_angular_velocity_body=angular_body,
        base_linear_velocity_world=base_rotation @ linear_body,
        base_angular_velocity_world=base_rotation @ angular_body,
        joint_velocity=joint_velocity,
        undesired_ground_contacts=3,
    )

    result = compensate_d1_state_constant_velocity(delayed)
    predicted = result.control_state

    horizon_s = delayed.age_s
    expected_rotation = base_rotation @ _yaw_rotation(horizon_s)
    expected_midpoint_rotation = base_rotation @ _yaw_rotation(0.5 * horizon_s)
    expected_position = delayed.base_position + expected_midpoint_rotation @ linear_body * horizon_s
    expected_foot_position = np.empty_like(delayed.foot_position)
    expected_foot_jacobian = np.empty_like(delayed.foot_jacobian)
    for leg_index in range(4):
        joint_slice = slice(4 * leg_index, 4 * (leg_index + 1))
        offset_body = base_rotation.T @ (delayed.foot_position[leg_index] - delayed.base_position)
        jacobian_body = base_rotation.T @ delayed.foot_jacobian[leg_index]
        expected_offset_body = offset_body + jacobian_body @ joint_velocity[joint_slice] * horizon_s
        expected_foot_position[leg_index] = (
            expected_position + expected_rotation @ expected_offset_body
        )
        expected_foot_jacobian[leg_index] = expected_rotation @ jacobian_body

    np.testing.assert_allclose(predicted.base_rotation, expected_rotation, atol=1e-10)
    np.testing.assert_allclose(predicted.base_position, expected_position, atol=1e-10)
    np.testing.assert_allclose(
        predicted.base_linear_velocity_world,
        expected_rotation @ linear_body,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        predicted.base_angular_velocity_world,
        expected_rotation @ angular_body,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        predicted.joint_position,
        delayed.joint_position + joint_velocity * horizon_s,
    )
    np.testing.assert_allclose(predicted.foot_position, expected_foot_position, atol=1e-10)
    np.testing.assert_allclose(predicted.foot_jacobian, expected_foot_jacobian, atol=1e-10)
    assert result.status == "applied"
    assert result.source_age_s == pytest.approx(0.02)
    assert result.estimate_time_s == pytest.approx(delayed.control_time_s)
    assert predicted.sequence == delayed.sequence
    assert predicted.control_time_s == delayed.control_time_s
    assert predicted.measurement_time_s == delayed.measurement_time_s
    np.testing.assert_array_equal(predicted.wheel_contact, delayed.wheel_contact)
    np.testing.assert_array_equal(predicted.wheel_contact_point, delayed.wheel_contact_point)
    assert predicted.undesired_ground_contacts == delayed.undesired_ground_contacts


@pytest.mark.parametrize(
    ("age_s", "expected_status"),
    (
        (0.049999, "applied"),
        (0.050000, "applied"),
        (0.050001, "horizon_exceeded"),
    ),
)
def test_latency_compensation_obeys_50_ms_horizon_boundary(
    age_s: float,
    expected_status: str,
) -> None:
    state = D1MujocoTruthStateSource(D1Plant()).reset()
    delayed = replace(state, control_time_s=age_s)

    result = compensate_d1_state_constant_velocity(delayed, max_horizon_s=0.05)

    assert result.status == expected_status
    if expected_status == "applied":
        assert result.control_state is not delayed
        assert result.applied_horizon_s == pytest.approx(age_s)
        assert result.estimate_time_s == pytest.approx(delayed.control_time_s)
    else:
        assert result.control_state is delayed
        assert result.applied_horizon_s == 0.0
        assert result.estimate_time_s == pytest.approx(delayed.measurement_time_s)


def test_latency_compensation_rejects_excessive_leg_joint_displacement() -> None:
    state = D1MujocoTruthStateSource(D1Plant()).reset()
    joint_velocity = np.zeros(16)
    joint_velocity[0] = 1.0
    delayed = replace(state, control_time_s=0.02, joint_velocity=joint_velocity)

    result = compensate_d1_state_constant_velocity(
        delayed,
        max_leg_joint_delta_rad=0.01,
    )

    assert result.control_state is delayed
    assert result.status == "kinematic_horizon_exceeded"
    assert result.applied_horizon_s == 0.0
    assert result.estimate_time_s == pytest.approx(delayed.measurement_time_s)


def test_latency_compensation_rejects_predicted_leg_joint_limit_violation() -> None:
    state = D1MujocoTruthStateSource(D1Plant()).reset()
    joint_position = state.joint_position.copy()
    joint_velocity = np.zeros(16)
    joint_position[0] = JOINT_POSITION_HIGH[0] - 0.001
    joint_velocity[0] = 0.1
    delayed = replace(
        state,
        control_time_s=0.02,
        joint_position=joint_position,
        joint_velocity=joint_velocity,
    )

    result = compensate_d1_state_constant_velocity(delayed)

    assert joint_velocity[0] * delayed.age_s < 0.35
    assert result.control_state is delayed
    assert result.status == "kinematic_horizon_exceeded"
    assert result.applied_horizon_s == 0.0


def test_prepare_control_state_none_preserves_delayed_measurement() -> None:
    state = D1MujocoTruthStateSource(D1Plant()).reset()
    delayed = replace(state, control_time_s=0.02)

    result = prepare_d1_control_state(delayed, latency_compensation="none")

    assert result.control_state is delayed
    assert result.method == "none"
    assert result.status == "disabled"
    assert result.source_age_s == pytest.approx(0.02)
    assert result.applied_horizon_s == 0.0
    assert result.estimate_time_s == pytest.approx(delayed.measurement_time_s)


def test_prepare_control_state_rejects_unknown_mode() -> None:
    state = D1MujocoTruthStateSource(D1Plant()).reset()

    with pytest.raises(ValueError, match="latency_compensation"):
        prepare_d1_control_state(state, latency_compensation="kalman")


@pytest.mark.parametrize(
    ("keyword", "value"),
    (("delay_steps", -1), ("joint_position_std_rad", -0.1), ("contact_flip_probability", 1.1)),
)
def test_impairments_reject_invalid_values(keyword: str, value: float) -> None:
    with pytest.raises(ValueError):
        D1EstimatorImpairments(**{keyword: value})
