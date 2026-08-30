from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.state_estimation import (
    D1EstimatorImpairments,
    D1MujocoTruthStateSource,
    D1NoisyDelayedStateSource,
    make_d1_state_source,
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


@pytest.mark.parametrize(
    ("keyword", "value"),
    (("delay_steps", -1), ("joint_position_std_rad", -0.1), ("contact_flip_probability", 1.1)),
)
def test_impairments_reject_invalid_values(keyword: str, value: float) -> None:
    with pytest.raises(ValueError):
        D1EstimatorImpairments(**{keyword: value})
