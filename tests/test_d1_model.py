import numpy as np
import pytest

from wheel_legged_control.d1.controllers import D1Command, D1VMCController
from wheel_legged_control.d1.hierarchical import (
    D1_LONGITUDINAL_FORCE_LIMIT_N,
    D1LQRVMCController,
    D1MPCVMCController,
)
from wheel_legged_control.d1.model import JOINT_TORQUE_LIMIT, D1Plant
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource


def test_d1_model_has_expected_full_body_dimensions() -> None:
    plant = D1Plant()
    summary = plant.summary
    assert (summary.nq, summary.nv, summary.nu) == (23, 22, 16)
    assert summary.body_count == 18
    assert summary.joint_count == 17
    assert summary.geom_count >= 43
    assert np.isclose(summary.total_mass_kg, 48.14686526)


def test_actuator_strength_updates_the_plant_and_allocator_visible_limit() -> None:
    plant = D1Plant()

    plant.set_domain(actuator_strength_scale=1.05)

    expected = 1.05 * JOINT_TORQUE_LIMIT
    np.testing.assert_allclose(plant.actuator_torque_limit_nm, expected)
    np.testing.assert_allclose(plant.model.actuator_ctrlrange[:, 1], expected)
    np.testing.assert_allclose(plant.model.actuator_forcerange[:, 1], expected)


def test_vmc_holds_four_wheel_contact_pose() -> None:
    plant = D1Plant()
    controller = D1VMCController(plant)
    states = D1MujocoTruthStateSource(plant)
    state = states.reset()
    for _ in range(100):
        plant.step(controller.compute(D1Command(), state))
        state = states.read()
    assert not plant.has_fallen()
    assert plant.wheel_ground_contacts == 4
    assert plant.undesired_ground_contacts == 0
    assert plant.base_position[2] > 0.40
    assert abs(plant.base_rpy[1]) < np.deg2rad(2.0)


def test_measured_wheel_contact_wrench_is_world_frame_ground_reaction() -> None:
    plant = D1Plant()
    controller = D1VMCController(plant)
    states = D1MujocoTruthStateSource(plant)
    state = states.reset()
    for _ in range(100):
        plant.step(
            controller.compute(D1Command(), state),
            measure_contact_wrench=True,
            contact_wrench_reference_world_m=state.base_position,
        )
        state = states.read()

    measurement = plant.measure_wheel_contact_wrench()
    interval_measurement = plant.last_control_interval_contact_wrench

    assert measurement.wheel_force_world_n.shape == (4, 3)
    assert measurement.wrench_world.shape == (6,)
    assert measurement.mean_contact_point_count_by_wheel.shape == (4,)
    assert np.all(measurement.mean_contact_point_count_by_wheel > 0)
    assert measurement.physics_sample_count == 1
    assert interval_measurement.physics_sample_count == plant.physics_steps
    assert np.all(interval_measurement.active_sample_fraction_by_wheel > 0.0)
    assert interval_measurement.wrench_world[2] == pytest.approx(
        plant.summary.total_mass_kg * 9.81,
        rel=0.01,
    )
    assert measurement.wrench_world[2] > 0.8 * plant.summary.total_mass_kg * 9.81
    np.testing.assert_allclose(
        measurement.wrench_world[:3],
        np.sum(measurement.wheel_force_world_n, axis=0),
        atol=1e-10,
    )
    assert not measurement.wheel_force_world_n.flags.writeable
    assert not measurement.wrench_world.flags.writeable

    shifted_reference = measurement.reference_position_world_m + np.asarray((0.1, -0.2, 0.3))
    shifted = plant.measure_wheel_contact_wrench(shifted_reference)
    expected_moment = measurement.wrench_world[3:] + np.cross(
        measurement.reference_position_world_m - shifted_reference,
        measurement.wrench_world[:3],
    )
    np.testing.assert_allclose(shifted.wrench_world[3:], expected_moment, atol=1e-10)


def test_identified_lqr_stabilizes_and_drives_forward() -> None:
    plant = D1Plant()
    controller = D1LQRVMCController(plant)
    states = D1MujocoTruthStateSource(plant)
    state = states.reset()
    assert np.max(np.abs(controller.closed_loop_eigenvalues)) < 1.0
    for _ in range(150):
        plant.step(controller.compute(D1Command(forward_velocity_mps=0.35), state))
        state = states.read()
    assert not plant.has_fallen()
    assert plant.base_position[0] > 0.15
    assert plant.base_velocity(local=False)[0][0] > 0.10


def test_mpc_respects_outer_force_constraint() -> None:
    plant = D1Plant()
    controller = D1MPCVMCController(plant, horizon=10)
    states = D1MujocoTruthStateSource(plant)
    state = states.reset()
    plant.step(controller.compute(D1Command(forward_velocity_mps=0.5), state))
    assert abs(controller.last_longitudinal_force_n) <= D1_LONGITUDINAL_FORCE_LIMIT_N
    controller.longitudinal_force_limit_n = 50.0
    state = states.read()
    plant.step(controller.compute(D1Command(forward_velocity_mps=0.8), state))
    assert abs(controller.last_longitudinal_force_n) <= 50.0
    assert np.isfinite(plant.data.qpos).all()


def test_vmc_feedback_cannot_bypass_the_state_estimate() -> None:
    plant = D1Plant()
    state = D1MujocoTruthStateSource(plant).reset()
    controller = D1VMCController(plant)
    command = D1Command(forward_velocity_mps=0.2, yaw_rate_rps=0.1)
    expected = controller.compute(command, state)

    qpos, qvel = plant.simulation_state()
    qpos[0] += 3.0
    qpos[2] += 0.2
    qvel[:6] += 1.0
    plant.set_simulation_state(qpos, qvel)

    actual = controller.compute(command, state)
    np.testing.assert_array_equal(actual, expected)


def test_vmc_body_level_command_is_invariant_to_global_yaw() -> None:
    command = D1Command(
        forward_velocity_mps=0.3,
        yaw_rate_rps=0.2,
        roll_rad=0.05,
        pitch_rad=-0.04,
    )
    torques = []
    for yaw_rad in (0.0, np.pi / 2.0):
        plant = D1Plant()
        plant.reset(
            base_quaternion=np.asarray((np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)))
        )
        state = D1MujocoTruthStateSource(plant).reset()
        torques.append(D1VMCController(plant).compute(command, state))

    np.testing.assert_allclose(torques[1], torques[0], atol=1e-8)


def test_vmc_nominal_constants_do_not_leak_randomized_mass() -> None:
    plant = D1Plant()
    nominal_mass = plant.nominal_total_mass_kg
    plant.set_domain(base_mass_scale=1.12)

    controller = D1VMCController(plant)

    assert controller.total_mass_kg == nominal_mass
    assert plant.summary.total_mass_kg > nominal_mass
