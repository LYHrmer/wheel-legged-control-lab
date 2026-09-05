from __future__ import annotations

import mujoco
import numpy as np
import pytest

from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.inverse_dynamics import D1InverseDynamicsController
from wheel_legged_control.d1.model import JOINT_TORQUE_LIMIT, D1Plant
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource


def test_airborne_robot_does_not_invent_support_and_its_com_falls_under_gravity() -> None:
    plant = D1Plant()
    plant.reset(base_position=np.asarray((0.0, 0.0, 2.0)))
    state = D1MujocoTruthStateSource(plant).reset()
    controller = D1InverseDynamicsController()

    torque = controller.compute(D1Command(base_height_m=2.0), state)
    result = controller.last_result

    assert result is not None and result.status == "solved"
    assert torque.shape == (16,)
    assert np.all(np.abs(torque) <= JOINT_TORQUE_LIMIT + 1e-6)
    np.testing.assert_allclose(result.contact_force_world_n, 0.0, atol=1e-7)
    # Independent whole-robot COM Jacobian: internal torques cannot change
    # ballistic COM acceleration, even though they can accelerate the base.
    com_jacobian = np.zeros((3, plant.model.nv))
    mujoco.mj_jacSubtreeCom(plant.model, plant.data, com_jacobian, plant.base_body_id)
    np.testing.assert_allclose(
        com_jacobian @ result.generalized_acceleration,
        np.asarray((0.0, 0.0, -9.81)), atol=2e-5,
    )


def test_rolling_acceleration_matches_independent_moving_point_difference() -> None:
    plant = D1Plant()
    plant.reset(base_position=np.asarray((0.0, 0.0, .45)))
    plant.data.qvel[:6] = (.15, .01, -.03, .2, -.15, .1)
    plant.data.qvel[6:] = np.tile((.05, -.04, .03, 3.0), 4)
    mujoco.mj_forward(plant.model, plant.data)
    state = D1MujocoTruthStateSource(plant).reset()
    controller = D1InverseDynamicsController()

    controller.compute(D1Command(base_height_m=.45), state)
    result = controller.last_result
    assert result is not None
    for wheel, body_id in enumerate(plant.wheel_body_ids_by_leg):
        # Recompute geometric rim points after integrating q, independently of
        # the controller's analytic derivative. A body-fixed point is wrong here.
        samples = []
        for dt in (-1e-6, 0.0, 1e-6):
            data = mujoco.MjData(plant.model)
            data.qpos[:] = plant.data.qpos
            mujoco.mj_integratePos(plant.model, data.qpos, plant.data.qvel, dt)
            mujoco.mj_fwdPosition(plant.model, data)
            axis = data.xaxis[plant.joint_ids[wheel * 4 + 3]]
            normal = np.asarray((0.0, 0.0, 1.0))
            radial = normal - axis * axis[2]
            point = data.xpos[body_id] - .087 * radial / np.linalg.norm(radial)
            jacobian = np.zeros((3, 22))
            mujoco.mj_jac(plant.model, data, jacobian, None, point, body_id)
            tangent = np.cross(axis, normal)
            tangent /= np.linalg.norm(tangent)
            frame = np.stack((tangent, np.cross(normal, tangent), normal))
            samples.append(frame @ jacobian)
        velocity = samples[1] @ plant.data.qvel
        bias = (samples[2] - samples[0]) @ plant.data.qvel / 2e-6
        acceleration = samples[1] @ result.generalized_acceleration + bias
        # Local rolling and normal velocities decay at 20/s; differentiate both
        # the rim Jacobian and the moving tangent frame, not only world velocity.
        np.testing.assert_allclose(acceleration[[0, 2]], -20 * velocity[[0, 2]], atol=2e-5)


def test_four_wheel_stance_supports_weight_without_base_acceleration() -> None:
    plant = D1Plant()
    plant.reset(base_position=np.asarray((0.0, 0.0, 0.45)))
    state = D1MujocoTruthStateSource(plant).reset()
    assert state.wheel_ground_contacts == 4
    controller = D1InverseDynamicsController()

    torque = controller.compute(D1Command(base_height_m=0.45), state)
    result = controller.last_result

    assert result is not None
    np.testing.assert_allclose(
        result.contact_force_world_n.sum(axis=0),
        (0.0, 0.0, plant.model.body_mass.sum() * 9.81), atol=1.0,
    )
    np.testing.assert_allclose(result.generalized_acceleration[:6], 0.0, atol=0.03)
    assert np.all(np.abs(torque) <= JOINT_TORQUE_LIMIT + 1e-6)


def test_nominal_standing_remains_upright_for_three_seconds() -> None:
    plant = D1Plant()
    source = D1MujocoTruthStateSource(plant)
    state = source.reset()
    controller = D1InverseDynamicsController()
    for _ in range(300):
        torque = controller.compute(D1Command(), state)
        plant.step(torque)
        state = source.read()
        assert abs(state.base_position[2] - .455) < .04
        assert np.max(np.abs(state.base_rpy[:2])) < np.deg2rad(5.0)
        assert state.undesired_ground_contacts == 0


def test_invalid_command_cannot_leave_a_stale_success_result() -> None:
    state = D1MujocoTruthStateSource(D1Plant()).reset()
    controller = D1InverseDynamicsController()
    controller.compute(D1Command(), state)
    with pytest.raises(ValueError, match="finite"):
        controller.compute(D1Command(forward_velocity_mps=float("nan")), state)
    assert controller.last_result is None


def test_bad_contact_normal_is_rejected_before_solving() -> None:
    from dataclasses import replace

    plant = D1Plant()
    plant.reset(base_position=np.asarray((0.0, 0.0, .45)))
    state = D1MujocoTruthStateSource(plant).reset()
    controller = D1InverseDynamicsController()
    with pytest.raises(ValueError, match="normal"):
        controller.compute(D1Command(), replace(state, wheel_contact_normal=np.zeros((4, 3))))


def test_prototype_rejects_turning_commands_instead_of_claiming_skid_steering() -> None:
    state = D1MujocoTruthStateSource(D1Plant()).reset()
    with pytest.raises(ValueError, match="yaw"):
        D1InverseDynamicsController().compute(D1Command(yaw_rate_rps=.2), state)


def test_returned_torque_never_exceeds_actuator_bounds_even_by_solver_tolerance() -> None:
    from scipy.spatial.transform import Rotation

    plant = D1Plant()
    xyzw = Rotation.from_euler("xyz", (-.0026730616913584783, -.006014092412498342, 0)).as_quat()
    plant.reset(base_quaternion=xyzw[[3, 0, 1, 2]])
    source = D1MujocoTruthStateSource(plant)
    state = source.reset()
    controller = D1InverseDynamicsController()
    for _ in range(100):
        torque = controller.compute(D1Command(), state)
        assert np.all(np.abs(torque) <= JOINT_TORQUE_LIMIT)
        plant.step(torque)
        state = source.read()


def test_standing_with_small_initial_tilt_does_not_accelerate_away() -> None:
    from scipy.spatial.transform import Rotation

    plant = D1Plant()
    xyzw = Rotation.from_euler("xyz", (.00562235177634942, .002116940591419361, 0)).as_quat()
    plant.reset(base_quaternion=xyzw[[3, 0, 1, 2]])
    source = D1MujocoTruthStateSource(plant)
    state = source.reset()
    controller = D1InverseDynamicsController()
    for _ in range(200):
        plant.step(controller.compute(D1Command(), state))
        state = source.read()
        assert np.linalg.norm(plant.data.qvel[:2]) < .15
        assert abs(state.base_position[2] - .455) < .04
        assert state.undesired_ground_contacts == 0


def test_prediction_agrees_with_independent_mujoco_inverse_dynamics() -> None:
    plant = D1Plant()
    plant.reset(base_position=np.asarray((0.0, 0.0, .45)))
    plant.data.qvel[:6] = (.15, .01, -.03, .2, -.15, .1)
    plant.data.qvel[6:] = np.tile((.05, -.04, .03, 3.0), 4)
    mujoco.mj_forward(plant.model, plant.data)
    state = D1MujocoTruthStateSource(plant).reset()
    controller = D1InverseDynamicsController()
    torque = controller.compute(D1Command(), state)
    result = controller.last_result
    assert result is not None

    # Independent oracle: no access to the controller's M, h, S, or J. Disable
    # MuJoCo contact solving and dry friction to match the declared nominal model.
    plant.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    plant.model.dof_frictionloss[:] = 0.0
    mujoco.mj_forward(plant.model, plant.data)
    external = np.zeros(22)
    for wheel, body_id in enumerate(plant.wheel_body_ids_by_leg):
        axis = plant.data.xaxis[plant.joint_ids[4 * wheel + 3]]
        radial = np.asarray((0, 0, 1)) - axis * axis[2]
        point = plant.data.xpos[body_id] - .087 * radial / np.linalg.norm(radial)
        force = result.contact_force_world_n[wheel]
        mujoco.mj_applyFT(plant.model, plant.data, force, np.zeros(3), point, body_id, external)
        assert force[2] >= -1e-6
        tangent = np.cross(axis, (0, 0, 1))
        tangent /= np.linalg.norm(tangent)
        lateral = np.cross((0, 0, 1), tangent)
        assert abs(tangent @ force) + abs(lateral @ force) <= .55 * force[2] + 1e-6
    external[plant.dof_addresses] += torque
    plant.data.qacc[:] = result.generalized_acceleration
    mujoco.mj_inverse(plant.model, plant.data)
    np.testing.assert_allclose(plant.data.qfrc_inverse, external, atol=1e-4)


def test_results_are_immutable_snapshots_and_reset_discards_last_result() -> None:
    state = D1MujocoTruthStateSource(D1Plant()).reset()
    controller = D1InverseDynamicsController()
    torque = controller.compute(D1Command(), state)
    result = controller.last_result
    assert result is not None
    saved = torque.copy()
    torque[:] = 0.0
    np.testing.assert_array_equal(result.command_torque_nm, saved)
    with pytest.raises(ValueError):
        result.command_torque_nm.setflags(write=True)
    controller.compute(D1Command(forward_velocity_mps=.2), state)
    np.testing.assert_array_equal(result.command_torque_nm, saved)
    controller.reset()
    assert controller.last_result is None


def test_infeasible_contact_braking_rejects_instead_of_reusing_old_torque() -> None:
    plant = D1Plant()
    plant.reset(base_position=np.asarray((0.0, 0.0, .45)))
    source = D1MujocoTruthStateSource(plant)
    controller = D1InverseDynamicsController()
    controller.compute(D1Command(), source.reset())
    # A deliberately out-of-envelope contact impact, not a normal operating
    # scenario. Its required contact braking exceeds this QP's feasible set.
    plant.data.qvel[2] = -30.0
    mujoco.mj_forward(plant.model, plant.data)
    with pytest.raises(RuntimeError, match="rejected"):
        controller.compute(D1Command(), source.read())
    assert controller.last_result is None
