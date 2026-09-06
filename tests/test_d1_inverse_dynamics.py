from __future__ import annotations

import mujoco
import numpy as np
import pytest

from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.inverse_dynamics import D1InverseDynamicsController
from wheel_legged_control.d1.model import JOINT_TORQUE_LIMIT, D1Plant
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource


def _slope_plant() -> D1Plant:
    plant = D1Plant(arena="course")
    ramp = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_ramp_up")
    rotation = plant.data.geom_xmat[ramp].reshape(3, 3)
    position = plant.data.geom_xpos[ramp] + (.04 + .45) * rotation[:, 2]
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    plant.reset(base_position=position, base_quaternion=quaternion)
    return plant


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


@pytest.mark.parametrize("on_slope", (False, True))
def test_rolling_acceleration_matches_independent_moving_point_difference(on_slope: bool) -> None:
    plant = _slope_plant() if on_slope else D1Plant()
    if not on_slope:
        plant.reset(base_position=np.asarray((0.0, 0.0, .45)))
    plant.data.qvel[:6] = (.15, .01, -.03, .2, -.15, .1)
    plant.data.qvel[6:] = np.tile((.05, -.04, .03, 3.0), 4)
    mujoco.mj_forward(plant.model, plant.data)
    state = D1MujocoTruthStateSource(plant).reset()
    controller = D1InverseDynamicsController()

    controller.compute(D1Command(
        base_height_m=float(state.base_position[2]), pitch_rad=float(state.base_rpy[1]),
    ), state)
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
            normal = state.wheel_contact_normal[wheel]
            normal = normal / np.linalg.norm(normal)
            radial = normal - axis * (axis @ normal)
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


@pytest.mark.parametrize("direction", (-1., 1.))
def test_low_speed_turn_changes_heading_and_then_brakes(direction: float) -> None:
    plant = D1Plant()
    source = D1MujocoTruthStateSource(plant)
    state = source.reset()
    controller = D1InverseDynamicsController(warm_start=True)
    for step in range(600):
        turning = 100 <= step < 400
        command = D1Command(
            forward_velocity_mps=.3 if turning else 0.,
            yaw_rate_rps=direction * .2 if turning else 0.,
        )
        plant.step(controller.compute(command, state))
        state = source.read()
        assert state.undesired_ground_contacts == 0
        assert abs(state.base_position[2] - .455) < .04
        assert np.max(np.abs(state.base_rpy[:2])) < np.deg2rad(5.)
    assert .4 < direction * state.base_rpy[2] < .8
    assert abs(state.base_angular_velocity_world[2]) < .02
    assert np.linalg.norm(plant.data.qvel[:2]) < .02


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


@pytest.mark.parametrize("warm_start", (False, True))
def test_prediction_agrees_with_independent_mujoco_inverse_dynamics(warm_start: bool) -> None:
    plant = D1Plant()
    plant.reset(base_position=np.asarray((0.0, 0.0, .45)))
    plant.data.qvel[:6] = (.15, .01, -.03, .2, -.15, .1)
    plant.data.qvel[6:] = np.tile((.05, -.04, .03, 3.0), 4)
    mujoco.mj_forward(plant.model, plant.data)
    state = D1MujocoTruthStateSource(plant).reset()
    controller = D1InverseDynamicsController(warm_start=warm_start)
    controller.compute(D1Command(forward_velocity_mps=.2), state)
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


@pytest.mark.parametrize("warm_start", (False, True))
def test_results_are_immutable_snapshots_and_reset_discards_last_result(warm_start: bool) -> None:
    state = D1MujocoTruthStateSource(D1Plant()).reset()
    controller = D1InverseDynamicsController(warm_start=warm_start)
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
    np.testing.assert_allclose(controller.compute(D1Command(), state), saved, atol=1e-6)


@pytest.mark.parametrize("warm_start", (False, True))
def test_infeasible_contact_braking_rejects_instead_of_reusing_old_torque(warm_start: bool) -> None:
    plant = D1Plant()
    plant.reset(base_position=np.asarray((0.0, 0.0, .45)))
    source = D1MujocoTruthStateSource(plant)
    controller = D1InverseDynamicsController(warm_start=warm_start)
    controller.compute(D1Command(), source.reset())
    # A deliberately out-of-envelope contact impact, not a normal operating
    # scenario. Its required contact braking exceeds this QP's feasible set.
    plant.data.qvel[2] = -30.0
    mujoco.mj_forward(plant.model, plant.data)
    with pytest.raises(RuntimeError, match="rejected"):
        controller.compute(D1Command(), source.read())
    assert controller.last_result is None
    plant.reset(base_position=np.asarray((0., 0., .45)))
    state = source.reset()
    reference = D1InverseDynamicsController(warm_start=warm_start)
    np.testing.assert_allclose(
        controller.compute(D1Command(), state), reference.compute(D1Command(), state), atol=1e-6,
    )


@pytest.mark.parametrize("warm_start", (False, True))
def test_contact_changes_and_reset_do_not_reuse_an_incompatible_solution(warm_start: bool) -> None:
    plant = D1Plant()
    source = D1MujocoTruthStateSource(plant)
    controller = D1InverseDynamicsController(warm_start=warm_start)
    for height in (.45, 2., .45):
        plant.reset(base_position=np.asarray((0., 0., height)))
        state = source.reset()
        command = D1Command(base_height_m=height)
        expected = D1InverseDynamicsController(warm_start=warm_start).compute(command, state)
        np.testing.assert_allclose(controller.compute(command, state), expected, atol=2e-5)
        if height == 2.:
            np.testing.assert_allclose(controller.last_result.contact_force_world_n, 0., atol=1e-7)
    controller.reset()
    np.testing.assert_allclose(controller.compute(command, state), expected, atol=2e-5)


def test_previous_commands_do_not_change_force_allocation_for_the_same_input() -> None:
    plant = D1Plant()
    plant.reset(base_position=np.asarray((0., 0., .45)))
    state = D1MujocoTruthStateSource(plant).reset()
    command = D1Command(forward_velocity_mps=.5, base_height_m=.45)
    reference = D1InverseDynamicsController()
    reference.compute(command, state)
    expected = reference.last_result
    controller = D1InverseDynamicsController()
    for previous in (command, D1Command(base_height_m=.45), command):
        controller.compute(previous, state)

    torque = controller.compute(command, state)
    result = controller.last_result
    assert result is not None and expected is not None
    np.testing.assert_allclose(torque, expected.command_torque_nm, atol=1e-3, rtol=0.)
    np.testing.assert_allclose(
        result.contact_force_world_n, expected.contact_force_world_n, atol=.05, rtol=0.,
    )
    np.testing.assert_allclose(
        result.generalized_acceleration, expected.generalized_acceleration, atol=.01, rtol=0.,
    )


@pytest.mark.parametrize("warm_start", (False, True))
def test_changed_torque_limits_keep_the_objective_consistent_with_the_bounds(warm_start: bool) -> None:
    plant = D1Plant()
    plant.reset(base_position=np.asarray((0., 0., .45)))
    state = D1MujocoTruthStateSource(plant).reset()
    command = D1Command(forward_velocity_mps=.5, base_height_m=.45)
    controller = D1InverseDynamicsController(warm_start=warm_start)
    controller.compute(command, state)
    controller.torque_limit_nm[:] = .6 * JOINT_TORQUE_LIMIT
    reference = D1InverseDynamicsController(
        torque_limit_nm=.6 * JOINT_TORQUE_LIMIT, warm_start=warm_start,
    )

    np.testing.assert_allclose(
        controller.compute(command, state), reference.compute(command, state),
        atol=1e-3, rtol=0.,
    )


def test_warm_contact_cache_distinguishes_which_two_wheels_are_in_contact() -> None:
    from dataclasses import replace

    plant = D1Plant()
    plant.reset(base_position=np.asarray((0., 0., .45)))
    state = D1MujocoTruthStateSource(plant).reset()
    command = D1Command(base_height_m=.45)
    controller = D1InverseDynamicsController(warm_start=True)
    for mask in ((True, False, False, True), (False, True, True, False)):
        snapshot = replace(state, wheel_contact=np.asarray(mask))
        reference = D1InverseDynamicsController(warm_start=True)
        np.testing.assert_allclose(
            controller.compute(command, snapshot), reference.compute(command, snapshot),
            atol=1e-5, rtol=0.,
        )
        np.testing.assert_allclose(
            controller.last_result.contact_force_world_n[~snapshot.wheel_contact], 0., atol=1e-7,
        )


def test_vertical_velocity_feedforward_is_separate_from_world_height() -> None:
    plant = D1Plant()
    plant.reset(base_position=np.asarray((0., 0., .45)))
    state = D1MujocoTruthStateSource(plant).reset()
    controller = D1InverseDynamicsController()
    controller.compute(D1Command(base_height_m=.45), state)
    zero_rate_acceleration = controller.last_result.generalized_acceleration[2]
    controller.compute(D1Command(base_height_m=.45, base_vertical_velocity_mps=.035), state)
    # This is a weighted task, not a hard acceleration constraint: posture tasks
    # can oppose it. Test the feedforward response, not the unconstrained PD target.
    assert controller.last_result.generalized_acceleration[2] > zero_rate_acceleration + 1e-3
    assert controller.last_result.constraint_violation_max <= 1e-4
    with pytest.raises(ValueError, match="finite"):
        controller.compute(D1Command(base_vertical_velocity_mps=float("nan")), state)
    assert controller.last_result is None


@pytest.mark.parametrize("kind", ("vmc", "lqr", "mpc"))
def test_legacy_paths_do_not_silently_drop_vertical_velocity_feedforward(kind: str) -> None:
    from wheel_legged_control.d1.controllers import D1VMCController
    from wheel_legged_control.d1.hierarchical import D1LQRVMCController, D1MPCVMCController

    plant = D1Plant()
    state = D1MujocoTruthStateSource(plant).reset()
    controller = {"vmc": D1VMCController, "lqr": D1LQRVMCController,
                  "mpc": D1MPCVMCController}[kind](plant)
    with pytest.raises(ValueError, match="vertical velocity"):
        controller.compute(D1Command(base_vertical_velocity_mps=.035), state)


def test_slope_contact_forces_close_independent_inverse_dynamics() -> None:
    plant = _slope_plant()
    state = D1MujocoTruthStateSource(plant).reset()
    assert state.wheel_ground_contacts == 4
    controller = D1InverseDynamicsController(warm_start=True)
    command = D1Command(base_height_m=plant.base_position[2], pitch_rad=-np.deg2rad(8.))
    torque = controller.compute(command, state)
    result = controller.last_result
    assert result is not None

    plant.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    plant.model.dof_frictionloss[:] = 0.
    mujoco.mj_forward(plant.model, plant.data)
    external = np.zeros(22)
    for wheel, body in enumerate(plant.wheel_body_ids_by_leg):
        n = state.wheel_contact_normal[wheel]
        n = n / np.linalg.norm(n)
        axis = plant.data.xaxis[plant.joint_ids[4 * wheel + 3]]
        radial = n - axis * (axis @ n)
        point = plant.data.xpos[body] - .087 * radial / np.linalg.norm(radial)
        force = result.contact_force_world_n[wheel]
        tangent = np.cross(axis, n)
        tangent /= np.linalg.norm(tangent)
        assert n @ force >= -1e-6
        assert abs(tangent @ force) + abs(np.cross(n, tangent) @ force) <= .55 * (n @ force) + 1e-6
        mujoco.mj_applyFT(plant.model, plant.data, force, np.zeros(3), point, body, external)
    external[plant.dof_addresses] += torque
    plant.data.qacc[:] = result.generalized_acceleration
    mujoco.mj_inverse(plant.model, plant.data)
    np.testing.assert_allclose(plant.data.qfrc_inverse, external, atol=1e-4)
