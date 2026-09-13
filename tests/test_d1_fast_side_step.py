"""Same-physics fast stepping and failure contracts for the Opus implementation."""
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from scripts.d1_fast_side_step import (
    FAST_PROFILE,
    SMOOTH_PEAK_ACCEL,
    FastSideStepController,
    G,
    _dense_inertia,
)
from scripts.d1_side_step import _edges
from wheel_legged_control.d1.interactive import D1InteractiveSimulation
from wheel_legged_control.d1.model import JOINT_VELOCITY_LIMIT, D1Plant


def snapshot(plant):
    arrays = [plant.data.qpos, plant.data.qvel, plant.data.ctrl, plant.data.xfrc_applied,
              plant.data.qfrc_applied, plant.data.qacc_warmstart, plant.model.body_mass,
              plant.model.body_inertia, plant.model.geom_friction,
              plant.measurement_data.qpos, plant.measurement_data.qvel,
              plant.measurement_data.xpos]
    return [x.copy() for x in arrays] + [np.array([plant.data.time, plant.control_dt,
                                                  plant.model.opt.timestep, plant.physics_steps])]


def unchanged(before, plant):
    for old, new in zip(before, snapshot(plant)):
        np.testing.assert_array_equal(old, new)


def tick(controller):
    plant = controller.plant
    before = snapshot(plant)
    torque = controller.compute()
    unchanged(before, plant)
    assert torque.shape == (16,) and np.isfinite(torque).all()
    assert np.all(np.abs(torque) <= plant.actuator_torque_limit_nm)
    old_time = plant.data.time
    plant.step(torque)
    assert plant.data.time-old_time == pytest.approx(.01)
    assert np.max(np.abs(plant.base_rpy[:2])) < .30
    assert plant.base_position[2] > .30
    assert plant.undesired_ground_contacts == 0
    assert np.all(np.abs(plant.joint_velocity) <= JOINT_VELOCITY_LIMIT)


def settled(yaw=0., course=False):
    if course:
        simulation = D1InteractiveSimulation(baseline="lqr", arena="course", state_mode="oracle")
        simulation.reset("start")
        for _ in range(200):
            simulation.step()
        plant = simulation.plant
    else:
        plant = D1Plant(sampling_mode="synchronized")
        plant.reset(base_quaternion=np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)]))
    before = snapshot(plant)
    controller = FastSideStepController(plant)
    unchanged(before, plant)
    if not course:
        for _ in range(200):
            tick(controller)
    return controller


@pytest.mark.parametrize("direction,yaw,course", [
    (1, 0., False), (-1, 0., False), (1, np.pi/2, False),
    (1, 0., True), (-1, 0., True),
])
def test_real_step_under_twelve_seconds(direction, yaw, course):
    controller = settled(yaw, course)
    plant = controller.plant
    initial, initial_yaw = plant.base_position.copy(), float(plant.base_rpy[2])
    before = snapshot(plant)
    assert controller.start(direction)
    unchanged(before, plant)
    assert not controller.start(-direction)
    started, lifted = float(plant.data.time), np.zeros(4, dtype=bool)
    for _ in range(1600):
        tick(controller)
        lowest, _ = controller._geometry(plant.measurement_data)
        lifted |= lowest[:, 2] > .012
        if controller.status["done"]:
            break
    assert controller.status["success"], controller.status
    assert plant.data.time-started < 12.
    assert lifted.all() and controller._contacts().all()
    assert np.linalg.norm(plant.base_origin_velocity()) < .04
    for _ in range(100):
        tick(controller)
    delta = plant.base_position-initial
    left = np.array([-np.sin(initial_yaw), np.cos(initial_yaw)])
    forward = np.array([np.cos(initial_yaw), np.sin(initial_yaw)])
    assert direction*(delta[:2]@left) > .02
    assert abs(delta[:2]@left-direction*.03) < .012
    assert abs(delta[:2]@forward) < .03
    error = plant.base_rpy[2]-initial_yaw
    assert abs(np.arctan2(np.sin(error), np.cos(error))) < .12
    assert controller.status["torque_source"] == "d1_fast_side_step"
    assert controller.status["state_source"] == "simulator_truth"


@pytest.mark.parametrize("phase", ["shift", "swing"])
def test_cancel_lands_before_handoff(phase):
    controller = settled()
    assert controller.start(1)
    for _ in range(400):
        tick(controller)
        if controller.phase == phase:
            break
    assert controller.phase == phase
    if phase == "swing":
        assert not controller._contacts()[controller.leg]
    before = snapshot(controller.plant)
    controller.cancel()
    unchanged(before, controller.plant)
    for _ in range(600):
        tick(controller)
        if controller.done:
            break
    assert controller.done and not controller.success
    assert controller.failure == "cancelled"
    assert controller._contacts().all()
    assert controller.leg is None
    assert np.linalg.norm(controller.plant.base_origin_velocity()) < .04


def test_missing_touchdown_cannot_claim_completion(monkeypatch):
    controller = settled()
    assert controller.start(1)
    for _ in range(400):
        tick(controller)
        if controller.phase == "swing":
            break
    assert controller.phase == "swing"
    leg, real_contacts = controller.leg, controller._contacts

    def missing():
        observed = real_contacts()
        observed[leg] = False
        return observed

    monkeypatch.setattr(controller, "_contacts", missing)
    for _ in range(450):
        tick(controller)
    assert controller.failure == "touchdown_timeout"
    assert controller.status["active"] and not controller.done


@pytest.mark.parametrize("fault", ["ik", "torque", "allocation"])
def test_completion_tick_fault_revokes_done(monkeypatch, fault):
    controller = settled()
    controller.phase, controller.phase_time = "recenter", 3.
    controller.body_from = controller.pose.copy()
    controller.body_to = controller.pose.copy()
    if fault == "allocation":
        real_solve = np.linalg.solve

        def invalid_allocation(a, b):
            return np.full_like(b, np.nan) if a.shape == (12, 12) else real_solve(a, b)

        monkeypatch.setattr(np.linalg, "solve", invalid_allocation)
    else:
        real_targets = controller._targets

        def invalid_target():
            target = real_targets()
            assert controller.done and controller.success
            if fault == "ik":
                controller.ik_error = .009
            else:
                target[0] = np.nan
            return target

        monkeypatch.setattr(controller, "_targets", invalid_target)
    before = snapshot(controller.plant)
    torque = controller.compute()
    unchanged(before, controller.plant)
    assert np.isfinite(torque).all()
    assert controller.status["active"] and not controller.done and not controller.success
    expected = {"ik": "ik_unreachable", "torque": "nonfinite_target", "allocation": "nonfinite_allocation"}
    assert controller.failure == expected[fault]


def test_zmp_inertial_sign_and_unclipped_acceleration_budget():
    controller = settled()
    controller.body_accel = np.array([.4, 0., 0.])
    before = snapshot(controller.plant)
    controller._advance()
    unchanged(before, controller.plant)
    points, _ = controller._geometry(controller.plant.measurement_data)
    com = controller._com(controller.plant.measurement_data)
    zmp = com[:2]-controller.com_height/G*controller.body_accel[:2]
    expected = min(normal@(zmp-origin) for origin, normal in _edges(points[:, :2]))
    assert controller.dynamic_margin == pytest.approx(expected)
    controller.cfg = replace(FAST_PROFILE, shift_time_bounds=(.3, .31))
    duration = controller._move_time(np.array([.12, 0., 0.]))
    assert duration > .31, "A nominal upper time must not override the acceleration budget"
    excursion = controller.com_height/G*SMOOTH_PEAK_ACCEL*.12/duration**2
    assert excursion <= controller.cfg.zmp_budget_m+1e-12


@pytest.mark.parametrize("legacy", [True, False])
def test_dense_inertia_supports_both_mujoco_bindings(monkeypatch, legacy):
    model = SimpleNamespace(nv=2)
    packed = np.array([2., .5, 3.])
    data = SimpleNamespace(qM=packed) if legacy else SimpleNamespace()
    expected = np.array([[2., .5], [.5, 3.]])

    if legacy:
        def full_m(received_model, destination, received_packed):
            assert received_model is model and received_packed is packed
            destination[:] = expected
    else:
        def full_m(received_model, received_data, destination):
            assert received_model is model and received_data is data
            destination[:] = expected

    monkeypatch.setattr(mujoco, "mj_fullM", full_m)
    np.testing.assert_array_equal(_dense_inertia(model, data), expected)
