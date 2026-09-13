"""Physical stepping contracts; controllers never receive permission to teleport."""
import numpy as np
import pytest

from scripts.d1_side_step import SideStepController
from wheel_legged_control.d1.model import D1Plant


def snapshot(plant):
    data = plant.data
    return [data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(),
            data.xfrc_applied.copy(), data.qfrc_applied.copy(),
            data.qacc_warmstart.copy(), np.array(data.time),
            plant.model.body_mass.copy(), plant.model.geom_friction.copy()]


def unchanged(before, plant):
    for previous, current in zip(before, snapshot(plant)):
        np.testing.assert_array_equal(previous, current)


def tick(controller):
    plant = controller.plant
    before = snapshot(plant)
    torque = controller.compute()
    unchanged(before, plant)
    assert torque.shape == (16,) and np.isfinite(torque).all()
    assert np.all(np.abs(torque) <= plant.actuator_torque_limit_nm)
    plant.step(torque)
    assert np.max(np.abs(plant.base_rpy[:2])) < .30
    assert plant.base_position[2] > .30
    assert plant.undesired_ground_contacts == 0


def settled(yaw=0.):
    plant = D1Plant(control_dt=.01, arena="flat", sampling_mode="synchronized")
    plant.reset(base_quaternion=np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)]))
    before = snapshot(plant)
    controller = SideStepController(plant)
    unchanged(before, plant)
    for _ in range(200):
        tick(controller)
    assert controller._contacts().all()
    return controller


@pytest.mark.parametrize("direction,yaw", [(1, 0.), (-1, 0.), (1, np.pi/2)])
def test_complete_physical_step(direction, yaw):
    controller = settled(yaw)
    plant = controller.plant
    initial = plant.base_position.copy()
    initial_yaw = float(plant.base_rpy[2])
    before = snapshot(plant)
    assert controller.start(direction)
    unchanged(before, plant)
    assert not controller.start(-direction), "A second request must not replace an active step"
    lifted = np.zeros(4, dtype=bool)
    phases = set()
    for _ in range(6000):
        tick(controller)
        phase = controller.status["phase"]
        phases.add(phase)
        contacts = controller._contacts()
        lowest, _ = controller._geometry(plant.measurement_data)
        lifted |= lowest[:, 2] > .012
        if phase == "swing":
            assert contacts.sum() == 3
            assert not contacts[controller.leg]
        if controller.status["done"]:
            break
    assert controller.status["done"] and controller.status["success"], controller.status
    assert lifted.all(), "Each foot must actually leave the ground"
    assert {"shift", "unload", "lift", "swing", "lower", "load", "recenter", "done"} <= phases
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
    assert controller._contacts().all()
    assert np.linalg.norm(plant.base_origin_velocity()) < .04
    assert controller.status["active_leg"] is None
    assert controller.status["torque_source"] == "d1_side_step"


@pytest.mark.parametrize("phase", ["shift", "swing"])
def test_cancel_lands_before_done(phase):
    controller = settled()
    assert controller.start(1)
    for _ in range(1200):
        tick(controller)
        if controller.phase == phase:
            break
    assert controller.phase == phase
    if phase == "swing":
        assert not controller._contacts()[controller.leg]
        assert controller._geometry(controller.plant.measurement_data)[0][controller.leg, 2] > .012
    before = snapshot(controller.plant)
    controller.cancel()
    unchanged(before, controller.plant)
    assert not controller.status["done"]
    for _ in range(600):
        tick(controller)
        if controller.status["done"]:
            break
    assert controller.status["done"] and not controller.status["success"]
    assert controller.status["failure"] == "cancelled"
    assert controller._contacts().all()
    assert np.linalg.norm(controller.plant.base_origin_velocity()) < .04
    assert controller.status["active_leg"] is None


def test_missing_touchdown_never_claims_safe_handoff(monkeypatch):
    controller = settled()
    assert controller.start(1)
    for _ in range(1200):
        tick(controller)
        if controller.phase == "swing":
            break
    swing_leg = controller.leg
    real_contacts = controller._contacts

    def missing():
        observed = real_contacts()
        observed[swing_leg] = False
        return observed

    monkeypatch.setattr(controller, "_contacts", missing)
    for _ in range(800):
        tick(controller)
    assert controller.status["failure"] == "touchdown_timeout"
    assert not controller.status["done"] and not controller.status["success"]
    assert controller.status["active_leg"] is not None


def test_invalid_requests_and_reset_do_not_touch_physics():
    controller = settled()
    plant = controller.plant
    for direction in [0, 2, True, 1., np.nan]:
        with pytest.raises(ValueError):
            controller.start(direction)
    for distance in [0., -.03, .05, np.inf, np.nan]:
        with pytest.raises(ValueError):
            controller.start(1, distance_m=distance)
    before = snapshot(plant)
    controller.reset()
    unchanged(before, plant)
    assert controller.status["phase"] == "idle"
    # Fault injection belongs to the test. The controller must not propagate NaNs.
    plant.data.qvel[0] = np.nan
    before = snapshot(plant)
    torque = controller.compute()
    unchanged(before, plant)
    assert np.isfinite(torque).all()
    assert controller.status["failure"] == "nonfinite_state"
    assert not controller.status["done"]


@pytest.mark.parametrize("fault", ["ik", "torque"])
def test_completion_tick_fault_revokes_handoff(monkeypatch, fault):
    controller = settled()
    # Reach the completion decision using valid physical state. Inject only
    # the subsequent target calculation failure, not a fictitious robot pose.
    controller.phase = "recenter"
    controller.phase_time = 3.
    controller.body_from = controller.pose.copy()
    controller.body_to = controller.pose.copy()
    real_targets = controller._targets

    def faulty_targets():
        target = real_targets()
        assert controller.done and controller.success
        if fault == "ik":
            controller.ik_error = .009
        else:
            target[0] = np.nan
        return target

    monkeypatch.setattr(controller, "_targets", faulty_targets)
    before = snapshot(controller.plant)
    torque = controller.compute()
    unchanged(before, controller.plant)
    assert np.isfinite(torque).all()
    assert controller.status["phase"] == "abort_hold"
    assert controller.status["active"]
    assert not controller.status["done"] and not controller.status["success"]
    assert controller.status["failure"] == ("ik_unreachable" if fault == "ik" else "nonfinite_torque")
