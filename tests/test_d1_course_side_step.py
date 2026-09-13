"""Real course dynamics and controller ownership across side-step handoffs."""

from dataclasses import asdict
from unittest.mock import Mock

import numpy as np
import pytest

from scripts.d1_course_side_step import CourseSideStepDrive
from scripts.d1_side_step import SideStepController
from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.interactive import D1InteractiveSimulation

STOP = D1MotionCommand()


def physical_snapshot(plant):
    return [plant.data.qpos.copy(), plant.data.qvel.copy(),
            plant.data.ctrl.copy(), np.array(plant.data.time),
            plant.model.body_mass.copy(), plant.model.geom_friction.copy()]


def assert_unchanged(before, plant):
    for old, new in zip(before, physical_snapshot(plant)):
        np.testing.assert_array_equal(old, new)


def settled():
    simulation = D1InteractiveSimulation(arena="course")
    for _ in range(200):
        simulation.step()
    drive = CourseSideStepDrive(simulation)
    return simulation, drive


def forbid_reset(*args, **kwargs):
    raise AssertionError("controller handoff must not reset the physical plant")


def test_factory_only_initializes_memory_and_preserves_controller_source():
    simulation, _ = settled()
    calls = []

    class TaggedSideController(SideStepController):
        # Exercise source propagation independently of the fast gait's tuning.
        @property
        def status(self):
            return {**super().status, "torque_source": "d1_fast_side_step"}

    def factory(plant):
        calls.append(plant)
        return TaggedSideController(plant)

    before = physical_snapshot(simulation.plant)
    drive = CourseSideStepDrive(simulation, side_controller_factory=factory)
    assert calls == [simulation.plant]
    assert_unchanged(before, simulation.plant)
    _, _, info = drive.step(STOP, 0)
    assert info["torque_source"] == "legacy"
    _, _, info = drive.step(STOP, 1)
    assert drive.active and info["torque_source"] == "d1_fast_side_step"
    before = physical_snapshot(simulation.plant)
    drive.reset()
    assert_unchanged(before, simulation.plant)
    assert calls == [simulation.plant], "reset must not create another controller"


def traced_step(simulation, drive, direction, monkeypatch):
    """Check one actual physics step and exclusive torque ownership each call."""
    legacy = Mock(wraps=simulation.teleop.compute)
    side = Mock(wraps=drive.side.compute)
    integrate = Mock(wraps=simulation.plant.step)
    update = Mock(wraps=simulation.teleop.update_state)
    monkeypatch.setattr(simulation.teleop, "compute", legacy)
    monkeypatch.setattr(drive.side, "compute", side)
    monkeypatch.setattr(simulation.plant, "step", integrate)
    monkeypatch.setattr(simulation.teleop, "update_state", update)

    def step(requested=STOP, next_direction=direction):
        counts = [m.call_count for m in (legacy, side, integrate, update)]
        before = float(simulation.plant.data.time)
        status, torque, info = drive.step(requested, next_direction)
        actual = [m.call_count-old for m, old in zip((legacy, side, integrate, update), counts)]
        expected = [0, 1, 1, 1] if info["torque_source"] == "d1_side_step" else [1, 0, 1, 1]
        assert actual == expected
        assert simulation.plant.data.time == pytest.approx(before+simulation.plant.control_dt)
        np.testing.assert_array_equal(integrate.call_args.args[0], torque)
        assert np.isfinite(torque).all()
        assert np.all(np.abs(torque) <= simulation.plant.actuator_torque_limit_nm)
        assert not simulation.plant.has_fallen()
        return status, torque, info

    return step


def test_inactive_matches_legacy_and_reset_only_clears_memory():
    first = D1InteractiveSimulation(arena="course")
    second = D1InteractiveSimulation(arena="course")
    drive = CourseSideStepDrive(first)
    commands = [STOP]*10+[D1MotionCommand(0.05, 0.02, 0.455)]*10
    for command in commands:
        second.teleop.forward_velocity_mps = command.forward_velocity_mps
        second.teleop.yaw_rate_rps = command.yaw_rate_rps
        second.teleop.base_height_m = command.clearance_m
        expected = second.step()
        status, torque, info = drive.step(command, 0)
        actual_fields, expected_fields = asdict(status), asdict(expected)
        # Solver wall time is an observation, not deterministic dynamics.
        assert actual_fields.pop("allocation_solve_ms") >= 0.0
        assert expected_fields.pop("allocation_solve_ms") >= 0.0
        assert actual_fields == expected_fields
        assert info["torque_source"] == "legacy" and not drive.active
        np.testing.assert_array_equal(first.plant.data.qpos, second.plant.data.qpos)
        np.testing.assert_array_equal(first.plant.data.qvel, second.plant.data.qvel)
        np.testing.assert_array_equal(torque, second.teleop._last_torque)
    first.reset("ramp")
    drive._blocked_until_release = True
    before = physical_snapshot(first.plant)
    drive.reset()
    assert_unchanged(before, first.plant)
    assert not drive.active and not drive._blocked_until_release
    assert drive.side.phase == "idle"
    np.testing.assert_array_equal(drive.side.pose, first.plant.measurement_data.qpos[:3])


def test_airborne_release_lands_then_returns_to_legacy(monkeypatch):
    simulation, drive = settled()
    monkeypatch.setattr(simulation, "reset", forbid_reset)
    monkeypatch.setattr(simulation.plant, "reset", forbid_reset)
    step = traced_step(simulation, drive, 1, monkeypatch)
    for _ in range(1500):
        status, _, info = step()
        if info["phase"] == "swing":
            break
    assert info["phase"] == "swing" and drive.active
    assert not drive.side._contacts()[drive.side.leg]
    assert status.state_estimation_mode == "simulator_truth"
    assert status.allocation_force_error_norm_n is None
    assert status.contact_allocation == "not_applicable"
    for _ in range(800):
        _, _, info = step(next_direction=0)
        assert info["torque_source"] == "d1_side_step"
        if info["done"]:
            break
        assert drive.active, "a failure string alone does not hand off torque"
    assert info["done"] and info["failure"] == "cancelled"
    assert not drive.active and drive.side._contacts().all()
    assert np.linalg.norm(simulation.plant.base_origin_velocity()) < 0.04
    assert simulation.teleop._stuck_steps == 0
    assert simulation.teleop._boost_steps_remaining == 0
    assert simulation.teleop.controller.control_memory.distance_m == 0.0
    np.testing.assert_array_equal(simulation.teleop._last_torque, np.zeros(16))
    for _ in range(100):
        _, _, info = step(next_direction=0)
        assert info["torque_source"] == "legacy"


def test_complete_held_cycle_restarts_without_reset(monkeypatch):
    simulation, drive = settled()
    initial = simulation.plant.base_position.copy()
    monkeypatch.setattr(simulation, "reset", forbid_reset)
    monkeypatch.setattr(simulation.plant, "reset", forbid_reset)
    step = traced_step(simulation, drive, -1, monkeypatch)
    for _ in range(6000):
        _, _, info = step()
        if info["done"]:
            break
    assert info["done"] and info["success"] and info["failure"] is None
    assert info["torque_source"] == "d1_side_step"
    assert not drive.active and drive.side._contacts().all()
    assert simulation.plant.base_position[1]-initial[1] < -0.02
    _, _, info = step()
    assert drive.active and info["phase"] == "shift"
    assert info["torque_source"] == "d1_side_step" and not info["done"]


def test_reversal_failure_requires_release_and_pending_jump_cannot_start(monkeypatch):
    simulation, drive = settled()
    step = traced_step(simulation, drive, 1, monkeypatch)
    _, _, info = step()
    assert drive.active
    for _ in range(200):
        _, _, info = step(next_direction=-1)
        if info["done"]:
            break
    assert info["done"] and info["failure"] == "cancelled"
    assert info["blocked_until_release"]
    for _ in range(20):
        _, _, info = step(next_direction=-1)
        assert info["torque_source"] == "legacy" and not drive.active
    step(next_direction=0)
    simulation.teleop.handle_key(32)
    status, _, info = step()
    assert info["torque_source"] == "legacy" and status.jump_phase == "crouch"
    for _ in range(5):
        status, _, info = step()
        assert info["torque_source"] == "legacy" and not drive.active


def test_jump_during_side_lands_without_deferred_jump(monkeypatch):
    simulation, drive = settled()
    step = traced_step(simulation, drive, 1, monkeypatch)
    step()
    simulation.teleop.handle_key(32)
    _, _, info = step()
    assert info["jump_request_discarded"]
    assert not simulation.teleop._jump_requested
    assert drive.active and info["failure"] == "cancelled"
    for _ in range(200):
        status, _, info = step()
        if info["done"]:
            break
    assert info["done"] and info["blocked_until_release"]
    for _ in range(20):
        status, _, info = step()
        assert info["torque_source"] == "legacy" and status.jump_phase == "ready"


@pytest.mark.parametrize("direction", [True, 1.0, 2, np.nan, "1"])
def test_invalid_direction_has_no_physics_or_command_side_effect(direction):
    simulation = D1InteractiveSimulation(arena="course")
    drive = CourseSideStepDrive(simulation)
    before = physical_snapshot(simulation.plant)
    with pytest.raises(ValueError):
        drive.step(D1MotionCommand(0.2), direction)
    assert_unchanged(before, simulation.plant)
    assert simulation.teleop.forward_velocity_mps == 0.0
