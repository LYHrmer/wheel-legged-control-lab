import math
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.d1_course_controls import CourseKeyboardCommands, wrap_angle
from scripts.run_d1_course_drive import apply_course_events
from wheel_legged_control.d1.interactive import D1InteractiveSimulation


def controller(yaw=0.0):
    clock = [0.0]
    commands = CourseKeyboardCommands(clock=lambda: clock[0])
    commands.reset(yaw, (0, 0))
    return commands, clock


def tick(commands, clock, keys=(), yaw=0.0, velocity=0.0, focused=True):
    clock[0] += 0.01
    commands.set_state(yaw_rad=yaw, position_xy=(0, 0), forward_velocity_mps=velocity,
                       simulation_time_s=clock[0])
    commands.update_pressed(set(map(ord, keys)), focused=focused)
    return commands(clock[0])


def test_heading_keys_retain_angle_and_follow_short_arc_across_pi():
    commands, clock = controller(math.pi-0.02)
    for _ in range(20):
        command = tick(commands, clock, "Q", yaw=math.pi-0.02)
    assert command.yaw_rate_rps > 0
    goal = commands.goal_yaw_rad
    assert goal < -3.0
    assert tick(commands, clock, yaw=math.pi-0.02).yaw_rate_rps > 0
    assert commands.goal_yaw_rad == goal
    assert tick(commands, clock, yaw=goal).yaw_rate_rps == pytest.approx(0)


def test_space_does_not_stop_and_height_uses_tg_not_r():
    commands, clock = controller()
    for _ in range(80):
        command = tick(commands, clock, "WT ")
    assert commands.requested_forward_mps == pytest.approx(0.3)
    assert command.forward_velocity_mps > 0 and command.clearance_m > 0.47
    assert tick(commands, clock, "R").forward_velocity_mps == 0
    assert commands.requested_forward_mps == 0


def test_focus_stop_watchdog_and_escape_cancel_goals():
    commands, clock = controller()
    for _ in range(40):
        tick(commands, clock, "WQ")
    assert tick(commands, clock, "WQ", focused=False).yaw_rate_rps == 0
    assert commands.goal_yaw_rad == 0
    tick(commands, clock, "WQ")
    clock[0] += 1
    assert commands(clock[0]).forward_velocity_mps == 0
    assert commands(clock[0]).yaw_rate_rps == 0
    commands.update_pressed({256})
    commands.reset(0, (0, 0))
    assert commands.stopped
    assert tick(commands, clock, "WQ").forward_velocity_mps == 0


def test_invalid_state_is_atomic_and_ad_has_explicit_feedback():
    commands, clock = controller()
    before = vars(commands).copy()
    with pytest.raises(ValueError):
        commands.set_state(yaw_rad=0, position_xy=(1, math.nan), simulation_time_s=0)
    assert vars(commands) == before
    for _ in range(30):
        command = tick(commands, clock, "A")
    assert command.forward_velocity_mps == 0 and command.yaw_rate_rps == 0
    assert "not available" in commands.message
    for _ in range(300):
        command = tick(commands, clock, "Q")
        assert abs(command.yaw_rate_rps) <= 1.0


def test_r_resets_actually_overturned_robot_and_clears_commands():
    simulation = D1InteractiveSimulation(arena="flat")
    commands, clock = controller()
    for _ in range(20):
        tick(commands, clock, "WQ")
    simulation.plant.reset(base_position=np.array([1., 1., .3]),
                           base_quaternion=np.array([0., 1., 0., 0.]))
    simulation.teleop.update_state()
    assert simulation.teleop._state.has_fallen()
    viewer = SimpleNamespace(events=[{"type": "key", "key": ord("R"), "action": 1}],
                             commands=commands, reset_camera=lambda: None)
    resets = []
    assert apply_course_events(simulation, viewer, 0,
                               lambda zone, index: resets.append((zone, index)), "start") == 1
    assert resets == [("start", 0)]
    assert not simulation.teleop._state.has_fallen()
    assert commands(0).forward_velocity_mps == commands(0).yaw_rate_rps == 0
    for _ in range(200):
        simulation.step()
    assert not simulation.teleop._state.has_fallen()
    assert abs(wrap_angle(float(simulation.teleop._state.base_rpy[2]))) < 0.05
