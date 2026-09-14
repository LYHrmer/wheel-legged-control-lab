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


def tick(commands, clock, keys=(), yaw=0.0, velocity=0.0, focused=True, extra_keys=()):
    clock[0] += 0.01
    commands.set_state(yaw_rad=yaw, position_xy=(0, 0), forward_velocity_mps=velocity,
                       simulation_time_s=clock[0])
    commands.update_pressed(set(map(ord, keys)) | set(extra_keys), focused=focused)
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


def test_shift_cycles_speed_once_per_press_and_accelerates_gradually():
    commands, clock = controller()
    first = tick(commands, clock, "W", extra_keys={340})
    assert commands.gear == 2 and first.forward_velocity_mps <= .0051
    for _ in range(130):
        command = tick(commands, clock, "W", extra_keys={340})
    assert commands.gear == 2 and command.forward_velocity_mps == pytest.approx(.40)
    tick(commands, clock, "W", extra_keys={340, 344})
    assert commands.gear == 2
    tick(commands, clock, "W")
    tick(commands, clock, "W", extra_keys={344})
    assert commands.gear == 3
    for _ in range(60):
        command = tick(commands, clock, "W", extra_keys={344})
    assert command.forward_velocity_mps == pytest.approx(.5)
    tick(commands, clock, "W")
    tick(commands, clock, "W", extra_keys={340})
    assert commands.gear == 1
    for _ in range(110):
        command = tick(commands, clock, "W")
    assert command.forward_velocity_mps == pytest.approx(.3)
    assert tick(commands, clock).forward_velocity_mps == 0


def test_gear_reset_focus_and_side_step_preserve_motion_cancellation():
    commands, clock = controller()
    tick(commands, clock, "S", extra_keys={340})
    for _ in range(100):
        command = tick(commands, clock, "S", extra_keys={340})
    assert command.forward_velocity_mps == pytest.approx(-.35)
    assert tick(commands, clock, "W", extra_keys={340}, focused=False).forward_velocity_mps == 0
    tick(commands, clock, "AW")
    assert commands.side_direction == 1 and commands.requested_forward_mps == 0
    commands.set_side_active(True)
    tick(commands, clock, "W", extra_keys={340})
    assert commands.gear == 3 and commands.requested_forward_mps == 0
    commands.reset(0.0, (0.0, 0.0), simulation_time_s=clock[0])
    assert commands.gear == 1
    tick(commands, clock, extra_keys={340})
    assert commands.gear == 1  # Reset cannot turn a held Shift into a fresh press.


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


def test_invalid_state_is_atomic_and_ad_requests_side_motion_without_yaw():
    commands, clock = controller()
    before = vars(commands).copy()
    with pytest.raises(ValueError):
        commands.set_state(yaw_rad=0, position_xy=(1, math.nan), simulation_time_s=0)
    assert vars(commands) == before
    for _ in range(30):
        command = tick(commands, clock, "A")
    assert command.forward_velocity_mps == 0 and command.yaw_rate_rps == 0
    assert commands.side_direction == 1 and "side step" in commands.message.lower()
    tick(commands, clock, "D")
    assert commands.side_direction == -1
    tick(commands, clock, "AD")
    assert commands.side_direction == 0
    for _ in range(300):
        command = tick(commands, clock, "Q")
        assert abs(command.yaw_rate_rps) <= 1.0


def test_side_motion_owns_drive_keys_until_landing_and_focus_loss_cancels():
    commands, clock = controller()
    tick(commands, clock, "AWQ")
    assert commands.side_direction == 1
    assert commands.requested_forward_mps == 0 and commands.goal_yaw_rad == 0
    commands.set_side_active(True)
    command = tick(commands, clock, "WQ")
    assert commands.side_direction == 0
    assert command.forward_velocity_mps == command.yaw_rate_rps == 0
    assert "Landing" in commands.message
    tick(commands, clock, "A", focused=False)
    assert commands.side_direction == 0 and commands.side_active
    commands.set_side_active(False)
    assert tick(commands, clock, "W").forward_velocity_mps > 0
    tick(commands, clock, "A")
    clock[0] += 1
    commands(clock[0])
    assert commands.side_direction == 0


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
