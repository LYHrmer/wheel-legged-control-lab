import pytest

from scripts.d1_keyboard_commands import HeldKeyboardCommands


def fixture():
    now = [0.0]
    return now, HeldKeyboardCommands(clock=lambda: now[0])


def hold(now, commands, keys, ticks=200):
    for _ in range(ticks):
        now[0] += 0.01
        commands.update_pressed({ord(k) for k in keys})


def test_long_hold_exceeds_old_timeout_without_repeat_events():
    now, commands = fixture()
    hold(now, commands, "W")
    assert commands(2).forward_velocity_mps == pytest.approx(0.30)
    commands.update_pressed(set())
    assert commands(2).forward_velocity_mps == 0
    hold(now, commands, "S")
    assert commands(4).forward_velocity_mps == pytest.approx(-0.20)


def test_axes_release_independently_opposites_cancel_and_focus_clears():
    now, commands = fixture()
    hold(now, commands, "WA")
    assert commands(0).forward_velocity_mps > 0.29
    commands.update_pressed({ord("A")})
    assert commands(0).forward_velocity_mps == 0
    assert commands(0).yaw_rate_rps == pytest.approx(0.25)
    commands.update_pressed({ord(k) for k in "WASD"})
    assert commands(0).forward_velocity_mps == commands(0).yaw_rate_rps == 0
    hold(now, commands, "WA")
    commands.update_pressed({ord("W"), ord("A")}, focused=False)
    assert commands(0).forward_velocity_mps == commands(0).yaw_rate_rps == 0


def test_space_and_escape_override_held_motion_and_preserve_height():
    now, commands = fixture()
    hold(now, commands, "WR")
    height = commands(0).clearance_m
    commands.update_pressed({ord("W"), ord("R"), 32})
    assert commands(0).forward_velocity_mps == 0
    assert commands(0).clearance_m == height
    commands.update_pressed({256})
    hold(now, commands, "WAF")
    assert commands.stopped
    assert commands(0).forward_velocity_mps == commands(0).yaw_rate_rps == 0
    assert commands(0).clearance_m == height


def test_height_hold_bounds_and_watchdog_cannot_revive_motion():
    now, commands = fixture()
    hold(now, commands, "WR")
    assert commands(0).clearance_m == pytest.approx(0.48)
    now[0] += 1
    assert commands(0).forward_velocity_mps == 0
    commands.update_pressed({ord("F")})
    assert commands(0).forward_velocity_mps == 0
    hold(now, commands, "F", ticks=300)
    assert commands(0).clearance_m == pytest.approx(0.43)
    before = commands(0)
    for _ in range(10):
        assert commands(0) == before  # Reading does not integrate another step.
    commands.update_pressed(set())
    now[0] += 60
    commands.update_pressed({ord("R")})
    assert commands(0).clearance_m <= 0.4325 + 1e-10


def test_clock_rollback_invalid_inputs_and_repeat_at_same_time():
    now, commands = fixture()
    commands.update_pressed({ord("W")})
    initial = commands(0)
    for _ in range(20):
        commands.update_pressed({ord("W")})
    assert commands(0) == initial
    now[0] = -1
    with pytest.raises(ValueError, match="backwards"):
        commands.update_pressed(set())
    for keys in [[87], {True}, {"W"}]:
        with pytest.raises(TypeError):
            HeldKeyboardCommands().update_pressed(keys)
    with pytest.raises(TypeError):
        HeldKeyboardCommands().update_pressed(set(), focused=1)
    for bad in [-1, float("nan"), float("inf"), True]:
        with pytest.raises(ValueError):
            HeldKeyboardCommands()(bad)
