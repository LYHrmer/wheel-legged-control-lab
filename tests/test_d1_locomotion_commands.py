import copy
import json
import math
import random
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.locomotion_commands import (
    D1CommandLimits,
    D1CommandSchedule,
    D1CommandSegment,
    random_command_schedule,
    schedule_from_dict,
    schedule_to_dict,
    validation_command_schedule,
)


def _vector(command):
    return np.asarray((command.forward_velocity_mps, command.yaw_rate_rps, command.clearance_m))


def test_cubic_transition_exact_boundaries_and_endpoint_derivatives():
    initial = D1MotionCommand()
    first = D1MotionCommand(0.4, 0.2, 0.48)
    second = D1MotionCommand(-0.2, -0.2, 0.43)
    schedule = D1CommandSchedule(
        (
            D1CommandSegment(2.0, first, 0.4),
            D1CommandSegment(1.0, second, 0.2),
        )
    )
    assert schedule.duration_s == 3.0
    assert schedule.cmd_at(0.0) == initial
    np.testing.assert_allclose(
        _vector(schedule.cmd_at(0.2)), (_vector(initial) + _vector(first)) / 2
    )
    assert schedule.cmd_at(0.4) is first
    assert schedule.cmd_at(2.0) is first
    np.testing.assert_allclose(
        _vector(schedule.cmd_at(2.1)), (_vector(first) + _vector(second)) / 2, atol=1e-15
    )
    assert schedule.cmd_at(2.2) is second
    assert schedule.cmd_at(3.0) is second
    assert schedule.cmd_at(60.0) is second
    epsilon = 1e-7
    for boundary in (0.0, 0.4, 2.0, 2.2):
        right_slope = (
            _vector(schedule.cmd_at(boundary + epsilon)) - _vector(schedule.cmd_at(boundary))
        ) / epsilon
        assert np.max(np.abs(right_slope)) < 1e-4


def test_zero_transition_is_explicit_and_boundaries_are_right_open():
    a, b = D1MotionCommand(0.1), D1MotionCommand(-0.1)
    schedule = D1CommandSchedule((D1CommandSegment(1.0, a, 0.0), D1CommandSegment(2.0, b, 0.0)))
    assert schedule.cmd_at(0.0) is a
    assert schedule.cmd_at(np.nextafter(1.0, 0.0)) is a
    assert schedule.cmd_at(1.0) is b
    assert schedule.cmd_at(3.0) is b


@pytest.mark.parametrize("time", (-0.1, np.nan, np.inf, -np.inf, True, np.bool_(False), "1", None))
def test_command_lookup_rejects_invalid_time(time):
    with pytest.raises(ValueError):
        validation_command_schedule("development", 12.0).cmd_at(time)


@pytest.mark.parametrize(
    "duration,transition", ((0, 0), (-1, 0), (1, -0.1), (1, 1.01), (np.inf, 0), (1, np.nan))
)
def test_invalid_segment(duration, transition):
    with pytest.raises(ValueError):
        D1CommandSegment(duration, D1MotionCommand(), transition)


def test_schedule_copies_input_list_and_all_command_objects_are_immutable():
    segments = [D1CommandSegment(1.0, D1MotionCommand(0.1))]
    schedule = D1CommandSchedule(segments)
    segments.clear()
    assert len(schedule.segments) == 1
    for obj, attr, value in (
        (schedule, "label", "other"),
        (schedule.segments[0], "duration_s", 2.0),
        (schedule.segments[0].command, "clearance_m", 0.5),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, attr, value)
    with pytest.raises(ValueError):
        D1CommandSchedule(())
    with pytest.raises(ValueError):
        D1CommandSchedule((D1CommandSegment(1e308, D1MotionCommand()),) * 2)
    with pytest.raises(TypeError):
        D1CommandSegment(1, {})


def test_random_construction_uses_only_local_rng_and_lookup_is_order_independent(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("global RNG or lookup-time RNG was used")

    for name in ("uniform", "choice", "seed", "random"):
        monkeypatch.setattr(np.random, name, forbidden)
    monkeypatch.setattr(random, "random", forbidden)
    first = random_command_schedule(73, 20.0)
    assert first == random_command_schedule(73, 20.0)
    assert first != random_command_schedule(74, 20.0)
    record = schedule_to_dict(first)
    times = (0.0, 3.75, 10.0, 19.9, 20.0, 21.0)
    expected = {t: first.cmd_at(t) for t in times}
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    for t in reversed(times):
        assert first.cmd_at(t) == expected[t]
        assert first.cmd_at(t) == expected[t]
    assert schedule_to_dict(first) == record


@pytest.mark.parametrize("duration", (0.1, 12.0, 16.0, 20.0, 60.0))
def test_random_limits_duration_and_command_coverage(duration):
    limits = D1CommandLimits(
        forward_velocity_mps=0.2,
        reverse_velocity_mps=0.1,
        yaw_rate_rps=0.15,
        clearance_min_m=0.445,
        clearance_max_m=0.465,
    )
    schedule = random_command_schedule(np.int64(120), duration, limits=limits)
    assert schedule.duration_s == pytest.approx(duration, abs=1e-13)
    targets = [_vector(s.command) for s in schedule.segments]
    assert any(v[0] > 0 for v in targets) and any(v[0] < 0 for v in targets)
    assert any(v[1] > 0 for v in targets) and any(v[1] < 0 for v in targets)
    assert any(v[2] < 0.455 for v in targets) and any(v[2] > 0.455 for v in targets)
    assert any(v[0] == v[1] == 0.0 for v in targets)
    for time in np.linspace(0.0, duration, 101):
        command = schedule.cmd_at(float(time))
        assert (
            -limits.reverse_velocity_mps
            <= command.forward_velocity_mps
            <= limits.forward_velocity_mps
        )
        assert abs(command.yaw_rate_rps) <= limits.yaw_rate_rps
        assert limits.clearance_min_m <= command.clearance_m <= limits.clearance_max_m
    assert schedule.cmd_at(duration) == D1MotionCommand()


@pytest.mark.parametrize(
    "kwargs",
    (
        {"forward_velocity_mps": 0},
        {"reverse_velocity_mps": 1.01},
        {"yaw_rate_rps": -1},
        {"clearance_min_m": 0.455},
        {"clearance_max_m": 0.455},
        {"transition_s": 0},
        {"transition_s": np.nan},
        {"nominal_clearance_m": True},
    ),
)
def test_invalid_random_limits(kwargs):
    with pytest.raises(ValueError):
        D1CommandLimits(**kwargs)


def test_tight_but_valid_clearance_limits_are_supported():
    schedule = random_command_schedule(
        1,
        12.0,
        limits=D1CommandLimits(
            clearance_min_m=0.4549,
            clearance_max_m=0.4551,
        ),
    )
    assert all(0.4549 <= s.command.clearance_m <= 0.4551 for s in schedule.segments)


@pytest.mark.parametrize("duration", (0.0, -1.0, np.inf, np.nan, True))
def test_invalid_factory_duration(duration):
    with pytest.raises(ValueError):
        random_command_schedule(1, duration)
    with pytest.raises(ValueError):
        validation_command_schedule("development", duration)


@pytest.mark.parametrize("seed", (-1, None, True, 1.5))
def test_explicit_integral_seed_is_required(seed):
    with pytest.raises(ValueError):
        random_command_schedule(seed, 12.0)


def test_fixed_splits_are_different_and_duration_never_rescales_speed():
    development = validation_command_schedule("development", 60)
    holdout = validation_command_schedule("holdout", 60)
    assert development.segments != holdout.segments
    for split in ("development", "holdout"):
        reference = validation_command_schedule(split, 60.0)
        assert reference.duration_s == 60.0
        for duration in (12.0, 20.0, 45.0):
            shorter = validation_command_schedule(split, duration)
            assert shorter.duration_s == pytest.approx(duration, abs=1e-13)
            assert [s.command for s in shorter.segments] == [s.command for s in reference.segments]
            assert shorter.label != reference.label
    with pytest.raises(ValueError):
        validation_command_schedule("train", 60.0)


def test_json_roundtrip_preserves_every_lookup_without_changing_input():
    schedule = random_command_schedule(937, 20.0)
    record = json.loads(json.dumps(schedule_to_dict(schedule), allow_nan=False))
    original = copy.deepcopy(record)
    restored = schedule_from_dict(record)
    assert record == original
    assert restored == schedule
    for time in np.linspace(0.0, 21.0, 311):
        assert restored.cmd_at(float(time)) == schedule.cmd_at(float(time))
    # A custom NumPy scalar command must also produce ordinary JSON numbers.
    custom = D1CommandSchedule((D1CommandSegment(1.0, D1MotionCommand(np.float32(0.1))),))
    json.dumps(schedule_to_dict(custom), allow_nan=False)


def test_corrupt_replay_record_is_rejected():
    record = schedule_to_dict(validation_command_schedule("development", 12))
    malformed = []
    for field, value in (("schema", "v0"), ("duration_s", 13.0), ("segments", []), ("extra", 1)):
        changed = copy.deepcopy(record)
        changed[field] = value
        malformed.append(changed)
    missing_command = copy.deepcopy(record)
    del missing_command["segments"][0]["command"]["yaw_rate_rps"]
    malformed.append(missing_command)
    for changed in malformed:
        with pytest.raises(ValueError):
            schedule_from_dict(changed)


@pytest.mark.parametrize("split", ("development", "holdout"))
@pytest.mark.parametrize("duration", (12.0, 20.0, 60.0))
def test_fixed_command_nominal_route_stays_inside_map_not_a_tracking_guarantee(split, duration):
    schedule = validation_command_schedule(split, duration)
    x, y, yaw = -3.8, 0.0, 0.0
    dt = 0.02
    # Nominal unicycle integration ignores dynamics, slip and estimator error.
    # The actual env must separately terminate true map exits and log exposure.
    for time in np.arange(0, duration, dt):
        command = schedule.cmd_at(float(time + dt / 2))
        x += command.forward_velocity_mps * math.cos(yaw) * dt
        y += command.forward_velocity_mps * math.sin(yaw) * dt
        yaw += command.yaw_rate_rps * dt
        assert -6.0 < x < 6.0
        assert -3.0 < y < 3.0
