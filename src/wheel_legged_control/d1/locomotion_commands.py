"""Explicit, replayable motion-command schedules for the common control loop.

Randomness is consumed once when constructing a training schedule. Looking up
a command never advances a cursor or RNG. Schedule duration is physical time;
choosing a different duration retimes its segments, never rescales velocity.
Command splits here are separate from road-layout splits in locomotion_terrain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np

from .control_loop import D1MotionCommand

D1_COMMAND_SCHEDULE_SCHEMA = "d1-explicit-motion-schedule-v1"


def _finite(value: object, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True, slots=True)
class D1CommandSegment:
    """One target and its duration, including the initial smooth transition.

    Over ``transition_s``, cubic smoothstep blends the previous target into
    this one with zero endpoint slope. The remaining interval holds the target.
    A zero transition explicitly requests an instantaneous change.
    """

    duration_s: float
    command: D1MotionCommand
    transition_s: float = 0.4

    def __post_init__(self) -> None:
        for name in ("duration_s", "transition_s"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.duration_s <= 0.0 or not 0.0 <= self.transition_s <= self.duration_s:
            raise ValueError("segment duration must be positive and contain its transition")
        if not isinstance(self.command, D1MotionCommand):
            raise TypeError("segment command must be D1MotionCommand")


@dataclass(frozen=True, slots=True)
class D1CommandSchedule:
    segments: tuple[D1CommandSegment, ...]
    initial_command: D1MotionCommand = field(default_factory=D1MotionCommand)
    label: str = "custom"

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        if not segments or not all(isinstance(s, D1CommandSegment) for s in segments):
            raise ValueError("schedule requires at least one D1CommandSegment")
        object.__setattr__(self, "segments", segments)
        if not isinstance(self.initial_command, D1MotionCommand):
            raise TypeError("initial_command must be D1MotionCommand")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("schedule label must be a nonempty string")
        try:
            duration = self.duration_s
        except OverflowError as exc:
            raise ValueError("total schedule duration must be finite") from exc
        if not math.isfinite(duration):
            raise ValueError("total schedule duration must be finite")

    @property
    def duration_s(self) -> float:
        return math.fsum(segment.duration_s for segment in self.segments)

    def cmd_at(self, time_s: float) -> D1MotionCommand:
        """Pure command lookup; hold the final target at/after the final time.

        Negative/nonfinite time is invalid. Segments are right-open: an exact
        boundary begins the next segment, whose transition starts at the prior
        target. After the schedule ends there is no loop or new random command;
        the environment separately owns episode termination.
        """
        time_s = _finite(time_s, "time_s")
        if time_s < 0.0:
            raise ValueError("time_s must be non-negative")
        if time_s >= self.duration_s:
            return self.segments[-1].command
        previous, start = self.initial_command, 0.0
        for index, segment in enumerate(self.segments):
            end = math.fsum(s.duration_s for s in self.segments[: index + 1])
            if time_s < end:
                elapsed = time_s - start
                if segment.transition_s == 0.0 or elapsed >= segment.transition_s:
                    return segment.command
                if elapsed == 0.0:
                    return previous
                fraction = elapsed / segment.transition_s
                blend = fraction * fraction * (3.0 - 2.0 * fraction)
                a, b = previous, segment.command
                return D1MotionCommand(
                    a.forward_velocity_mps
                    + blend * (b.forward_velocity_mps - a.forward_velocity_mps),
                    a.yaw_rate_rps + blend * (b.yaw_rate_rps - a.yaw_rate_rps),
                    a.clearance_m + blend * (b.clearance_m - a.clearance_m),
                )
            previous, start = segment.command, end
        raise AssertionError("finite schedule time was not assigned a segment")


@dataclass(frozen=True, slots=True)
class D1CommandLimits:
    """Construction limits, not proven tracking or traversability limits."""

    forward_velocity_mps: float = 0.30
    reverse_velocity_mps: float = 0.20
    yaw_rate_rps: float = 0.25
    clearance_min_m: float = 0.43
    clearance_max_m: float = 0.48
    nominal_clearance_m: float = 0.455
    transition_s: float = 0.4

    def __post_init__(self) -> None:
        for name in (
            "forward_velocity_mps",
            "reverse_velocity_mps",
            "yaw_rate_rps",
            "clearance_min_m",
            "clearance_max_m",
            "nominal_clearance_m",
            "transition_s",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if not all(
            0.0 < value <= 1.0
            for value in (
                self.forward_velocity_mps,
                self.reverse_velocity_mps,
                self.yaw_rate_rps,
            )
        ):
            raise ValueError("positive velocity/yaw limits must not exceed 1")
        if (
            not 0.38
            <= self.clearance_min_m
            < self.nominal_clearance_m
            < self.clearance_max_m
            <= 0.53
        ):
            raise ValueError("clearance limits must bracket nominal within [.38, .53] m")
        if self.transition_s <= 0.0:
            raise ValueError("random schedule transition_s must be positive")


DEFAULT_COMMAND_LIMITS = D1CommandLimits()


def _duration(value: float) -> float:
    value = _finite(value, "duration_s")
    if value <= 0.0:
        raise ValueError("duration_s must be positive")
    return value


def _schedule(commands, durations, initial, label, transition_s=0.4):
    return D1CommandSchedule(
        tuple(
            D1CommandSegment(duration, command, min(transition_s, duration * 0.5))
            for duration, command in zip(durations, commands, strict=True)
        ),
        initial_command=initial,
        label=label,
    )


def random_command_schedule(
    seed: int,
    duration_s: float,
    *,
    limits: D1CommandLimits = DEFAULT_COMMAND_LIMITS,
) -> D1CommandSchedule:
    """Freeze a seeded stop/forward/reverse/turn/clearance training sequence.

    After initial settling, forward motion gets 40% of the requested duration.
    The other targets are shuffled and assigned random durations at construction.
    Short episodes can still stay on the spawn strip: commanded motion is not
    evidence that any terrain feature was reached. Log actual exposure in env.
    """
    duration_s = _duration(duration_s)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(limits, D1CommandLimits):
        raise TypeError("limits must be D1CommandLimits")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    neutral = D1MotionCommand(clearance_m=limits.nominal_clearance_m)
    forward = D1MotionCommand(
        float(rng.uniform(0.75, 1.0) * limits.forward_velocity_mps),
        0.0,
        limits.nominal_clearance_m,
    )
    targets = [
        D1MotionCommand(
            -float(rng.uniform(0.5, 1.0) * limits.reverse_velocity_mps),
            0.0,
            limits.nominal_clearance_m,
        ),
        D1MotionCommand(
            float(rng.uniform(0.2, 0.6) * limits.forward_velocity_mps),
            float(rng.uniform(0.4, 1.0) * limits.yaw_rate_rps),
            limits.nominal_clearance_m,
        ),
        D1MotionCommand(
            float(rng.uniform(0.2, 0.6) * limits.forward_velocity_mps),
            -float(rng.uniform(0.4, 1.0) * limits.yaw_rate_rps),
            limits.nominal_clearance_m,
        ),
        D1MotionCommand(
            clearance_m=float(
                limits.nominal_clearance_m
                - rng.uniform(0.3, 1.0) * (limits.nominal_clearance_m - limits.clearance_min_m)
            )
        ),
        D1MotionCommand(
            clearance_m=float(
                limits.nominal_clearance_m
                + rng.uniform(0.3, 1.0) * (limits.clearance_max_m - limits.nominal_clearance_m)
            )
        ),
    ]
    targets = [targets[int(i)] for i in rng.permutation(len(targets))]
    settle_s = min(1.0, duration_s * 0.08)
    forward_s = duration_s * 0.4
    remainder = duration_s - 2.0 * settle_s - forward_s
    weights = rng.uniform(0.8, 1.2, len(targets))
    middle = (remainder * weights / weights.sum()).tolist()
    durations = [settle_s, forward_s, *middle]
    durations.append(duration_s - math.fsum(durations))
    return _schedule(
        [neutral, forward, *targets, neutral],
        durations,
        neutral,
        f"commands-random-seed-{seed}-{duration_s:g}s",
        limits.transition_s,
    )


def validation_command_schedule(split: str, duration_s: float) -> D1CommandSchedule:
    """Fixed command-combination splits; independent of terrain split selection.

    Targets have the same physical velocities at 12, 20 or 60 s. Changing
    duration changes dwell/transition fractions and hence the task; evaluations
    must report it. The finite map and actual tracking remain the env's concern.
    """
    duration_s = _duration(duration_s)
    # fraction, forward velocity, yaw rate, clearance; no terrain-success labels.
    rows = {
        "development": (
            (0.06, 0.0, 0.0, 0.455),
            (0.20, 0.22, 0.0, 0.455),
            (0.12, 0.14, 0.12, 0.455),
            (0.08, 0.0, 0.0, 0.43),
            (0.10, -0.12, 0.0, 0.455),
            (0.12, 0.08, -0.16, 0.455),
            (0.12, 0.16, 0.0, 0.48),
            (0.06, 0.0, -0.10, 0.455),
            (0.06, 0.0, 0.13, 0.455),
            (0.08, 0.0, 0.0, 0.455),
        ),
        "holdout": (
            (0.06, 0.0, 0.0, 0.455),
            (0.10, -0.10, 0.0, 0.455),
            (0.22, 0.24, 0.0, 0.455),
            (0.10, 0.12, -0.14, 0.455),
            (0.08, 0.0, 0.16, 0.44),
            (0.08, 0.0, 0.0, 0.455),
            (0.12, 0.16, 0.0, 0.47),
            (0.10, 0.10, 0.10, 0.455),
            (0.08, -0.12, -0.07, 0.455),
            (0.06, 0.0, 0.0, 0.455),
        ),
    }
    if split not in rows:
        raise ValueError("command split must be development or holdout")
    commands = [D1MotionCommand(*row[1:]) for row in rows[split]]
    durations = [row[0] * duration_s for row in rows[split][:-1]]
    durations.append(duration_s - math.fsum(durations))
    return _schedule(commands, durations, D1MotionCommand(), f"commands-{split}-{duration_s:g}s")


def schedule_to_dict(schedule: D1CommandSchedule) -> dict:
    if not isinstance(schedule, D1CommandSchedule):
        raise TypeError("schedule must be D1CommandSchedule")

    def command_dict(command):
        return {
            name: float(getattr(command, name))
            for name in (
                "forward_velocity_mps",
                "yaw_rate_rps",
                "clearance_m",
            )
        }

    return {
        "schema": D1_COMMAND_SCHEDULE_SCHEMA,
        "duration_s": schedule.duration_s,
        "label": schedule.label,
        "initial_command": command_dict(schedule.initial_command),
        "segments": [
            {
                "duration_s": segment.duration_s,
                "transition_s": segment.transition_s,
                "command": command_dict(segment.command),
            }
            for segment in schedule.segments
        ],
    }


def schedule_from_dict(value: dict) -> D1CommandSchedule:
    if not isinstance(value, dict):
        raise TypeError("schedule record must be a dict")
    if set(value) != {"schema", "duration_s", "segments", "initial_command", "label"}:
        raise ValueError("schedule record has missing or unexpected fields")
    if value["schema"] != D1_COMMAND_SCHEDULE_SCHEMA:
        raise ValueError("unsupported command schedule schema")
    if not isinstance(value["segments"], (list, tuple)):
        raise TypeError("segments must be an explicit sequence")

    def command_from_dict(fields):
        if not isinstance(fields, dict) or set(fields) != {
            "forward_velocity_mps",
            "yaw_rate_rps",
            "clearance_m",
        }:
            raise ValueError("command record has missing or unexpected fields")
        return D1MotionCommand(**fields)

    segments = []
    for row in value["segments"]:
        if not isinstance(row, dict) or set(row) != {"duration_s", "transition_s", "command"}:
            raise ValueError("segment record has missing or unexpected fields")
        segments.append(
            D1CommandSegment(
                row["duration_s"],
                command_from_dict(row["command"]),
                row["transition_s"],
            )
        )
    schedule = D1CommandSchedule(
        tuple(segments), command_from_dict(value["initial_command"]), value["label"]
    )
    if schedule.duration_s != _duration(value["duration_s"]):
        raise ValueError("recorded duration differs from segment durations")
    return schedule
