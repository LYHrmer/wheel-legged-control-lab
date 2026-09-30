"""Pure, preregistered requests for the 08-R authority and RL pilot.

This module constructs no model and imports no engine.  It publishes the raw
operator/training target for each control tick; ``FullDriveCommandServo`` is
the separate, recorded applied request.  A training runner owns its seeded
choice of level/terrain/speed and promotion history, and must save those
choices before constructing a physical episode.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from full_drive_command_08 import FullDriveCommand, QualifiedCommandCaps
from full_drive_servo_08 import CONTROL_DT_S, FullDriveCommandServo

SCHEDULE_SCHEMA = "d1-course-rl16-preregistered-request-schedule-v1"
FLAT_SPAWN_M = (-8.0, -4.7, 0.455)
COURSE_SPAWNS_M = {
    "flat": FLAT_SPAWN_M,
    "bumps": (0.35, -2.2, 0.455),
    "rough": (0.35, 0.0, 0.455),
    "ramp": (2.75, 0.0, 0.455),
}
HELDOUT_CASES = (
    ("flat_0p6", "flat", 0.6, 0.0, 88501),
    ("flat_1p6", "flat", 1.6, 0.0, 88502),
    ("flat_1p2_yaw", "flat", 1.2, 0.3, 88503),
    ("bumps_0p4", "bumps", 0.4, 0.0, 88504),
    ("rough_0p35", "rough", 0.35, 0.0, 88505),
    ("ramp_0p35", "ramp", 0.35, 0.0, 88506),
)


def _real(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer,
                                                                      np.floating)):
        raise TypeError(f"{label} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _rate_ticks(speed: float) -> int:
    # Uniformly sampled training targets need not lie on the .005 grid; the
    # last servo increment can be shorter than a full rate-limited tick.
    return math.ceil(speed / 0.005 - 1e-12)


@dataclass(frozen=True, slots=True)
class FullDriveEpisodeSchedule:
    """Exactly one immutable 1600-tick raw-command episode."""

    case_id: str
    terrain: str
    spawn_position_m: tuple[float, float, float]
    target_speed_mps: float
    yaw_amplitude_rps: float
    seed: int | None
    source_kind: str
    raw_commands: tuple[FullDriveCommand, ...]
    schema: str = SCHEDULE_SCHEMA

    def __post_init__(self) -> None:
        if not self.case_id or self.terrain not in COURSE_SPAWNS_M:
            raise ValueError("episode needs a named case and known course lane")
        if tuple(self.spawn_position_m) != COURSE_SPAWNS_M[self.terrain]:
            raise ValueError("episode spawn must match the preregistered course lane")
        if len(self.raw_commands) != 1600 or not all(
            isinstance(command, FullDriveCommand) for command in self.raw_commands
        ):
            raise ValueError("episode must contain exactly 1600 typed raw commands")
        if self.source_kind not in ("train", "heldout", "e1_baseline"):
            raise ValueError("unknown episode request source")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("episode seed must be a nonnegative integer or None")

    def command_source(self, tick: int, time_s: float) -> FullDriveCommand:
        if type(tick) is not int or not 0 <= tick < 1600:
            raise ValueError("request tick is outside this episode")
        if not math.isclose(_real(time_s, "control time"), tick * CONTROL_DT_S,
                            rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("request time differs from the actual control tick")
        return self.raw_commands[tick]

    @property
    def command_sha256(self) -> str:
        values = [
            [command.forward_velocity_mps, command.lateral_velocity_mps,
             command.yaw_rate_rps, command.clearance_m, command.jump_requested]
            for command in self.raw_commands
        ]
        return hashlib.sha256(json.dumps(values, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()


def _ramp_hold_commands(speed: float, yaw_amplitude: float) -> tuple[FullDriveCommand, ...]:
    ramp_ticks = _rate_ticks(speed)
    hold_begin = 175 + ramp_ticks
    hold_end = hold_begin + 400
    down_end = hold_end + ramp_ticks
    if down_end > 1600:
        raise ValueError("requested ramp/hold cannot fit a 1600-tick episode")
    commands = []
    for tick in range(1600):
        if tick < 175:
            vx = 0.0
        elif tick < hold_end:
            vx = speed
        else:
            vx = 0.0
        yaw = 0.0
        hold_tick = tick - hold_begin
        if 0 <= hold_tick < 400 and yaw_amplitude != 0.0:
            yaw = yaw_amplitude if hold_tick < 100 or hold_tick >= 300 else -yaw_amplitude
        commands.append(FullDriveCommand(vx, 0.0, yaw, 0.455, False))
    return tuple(commands)


def training_schedule(
    *, level: int, terrain: str, target_speed_mps: float,
    yaw_amplitude_rps: float, episode_index: int, flat_episode_ordinal: int | None,
    schedule_seed: int,
) -> FullDriveEpisodeSchedule:
    """Bind one externally selected, seeded curriculum choice before physics."""
    if type(level) is not int or level not in (0, 1, 2, 3):
        raise ValueError("training level must be L0 through L3")
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError("episode index must be nonnegative")
    if type(schedule_seed) is not int or schedule_seed < 0:
        raise ValueError("schedule seed must be nonnegative")
    if terrain not in COURSE_SPAWNS_M:
        raise ValueError("training terrain is not an authorized course lane")
    if (level < 2 and terrain != "flat"
            or level == 2 and terrain not in ("flat", "bumps")):
        raise ValueError("terrain is not enabled at this curriculum level")
    speed = _real(target_speed_mps, "training target speed")
    yaw = _real(yaw_amplitude_rps, "training yaw amplitude")
    if terrain == "flat":
        ceiling = (0.4, 0.9, 1.6, 1.6)[level]
        if not 0.0 <= speed <= ceiling:
            raise ValueError("flat target speed exceeds the selected level")
        if level == 0 and yaw != 0.0 or level > 0 and abs(yaw) > 0.3:
            raise ValueError("flat yaw amplitude exceeds the corrected curriculum")
        if flat_episode_ordinal is None or type(flat_episode_ordinal) is not int or flat_episode_ordinal < 0:
            raise ValueError("flat episode ordinal is required for the every-fourth ceiling")
        if (flat_episode_ordinal + 1) % 4 == 0 and speed != ceiling:
            raise ValueError("every fourth flat episode must request its level ceiling")
    elif not 0.2 <= speed <= 0.4 or yaw != 0.0:
        raise ValueError("terrain pilot needs speed .2-.4 and zero yaw")
    elif flat_episode_ordinal is not None:
        raise ValueError("terrain episodes must not advance the flat ordinal")
    return FullDriveEpisodeSchedule(
        f"train_{episode_index:06d}_L{level}_{terrain}", terrain,
        COURSE_SPAWNS_M[terrain], speed, yaw, schedule_seed, "train",
        _ramp_hold_commands(speed, yaw),
    )


def heldout_schedule(case_id: str) -> FullDriveEpisodeSchedule:
    for name, terrain, speed, yaw, seed in HELDOUT_CASES:
        if name == case_id:
            return FullDriveEpisodeSchedule(
                name, terrain, COURSE_SPAWNS_M[terrain], speed, yaw, seed,
                "heldout", _ramp_hold_commands(speed, yaw),
            )
    raise ValueError("unknown preregistered heldout case")


def e1_ladder_schedule() -> FullDriveEpisodeSchedule:
    """The finite zero-residual speed ladder: 175 settle, five holds, stop."""
    segments = (
        (175, 0.0), (180, 0.4), (140, 0.6), (160, 0.9),
        (160, 1.2), (280, 1.6), (505, 0.0),
    )
    commands = tuple(
        FullDriveCommand(speed) for count, speed in segments for _ in range(count)
    )
    return FullDriveEpisodeSchedule(
        "e1_zero_residual_speed_ladder", "flat", FLAT_SPAWN_M,
        1.6, 0.0, None, "e1_baseline", commands,
    )


def e0_stationary_probe_command(tick: int, time_s: float) -> FullDriveCommand:
    if type(tick) is not int or not 0 <= tick < 800:
        raise ValueError("E0 probe tick must be in [0,800)")
    if not math.isclose(_real(time_s, "control time"), tick * CONTROL_DT_S,
                        rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("E0 request time differs from control tick")
    return FullDriveCommand()


def e0_stationary_probe_action(tick: int) -> np.ndarray:
    """Single normalized +.15 action for ten ticks, followed by ten zero."""
    if type(tick) is not int or not 0 <= tick < 800:
        raise ValueError("E0 probe action tick must be in [0,800)")
    result = np.zeros(16, dtype=np.float64)
    phase = tick - 200
    if 0 <= phase < 320 and phase % 20 < 10:
        result[phase // 20] = 0.15
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class NominalPathReceipt:
    """Command-only geometry precheck, never a simulated trajectory claim."""

    case_id: str
    passed: bool
    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float
    final_heading_rad: float
    raw_command_sha256: str
    servo_product_delays: int


def precheck_nominal_path(schedule: FullDriveEpisodeSchedule,
                          caps: QualifiedCommandCaps) -> NominalPathReceipt:
    """Integrate actual pure servo requests before any model construction."""
    if not isinstance(schedule, FullDriveEpisodeSchedule):
        raise TypeError("nominal path precheck requires a frozen episode schedule")
    servo = FullDriveCommandServo(caps)
    x, y, _ = schedule.spawn_position_m
    heading = 0.0
    min_x = max_x = x
    min_y = max_y = y
    delays = 0
    for tick, raw in enumerate(schedule.raw_commands):
        consumed = servo.advance(tick, raw)
        applied = consumed.applied
        delays += consumed.product_delayed_axis is not None
        x += math.cos(heading) * applied.forward_velocity_mps * CONTROL_DT_S
        y += math.sin(heading) * applied.forward_velocity_mps * CONTROL_DT_S
        heading += applied.yaw_rate_rps * CONTROL_DT_S
        min_x, max_x = min(min_x, x), max(max_x, x)
        min_y, max_y = min(min_y, y), max(max_y, y)
    if schedule.terrain == "flat":
        passed = max(abs(min_x), abs(max_x)) < 10.5 and -5.8 < min_y and max_y < -3.6
    else:
        passed = max(abs(min_x), abs(max_x)) < 10.5 and max(abs(min_y), abs(max_y)) < 5.8
    return NominalPathReceipt(
        schedule.case_id, passed, min_x, max_x, min_y, max_y,
        heading, schedule.command_sha256, delays,
    )
