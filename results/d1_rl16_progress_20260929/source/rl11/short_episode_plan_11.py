"""Immutable 1000-tick request and deterministic selection seam for 11 training.

This module fixes only episode shape and record identity. A later reviewed
contract must supply the exact terrain/speed/yaw selection function and seeds.
Nothing here creates a plant, policy or engine object.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from full_drive_command_08 import FullDriveCommand, QualifiedCommandCaps
from full_drive_schedule_08 import COURSE_SPAWNS_M

SHORT_SCHEDULE_SCHEMA = "d1-course-world-upright-short1000-request-v1"
SHORT_EPISODE_TICKS = 1000
SETTLE_TICKS = 175


@dataclass(frozen=True, slots=True)
class ShortEpisodeSchedule:
    """One preregistered raw episode; no success-driven redraw or promotion."""

    episode_index: int
    case_id: str
    terrain: str
    spawn_position_m: tuple[float, float, float]
    selection_seed: int
    choice_json: str
    raw_commands: tuple[FullDriveCommand, ...]
    schema: str = SHORT_SCHEDULE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.episode_index) is not int or self.episode_index < 0:
            raise ValueError("episode index must be a nonnegative integer")
        if type(self.selection_seed) is not int or not 0 <= self.selection_seed < 2**32:
            raise ValueError("schedule seed must be an explicit uint32")
        if (type(self.case_id) is not str or not self.case_id
                or self.terrain not in COURSE_SPAWNS_M
                or tuple(self.spawn_position_m) != COURSE_SPAWNS_M[self.terrain]
                or self.schema != SHORT_SCHEDULE_SCHEMA):
            raise ValueError("episode identity or actual course spawn is invalid")
        if type(self.choice_json) is not str:
            raise TypeError("schedule choice must be canonical JSON")
        choice = json.loads(self.choice_json)
        if (not isinstance(choice, dict)
                or any(type(key) is not str for key in choice)
                or json.dumps(choice, sort_keys=True, separators=(",", ":"),
                              allow_nan=False) != self.choice_json):
            raise ValueError("schedule choice is not a canonical string-keyed record")
        if (type(self.raw_commands) is not tuple
                or len(self.raw_commands) != SHORT_EPISODE_TICKS
                or any(not isinstance(row, FullDriveCommand) for row in self.raw_commands)):
            raise ValueError("short episode requires exactly 1000 typed raw commands")
        if any(command.forward_velocity_mps != 0.0 or command.lateral_velocity_mps != 0.0
               or command.yaw_rate_rps != 0.0 or command.jump_requested
               for command in self.raw_commands[:SETTLE_TICKS]):
            raise ValueError("the first 175 controls must be an actual zero-request settle")
        if any(command.forward_velocity_mps <= 0.0 or command.lateral_velocity_mps != 0.0
               or command.jump_requested
               for command in self.raw_commands[SETTLE_TICKS:]):
            raise ValueError("the remaining 825 controls require positive raw forward motion")

    def command_source(self, tick: int, time_s: float) -> FullDriveCommand:
        if (type(tick) is not int or not 0 <= tick < SHORT_EPISODE_TICKS
                or not isinstance(time_s, (int, float)) or isinstance(time_s, bool)
                or not math.isclose(float(time_s), tick * 0.01, rel_tol=0.0,
                                    abs_tol=1e-10)):
            raise ValueError("request differs from the preregistered control tick")
        return self.raw_commands[tick]

    @property
    def command_sha256(self) -> str:
        rows = [asdict(command) for command in self.raw_commands]
        return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()

    def record(self, caps: QualifiedCommandCaps) -> dict[str, Any]:
        if not isinstance(caps, QualifiedCommandCaps):
            raise TypeError("actual command caps are required for schedule validation")
        for command in self.raw_commands:
            caps.check(command)
        return {
            "schema": self.schema,
            "episode_index": self.episode_index,
            "case_id": self.case_id,
            "terrain": self.terrain,
            "spawn_position_m": self.spawn_position_m,
            "selection_seed": self.selection_seed,
            "choice": json.loads(self.choice_json),
            "raw_command_sha256": self.command_sha256,
            "raw_commands": [asdict(command) for command in self.raw_commands],
        }


@dataclass(frozen=True, slots=True)
class FrozenShortPlan:
    """Reviewed deterministic recipe supplied by a later execution contract."""

    schema: str
    source_sha256: str
    select: Callable[[int], ShortEpisodeSchedule]

    def __post_init__(self) -> None:
        if (type(self.schema) is not str or not self.schema
                or type(self.source_sha256) is not str or len(self.source_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.source_sha256)
                or not callable(self.select)):
            raise ValueError("short plan requires a named frozen selector source")

    def episode(self, index: int) -> ShortEpisodeSchedule:
        if type(index) is not int or index < 0:
            raise ValueError("episode index must be nonnegative int")
        schedule = self.select(index)
        if not isinstance(schedule, ShortEpisodeSchedule) or schedule.episode_index != index:
            raise RuntimeError("frozen plan did not return the requested episode")
        return schedule


__all__ = (
    "SETTLE_TICKS",
    "SHORT_EPISODE_TICKS",
    "SHORT_SCHEDULE_SCHEMA",
    "FrozenShortPlan",
    "ShortEpisodeSchedule",
)
