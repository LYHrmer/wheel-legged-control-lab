"""C18 raw command recipes with explicit session seeds and yaw mirror."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from budget_spec_11 import BudgetSpec
from full_drive_command_08 import FullDriveCommand
from full_drive_schedule_08 import COURSE_SPAWNS_M
from short_episode_plan_11 import (
    SETTLE_TICKS, SHORT_EPISODE_TICKS, FrozenShortPlan, ShortEpisodeSchedule,
)
from short_heldout_11 import HELDOUT_SCHEMA, WorldUprightHeldoutSchedule

PLAN_SCHEMA = "d1-world-upright-short1000-stage15-preregistered-plan-v1"
HELDOUT_TABLE_SCHEMA = "d1-world-upright-stage18-six-task-patterns-v1"
RECIPE_SCHEMA = "d1-world-upright-groupedclip-stage15-recipe-v1"
PPO_SEED = 151001
SELECTION_SEED = 151002
MEASUREMENT_SEED = 151003
FLOOR_SEED = 151090
TRAIN_BUDGET = BudgetSpec(16384, 1024, 256, 4)

# The 11-S first-eight and eight-slot rules are copied as values. Their module
# globals are never changed, and each index gets an independent PCG64 stream.
_WARMUP = (
    (0.4, 0.0, 1), (0.6, 0.0, 1), (0.8, 0.0, 1), (1.0, 0.0, 1),
    (1.2, 0.0, 1), (1.6, 0.0, 1), (0.6, 0.15, 1), (0.9, 0.2, -1),
)
_SLOTS = (
    ("flat", (0.4, 0.9), None),
    ("flat", (1.1, 1.6), None),
    ("flat", (0.6, 1.2), (0.15, 0.3)),
    ("bumps", (0.25, 0.4), None),
    ("rough", (0.25, 0.4), None),
    ("ramp", (0.35, 0.5), None),
    ("flat", 1.6, None),
    ("ramp", (0.35, 0.5), None),
)


def select_episode(index: int) -> ShortEpisodeSchedule:
    """Select local episode index 0 onward using only the new selection seed."""
    if type(index) is not int or index < 0:
        raise ValueError("episode index must be a nonnegative integer")
    rng = np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([SELECTION_SEED, index]),
    ))
    if index < len(_WARMUP):
        terrain = "flat"
        speed, amplitude, sign = _WARMUP[index]
        slot = None
        phase = "first_eight_fixed_flat"
        draws: list[str] = []
    else:
        slot = (index - len(_WARMUP)) % len(_SLOTS)
        terrain, speed_rule, yaw_rule = _SLOTS[slot]
        phase = "fixed_eight_slot_cycle"
        draws = []
        if isinstance(speed_rule, tuple):
            speed = float(rng.uniform(*speed_rule))
            draws.append("speed_uniform_half_open")
        else:
            speed = float(speed_rule)
        if yaw_rule is None:
            amplitude, sign = 0.0, 1
        else:
            amplitude = float(rng.uniform(*yaw_rule))
            draws.append("yaw_amplitude_uniform_half_open")
            sign = 1 if int(rng.integers(0, 2)) == 0 else -1
            draws.append("first_yaw_sign_integers_0_or_1")
    if abs(speed * amplitude) > 0.8:
        raise ValueError("frozen selector exceeded the raw forward-yaw product cap")
    zero = FullDriveCommand()
    raw: list[FullDriveCommand] = []
    for tick in range(SHORT_EPISODE_TICKS):
        if tick < SETTLE_TICKS:
            raw.append(zero)
            continue
        yaw = 0.0
        if amplitude and 450 <= tick < 850:
            yaw = amplitude * (sign if tick < 550 or tick >= 750 else -sign)
        raw.append(FullDriveCommand(
            forward_velocity_mps=speed, lateral_velocity_mps=0.0,
            yaw_rate_rps=yaw, clearance_m=0.455, jump_requested=False,
        ))
    choice = {
        "schema": PLAN_SCHEMA, "episode_index": index,
        "selection_seed": SELECTION_SEED,
        "rng": "PCG64(SeedSequence([151002,episode_index]))",
        "phase": phase, "slot": slot, "draw_order": draws,
        "terrain": terrain, "speed_mps": speed,
        "yaw_amplitude_rps": amplitude, "yaw_first_sign": sign,
        "yaw_windows": [[450, 550], [550, 750], [750, 850]],
        "settle_end_tick": SETTLE_TICKS,
        "horizon_ticks": SHORT_EPISODE_TICKS,
    }
    return ShortEpisodeSchedule(
        episode_index=index, case_id=f"short_train_{index:06d}_{terrain}",
        terrain=terrain, spawn_position_m=COURSE_SPAWNS_M[terrain],
        selection_seed=SELECTION_SEED,
        choice_json=json.dumps(choice, sort_keys=True, separators=(",", ":"),
                               allow_nan=False),
        raw_commands=tuple(raw),
    )


def build_plan(*, source_sha256: str) -> FrozenShortPlan:
    """Bind the root-verified SHA256 of this source to the selector."""
    return FrozenShortPlan(PLAN_SCHEMA, source_sha256, select_episode)


@dataclass(frozen=True, slots=True)
class HeldoutTask:
    case_id: str
    terrain: str
    speed_mps: float
    yaw_amplitude_rps: float
    seed: int
    horizon: int
    hold_start: int
    hold_end: int
    release_tick: int
    final_start: int


TASKS = (
    HeldoutTask("flat_0p6", "flat", .6, 0.0, 151101, 1600, 295, 695, 695, 1500),
    HeldoutTask("flat_1p6", "flat", 1.6, 0.0, 151102, 1600, 495, 895, 895, 1500),
    HeldoutTask("flat_1p2_yaw", "flat", 1.2, .3, 151103, 1600, 415, 815, 815, 1500),
    HeldoutTask("bumps_0p4", "bumps", .4, 0.0, 151104, 1600, 255, 655, 655, 1500),
    HeldoutTask("rough_0p35", "rough", .35, 0.0, 151105, 1600, 245, 645, 645, 1500),
    HeldoutTask("ramp_0p45_complete", "ramp", .45, 0.0, 151106, 1800,
                600, 1000, 1355, 1700),
)
ORDER = tuple(task.case_id for task in TASKS)
TASK_BY_ID = {task.case_id: task for task in TASKS}


def make_schedule(case_id: str, seed: int, mirror: bool = False) -> WorldUprightHeldoutSchedule:
    """Build a frozen command pattern with a new fixed seed and optional yaw mirror."""
    if type(case_id) is not str or case_id not in TASK_BY_ID:
        raise ValueError("unknown heldout task")
    if type(seed) is not int or not 0 <= seed < 2**32 or type(mirror) is not bool:
        raise ValueError("heldout seed/mirror must be explicit and valid")
    if mirror and case_id != "flat_1p2_yaw":
        raise ValueError("mirror only applies to flat_1p2_yaw")
    task = TASK_BY_ID[case_id]
    if (task.hold_end - task.hold_start != 400
            or task.final_start != task.horizon - 100
            or not 175 < task.hold_start < task.hold_end <= task.release_tick
            or task.release_tick >= task.final_start):
        raise RuntimeError("heldout table window identities differ")
    raw = []
    for tick in range(task.horizon):
        speed = task.speed_mps if 175 <= tick < task.release_tick else 0.0
        yaw = 0.0
        if task.yaw_amplitude_rps and task.hold_start <= tick < task.hold_end:
            within_hold = tick - task.hold_start
            yaw = (task.yaw_amplitude_rps if within_hold < 100
                   or within_hold >= 300 else -task.yaw_amplitude_rps)
            if mirror:
                yaw = -yaw
        raw.append(FullDriveCommand(
            forward_velocity_mps=speed, lateral_velocity_mps=0.0,
            yaw_rate_rps=yaw, clearance_m=0.455, jump_requested=False,
        ))
    return WorldUprightHeldoutSchedule(
        case_id=task.case_id, terrain=task.terrain, seed=seed,
        spawn_position_m=COURSE_SPAWNS_M[task.terrain], raw_commands=tuple(raw),
    )


@dataclass(frozen=True, slots=True)
class FloorSchedule(WorldUprightHeldoutSchedule):
    """The same heldout interface with the separately reviewed 600-tick cap."""

    def __post_init__(self) -> None:
        zero = FullDriveCommand()
        drive = FullDriveCommand(forward_velocity_mps=.4, lateral_velocity_mps=0.0,
                                 yaw_rate_rps=0.0, clearance_m=.455,
                                 jump_requested=False)
        expected = (zero,) * 175 + (drive,) * 250 + (zero,) * 175
        if (self.case_id != "floor_0p4_600" or self.terrain != "flat"
                or type(self.seed) is not int or not 0 <= self.seed < 2**32
                or tuple(self.spawn_position_m) != COURSE_SPAWNS_M["flat"]
                or type(self.raw_commands) is not tuple
                or len(self.raw_commands) != 600
                or any(not isinstance(row, FullDriveCommand)
                       for row in self.raw_commands)
                or self.raw_commands != expected
                or self.schema != HELDOUT_SCHEMA):
            raise ValueError("600-control floor schedule differs from its identity")


def floor_schedule(seed: int) -> FloorSchedule:
    """One 175-settle, 250-drive, 175-zero raw flat probe at 0.4 m/s."""
    zero = FullDriveCommand()
    drive = FullDriveCommand(forward_velocity_mps=.4, lateral_velocity_mps=0.0,
                             yaw_rate_rps=0.0, clearance_m=.455,
                             jump_requested=False)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("floor seed must be an explicit uint32")
    return FloorSchedule(
        case_id="floor_0p4_600", terrain="flat", seed=seed,
        spawn_position_m=COURSE_SPAWNS_M["flat"],
        raw_commands=(zero,) * 175 + (drive,) * 250 + (zero,) * 175,
    )


__all__ = (
    "PLAN_SCHEMA", "HELDOUT_TABLE_SCHEMA", "RECIPE_SCHEMA", "PPO_SEED",
    "SELECTION_SEED", "MEASUREMENT_SEED", "FLOOR_SEED", "TRAIN_BUDGET",
    "TASKS", "ORDER", "HeldoutTask", "FloorSchedule", "select_episode",
    "build_plan", "make_schedule", "floor_schedule",
)
