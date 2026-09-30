"""Frozen per-index PCG64 selector for the 11-S 1000-control train plan."""

from __future__ import annotations

import json

import numpy as np
from full_drive_command_08 import FullDriveCommand
from full_drive_schedule_08 import COURSE_SPAWNS_M
from short_episode_plan_11 import (
    SETTLE_TICKS,
    SHORT_EPISODE_TICKS,
    FrozenShortPlan,
    ShortEpisodeSchedule,
)

PLAN_SCHEMA = "d1-world-upright-short1000-preregistered-plan-v1"
SELECTION_SEED = 88622
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
    """Select exactly once from an index-local PCG64 stream, independent of outcomes."""
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
            amplitude = 0.0
            sign = 1
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
            forward_velocity_mps=speed,
            lateral_velocity_mps=0.0,
            yaw_rate_rps=yaw,
            clearance_m=0.455,
            jump_requested=False,
        ))
    choice = {
        "schema": PLAN_SCHEMA,
        "episode_index": index,
        "selection_seed": SELECTION_SEED,
        "rng": "PCG64(SeedSequence([88622,episode_index]))",
        "phase": phase,
        "slot": slot,
        "draw_order": draws,
        "terrain": terrain,
        "speed_mps": speed,
        "yaw_amplitude_rps": amplitude,
        "yaw_first_sign": sign,
        "yaw_windows": [[450, 550], [550, 750], [750, 850]],
        "settle_end_tick": SETTLE_TICKS,
        "horizon_ticks": SHORT_EPISODE_TICKS,
    }
    return ShortEpisodeSchedule(
        episode_index=index,
        case_id=f"short_train_{index:06d}_{terrain}",
        terrain=terrain,
        spawn_position_m=COURSE_SPAWNS_M[terrain],
        selection_seed=SELECTION_SEED,
        choice_json=json.dumps(choice, sort_keys=True, separators=(",", ":"),
                               allow_nan=False),
        raw_commands=tuple(raw),
    )


def build_plan(*, source_sha256: str) -> FrozenShortPlan:
    """Bind a root-verified source hash to this exact selector."""
    return FrozenShortPlan(PLAN_SCHEMA, source_sha256, select_episode)


__all__ = ("PLAN_SCHEMA", "SELECTION_SEED", "build_plan", "select_episode")
