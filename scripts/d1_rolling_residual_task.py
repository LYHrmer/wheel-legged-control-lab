"""Pure task and action contract for one rolling 15 mm obstacle.

This module has no MuJoCo, controller, model, or training imports. It defines
the single scalar policy action and its exact eight-channel physical expansion;
the existing heading observation and reward remain unchanged.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

ROLLING_TASK_SCHEMA = "d1-rolling-15mm-shared-leg-residual-v1"
ROLLING_ACTION_SCHEMA = "d1-shared-leg1-wheel-zero-gated-v1"
ROLLING_CONTROL_SCHEMA = "d1-stop-turn-shared-leg-residual-v1"
HEADING_OBSERVATION_SCHEMA = "d1-proprio-servo82-heading85-v1"
HEADING_REWARD_SCHEMA = "d1-heading-goal-servo-rate-v1"
CONTROL_DT_S = 0.01
TERMINAL_TICK = 1200
DRIVE_TICKS = 800
RAW_WORLD_HEIGHT_M = 0.455
POLICY_ACTION_SIZE = 1
PHYSICAL_ACTION_SIZE = 8
LEG_NORMALIZED_SCALE = 0.25
ORIGINAL_LEG_EXTENSION_SCALE_M = 0.04
MAX_LEG_EXTENSION_RESIDUAL_M = LEG_NORMALIZED_SCALE * ORIGINAL_LEG_EXTENSION_SCALE_M


def _finite_speed(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError("forward speed must be a real numeric scalar")
    speed = float(value)
    if not math.isfinite(speed) or not 0.0 < speed <= 0.25:
        raise ValueError("forward speed must be finite and in (0, 0.25] m/s")
    return speed


def _tick(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer tick")
    tick = int(value)
    if not 0 <= tick <= TERMINAL_TICK:
        raise ValueError(f"{name} must lie in [0, {TERMINAL_TICK}]")
    return tick


@dataclass(frozen=True, slots=True)
class RollingEpisodeSpec:
    """One fixed-speed, 800-tick drive and 15 mm box/plane selection."""

    forward_velocity_mps: float
    forward_start_tick: int
    forward_stop_tick: int
    obstacle_enabled: bool = True

    def __post_init__(self) -> None:
        speed = _finite_speed(self.forward_velocity_mps)
        start = _tick(self.forward_start_tick, "forward_start_tick")
        stop = _tick(self.forward_stop_tick, "forward_stop_tick")
        if stop - start != DRIVE_TICKS:
            raise ValueError(f"forward interval must have exactly {DRIVE_TICKS} ticks")
        if type(self.obstacle_enabled) is not bool:
            raise TypeError("obstacle_enabled must be bool")
        object.__setattr__(self, "forward_velocity_mps", speed)
        object.__setattr__(self, "forward_start_tick", start)
        object.__setattr__(self, "forward_stop_tick", stop)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RollingAction:
    """Exact policy input, clipped scalar, and applied physical action."""

    policy_scalar: float
    clipped_scalar: float
    forward_gate_active: bool
    physical_action: tuple[float, float, float, float, float, float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def raw_command_at_tick(spec: RollingEpisodeSpec, tick: int) -> dict[str, float | int]:
    if not isinstance(spec, RollingEpisodeSpec):
        raise TypeError("spec must be a RollingEpisodeSpec")
    now = _tick(tick, "tick")
    speed = (spec.forward_velocity_mps
             if spec.forward_start_tick <= now < spec.forward_stop_tick else 0.0)
    return {
        "tick": now,
        "forward_velocity_mps": speed,
        "yaw_rate_rps": 0.0,
        "world_height_m": RAW_WORLD_HEIGHT_M,
    }


def expand_policy_action(action: Any, raw_forward_velocity_mps: Any) -> RollingAction:
    """Validate scalar policy action; gate it by the current *raw* command.

    The original controller scales each leg's normalized residual by 0.04 m.
    Here the four shared normalized inputs are limited to ±0.25, hence ±0.01 m.
    All four wheel residuals are exact zeros, including during forward drive.
    """
    array = np.asarray(action)
    if array.dtype.kind not in ("f", "i", "u"):
        raise TypeError("policy action must be a real numeric length-one array")
    if array.shape != (POLICY_ACTION_SIZE,):
        raise ValueError("policy action must have shape (1,)")
    scalar = float(array[0])
    if not math.isfinite(scalar):
        raise ValueError("policy action must be finite")
    if isinstance(raw_forward_velocity_mps, (bool, np.bool_)) or not isinstance(
        raw_forward_velocity_mps, (int, float, np.integer, np.floating)
    ):
        raise TypeError("raw forward velocity must be a real numeric scalar")
    forward = float(raw_forward_velocity_mps)
    if not math.isfinite(forward):
        raise ValueError("raw forward velocity must be finite")
    clipped = float(np.clip(scalar, -1.0, 1.0))
    active = forward != 0.0
    leg = LEG_NORMALIZED_SCALE * clipped if active else 0.0
    return RollingAction(
        policy_scalar=scalar,
        clipped_scalar=clipped,
        forward_gate_active=active,
        physical_action=(leg, leg, leg, leg, 0.0, 0.0, 0.0, 0.0),
    )


def rolling_task_definition() -> dict[str, Any]:
    """Comparable static task definition; per-episode speed/onset live elsewhere."""
    return {
        "task_schema": ROLLING_TASK_SCHEMA,
        "action_schema": ROLLING_ACTION_SCHEMA,
        "control_schema": ROLLING_CONTROL_SCHEMA,
        "observation_schema": HEADING_OBSERVATION_SCHEMA,
        "reward_schema": HEADING_REWARD_SCHEMA,
        "policy_action_size": POLICY_ACTION_SIZE,
        "physical_action_size": PHYSICAL_ACTION_SIZE,
        "policy_to_physical_indices": [0, 0, 0, 0, None, None, None, None],
        "constant_physical_channels": {"wheel_indices": [4, 5, 6, 7], "value": 0.0},
        "leg_normalized_scale": LEG_NORMALIZED_SCALE,
        "max_leg_extension_residual_m": MAX_LEG_EXTENSION_RESIDUAL_M,
        "forward_gate": "raw_forward_velocity_mps != 0.0",
        "duration_s": TERMINAL_TICK * CONTROL_DT_S,
        "drive_ticks": DRIVE_TICKS,
        "world_height_m": RAW_WORLD_HEIGHT_M,
        "yaw_rate_rps": 0.0,
        "observation_truth_appended": False,
        "reward_changes_from_heading_parent": False,
    }
