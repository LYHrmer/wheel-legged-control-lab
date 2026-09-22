"""Frozen single 15 mm box, zero-action readiness task environment.

This module exposes :class:`D1SingleStepEnv`, a fixed (non-learning) task
wrapper around :class:`scripts.d1_heading_tracking_env.D1HeadingTrackingEnv`.
The task is completely scripted: the agent is only allowed to submit the
zero action, the forward command schedule is a frozen function of the control
tick, and the commanded world height is pinned at 0.455 m.

Plant construction, recording and physical episode execution remain the
responsibility of the root caller; this module performs no integration.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from scripts.d1_heading_tracking_env import D1HeadingTrackingEnv
from scripts.d1_single_step_plant import D1SingleStepPlant
from scripts.d1_stop_turn_composition import StopTurnCompositionController
from wheel_legged_control.d1.control_loop import D1WheelLegControllerAdapter
from wheel_legged_control.d1.locomotion_terrain import D1LocomotionTerrainConfig

try:  # D1MotionCommand lives with the control loop contract.
    from wheel_legged_control.d1.control_loop import D1MotionCommand
except ImportError:  # pragma: no cover - re-exported by the heading env.
    from scripts.d1_heading_tracking_env import D1MotionCommand

TASK_SCHEMA = "d1-single-15mm-box-zero-world-z-task-v1"
CONTROL_DT = 0.01
RAW_WORLD_HEIGHT_M = 0.455
TERMINAL_TICK = 1200
FORWARD_START_TICK = 200
FORWARD_STOP_TICK = 1000
FORWARD_VELOCITY_MPS = 0.2
REQUIRED_SEED = 77301
_TICK_TOL = 1e-10
_HEIGHT_TOL = 1e-9
_PLANT_ATTRS = ("plant", "_plant")


def raw_command_at_tick(tick: int) -> Dict[str, Any]:
    """Return the frozen raw command for ``tick`` (0..1200 inclusive)."""
    if isinstance(tick, bool):
        raise TypeError("tick must be an integer, not bool")
    if isinstance(tick, float):
        if not float(tick).is_integer():
            raise TypeError("tick must be integral")
        tick = int(tick)
    if not isinstance(tick, (int, np.integer)):
        raise TypeError(f"tick must be an integer, got {type(tick)!r}")
    tick = int(tick)
    if tick < 0 or tick > TERMINAL_TICK:
        raise ValueError(f"tick out of range [0, {TERMINAL_TICK}]: {tick}")
    forward = FORWARD_VELOCITY_MPS if FORWARD_START_TICK <= tick < FORWARD_STOP_TICK else 0.0
    return {
        "tick": tick,
        "forward_velocity_mps": forward,
        "yaw_rate_rps": 0.0,
        "world_height_m": RAW_WORLD_HEIGHT_M,
    }


def _raw_schedule() -> List[Dict[str, Any]]:
    return [
        {"tick_start": 0, "tick_end": FORWARD_START_TICK, "forward_velocity_mps": 0.0},
        {"tick_start": FORWARD_START_TICK, "tick_end": FORWARD_STOP_TICK,
         "forward_velocity_mps": FORWARD_VELOCITY_MPS},
        {"tick_start": FORWARD_STOP_TICK, "tick_end": TERMINAL_TICK, "forward_velocity_mps": 0.0},
        {"tick": TERMINAL_TICK, "prepared_only": True, "executed": False},
    ]


def _require_zero_action(action: Any) -> Any:
    arr = np.asarray(action)
    if arr.dtype.kind not in ("f", "i", "u"):
        raise TypeError(f"action must be real numeric, got dtype {arr.dtype!r}")
    if arr.shape != (8,):
        raise ValueError(f"action must have shape (8,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("action must be finite")
    if bool(np.any(arr != 0)):
        raise ValueError("this task accepts the zero action only")
    return action


class D1SingleStepEnv(D1HeadingTrackingEnv):
    """Zero-action readiness task over a single 15 mm box (or bare flat floor)."""

    def __init__(self, obstacle_enabled: bool) -> None:
        if not isinstance(obstacle_enabled, bool):
            raise TypeError("obstacle_enabled must be a bool")
        self.obstacle_enabled = obstacle_enabled  # consumed by _build_heading_task_config
        self.command_records: List[Dict[str, Any]] = []
        self.raw_callback_count = 0
        super().__init__(
            episode_seconds=12.0,
            terrain=D1LocomotionTerrainConfig(layout="flat"),
            command_source=self._command_source,
        )
        if self.loop is not None or self.decision is not None or getattr(self, "_active", False):
            raise RuntimeError("parent constructor must leave loop/decision None and inactive")
        attr = self._plant_attr()
        old_plant = getattr(self, attr)
        if float(old_plant.data.time) != 0.0:
            raise RuntimeError("constructed plant must never have been stepped")
        channel = getattr(old_plant, "actuator_channel", None)
        setattr(self, attr, D1SingleStepPlant(
            obstacle_enabled=obstacle_enabled,
            control_dt=CONTROL_DT,
            actuator_channel=channel,
        ))
        self._controller = self._build_composition_adapter()

    # ---------------------------------------------------------------- helpers
    def _plant_attr(self) -> str:
        for name in _PLANT_ATTRS:
            if getattr(self, name, None) is not None:
                return name
        raise AttributeError("unable to locate the parent plant attribute")

    @property
    def plant(self):  # type: ignore[override]
        return self.__dict__.get("plant") or self.__dict__.get("_plant")

    def _build_composition_adapter(self) -> D1WheelLegControllerAdapter:
        composition = StopTurnCompositionController(enabled=True, **asdict(self.wheel_leg_control))
        try:
            return D1WheelLegControllerAdapter(composition)
        except TypeError:
            return D1WheelLegControllerAdapter(controller=composition)

    def _build_heading_task_config(self, *args: Any, **kwargs: Any):
        config = super()._build_heading_task_config(*args, **kwargs)
        config["task_schema"] = TASK_SCHEMA
        config["collision_identity"] = (
            "single-15mm-box" if self.obstacle_enabled else "flat-floor-no-obstacle"
        )
        config["obstacle_enabled"] = self.obstacle_enabled
        config["controller_schema"] = "stop-turn-composition"
        config["height_semantics"] = {
            "raw_world_height_m": RAW_WORLD_HEIGHT_M,
            "raw_is_world_z": True,
            "user_command_clearance_is_adapted": True,
            "note": "callback returns clearance = raw world z - provider ground height",
        }
        return config

    # -------------------------------------------------------------- callback
    def _command_source(self, time_s: float) -> D1MotionCommand:
        tick = int(round(float(time_s) / CONTROL_DT))
        if abs(float(time_s) - tick * CONTROL_DT) > _TICK_TOL:
            raise ValueError(f"time {time_s!r} is not tick aligned")
        if tick != int(self._steps):
            raise RuntimeError(f"tick {tick} does not match step counter {self._steps}")
        raw = raw_command_at_tick(tick)
        ground = self.loop.provider.ground_reference()
        ground_h = float(ground.height_m)
        clearance = RAW_WORLD_HEIGHT_M - ground_h
        self._controller.controller.bind_raw_command(
            forward_velocity_mps=raw["forward_velocity_mps"],
            yaw_rate_rps=raw["yaw_rate_rps"],
            control_time_s=float(time_s),
        )
        self.raw_callback_count += 1
        self.command_records.append({
            "tick": tick,
            "time_s": float(time_s),
            "forward_velocity_mps": raw["forward_velocity_mps"],
            "yaw_rate_rps": raw["yaw_rate_rps"],
            "raw_world_height_m": RAW_WORLD_HEIGHT_M,
            "ground_height_m": ground_h,
            "motion_clearance_m": clearance,
            "prepared_world_height_m": None,
        })
        return D1MotionCommand(
            forward_velocity_mps=raw["forward_velocity_mps"],
            yaw_rate_rps=0.0,
            clearance_m=clearance,
        )

    def _prepare(self, *args: Any, **kwargs: Any):
        result = super()._prepare(*args, **kwargs)
        world_height = float(self.decision.world_command.base_height_m)
        if abs(world_height - RAW_WORLD_HEIGHT_M) > _HEIGHT_TOL:
            raise RuntimeError(
                f"prepared world height {world_height!r} != {RAW_WORLD_HEIGHT_M}"
            )
        if self.command_records:
            self.command_records[-1]["prepared_world_height_m"] = world_height
        return result

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is None or isinstance(seed, bool) or int(seed) != REQUIRED_SEED:
            raise ValueError(f"this fixed task requires seed={REQUIRED_SEED}")
        if options:
            raise ValueError("options must be None or empty for this fixed task")
        self.plant.validate_single_step()
        self.raw_callback_count = 0
        self.command_records = []
        obs, info = super().reset(seed=seed, options=options)
        metadata = self._episode_metadata
        metadata["collision_terrain"] = dict(self.plant.collision_terrain_metadata)
        metadata["raw_world_height_m"] = RAW_WORLD_HEIGHT_M
        metadata["raw_schedule"] = _raw_schedule()
        metadata["no_learning"] = True
        metadata["zero_only"] = True
        metadata["record_semantics"] = {
            "raw_world_height_m": "commanded world z (frozen)",
            "motion_clearance_m": "adapted clearance handed to the parent loop",
            "prepared_world_height_m": "world z reconstructed by the parent command loop",
            "user_command_clearance_m": "adapted clearance, NOT raw world z",
        }
        info["episode_metadata"] = self.episode_metadata
        return obs, info

    def step(self, action):
        _require_zero_action(action)
        return super().step(action)
