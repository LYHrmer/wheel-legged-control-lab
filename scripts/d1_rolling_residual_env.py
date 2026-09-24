"""Thin shared-leg residual actor over the frozen heading/15 mm plant chain.

The inner heading environment keeps its original eight-channel control path,
reward, 85-entry observation, reset, and single physical step. An outer wrapper
exposes one policy scalar and records its exact gated physical expansion.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scripts.d1_heading_tracking_env import (
    HEADING_OBSERVATION_SCHEMA,
    HEADING_REWARD_SCHEMA,
    D1HeadingTrackingEnv,
)
from scripts.d1_rolling_residual_task import (
    CONTROL_DT_S,
    PHYSICAL_ACTION_SIZE,
    POLICY_ACTION_SIZE,
    RAW_WORLD_HEIGHT_M,
    ROLLING_ACTION_SCHEMA,
    ROLLING_CONTROL_SCHEMA,
    ROLLING_TASK_SCHEMA,
    RollingEpisodeSpec,
    expand_policy_action,
    raw_command_at_tick,
    rolling_task_definition,
)
from scripts.d1_single_step_plant import D1SingleStepPlant
from scripts.d1_stop_turn_composition import StopTurnCompositionController
from wheel_legged_control.d1.control_loop import D1MotionCommand, D1WheelLegControllerAdapter
from wheel_legged_control.d1.locomotion_terrain import D1LocomotionTerrainConfig


class RollingResidualCompositionController(StopTurnCompositionController):
    """Open only the shared leg residual; preserve all inherited control math."""

    def __init__(self, *, enabled: bool = True, **kwargs: Any) -> None:
        super().__init__(enabled=enabled, **kwargs)
        if not self.enabled:
            raise ValueError("rolling residual requires the enabled stop/turn composition")
        self.control_schema = ROLLING_CONTROL_SCHEMA

    @staticmethod
    def _checked_zero_action(action: Any) -> np.ndarray:
        """Validate before either inherited compute can change controller memory."""
        array = np.asarray(action)
        if array.dtype.kind not in ("f", "i", "u"):
            raise TypeError("physical action must contain eight real numeric values")
        checked = np.ascontiguousarray(array, dtype=np.float64)
        if checked.shape != (PHYSICAL_ACTION_SIZE,):
            raise ValueError("physical action must have shape (8,)")
        if not np.isfinite(checked).all():
            raise ValueError("physical action must be finite")
        if not np.all(checked[:4] == checked[0]) or np.any(np.abs(checked[:4]) > 0.25):
            raise ValueError("four shared leg residuals must agree and remain within ±0.25")
        if np.any(checked[4:] != 0.0):
            raise ValueError("all wheel residuals must be exactly zero")
        return checked


class _RollingPhysicalEnv(D1HeadingTrackingEnv):
    """Keep the heading parent's actual 8D action and replace only the plant."""

    task_schema = ROLLING_TASK_SCHEMA

    def __init__(self, spec: RollingEpisodeSpec) -> None:
        if not isinstance(spec, RollingEpisodeSpec):
            raise TypeError("spec must be a RollingEpisodeSpec")
        self.rolling_spec = spec
        self._pending_spec: RollingEpisodeSpec | None = None
        self.command_records: list[dict[str, Any]] = []
        self.raw_callback_count = 0
        super().__init__(
            episode_seconds=12.0,
            terrain=D1LocomotionTerrainConfig(layout="flat"),
            command_source=self._command_source,
        )
        if self.loop is not None or self.decision is not None or getattr(self, "_active", False):
            raise RuntimeError("parent construction must leave an inactive, unprepared plant")
        old_plant = self.plant
        if float(old_plant.data.time) != 0.0:
            raise RuntimeError("parent construction advanced physical time")
        self.plant = D1SingleStepPlant(
            obstacle_enabled=spec.obstacle_enabled,
            control_dt=CONTROL_DT_S,
            actuator_channel=old_plant.actuator_channel,
        )
        self._controller = D1WheelLegControllerAdapter(
            RollingResidualCompositionController(enabled=True, **asdict(self.wheel_leg_control))
        )

    def configure_next_episode(self, spec: RollingEpisodeSpec) -> None:
        if not isinstance(spec, RollingEpisodeSpec):
            raise TypeError("spec must be a RollingEpisodeSpec")
        if spec.obstacle_enabled != self.rolling_spec.obstacle_enabled:
            raise ValueError("switching physical box presence requires a fresh environment")
        # PPO may stop after a partial final episode. Stage the next spec only;
        # the current prepared command and its callback keep the current spec.
        self._pending_spec = spec

    def _command_source(self, time_s: float) -> D1MotionCommand:
        tick = round(float(time_s) / CONTROL_DT_S)
        if abs(float(time_s) - tick * CONTROL_DT_S) > 1e-10:
            raise ValueError("command time is not control-tick aligned")
        if tick != int(self._steps):
            raise RuntimeError("command tick differs from the environment step counter")
        raw = raw_command_at_tick(self.rolling_spec, tick)
        ground_height = float(self.loop.provider.ground_reference().height_m)
        clearance = RAW_WORLD_HEIGHT_M - ground_height
        self._controller.controller.bind_raw_command(
            forward_velocity_mps=raw["forward_velocity_mps"],
            yaw_rate_rps=0.0,
            control_time_s=float(time_s),
        )
        self.raw_callback_count += 1
        self.command_records.append({
            "tick": tick,
            "time_s": float(time_s),
            "forward_velocity_mps": raw["forward_velocity_mps"],
            "yaw_rate_rps": 0.0,
            "raw_world_height_m": RAW_WORLD_HEIGHT_M,
            "ground_height_m": ground_height,
            "motion_clearance_m": clearance,
            "prepared_world_height_m": None,
        })
        return D1MotionCommand(
            forward_velocity_mps=raw["forward_velocity_mps"],
            yaw_rate_rps=0.0,
            clearance_m=clearance,
        )

    def _prepare(self):
        result = super()._prepare()
        world_height = float(self.decision.world_command.base_height_m)
        if world_height != RAW_WORLD_HEIGHT_M:
            raise RuntimeError("prepared world height changed from the frozen 0.455 m")
        if self.command_records:
            self.command_records[-1]["prepared_world_height_m"] = world_height
        return result

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options:
            raise ValueError("rolling task does not accept reset options")
        if self._pending_spec is not None:
            self.rolling_spec = self._pending_spec
            self._pending_spec = None
        self.plant.validate_single_step()
        self.raw_callback_count = 0
        self.command_records = []
        observation, info = super().reset(seed=seed, options=options)
        metadata = self._episode_metadata
        metadata["requested_terrain_config"] = {
            "physical_obstacle_enabled": self.rolling_spec.obstacle_enabled,
            "physical_plant": "d1_single_step_15mm_box_or_plane",
            "heading_parent_terrain_layout": "flat",
        }
        metadata["rolling_episode_spec"] = self.rolling_spec.as_dict()
        metadata["terrain"] = dict(self.plant.collision_terrain_metadata)
        metadata["collision_terrain"] = dict(self.plant.collision_terrain_metadata)
        metadata["raw_world_height_m"] = RAW_WORLD_HEIGHT_M
        metadata["raw_schedule"] = [
            {"tick_start": 0, "tick_end": self.rolling_spec.forward_start_tick,
             "forward_velocity_mps": 0.0},
            {"tick_start": self.rolling_spec.forward_start_tick,
             "tick_end": self.rolling_spec.forward_stop_tick,
             "forward_velocity_mps": self.rolling_spec.forward_velocity_mps},
            {"tick_start": self.rolling_spec.forward_stop_tick, "tick_end": 1200,
             "forward_velocity_mps": 0.0},
        ]
        result = dict(info)
        result["episode_metadata"] = self.episode_metadata
        return observation, result


class D1RollingResidualEnv(gym.Wrapper):
    """A one-scalar policy view of `_RollingPhysicalEnv` and its true 8D plant."""

    task_schema = ROLLING_TASK_SCHEMA
    observation_schema = HEADING_OBSERVATION_SCHEMA
    reward_schema = HEADING_REWARD_SCHEMA
    action_schema = ROLLING_ACTION_SCHEMA
    action_mode = "shared_leg1_wheel_zero"
    policy_action_size = POLICY_ACTION_SIZE
    physical_action_size = PHYSICAL_ACTION_SIZE
    policy_to_physical_indices = (0, 0, 0, 0, None, None, None, None)

    def __init__(self, spec: RollingEpisodeSpec) -> None:
        super().__init__(_RollingPhysicalEnv(spec))
        self.action_space = spaces.Box(-1.0, 1.0, (POLICY_ACTION_SIZE,), dtype=np.float32)
        self.observation_space = self.env.observation_space
        self.physical_action_schema = self.env.physical_action_schema
        self.last_rolling_action: dict[str, Any] | None = None

    @property
    def rolling_spec(self) -> RollingEpisodeSpec:
        return self.env.rolling_spec

    @property
    def episode_metadata(self) -> dict[str, Any]:
        metadata = self.env.episode_metadata
        metadata.update({
            "task_schema": ROLLING_TASK_SCHEMA,
            "action_schema": ROLLING_ACTION_SCHEMA,
            "action_mode": self.action_mode,
            "policy_action_size": POLICY_ACTION_SIZE,
            "physical_action_size": PHYSICAL_ACTION_SIZE,
            "policy_to_physical_indices": list(self.policy_to_physical_indices),
            "rolling_task_definition": rolling_task_definition(),
            "heading_task_config_scope": "inner original eight-channel physical controller",
            "rolling_action_scope": "outer one-scalar actor and exact wheel-zero expansion",
        })
        return metadata

    def configure_next_episode(self, spec: RollingEpisodeSpec) -> None:
        self.env.configure_next_episode(spec)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.last_rolling_action = None
        observation, info = self.env.reset(seed=seed, options=options)
        result = dict(info)
        result["episode_metadata"] = self.episode_metadata
        return observation, result

    def step(self, action: Any):
        if not self.env._active:
            raise RuntimeError("reset must precede rolling residual step")
        decision = self.env.heading_decision
        if decision is None or not self.env.command_records:
            raise RuntimeError("rolling residual step lacks a prepared command")
        raw = self.env.command_records[-1]
        if raw["tick"] != decision.tick:
            raise RuntimeError("raw command and decision ticks differ")
        receipt = expand_policy_action(action, raw["forward_velocity_mps"])
        physical = np.asarray(receipt.physical_action, dtype=np.float64)
        observation, reward, terminated, truncated, info = self.env.step(physical)
        if not np.array_equal(np.asarray(info["applied_action"]), physical):
            raise RuntimeError("actual applied physical action differs from the actor expansion")
        self.last_rolling_action = receipt.as_dict()
        result = dict(info)
        result["rolling_action"] = dict(self.last_rolling_action)
        result["policy_action"] = np.asarray([receipt.policy_scalar], dtype=np.float64)
        result["physical_action"] = physical.copy()
        return observation, reward, terminated, truncated, result
