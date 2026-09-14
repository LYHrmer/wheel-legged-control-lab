"""Terrain-condition curriculum wrapper for D1 locomotion PPO training.

This module adapts the frozen ``D1LocomotionEnv`` to three terrain conditions
(``flat``, ``mixed``, ``curriculum``) without touching observations, actions,
rewards, controller limits or physics.  Difficulty is selected only at reset;
an episode that has already begun keeps its terrain until it ends.

Nothing in this file trains, saves or evaluates a policy: a separate local
runner owns the SB3 PPO protocol.  No frozen source is modified.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, replace
from numbers import Real
from typing import Any

import gymnasium as gym
import numpy as np

from wheel_legged_control.d1.locomotion_commands import D1CommandSchedule
from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv
from wheel_legged_control.d1.locomotion_terrain import (
    D1LocomotionTerrainConfig,
    locomotion_terrain_configs,
)

CONDITIONS = ("flat", "mixed", "curriculum")
MAX_LEVEL = 3
TERRAIN_COUNT = 4
_EPISODE_SEED_LIMIT = 2**31 - 1


def _validated_int(value: Any, name: str, low: int, high: int) -> int:
    """Return ``value`` as an int, rejecting bools, floats and out-of-range."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-boolean integer, got {type(value).__name__}")
    number = int(value)
    if not low <= number <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}], got {number}")
    return number


def _finite_float(value: Any) -> float | None:
    """Return a JSON-native finite float, or None when unavailable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def scaled_training_terrain(level: int, terrain_index: int) -> D1LocomotionTerrainConfig:
    """Scale one training terrain to a difficulty level in ``0..3``.

    Level 0 is the default flat road, independent of the terrain index. For
    other levels the nominal amplitudes are scaled by ``level / 3``; wavelengths,
    phases, layout and schema are untouched, so the frozen construction limits
    still validate the result.
    """
    lvl = _validated_int(level, "level", 0, MAX_LEVEL)
    index = _validated_int(terrain_index, "terrain_index", 0, TERRAIN_COUNT - 1)
    configs = tuple(locomotion_terrain_configs("train"))
    if len(configs) < TERRAIN_COUNT:
        raise RuntimeError(
            f"expected at least {TERRAIN_COUNT} training terrains, got {len(configs)}"
        )
    if lvl == 0:
        return D1LocomotionTerrainConfig()
    original = configs[index]
    scale = lvl / float(MAX_LEVEL)
    return replace(
        original,
        slope_deg=original.slope_deg * scale,
        cross_slope_deg=original.cross_slope_deg * scale,
        ripple_amplitude_m=original.ripple_amplitude_m * scale,
        step_height_m=original.step_height_m * scale,
    )


class D1CourseCurriculumEnv(gym.Wrapper):
    """Gymnasium wrapper selecting terrain difficulty at every reset."""

    def __init__(
        self,
        *,
        condition: str,
        total_steps: int,
        seed: int,
        episode_seconds: float = 12.0,
        command_source: Callable[..., Any] | None = None,
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
        if isinstance(total_steps, bool) or not isinstance(total_steps, (int, np.integer)):
            raise TypeError("total_steps must be a non-boolean integer")
        if int(total_steps) <= 0:
            raise ValueError("total_steps must be positive")
        if isinstance(episode_seconds, (bool, np.bool_)) or not isinstance(episode_seconds, Real):
            raise TypeError("episode_seconds must be a non-boolean real number")
        seconds = float(episode_seconds)
        if (
            not math.isfinite(seconds)
            or seconds < 0.01
            or not np.isclose(seconds / 0.01, round(seconds / 0.01), rtol=0, atol=1e-8)
        ):
            raise ValueError("episode_seconds must be a positive multiple of 0.01")
        if command_source is not None and not callable(command_source):
            raise TypeError("command_source must be callable or None")

        self._condition = condition
        self._total_steps = int(total_steps)
        self._episode_seconds = seconds
        self._command_source = command_source
        self._seed = _validated_int(seed, "seed", 0, 2**63 - 1)

        self.total_transitions = 0
        self.episode_records: list[dict[str, Any]] = []
        self.current_level: int | None = None
        self.current_terrain_index: int | None = None
        self.current_episode_seed: int | None = None

        self._closed = False
        self._started = False
        self._episode_active = False
        self._step_failed = False
        self._episode_counter = 0
        self._episode_transitions = 0
        self._episode_reward_sum = 0.0
        self._episode_start_transitions = 0
        self._reset_x: float | None = None
        self._last_x: float | None = None
        self._min_clearance: float | None = None
        self._max_abs_tilt: float | None = None
        self._last_exposure: dict[str, Any] | None = None

        self._spawn_streams(self._seed)
        terrain = scaled_training_terrain(0, 0)
        self._terrain_config = terrain
        base_env = self._build_env(terrain)
        super().__init__(base_env)
        self._reference_observation_space = base_env.observation_space
        self._reference_action_space = base_env.action_space

    # ---------------------------------------------------------------- setup

    def _spawn_streams(self, seed: int) -> None:
        level_seq, terrain_seq, episode_seq = np.random.SeedSequence(seed).spawn(3)
        self._level_rng = np.random.default_rng(level_seq)
        self._terrain_rng = np.random.default_rng(terrain_seq)
        self._episode_seed_rng = np.random.default_rng(episode_seq)

    def _build_env(self, terrain: D1LocomotionTerrainConfig) -> D1LocomotionEnv:
        return D1LocomotionEnv(
            baseline="wheel_leg",
            action_mode="independent8",
            episode_seconds=self._episode_seconds,
            terrain=terrain,
            command_source=self._command_source,
        )

    def _select_level(self, level_draw: int) -> int:
        if self._condition == "flat":
            return 0
        if self._condition == "mixed":
            return level_draw
        return min(MAX_LEVEL, (4 * self.total_transitions) // self._total_steps)

    # ---------------------------------------------------------------- reset

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if self._closed:
            raise RuntimeError("D1CourseCurriculumEnv is closed; reset is not allowed")
        if seed is not None:
            if self._started:
                raise ValueError("reset(seed=...) is rejected once an episode has started")
            seed = _validated_int(seed, "seed", 0, 2**63 - 1)
        base_options: dict[str, Any] | None = None
        if options is not None:
            if not isinstance(options, dict):
                raise TypeError("options must be a dict or None")
            unexpected = set(options) - {"schedule"}
            if unexpected:
                raise ValueError("only schedule may be supplied as a reset option")
            if "schedule" in options:
                schedule = options["schedule"]
                if schedule is not None and (
                    not isinstance(schedule, D1CommandSchedule)
                    or not np.isclose(
                        schedule.duration_s, self._episode_seconds, rtol=0, atol=1e-10
                    )
                ):
                    raise ValueError("schedule duration must match the episode")
                if schedule is not None and self._command_source is not None:
                    raise ValueError("external commands cannot also consume a schedule")
                base_options = {"schedule": schedule}

        if self._episode_active:
            self._finalize_episode("manual_reset", completed=False)
        if seed is not None:
            self._seed = seed
            self._spawn_streams(seed)

        level_draw = int(self._level_rng.integers(0, MAX_LEVEL + 1))
        terrain_index = int(self._terrain_rng.integers(0, TERRAIN_COUNT))
        episode_seed = int(self._episode_seed_rng.integers(0, _EPISODE_SEED_LIMIT))
        level = self._select_level(level_draw)
        terrain = scaled_training_terrain(level, terrain_index)

        if terrain != self._terrain_config:
            old_env = self.env
            new_env = self._build_env(terrain)
            if new_env.observation_space != self._reference_observation_space:
                new_env.close()
                raise RuntimeError("rebuilt environment changed the observation space")
            if new_env.action_space != self._reference_action_space:
                new_env.close()
                raise RuntimeError("rebuilt environment changed the action space")
            self.env = new_env
            self.observation_space = new_env.observation_space
            self.action_space = new_env.action_space
            self._terrain_config = terrain
            old_env.close()

        observation, info = self.env.reset(seed=episode_seed, options=base_options)

        self.current_level = level
        self.current_terrain_index = terrain_index
        self.current_episode_seed = episode_seed
        self._started = True
        self._episode_active = True
        self._step_failed = False
        self._episode_transitions = 0
        self._episode_reward_sum = 0.0
        self._episode_start_transitions = self.total_transitions
        self._reset_x = _finite_float(float(np.asarray(self.env.plant.data.qpos[:3])[0]))
        self._last_x = None
        self._min_clearance = None
        self._max_abs_tilt = None
        self._last_exposure = None

        out_info = dict(info)
        out_info["course_curriculum"] = {
            "condition": self._condition,
            "level": level,
            "terrain_index": terrain_index,
            "episode_seed": episode_seed,
            "transitions_before_episode": self._episode_start_transitions,
        }
        return observation, out_info

    # ----------------------------------------------------------------- step

    def step(self, action):
        if self._closed:
            raise RuntimeError("D1CourseCurriculumEnv is closed; step is not allowed")
        if not self._episode_active or self._step_failed:
            raise RuntimeError("step() requires an active episode; call reset() first")
        # Reject malformed actions before the base can consume its decision.
        # Valid actions pass through unchanged; clipping still belongs to D1.
        probe = np.asarray(action, dtype=np.float64)
        if probe.shape != self.action_space.shape or not np.isfinite(probe).all():
            raise ValueError("action must match the finite action schema (policy dimensions)")
        try:
            observation, reward, terminated, truncated, info = self.env.step(action)
        except Exception:
            # Failed physics may have partly advanced. Do not retry that state,
            # invent a completed transition, or hide the original exception.
            self._step_failed = True
            raise
        self._episode_transitions += 1
        self.total_transitions += 1
        self._episode_reward_sum += float(reward)
        self._absorb_step_truth(info)

        out_info = dict(info)
        out_info["course_curriculum"] = {
            "condition": self._condition,
            "level": self.current_level,
            "terrain_index": self.current_terrain_index,
            "episode_seed": self.current_episode_seed,
            "episode_index": self._episode_counter,
            "transitions_in_episode": self._episode_transitions,
            "total_transitions": self.total_transitions,
        }
        if terminated or truncated:
            self._finalize_episode(info.get("terminal_reason"), completed=True)
        return observation, reward, terminated, truncated, out_info

    def _absorb_step_truth(self, info: dict[str, Any]) -> None:
        metrics = info.get("metrics")
        if isinstance(metrics, dict):
            clearance = _finite_float(metrics.get("clearance_m"))
            if clearance is not None:
                self._min_clearance = (
                    clearance
                    if self._min_clearance is None
                    else min(self._min_clearance, clearance)
                )
        exposure = info.get("terrain_exposure")
        if isinstance(exposure, dict):
            self._last_exposure = dict(exposure)
        transition = self.env.last_transition
        if transition is None:
            return
        position = np.asarray(transition.truth.base_position, dtype=float)
        x_value = _finite_float(float(position[0]))
        if x_value is not None:
            self._last_x = x_value
        rpy = np.asarray(transition.truth.base_rpy, dtype=float)
        tilt = _finite_float(float(np.max(np.abs(rpy[:2]))))
        if tilt is not None:
            self._max_abs_tilt = (
                tilt if self._max_abs_tilt is None else max(self._max_abs_tilt, tilt)
            )

    # ------------------------------------------------------------ recording

    def _finalize_episode(self, reason: Any, *, completed: bool) -> None:
        self._episode_active = False
        self._episode_counter += 1
        if self._episode_transitions == 0:
            return
        progress: float | None = None
        if self._last_x is not None and self._reset_x is not None:
            progress = _finite_float(self._last_x - self._reset_x)
        terminal_reason = reason if isinstance(reason, str) or reason is None else str(reason)
        self.episode_records.append(
            {
                "episode_index": self._episode_counter - 1,
                "condition": self._condition,
                "level": self.current_level,
                "terrain_index": self.current_terrain_index,
                "episode_seed": self.current_episode_seed,
                "terrain_parameters": asdict(self._terrain_config),
                "transitions": self._episode_transitions,
                "start_total_transitions": self._episode_start_transitions,
                "end_total_transitions": self.total_transitions,
                "reward_sum": _finite_float(self._episode_reward_sum),
                "terminal_reason": terminal_reason,
                "completed": bool(completed),
                "forward_progress_m": progress,
                "last_terrain_exposure": (
                    dict(self._last_exposure) if self._last_exposure is not None else None
                ),
                "minimum_clearance_m": self._min_clearance,
                "maximum_abs_roll_pitch_rad": self._max_abs_tilt,
            }
        )

    # ---------------------------------------------------------------- close

    def close(self) -> None:
        if self._closed:
            return
        if self._episode_active:
            self._finalize_episode("budget_cut", completed=False)
        self._closed = True
        self.env.close()

    # ------------------------------------------------------------ read-only

    @property
    def condition(self) -> str:
        return self._condition

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def seed_value(self) -> int:
        return self._seed

    @property
    def terrain_config(self) -> D1LocomotionTerrainConfig:
        return self._terrain_config

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def episode_active(self) -> bool:
        return self._episode_active
