"""Flat, oldest-first history of observations already emitted by a D1 env.

This wrapper buffers observations; it neither reads simulator state nor adds or
removes sensor fields. Any privileged fields in the base observation stay
privileged. It never resets an episode implicitly.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

HISTORY_SCHEMA_VERSION = "v1"
STACKING_ORDER = "oldest_first"
RESET_PADDING = "repeat_initial_observation"


def _flat_float_box(space: Any, label: str) -> tuple[int, np.dtype]:
    if not isinstance(space, Box):
        raise TypeError(f"{label} space must be a Box")
    if len(space.shape) != 1 or space.shape[0] < 1:
        raise ValueError(f"{label} space must be a nonempty one-dimensional Box")
    dtype = np.dtype(space.dtype)
    if dtype.kind != "f":
        raise TypeError(f"{label} Box must have a floating-point dtype")
    return int(space.shape[0]), dtype


def _required_schema(env: gym.Env, name: str) -> str:
    try:
        value = env.get_wrapper_attr(name)
    except AttributeError as exc:
        raise AttributeError(f"D1ObservationHistory requires '{name}'") from exc
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


class D1ObservationHistory(gym.Wrapper):
    """Concatenate the last ``history_length`` frames, oldest to newest.

    Both spaces must be one-dimensional floating-point Boxes. Emitted frames
    must have the exact observation dtype and shape and contain finite values.
    Actions must be finite real numeric vectors of the declared shape; their
    values and dtype pass through without clipping or normalization. The base
    environment remains responsible for its action-bound semantics.

    Reset repeats the first frame in every slot. A failed base transition or an
    invalid emitted frame requires another explicit reset because the base may
    have advanced. Invalid actions are rejected before advancing the base.
    Returned observation arrays own their data and are safe to modify.
    """

    def __init__(self, env: gym.Env, history_length: int = 1):
        if isinstance(history_length, bool) or not isinstance(history_length, (int, np.integer)):
            raise TypeError("history_length must be an integer, not a boolean")
        history_length = int(history_length)
        if history_length < 1:
            raise ValueError("history_length must be at least 1")
        super().__init__(env)
        self._frame_dim, self._frame_dtype = _flat_float_box(env.observation_space, "observation")
        self._action_dim, _ = _flat_float_box(env.action_space, "action")
        self._history_length = history_length
        self._base_schema = _required_schema(env, "observation_schema")
        self._source_schema_value = _required_schema(env, "source_schema")
        self.observation_space = Box(
            low=np.tile(env.observation_space.low, history_length),
            high=np.tile(env.observation_space.high, history_length),
            dtype=self._frame_dtype,
        )
        self._history: list[np.ndarray] = []
        self._needs_reset = True

    @property
    def history_length(self) -> int:
        return self._history_length

    @property
    def observation_schema(self) -> str:
        if self._history_length == 1:
            return self._base_schema
        return (
            f"{self._base_schema}+history/{HISTORY_SCHEMA_VERSION}"
            f";length={self._history_length};order={STACKING_ORDER}"
            f";padding={RESET_PADDING}"
        )

    @property
    def source_schema(self) -> str:
        return self._source_schema_value

    @property
    def history_metadata(self) -> dict[str, Any]:
        """Fresh JSON-serializable description; does not claim new sensors."""
        return {
            "base_observation_schema": self._base_schema,
            "observation_schema": self.observation_schema,
            "source_schema": self._source_schema_value,
            "frame_dim": self._frame_dim,
            "frame_dtype": self._frame_dtype.name,
            "history_length": self._history_length,
            "observation_dim": self._frame_dim * self._history_length,
            "stacking_order": STACKING_ORDER,
            "reset_padding": RESET_PADDING,
        }

    def _check_action(self, action: Any) -> None:
        # Check Python sequences before NumPy can coerce mixed bool/float input.
        if isinstance(action, (list, tuple)) and any(
            isinstance(value, (bool, np.bool_)) for value in action
        ):
            raise TypeError("action must contain real numbers, not booleans")
        probe = np.asarray(action)
        if probe.dtype.kind not in ("f", "i", "u"):
            raise TypeError("action must contain real numeric values")
        if probe.shape != (self._action_dim,):
            raise ValueError(f"action must have shape ({self._action_dim},)")
        if not np.all(np.isfinite(probe)):
            raise ValueError("action must contain only finite values")

    def _check_observation(self, observation: Any) -> np.ndarray:
        if not isinstance(observation, np.ndarray):
            raise TypeError("observation must be a NumPy array")
        if observation.dtype != self._frame_dtype:
            raise TypeError(f"observation must have dtype {self._frame_dtype}")
        if observation.shape != (self._frame_dim,):
            raise ValueError(f"observation must have shape ({self._frame_dim},)")
        if not np.all(np.isfinite(observation)):
            raise ValueError("observation must contain only finite values")
        return observation

    def _stacked(self) -> np.ndarray:
        return np.concatenate(self._history)

    def reset(self, *, seed=None, options=None):
        self._history = []
        self._needs_reset = True
        observation, info = self.env.reset(seed=seed, options=options)
        frame = self._check_observation(observation)
        self._history = [frame.copy() for _ in range(self._history_length)]
        self._needs_reset = False
        return self._stacked(), info

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("D1ObservationHistory requires reset() before step()")
        self._check_action(action)
        # If the base raises or emits an invalid frame, this remains True.
        self._needs_reset = True
        observation, reward, terminated, truncated, info = self.env.step(action)
        frame = self._check_observation(observation)
        self._history.append(frame.copy())
        del self._history[0]
        self._needs_reset = bool(terminated) or bool(truncated)
        return self._stacked(), reward, terminated, truncated, info
