"""Gymnasium environment for residual reinforcement learning."""

from __future__ import annotations

from collections import deque
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .controllers import LinearMPCController, LQRController, TrackingCommand
from .model import ACTION_SIZE, STATE_SIZE, WheelLeggedPlant
from .rewards import calculate_reward

OBSERVATION_SIZE = 11
RESIDUAL_SCALE = np.array([18.0, 35.0], dtype=np.float64)


class WheelLeggedResidualEnv(gym.Env[np.ndarray, np.ndarray]):
    """Track velocity and height with a bounded residual over LQR or MPC.

    The agent never commands raw force.  Its normalized two-dimensional action is
    scaled and added to a stabilizing controller, which keeps exploration focused
    on model mismatch, delay, noise, and push recovery.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        baseline: str = "lqr",
        episode_seconds: float = 8.0,
        randomize: bool = True,
    ) -> None:
        super().__init__()
        if baseline not in {"lqr", "mpc"}:
            raise ValueError("baseline must be 'lqr' or 'mpc'")
        self.baseline_name = baseline
        self.randomize = randomize
        self.plant = WheelLeggedPlant()
        self.controller = (
            LQRController(self.plant)
            if baseline == "lqr"
            else LinearMPCController(self.plant)
        )
        self.max_steps = round(episode_seconds / self.plant.control_dt)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_SIZE,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -5.0, 5.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )

        self._step_count = 0
        self._scenario = "training"
        self._command = TrackingCommand()
        self._baseline_control = self.plant.equilibrium_control
        self._previous_residual = np.zeros(ACTION_SIZE)
        self._delay_queue: deque[np.ndarray] = deque()
        self._delay_steps = 0
        self._noise_scale = 0.0
        self._push_start = -1
        self._push_end = -1
        self._push_force_n = 0.0
        self._training_velocity = 0.0
        self._training_height = 0.0

    def _configure_episode(self, options: dict[str, Any]) -> np.ndarray:
        self._scenario = str(options.get("scenario", "training"))
        use_randomization = bool(options.get("randomize", self.randomize))

        if use_randomization:
            mass_scale = float(options.get("mass_scale", self.np_random.uniform(0.80, 1.25)))
            damping_scale = float(
                options.get("damping_scale", self.np_random.uniform(0.70, 1.40))
            )
            self._delay_steps = int(options.get("delay_steps", self.np_random.integers(0, 3)))
            self._noise_scale = float(
                options.get("sensor_noise", self.np_random.uniform(0.0, 1.0))
            )
        else:
            mass_scale = float(options.get("mass_scale", 1.0))
            damping_scale = float(options.get("damping_scale", 1.0))
            self._delay_steps = int(options.get("delay_steps", 0))
            self._noise_scale = float(options.get("sensor_noise", 0.0))

        self.plant.set_domain(mass_scale, damping_scale)
        initial_pitch = float(
            options.get(
                "initial_pitch",
                self.np_random.uniform(-0.09, 0.09) if use_randomization else 0.04,
            )
        )
        initial_state = np.zeros(STATE_SIZE, dtype=np.float64)
        initial_state[1] = initial_pitch
        if use_randomization:
            initial_state[4] = self.np_random.uniform(-0.08, 0.08)

        if self._scenario == "training":
            self._push_start = int(self.np_random.integers(70, max(71, self.max_steps - 70)))
            duration = int(self.np_random.integers(4, 10))
            self._push_end = self._push_start + duration
            magnitude = self.np_random.uniform(25.0, 55.0)
            self._push_force_n = float(magnitude * self.np_random.choice((-1.0, 1.0)))
            self._sample_training_command()
        elif self._scenario == "push":
            self._push_start = round(3.0 / self.plant.control_dt)
            self._push_end = self._push_start + round(0.12 / self.plant.control_dt)
            self._push_force_n = float(options.get("push_force_n", 55.0))
        elif self._scenario == "mismatch_delay":
            self._push_start = round(4.2 / self.plant.control_dt)
            self._push_end = self._push_start + round(0.10 / self.plant.control_dt)
            self._push_force_n = float(options.get("push_force_n", -45.0))
        else:
            self._push_start = self._push_end = -1
            self._push_force_n = 0.0
        return initial_state

    def _sample_training_command(self) -> None:
        self._training_velocity = float(self.np_random.uniform(-1.0, 1.0))
        self._training_height = float(self.np_random.uniform(-0.07, 0.07))

    def _command_at_step(self) -> TrackingCommand:
        time_s = self._step_count * self.plant.control_dt
        if self._scenario == "training":
            if self._step_count > 0 and self._step_count % 100 == 0:
                self._sample_training_command()
            return TrackingCommand(self._training_velocity, self._training_height)
        if time_s < 1.0:
            return TrackingCommand(0.0, 0.0)
        if time_s < 3.0:
            return TrackingCommand(0.80, 0.045 if time_s >= 2.0 else 0.0)
        if time_s < 5.5:
            return TrackingCommand(-0.50, -0.035 if time_s >= 4.0 else 0.045)
        return TrackingCommand(0.35, 0.0)

    def _measure_state(self) -> np.ndarray:
        state = self.plant.state()
        if self._noise_scale == 0.0:
            return state
        standard_deviation = self._noise_scale * np.array(
            [0.0005, 0.0020, 0.0005, 0.010, 0.012, 0.006]
        )
        return state + self.np_random.normal(0.0, standard_deviation)

    def _observation(self, measured_state: np.ndarray) -> np.ndarray:
        equilibrium = self.plant.equilibrium_control
        observation = np.array(
            [
                measured_state[3] / 1.5,
                measured_state[1] / 0.5,
                measured_state[4] / 4.0,
                (measured_state[2] - self._command.leg_extension_m) / 0.12,
                measured_state[5] / 1.0,
                self._command.velocity_mps / 1.0,
                self._command.leg_extension_m / 0.08,
                self._baseline_control[0] / self.plant.limits.wheel_force_n,
                (self._baseline_control[1] - equilibrium[1]) / 100.0,
                self._previous_residual[0],
                self._previous_residual[1],
            ],
            dtype=np.float32,
        )
        return np.clip(observation, -5.0, 5.0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = {} if options is None else dict(options)
        self._step_count = 0
        initial_state = self._configure_episode(options)
        self.plant.reset(initial_state)
        self.controller.reset(initial_state)
        self._command = self._command_at_step()
        measured_state = self._measure_state()
        self._baseline_control = self.controller.compute(measured_state, self._command)
        self._previous_residual[:] = 0.0
        self._delay_queue = deque(
            [self.plant.equilibrium_control.copy() for _ in range(self._delay_steps)]
        )
        info = self._info(self.plant.state(), self.plant.equilibrium_control, 0.0)
        return self._observation(measured_state), info

    def _info(
        self, state: np.ndarray, applied_control: np.ndarray, push_force_n: float
    ) -> dict[str, Any]:
        return {
            "state": state.astype(np.float64, copy=True),
            "control": applied_control.astype(np.float64, copy=True),
            "command_velocity_mps": self._command.velocity_mps,
            "command_leg_extension_m": self._command.leg_extension_m,
            "push_force_n": push_force_n,
            "baseline": self.baseline_name,
            "delay_steps": self._delay_steps,
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        normalized_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if normalized_action.shape != (ACTION_SIZE,):
            raise ValueError(f"action must have shape ({ACTION_SIZE},)")
        desired_control = self._baseline_control + normalized_action * RESIDUAL_SCALE
        desired_control = np.clip(
            desired_control, self.plant.actuator_low, self.plant.actuator_high
        )
        self._delay_queue.append(desired_control)
        applied_control = self._delay_queue.popleft()

        push_force = (
            self._push_force_n
            if self._push_start <= self._step_count < self._push_end
            else 0.0
        )
        state = self.plant.step(applied_control, push_force)
        self._step_count += 1
        self._command = self._command_at_step()
        measured_state = self._measure_state()
        self._baseline_control = self.controller.compute(measured_state, self._command)
        self._previous_residual = normalized_action.copy()

        terminated = bool(
            not np.all(np.isfinite(state))
            or abs(state[1]) >= self.plant.limits.fall_pitch_rad
        )
        truncated = bool(self._step_count >= self.max_steps)
        reward_terms = calculate_reward(
            state,
            self._command.velocity_mps,
            self._command.leg_extension_m,
            normalized_action,
            terminated,
        )
        reward = reward_terms.total
        info = self._info(state, applied_control, push_force)
        info["is_success"] = bool(truncated and not terminated)
        info["reward_terms"] = reward_terms.as_dict()
        return self._observation(measured_state), reward, terminated, truncated, info

    def close(self) -> None:
        self._delay_queue.clear()
