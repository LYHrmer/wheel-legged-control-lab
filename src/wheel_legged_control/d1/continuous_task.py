"""A continuous D1 route: one static road, one reset, and 45 seconds of commands.

The legacy 44-value layout supports explicitly labeled oracle-v2 task transfer.
The command45 layout appends commanded yaw rate and has separate oracle/sensor
schemas for policies trained on the complete continuous task.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .control_primitives import YawRatePI, terrain_normal_to_rpy
from .controllers import D1Command
from .env import encode_d1_observation
from .hierarchical import D1LQRVMCController
from .model import D1Plant
from .state_estimation import D1MujocoTruthStateSource, D1StateEstimate
from .training_terrain import TrainingGroundReference, TrainingTerrainConfig

TASK_SCHEMA = "d1-continuous-static-road-v1"
OBSERVATION_SCHEMA = "d1-terrain-tracking-oracle-v2"
ACTION_SCHEMA = "d1-terrain-residual-quarter-v1"
CONTROL_SCHEMA = "d1-lqr-vmc-local-tangent-yaw-pi-v1"
SENSOR_CONTROL_SCHEMA = "d1-lqr-vmc-sensor-world-slopes-yaw-pi-v2"
RESIDUAL_SCALE_N = np.asarray((11.25, 20.0))
# Start/end times are specified on the 45-second template. Other durations
# rescale time and commands inversely, preserving the commanded route length.
STAGES = (
    ("settle", 0.0, 2.0, 0.0, 0.0),
    ("cruise_bumps", 2.0, 10.0, 0.25, 0.0),
    ("stop_one", 10.0, 13.0, 0.0, 0.0),
    ("climb_and_descend", 13.0, 23.0, 0.25, 0.0),
    ("slow_exit", 23.0, 29.0, 0.16, 0.0),
    ("turn_left", 29.0, 35.0, 0.12, 0.08),
    ("turn_right", 35.0, 41.0, 0.12, -0.08),
    ("stop_final", 41.0, 45.0, 0.0, 0.0),
)


def _smoothstep(value: np.ndarray | float) -> np.ndarray:
    t = np.clip(value, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def road_height(x: np.ndarray | float) -> np.ndarray:
    """C1 analytic design sampled once into the actual collision heightfield.

    Bumps have 5 mm amplitude and 0.8 m wavelength. Ramp grade rises smoothly
    to 4 degrees, with a 0.4 m plateau between ascending/descending branches.
    Smooth edge windows avoid steps when entering/leaving a terrain section.
    """

    x = np.asarray(x, dtype=np.float64)
    bump_window = _smoothstep((x + 3.0) / 0.2) * _smoothstep((-1.5 - x) / 0.2)
    bumps = 0.005 * bump_window * np.sin(2.0 * np.pi * (x + 3.0) / 0.8)

    def smooth_ramp_integral(start: float, end: float) -> np.ndarray:
        width = 0.2

        def integral(value: np.ndarray) -> np.ndarray:
            t = np.clip(value / width, 0.0, 1.0)
            return width * (t**3 - 0.5 * t**4) + np.maximum(value - width, 0.0)

        return integral(x - start) - integral(x - (end - width))

    slope = math.tan(math.radians(4.0))
    return bumps + slope * (smooth_ramp_integral(-1.1, 0.1) - smooth_ramp_integral(0.5, 1.7))


def road_section(x: float) -> str:
    if -3.0 <= x <= -1.5:
        return "bumps"
    if -1.1 <= x < 0.1:
        return "ramp_up"
    if 0.1 <= x < 0.5:
        return "plateau"
    if 0.5 <= x <= 1.7:
        return "ramp_down"
    return "flat"


def install_static_road(plant: D1Plant) -> dict[str, Any]:
    """Reuse an allocated training heightfield, without changing its dimensions."""

    model = plant.model
    hfield_id = int(model.geom_dataid[plant.floor_geom_id])
    nrow, ncol = int(model.hfield_nrow[hfield_id]), int(model.hfield_ncol[hfield_id])
    size = model.hfield_size[hfield_id].copy()
    offset = float(model.geom_pos[plant.floor_geom_id, 2])
    x = np.linspace(-size[0], size[0], ncol)
    heights = road_height(x)
    normalized = (heights - offset) / size[2]
    if np.any(normalized < 0) or np.any(normalized > 1):
        raise ValueError("road exceeds allocated heightfield elevation bounds")
    address = int(model.hfield_adr[hfield_id])
    model.hfield_data[address : address + nrow * ncol].reshape(nrow, ncol)[:] = normalized
    return {
        "schema": "static-composite-heightfield-v1",
        "size": size.tolist(),
        "nrow": nrow,
        "ncol": ncol,
        "geom_z_offset_m": offset,
        "bounds_x_m": [-float(size[0]), float(size[0])],
        "bounds_y_m": [-float(size[1]), float(size[1])],
        "bump_amplitude_m": 0.005,
        "bump_wavelength_m": 0.8,
        "maximum_ramp_grade_deg": 4.0,
        "switching": "static spatial terrain; never rebuilt or switched during the episode",
    }


def scheduled_command(time_s: float, duration_s: float) -> tuple[str, float, float, float]:
    """Return stage, forward speed, yaw rate, and elapsed stage time."""

    template_time = time_s * 45.0 / duration_s
    index = next((i for i, stage in enumerate(STAGES) if template_time < stage[2]), len(STAGES) - 1)
    name, start, _, velocity, yaw_rate = STAGES[index]
    previous = STAGES[max(0, index - 1)]
    blend = float(_smoothstep((template_time - start) / 0.6))
    speed_scale = 45.0 / duration_s
    return (
        name,
        (previous[3] + blend * (velocity - previous[3])) * speed_scale,
        (previous[4] + blend * (yaw_rate - previous[4])) * speed_scale,
        (template_time - start) / speed_scale,
    )


@dataclass(frozen=True)
class ContinuousTaskConfig:
    duration_s: float = 45.0
    ground_reference_mode: str = "oracle"
    observation_layout: str = "legacy44"
    yaw_controller: str = "pi"
    yaw_kp: float = 2.0
    yaw_ki: float = 3.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.duration_s, bool)
            or not isinstance(self.duration_s, Real)
            or not np.isfinite(self.duration_s)
            or not 30 <= self.duration_s <= 60
        ):
            raise ValueError("duration_s must lie in [30, 60]")
        if self.ground_reference_mode not in ("oracle", "estimated"):
            raise ValueError("ground_reference_mode must be oracle or estimated")
        if self.observation_layout not in ("legacy44", "command45"):
            raise ValueError("observation_layout must be legacy44 or command45")
        if self.yaw_controller not in ("legacy-p", "pi"):
            raise ValueError("yaw_controller must be legacy-p or pi")
        for name in ("yaw_kp", "yaw_ki"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 5:
                raise ValueError(f"{name} must be finite and lie in [0, 5]")


class D1ContinuousTask(gym.Env):
    """Full-step task with a replaceable state/ground estimate and explicit truth audit.

    ``state_source_factory(plant)`` returns an object with reset(seed=...) and
    read(). Estimated ground mode also requires ``ground_reference(state)``;
    its world-axis height/pitch/roll must come from measured state, not the map.
    No trained-policy compatibility is claimed for a different state source.
    """

    observation_schema = OBSERVATION_SCHEMA
    action_schema = ACTION_SCHEMA
    control_schema = CONTROL_SCHEMA
    reward_schema = "d1-continuous-task-reward-v1"

    def __init__(
        self,
        config: ContinuousTaskConfig | None = None,
        *,
        state_source_factory: Callable | None = None,
        ground_reference: Callable[[D1StateEstimate], TrainingGroundReference] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ContinuousTaskConfig()
        if self.config.ground_reference_mode == "estimated" and state_source_factory is None:
            raise ValueError("estimated mode requires explicit measured state and ground providers")
        self.plant = D1Plant(training_terrain=TrainingTerrainConfig())
        self.road = install_static_road(self.plant)
        self.controller = D1LQRVMCController(self.plant)
        self.controller.low_level.yaw_rate_gain = (
            self.config.yaw_kp if self.config.yaw_controller == "legacy-p" else 0.0
        )
        self.yaw_pi = YawRatePI(self.config.yaw_kp, self.config.yaw_ki)
        if self.config.yaw_controller == "legacy-p":
            self.control_schema = "d1-lqr-vmc-local-tangent-v2"
        self.source = (state_source_factory or D1MujocoTruthStateSource)(self.plant)
        self._estimated_ground = ground_reference or getattr(self.source, "ground_reference", None)
        if self.config.ground_reference_mode == "estimated":
            if self._estimated_ground is None:
                raise ValueError("estimated mode requires a measured ground provider")
            self.observation_schema = "d1-continuous-estimated-layout44-v1"
            if self.config.yaw_controller == "pi":
                self.control_schema = SENSOR_CONTROL_SCHEMA
        if self.config.observation_layout == "command45":
            source_name = "sensor" if self.config.ground_reference_mode == "estimated" else "oracle"
            self.observation_schema = f"d1-continuous-{source_name}-command45-v1"
        self._truth = D1MujocoTruthStateSource(self.plant)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        dimension = 45 if self.config.observation_layout == "command45" else 44
        self.observation_space = spaces.Box(-5.0, 5.0, (dimension,), dtype=np.float32)
        self.max_steps = round(self.config.duration_s / self.plant.control_dt)
        self.reset()

    def _control_ground(self) -> TrainingGroundReference:
        if self.config.ground_reference_mode == "estimated":
            return self._estimated_ground(self.state)
        x, y = self.state.base_position[:2]
        return self._truth_ground(x, y)

    def _truth_ground(self, x: float, y: float) -> TrainingGroundReference:
        return self.plant.training_ground_reference(
            float(np.clip(x, *self.road["bounds_x_m"])),
            float(np.clip(y, *self.road["bounds_y_m"])),
        )

    def _command(self) -> D1Command:
        _, velocity, yaw_rate, _ = scheduled_command(
            self.steps * self.plant.control_dt, self.config.duration_s
        )
        ground = self._control_ground()
        # The measured support plane publishes independent world-axis tilt
        # angles. They are not the roll/pitch of a composed Euler rotation.
        slope_x = -math.tan(ground.pitch_rad)
        slope_y = math.tan(ground.roll_rad)
        if self.config.ground_reference_mode == "oracle":
            # Preserve the historical oracle convention (our map has roll=0).
            slope_y /= math.cos(ground.pitch_rad)
        roll, pitch = terrain_normal_to_rpy(slope_x, slope_y, float(self.state.base_rpy[2]))
        return D1Command(velocity, yaw_rate, ground.height_m + 0.455, roll, pitch)

    def observation(self) -> np.ndarray:
        command = self._command()
        base = encode_d1_observation(
            state=self.state,
            command=command,
            baseline_longitudinal_force_n=self.controller.last_longitudinal_force_n,
            previous_applied_action=self.previous_action,
        )
        observation = np.r_[base, command.pitch_rad, command.roll_rad]
        if self.config.observation_layout == "command45":
            observation = np.r_[observation, command.yaw_rate_rps]
        return np.clip(observation, -5, 5).astype(np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("continuous-task reset does not accept implicit task overrides")
        self.plant.reset(base_position=np.asarray((-3.8, 0.0, 0.455)))
        self.controller.reset()
        self.yaw_pi.reset()
        self.state = self.source.reset(seed=seed)
        self._truth.reset(seed=seed)
        self.steps = 0
        self.previous_action = np.zeros(2)
        self.reference_xy = np.asarray((-3.8, 0.0))
        self.reference_yaw = 0.0
        self._done = False
        return self.observation(), {"task_schema": TASK_SCHEMA}

    def step(self, action):
        if self._done:
            raise RuntimeError("episode ended; no hidden automatic reset is performed")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("action must contain two finite values")
        applied = np.clip(action, -1, 1)
        command = self._command()
        stage, _, _, stage_time = scheduled_command(
            self.steps * self.plant.control_dt, self.config.duration_s
        )
        torque = self.controller.compute(
            command, self.state, residual_force_n=applied * RESIDUAL_SCALE_N
        )
        yaw_torque = 0.0
        if self.config.yaw_controller == "pi":
            yaw_torque = self.yaw_pi.compute(
                command.yaw_rate_rps - self.state.base_angular_velocity_body[2],
                self.plant.control_dt,
            )
            torque[[3, 11]] -= yaw_torque
            torque[[7, 15]] += yaw_torque
            torque = np.clip(
                torque, -self.plant.actuator_torque_limit_nm, self.plant.actuator_torque_limit_nm
            )
        self.plant.step(torque)
        self.state = self.source.read()
        truth = self._truth.read()
        dt = self.plant.control_dt
        heading_midpoint = self.reference_yaw + 0.5 * command.yaw_rate_rps * dt
        self.reference_xy += (
            dt
            * command.forward_velocity_mps
            * np.asarray((math.cos(heading_midpoint), math.sin(heading_midpoint)))
        )
        self.reference_yaw += command.yaw_rate_rps * dt
        self.steps += 1
        self.previous_action = applied.copy()
        x, y, z = truth.base_position
        ground = self._truth_ground(x, y)
        linear, angular = truth.base_velocity(local=True)
        yaw = float(truth.base_rpy[2])
        yaw_error = math.atan2(
            math.sin(yaw - self.reference_yaw), math.cos(yaw - self.reference_yaw)
        )
        position_error = truth.base_position[:2] - self.reference_xy
        lateral_error = float(
            position_error
            @ np.asarray((-math.sin(self.reference_yaw), math.cos(self.reference_yaw)))
        )
        longitudinal_error = float(
            position_error
            @ np.asarray((math.cos(self.reference_yaw), math.sin(self.reference_yaw)))
        )
        reason = "ongoing"
        if (
            not np.isfinite(self.plant.data.qpos).all()
            or not np.isfinite(self.plant.data.qvel).all()
        ):
            reason = "nonfinite"
        elif abs(x) > self.road["bounds_x_m"][1] - 0.5 or abs(y) > self.road["bounds_y_m"][1] - 0.5:
            reason = "map_boundary"
        elif z - ground.height_m < 0.22 or max(abs(truth.base_rpy[:2])) > 0.85:
            reason = "fallen"
        terminated = reason != "ongoing"
        truncated = self.steps >= self.max_steps and not terminated
        if truncated:
            reason = "time_limit"
        self._done = terminated or truncated
        info = {
            "step": self.steps,
            "time_s": self.steps * dt,
            "stage": stage,
            "stage_time_s": stage_time + dt,
            "terrain_section": road_section(float(x)),
            "ground_height_m": ground.height_m,
            "ground_pitch_rad": ground.pitch_rad,
            "x_m": float(x),
            "y_m": float(y),
            "z_m": float(z),
            "command_velocity_mps": command.forward_velocity_mps,
            "command_yaw_rate_rps": command.yaw_rate_rps,
            "velocity_mps": float(linear[0]),
            "lateral_velocity_mps": float(linear[1]),
            "yaw_rate_rps": float(angular[2]),
            "velocity_error_mps": float(linear[0] - command.forward_velocity_mps),
            "yaw_rate_error_rps": float(angular[2] - command.yaw_rate_rps),
            "yaw_rad": yaw,
            "yaw_error_rad": yaw_error,
            "lateral_path_error_m": lateral_error,
            "longitudinal_path_error_m": longitudinal_error,
            "reference_x_m": float(self.reference_xy[0]),
            "reference_y_m": float(self.reference_xy[1]),
            "reference_yaw_rad": self.reference_yaw,
            "clearance_m": float(z - ground.height_m),
            "clearance_error_m": float(z - ground.height_m - 0.455),
            "command_pitch_rad": command.pitch_rad,
            "command_roll_rad": command.roll_rad,
            "action_longitudinal": float(applied[0]),
            "action_vertical": float(applied[1]),
            "residual_longitudinal_n": float(applied[0] * RESIDUAL_SCALE_N[0]),
            "residual_vertical_n": float(applied[1] * RESIDUAL_SCALE_N[1]),
            "yaw_differential_torque_nm": yaw_torque,
            "yaw_integral_nm": self.yaw_pi.integral_nm,
            "torque_saturation_fraction": float(
                np.mean(np.abs(torque) >= self.plant.actuator_torque_limit_nm - 1e-9)
            ),
            "terminated": int(terminated),
            "truncated": int(truncated),
            "termination_reason": reason,
            "ground_reference_mode": self.config.ground_reference_mode,
        }
        reward = float(
            math.exp(-((info["velocity_error_mps"] / 0.2) ** 2))
            + math.exp(-((info["yaw_rate_error_rps"] / 0.15) ** 2))
            - (info["clearance_error_m"] / 0.08) ** 2
            - 5 * terminated
        )
        return self.observation(), reward, terminated, truncated, info


def summarize_task(rows: list[dict]) -> dict[str, Any]:
    """Separate completion from command-tracking quality; steps are not replicates."""

    if not rows:
        raise ValueError("cannot summarize an empty task")
    summaries = []
    for name, *_ in STAGES:
        stage = [row for row in rows if row["stage"] == name]
        steady = [row for row in stage if row["stage_time_s"] >= 1.0]
        if not stage:
            summaries.append({"stage": name, "completed": False, "quality_pass": False})
            continue
        selected = steady or stage
        metrics = {
            field: float(np.sqrt(np.mean([row[field] ** 2 for row in selected])))
            for field in (
                "velocity_error_mps",
                "lateral_velocity_mps",
                "yaw_rate_error_rps",
                "clearance_error_m",
            )
        }
        quality = (
            metrics["velocity_error_mps"] <= 0.14
            and metrics["lateral_velocity_mps"] <= 0.10
            and metrics["yaw_rate_error_rps"] <= 0.06
            and metrics["clearance_error_m"] <= 0.025
        )
        yaw_change = stage[-1]["yaw_rad"] - stage[0]["yaw_rad"]
        commanded_yaw_change = stage[-1]["reference_yaw_rad"] - stage[0]["reference_yaw_rad"]
        turn_progress = (
            yaw_change / commanded_yaw_change if abs(commanded_yaw_change) > 0.1 else None
        )
        if name in ("turn_left", "turn_right"):
            quality = quality and turn_progress is not None and 0.25 <= turn_progress <= 1.75
        summaries.append(
            {
                "stage": name,
                "steps": len(stage),
                "steady_steps": len(steady),
                "completed": not bool(stage[-1]["terminated"]),
                "quality_pass": quality,
                "steady_rmse": metrics,
                "yaw_change_rad": yaw_change,
                "commanded_yaw_change_rad": commanded_yaw_change,
                "turn_progress_fraction": turn_progress,
            }
        )
    sections = sorted({row["terrain_section"] for row in rows})
    completion = bool(rows[-1]["truncated"]) and not bool(rows[-1]["terminated"])
    covered = {"flat", "bumps", "ramp_up", "ramp_down"}.issubset(sections)
    return {
        "completed": completion,
        "quality_pass": completion and covered and all(row["quality_pass"] for row in summaries),
        "duration_s": rows[-1]["time_s"],
        "stage_summaries": summaries,
        "visited_sections": sections,
        "required_terrain_visited": covered,
        "final_lateral_path_error_m": rows[-1]["lateral_path_error_m"],
        "max_abs_yaw_error_rad": max(abs(row["yaw_error_rad"]) for row in rows),
        "whole_task_rmse": {
            field: float(np.sqrt(np.mean([row[field] ** 2 for row in rows])))
            for field in (
                "velocity_error_mps",
                "lateral_velocity_mps",
                "yaw_rate_error_rps",
                "clearance_error_m",
                "lateral_path_error_m",
                "longitudinal_path_error_m",
            )
            if field in rows[0]
        },
        "quality_criteria": {
            "steady_window_excludes_first_s": 1.0,
            "velocity_rmse_max_mps": 0.14,
            "lateral_velocity_rmse_max_mps": 0.10,
            "yaw_rate_rmse_max_rps": 0.06,
            "clearance_rmse_max_m": 0.025,
            "turn_progress_fraction_range": [0.25, 1.75],
            "all_terrain_sections_and_stages_required": True,
        },
        "interpretation": "one continuous development task, not statistical generalization evidence",
    }
