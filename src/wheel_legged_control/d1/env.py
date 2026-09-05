"""Lightweight full-body D1 environment for residual reinforcement learning."""

from __future__ import annotations

from collections import deque
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .contact_allocation import D1_CONTACT_ALLOCATION_MODES, make_d1_contact_allocator
from .controllers import D1Command
from .hierarchical import D1LQRVMCController, D1MPCVMCController
from .model import JOINT_VELOCITY_LIMIT, NOMINAL_JOINT_POSITION, D1Plant
from .rewards import D1_REWARD_SCHEMA, calculate_d1_reward
from .state_estimation import (
    D1StateEstimate,
    make_d1_state_source,
    make_default_d1_estimator_impairments,
    prepare_d1_control_state,
)

D1_OBSERVATION_SIZE = 42
D1_OBSERVATION_SCHEMA = "d1-base-link-velocity-v2"
D1_RESIDUAL_SCALE = np.asarray((45.0, 80.0), dtype=np.float64)
D1_LEG_INDICES = np.asarray([index for index in range(16) if index % 4 != 3], dtype=np.int32)
D1_LEG_POSITION_SCALE = np.tile((0.8, 2.0, 1.0), 4)


def encode_d1_observation(
    *,
    state: D1StateEstimate,
    command: D1Command,
    baseline_longitudinal_force_n: float,
    previous_applied_action: np.ndarray,
) -> np.ndarray:
    """Build the versioned 42-value policy observation without adding noise."""

    previous = np.asarray(previous_applied_action, dtype=np.float64)
    if previous.shape != (2,):
        raise ValueError("previous_applied_action must have shape (2,)")
    linear_velocity, angular_velocity = state.base_velocity(local=True)
    leg_position_error = (
        state.joint_position[D1_LEG_INDICES] - NOMINAL_JOINT_POSITION[D1_LEG_INDICES]
    ) / D1_LEG_POSITION_SCALE
    observation = np.concatenate(
        (
            linear_velocity / np.asarray((2.0, 1.0, 1.0)),
            angular_velocity / 4.0,
            state.projected_gravity_body,
            np.asarray(
                (
                    command.forward_velocity_mps,
                    (state.base_position[2] - command.base_height_m) / 0.08,
                )
            ),
            leg_position_error,
            state.joint_velocity / JOINT_VELOCITY_LIMIT,
            np.asarray((baseline_longitudinal_force_n / 180.0,)),
            previous,
        )
    ).astype(np.float32)
    return np.clip(observation, -5.0, 5.0)


def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.asarray(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dtype=np.float64,
    )


class D1ResidualEnv(gym.Env[np.ndarray, np.ndarray]):
    """Residual force policy over a contact-aware D1 LQR/MPC+VMC baseline."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}
    observation_schema = D1_OBSERVATION_SCHEMA
    reward_schema = D1_REWARD_SCHEMA

    def __init__(
        self,
        baseline: str = "lqr",
        episode_seconds: float = 6.0,
        randomize: bool = True,
        state_mode: str = "oracle",
        latency_compensation: str = "none",
        contact_allocation: str = "legacy",
        measure_contact_wrench: bool = False,
        profile_allocation_timing: bool = False,
    ) -> None:
        super().__init__()
        if baseline not in {"lqr", "mpc"}:
            raise ValueError("baseline must be 'lqr' or 'mpc'")
        if episode_seconds <= 0.0:
            raise ValueError("episode_seconds must be positive")
        if state_mode not in {"oracle", "estimated"}:
            raise ValueError("state_mode must be 'oracle' or 'estimated'")
        if latency_compensation not in {"none", "constant_velocity"}:
            raise ValueError("latency_compensation must be 'none' or 'constant_velocity'")
        if state_mode == "oracle" and latency_compensation != "none":
            raise ValueError("latency compensation requires state_mode='estimated'")
        if contact_allocation not in D1_CONTACT_ALLOCATION_MODES:
            raise ValueError(f"contact_allocation must be one of {D1_CONTACT_ALLOCATION_MODES}")
        if not isinstance(measure_contact_wrench, bool):
            raise TypeError("measure_contact_wrench must be a bool")
        if not isinstance(profile_allocation_timing, bool):
            raise TypeError("profile_allocation_timing must be a bool")
        self.baseline_name = baseline
        self.randomize = randomize
        self.state_mode = state_mode
        self.latency_compensation = latency_compensation
        self.contact_allocation = contact_allocation
        self.measure_contact_wrench = measure_contact_wrench
        self.profile_allocation_timing = profile_allocation_timing
        self.plant = D1Plant(control_dt=0.01)
        allocator = make_d1_contact_allocator(self.plant, contact_allocation)
        self.controller = (
            D1LQRVMCController(self.plant, contact_allocator=allocator)
            if baseline == "lqr"
            else D1MPCVMCController(self.plant, contact_allocator=allocator)
        )
        self.max_steps = round(episode_seconds / self.plant.control_dt)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -5.0,
            5.0,
            shape=(D1_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self._leg_indices = D1_LEG_INDICES
        self._step_count = 0
        self._scenario = "training"
        self._command = D1Command()
        self._training_velocity = 0.0
        self._training_height = self.plant.nominal_base_height_m
        self._delay_steps = 0
        self._state_delay_steps = 0
        self._noise_scale = 0.0
        self._estimator_seed: int | None = None
        self._delay_queue: deque[np.ndarray] = deque()
        self._previous_applied_action = np.zeros(2, dtype=np.float64)
        self._push_start = -1
        self._push_end = -1
        self._push_force_n = 0.0
        self._domain = {
            "base_mass_scale": 1.0,
            "damping_scale": 1.0,
            "friction_scale": 1.0,
            "actuator_strength_scale": 1.0,
        }
        self._state_source = make_d1_state_source(self.plant)
        self._raw_state = self._state_source.reset()
        self._state = self._raw_state
        self._compensation_status = "disabled"
        self._compensation_horizon_s = 0.0

    def _publish_control_state(self, raw_state: D1StateEstimate) -> None:
        self._raw_state = raw_state
        result = prepare_d1_control_state(
            raw_state,
            latency_compensation=self.latency_compensation,
        )
        self._state = result.control_state
        self._compensation_status = result.status
        self._compensation_horizon_s = result.applied_horizon_s

    def _sample_training_command(self) -> None:
        self._training_velocity = float(self.np_random.uniform(-0.75, 0.75))
        self._training_height = float(self.np_random.uniform(0.445, 0.475))

    def _command_at_step(self) -> D1Command:
        time_s = self._step_count * self.plant.control_dt
        if self._scenario == "training":
            if self._step_count > 0 and self._step_count % 150 == 0:
                self._sample_training_command()
            return D1Command(self._training_velocity, 0.0, self._training_height)
        if time_s < 0.75:
            return D1Command(0.0, 0.0, 0.455)
        if time_s < 2.75:
            return D1Command(0.50, 0.0, 0.465 if time_s >= 1.75 else 0.455)
        if time_s < 4.75:
            return D1Command(-0.30, 0.0, 0.445)
        return D1Command(0.25, 0.0, 0.455)

    def _configure_episode(self, options: dict[str, Any]) -> None:
        self._scenario = str(options.get("scenario", "training"))
        use_randomization = bool(options.get("randomize", self.randomize))
        if use_randomization:
            self._domain = {
                "base_mass_scale": float(
                    options.get("base_mass_scale", self.np_random.uniform(0.90, 1.12))
                ),
                "damping_scale": float(
                    options.get("damping_scale", self.np_random.uniform(0.80, 1.25))
                ),
                "friction_scale": float(
                    options.get("friction_scale", self.np_random.uniform(0.65, 1.30))
                ),
                "actuator_strength_scale": float(
                    options.get("actuator_strength_scale", self.np_random.uniform(0.85, 1.05))
                ),
            }
            sampled_action_delay = int(self.np_random.integers(0, 4))
            sampled_state_delay = (
                int(self.np_random.integers(0, 4)) if self.state_mode == "estimated" else 0
            )
            self._delay_steps = int(
                options.get("action_delay_steps", options.get("delay_steps", sampled_action_delay))
            )
            self._state_delay_steps = (
                int(options.get("state_delay_steps", sampled_state_delay))
                if self.state_mode == "estimated"
                else 0
            )
            self._noise_scale = float(options.get("sensor_noise", self.np_random.uniform(0.0, 1.0)))
        else:
            self._domain = {
                "base_mass_scale": float(options.get("base_mass_scale", 1.0)),
                "damping_scale": float(options.get("damping_scale", 1.0)),
                "friction_scale": float(options.get("friction_scale", 1.0)),
                "actuator_strength_scale": float(options.get("actuator_strength_scale", 1.0)),
            }
            self._delay_steps = int(
                options.get("action_delay_steps", options.get("delay_steps", 0))
            )
            self._state_delay_steps = (
                int(options.get("state_delay_steps", 0)) if self.state_mode == "estimated" else 0
            )
            self._noise_scale = float(options.get("sensor_noise", 0.0))
        if self._delay_steps < 0 or self._state_delay_steps < 0:
            raise ValueError("action and state delays must be non-negative")
        if not np.isfinite(self._noise_scale) or self._noise_scale < 0.0:
            raise ValueError("sensor_noise must be finite and non-negative")
        self.plant.set_domain(**self._domain)
        allocator = self.controller.low_level.contact_allocator
        if allocator is not None:
            allocator.set_torque_limits(self.plant.actuator_torque_limit_nm)

        if self._scenario == "training":
            self._sample_training_command()
            latest_start = max(81, self.max_steps - 80)
            self._push_start = int(self.np_random.integers(80, latest_start))
            self._push_end = self._push_start + int(self.np_random.integers(8, 15))
            self._push_force_n = float(
                self.np_random.uniform(90.0, 170.0) * self.np_random.choice((-1.0, 1.0))
            )
        elif self._scenario == "push":
            self._push_start = round(2.6 / self.plant.control_dt)
            self._push_end = self._push_start + round(0.12 / self.plant.control_dt)
            self._push_force_n = float(options.get("push_force_n", 140.0))
        elif self._scenario == "mismatch_delay":
            self._push_start = round(3.6 / self.plant.control_dt)
            self._push_end = self._push_start + round(0.12 / self.plant.control_dt)
            self._push_force_n = float(options.get("push_force_n", -130.0))
        else:
            self._push_start = self._push_end = -1
            self._push_force_n = 0.0

        roll = float(
            options.get(
                "initial_roll", self.np_random.uniform(-0.035, 0.035) if use_randomization else 0.0
            )
        )
        pitch = float(
            options.get(
                "initial_pitch",
                self.np_random.uniform(-0.045, 0.045) if use_randomization else 0.02,
            )
        )
        yaw = float(
            options.get(
                "initial_yaw", self.np_random.uniform(-0.05, 0.05) if use_randomization else 0.0
            )
        )
        joint_position = NOMINAL_JOINT_POSITION.copy()
        if use_randomization:
            joint_position[self._leg_indices] += self.np_random.uniform(
                -0.025, 0.025, size=len(self._leg_indices)
            )
        base_height = self.plant.nominal_base_height_m + (
            float(self.np_random.uniform(-0.008, 0.012)) if use_randomization else 0.0
        )
        self.plant.reset(
            base_position=np.asarray((0.0, 0.0, base_height)),
            base_quaternion=_quaternion_from_rpy(roll, pitch, yaw),
            joint_position=joint_position,
        )
        if self.state_mode == "estimated":
            impairments = make_default_d1_estimator_impairments(
                delay_steps=self._state_delay_steps,
                noise_scale=self._noise_scale,
            )
            self._estimator_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))
            self._state_source = make_d1_state_source(
                self.plant,
                impairments=impairments,
                seed=self._estimator_seed,
            )
        else:
            self._estimator_seed = None
            self._state_source = make_d1_state_source(self.plant)
        self._publish_control_state(self._state_source.reset())

    def _observation(self) -> np.ndarray:
        observation = encode_d1_observation(
            state=self._state,
            command=self._command,
            baseline_longitudinal_force_n=self.controller.last_longitudinal_force_n,
            previous_applied_action=self._previous_applied_action,
        )
        if self.state_mode == "oracle" and self._noise_scale:
            observation += self.np_random.normal(
                0.0,
                0.008 * self._noise_scale,
                size=D1_OBSERVATION_SIZE,
            ).astype(np.float32)
        return np.clip(observation, -5.0, 5.0)

    def _info(
        self,
        *,
        command: D1Command,
        torque_nm: np.ndarray,
        applied_action: np.ndarray,
        push_force_n: float,
    ) -> dict[str, Any]:
        truth_state = self.plant.reduced_state()
        truth_forward_velocity = float(self.plant.base_velocity(local=True)[0][0])
        truth_joint_position = self.plant.joint_position
        truth_joint_velocity = self.plant.joint_velocity
        truth_wheel_contacts = self.plant.wheel_ground_contacts
        truth_undesired_contacts = self.plant.undesired_ground_contacts
        allocation = self.controller.low_level.last_breakdown
        allocation_status = (
            "not_run" if allocation.allocation_status is None else allocation.allocation_status.value
        )
        allocation_wrench_tracking_status = (
            "not_run"
            if allocation.allocation_wrench_tracking_status is None
            else allocation.allocation_wrench_tracking_status.value
        )
        measured_contact = self.plant.last_control_interval_contact_wrench
        return {
            # Legacy names remain truth-valued for reward/evaluation compatibility.
            "state": truth_state,
            "truth_state": truth_state,
            "raw_estimated_state": self._raw_state.reduced_state(),
            "control_state": self._state.reduced_state(),
            # Backward-compatible alias: this is the state consumed by control.
            "estimated_state": self._state.reduced_state(),
            "forward_velocity_mps": truth_forward_velocity,
            "truth_forward_velocity_mps": truth_forward_velocity,
            "estimated_forward_velocity_mps": float(self._state.base_linear_velocity_body[0]),
            "joint_position": truth_joint_position,
            "truth_joint_position": truth_joint_position,
            "estimated_joint_position": self._state.joint_position.copy(),
            "joint_velocity": truth_joint_velocity,
            "truth_joint_velocity": truth_joint_velocity,
            "estimated_joint_velocity": self._state.joint_velocity.copy(),
            "torque_nm": np.asarray(torque_nm, dtype=np.float64).copy(),
            "command_velocity_mps": command.forward_velocity_mps,
            "command_height_m": command.base_height_m,
            "residual_action": np.asarray(applied_action, dtype=np.float64).copy(),
            "longitudinal_force_n": self.controller.last_longitudinal_force_n,
            "vertical_residual_n": self.controller.last_vertical_residual_n,
            "push_force_n": push_force_n,
            "wheel_contacts": truth_wheel_contacts,
            "truth_wheel_contacts": truth_wheel_contacts,
            "estimated_wheel_contacts": self._state.wheel_ground_contacts,
            "undesired_contacts": truth_undesired_contacts,
            "truth_undesired_contacts": truth_undesired_contacts,
            "estimated_undesired_contacts": self._state.undesired_ground_contacts,
            "delay_steps": self._delay_steps,
            "action_delay_steps": self._delay_steps,
            "state_delay_steps": self._state_delay_steps,
            "sensor_noise_scale": self._noise_scale,
            "state_estimator_seed": self._estimator_seed,
            "state_age_ms": 1e3 * self._state.age_s,
            "state_estimation_mode": self.state_mode,
            "latency_compensation": self.latency_compensation,
            "latency_compensation_status": self._compensation_status,
            "latency_compensation_horizon_ms": 1e3 * self._compensation_horizon_s,
            "domain": dict(self._domain),
            "baseline": self.controller.baseline_name,
            "solve_ms": self.controller.last_solve_ms,
            "contact_allocation": self.contact_allocation,
            "allocation_status": allocation_status,
            "allocation_wrench_tracking_status": allocation_wrench_tracking_status,
            "allocation_status_reason": allocation.allocation_status_reason,
            "allocation_solve_ms": (
                allocation.allocation_solve_ms
                if self.profile_allocation_timing
                else 0.0
            ),
            "allocation_timing_measured": self.profile_allocation_timing,
            "allocation_constraint_violation": allocation.allocation_constraint_violation,
            "allocation_desired_wrench_world": allocation.desired_wrench_world.copy(),
            "allocation_achieved_wrench_world": allocation.achieved_wrench_world.copy(),
            "measured_wheel_contact_force_world_n": (
                measured_contact.wheel_force_world_n.copy()
            ),
            "measured_contact_wrench_world": measured_contact.wrench_world.copy(),
            "contact_wrench_reference_position_world_m": (
                measured_contact.reference_position_world_m.copy()
            ),
            "contact_wrench_physics_samples": measured_contact.physics_sample_count,
            "wheel_contact_active_sample_fraction": (
                measured_contact.active_sample_fraction_by_wheel.copy()
            ),
            "allocation_force_error_norm_n": float(
                np.linalg.norm(
                    allocation.achieved_wrench_world[:3]
                    - allocation.desired_wrench_world[:3]
                )
            ),
            "allocation_moment_error_norm_nm": float(
                np.linalg.norm(
                    allocation.achieved_wrench_world[3:]
                    - allocation.desired_wrench_world[3:]
                )
            ),
            "contact_force_model_error_norm_n": float(
                np.linalg.norm(
                    measured_contact.wrench_world[:3]
                    - allocation.achieved_wrench_world[:3]
                )
            ),
            "contact_moment_model_error_norm_nm": float(
                np.linalg.norm(
                    measured_contact.wrench_world[3:]
                    - allocation.achieved_wrench_world[3:]
                )
            ),
            "contact_force_tracking_error_norm_n": float(
                np.linalg.norm(
                    measured_contact.wrench_world[:3]
                    - allocation.desired_wrench_world[:3]
                )
            ),
            "contact_moment_tracking_error_norm_nm": float(
                np.linalg.norm(
                    measured_contact.wrench_world[3:]
                    - allocation.desired_wrench_world[3:]
                )
            ),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._configure_episode({} if options is None else dict(options))
        self.controller.reset()
        self._command = self._command_at_step()
        self._previous_applied_action[:] = 0.0
        self._delay_queue = deque([np.zeros(2, dtype=np.float64) for _ in range(self._delay_steps)])
        zero_torque = np.zeros(16, dtype=np.float64)
        info = self._info(
            command=self._command,
            torque_nm=zero_torque,
            applied_action=np.zeros(2),
            push_force_n=0.0,
        )
        return self._observation(), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        normalized_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if normalized_action.shape != (2,):
            raise ValueError("D1 residual action must have shape (2,)")
        self._delay_queue.append(normalized_action)
        applied_action = self._delay_queue.popleft()

        transition_command = self._command
        torque = self.controller.compute(
            transition_command,
            self._state,
            residual_force_n=applied_action * D1_RESIDUAL_SCALE,
        )
        push_force = (
            self._push_force_n if self._push_start <= self._step_count < self._push_end else 0.0
        )
        self.plant.step(
            torque,
            push_force_world_n=np.asarray((push_force, 0.0, 0.0)),
            measure_contact_wrench=self.measure_contact_wrench,
            contact_wrench_reference_world_m=(
                self.controller.low_level.last_breakdown.wrench_reference_position_world_m
            ),
        )
        self._publish_control_state(self._state_source.read())
        linear_velocity, _ = self.plant.base_velocity(local=True)
        roll, pitch, _ = self.plant.base_rpy
        terminated = self.plant.has_fallen()
        reward_terms = calculate_d1_reward(
            forward_velocity_mps=float(linear_velocity[0]),
            roll_rad=float(roll),
            pitch_rad=float(pitch),
            base_height_m=float(self.plant.base_position[2]),
            joint_position=self.plant.joint_position,
            command_velocity_mps=transition_command.forward_velocity_mps,
            command_height_m=transition_command.base_height_m,
            normalized_action=applied_action,
            previous_normalized_action=self._previous_applied_action,
            undesired_contacts=self.plant.undesired_ground_contacts,
            terminated=terminated,
        )
        info = self._info(
            command=transition_command,
            torque_nm=torque,
            applied_action=applied_action,
            push_force_n=push_force,
        )
        info["reward_terms"] = reward_terms.as_dict()

        self._previous_applied_action = applied_action.copy()
        self._step_count += 1
        truncated = bool(self._step_count >= self.max_steps)
        info["is_success"] = bool(truncated and not terminated)
        self._command = self._command_at_step()
        return self._observation(), reward_terms.total, terminated, truncated, info

    def close(self) -> None:
        self._delay_queue.clear()
