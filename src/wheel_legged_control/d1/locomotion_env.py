"""Command-conditioned Gym task on the shared D1 control loop.

Each worker owns one fixed collision road. Reset changes commands and optional
synthetic dynamics, never moves terrain under the robot. Reward/evaluation truth
is confined to info and reward; actor observations encode the prepared decision.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from numbers import Integral, Real
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actuator_channel import ActuatorChannel, ActuatorChannelConfig
from .control_loop import (
    D1ControlLoop,
    D1ForceControllerAdapter,
    D1MotionCommand,
    D1WheelLegControllerAdapter,
)
from .control_primitives import terrain_normal_to_rpy
from .hierarchical import D1LQRVMCController, D1MPCVMCController
from .locomotion_commands import (
    D1CommandSchedule,
    random_command_schedule,
    schedule_to_dict,
    validation_command_schedule,
)
from .locomotion_observation import (
    LOCOMOTION_OBSERVATION_SCHEMA,
    LOCOMOTION_OBSERVATION_SIZE,
    encode_d1_locomotion_observation,
)
from .locomotion_rewards import D1LocomotionRewardConfig, d1_locomotion_reward_terms
from .locomotion_terrain import (
    LOCOMOTION_MAP_HALF_SIZE_M,
    LOCOMOTION_SPAWN_XY_M,
    D1LocomotionTerrainConfig,
)
from .model import JOINT_TORQUE_LIMIT, D1Plant
from .state_provider import D1StateProviderConfig, build_d1_state_provider
from .wheel_leg_controller import (
    D1ControlEnvelopeError,
    D1WheelLegControlConfig,
    D1WheelLegController,
)

LOCOMOTION_REWARD_SCHEMA = "d1-command-tracking-rate-v1"
LOCOMOTION_TASK_SCHEMA = "d1-fixed-road-command-task-v1"
SHARED_POLICY_ACTION_SCHEMA = "d1-shared-wheel-leg-extension-speed-v1"
# One leg-extension and one wheel-speed policy dimension replicated onto the
# unchanged physical eight-dimensional schema. The map is a pure gather: no
# normalization, reordering or scaling happens here.
SHARED_POLICY_TO_PHYSICAL_INDICES = (0, 0, 0, 0, 1, 1, 1, 1)
POLICY_ACTION_MODES = ("shared2", "independent8")
LOCOMOTION_SPAWN_POSITION_M = (*LOCOMOTION_SPAWN_XY_M, 0.455)
# The body-origin boundary leaves room for the whole footprint before the
# collision map ends. It is a safety termination, not task completion.
LOCOMOTION_SAFE_HALF_SIZE_M = (5.3, 2.3)


@dataclass(frozen=True, slots=True)
class D1LocomotionRandomization:
    """Episode-wise synthetic ranges; None preserves the explicit base config.

    Measurement delays count 10 ms decisions. Actuator delays count 2 ms
    physical substeps. Strength changes saturation; gain changes motor output.
    No field represents an identified real D1 parameter.
    """

    base_mass_scale: tuple[float, float] = (1.0, 1.0)
    damping_scale: tuple[float, float] = (1.0, 1.0)
    friction_scale: tuple[float, float] = (1.0, 1.0)
    actuator_strength_scale: tuple[float, float] = (1.0, 1.0)
    measurement_delay_steps: tuple[int, int] | None = None
    actuator_delay_steps: tuple[int, int] | None = None
    actuator_time_constant_s: tuple[float, float] | None = None
    actuator_gain: tuple[float, float] | None = None

    def __post_init__(self):
        for item in fields(self):
            name, value = item.name, getattr(self, item.name)
            if value is None and not name.endswith("_scale"):
                continue
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError(f"{name} must be a lower/upper pair")
            integer = name.endswith("delay_steps")
            kind = Integral if integer else Real
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, kind) for v in value):
                raise ValueError(f"{name} must contain finite real values or integer delays")
            low, high = value
            if not np.isfinite(value).all() or low > high or low < 0:
                raise ValueError(f"{name} requires finite 0 <= lower <= upper")
            if (name.endswith("_scale") or name == "actuator_gain") and low <= 0:
                raise ValueError(f"{name} must be positive")
            if integer and high > 20:
                raise ValueError("development delay ranges are limited to 20 steps")
            object.__setattr__(self, name, tuple(int(v) if integer else float(v) for v in value))


class D1LocomotionEnv(gym.Env):
    """The same current-command decision is used by Gym and keyboard callbacks.

    A command_source is sampled once for each prepared tick (at reset and at
    the end of step). External input arriving later applies to the following
    tick; it cannot replace a command after the actor has observed it.

    A finite next reference outside the controller's envelope terminates the
    completed transition. With no executable next preview, its terminal-only
    observation encodes the last executed decision, explicitly labelled in
    info. It is not a fresh state estimate and must not be bootstrapped. Normal
    observations and schemas are unchanged; numerical/configuration failures
    still raise instead of becoming training samples.
    """

    metadata: ClassVar[dict] = {"render_modes": []}
    observation_schema = LOCOMOTION_OBSERVATION_SCHEMA
    reward_schema = LOCOMOTION_REWARD_SCHEMA
    task_schema = LOCOMOTION_TASK_SCHEMA

    def __init__(
        self,
        baseline: str = "wheel_leg",
        episode_seconds: float = 60.0,
        terrain: D1LocomotionTerrainConfig | None = None,
        command_mode: str = "random",
        provider_config: D1StateProviderConfig | None = None,
        actuator_config: ActuatorChannelConfig | None = None,
        randomization: D1LocomotionRandomization | None = None,
        reward_config: D1LocomotionRewardConfig | None = None,
        command_source: Callable[[float], D1MotionCommand] | None = None,
        wheel_leg_control: D1WheelLegControlConfig | None = None,
        action_mode: str | None = None,
    ):
        super().__init__()
        if baseline not in ("wheel_leg", "lqr", "mpc"):
            raise ValueError("baseline must be wheel_leg, lqr or mpc")
        if action_mode is not None:
            # Rejected before any simulator/controller exists: an unsupported
            # policy mapping must never build a half-configured environment.
            if action_mode not in POLICY_ACTION_MODES:
                raise ValueError("action_mode must be None, 'shared2' or 'independent8'")
            if baseline != "wheel_leg":
                raise ValueError("action_mode requires the wheel_leg baseline")
        if wheel_leg_control is not None and baseline != "wheel_leg":
            raise ValueError("wheel_leg_control requires the wheel_leg baseline")
        if wheel_leg_control is not None and not isinstance(
            wheel_leg_control, D1WheelLegControlConfig
        ):
            raise TypeError("wheel_leg_control must be D1WheelLegControlConfig")
        self.wheel_leg_control = (
            D1WheelLegControlConfig() if wheel_leg_control is None else wheel_leg_control
        )
        if command_mode not in ("random", "development", "holdout"):
            raise ValueError("command_mode must be random, development or holdout")
        if (
            isinstance(episode_seconds, (bool, np.bool_))
            or not isinstance(episode_seconds, Real)
            or not np.isfinite(episode_seconds)
            or episode_seconds < 0.01
        ):
            raise ValueError("episode_seconds must be finite and at least one control tick")
        steps = episode_seconds / 0.01
        if not np.isclose(steps, round(steps), rtol=0, atol=1e-8):
            raise ValueError("episode_seconds must be an integer multiple of 10 ms")
        if command_source is not None and not callable(command_source):
            raise TypeError("command_source must be callable")
        self.baseline_name, self.command_mode = baseline, command_mode
        self.episode_seconds, self.max_steps = float(episode_seconds), round(steps)
        self.terrain = D1LocomotionTerrainConfig() if terrain is None else terrain
        self.provider_config = (
            D1StateProviderConfig("oracle") if provider_config is None else provider_config
        )
        self.actuator_config = (
            ActuatorChannelConfig(torque_limit_nm=tuple(JOINT_TORQUE_LIMIT))
            if actuator_config is None
            else actuator_config
        )
        self.randomization = D1LocomotionRandomization() if randomization is None else randomization
        self.reward_config = D1LocomotionRewardConfig() if reward_config is None else reward_config
        for value, expected in (
            (self.terrain, D1LocomotionTerrainConfig),
            (self.provider_config, D1StateProviderConfig),
            (self.actuator_config, ActuatorChannelConfig),
            (self.randomization, D1LocomotionRandomization),
            (self.reward_config, D1LocomotionRewardConfig),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"expected {expected.__name__}")
        if self.actuator_config.n_axes != 16 or self.actuator_config.physics_dt_s != 0.002:
            raise ValueError("task actuator requires 16 axes at a 2 ms physical timestep")
        if self.actuator_config.torque_limit_nm is None or not np.array_equal(
            np.broadcast_to(self.actuator_config.torque_limit_nm, (16,)), JOINT_TORQUE_LIMIT
        ):
            raise ValueError(
                "base actuator limits must match nominal D1 limits; use strength randomization"
            )
        if (
            self.provider_config.kind == "oracle"
            and self.randomization.measurement_delay_steps is not None
        ):
            raise ValueError("oracle cannot randomize measurement delays")
        if self.provider_config.kind == "imu_encoder_fusion" and (
            self.provider_config.initial_position_m != LOCOMOTION_SPAWN_POSITION_M
            or self.provider_config.initial_rpy_rad != (0, 0, 0)
        ):
            raise ValueError("fusion placement prior must match the known flat-strip spawn")
        self.command_source = command_source
        self.plant = D1Plant(
            sampling_mode="synchronized",
            locomotion_terrain=self.terrain,
            actuator_channel=ActuatorChannel(self.actuator_config),
        )
        self._controller = (
            D1WheelLegControllerAdapter(D1WheelLegController(**asdict(self.wheel_leg_control)))
            if baseline == "wheel_leg"
            else D1ForceControllerAdapter(
                D1LQRVMCController(self.plant)
                if baseline == "lqr"
                else D1MPCVMCController(self.plant)
            )
        )
        self.physical_action_schema = self._controller.action_schema
        self.physical_action_size = self._controller.action_size
        self.action_mode = (
            ("independent8" if action_mode is None else action_mode)
            if baseline == "wheel_leg"
            else "legacy_force2"
        )
        shared = self.action_mode == "shared2"
        self.action_schema = SHARED_POLICY_ACTION_SCHEMA if shared else self.physical_action_schema
        self.policy_action_size = 2 if shared else self.physical_action_size
        self.policy_to_physical_indices = (
            SHARED_POLICY_TO_PHYSICAL_INDICES if shared else tuple(range(self.physical_action_size))
        )
        self._policy_gather = np.asarray(self.policy_to_physical_indices, dtype=np.intp)
        self.source_schema = self.provider_config.source_schema
        self.action_space = spaces.Box(-1.0, 1.0, (self.policy_action_size,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -5.0, 5.0, (LOCOMOTION_OBSERVATION_SIZE,), dtype=np.float32
        )
        self.loop = None
        self.decision = self.last_transition = self.schedule = None
        self._active = False
        self._episode_metadata = {}

    @property
    def episode_metadata(self):
        return deepcopy(self._episode_metadata)

    def _bounded_ground_query(self, x, y):
        # No fictitious extrapolation: clamp only to produce a terminal-ready
        # publication. The episode is immediately terminated if this occurs.
        point = np.asarray((x, y), dtype=float)
        clipped = np.clip(
            point, -np.asarray(LOCOMOTION_MAP_HALF_SIZE_M), LOCOMOTION_MAP_HALF_SIZE_M
        )
        if not np.array_equal(point, clipped):
            self._ground_query_clamped = True
        return self.plant.locomotion_ground_reference(*clipped)

    def _sample_range(self, bounds, integer=False):
        low, high = bounds
        if low == high:
            return low
        return (
            int(self.np_random.integers(low, high + 1))
            if integer
            else float(self.np_random.uniform(low, high))
        )

    def _configure_dynamics(self):
        randomized = self.randomization
        domain = {
            name: self._sample_range(getattr(randomized, name))
            for name in (
                "base_mass_scale",
                "damping_scale",
                "friction_scale",
                "actuator_strength_scale",
            )
        }
        channel = {"torque_limit_nm": tuple(JOINT_TORQUE_LIMIT * domain["actuator_strength_scale"])}
        for field, name in (
            ("actuator_delay_steps", "delay_steps"),
            ("actuator_time_constant_s", "time_constant_s"),
            ("actuator_gain", "gain"),
        ):
            bounds = getattr(randomized, field)
            if bounds is not None:
                channel[name] = self._sample_range(bounds, integer=name == "delay_steps")
        channel_config = replace(self.actuator_config, **channel)
        config = self.provider_config
        if randomized.measurement_delay_steps is not None:
            delay = self._sample_range(randomized.measurement_delay_steps, integer=True)
            config = (
                replace(config, sensor_delay_steps=delay)
                if config.kind == "imu_encoder_fusion"
                else replace(config, impairments=replace(config.impairments, delay_steps=delay))
            )
        self.plant.set_domain(**domain)
        self.plant.actuator_channel = ActuatorChannel(channel_config)
        provider = build_d1_state_provider(
            self.plant,
            config,
            **(
                {}
                if config.kind == "imu_encoder_fusion"
                else {"oracle_ground_query": self._bounded_ground_query}
            ),
        )
        self.loop = D1ControlLoop(self.plant, provider, self._controller)
        return domain, channel_config, config

    def _prepare(self):
        time_s = self._steps * self.plant.control_dt
        command = (
            self.command_source(time_s)
            if self.command_source is not None
            else self.schedule.cmd_at(time_s)
        )
        self.decision = self.loop.prepare(command)
        return encode_d1_locomotion_observation(self.decision)

    def reset(self, *, seed=None, options=None):
        options = {} if options is None else dict(options)
        if set(options) - {"schedule"}:
            raise ValueError("only an explicit schedule may be supplied as a reset option")
        schedule = options.get("schedule")
        if schedule is not None and (
            not isinstance(schedule, D1CommandSchedule)
            or not np.isclose(schedule.duration_s, self.episode_seconds, rtol=0, atol=1e-10)
        ):
            raise ValueError("schedule duration must match the episode")
        if schedule is not None and self.command_source is not None:
            raise ValueError("external commands cannot also consume a schedule")
        super().reset(seed=seed)
        self._active = False
        self._steps = 0
        self._ground_query_clamped = False
        domain, channel, provider = self._configure_dynamics()
        command_seed, measurement_seed = (int(v) for v in self.np_random.integers(0, 2**31, 2))
        self.schedule = (
            None
            if self.command_source is not None
            else schedule
            if schedule is not None
            else random_command_schedule(command_seed, self.episode_seconds)
            if self.command_mode == "random"
            else validation_command_schedule(self.command_mode, self.episode_seconds)
        )
        self.loop.reset(seed=measurement_seed, base_position=LOCOMOTION_SPAWN_POSITION_M)
        self.last_transition = None
        self._last_xy = np.asarray(LOCOMOTION_SPAWN_XY_M)
        self._nonflat_steps, self._path_m, self._nonflat_path_m = 0, 0.0, 0.0
        self._terrain_cells = set()
        self._episode_metadata = {
            **(
                {"controller_parameters": asdict(self.wheel_leg_control)}
                if self.baseline_name == "wheel_leg"
                else {}
            ),
            "task_schema": self.task_schema,
            "terminal_reference_handling": "finite-envelope-exit-cached-terminal-observation-v1",
            "control_schema": self.loop.schema,
            "controller_schema": getattr(
                self._controller.controller, "control_schema", self.action_schema
            ),
            "observation_schema": self.observation_schema,
            "action_schema": self.action_schema,
            "action_mode": self.action_mode,
            "policy_action_size": self.policy_action_size,
            "physical_action_size": self.physical_action_size,
            "physical_action_schema": self.physical_action_schema,
            "policy_to_physical_indices": list(self.policy_to_physical_indices),
            "source_schema": self.source_schema,
            "reward_schema": self.reward_schema,
            "baseline": self.baseline_name,
            "duration_s": self.episode_seconds,
            "control_dt_s": self.plant.control_dt,
            "physics_dt_s": float(self.plant.model.opt.timestep),
            "terrain": asdict(self.terrain),
            "provider": asdict(provider),
            "actuator": asdict(channel),
            "domain": domain,
            "randomization": asdict(self.randomization),
            "reward": asdict(self.reward_config),
            "command_seed": command_seed,
            "measurement_seed": measurement_seed,
            "command_source": "external_callback"
            if self.command_source is not None
            else "schedule",
            "schedule": None if self.schedule is None else schedule_to_dict(self.schedule),
            "spawn_position_m": LOCOMOTION_SPAWN_POSITION_M,
            "safe_half_size_m": LOCOMOTION_SAFE_HALF_SIZE_M,
            "parameter_origin": "synthetic simulation settings, not real D1 identification",
        }
        observation = self._prepare()
        self._active = True
        return observation, {"episode_metadata": self.episode_metadata}

    def _metrics(self, transition, ground):
        truth, target = transition.truth, transition.decision.motion_command
        roll, pitch = terrain_normal_to_rpy(
            -np.tan(ground.pitch_rad), np.tan(ground.roll_rad), truth.base_rpy[2]
        )
        traces = self.plant.last_control_interval_actuator_traces
        return {
            "time_s": truth.control_time_s,
            "x_m": float(truth.base_position[0]),
            "y_m": float(truth.base_position[1]),
            "z_m": float(truth.base_position[2]),
            "ground_height_m": ground.height_m,
            "clearance_m": float(truth.base_position[2] - ground.height_m),
            "velocity_mps": float(truth.base_linear_velocity_body[0]),
            "yaw_rate_rps": float(truth.base_angular_velocity_body[2]),
            "yaw_rad": float(truth.base_rpy[2]),
            "velocity_error_mps": float(
                truth.base_linear_velocity_body[0] - target.forward_velocity_mps
            ),
            "yaw_rate_error_rps": float(truth.base_angular_velocity_body[2] - target.yaw_rate_rps),
            "height_error_m": float(truth.base_position[2] - ground.height_m - target.clearance_m),
            "roll_error_rad": float(truth.base_rpy[0] - roll),
            "pitch_error_rad": float(truth.base_rpy[1] - pitch),
            "mechanical_power_w": float(
                np.mean([np.sum(np.abs(t.applied_nm * t.joint_velocity_rps)) for t in traces])
            ),
            "torque_input_clipped_fraction": float(
                np.mean([t.requested_nm != t.limited_nm for t in traces])
            ),
            "measurement_age_s": transition.state.age_s,
            "undesired_ground_contacts": self.plant.undesired_ground_contacts,
        }

    def _exposure(self, transition, ground):
        truth = transition.truth
        references = [ground]
        for point in truth.foot_position:
            references.append(self._bounded_ground_query(float(point[0]), float(point[1])))
        nonflat = any(
            abs(g.height_m) > 0.001 or max(abs(g.roll_rad), abs(g.pitch_rad)) > np.deg2rad(0.25)
            for g in references
        )
        distance = float(np.linalg.norm(truth.base_position[:2] - self._last_xy))
        self._last_xy = truth.base_position[:2].copy()
        self._path_m += distance
        if nonflat:
            self._nonflat_steps += 1
            self._nonflat_path_m += distance
            self._terrain_cells.add(tuple(np.floor(truth.base_position[:2] / 0.25).astype(int)))
        return {
            "nonflat_now": bool(nonflat),
            "nonflat_steps": self._nonflat_steps,
            "nonflat_fraction": self._nonflat_steps / self._steps,
            "path_m": self._path_m,
            "nonflat_path_m": self._nonflat_path_m,
            "unique_nonflat_0p25m_cells": len(self._terrain_cells),
            "definition": "body/wheel-center ground: |z|>1mm or local slope>0.25deg; geometric exposure, not contact or traversal proof",
        }

    def _policy_action(self, action):
        """Validate the policy action, then gather it onto physical dimensions.

        Validation does not advance physics or consume the prepared decision.
        The environment still requires reset after an error, as before this
        mapping existed. Clipping belongs to the physical control loop.
        """
        policy = np.asarray(action, dtype=np.float64)
        if policy.shape != (self.policy_action_size,) or not np.isfinite(policy).all():
            raise ValueError("action must match the finite action schema (policy dimensions)")
        return policy.copy(), policy[self._policy_gather]

    def step(self, action):
        if not self._active:
            raise RuntimeError("reset must precede step or follow episode end")
        try:
            policy, physical = self._policy_action(action)
            self._ground_query_clamped = False
            transition = self.loop.step(physical)
            self.last_transition = transition
            self._steps += 1
            ground = self._bounded_ground_query(*transition.truth.base_position[:2])
            metrics = self._metrics(transition, ground)
            exposure = self._exposure(transition, ground)
            fallen = (
                metrics["clearance_m"] < 0.22
                or np.max(np.abs(transition.truth.base_rpy[:2])) > 0.85
                or metrics["undesired_ground_contacts"] > 0
            )
            outside = np.any(
                np.abs(transition.truth.base_position[:2]) >= LOCOMOTION_SAFE_HALF_SIZE_M
            )
            reason = (
                "fall_or_body_contact"
                if fallen
                else "map_boundary"
                if outside
                else "ground_query_outside_map"
                if self._ground_query_clamped
                else None
            )
            terminated = reason is not None
            truncated = self._steps >= self.max_steps and not terminated
            rejection_info = {}
            try:
                observation = self._prepare()
            except D1ControlEnvelopeError as exc:
                # Physics and the next measurement already completed. Reject
                # only this finite reference-domain exit, not arbitrary errors.
                reason = reason or "control_reference_out_of_envelope"
                terminated, truncated = True, False
                self.decision = transition.decision
                observation = encode_d1_locomotion_observation(transition.decision)
                rejection_info = {
                    "terminal_observation_source": "last_executable_decision",
                    "control_reference_rejection": str(exc),
                }
            terms = d1_locomotion_reward_terms(
                transition,
                ground,
                self.plant.last_control_interval_actuator_traces,
                terminated=terminated,
                config=self.reward_config,
            )
            info = {
                **rejection_info,
                "reward_terms": terms,
                "command": asdict(transition.decision.motion_command),
                "metrics": metrics,
                "terrain_exposure": exposure,
                "terminal_reason": reason if terminated else "time_limit" if truncated else None,
                "ground_query_clamped": self._ground_query_clamped,
                "raw_action": transition.raw_action.copy(),
                "applied_action": transition.receipt.normalized_action.copy(),
                "policy_action": policy,
            }
        except Exception:
            # Do not turn numerical/invariant failures into fabricated terminal
            # rewards. A partly consumed physics/filter state is not retryable.
            self._active = False
            raise
        self._active = not (terminated or truncated)
        return observation, float(sum(terms.values())), terminated, truncated, info

    def close(self):
        self._active = False
