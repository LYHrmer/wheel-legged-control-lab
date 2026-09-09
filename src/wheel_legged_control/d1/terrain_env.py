"""Oracle-terrain residual learning, separate from legacy 42-value policies.

The controller tracks world-upright attitude and ground-relative clearance.
Terrain height and angles are simulator measurements, not a real estimator.
The first 0.3 s are stationary; a 0.3 s smoothstep then raises forward speed.
Curriculum stages are externally scheduled and change only on reset.
"""

from __future__ import annotations

from dataclasses import asdict
from numbers import Integral, Real
from typing import Any

import numpy as np
from gymnasium import spaces

from .controllers import D1Command
from .env import D1ResidualEnv, encode_d1_observation
from .model import D1Plant
from .rewards import D1RewardBreakdown, calculate_d1_reward
from .training_terrain import (
    TRAINING_TERRAIN_X_HALF_SIZE_M,
    TRAINING_TERRAIN_Y_HALF_SIZE_M,
    TrainingGroundReference,
    TrainingTerrainConfig,
)

D1_TERRAIN_OBSERVATION_SIZE = 44
D1_TERRAIN_OBSERVATION_SCHEMA = "d1-terrain-oracle-v1"
D1_TERRAIN_REWARD_SCHEMA = "d1-terrain-clearance-v1"
D1_TERRAIN_ACTION_SCHEMA = "d1-terrain-residual-quarter-v1"
D1_TERRAIN_RESIDUAL_SCALE = np.asarray((11.25, 20.0), dtype=np.float64)
TRAINING_WAVELENGTHS_M = (0.8, 1.0, 1.2)


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _validate_stage(stage: int) -> int:
    if isinstance(stage, bool) or not isinstance(stage, Integral) or stage not in range(4):
        raise ValueError("curriculum stage must be an integer from 0 through 3")
    return int(stage)


def sample_training_terrain(
    rng: np.random.Generator,
    stage: int,
) -> TrainingTerrainConfig:
    """Sample a fixed stage distribution; this function never sees test scores."""

    stage = _validate_stage(stage)
    probabilities = (
        (1.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.3, 0.35, 0.35),
        (0.25, 0.25, 0.5),
    )[stage]
    kind = str(rng.choice(("flat", "bumps", "ramp"), p=probabilities))
    if kind == "bumps":
        return TrainingTerrainConfig(
            kind=kind,
            amplitude_m=0.005 if stage == 1 else float(rng.uniform(0.005, 0.010)),
            wavelength_m=float(rng.choice(TRAINING_WAVELENGTHS_M)),
            phase_rad=float(rng.uniform(-np.pi, np.pi)),
        )
    if kind == "ramp":
        slopes = (-2.0, 2.0) if stage == 2 else (-4.0, -2.0, 2.0, 4.0)
        return TrainingTerrainConfig(kind=kind, slope_deg=float(rng.choice(slopes)))
    return TrainingTerrainConfig()


def calculate_d1_terrain_reward(
    *,
    ground_height_m: float,
    command_clearance_m: float,
    base_height_m: float,
    **kwargs: Any,
) -> D1RewardBreakdown:
    """Use the existing transparent weights with a relative-height target.

    Attitude remains world-upright, as in the controller. The terrain pitch and
    roll are observations only; they do not redefine the upright reward.
    """

    return calculate_d1_reward(
        base_height_m=base_height_m - ground_height_m,
        command_height_m=command_clearance_m,
        **kwargs,
    )


class D1TerrainResidualEnv(D1ResidualEnv):
    """Small-bump/ramp curriculum with a distinct observation/action contract.

    ``height_m`` in reset options means clearance above local ground, whereas
    the inherited D1Command height is world-z. Old 42-value policies cannot be
    loaded into this 44-value environment. Neither training nor evaluation
    uses the interactive demo's policy gate.
    """

    observation_schema = D1_TERRAIN_OBSERVATION_SCHEMA
    reward_schema = D1_TERRAIN_REWARD_SCHEMA
    action_schema = D1_TERRAIN_ACTION_SCHEMA

    def __init__(
        self,
        baseline: str = "lqr",
        episode_seconds: float = 4.0,
        training_mode: str = "curriculum",
        randomize: bool = False,
        stage: int = 0,
    ) -> None:
        if training_mode not in {"flat", "curriculum"}:
            raise ValueError("training_mode must be 'flat' or 'curriculum'")
        if not isinstance(randomize, bool):
            raise TypeError("randomize must be a bool")
        duration = _finite_scalar(episode_seconds, "episode_seconds")
        if duration < 0.01:
            raise ValueError("episode_seconds must be at least one control step (0.01 s)")
        self.training_mode = training_mode
        self.curriculum_stage = _validate_stage(stage)
        self._episode_stage = self.curriculum_stage
        self._terrain = TrainingTerrainConfig()
        self._command_clearance_m = 0.455
        self._termination_reason = "ongoing"
        self._initial_height_lift_m = 0.0
        super().__init__(
            baseline=baseline,
            episode_seconds=duration,
            randomize=randomize,
            plant=D1Plant(control_dt=0.01, training_terrain=self._terrain),
        )
        self.observation_space = spaces.Box(
            -5.0, 5.0, shape=(D1_TERRAIN_OBSERVATION_SIZE,), dtype=np.float32
        )

    def set_curriculum_stage(self, stage: int) -> None:
        """Set the next episode's stage without changing the current terrain."""

        self.curriculum_stage = _validate_stage(stage)

    def _ground_reference(self) -> TrainingGroundReference:
        # A final state may lie beyond the safe interior. Clamp only the query
        # for reporting that terminal transition; _has_fallen rejects it.
        x, y = self.plant.base_position[:2]
        if not np.isfinite((x, y)).all():
            x = y = 0.0
        return self.plant.training_ground_reference(
            float(np.clip(x, -TRAINING_TERRAIN_X_HALF_SIZE_M, TRAINING_TERRAIN_X_HALF_SIZE_M)),
            float(np.clip(y, -TRAINING_TERRAIN_Y_HALF_SIZE_M, TRAINING_TERRAIN_Y_HALF_SIZE_M)),
        )

    def _reset_base_height_m(self) -> float:
        ground = self.plant.training_ground_reference(0.0, 0.0)
        return ground.height_m + self.plant.nominal_base_height_m

    def _configure_episode(self, options: dict[str, Any]) -> None:
        allowed = {
            "terrain",
            "velocity_mps",
            "height_m",
            "randomize",
            "base_mass_scale",
            "damping_scale",
            "friction_scale",
            "actuator_strength_scale",
            "action_delay_steps",
            "sensor_noise",
            "initial_roll",
            "initial_pitch",
            "initial_yaw",
        }
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"unsupported terrain reset options: {sorted(unknown)}")
        if "randomize" in options and not isinstance(options["randomize"], bool):
            raise TypeError("randomize must be a bool")
        if "action_delay_steps" in options:
            delay = options["action_delay_steps"]
            if isinstance(delay, bool) or not isinstance(delay, Integral) or delay < 0:
                raise ValueError("action_delay_steps must be a non-negative integer")
        self._episode_stage = self.curriculum_stage
        if "terrain" in options:
            terrain = options["terrain"]
            if isinstance(terrain, dict):
                terrain = TrainingTerrainConfig(**terrain)
            if not isinstance(terrain, TrainingTerrainConfig):
                raise TypeError("terrain must be TrainingTerrainConfig or a dictionary")
        else:
            terrain = (
                TrainingTerrainConfig()
                if self.training_mode == "flat"
                else sample_training_terrain(self.np_random, self._episode_stage)
            )
        velocity = _finite_scalar(
            options.get("velocity_mps", self.np_random.uniform(0.20, 0.45)), "velocity_mps"
        )
        clearance = _finite_scalar(options.get("height_m", 0.455), "height_m")
        if not 0.0 <= velocity <= 0.6:
            raise ValueError("velocity_mps must be in [0, 0.6] for this forward-only curriculum")
        if not 0.35 <= clearance <= 0.55:
            raise ValueError("height_m clearance must be in [0.35, 0.55]")
        for key in ("initial_roll", "initial_pitch", "initial_yaw"):
            if key in options:
                _finite_scalar(options[key], key)
        self._terrain = terrain
        self.plant.set_training_terrain(terrain)
        self._training_velocity = velocity
        self._command_clearance_m = clearance
        self._termination_reason = "ongoing"
        inherited_options = {
            key: value
            for key, value in options.items()
            if key not in {"terrain", "velocity_mps", "height_m"}
        }
        # The legacy "training" scenario injects 90--170 N pushes regardless
        # of randomize. A separate scenario deliberately excludes those pushes.
        inherited_options["scenario"] = "terrain"
        inherited_options.setdefault("initial_pitch", 0.0)
        super()._configure_episode(inherited_options)
        self._lift_initial_penetration()
        self._publish_control_state(self._state_source.reset())

    def _lift_initial_penetration(self) -> None:
        """Lift an intersecting initial pose, without stepping the simulation."""

        self._initial_height_lift_m = 0.0
        for _ in range(4):
            lifts = [
                (-float(contact.dist) + 0.002) / max(abs(float(contact.frame[2])), 0.2)
                for contact in self.plant.data.contact
                if contact.dist < 0.0
                and (
                    int(contact.geom1) in self.plant.terrain_geom_ids
                    or int(contact.geom2) in self.plant.terrain_geom_ids
                )
            ]
            if not lifts:
                return
            lift = max(lifts)
            qpos, qvel = self.plant.simulation_state()
            qpos[2] += lift
            self.plant.set_simulation_state(qpos, qvel)
            self._initial_height_lift_m += lift
        raise RuntimeError("terrain reset still intersects the ground after lifting")

    def _command_at_step(self) -> D1Command:
        time_s = self._step_count * self.plant.control_dt
        progress = float(np.clip((time_s - 0.3) / 0.3, 0.0, 1.0))
        velocity = self._training_velocity * progress**2 * (3.0 - 2.0 * progress)
        return D1Command(
            forward_velocity_mps=velocity,
            base_height_m=self._ground_reference().height_m + self._command_clearance_m,
        )

    def _residual_force(self, applied_action: np.ndarray) -> np.ndarray:
        return applied_action * D1_TERRAIN_RESIDUAL_SCALE

    def _observation(self) -> np.ndarray:
        ground = self._ground_reference()
        # self._command belongs to the observed state, not the prior transition.
        observation = encode_d1_observation(
            state=self._state,
            command=self._command,
            baseline_longitudinal_force_n=self.controller.last_longitudinal_force_n,
            previous_applied_action=self._previous_applied_action,
        )
        if self._noise_scale:
            observation += self.np_random.normal(0.0, 0.008 * self._noise_scale, size=42).astype(
                np.float32
            )
        return np.clip(
            np.concatenate((observation, (ground.pitch_rad, ground.roll_rad))), -5.0, 5.0
        ).astype(np.float32)

    def _has_fallen(self) -> bool:
        x, y, z = self.plant.base_position
        roll, pitch, _ = self.plant.base_rpy
        if (
            not np.isfinite(self.plant.data.qpos).all()
            or not np.isfinite(self.plant.data.qvel).all()
        ):
            reason = "nonfinite"
        elif (
            abs(x) > TRAINING_TERRAIN_X_HALF_SIZE_M - 0.5
            or abs(y) > TRAINING_TERRAIN_Y_HALF_SIZE_M - 0.5
        ):
            reason = "out_of_terrain"
        elif z - self._ground_reference().height_m < 0.22:
            reason = "low_clearance"
        elif abs(roll) > 0.85:
            reason = "excess_roll"
        elif abs(pitch) > 0.85:
            reason = "excess_pitch"
        else:
            reason = "ongoing"
        self._termination_reason = reason
        return reason != "ongoing"

    def _reward_terms(
        self,
        *,
        command: D1Command,
        applied_action: np.ndarray,
        terminated: bool,
    ) -> D1RewardBreakdown:
        roll, pitch, _ = self.plant.base_rpy
        return calculate_d1_terrain_reward(
            ground_height_m=self._ground_reference().height_m,
            command_clearance_m=self._command_clearance_m,
            base_height_m=float(self.plant.base_position[2]),
            forward_velocity_mps=float(self.plant.base_velocity(local=True)[0][0]),
            roll_rad=float(roll),
            pitch_rad=float(pitch),
            joint_position=self.plant.joint_position,
            command_velocity_mps=command.forward_velocity_mps,
            normalized_action=applied_action,
            previous_normalized_action=self._previous_applied_action,
            undesired_contacts=self.plant.undesired_ground_contacts,
            terminated=terminated,
        )

    def _info(self, **kwargs: Any) -> dict[str, Any]:
        info = super()._info(**kwargs)
        ground = self._ground_reference()
        clearance = float(self.plant.base_position[2] - ground.height_m)
        torque = np.asarray(kwargs["torque_nm"])
        info.update(
            terrain_config=asdict(self._terrain),
            curriculum_stage=self._episode_stage,
            training_mode=self.training_mode,
            ground_height_m=ground.height_m,
            ground_pitch_rad=ground.pitch_rad,
            ground_roll_rad=ground.roll_rad,
            clearance_m=clearance,
            command_clearance_m=self._command_clearance_m,
            clearance_error_m=clearance - self._command_clearance_m,
            world_height_target_m=info["command_height_m"],
            reward_world_height_target_m=ground.height_m + self._command_clearance_m,
            velocity_error_mps=info["forward_velocity_mps"] - info["command_velocity_mps"],
            target_velocity_mps=self._training_velocity,
            episode_step=self._step_count,
            initial_height_lift_m=self._initial_height_lift_m,
            position_x_m=float(self.plant.base_position[0]),
            position_y_m=float(self.plant.base_position[1]),
            yaw_rad=float(self.plant.base_rpy[2]),
            termination_reason=self._termination_reason,
            torque_saturation_fraction=float(
                np.mean(np.abs(torque) >= self.plant.actuator_torque_limit_nm - 1e-8)
            ),
            policy_applied=False,
            policy_gated=False,
            observation_schema=self.observation_schema,
            reward_schema=self.reward_schema,
            action_schema=self.action_schema,
            residual_scale_n=D1_TERRAIN_RESIDUAL_SCALE.copy(),
            attitude_target="world_upright",
            ground_reference_source="simulation_oracle",
        )
        return info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        checked = np.asarray(action, dtype=np.float64)
        if checked.shape != (2,) or not np.isfinite(checked).all():
            raise ValueError("D1 terrain residual action must contain two finite values")
        observation, reward, terminated, truncated, info = super().step(checked)
        info["episode_step"] = self._step_count
        info["policy_applied"] = True
        if truncated and not terminated:
            info["termination_reason"] = "time_limit"
        return observation, reward, terminated, truncated, info
