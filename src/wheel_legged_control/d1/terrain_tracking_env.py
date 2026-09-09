"""D1 local-tangent tracking with explicit oracle observation and reward schemas."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from .control_primitives import terrain_normal_to_rpy
from .controllers import D1Command
from .rewards import D1RewardBreakdown
from .terrain_env import D1TerrainResidualEnv, _validate_stage


class D1TerrainTrackingEnv(D1TerrainResidualEnv):
    """Track the local collision-cell tangent with unchanged LQR/VMC gains.

    Height, spawn, action scale, terrain geometry, and command timing remain
    those of v1. Terrain information is a simulation oracle, not an estimator.
    """

    observation_schema = "d1-terrain-tracking-oracle-v2"
    reward_schema = "d1-terrain-tracking-v2"
    control_schema = "d1-lqr-vmc-local-tangent-v2"

    def __init__(
        self,
        baseline: str = "lqr",
        episode_seconds: float = 4.0,
        training_mode: str = "curriculum",
        randomize: bool = False,
        stage: int = 0,
    ) -> None:
        if baseline != "lqr":
            raise ValueError("the tracking-v2 control schema supports baseline='lqr' only")
        if training_mode not in {"flat", "mixed", "curriculum"}:
            raise ValueError("training_mode must be 'flat', 'mixed', or 'curriculum'")
        stage = _validate_stage(stage)
        super().__init__(
            baseline=baseline,
            episode_seconds=episode_seconds,
            training_mode="curriculum" if training_mode == "mixed" else training_mode,
            randomize=randomize,
            stage=3 if training_mode == "mixed" else stage,
        )
        self.training_mode = training_mode

    def set_curriculum_stage(self, stage: int) -> None:
        """Mixed sampling always stays at stage 3; curriculum changes on reset."""

        stage = _validate_stage(stage)
        if self.training_mode == "mixed" and stage != 3:
            raise ValueError("mixed mode is fixed at curriculum stage 3")
        super().set_curriculum_stage(stage)

    def _terrain_attitude_target(self) -> tuple[float, float]:
        ground = self._ground_reference()
        # The supported heightfields are extruded along y: dh/dy is zero.
        return terrain_normal_to_rpy(
            -math.tan(ground.pitch_rad), 0.0, float(self.plant.base_rpy[2])
        )

    def _command_at_step(self) -> D1Command:
        command = super()._command_at_step()
        roll, pitch = self._terrain_attitude_target()
        return replace(command, roll_rad=roll, pitch_rad=pitch)

    def _observation(self) -> np.ndarray:
        observation = super()._observation()
        roll, pitch = self._terrain_attitude_target()
        observation[-2:] = (pitch, roll)
        return observation

    def _reward_terms(
        self,
        *,
        command: D1Command,
        applied_action: np.ndarray,
        terminated: bool,
    ) -> D1RewardBreakdown:
        terms = super()._reward_terms(
            command=command, applied_action=applied_action, terminated=terminated
        )
        target_roll, target_pitch = self._terrain_attitude_target()
        roll, pitch, _ = self.plant.base_rpy
        # Replace only the attitude error. Every coefficient and other term is
        # inherited verbatim; a training-only reward scale belongs in a wrapper.
        upright = float(
            np.exp(-(((roll - target_roll) / 0.22) ** 2) - (((pitch - target_pitch) / 0.22) ** 2))
        )
        return replace(terms, upright=upright)

    def _info(self, **kwargs: Any) -> dict[str, Any]:
        info = super()._info(**kwargs)
        transition_command = kwargs["command"]
        target_roll, target_pitch = self._terrain_attitude_target()
        roll, pitch, _ = self.plant.base_rpy
        info.update(
            command_roll_rad=transition_command.roll_rad,
            command_pitch_rad=transition_command.pitch_rad,
            reward_roll_target_rad=target_roll,
            reward_pitch_target_rad=target_pitch,
            observation_roll_target_rad=target_roll,
            observation_pitch_target_rad=target_pitch,
            measured_roll_rad=float(roll),
            measured_pitch_rad=float(pitch),
            roll_error_rad=float(roll - target_roll),
            pitch_error_rad=float(pitch - target_pitch),
            control_schema=self.control_schema,
            attitude_target="local_collision_tangent",
        )
        return info
