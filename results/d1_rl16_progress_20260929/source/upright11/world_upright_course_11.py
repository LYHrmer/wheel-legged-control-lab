"""New world-upright course task over the unchanged compiled geometry.

The vertical ray's height, normal and geom identity remain physical evidence.
World roll/pitch zero is an explicit *task reference*, not a modified hit.
Importing this module does not construct a model or run a physical step.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from full_drive_command_08 import FullDriveCommand
from full_drive_env_08 import SAFE_X_M, SAFE_Y_M, FullDriveCourseEnv
from full_drive_loop_08 import (
    FullDriveDecision,
    FullDriveLoop,
    FullDriveWorldCommand,
)

from wheel_legged_control.d1.control_context import D1ControlContext
from wheel_legged_control.d1.training_terrain import TrainingGroundReference

WORLD_UPRIGHT_REFERENCE_SCHEMA = "d1-course-world-upright-roll-pitch-zero-v1"
WORLD_UPRIGHT_LOOP_SCHEMA = "d1-full-drive-world-upright-loop-v1"
WORLD_UPRIGHT_TASK_SCHEMA = "d1-course-world-upright-rl16-task-v1"
WORLD_UPRIGHT_REWARD_SCHEMA = "d1-course-world-upright-bodycom-yaw-terrain-action-torque-v1"


def world_upright_command(applied: FullDriveCommand, raw: FullDriveCommand,
                          ground: TrainingGroundReference) -> FullDriveWorldCommand:
    """Preserve the applied/raw command and real height; target world R/P zero."""
    if (not isinstance(applied, FullDriveCommand)
            or not isinstance(raw, FullDriveCommand)
            or not isinstance(ground, TrainingGroundReference)):
        raise TypeError("world-upright command needs typed applied/raw and actual ground")
    if not np.isfinite((ground.height_m, ground.pitch_rad, ground.roll_rad)).all():
        raise ValueError("actual compiled ground reference is nonfinite")
    base_height = ground.height_m + applied.clearance_m
    if not math.isfinite(base_height):
        raise ValueError("world-upright base height is nonfinite")
    return FullDriveWorldCommand(
        forward_velocity_mps=applied.forward_velocity_mps,
        lateral_velocity_mps=applied.lateral_velocity_mps,
        yaw_rate_rps=applied.yaw_rate_rps,
        base_height_m=base_height,
        roll_rad=0.0, pitch_rad=0.0,
        base_vertical_velocity_mps=0.0, jump_phase=0,
        raw_forward_velocity_mps=raw.forward_velocity_mps,
        raw_lateral_velocity_mps=raw.lateral_velocity_mps,
        raw_yaw_rate_rps=raw.yaw_rate_rps,
        raw_jump_requested=raw.jump_requested,
    )


class WorldUprightLoop(FullDriveLoop):
    """Own one new world reference, while inheriting the original single step."""

    schema = WORLD_UPRIGHT_LOOP_SCHEMA

    def prepare(self, command: FullDriveCommand, *,
                raw_operator_command: FullDriveCommand | None = None) -> FullDriveDecision:
        if not self._initialized:
            raise RuntimeError("reset must precede world-upright prepare")
        if not isinstance(command, FullDriveCommand):
            raise TypeError("applied request must be FullDriveCommand")
        raw = command if raw_operator_command is None else raw_operator_command
        if not isinstance(raw, FullDriveCommand):
            raise TypeError("raw request must be FullDriveCommand")
        self.caps.check(command)
        self.caps.check(raw)
        if (command.lateral_velocity_mps != 0.0 or command.jump_requested
                or raw.lateral_velocity_mps != 0.0 or raw.jump_requested):
            raise ValueError("first world-upright pilot has no lateral or jump skill")
        existing = self._decision
        if existing is not None:
            if (not isinstance(existing, FullDriveDecision)
                    or command != existing.motion_command
                    or raw != existing.raw_operator_command):
                raise RuntimeError("current-tick prepared command cannot be replaced")
            return existing
        state = self.provider.read()
        ground = self.provider.ground_reference()
        world = world_upright_command(command, raw, ground)
        proposal = self.controller.preview(world, state, ground)
        context = D1ControlContext(
            state, proposal, self.controller.action_schema,
            self.controller.action_size, self._receipt,
        )
        decision = FullDriveDecision(command, raw, world, ground, context,
                                     schema=WORLD_UPRIGHT_LOOP_SCHEMA)
        self._decision = decision
        return decision

    def terminal_observation_decision(self, consumed, transition) -> FullDriveDecision:
        if (self._decision is not None or transition.decision is not consumed
                or self._receipt is not transition.receipt):
            raise RuntimeError("terminal observation requires the just-completed interval")
        state = self.provider.read()
        if state is not transition.state:
            raise RuntimeError("terminal provider state differs from the real post state")
        ground = self.provider.ground_reference()
        world = world_upright_command(
            consumed.motion_command, consumed.raw_operator_command, ground,
        )
        proposal = self.controller.terminal_observation_proposal(world, state, ground)
        context = D1ControlContext(
            state, proposal, self.controller.action_schema,
            self.controller.action_size, self._receipt,
        )
        return FullDriveDecision(
            consumed.motion_command, consumed.raw_operator_command,
            world, ground, context, schema=WORLD_UPRIGHT_LOOP_SCHEMA,
        )


class WorldUprightCourseEnv(FullDriveCourseEnv):
    """One real CoursePlant with a distinct world-leveling task identity."""

    task_schema = WORLD_UPRIGHT_TASK_SCHEMA
    reward_schema = WORLD_UPRIGHT_REWARD_SCHEMA
    reference_schema = WORLD_UPRIGHT_REFERENCE_SCHEMA

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        previous = self.loop
        if self.pure_test_components:
            if not isinstance(previous, WorldUprightLoop):
                raise RuntimeError("pure test injection requires the actual new loop class")
            return
        if (type(previous) is not FullDriveLoop or self._steps is not None
                or self._active or self._episode_index != -1
                or self.decision is not None or self.last_transition is not None
                or previous._initialized or previous._decision is not None
                or previous._receipt is not None or self.controller._preview is not None
                or previous.plant is not self.plant or previous.controller is not self.controller):
            raise RuntimeError("world-upright loop replacement requires a never-reset env")
        self.loop = WorldUprightLoop(
            self.plant, previous.provider, self.controller, caps=self.caps,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, _ = super().reset(seed=seed, options=options)
        self._episode_metadata.update({
            "task_schema": self.task_schema,
            "reward_schema": self.reward_schema,
            "reference_schema": self.reference_schema,
            "reference_roll_rad": 0.0,
            "reference_pitch_rad": 0.0,
            "geometry_ground_source": "unaltered_compiled_course_vertical_hit",
            "control_loop_schema": self.loop.schema,
        })
        return observation, {"episode_metadata": self.episode_metadata}

    def _endpoint(self, transition, native_interval):
        metrics, _ = super()._endpoint(transition, native_interval)
        truth = transition.truth
        task_roll = float(truth.base_rpy[0])
        task_pitch = float(truth.base_rpy[1])
        if not math.isfinite(task_roll) or not math.isfinite(task_pitch):
            raise RuntimeError("world-upright task attitude is nonfinite")
        metrics = dict(metrics)
        metrics["task_roll_error_rad"] = task_roll
        metrics["task_pitch_error_rad"] = task_pitch
        metrics["reference_roll_rad"] = 0.0
        metrics["reference_pitch_rad"] = 0.0
        undesired = int(self.plant.undesired_ground_contacts)
        outside = bool(abs(truth.base_position[0]) > SAFE_X_M
                       or abs(truth.base_position[1]) > SAFE_Y_M)
        reason = (
            "nonwheel_ground_contact"
            if native_interval["nonwheel_contact_count"] > 0 or undesired > 0 else
            "course_map_boundary" if outside else
            "fall_or_low_clearance"
            if (metrics["clearance_m"] < 0.28
                or max(abs(task_roll), abs(task_pitch)) > math.radians(20))
            else None
        )
        return metrics, reason

    def _reward(self, transition, metrics: dict[str, float], terminated: bool):
        terms = super()._reward(transition, metrics, terminated)
        roll, pitch = metrics["task_roll_error_rad"], metrics["task_pitch_error_rad"]
        terms["attitude"] = -0.2 * ((roll / 0.2) ** 2 + (pitch / 0.2) ** 2)
        if not np.isfinite(tuple(terms.values())).all():
            raise RuntimeError("world-upright reward terms are nonfinite")
        return terms

    def step(self, policy_action):
        observation, reward, terminated, truncated, info = super().step(policy_action)
        info = dict(info)
        info["task_schema"] = self.task_schema
        info["reward_schema"] = self.reward_schema
        info["reference_schema"] = self.reference_schema
        info["reference_roll_rad"] = 0.0
        info["reference_pitch_rad"] = 0.0
        return observation, reward, terminated, truncated, info


__all__ = (
    "WORLD_UPRIGHT_LOOP_SCHEMA",
    "WORLD_UPRIGHT_REFERENCE_SCHEMA",
    "WORLD_UPRIGHT_REWARD_SCHEMA",
    "WORLD_UPRIGHT_TASK_SCHEMA",
    "WorldUprightCourseEnv",
    "WorldUprightLoop",
    "world_upright_command",
)
