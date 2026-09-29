"""New-command decision seam over the original single-step control transaction.

Importing this module constructs nothing. The root-owned runtime may create
one actual CoursePlant/provider/controller and then use inherited reset/step;
only prepare differs because the old D1MotionCommand forbids 1.6 m/s and has
no lateral/jump fields. The first pilot deliberately rejects those later
skills until their phase and reward contracts are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from full_drive_command_08 import FullDriveCommand, QualifiedCommandCaps

from wheel_legged_control.d1.control_context import D1ControlContext
from wheel_legged_control.d1.control_loop import D1ControlLoop, D1ControlTransition
from wheel_legged_control.d1.control_primitives import terrain_normal_to_rpy
from wheel_legged_control.d1.training_terrain import TrainingGroundReference

FULL_DRIVE_LOOP_SCHEMA = "d1-full-drive-single-prepared-decision-loop-v1"


@dataclass(frozen=True, slots=True)
class FullDriveWorldCommand:
    """Applied servo request with ground-relative clearance resolved to world z."""

    forward_velocity_mps: float
    lateral_velocity_mps: float
    yaw_rate_rps: float
    base_height_m: float
    roll_rad: float
    pitch_rad: float
    base_vertical_velocity_mps: float
    jump_phase: int
    raw_forward_velocity_mps: float
    raw_lateral_velocity_mps: float
    raw_yaw_rate_rps: float
    raw_jump_requested: bool


@dataclass(frozen=True, slots=True)
class FullDriveDecision:
    """One immutable current-tick request and matching provider publication."""

    motion_command: FullDriveCommand
    raw_operator_command: FullDriveCommand
    world_command: FullDriveWorldCommand
    ground: TrainingGroundReference
    context: D1ControlContext
    schema: str = FULL_DRIVE_LOOP_SCHEMA


@dataclass(frozen=True, slots=True)
class FullDriveTransition(D1ControlTransition):
    """Distinguish env-input policy action from action sent to targets.

    SB3 may already clip its sampled Gaussian before calling the environment;
    any true pre-clip sample needs a separate rollout callback record.
    """

    policy_input_action: np.ndarray
    policy_clipped_action: np.ndarray
    action_enabled: bool

    def __post_init__(self) -> None:
        D1ControlTransition.__post_init__(self)
        for name in ("policy_input_action", "policy_clipped_action"):
            value = np.asarray(getattr(self, name))
            if value.shape != (16,) or value.dtype.kind not in "fiu":
                raise TypeError(f"{name} must be a real shape-16 vector")
            checked = np.asarray(value, dtype=np.float64)
            if not np.isfinite(checked).all():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name,
                               np.frombuffer(np.ascontiguousarray(checked).tobytes(),
                                             dtype=np.float64))
        if type(self.action_enabled) is not bool:
            raise TypeError("action_enabled must be an explicit bool")
        expected = self.policy_clipped_action if self.action_enabled else np.zeros(16)
        if (not np.array_equal(np.clip(self.policy_input_action, -1.0, 1.0),
                               self.policy_clipped_action)
                or not np.array_equal(self.raw_action, expected)
                or not np.array_equal(self.receipt.normalized_action, expected)):
            raise ValueError("actor clip/gate differs from consumed physical action")


class FullDriveLoop(D1ControlLoop):
    """Own one prepare/infer/compute/plant.step cycle, with explicit capability."""

    schema = FULL_DRIVE_LOOP_SCHEMA

    def __init__(self, plant, provider, controller, *, caps: QualifiedCommandCaps) -> None:
        if not isinstance(caps, QualifiedCommandCaps):
            raise TypeError("full-drive loop needs explicit qualified command caps")
        super().__init__(plant, provider, controller)
        self.caps = caps

    def prepare(self, command: FullDriveCommand, *, raw_operator_command: FullDriveCommand | None = None
                ) -> FullDriveDecision:
        if not self._initialized:
            raise RuntimeError("reset must precede full-drive prepare")
        if not isinstance(command, FullDriveCommand):
            raise TypeError("applied request must be FullDriveCommand")
        raw = command if raw_operator_command is None else raw_operator_command
        if not isinstance(raw, FullDriveCommand):
            raise TypeError("raw operator request must be FullDriveCommand")
        self.caps.check(command)
        self.caps.check(raw)
        if (command.lateral_velocity_mps != 0.0 or command.jump_requested
                or raw.lateral_velocity_mps != 0.0 or raw.jump_requested):
            raise ValueError("first full-drive pilot has no lateral or jump skill")
        existing = self._decision
        if existing is not None:
            if (not isinstance(existing, FullDriveDecision)
                    or command != existing.motion_command
                    or raw != existing.raw_operator_command):
                raise RuntimeError("cannot replace the command after current-tick prepare")
            return existing
        state = self.provider.read()
        ground = self.provider.ground_reference()
        if not isinstance(ground, TrainingGroundReference):
            raise TypeError("provider must publish an actual local ground reference")
        roll, pitch = terrain_normal_to_rpy(
            -np.tan(ground.pitch_rad), np.tan(ground.roll_rad), float(state.base_rpy[2])
        )
        world = FullDriveWorldCommand(
            forward_velocity_mps=command.forward_velocity_mps,
            lateral_velocity_mps=command.lateral_velocity_mps,
            yaw_rate_rps=command.yaw_rate_rps,
            base_height_m=ground.height_m + command.clearance_m,
            roll_rad=roll, pitch_rad=pitch,
            base_vertical_velocity_mps=0.0, jump_phase=0,
            raw_forward_velocity_mps=raw.forward_velocity_mps,
            raw_lateral_velocity_mps=raw.lateral_velocity_mps,
            raw_yaw_rate_rps=raw.yaw_rate_rps,
            raw_jump_requested=raw.jump_requested,
        )
        proposal = self.controller.preview(world, state, ground)
        context = D1ControlContext(
            state, proposal, self.controller.action_schema,
            self.controller.action_size, self._receipt,
        )
        decision = FullDriveDecision(command, raw, world, ground, context)
        self._decision = decision
        return decision

    def step(self, policy_action: Any, *, push_force_world_n=None) -> FullDriveTransition:
        """Gate before the inherited physical step so its receipt is truthful."""
        decision = self._decision
        if decision is None:
            raise RuntimeError("prepare a full-drive decision before stepping")
        if not isinstance(decision, FullDriveDecision):
            raise TypeError("prepared decision has a different command schema")
        raw = np.asarray(policy_action)
        if raw.shape != (16,) or raw.dtype.kind not in "fiu":
            raise TypeError("policy action must be a real shape-16 vector")
        actor = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(actor).all():
            raise ValueError("policy action must be finite")
        clipped = np.clip(actor, -1.0, 1.0)
        enabled = self.controller.action_gate(decision.world_command)
        if type(enabled) is not bool:
            raise TypeError("controller action gate must return bool")
        physical = clipped if enabled else np.zeros(16, dtype=np.float64)
        transition = super().step(physical, push_force_world_n=push_force_world_n)
        try:
            return FullDriveTransition(
                decision=transition.decision, receipt=transition.receipt,
                raw_action=transition.raw_action,
                requested_torque_nm=transition.requested_torque_nm,
                state=transition.state, truth=transition.truth,
                policy_input_action=actor, policy_clipped_action=clipped,
                action_enabled=enabled,
            )
        except Exception:
            # The physical interval already completed; a receipt mismatch is
            # fatal and must never make this tick retryable.
            self._initialized = False
            raise

    def terminal_observation_decision(
        self, consumed: FullDriveDecision, transition: FullDriveTransition,
    ) -> FullDriveDecision:
        """Describe actual post-step state without preparing or consuming tick+1.

        The command is the already consumed servo/raw request, while state,
        ground, receipt and wheel integral are the actual post-control values.
        This is solely for Gym's terminal observation and cannot be stepped.
        """
        if (self._decision is not None or transition.decision is not consumed
                or self._receipt is not transition.receipt):
            raise RuntimeError("terminal observation requires the just completed transition")
        state = self.provider.read()
        if state is not transition.state:
            raise RuntimeError("terminal observation provider differs from post-control state")
        ground = self.provider.ground_reference()
        command = consumed.motion_command
        raw = consumed.raw_operator_command
        roll, pitch = terrain_normal_to_rpy(
            -np.tan(ground.pitch_rad), np.tan(ground.roll_rad), float(state.base_rpy[2])
        )
        world = FullDriveWorldCommand(
            forward_velocity_mps=command.forward_velocity_mps,
            lateral_velocity_mps=command.lateral_velocity_mps,
            yaw_rate_rps=command.yaw_rate_rps,
            base_height_m=ground.height_m + command.clearance_m,
            roll_rad=roll, pitch_rad=pitch,
            base_vertical_velocity_mps=0.0, jump_phase=0,
            raw_forward_velocity_mps=raw.forward_velocity_mps,
            raw_lateral_velocity_mps=raw.lateral_velocity_mps,
            raw_yaw_rate_rps=raw.yaw_rate_rps,
            raw_jump_requested=raw.jump_requested,
        )
        proposal = self.controller.terminal_observation_proposal(world, state, ground)
        context = D1ControlContext(
            state, proposal, self.controller.action_schema,
            self.controller.action_size, self._receipt,
        )
        return FullDriveDecision(command, raw, world, ground, context)
