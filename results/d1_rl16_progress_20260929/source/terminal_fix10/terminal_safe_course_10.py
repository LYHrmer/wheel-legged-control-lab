"""Narrow observation-only fallback for a genuine course task termination.

The compiled terrain, controller, action, reward, step order and task failure
remain those of the frozen 08 implementation. In particular this module never
turns a non-executable terminal ground normal into a valid next control target.
No model is created by importing this module.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from full_drive_env_08 import FullDriveCourseEnv
from full_drive_loop_08 import FullDriveDecision, FullDriveLoop, FullDriveWorldCommand

from wheel_legged_control.d1.control_context import (
    D1ControlContext,
    D1ControllerMemory,
    D1ControlProposal,
    D1JointTargetBaseline,
)
from wheel_legged_control.d1.control_primitives import terrain_normal_to_rpy

TERMINAL_FALLBACK_SOURCE = "consumed_precontrol_baseline_for_nonexecutable_task_terminal"
ORIGINAL_POST_SOURCE = "original_poststate_nominal_within_executable_envelope"


def _terminal_world(consumed: FullDriveDecision, state: Any, ground: Any
                    ) -> FullDriveWorldCommand:
    """Exactly the original terminal command conversion, without side effects."""
    command = consumed.motion_command
    raw = consumed.raw_operator_command
    terrain = np.asarray((ground.height_m, ground.pitch_rad, ground.roll_rad), dtype=np.float64)
    pose = np.asarray(state.base_rpy, dtype=np.float64)
    if (terrain.shape != (3,) or pose.shape != (3,)
            or not np.isfinite(terrain).all() or not np.isfinite(pose).all()):
        raise ValueError("terminal ground/state attitude must be finite")
    roll, pitch = terrain_normal_to_rpy(
        -np.tan(ground.pitch_rad), np.tan(ground.roll_rad), float(pose[2]),
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
    values = np.asarray((
        world.forward_velocity_mps, world.lateral_velocity_mps, world.yaw_rate_rps,
        world.base_height_m, world.roll_rad, world.pitch_rad,
        world.base_vertical_velocity_mps, ground.height_m,
    ), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("terminal world command must be finite")
    return world


def _original_extension_violations(world: FullDriveWorldCommand, ground: Any
                                   ) -> tuple[str, ...]:
    """Use the original _extension clearance and tilt inequalities verbatim."""
    clearance = world.base_height_m - ground.height_m
    values = np.asarray((clearance, world.roll_rad, world.pitch_rad), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("terminal clearance/attitude must be finite")
    violations = []
    if not 0.32 <= clearance <= 0.56:
        violations.append("nominal_clearance_outside_0p32_to_0p56_m")
    if max(abs(world.roll_rad), abs(world.pitch_rad)) > 0.3:
        violations.append("nominal_roll_or_pitch_above_0p3_rad")
    return tuple(violations)


class TerminalSafeFullDriveLoop(FullDriveLoop):
    """Preserve the original executable path; isolate failed-terminal encoding."""

    def __init__(self, plant: Any, provider: Any, controller: Any, *, caps: Any) -> None:
        super().__init__(plant, provider, controller, caps=caps)
        self._terminal_transition: Any | None = None
        self._terminal_reason: str | None = None
        self.terminal_observation_provenance: dict[str, Any] | None = None

    def clear_terminal_marker(self) -> None:
        self._terminal_transition = None
        self._terminal_reason = None
        self.terminal_observation_provenance = None

    def mark_real_task_termination(self, transition: Any, reason: str) -> None:
        if (type(reason) is not str or not reason
                or self._decision is not None
                or self._receipt is not transition.receipt):
            raise RuntimeError("terminal marker requires the just-completed physical transition")
        self._terminal_transition = transition
        self._terminal_reason = reason

    def terminal_observation_decision(self, consumed, transition):
        real_termination = (self._terminal_transition is transition
                            and self._terminal_reason is not None)
        if (not real_termination or self._decision is not None
                or transition.decision is not consumed
                or self._receipt is not transition.receipt
                or self.controller._preview is not None):
            result = super().terminal_observation_decision(consumed, transition)
            self.terminal_observation_provenance = {
                "nominal_source": ORIGINAL_POST_SOURCE,
                "terminal_tick": result.context.proposal.tick,
                "task_failure_reason": self._terminal_reason if real_termination else None,
                "fallback_used": False,
            }
            return result

        state = self.provider.read()
        if state is not transition.state:
            raise RuntimeError("terminal observation provider differs from actual post-control state")
        ground = self.provider.ground_reference()
        world = _terminal_world(consumed, state, ground)
        violations = _original_extension_violations(world, ground)
        if not violations:
            result = super().terminal_observation_decision(consumed, transition)
            self.terminal_observation_provenance = {
                "nominal_source": ORIGINAL_POST_SOURCE,
                "terminal_tick": result.context.proposal.tick,
                "task_failure_reason": self._terminal_reason,
                "fallback_used": False,
            }
            return result

        baseline = consumed.context.proposal.baseline
        if not isinstance(baseline, D1JointTargetBaseline):
            raise TypeError("consumed terminal baseline must contain real leg/wheel targets")
        if (state.sequence != transition.receipt.tick + 1
                or not math.isclose(state.control_time_s, transition.receipt.end_time_s,
                                    rel_tol=0.0, abs_tol=1e-10)):
            raise RuntimeError("terminal fallback does not match the returned physical interval")
        wheel_integral = self.controller.wheel_integral_nm
        proposal = D1ControlProposal(
            state.sequence, state.control_time_s, baseline,
            D1ControllerMemory(wheel_integral_nm=wheel_integral),
        )
        context = D1ControlContext(
            state, proposal, self.controller.action_schema,
            self.controller.action_size, transition.receipt,
        )
        result = FullDriveDecision(
            consumed.motion_command, consumed.raw_operator_command,
            world, ground, context,
        )
        self.terminal_observation_provenance = {
            "nominal_source": TERMINAL_FALLBACK_SOURCE,
            "source_tick": consumed.context.proposal.tick,
            "terminal_tick": proposal.tick,
            "envelope_violations": list(violations),
            "original_nominal_clearance_m": world.base_height_m - ground.height_m,
            "original_nominal_roll_rad": world.roll_rad,
            "original_nominal_pitch_rad": world.pitch_rad,
            "task_failure_reason": self._terminal_reason,
            "fallback_used": True,
            "observation_only_not_executable": True,
            "additional_prepare_or_control_calls": 0,
        }
        return result


class TerminalSafeCourseEnv(FullDriveCourseEnv):
    """Use one fresh original CoursePlant with the narrow terminal-safe loop."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        previous = self.loop
        if self.pure_test_components:
            if not isinstance(previous, TerminalSafeFullDriveLoop):
                raise RuntimeError("pure test injection must supply the actual new loop class")
            return
        if (type(previous) is not FullDriveLoop or self._steps is not None
                or self._active or self._episode_index != -1
                or self.decision is not None or self.last_transition is not None
                or previous._initialized or previous._decision is not None
                or previous._receipt is not None or self.controller._preview is not None
                or previous.plant is not self.plant or previous.controller is not self.controller):
            raise RuntimeError("terminal-safe loop replacement requires one never-reset fresh env")
        self.loop = TerminalSafeFullDriveLoop(
            self.plant, previous.provider, self.controller, caps=self.caps,
        )

    def _endpoint(self, transition, native_interval):
        metrics, reason = super()._endpoint(transition, native_interval)
        if reason is not None:
            self.loop.mark_real_task_termination(transition, reason)
        return metrics, reason

    def step(self, policy_action):
        self.loop.clear_terminal_marker()
        observation, reward, terminated, truncated, info = super().step(policy_action)
        if terminated or truncated:
            provenance = self.loop.terminal_observation_provenance
            if provenance is None:
                raise RuntimeError("terminal observation lacks source provenance")
            info = dict(info)
            info["terminal_observation_provenance"] = dict(provenance)
            if (provenance["fallback_used"] and not terminated
                    or provenance["fallback_used"] and provenance["task_failure_reason"]
                    != info["terminal_reason"]):
                raise RuntimeError("nonexecutable terminal fallback changed the real task result")
        return observation, reward, terminated, truncated, info


__all__ = (
    "ORIGINAL_POST_SOURCE", "TERMINAL_FALLBACK_SOURCE",
    "TerminalSafeCourseEnv", "TerminalSafeFullDriveLoop",
)
