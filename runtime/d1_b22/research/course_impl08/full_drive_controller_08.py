"""Bind the pure new 16D law to one synchronized D1 control decision.

The historical D1WheelLegController is used only for its current-state,
action-free nominal geometry preview. Its eight-action compute is never
called. One pure residual16 calculation consumes that preview, provider
state and one explicit wheel-integral snapshot. This is a new control schema,
not a bitwise claim about the previously qualified 06 composite controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from full_drive_loop_08 import FullDriveWorldCommand
from residual16_math_08 import (
    ACTION_SCHEMA,
    CONTROL_SCHEMA,
    JOINT_POSITION_HIGH_RAD,
    JOINT_POSITION_LOW_RAD,
    JOINT_TORQUE_LIMIT_NM,
    JOINT_VELOCITY_LIMIT_RAD_S,
    compute_residual16,
    nominal_support,
)

from wheel_legged_control.d1.control_context import (
    D1ControllerMemory,
    D1ControlProposal,
    D1JointTargetBaseline,
)
from wheel_legged_control.d1.model import (
    JOINT_POSITION_HIGH,
    JOINT_POSITION_LOW,
    JOINT_TORQUE_LIMIT,
    JOINT_VELOCITY_LIMIT,
)
from wheel_legged_control.d1.wheel_leg_controller import (
    NOMINAL_LEG_KD,
    NOMINAL_LEG_KP,
    D1WheelLegController,
)

ADAPTER_SCHEMA = "d1-course-newrl16-singlepi-damping-commonp-stopgate-v1"
RuntimeMode = Literal["test_actor", "train", "eval", "manual"]


@dataclass(frozen=True, slots=True)
class _CurrentPreview:
    command: FullDriveWorldCommand
    state: Any
    ground: Any
    nominal: Any
    proposal: D1ControlProposal


class FullDriveControllerAdapter:
    """One actual torque producer for the full-drive loop's 16-action policy."""

    action_size = 16
    action_schema = ACTION_SCHEMA
    control_schema = ADAPTER_SCHEMA

    def __init__(self, plant, *, mode: RuntimeMode,
                 enable_stationary_action_probe: bool = False) -> None:
        if mode not in ("test_actor", "train", "eval", "manual"):
            raise ValueError("unknown full-drive runtime mode")
        if type(enable_stationary_action_probe) is not bool:
            raise TypeError("stationary action probe switch must be bool")
        if enable_stationary_action_probe and mode != "test_actor":
            raise ValueError("stationary residual probe is forbidden in train/eval/manual")
        if (not np.array_equal(JOINT_POSITION_HIGH_RAD, JOINT_POSITION_HIGH)
                or not np.array_equal(JOINT_POSITION_LOW_RAD, JOINT_POSITION_LOW)
                or not np.array_equal(JOINT_TORQUE_LIMIT_NM, JOINT_TORQUE_LIMIT)
                or not np.array_equal(JOINT_VELOCITY_LIMIT_RAD_S, JOINT_VELOCITY_LIMIT)
                or not np.all(NOMINAL_LEG_KP[[i for i in range(16) if i % 4 != 3]] == 80.0)
                or not np.all(NOMINAL_LEG_KD[[i for i in range(16) if i % 4 != 3]] == 3.0)):
            raise RuntimeError("new 16D law's physical constants differ from D1 source")
        self.plant = plant
        self.mode = mode
        self.enable_stationary_action_probe = enable_stationary_action_probe
        self._nominal = D1WheelLegController(control_dt=plant.control_dt)
        if (self._nominal.wheel_kp != 2.2 or self._nominal.wheel_ki != 3.0
                or self._nominal.control_dt != 0.01
                or self._nominal._mass != plant.nominal_total_mass_kg):
            raise RuntimeError("nominal geometry/PI provenance differs from course plant")
        self.reset()

    def reset(self) -> None:
        self._nominal.reset()
        self._wheel_integral_nm = np.zeros(4, dtype=np.float64)
        self._previous_servo_forward_mps: float | None = None
        self._stop_latched = False
        self._preview: _CurrentPreview | None = None
        self.last_record: dict[str, Any] | None = None

    @property
    def wheel_integral_nm(self) -> np.ndarray:
        return self._wheel_integral_nm.copy()

    @property
    def stop_latched(self) -> bool:
        return self._stop_latched

    def action_gate(self, command: FullDriveWorldCommand) -> bool:
        """Raw request owns residual permission; E0 probe is explicit only."""
        if not isinstance(command, FullDriveWorldCommand):
            raise TypeError("action gate needs the current full-drive world command")
        raw_motion = bool(
            command.raw_forward_velocity_mps != 0.0
            or command.raw_lateral_velocity_mps != 0.0
            or command.raw_yaw_rate_rps != 0.0
            or command.raw_jump_requested
        )
        return bool(raw_motion or self.enable_stationary_action_probe)

    def preview(self, command, state, ground) -> D1ControlProposal:
        if not isinstance(command, FullDriveWorldCommand):
            raise TypeError("nominal preview needs FullDriveWorldCommand")
        if self._preview is not None:
            raise RuntimeError("one prepared controller preview must be consumed before another")
        nominal = self._nominal.nominal_targets(
            command, state, ground_height_m=ground.height_m,
        )
        proposal = D1ControlProposal(
            state.sequence, state.control_time_s,
            D1JointTargetBaseline(
                nominal.nominal_joint_target_rad,
                nominal.nominal_wheel_speed_rad_s,
            ),
            D1ControllerMemory(wheel_integral_nm=self._wheel_integral_nm),
        )
        self._preview = _CurrentPreview(command, state, ground, nominal, proposal)
        return proposal

    def terminal_observation_proposal(self, command, state, ground) -> D1ControlProposal:
        """Read-only post-control nominal for a real terminal observation.

        This has no pending compute and must not create or replace ``_preview``.
        The original nominal-target calculation is action-free and does not
        update the wheel integral; the current committed integral is published.
        """
        if not isinstance(command, FullDriveWorldCommand) or self._preview is not None:
            raise RuntimeError("terminal observation requires no pending controller preview")
        nominal = self._nominal.nominal_targets(
            command, state, ground_height_m=ground.height_m,
        )
        return D1ControlProposal(
            state.sequence, state.control_time_s,
            D1JointTargetBaseline(
                nominal.nominal_joint_target_rad,
                nominal.nominal_wheel_speed_rad_s,
            ),
            D1ControllerMemory(wheel_integral_nm=self._wheel_integral_nm),
        )

    def compute(self, command, state, ground, action) -> np.ndarray:
        preview = self._preview
        if preview is None or command is not preview.command or state is not preview.state or ground is not preview.ground:
            raise RuntimeError("controller compute must consume its exact current preview")
        support = nominal_support({
            "base_rpy_rad": state.base_rpy,
            "base_angular_velocity_world": state.base_angular_velocity_world,
            "base_rotation_world_from_body": state.base_rotation,
            "foot_offset_world": state.foot_offset_world,
            "foot_jacobian_world": state.foot_jacobian,
            "nominal_mass_kg": self._nominal._mass,
            "nominal_base_com_offset_body_m": self._nominal._com,
            "commanded_roll_rad": command.roll_rad,
            "commanded_pitch_rad": command.pitch_rad,
        })
        servo_forward = float(command.forward_velocity_mps)
        previous = self._previous_servo_forward_mps
        stop_before = self._stop_latched
        if servo_forward != 0.0:
            stop_after = False
        elif previous is not None and previous != 0.0:
            stop_after = True
        else:
            stop_after = stop_before
        gate = self.action_gate(command)
        result = compute_residual16({
            "joint_position_rad": state.joint_position,
            "joint_velocity_rad_s": state.joint_velocity,
            "nominal_joint_target_rad": preview.nominal.nominal_joint_target_rad,
            "nominal_wheel_speed_rad_s": preview.nominal.nominal_wheel_speed_rad_s,
            "support_torque_nm": support["support_torque_nm"],
            "base_rotation_world_from_body": state.base_rotation,
            "foot_jacobian_world": state.foot_jacobian,
            "body_com_forward_mps": state.base_linear_velocity_body[0],
            "action_enabled": gate,
            "leg_longitudinal_damping_active": servo_forward != 0.0 or stop_after,
            "body_common_p_active": servo_forward != 0.0,
        }, action, self._wheel_integral_nm)
        if result["schema"] != CONTROL_SCHEMA or result["action_schema"] != ACTION_SCHEMA:
            raise RuntimeError("pure 16D calculation returned another control schema")
        # The pure calculation and support check have completed. A later plant
        # failure latches the enclosing loop; reset is mandatory, never retry.
        self._wheel_integral_nm = result["wheel_integral_after_nm"].copy()
        self._previous_servo_forward_mps = servo_forward
        self._stop_latched = stop_after
        self._preview = None
        self.last_record = {
            "schema": ADAPTER_SCHEMA,
            "mode": self.mode,
            "test_actor_probe": self.enable_stationary_action_probe,
            "control_time_s": float(state.control_time_s),
            "provider_sequence": int(state.sequence),
            "raw_forward_mps": float(command.raw_forward_velocity_mps),
            "raw_lateral_mps": float(command.raw_lateral_velocity_mps),
            "raw_yaw_rps": float(command.raw_yaw_rate_rps),
            "servo_forward_mps": servo_forward,
            "servo_lateral_mps": float(command.lateral_velocity_mps),
            "servo_yaw_rps": float(command.yaw_rate_rps),
            "action_gate_enabled": gate,
            "previous_servo_forward_mps": previous,
            "stop_latched_before": stop_before,
            "stop_latched_after": stop_after,
            "nominal_effective_yaw_rps": preview.nominal.effective_yaw_request_rps,
            "nominal_leg_extension_m": preview.nominal.leg_extension_target_m,
            "nominal_support": support,
            "calculation": result,
        }
        return result["safe_torque_nm"].copy()
