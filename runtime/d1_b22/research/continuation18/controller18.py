"""C18 filtered-common-integral controller installed before first reset.

This module constructs no model on import. The installer keeps the already
constructed plant and provider, and replaces the controller/loop once before
the first reset. Variant choice is fixed for the lifetime of that environment.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from full_drive_controller_08 import FullDriveControllerAdapter
from residual18 import (
    ACTION_SCHEMA,
    CONTROL_SCHEMA,
    VARIANTS,
    compute_residual18,
    nominal_support,
)
from world_upright_course_11 import WorldUprightCourseEnv, WorldUprightLoop
from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry


ADAPTER_SCHEMA = "d1-course18-yawlimit-filteredcommonintegral-adapter-v1"
YAW_LIMIT_BY_VARIANT = {
    "baseline": .6,
    "yaw": 1.2,
    "ramp": .6,
    "combined": 1.2,
}
YAW_FEEDBACK_GAIN = 4.0


class FullDriveController18(FullDriveControllerAdapter):
    """Old nominal geometry and 16D action seam with explicit C18 arithmetic."""

    action_schema = ACTION_SCHEMA
    control_schema = ADAPTER_SCHEMA

    def __init__(self, plant, *, mode: str, variant: str,
                 enable_stationary_action_probe: bool = False) -> None:
        if type(variant) is not str or variant not in VARIANTS:
            raise ValueError("controller variant must be one of the four frozen C18 names")
        self.variant = variant
        super().__init__(
            plant, mode=mode,
            enable_stationary_action_probe=enable_stationary_action_probe,
        )
        if self._nominal.yaw_feedback_gain != YAW_FEEDBACK_GAIN:
            raise RuntimeError("nominal yaw feedback gain differs from frozen source")
        self._nominal.yaw_request_limit_rps = YAW_LIMIT_BY_VARIANT[variant]

    def reset(self) -> None:
        super().reset()
        self._wheel_common_reference_z_rad_s = 0.0

    @property
    def wheel_common_reference_z_rad_s(self) -> float:
        return float(self._wheel_common_reference_z_rad_s)

    def compute(self, command, state, ground, action) -> np.ndarray:
        preview = self._preview
        if (preview is None or command is not preview.command
                or state is not preview.state or ground is not preview.ground):
            raise RuntimeError("C18 compute must consume its exact current preview")
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
        result = compute_residual18({
            "joint_position_rad": state.joint_position,
            "joint_velocity_rad_s": state.joint_velocity,
            "nominal_joint_target_rad": preview.nominal.nominal_joint_target_rad,
            "nominal_wheel_speed_rad_s": preview.nominal.nominal_wheel_speed_rad_s,
            "support_torque_nm": support["support_torque_nm"],
            "base_rotation_world_from_body": state.base_rotation,
            "foot_jacobian_world": state.foot_jacobian,
            "body_com_forward_mps": state.base_linear_velocity_body[0],
            "servo_forward_mps": servo_forward,
            "controller_variant": self.variant,
            "action_enabled": gate,
            "leg_longitudinal_damping_active": servo_forward != 0.0 or stop_after,
            "body_common_p_active": servo_forward != 0.0,
        }, action, self._wheel_integral_nm, self._wheel_common_reference_z_rad_s)
        if result["schema"] != CONTROL_SCHEMA or result["action_schema"] != ACTION_SCHEMA:
            raise RuntimeError("C18 pure calculation returned another schema")
        self._wheel_integral_nm = result["wheel_integral_after_nm"].copy()
        self._wheel_common_reference_z_rad_s = result["wheel_common_reference_z_after_rad_s"]
        self._previous_servo_forward_mps = servo_forward
        self._stop_latched = stop_after
        self._preview = None
        pre_yaw = float(state.base_rpy[2])
        pre_yaw_rate = float(state.base_angular_velocity_body[2])
        servo_yaw = float(command.yaw_rate_rps)
        lateral_axis_world = np.asarray((-math.sin(pre_yaw), math.cos(pre_yaw), 0.0))
        self.last_record = {
            "schema": ADAPTER_SCHEMA,
            "controller_variant": self.variant,
            "mode": self.mode,
            "test_actor_probe": self.enable_stationary_action_probe,
            "control_time_s": float(state.control_time_s),
            "provider_sequence": int(state.sequence),
            "raw_forward_mps": float(command.raw_forward_velocity_mps),
            "raw_lateral_mps": float(command.raw_lateral_velocity_mps),
            "raw_yaw_rps": float(command.raw_yaw_rate_rps),
            "servo_forward_mps": servo_forward,
            "servo_lateral_mps": float(command.lateral_velocity_mps),
            "servo_yaw_rps": servo_yaw,
            "pre_body_yaw_rad": pre_yaw,
            "pre_body_yaw_rate_rps": pre_yaw_rate,
            "nominal_yaw_feedback_gain": YAW_FEEDBACK_GAIN,
            "nominal_yaw_limit_rps": YAW_LIMIT_BY_VARIANT[self.variant],
            "nominal_unclipped_yaw_rps": servo_yaw + YAW_FEEDBACK_GAIN*(servo_yaw-pre_yaw_rate),
            "nominal_unlimited_yaw_rps": servo_yaw + YAW_FEEDBACK_GAIN*(servo_yaw-pre_yaw_rate),
            "nominal_foot_lateral_m": state.foot_offset_world @ lateral_axis_world,
            "wheel_common_reference_z_before_rad_s": result["wheel_common_reference_z_before_rad_s"],
            "wheel_common_reference_z_after_rad_s": result["wheel_common_reference_z_after_rad_s"],
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


def install_controller18(env: WorldUprightCourseEnv, variant: str) -> FullDriveController18:
    """Atomically replace the C16 controller and loop before the first reset."""
    if type(variant) is not str or variant not in VARIANTS:
        raise ValueError("controller variant must be one of the four frozen C18 names")
    if type(env) is not WorldUprightCourseEnv or env.pure_test_components:
        raise TypeError("C18 installer needs the exact production world-upright env")
    old_loop = env.loop
    old_controller = env.controller
    if (type(old_loop) is not WorldUprightLoop
            or type(old_controller) is not FullDriveControllerAdapter
            or env._steps is not None or env._episode_index != -1 or env._active
            or env.decision is not None or env.last_transition is not None
            or old_loop._initialized or old_loop._decision is not None
            or old_loop._receipt is not None or old_controller._preview is not None
            or old_loop.plant is not env.plant
            or old_loop.controller is not old_controller
            or old_controller.plant is not env.plant):
        raise RuntimeError("C18 controller installation requires the never-reset source chain")
    cache_before = _nominal_geometry.cache_info()
    if cache_before.currsize != 1:
        raise RuntimeError("C18 nominal geometry must already be warm")
    controller = FullDriveController18(
        env.plant, mode=env.mode, variant=variant,
        enable_stationary_action_probe=old_controller.enable_stationary_action_probe,
    )
    cache_after = _nominal_geometry.cache_info()
    if cache_after.misses != cache_before.misses or cache_after.hits != cache_before.hits + 1:
        raise RuntimeError("C18 nominal geometry construction was not a cache hit")
    loop = WorldUprightLoop(env.plant, old_loop.provider, controller, caps=env.caps)
    if (loop.plant is not env.plant or loop.controller is not controller
            or loop.provider is not old_loop.provider or controller.mode != env.mode
            or loop.caps is not env.caps):
        raise RuntimeError("C18 replacement chain identity differs")
    env.controller, env.loop = controller, loop
    return controller


__all__ = (
    "ADAPTER_SCHEMA", "FullDriveController18", "YAW_FEEDBACK_GAIN",
    "YAW_LIMIT_BY_VARIANT", "install_controller18",
)
