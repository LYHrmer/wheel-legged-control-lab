"""Pure C17 16D controller with a selectable common wheel-integral feedback.

This is a NEW control law for the 16-action residual schema of
astra_rl_first_v2_addendum_08.md, not an inherited qualification of the
published 06 fixed-policy record. Nothing here imports the engine, builds a
model, keeps mutable state or owns a controller class: every call is a pure
function of the arguments, and every stage is returned so the adapter and the
validator can audit it.

Bound inputs are the adapter's responsibility: ``nominal_joint_target_rad``,
``nominal_wheel_speed_rad_s`` and ``support_torque_nm`` must come from the
actual decision of the tick, and the three booleans must come from the consumed
command / stop latch. This module never re-derives them, never infers contact
from torque or velocity signs, and never runs the old 8D law.

The wheel PI integral is advanced exactly once per call, also when
``action_enabled`` is false. Ramp/combined use the common body-forward error
for its mean whenever common wheel P is active; the differential wheel-error
component remains. The candidate torque is tested before common P is added,
and unwind uses the chosen integral error's sign. There is no second integrator.

The leg damping gain and the common wheel-P formula are the published 06
baseline components (drive_damping_math_04.py, body_speed_math_06.py); they are
traditional terms, credited to that baseline and not to the policy.
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np

CONTROL_SCHEMA = "d1-course17-residual16-common-integral-v1"
ACTION_SCHEMA = "d1-course-residual16-leg-angle-wheel-speed-v1"
SUPPORT_SCHEMA = "d1-course-nominal-gravity-attitude-support-v1"

CONTROL_DT_S = 0.01
WHEEL_INDICES = (3, 7, 11, 15)
LEG_JOINT_INDICES = tuple(index for index in range(16) if index % 4 != 3)

LEG_KP = 80.0
LEG_KD = 3.0
WHEEL_KP = 2.2
WHEEL_KI = 3.0
WHEEL_INTEGRAL_LIMIT_NM = 4.0
WHEEL_SPEED_LIMIT_RAD_S = 30.0
WHEEL_RADIUS_M = 0.087
DRIVE_DAMPING_GAIN_NSPM = 126.4374005337902

ATTITUDE_KP = 180.0
ATTITUDE_KD = 24.0
GRAVITY_MPS2 = 9.81
SUPPORT_FORCE_FRACTION = 0.65

LEG_ACTION_SCALE_RAD = np.tile((0.12, 0.25, 0.25), 4)
WHEEL_ACTION_SCALE_RAD_S = 4.0

JOINT_TORQUE_LIMIT_NM = np.tile((80.0, 80.0, 80.0, 12.0), 4)
JOINT_VELOCITY_LIMIT_RAD_S = np.tile((20.0, 20.0, 20.0, 30.0), 4)
JOINT_POSITION_LOW_RAD = np.tile((-0.785398, -1.8326, -2.775, -np.inf), 4)
JOINT_POSITION_HIGH_RAD = np.tile((0.785398, 3.40339, -0.855, np.inf), 4)
for _array in (
    LEG_ACTION_SCALE_RAD,
    JOINT_TORQUE_LIMIT_NM,
    JOINT_VELOCITY_LIMIT_RAD_S,
    JOINT_POSITION_LOW_RAD,
    JOINT_POSITION_HIGH_RAD,
):
    _array.setflags(write=False)

ROTATION_TOLERANCE = 1e-10
ANTIWINDUP_SCOPE = (
    "wheel integral committed once per call against the differential wheel PI "
    "request before the common wheel-P increment; unwind product uses the "
    "selected wheel integral error"
)
VARIANTS = ("baseline", "yaw", "ramp", "combined")
COMMON_INTEGRAL_VARIANTS = ("ramp", "combined")

RESIDUAL_ARRAY_SHAPES = {
    "joint_position_rad": (16,),
    "joint_velocity_rad_s": (16,),
    "nominal_joint_target_rad": (16,),
    "nominal_wheel_speed_rad_s": (4,),
    "support_torque_nm": (16,),
    "base_rotation_world_from_body": (3, 3),
    "foot_jacobian_world": (4, 3, 4),
}
RESIDUAL_SCALAR_KEYS = ("body_com_forward_mps", "servo_forward_mps")
RESIDUAL_GATE_KEYS = (
    "action_enabled",
    "leg_longitudinal_damping_active",
    "body_common_p_active",
)
RESIDUAL_INPUT_KEYS = (
    tuple(RESIDUAL_ARRAY_SHAPES) + RESIDUAL_SCALAR_KEYS + RESIDUAL_GATE_KEYS
    + ("controller_variant",)
)

SUPPORT_ARRAY_SHAPES = {
    "base_rpy_rad": (3,),
    "base_angular_velocity_world": (3,),
    "base_rotation_world_from_body": (3, 3),
    "foot_offset_world": (4, 3),
    "foot_jacobian_world": (4, 3, 4),
    "nominal_base_com_offset_body_m": (3,),
}
SUPPORT_SCALAR_KEYS = ("nominal_mass_kg", "commanded_roll_rad", "commanded_pitch_rad")
SUPPORT_INPUT_KEYS = tuple(SUPPORT_ARRAY_SHAPES) + SUPPORT_SCALAR_KEYS


def _finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    """Return an independent float64 copy; bool and object arrays are rejected."""
    array = np.asarray(value)
    if array.dtype.kind not in ("f", "i", "u") or array.shape != shape:
        raise TypeError(f"{label} must be a real numeric array of shape {shape}")
    checked = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(checked).all():
        raise ValueError(f"{label} must be finite")
    return checked


def _finite_scalar(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real numeric scalar")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _gate(value: Any, label: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an explicit boolean gate")
    return bool(value)


def _rotation(value: Any, label: str) -> np.ndarray:
    matrix = _finite_array(value, (3, 3), label)
    orthonormal = float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
    determinant = float(np.linalg.det(matrix))
    if orthonormal > ROTATION_TOLERANCE or abs(determinant - 1.0) > ROTATION_TOLERANCE:
        raise ValueError(
            f"{label} must be orthonormal with determinant +1 within "
            f"{ROTATION_TOLERANCE}"
        )
    return matrix


def _checked_keys(inputs: Any, required: tuple[str, ...], label: str) -> None:
    if not isinstance(inputs, dict):
        raise TypeError(f"{label} must be a dict")
    missing = tuple(key for key in required if key not in inputs)
    if missing:
        raise KeyError(f"{label} is missing {missing}")
    unknown = tuple(key for key in inputs if key not in required)
    if unknown:
        raise TypeError(f"{label} carries unsupported keys {unknown}")


def _frozen(value: Any) -> np.ndarray:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


def compute_residual17(
    inputs: dict, action: Any, wheel_integral_before_nm: Any
) -> dict[str, Any]:
    """Return every stage of one 16D residual control tick.

    The result is self-contained: consumed raw inputs, the raw/clipped/applied
    action, each target stage with its limiting mask, the single wheel PI
    update, the two traditional increments and both the unprotected request and
    the protected torque with all protection masks. Inputs are never mutated and
    all returned arrays are independent read-only copies.
    """
    _checked_keys(inputs, RESIDUAL_INPUT_KEYS, "inputs")
    joint_position = _finite_array(inputs["joint_position_rad"], (16,), "joint_position_rad")
    joint_velocity = _finite_array(
        inputs["joint_velocity_rad_s"], (16,), "joint_velocity_rad_s"
    )
    nominal_target = _finite_array(
        inputs["nominal_joint_target_rad"], (16,), "nominal_joint_target_rad"
    )
    nominal_wheel = _finite_array(
        inputs["nominal_wheel_speed_rad_s"], (4,), "nominal_wheel_speed_rad_s"
    )
    support = _finite_array(inputs["support_torque_nm"], (16,), "support_torque_nm")
    rotation = _rotation(
        inputs["base_rotation_world_from_body"], "base_rotation_world_from_body"
    )
    jacobian = _finite_array(inputs["foot_jacobian_world"], (4, 3, 4), "foot_jacobian_world")
    body_forward = _finite_scalar(inputs["body_com_forward_mps"], "body_com_forward_mps")
    servo_forward = _finite_scalar(inputs["servo_forward_mps"], "servo_forward_mps")
    variant = inputs["controller_variant"]
    if type(variant) is not str or variant not in VARIANTS:
        raise ValueError("controller_variant must be one of the four C17 variants")
    action_enabled = _gate(inputs["action_enabled"], "action_enabled")
    damping_active = _gate(
        inputs["leg_longitudinal_damping_active"], "leg_longitudinal_damping_active"
    )
    common_p_active = _gate(inputs["body_common_p_active"], "body_common_p_active")
    if not np.all(support[list(WHEEL_INDICES)] == 0.0):
        raise ValueError("support_torque_nm wheel entries must be exactly zero")

    integral_before = _finite_array(
        wheel_integral_before_nm, (4,), "wheel_integral_before_nm"
    )
    if np.any(np.abs(integral_before) > WHEEL_INTEGRAL_LIMIT_NM):
        raise ValueError(
            f"wheel_integral_before_nm must lie within +/-{WHEEL_INTEGRAL_LIMIT_NM} Nm"
        )

    raw_action = _finite_array(action, (16,), "action")
    clipped_action = np.clip(raw_action, -1.0, 1.0)
    action_clip_mask = clipped_action != raw_action
    applied_action = clipped_action.copy() if action_enabled else np.zeros(16)

    leg_offset = LEG_ACTION_SCALE_RAD * applied_action[:12]
    wheel_offset = WHEEL_ACTION_SCALE_RAD_S * applied_action[12:]

    geometric_target = nominal_target.copy()
    geometric_target[list(LEG_JOINT_INDICES)] += leg_offset
    rate_limited_target = np.clip(
        geometric_target,
        joint_position - JOINT_VELOCITY_LIMIT_RAD_S * CONTROL_DT_S,
        joint_position + JOINT_VELOCITY_LIMIT_RAD_S * CONTROL_DT_S,
    )
    rate_limit_mask = rate_limited_target != geometric_target
    position_limited_target = np.clip(
        rate_limited_target, JOINT_POSITION_LOW_RAD, JOINT_POSITION_HIGH_RAD
    )
    position_clip_mask = position_limited_target != rate_limited_target
    joint_target = position_limited_target.copy()
    joint_target[list(WHEEL_INDICES)] = joint_position[list(WHEEL_INDICES)]

    leg_pd = LEG_KP * (joint_target - joint_position) - LEG_KD * joint_velocity
    leg_pd[list(WHEEL_INDICES)] = 0.0

    wheel_omega = joint_velocity[list(WHEEL_INDICES)]
    wheel_speed_request = nominal_wheel + wheel_offset
    wheel_speed_target = np.clip(
        wheel_speed_request, -WHEEL_SPEED_LIMIT_RAD_S, WHEEL_SPEED_LIMIT_RAD_S
    )
    wheel_speed_target_clip_mask = wheel_speed_target != wheel_speed_request
    wheel_error = wheel_speed_target - wheel_omega
    if variant in COMMON_INTEGRAL_VARIANTS and common_p_active:
        integral_error = (
            wheel_error - np.mean(wheel_error)
            + (servo_forward - body_forward) / WHEEL_RADIUS_M
        )
        integral_mode = "common_body_feedback"
    else:
        integral_error = wheel_error.copy()
        integral_mode = "differential_wheel_feedback"
    integral_candidate = np.clip(
        integral_before + WHEEL_KI * integral_error * CONTROL_DT_S,
        -WHEEL_INTEGRAL_LIMIT_NM,
        WHEEL_INTEGRAL_LIMIT_NM,
    )
    candidate_request = WHEEL_KP * wheel_error + integral_candidate
    within_torque = np.abs(candidate_request) <= JOINT_TORQUE_LIMIT_NM[list(WHEEL_INDICES)]
    candidate_error_product = candidate_request * integral_error
    if not np.isfinite(candidate_error_product).all():
        raise ValueError("wheel PI antiwindup arithmetic produced nonfinite values")
    unwinding = candidate_error_product < 0
    commit_mask = within_torque | unwinding
    integral_after = np.where(commit_mask, integral_candidate, integral_before)
    base_wheel = np.zeros(16)
    base_wheel[list(WHEEL_INDICES)] = WHEEL_KP * wheel_error + integral_after

    base_request = leg_pd + support + base_wheel

    forward_axis = rotation[:, 0]
    projected_jx = np.zeros((4, 3))
    relative_forward = np.zeros(4)
    leg_damping_delta = np.zeros(16)
    for leg in range(4):
        jx = forward_axis @ jacobian[leg, :, :3]
        projected_jx[leg] = jx
        relative_forward[leg] = float(jx @ joint_velocity[4 * leg : 4 * leg + 3])
        if damping_active:
            leg_damping_delta[4 * leg : 4 * leg + 3] = (
                -DRIVE_DAMPING_GAIN_NSPM * jx * relative_forward[leg]
            )
    leg_damping_power = float(np.sum(leg_damping_delta * joint_velocity))

    mean_omega = float(np.mean(wheel_omega))
    body_equivalent_omega = body_forward / WHEEL_RADIUS_M
    common_error = mean_omega - body_equivalent_omega
    common_scalar = WHEEL_KP * common_error if common_p_active else 0.0
    common_delta = np.zeros(16)
    common_delta[list(WHEEL_INDICES)] = common_scalar

    final_request = base_request + leg_damping_delta + common_delta
    safe_torque = np.clip(final_request, -JOINT_TORQUE_LIMIT_NM, JOINT_TORQUE_LIMIT_NM)
    torque_clip_mask = safe_torque != final_request
    upper_outward = (joint_position >= JOINT_POSITION_HIGH_RAD) & (safe_torque > 0)
    lower_outward = (joint_position <= JOINT_POSITION_LOW_RAD) & (safe_torque < 0)
    speed_outward = (np.abs(joint_velocity) >= JOINT_VELOCITY_LIMIT_RAD_S) & (
        safe_torque * joint_velocity > 0
    )
    safe_torque[upper_outward | lower_outward | speed_outward] = 0.0
    protection_changed_mask = safe_torque != final_request

    for label, value in (
        ("leg action offset", leg_offset),
        ("wheel action offset", wheel_offset),
        ("geometric joint target", geometric_target),
        ("rate-limited joint target", rate_limited_target),
        ("position-limited joint target", position_limited_target),
        ("joint target", joint_target),
        ("wheel speed request", wheel_speed_request),
        ("wheel speed target", wheel_speed_target),
        ("wheel speed error", wheel_error),
        ("wheel integral error", integral_error),
        ("wheel integral candidate", integral_candidate),
        ("wheel candidate request", candidate_request),
        ("wheel integral after", integral_after),
        ("leg PD", leg_pd),
        ("wheel PI", base_wheel),
        ("base request", base_request),
        ("projected foot Jacobian", projected_jx),
        ("relative forward velocity", relative_forward),
        ("leg damping", leg_damping_delta),
        ("leg damping raw power", leg_damping_power),
        ("mean wheel angular speed", mean_omega),
        ("body equivalent wheel speed", body_equivalent_omega),
        ("common wheel-P error", common_error),
        ("common wheel-P scalar", common_scalar),
        ("common wheel-P", common_delta),
        ("final request", final_request),
        ("protected torque", safe_torque),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"{label} arithmetic produced nonfinite values")

    return {
        "schema": CONTROL_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "controller_variant": variant,
        "wheel_integral_mode": integral_mode,
        "wheel_kp_nm_per_rad_s": WHEEL_KP,
        "wheel_ki_nm_per_rad": WHEEL_KI,
        "wheel_integral_limit_nm": WHEEL_INTEGRAL_LIMIT_NM,
        "wheel_radius_m": WHEEL_RADIUS_M,
        "control_dt_s": CONTROL_DT_S,
        "action_enabled": action_enabled,
        "leg_longitudinal_damping_active": damping_active,
        "body_common_p_active": common_p_active,
        "wheel_integral_antiwindup_scope": ANTIWINDUP_SCOPE,
        "consumed_joint_position_rad": _frozen(joint_position),
        "consumed_joint_velocity_rad_s": _frozen(joint_velocity),
        "consumed_nominal_joint_target_rad": _frozen(nominal_target),
        "consumed_nominal_wheel_speed_rad_s": _frozen(nominal_wheel),
        "consumed_support_torque_nm": _frozen(support),
        "consumed_base_rotation_world_from_body": _frozen(rotation),
        "consumed_foot_jacobian_world": _frozen(jacobian),
        "consumed_body_com_forward_mps": body_forward,
        "consumed_servo_forward_mps": servo_forward,
        "consumed_wheel_omega_rad_s": _frozen(wheel_omega),
        "raw_action": _frozen(raw_action),
        "clipped_action": _frozen(clipped_action),
        "action_clip_mask": _frozen(action_clip_mask),
        "applied_action": _frozen(applied_action),
        "leg_action_scale_rad": _frozen(LEG_ACTION_SCALE_RAD),
        "wheel_action_scale_rad_s": WHEEL_ACTION_SCALE_RAD_S,
        "leg_action_offset_rad": _frozen(leg_offset),
        "wheel_action_offset_rad_s": _frozen(wheel_offset),
        "geometric_joint_target_rad": _frozen(geometric_target),
        "rate_limited_joint_target_rad": _frozen(rate_limited_target),
        "rate_limit_mask": _frozen(rate_limit_mask),
        "position_limited_joint_target_rad": _frozen(position_limited_target),
        "position_clip_mask": _frozen(position_clip_mask),
        "joint_target_rad": _frozen(joint_target),
        "wheel_target_position_rad": _frozen(joint_position[list(WHEEL_INDICES)]),
        "wheel_speed_request_rad_s": _frozen(wheel_speed_request),
        "wheel_speed_target_rad_s": _frozen(wheel_speed_target),
        "wheel_speed_target_clip_mask": _frozen(wheel_speed_target_clip_mask),
        "wheel_speed_error_rad_s": _frozen(wheel_error),
        "wheel_integral_error_rad_s": _frozen(integral_error),
        "wheel_integral_before_nm": _frozen(integral_before),
        "wheel_integral_candidate_nm": _frozen(integral_candidate),
        "wheel_candidate_request_nm": _frozen(candidate_request),
        "wheel_integral_commit_mask": _frozen(commit_mask),
        "wheel_integral_after_nm": _frozen(integral_after),
        "leg_pd_nm": _frozen(leg_pd),
        "base_wheel_torque_nm": _frozen(base_wheel),
        "base_request_torque_nm": _frozen(base_request),
        "projected_jx": _frozen(projected_jx),
        "relative_forward_mps": _frozen(relative_forward),
        "leg_damping_delta_torque_nm": _frozen(leg_damping_delta),
        "leg_damping_raw_joint_power_w": leg_damping_power,
        "common_wheel_mean_omega_rad_s": mean_omega,
        "common_wheel_body_equivalent_omega_rad_s": body_equivalent_omega,
        "common_wheel_error_rad_s": common_error,
        "common_wheel_scalar_delta_nm": common_scalar,
        "common_wheel_delta_torque_nm": _frozen(common_delta),
        "final_request_torque_nm": _frozen(final_request),
        "safe_torque_nm": _frozen(safe_torque),
        "torque_clip_mask": _frozen(torque_clip_mask),
        "upper_outward_mask": _frozen(upper_outward),
        "lower_outward_mask": _frozen(lower_outward),
        "speed_outward_mask": _frozen(speed_outward),
        "protection_changed_mask": _frozen(protection_changed_mask),
    }


def nominal_support(inputs: dict) -> dict[str, Any]:
    """Return the source-equivalent nominal gravity/attitude support arithmetic.

    Original heading rotation, 180/24 attitude pair, 9.81 gravity, the original
    least-squares [1; y; -x] allocation with the [0, .65*mg] force clip and the
    original -Jz * force leg support. No cached state, no engine, no alternative
    controller: the caller owns the geometry and the commanded attitude.
    """
    _checked_keys(inputs, SUPPORT_INPUT_KEYS, "inputs")
    base_rpy = _finite_array(inputs["base_rpy_rad"], (3,), "base_rpy_rad")
    angular_world = _finite_array(
        inputs["base_angular_velocity_world"], (3,), "base_angular_velocity_world"
    )
    rotation = _rotation(
        inputs["base_rotation_world_from_body"], "base_rotation_world_from_body"
    )
    foot_offset = _finite_array(inputs["foot_offset_world"], (4, 3), "foot_offset_world")
    jacobian = _finite_array(inputs["foot_jacobian_world"], (4, 3, 4), "foot_jacobian_world")
    com_offset = _finite_array(
        inputs["nominal_base_com_offset_body_m"], (3,), "nominal_base_com_offset_body_m"
    )
    mass = _finite_scalar(inputs["nominal_mass_kg"], "nominal_mass_kg")
    if mass <= 0.0:
        raise ValueError("nominal_mass_kg must be positive")
    commanded_roll = _finite_scalar(inputs["commanded_roll_rad"], "commanded_roll_rad")
    commanded_pitch = _finite_scalar(inputs["commanded_pitch_rad"], "commanded_pitch_rad")

    roll, pitch, yaw = base_rpy
    heading = np.asarray(
        (
            (np.cos(yaw), -np.sin(yaw), 0.0),
            (np.sin(yaw), np.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    angular_yaw_frame = heading.T @ angular_world
    feedback = (
        ATTITUDE_KP * (np.asarray((commanded_roll, commanded_pitch)) - (roll, pitch))
        - ATTITUDE_KD * angular_yaw_frame[:2]
    )
    upward = mass * GRAVITY_MPS2
    for label, value in (
        ("heading rotation", heading),
        ("yaw-frame angular velocity", angular_yaw_frame),
        ("attitude feedback", feedback),
        ("upward force", upward),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"nominal support {label} produced nonfinite values")
    moment = np.cross(
        rotation @ com_offset, np.asarray((0.0, 0.0, upward))
    ) + heading @ np.r_[feedback, 0.0]
    allocation = np.vstack((np.ones(4), foot_offset[:, 1], -foot_offset[:, 0]))
    if not np.isfinite(moment).all() or not np.isfinite(allocation).all():
        raise ValueError("nominal support moment/allocation produced nonfinite values")
    least_squares = np.linalg.lstsq(allocation, np.r_[upward, moment[:2]], rcond=None)[0]
    if not np.isfinite(least_squares).all():
        raise ValueError("nominal support least-squares force produced nonfinite values")
    forces = np.clip(least_squares, 0.0, SUPPORT_FORCE_FRACTION * upward)
    support = np.zeros(16)
    for leg in range(4):
        support[4 * leg : 4 * leg + 3] = -jacobian[leg, 2, :3] * forces[leg]
    if not (
        np.isfinite(moment).all() and np.isfinite(forces).all() and np.isfinite(support).all()
    ):
        raise ValueError("nominal support arithmetic produced nonfinite values")

    return {
        "schema": SUPPORT_SCHEMA,
        "consumed_base_rpy_rad": _frozen(base_rpy),
        "consumed_base_angular_velocity_world": _frozen(angular_world),
        "consumed_base_rotation_world_from_body": _frozen(rotation),
        "consumed_foot_offset_world": _frozen(foot_offset),
        "consumed_foot_jacobian_world": _frozen(jacobian),
        "consumed_nominal_base_com_offset_body_m": _frozen(com_offset),
        "consumed_nominal_mass_kg": mass,
        "consumed_commanded_roll_rad": commanded_roll,
        "consumed_commanded_pitch_rad": commanded_pitch,
        "heading_rotation_world_from_yaw": _frozen(heading),
        "base_angular_velocity_yaw_frame_rad_s": _frozen(angular_yaw_frame),
        "attitude_feedback_nm": _frozen(feedback),
        "upward_force_n": float(upward),
        "support_moment_world_nm": _frozen(moment),
        "allocation_matrix": _frozen(allocation),
        "least_squares_force_n": _frozen(least_squares),
        "force_clip_mask": _frozen(forces != least_squares),
        "support_force_n": _frozen(forces),
        "support_torque_nm": _frozen(support),
    }
