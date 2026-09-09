"""Eight physical wheel/leg targets with one standalone low-level torque law.

There is no LQR/VMC object hidden inside this controller. Leg position PD is
combined once with nominal gravity/attitude support; wheel joints receive only
wheel-speed feedback. Runtime feedback is exclusively a D1StateEstimate.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
from numbers import Real

import mujoco
import numpy as np

from .control_context import D1ControllerMemory
from .controllers import D1Command
from .model import (
    JOINT_POSITION_HIGH,
    JOINT_POSITION_LOW,
    JOINT_TORQUE_LIMIT,
    JOINT_VELOCITY_LIMIT,
    NOMINAL_JOINT_POSITION,
    D1Plant,
)
from .state_estimation import D1StateEstimate

WHEEL_INDICES = np.asarray((3, 7, 11, 15))
LEG_INDICES = np.asarray([i for i in range(16) if i % 4 != 3])
NOMINAL_LEG_KP = np.tile((80.0, 80.0, 80.0, 0.0), 4)
NOMINAL_LEG_KD = np.tile((3.0, 3.0, 3.0, 0.0), 4)
for _array in (NOMINAL_LEG_KP, NOMINAL_LEG_KD):
    _array.setflags(write=False)
NOMINAL_ROLL_PITCH_KP = 180.0
NOMINAL_ROLL_PITCH_KD = 24.0
# Strictly positive: a zero here would silently delete a feedback law instead of
# retuning it, and the wheel loop has no other proportional term.
_POSITIVE_FIELDS = ("wheel_kp", "leg_feedback_scale", "attitude_feedback_scale")


class D1ControlEnvelopeError(ValueError):
    """A finite nominal reference lies outside the controller's supported domain.

    This does not represent malformed/nonfinite input or a numerical failure.
    A task may terminate after a completed step if its next preview raises it;
    rejecting an initial configuration or an unexecuted action still raises.
    """


@dataclass(frozen=True)
class D1WheelLegControlConfig:
    """Fixed episode gains; synthetic tuning, not identified hardware gains."""

    wheel_kp: float = 2.2
    wheel_ki: float = 3.0
    yaw_feedback_gain: float = 4.0
    leg_feedback_scale: float = 1.0
    attitude_feedback_scale: float = 1.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{field.name} must be a real scalar")
            positive = field.name in _POSITIVE_FIELDS
            if not np.isfinite(value) or value < 0 or (positive and value == 0):
                raise ValueError(
                    f"{field.name} must be finite and nonnegative; "
                    "wheel_kp and the feedback scales must be positive"
                )
            object.__setattr__(self, field.name, float(value))

    @property
    def feedback_scaled(self) -> bool:
        """True when either bandwidth scale leaves its historical value of one."""
        return self.leg_feedback_scale != 1.0 or self.attitude_feedback_scale != 1.0


def _freeze_arrays(instance) -> None:
    for field in fields(instance):
        value = getattr(instance, field.name)
        if isinstance(value, np.ndarray):
            value = np.ascontiguousarray(value)
            object.__setattr__(
                instance,
                field.name,
                np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape),
            )


@dataclass(frozen=True)
class D1WheelLegTargets:
    nominal_joint_target_rad: np.ndarray
    nominal_wheel_speed_rad_s: np.ndarray
    leg_extension_target_m: np.ndarray
    effective_yaw_request_rps: float

    def __post_init__(self):
        _freeze_arrays(self)


@dataclass(frozen=True)
class D1WheelLegResult:
    nominal_joint_target_rad: np.ndarray
    nominal_wheel_speed_rad_s: np.ndarray
    joint_target_rad: np.ndarray
    wheel_speed_target_rad_s: np.ndarray
    leg_extension_target_m: np.ndarray
    requested_extension_m: np.ndarray
    clipped_action: np.ndarray
    leg_pd_nm: np.ndarray
    support_nm: np.ndarray
    wheel_nm: np.ndarray
    requested_torque_nm: np.ndarray
    torque_nm: np.ndarray
    support_force_n: np.ndarray
    joint_target_rate_limited: np.ndarray
    torque_limited: np.ndarray
    memory_before: D1ControllerMemory
    memory_after: D1ControllerMemory
    effective_yaw_request_rps: float

    def __post_init__(self):
        _freeze_arrays(self)


def build_leg_extension_table(plant):
    """Build body-origin wheel IK, warm-starting outward from the nominal pose.

    The DLS/precompute implementation was drafted by Claude Opus and checked
    locally; public validation is recorded in results/d1_v3_action_probes.
    Only caller-owned nominal scratch data is modified.
    """
    model, data = plant.model, plant.data
    qadr = np.asarray(plant.qpos_addresses).reshape(4, 4)
    vadr = np.asarray(plant.dof_addresses).reshape(4, 4)
    base = plant.base_body_id
    bodies = plant.wheel_body_ids_by_leg
    low, high = JOINT_POSITION_LOW[:3], JOINT_POSITION_HIGH[:3]
    grid = np.linspace(-0.08, 0.08, 161)
    table = np.zeros((4, len(grid), 3))
    q_backup, v_backup = data.qpos.copy(), data.qvel.copy()
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def fk(leg, q):
        data.qpos[qadr[leg, :3]] = q
        data.qpos[qadr[leg, 3]] = 0.0
        mujoco.mj_forward(model, data)
        rotation = data.xmat[base].reshape(3, 3).copy()
        return rotation.T @ (data.xpos[bodies[leg]] - data.xpos[base]), rotation

    def solve(leg, extension, seed):
        target = offsets[leg].copy()
        target[2] -= extension
        q = np.clip(seed.copy(), low, high)
        for _ in range(60):
            current, rotation = fk(leg, q)
            error = target - current
            if np.linalg.norm(error) <= 1e-8:
                break
            mujoco.mj_jacBody(model, data, jacp, jacr, bodies[leg])
            jacobian = rotation.T @ jacp[:, vadr[leg, :3]]
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-6 * np.eye(3), error)
            candidate = np.clip(q + np.clip(delta, -0.15, 0.15), low, high)
            if np.max(np.abs(candidate - q)) < 1e-14:
                break
            q = candidate
        current, _ = fk(leg, q)
        residual = float(np.linalg.norm(target - current))
        if not np.isfinite(residual) or residual > 1e-5:
            raise RuntimeError(f"leg {leg} extension {extension} m IK residual {residual} m")
        return q, residual

    try:
        data.qpos[plant.qpos_addresses] = NOMINAL_JOINT_POSITION
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        rotation = data.xmat[base].reshape(3, 3)
        offsets = (data.xpos[np.asarray(bodies)] - data.xpos[base]) @ rotation
        max_error = 0.0
        zero = int(np.argmin(np.abs(grid)))
        for leg in range(4):
            q_zero, error = solve(leg, grid[zero], NOMINAL_JOINT_POSITION[:3])
            table[leg, zero] = q_zero
            max_error = max(max_error, error)
            for indices in (range(zero + 1, len(grid)), range(zero - 1, -1, -1)):
                seed = q_zero.copy()
                for index in indices:
                    seed, error = solve(leg, grid[index], seed)
                    table[leg, index] = seed
                    max_error = max(max_error, error)
        return grid, table, offsets.copy(), max_error
    finally:
        data.qpos[:] = q_backup
        data.qvel[:] = v_backup
        mujoco.mj_forward(model, data)


@lru_cache(maxsize=1)
def _nominal_geometry():
    plant = D1Plant()
    grid, table, offsets, error = build_leg_extension_table(plant)
    mass = plant.nominal_total_mass_kg
    rotation = plant.data.xmat[plant.base_body_id].reshape(3, 3)
    com = (
        np.sum(plant.model.body_mass[:, None] * plant.data.xipos, axis=0) / mass
        - plant.data.xpos[plant.base_body_id]
    ) @ rotation
    for array in (grid, table, offsets, com):
        array.setflags(write=False)
    return grid, table, offsets, mass, com, error


class D1WheelLegController:
    """Map [four extension, four wheel-speed] residuals to 16 ordered torques.

    Positive leg action extends the leg (wheel moves down in the body frame).
    Positive wheel action increases forward rolling speed. Commands keep the
    existing world-z height contract; ground_height_m is supplied by the caller.
    Four wheel integral torques are explicitly exposed through control_memory.
    Call compute exactly once per control tick; nominal_targets is pure and safe
    before policy inference. reset clears all integrals and diagnostics.

    Joint reference increments are bounded relative to measured q by the rated
    speed times dt. Outward torque is removed at position/speed boundaries. This
    does not guarantee actual speed under external impacts; probes report it.

    leg_feedback_scale and attitude_feedback_scale multiply the nominal leg PD
    pair (80/3) and the nominal body attitude pair (180/24). They only retune
    feedback bandwidth: gravity support, the wheel PI law, the yaw feedback and
    the IK targets are unchanged, and any scale other than one selects its own
    control schema so recorded policies cannot load against a different law.
    """

    action_schema = "d1-wheel-leg-extension-speed-v1"
    control_schema = "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-v2"
    leg_action_scale_m = 0.04
    wheel_action_scale_rad_s = 4.0

    def __init__(
        self,
        *,
        control_dt: float = 0.01,
        yaw_feedback_gain: float = 4.0,
        wheel_kp: float = 2.2,
        wheel_ki: float = 3.0,
        leg_feedback_scale: float = 1.0,
        attitude_feedback_scale: float = 1.0,
    ):
        if not np.isfinite(control_dt) or not 0 < control_dt <= 0.02:
            raise ValueError("control_dt must be finite and in (0, .02] s")
        gains = D1WheelLegControlConfig(
            wheel_kp, wheel_ki, yaw_feedback_gain, leg_feedback_scale, attitude_feedback_scale
        )
        if gains.feedback_scaled:
            # Scaled position/attitude bandwidth is a different control law, so
            # old checkpoints must not silently load against it.
            self.control_schema = "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-configured-v4"
        elif gains != D1WheelLegControlConfig():
            self.control_schema = "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-configured-v3"
        self.control_dt = float(control_dt)
        self._grid, self._table, self._offsets, self._mass, self._com, self.ik_max_error_m = (
            _nominal_geometry()
        )
        # One scale moves the whole leg PD pair, and one moves the whole body
        # attitude pair; gravity support, wheel PI and the IK stay untouched.
        self.leg_feedback_scale = gains.leg_feedback_scale
        self.attitude_feedback_scale = gains.attitude_feedback_scale
        self.leg_kp = self.leg_feedback_scale * NOMINAL_LEG_KP
        self.leg_kd = self.leg_feedback_scale * NOMINAL_LEG_KD
        self.wheel_kp = gains.wheel_kp
        self.wheel_ki = gains.wheel_ki
        self.yaw_feedback_gain = gains.yaw_feedback_gain
        self.yaw_request_limit_rps = 0.6
        self._wheel_integral_nm = np.zeros(4)
        self.roll_pitch_kp = self.attitude_feedback_scale * NOMINAL_ROLL_PITCH_KP
        self.roll_pitch_kd = self.attitude_feedback_scale * NOMINAL_ROLL_PITCH_KD
        self.last_result: D1WheelLegResult | None = None

    def reset(self):
        self.last_result = None
        self._wheel_integral_nm[:] = 0.0

    @property
    def control_memory(self) -> D1ControllerMemory:
        return D1ControllerMemory(wheel_integral_nm=self._wheel_integral_nm)

    def _extension(self, command: D1Command, ground_height_m: float) -> np.ndarray:
        values = np.asarray(
            (
                command.forward_velocity_mps,
                command.yaw_rate_rps,
                command.base_height_m,
                command.roll_rad,
                command.pitch_rad,
                command.base_vertical_velocity_mps,
                ground_height_m,
            )
        )
        if not np.isfinite(values).all():
            raise ValueError("command and ground height must be finite")
        if command.base_vertical_velocity_mps != 0:
            raise ValueError(
                "nonzero vertical velocity is not implemented by the v3 position controller"
            )
        clearance = command.base_height_m - ground_height_m
        if not np.isfinite(clearance):
            raise ValueError("nominal clearance must be finite")
        if (
            not 0.32 <= clearance <= 0.56
            or max(abs(command.roll_rad), abs(command.pitch_rad)) > 0.3
        ):
            raise D1ControlEnvelopeError(
                "nominal clearance must be .32-.56 m and attitude commands within .3 rad"
            )
        # Small commanded attitude tilts the desired support geometry. Measured
        # attitude feedback is handled by the support moments, not a second PD.
        return (
            (clearance - 0.455)
            + self._offsets[:, 1] * np.tan(command.roll_rad)
            - self._offsets[:, 0] * np.tan(command.pitch_rad)
        )

    def _joint_targets(self, extension: np.ndarray) -> np.ndarray:
        target = NOMINAL_JOINT_POSITION.copy()
        for leg in range(4):
            for joint in range(3):
                target[4 * leg + joint] = np.interp(
                    extension[leg], self._grid, self._table[leg, :, joint]
                )
        return target

    def nominal_targets(
        self, command: D1Command, state: D1StateEstimate, *, ground_height_m: float = 0.0
    ) -> D1WheelLegTargets:
        extension = np.clip(
            self._extension(command, ground_height_m), self._grid[0], self._grid[-1]
        )
        yaw = float(state.base_rpy[2])
        lateral_world = np.asarray((-np.sin(yaw), np.cos(yaw), 0.0))
        lateral = state.foot_offset_world @ lateral_world
        effective_yaw = float(
            np.clip(
                command.yaw_rate_rps
                + self.yaw_feedback_gain
                * (command.yaw_rate_rps - state.base_angular_velocity_body[2]),
                -self.yaw_request_limit_rps,
                self.yaw_request_limit_rps,
            )
        )
        wheels = np.clip(
            (command.forward_velocity_mps - effective_yaw * lateral) / 0.087, -30.0, 30.0
        )
        return D1WheelLegTargets(self._joint_targets(extension), wheels, extension, effective_yaw)

    def compute(
        self,
        command: D1Command,
        state: D1StateEstimate,
        action: np.ndarray,
        *,
        ground_height_m: float = 0.0,
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("action must contain eight finite values")
        clipped = np.clip(action, -1.0, 1.0)
        memory_before = self.control_memory
        nominal = self.nominal_targets(command, state, ground_height_m=ground_height_m)
        requested_extension = (
            self._extension(command, ground_height_m) + self.leg_action_scale_m * clipped[:4]
        )
        extension = np.clip(requested_extension, self._grid[0], self._grid[-1])
        geometric_target = self._joint_targets(extension)
        target = np.clip(
            geometric_target,
            state.joint_position - JOINT_VELOCITY_LIMIT * self.control_dt,
            state.joint_position + JOINT_VELOCITY_LIMIT * self.control_dt,
        )
        target = np.clip(target, JOINT_POSITION_LOW, JOINT_POSITION_HIGH)
        target[WHEEL_INDICES] = state.joint_position[WHEEL_INDICES]
        wheel_target = np.clip(
            nominal.nominal_wheel_speed_rad_s + self.wheel_action_scale_rad_s * clipped[4:],
            -30.0,
            30.0,
        )
        leg_pd = self.leg_kp * (target - state.joint_position) - self.leg_kd * state.joint_velocity
        leg_pd[WHEEL_INDICES] = 0.0

        roll, pitch, yaw = state.base_rpy
        heading = np.asarray(
            ((np.cos(yaw), -np.sin(yaw), 0.0), (np.sin(yaw), np.cos(yaw), 0.0), (0.0, 0.0, 1.0))
        )
        angular = heading.T @ state.base_angular_velocity_world
        feedback = (
            self.roll_pitch_kp * (np.asarray((command.roll_rad, command.pitch_rad)) - (roll, pitch))
            - self.roll_pitch_kd * angular[:2]
        )
        upward = self._mass * 9.81
        moment = (
            np.cross(state.base_rotation @ self._com, np.asarray((0.0, 0.0, upward)))
            + heading @ np.r_[feedback, 0.0]
        )
        offsets = state.foot_offset_world
        allocation = np.vstack((np.ones(4), offsets[:, 1], -offsets[:, 0]))
        forces = np.clip(
            np.linalg.lstsq(allocation, np.r_[upward, moment[:2]], rcond=None)[0], 0, 0.65 * upward
        )
        support = np.zeros(16)
        for leg in range(4):
            support[4 * leg : 4 * leg + 3] = -state.foot_jacobian[leg, 2, :3] * forces[leg]
        wheels = np.zeros(16)
        wheel_error = wheel_target - state.joint_velocity[WHEEL_INDICES]
        candidate = np.clip(
            self._wheel_integral_nm + self.wheel_ki * wheel_error * self.control_dt, -4.0, 4.0
        )
        wheel_request = self.wheel_kp * wheel_error + candidate
        within_torque = np.abs(wheel_request) <= JOINT_TORQUE_LIMIT[WHEEL_INDICES]
        unwinding = wheel_request * wheel_error < 0
        self._wheel_integral_nm = np.where(
            within_torque | unwinding, candidate, self._wheel_integral_nm
        )
        wheels[WHEEL_INDICES] = self.wheel_kp * wheel_error + self._wheel_integral_nm
        request = leg_pd + support + wheels
        torque = np.clip(request, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
        upper_outward = (state.joint_position >= JOINT_POSITION_HIGH) & (torque > 0)
        lower_outward = (state.joint_position <= JOINT_POSITION_LOW) & (torque < 0)
        speed_outward = (np.abs(state.joint_velocity) >= JOINT_VELOCITY_LIMIT) & (
            torque * state.joint_velocity > 0
        )
        torque[upper_outward | lower_outward | speed_outward] = 0.0
        self.last_result = D1WheelLegResult(
            nominal.nominal_joint_target_rad,
            nominal.nominal_wheel_speed_rad_s,
            target,
            wheel_target,
            extension,
            requested_extension,
            clipped,
            leg_pd,
            support,
            wheels,
            request,
            torque,
            forces,
            target != geometric_target,
            torque != request,
            memory_before,
            self.control_memory,
            nominal.effective_yaw_request_rps,
        )
        return torque.copy()
