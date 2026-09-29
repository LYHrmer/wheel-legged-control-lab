"""Pure 99-value observation for the new 16-action full-drive policy.

The encoder consumes a single prepared decision, provider state and a static
compiled-course scan. It never reads a MuJoCo model, truth reward, controller
private state, or any already-applied future action. The first training pilot
has lateral command and jump phase fixed to zero; later capabilities need a
new reviewed command/phase contract rather than an implicit reinterpretation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from course_ground_08 import CourseGroundMap
from residual16_math_08 import ACTION_SCHEMA

OBSERVATION_SCHEMA = "d1-full-drive-proprio-course99-action16-v1"
OBSERVATION_SIZE = 99
LEG_INDICES = np.asarray([index for index in range(16) if index % 4 != 3])
WHEEL_INDICES = np.asarray((3, 7, 11, 15))
LEG_POSITION_SCALE_RAD = np.tile((0.8, 2.0, 1.0), 4)
NOMINAL_JOINT_POSITION_RAD = np.tile((0.0, 0.8, -1.5, 0.0), 4)
JOINT_VELOCITY_LIMIT_RAD_S = np.tile((20.0, 20.0, 20.0, 30.0), 4)
SCAN_FORWARD_OFFSETS_M = (-0.3, 0.0, 0.3, 0.6, 0.9)
SCAN_LATERAL_OFFSETS_M = (-0.3, 0.0, 0.3)
OBSERVATION_SLICES = {
    "com_velocity_body_over2": (0, 3),
    "angular_velocity_body_over4": (3, 6),
    "projected_gravity_body": (6, 9),
    "clearance_error_over_0p08m": (9, 10),
    "leg_position_error": (10, 22),
    "joint_velocity_over_limit": (22, 38),
    "applied_vx_vy_yaw_clearance_jump_phase": (38, 43),
    "nominal_leg_joint_targets": (43, 55),
    "nominal_wheel_speed_targets_over30": (55, 59),
    "actual_wheel_integral_over4": (59, 63),
    "previous_effective_action16": (63, 79),
    "measured_wheel_contacts": (79, 83),
    "compiled_map_scan_relative_over0p20m": (83, 98),
    "measurement_age_over0p05s": (98, 99),
}


def _scalar(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise TypeError(f"{label} must be a real array of shape {shape}")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class CourseScan:
    """15 x-major samples, relative to ground at the current body origin."""

    center_x_m: float
    center_y_m: float
    yaw_rad: float
    center_height_m: float
    center_geom_id: int
    sampled_height_m: np.ndarray
    relative_height_m: np.ndarray
    sampled_geom_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("center_x_m", "center_y_m", "yaw_rad"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name))
        center = _scalar(self.center_height_m, "scan center height")
        if type(self.center_geom_id) is not int or self.center_geom_id < 0:
            raise ValueError("scan center geom id must be non-negative int")
        sampled = _array(self.sampled_height_m, (15,), "sampled heights")
        relative = _array(self.relative_height_m, (15,), "relative heights")
        if not np.array_equal(relative, sampled - center):
            raise ValueError("relative scan heights differ from sampled minus center")
        ids = tuple(self.sampled_geom_ids)
        if len(ids) != 15 or any(type(value) is not int or value < 0 for value in ids):
            raise ValueError("scan must name fifteen compiled geom ids")
        for name, value in (("sampled_height_m", sampled),
                            ("relative_height_m", relative)):
            object.__setattr__(self, name,
                               np.frombuffer(np.ascontiguousarray(value).tobytes(),
                                             dtype=np.float64))
        object.__setattr__(self, "sampled_geom_ids", ids)


def scan_compiled_course_ground(ground_map: CourseGroundMap, *, x_m: Real,
                                y_m: Real, yaw_rad: Real) -> CourseScan:
    if not isinstance(ground_map, CourseGroundMap):
        raise TypeError("scan source must be the compiled CourseGroundMap")
    x, y, yaw = (_scalar(value, label) for value, label in
                 ((x_m, "scan x"), (y_m, "scan y"), (yaw_rad, "scan yaw")))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    points = [(x, y)]
    for forward in SCAN_FORWARD_OFFSETS_M:
        for lateral in SCAN_LATERAL_OFFSETS_M:
            points.append((
                x + cosine * forward - sine * lateral,
                y + sine * forward + cosine * lateral,
            ))
    center, *samples = ground_map.sample_many(np.asarray(points, dtype=np.float64))
    sampled = np.asarray([hit.height_m for hit in samples], dtype=np.float64)
    geom_ids = tuple(hit.geom_id for hit in samples)
    relative = sampled - center.height_m
    if sampled.shape != (15,) or not np.isfinite(relative).all():
        raise ValueError("compiled course scan is not finite shape 15")
    return CourseScan(x, y, yaw, center.height_m, center.geom_id,
                      sampled, relative, geom_ids)


@dataclass(frozen=True, slots=True)
class ObservationPacket:
    values: np.ndarray
    clipped_mask: np.ndarray
    clipped_count: int
    preclip_min: float
    preclip_max: float
    scan_center_height_m: float
    scan_center_geom_id: int
    scan_geom_ids: tuple[int, ...]


def encode_full_drive_observation(decision: Any, scan: CourseScan,
                                  *, jump_phase: int = 0) -> ObservationPacket:
    """Encode one matching current-tick decision; no hidden state mutation."""
    if not isinstance(scan, CourseScan):
        raise TypeError("observation requires one compiled course scan")
    if type(jump_phase) is not int or jump_phase != 0:
        raise ValueError("the first pilot permits only idle jump phase 0")
    command = decision.motion_command
    if _scalar(command.lateral_velocity_mps, "applied vy") != 0.0 or command.jump_requested:
        raise ValueError("pilot observation requires zero lateral and jump commands")
    context = decision.context
    if context.action_size != 16 or context.action_schema != ACTION_SCHEMA:
        raise ValueError("current decision has a different 16-action schema")
    state, proposal = context.state, context.proposal
    if state.sequence != proposal.tick or abs(state.control_time_s - proposal.control_time_s) > 1e-10:
        raise ValueError("proposal and provider publication do not match")
    base_xy = _array(state.base_position, (3,), "base position")[:2]
    if (abs(scan.center_x_m - base_xy[0]) > 1e-12
            or abs(scan.center_y_m - base_xy[1]) > 1e-12
            or abs(scan.yaw_rad - _scalar(state.base_rpy[2], "base yaw")) > 1e-12):
        raise ValueError("compiled scan pose differs from current provider pose")
    baseline = proposal.baseline
    nominal_joint = _array(baseline.nominal_joint_target_rad, (16,), "nominal joint targets")
    nominal_wheel = _array(baseline.nominal_wheel_speed_rad_s, (4,), "nominal wheel targets")
    wheel_integral = _array(proposal.memory.wheel_integral_nm, (4,), "actual wheel integral")
    position = _array(state.joint_position, (16,), "joint position")
    velocity = _array(state.joint_velocity, (16,), "joint velocity")
    contacts = np.asarray(state.wheel_contact)
    if contacts.shape != (4,) or contacts.dtype.kind != "b":
        raise TypeError("wheel contacts must be four measured booleans")
    previous = _array(context.previous_normalized_action, (16,), "previous action")
    if np.any(np.abs(previous) > 1.0):
        raise ValueError("previous applied action must be clipped")
    scan_relative = _array(scan.relative_height_m, (15,), "course scan")
    parts = (
        _array(state.base_linear_velocity_body, (3,), "COM body velocity") / 2.0,
        _array(state.base_angular_velocity_body, (3,), "body angular velocity") / 4.0,
        _array(state.projected_gravity_body, (3,), "projected gravity"),
        np.asarray(((_scalar(state.base_position[2], "base z")
                     - _scalar(decision.world_command.base_height_m, "world target z")) / 0.08,)),
        (position[LEG_INDICES] - NOMINAL_JOINT_POSITION_RAD[LEG_INDICES]) / LEG_POSITION_SCALE_RAD,
        velocity / JOINT_VELOCITY_LIMIT_RAD_S,
        np.asarray((
            _scalar(command.forward_velocity_mps, "applied vx") / 1.6,
            _scalar(command.lateral_velocity_mps, "applied vy") / 0.12,
            _scalar(command.yaw_rate_rps, "applied yaw") / 0.6,
            (_scalar(command.clearance_m, "applied clearance") - 0.455) / 0.08,
            float(jump_phase),
        )),
        (nominal_joint[LEG_INDICES] - NOMINAL_JOINT_POSITION_RAD[LEG_INDICES])
        / LEG_POSITION_SCALE_RAD,
        nominal_wheel / 30.0,
        wheel_integral / 4.0,
        previous,
        contacts.astype(np.float64),
        scan_relative / 0.20,
        np.asarray((_scalar(state.age_s, "measurement age") / 0.05,)),
    )
    raw = np.concatenate(parts)
    if raw.shape != (OBSERVATION_SIZE,) or not np.isfinite(raw).all():
        raise ValueError("new full-drive observation is not finite shape 99")
    clipped = np.clip(raw, -5.0, 5.0)
    mask = clipped != raw
    return ObservationPacket(
        values=np.frombuffer(clipped.astype(np.float32).tobytes(), dtype=np.float32),
        clipped_mask=np.frombuffer(mask.tobytes(), dtype=np.bool_),
        clipped_count=int(np.count_nonzero(mask)),
        preclip_min=float(np.min(raw)), preclip_max=float(np.max(raw)),
        scan_center_height_m=scan.center_height_m,
        scan_center_geom_id=scan.center_geom_id,
        scan_geom_ids=scan.sampled_geom_ids,
    )
