"""Procedural flat and skills-course terrain for the D1 simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from .state_estimation import D1StateEstimate

SUPPORTED_D1_ARENAS = ("flat", "course")


@dataclass(frozen=True)
class D1CourseSpawn:
    """A safe reset pose immediately before one course feature."""

    label: str
    position_m: tuple[float, float, float]
    yaw_rad: float = 0.0


D1_COURSE_SPAWNS: dict[str, D1CourseSpawn] = {
    "start": D1CourseSpawn("course start", (0.0, 0.0, 0.455)),
    "rough": D1CourseSpawn("rough blocks", (0.35, 0.0, 0.455)),
    "ramp": D1CourseSpawn("8 degree ramp", (2.75, 0.0, 0.455)),
    "stairs": D1CourseSpawn("15 mm stairs", (0.35, 2.2, 0.455)),
    "bumps": D1CourseSpawn("wave bumps", (0.35, -2.2, 0.455)),
    "jump": D1CourseSpawn("jump lane", (0.35, 4.0, 0.455)),
}


@dataclass(frozen=True)
class D1TerrainAttitude:
    """Filtered local terrain attitude fitted from wheel contact points."""

    roll_rad: float
    pitch_rad: float
    contact_wheels: int


class D1TerrainAttitudeEstimator:
    """Fit a support plane to wheel contacts without privileged terrain labels."""

    def __init__(self, *, filter_gain: float = 0.12, limit_rad: float = 0.22) -> None:
        if not 0.0 < filter_gain <= 1.0:
            raise ValueError("filter_gain must be in (0, 1]")
        if limit_rad <= 0.0:
            raise ValueError("limit_rad must be positive")
        self.filter_gain = float(filter_gain)
        self.limit_rad = float(limit_rad)
        self._roll = 0.0
        self._pitch = 0.0

    def reset(self) -> None:
        self._roll = 0.0
        self._pitch = 0.0

    def update(self, state: D1StateEstimate) -> D1TerrainAttitude:
        points = state.wheel_contact_point[state.wheel_contact]
        if len(points) >= 3:
            design = np.column_stack((points[:, 0], points[:, 1], np.ones(len(points))))
            slope_x, slope_y, _ = np.linalg.lstsq(design, points[:, 2], rcond=None)[0]
            yaw = float(state.base_rpy[2])
            forward_slope = slope_x * np.cos(yaw) + slope_y * np.sin(yaw)
            lateral_slope = -slope_x * np.sin(yaw) + slope_y * np.cos(yaw)
            target_roll = float(np.arctan(lateral_slope))
            target_pitch = float(-np.arctan(forward_slope))
            target_roll = float(np.clip(target_roll, -self.limit_rad, self.limit_rad))
            target_pitch = float(np.clip(target_pitch, -self.limit_rad, self.limit_rad))
            self._roll += self.filter_gain * (target_roll - self._roll)
            self._pitch += self.filter_gain * (target_pitch - self._pitch)
        return D1TerrainAttitude(self._roll, self._pitch, state.wheel_ground_contacts)


def _pitch_quaternion(angle_rad: float) -> tuple[float, float, float, float]:
    return (float(np.cos(angle_rad / 2.0)), 0.0, float(np.sin(angle_rad / 2.0)), 0.0)


def _add_box(
    spec: mujoco.MjSpec,
    *,
    name: str,
    position: tuple[float, float, float],
    half_size: tuple[float, float, float],
    friction: float,
    rgba: tuple[float, float, float, float],
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=position,
        quat=quaternion,
        size=half_size,
        contype=1,
        conaffinity=1,
        condim=3,
        friction=(friction, 0.005, 0.0001),
        rgba=rgba,
    )


def _add_rough_blocks(spec: mujoco.MjSpec, friction: float) -> None:
    for row, y_position in enumerate(np.linspace(-0.54, 0.54, 7)):
        for column, x_position in enumerate(np.linspace(0.85, 2.45, 9)):
            height = 0.012 + 0.010 * (1.0 + np.sin(1.7 * column + 2.3 * row))
            _add_box(
                spec,
                name=f"terrain_rough_{row}_{column}",
                position=(float(x_position), float(y_position), float(height / 2.0)),
                half_size=(0.095, 0.075, float(height / 2.0)),
                friction=friction,
                rgba=(0.38, 0.43, 0.47, 1.0),
            )


def _add_ramp(spec: mujoco.MjSpec, friction: float) -> None:
    angle = np.deg2rad(8.0)
    half_length = 0.75
    half_thickness = 0.04
    center_height = half_length * np.sin(angle) - half_thickness * np.cos(angle)
    deck_height = 2.0 * half_length * np.sin(angle)
    color = (0.26, 0.50, 0.68, 1.0)
    _add_box(
        spec,
        name="terrain_ramp_up",
        position=(3.85, 0.0, float(center_height)),
        half_size=(half_length, 0.65, half_thickness),
        friction=friction,
        rgba=color,
        quaternion=_pitch_quaternion(-angle),
    )
    _add_box(
        spec,
        name="terrain_ramp_deck",
        position=(5.00, 0.0, float(deck_height / 2.0)),
        half_size=(0.40, 0.65, float(deck_height / 2.0)),
        friction=friction,
        rgba=color,
    )
    _add_box(
        spec,
        name="terrain_ramp_down",
        position=(6.15, 0.0, float(center_height)),
        half_size=(half_length, 0.65, half_thickness),
        friction=friction,
        rgba=color,
        quaternion=_pitch_quaternion(angle),
    )


def _add_stairs(spec: mujoco.MjSpec, friction: float) -> None:
    heights = (0.015, 0.030, 0.045, 0.060, 0.075, 0.060, 0.045, 0.030, 0.015)
    for index, height in enumerate(heights):
        _add_box(
            spec,
            name=f"terrain_stair_{index}",
            position=(1.25 + 0.36 * index, 2.2, height / 2.0),
            half_size=(0.18, 0.62, height / 2.0),
            friction=friction,
            rgba=(0.55, 0.48, 0.70, 1.0),
        )


def _add_wave_bumps(spec: mujoco.MjSpec, friction: float) -> None:
    for index in range(13):
        height = 0.005 + 0.010 * (0.5 + 0.5 * np.sin(index * np.pi / 2.0))
        _add_box(
            spec,
            name=f"terrain_bump_{index}",
            position=(0.95 + 0.28 * index, -2.2, float(height / 2.0)),
            half_size=(0.14, 0.62, float(height / 2.0)),
            friction=friction,
            rgba=(0.36, 0.62, 0.48, 1.0),
        )


def _add_jump_lane(spec: mujoco.MjSpec, friction: float) -> None:
    for index, (x_position, height) in enumerate(((1.8, 0.02), (3.0, 0.04), (4.3, 0.06))):
        _add_box(
            spec,
            name=f"terrain_jump_{index}",
            position=(x_position, 4.0, height / 2.0),
            half_size=(0.08, 0.70, height / 2.0),
            friction=friction,
            rgba=((0.90, 0.48, 0.18, 1.0) if index == 2 else (0.90, 0.68, 0.20, 1.0)),
        )


def add_d1_terrain(spec: mujoco.MjSpec, arena: str, friction: float) -> None:
    """Add collision-identical visual terrain behind one construction seam."""

    if arena not in SUPPORTED_D1_ARENAS:
        raise ValueError(f"arena must be one of {SUPPORTED_D1_ARENAS}")
    spec.add_texture(
        name="sky_gradient",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=(0.16, 0.24, 0.34),
        rgb2=(0.01, 0.02, 0.04),
        width=512,
        height=3072,
    )
    spec.add_texture(
        name="floor_checker",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=(0.09, 0.13, 0.17),
        rgb2=(0.16, 0.22, 0.28),
        markrgb=(0.25, 0.34, 0.42),
        width=512,
        height=512,
    )
    spec.add_material(
        name="floor_material",
        textures=("floor_checker",),
        texrepeat=((40.0, 28.0) if arena == "course" else (30.0, 30.0)),
        reflectance=0.08,
        roughness=0.8,
    )
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=((20.0, 20.0, 0.05) if arena == "flat" else (12.0, 7.0, 0.05)),
        contype=1,
        conaffinity=1,
        condim=3,
        friction=(friction, 0.005, 0.0001),
        material="floor_material",
        rgba=((0.18, 0.22, 0.26, 1.0) if arena == "flat" else (0.12, 0.16, 0.20, 1.0)),
    )
    if arena == "flat":
        return
    _add_rough_blocks(spec, friction)
    _add_ramp(spec, friction)
    _add_stairs(spec, friction)
    _add_wave_bumps(spec, friction)
    _add_jump_lane(spec, friction)
