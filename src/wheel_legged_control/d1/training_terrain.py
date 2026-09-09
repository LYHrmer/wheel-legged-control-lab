"""Small, mutable heightfields for straight-line D1 training experiments.

The reference is an oracle read from the actual collision heightfield, not a
terrain estimate or a deployable sensor. Profiles are constant across y. Their
piecewise-linear x interpolation is therefore also the height of both collision
triangles in each grid cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import mujoco
import numpy as np

TRAINING_TERRAIN_X_HALF_SIZE_M = 6.0
TRAINING_TERRAIN_Y_HALF_SIZE_M = 3.0
TRAINING_TERRAIN_GRID_SPACING_M = 0.01
_HEIGHTFIELD_NAME = "training_ground"
_HEIGHTFIELD_NCOL = 1201
_HEIGHTFIELD_NROW = 2
_HEIGHTFIELD_Z_SPAN_M = 1.4
_HEIGHTFIELD_Z_OFFSET_M = -0.7


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class TrainingTerrainConfig:
    """Physical profile parameters, in metres and degrees.

    ``amplitude_m`` is the sine amplitude, not its peak-to-peak height. Positive
    ``slope_deg`` rises towards world +x. The ramp-angle limit does not constrain
    the local slope of bumps. Unused amplitude/slope parameters must be zero so
    recorded configurations cannot silently describe a different surface.
    """

    kind: str = "flat"
    amplitude_m: float = 0.0
    slope_deg: float = 0.0
    wavelength_m: float = 0.8
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("flat", "bumps", "ramp"):
            raise ValueError("kind must be flat, bumps, or ramp")
        for name in ("amplitude_m", "slope_deg", "wavelength_m", "phase_rad"):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))
        if not 0.0 <= self.amplitude_m <= 0.03:
            raise ValueError("amplitude_m must lie in [0, 0.03]")
        if abs(self.slope_deg) > 6.0:
            raise ValueError("slope_deg must lie in [-6, 6]")
        if self.wavelength_m < 0.3:
            raise ValueError("wavelength_m must be at least 0.3")
        if self.kind != "bumps" and self.amplitude_m != 0.0:
            raise ValueError("amplitude_m must be zero unless kind is bumps")
        if self.kind != "ramp" and self.slope_deg != 0.0:
            raise ValueError("slope_deg must be zero unless kind is ramp")


@dataclass(frozen=True)
class TrainingGroundReference:
    """Oracle collision height and world-axis tangent orientation.

    ``pitch_rad`` follows D1's RPY convention: an uphill +x tangent has negative
    pitch. ``roll_rad`` is zero because these profiles do not vary across y.
    At a grid knot the right-hand x segment is selected, except at the last knot.
    """

    height_m: float
    pitch_rad: float
    roll_rad: float


def add_training_heightfield(spec: mujoco.MjSpec) -> None:
    """Replace the existing flat floor before compiling; preserve its identity."""

    # Two equal rows suffice for an extruded 1-D surface and avoid unnecessary
    # collision cells across the wheel width. Size/geom position stay fixed for
    # every episode, including flat ones; only normalized elevations change.
    spec.add_hfield(
        name=_HEIGHTFIELD_NAME,
        size=(
            TRAINING_TERRAIN_X_HALF_SIZE_M,
            TRAINING_TERRAIN_Y_HALF_SIZE_M,
            _HEIGHTFIELD_Z_SPAN_M,
            0.1,
        ),
        nrow=_HEIGHTFIELD_NROW,
        ncol=_HEIGHTFIELD_NCOL,
        userdata=[0.0] * (_HEIGHTFIELD_NROW * _HEIGHTFIELD_NCOL),
    )
    floor = spec.geom("floor")
    floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
    floor.hfieldname = _HEIGHTFIELD_NAME
    floor.pos = (0.0, 0.0, _HEIGHTFIELD_Z_OFFSET_M)


def _heightfield_data(model: mujoco.MjModel) -> np.ndarray:
    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, _HEIGHTFIELD_NAME)
    if hfield_id < 0:
        raise ValueError("plant was not constructed with a training terrain")
    address = int(model.hfield_adr[hfield_id])
    count = _HEIGHTFIELD_NROW * _HEIGHTFIELD_NCOL
    return model.hfield_data[address : address + count].reshape(
        _HEIGHTFIELD_NROW, _HEIGHTFIELD_NCOL
    )


def update_training_heightfield(model: mujoco.MjModel, config: TrainingTerrainConfig) -> None:
    """Write a compiled collision surface; caller must reset before stepping.

    Compile-time normalization is deliberately not used: it would turn a 5 mm
    and a 10 mm sine wave into the same height. A renderer created before this
    update additionally needs ``mjr_uploadHField`` to refresh its GPU copy.
    """

    if not isinstance(config, TrainingTerrainConfig):
        raise TypeError("config must be a TrainingTerrainConfig")
    data = _heightfield_data(model)
    x = np.linspace(
        -TRAINING_TERRAIN_X_HALF_SIZE_M,
        TRAINING_TERRAIN_X_HALF_SIZE_M,
        _HEIGHTFIELD_NCOL,
    )
    if config.kind == "ramp":
        height = np.tan(np.deg2rad(config.slope_deg)) * x
    elif config.kind == "bumps":
        height = config.amplitude_m * np.sin(
            (2.0 * np.pi / config.wavelength_m) * x + config.phase_rad
        )
    else:
        height = np.zeros_like(x)
    # Fixed bounds contain the full allowed profile family. MuJoCo stores the
    # data as float32; oracle queries below read that same rounded data.
    data[:] = (height - _HEIGHTFIELD_Z_OFFSET_M) / _HEIGHTFIELD_Z_SPAN_M


def training_ground_reference(model: mujoco.MjModel, x: float, y: float) -> TrainingGroundReference:
    """Query the collision triangles in O(1), refusing points outside the map."""

    x = _finite_real(x, "x")
    y = _finite_real(y, "y")
    if abs(x) > TRAINING_TERRAIN_X_HALF_SIZE_M or abs(y) > TRAINING_TERRAIN_Y_HALF_SIZE_M:
        raise ValueError("reference point lies outside the finite training terrain")
    data = _heightfield_data(model)
    coordinate = (x + TRAINING_TERRAIN_X_HALF_SIZE_M) / TRAINING_TERRAIN_GRID_SPACING_M
    index = min(int(np.floor(coordinate)), _HEIGHTFIELD_NCOL - 2)
    fraction = coordinate - index
    left = float(data[0, index])
    right = float(data[0, index + 1])
    height = _HEIGHTFIELD_Z_OFFSET_M + _HEIGHTFIELD_Z_SPAN_M * (left + fraction * (right - left))
    slope = _HEIGHTFIELD_Z_SPAN_M * (right - left) / TRAINING_TERRAIN_GRID_SPACING_M
    return TrainingGroundReference(
        height_m=height, pitch_rad=-float(np.arctan(slope)), roll_rad=0.0
    )
