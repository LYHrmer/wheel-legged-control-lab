"""Fixed, finite 2-D roads for CPU locomotion experiments.

The 12 x 6 m collision map is built once, before model compilation. Nothing
is translated under the robot or changed during an episode. The initial
strip x <= -2.4 m is level at world z=0; the intended spawn is (-3.8, 0).

Ground queries read compiled float32 collision samples, not the generating
formula. ``pitch`` and ``roll`` are world-x/world-y tangent angles, matching
the sensor provider: normal = normalize([tan(pitch), -tan(roll), 1]). They
are not a pair of coupled Euler angles for a body aligned with that normal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from numbers import Real

import mujoco
import numpy as np

from .training_terrain import TrainingGroundReference

LOCOMOTION_TERRAIN_SCHEMA = "d1-fixed-2d-road-v1"
LOCOMOTION_MAP_HALF_SIZE_M = (6.0, 3.0)
LOCOMOTION_GRID_SPACING_M = 0.05
LOCOMOTION_SPAWN_XY_M = (-3.8, 0.0)
LOCOMOTION_FLAT_STRIP_END_X_M = -2.4
_HFIELD_NAME = "locomotion_ground"
_NCOL, _NROW = 241, 121
# Each layout changes feature order/overlap as well as the lateral road path.
# Entries are (ramp start, ripple start, ripple end, step start, step end).
_LAYOUTS = {
    "straight": (-2.2, 0.5, 2.5, 3.0, 4.2),
    "left_offset": (0.0, -2.2, -0.2, 2.8, 4.0),
    "right_offset": (-0.9, 1.8, 4.6, -2.2, -1.2),
    "s_bend": (-2.1, -0.8, 2.8, 3.1, 4.3),
    "diagonal": (1.1, -2.2, 0.4, 0.0, 0.9),
}


def _real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class D1LocomotionTerrainConfig:
    """Reproducible road parameters, not a claim of traversability.

    Slopes are nominal feature slopes, not a bound on every triangle: taper,
    feature overlap and ripples also contribute. Ripples have at most the
    specified amplitude before composition. A step is a raised platform with
    a 5 cm bevel (one grid cell), not an ideal vertical riser. All distances
    and amplitudes are metres; angular parameters explicitly name their units.
    """

    layout: str = "flat"
    slope_deg: float = 0.0
    cross_slope_deg: float = 0.0
    ripple_amplitude_m: float = 0.0
    wavelength_x_m: float = 0.8
    wavelength_y_m: float = 1.2
    phase_x_rad: float = 0.0
    phase_y_rad: float = 0.0
    step_height_m: float = 0.0
    schema: str = LOCOMOTION_TERRAIN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LOCOMOTION_TERRAIN_SCHEMA:
            raise ValueError("unsupported locomotion terrain schema")
        if self.layout not in ("flat", *_LAYOUTS):
            raise ValueError(f"layout must be flat or one of {tuple(_LAYOUTS)}")
        for name in (
            "slope_deg",
            "cross_slope_deg",
            "ripple_amplitude_m",
            "wavelength_x_m",
            "wavelength_y_m",
            "phase_x_rad",
            "phase_y_rad",
            "step_height_m",
        ):
            object.__setattr__(self, name, _real(getattr(self, name), name))
        if abs(self.slope_deg) > 3.0 or abs(self.cross_slope_deg) > 2.0:
            raise ValueError("nominal slope/cross slope exceed the 3/2 degree construction limits")
        if not 0.0 <= self.ripple_amplitude_m <= 0.01:
            raise ValueError("ripple amplitude must lie in [0, 0.01] m")
        if not 0.0 <= self.step_height_m <= 0.01:
            raise ValueError("beveled step height must lie in [0, 0.01] m")
        if min(self.wavelength_x_m, self.wavelength_y_m) < 0.4:
            raise ValueError("wavelengths must be at least 0.4 m (8 grid cells)")
        if self.layout == "flat" and any(
            (
                self.slope_deg,
                self.cross_slope_deg,
                self.ripple_amplitude_m,
                self.step_height_m,
                self.phase_x_rad,
                self.phase_y_rad,
            )
        ):
            raise ValueError("flat layout cannot carry active terrain parameters")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> D1LocomotionTerrainConfig:
        fields = json.loads(value)
        if not isinstance(fields, dict):
            raise TypeError("terrain JSON must contain an object")
        return cls(**fields)


def locomotion_terrain_configs(split: str) -> tuple[D1LocomotionTerrainConfig, ...]:
    """Fixed parameter sets with disjoint layouts, wavelengths and phases.

    Development is for tuning; holdout is only for the frozen final policy.
    This is a small interpolation-oriented split, not evidence for arbitrary
    terrain generalization. Flat construction is a separate diagnostic and
    is deliberately not duplicated across the three nonflat sets.
    """
    # layout, slope, cross, amplitude, wavelength x/y, phase x/y, step
    rows = {
        "train": (
            ("straight", 1.0, 0.5, 0.005, 0.6, 0.9, 0.0, 0.2, 0.005),
            ("straight", -1.5, -0.5, 0.01, 0.9, 1.2, 0.6, 0.8, 0.01),
            ("left_offset", 1.5, -1.0, 0.0075, 0.6, 1.2, 0.6, 0.2, 0.0075),
            ("left_offset", -1.0, 1.0, 0.005, 0.9, 0.9, 0.0, 0.8, 0.005),
        ),
        "development": (
            ("right_offset", 1.25, -0.75, 0.006, 0.75, 1.05, 0.3, 0.5, 0.006),
            ("right_offset", -1.25, 0.75, 0.008, 0.85, 1.15, 0.9, 1.1, 0.008),
        ),
        "holdout": (
            ("s_bend", 1.4, 0.65, 0.009, 0.7, 1.0, 1.4, 1.6, 0.009),
            ("diagonal", -1.4, -0.65, 0.0065, 0.8, 1.1, 2.0, 2.2, 0.0065),
        ),
    }
    if split not in rows:
        raise ValueError("split must be train, development, or holdout")
    return tuple(D1LocomotionTerrainConfig(*row) for row in rows[split])


def _smooth_step(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _window(x: np.ndarray, start: float, end: float, taper: float = 0.3) -> np.ndarray:
    return _smooth_step((x - start) / taper) * _smooth_step((end - x) / taper)


def _road_heights(config: D1LocomotionTerrainConfig) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(-6.0, 6.0, _NCOL), np.linspace(-3.0, 3.0, _NROW))
    if config.layout == "flat":
        return np.zeros_like(x)
    progress = _smooth_step((x + 2.4) / 1.5)
    if config.layout == "left_offset":
        center = -0.4 * progress
    elif config.layout == "right_offset":
        center = 0.4 * progress
    elif config.layout == "s_bend":
        center = 0.45 * progress * np.sin(0.85 * (x + 2.4))
    elif config.layout == "diagonal":
        center = 0.12 * np.clip(x + 2.4, 0.0, 7.2)
    else:
        center = 0.0
    lateral = y - center
    # 2.2 m wide full-feature road, plus 0.6 m shoulders tapering to flat.
    shoulder = _smooth_step((1.7 - np.abs(lateral)) / 0.6)
    ramp_start, wave_start, wave_end, step_start, step_end = _LAYOUTS[config.layout]
    ramp_distance = np.clip(x - ramp_start, 0.0, 1.2) - np.clip(x - ramp_start - 1.2, 0.0, 1.2)
    ramp = np.tan(np.deg2rad(config.slope_deg)) * ramp_distance
    cross = (
        np.tan(np.deg2rad(config.cross_slope_deg))
        * lateral
        * _window(x, ramp_start, ramp_start + 2.4)
    )
    waves = (
        config.ripple_amplitude_m
        * np.sin(2.0 * np.pi * x / config.wavelength_x_m + config.phase_x_rad)
        * np.cos(2.0 * np.pi * lateral / config.wavelength_y_m + config.phase_y_rad)
        * _window(x, wave_start, wave_end)
    )
    step = config.step_height_m * np.minimum(
        np.clip((x - step_start) / LOCOMOTION_GRID_SPACING_M, 0.0, 1.0),
        np.clip((step_end - x) / LOCOMOTION_GRID_SPACING_M, 0.0, 1.0),
    )
    return shoulder * (ramp + cross + waves + step)


def add_locomotion_terrain(spec: mujoco.MjSpec, config: D1LocomotionTerrainConfig) -> None:
    """Replace ``floor`` before compiling, preserving friction/contact identity.

    MuJoCo normalizes supplied elevations at compile time. Matching the geom
    offset and elevation scale to the sampled min/max preserves metric relief
    rather than stretching every amplitude to one fixed height. Quantization
    is float32; the known z=0 strip can differ by a few nanometres after compile.
    """
    if not isinstance(config, D1LocomotionTerrainConfig):
        raise TypeError("config must be D1LocomotionTerrainConfig")
    floor = spec.geom("floor")
    if floor is None or floor.parent != spec.worldbody:
        raise ValueError("locomotion terrain requires a static world floor geom")
    if spec.hfield(_HFIELD_NAME) is not None:
        raise ValueError("locomotion terrain has already been added")
    height = _road_heights(config)
    minimum, maximum = float(height.min()), float(height.max())
    span = maximum - minimum
    spec.add_hfield(
        name=_HFIELD_NAME,
        size=(*LOCOMOTION_MAP_HALF_SIZE_M, span if span > 0.0 else 1.0, 0.1),
        nrow=_NROW,
        ncol=_NCOL,
        userdata=height.ravel().tolist(),
    )
    floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
    floor.hfieldname = _HFIELD_NAME
    floor.pos = (0.0, 0.0, minimum)
    floor.quat = (1.0, 0.0, 0.0, 0.0)


def locomotion_ground_reference(
    model: mujoco.MjModel,
    x: float,
    y: float,
) -> TrainingGroundReference:
    """O(1) query of the actual top-surface collision triangle, including edges.

    MuJoCo splits each cell on its bottom-left/top-right diagonal. At knots
    use the positive-side cell (last cell at the outer edge); on the diagonal
    use its x-major triangle. Height is continuous there but a crease has two
    valid one-sided normals. No implicit extrapolation exists beyond the map.

    Triangulation reference: MuJoCo 3.12.0 ``mj_rayHfield`` in
    https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_ray.c
    """
    x, y = _real(x, "x"), _real(y, "y")
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, _HFIELD_NAME)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if (
        hid < 0
        or gid < 0
        or model.geom_dataid[gid] != hid
        or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_HFIELD
    ):
        raise ValueError("model was not built with locomotion terrain")
    if (
        model.geom_bodyid[gid] != 0
        or not np.array_equal(model.geom_quat[gid], (1.0, 0.0, 0.0, 0.0))
        or np.any(model.geom_pos[gid, :2] != 0.0)
    ):
        raise ValueError("locomotion terrain must remain fixed in its original world frame")
    half_x, half_y, scale, _ = model.hfield_size[hid]
    if abs(x) > half_x or abs(y) > half_y:
        raise ValueError("reference point lies outside the finite locomotion map")
    nrow, ncol = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
    dx, dy = 2.0 * half_x / (ncol - 1), 2.0 * half_y / (nrow - 1)
    cx, cy = (x + half_x) / dx, (y + half_y) / dy
    col, row = min(int(np.floor(cx)), ncol - 2), min(int(np.floor(cy)), nrow - 2)
    fx, fy = cx - col, cy - row
    address = int(model.hfield_adr[hid])
    data = model.hfield_data[address : address + nrow * ncol].reshape(nrow, ncol)
    h00, h10 = float(data[row, col]), float(data[row, col + 1])
    h01, h11 = float(data[row + 1, col]), float(data[row + 1, col + 1])
    if fx >= fy:
        sx, sy = h10 - h00, h11 - h10
    else:
        sx, sy = h11 - h01, h01 - h00
    return TrainingGroundReference(
        height_m=float(model.geom_pos[gid, 2] + scale * (h00 + fx * sx + fy * sy)),
        pitch_rad=-float(np.arctan(scale * sx / dx)),
        roll_rad=float(np.arctan(scale * sy / dy)),
    )
