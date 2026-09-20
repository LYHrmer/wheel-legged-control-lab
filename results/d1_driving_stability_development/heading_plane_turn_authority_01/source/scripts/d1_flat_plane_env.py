"""Independently versioned D1 flat *native plane* collision diagnostic environment.

Motivation
----------
The frozen D1 flat locomotion terrain is an all-zero MuJoCo heightfield.  Saved
pose geometry combined with executed contact forces showed loaded, nearly
horizontal contact normals at interior heightfield grid lines, while a native
MuJoCo plane at the same height yields vertical normals.  This module provides a
separately schema-versioned, zero-residual-only diagnostic environment so the
difference can be measured without touching any frozen artifact.

Scope discipline (intentional non-goals)
---------------------------------------
* Nothing here modifies frozen sources, model arrays after construction,
  controllers, scoring, command schedules, gains, reward, state estimation,
  actuation or dynamics.
* Release reference shaping and leg damping candidates are neither combined nor
  imported.
* No training, no policy/model loading, no rollout driving.  The root runner owns
  integration, logging and the physical test matrix.
"""

from __future__ import annotations

import numbers
from dataclasses import asdict
from typing import Any

import mujoco
import numpy as np

from scripts.d1_heading_tracking_env import D1HeadingTrackingEnv
from wheel_legged_control.d1.actuator_channel import ActuatorChannel
from wheel_legged_control.d1.locomotion_env import D1WheelLegControlConfig
from wheel_legged_control.d1.locomotion_terrain import (
    LOCOMOTION_MAP_HALF_SIZE_M,
    D1LocomotionTerrainConfig,
)
from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.training_terrain import TrainingGroundReference

__all__ = [
    "FLAT_PLANE_SCHEMA",
    "FLAT_PLANE_TASK_SCHEMA",
    "D1FlatPlaneHeadingEnv",
    "D1FlatPlanePlant",
]


#: Collision-identity schema for the native-plane diagnostic arena.
FLAT_PLANE_SCHEMA = "d1-flat-plane-collision-v1"

#: Heading-task schema.  Deliberately distinct from every frozen heading schema so
#: that ``load_heading_policy`` rejects old sidecars while comparing the recorded
#: ``heading_task_config``, i.e. before any zip deserialization happens.
FLAT_PLANE_TASK_SCHEMA = "d1-heading-flat-plane-zero-task-v1"
FLAT_PLANE_PLANT_SCHEMA = "d1-native-flat-plane-plant-v1"

#: Requested-layout name that this module accepts (and only this one).
_REQUIRED_LAYOUT = "flat"

#: Expected number of independent baseline action channels (wheel_leg).
_ACTION_DIM = 8

def _as_float_list(values: Any) -> list[float]:
    """Return a plain, JSON-serializable list of floats."""
    return [float(v) for v in np.asarray(values, dtype=float).reshape(-1)]


def _is_real_scalar(value: Any) -> bool:
    """True only for genuine real scalars (bool and complex are rejected)."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, numbers.Real):
        return True
    if isinstance(value, np.generic):
        return bool(np.issubdtype(np.asarray(value).dtype, np.floating)) or bool(
            np.issubdtype(np.asarray(value).dtype, np.integer)
        )
    return False


class D1FlatPlanePlant(D1Plant):
    """D1 plant on the frozen builder's *native* flat plane arena.

    The frozen ``build_d1_model`` flat arena already carries a native
    ``mjGEOM_PLANE`` floor with the identical robot, friction, margin, integrator
    and solver configuration used by the heightfield arena; model dimensions and
    element IDs are unchanged, only the heightfield asset is absent.  This class
    therefore does not copy any builder code and does not re-implement ``reset``:
    ``D1Plant.__init__`` already compiles the model, allocates ``data`` /
    ``measurement_data``, resolves IDs, caches nominal arrays and calls its own
    ``reset()`` (which performs no time integration).

    There is deliberately no keyword by which a non-flat arena or a terrain object
    could be substituted.
    """

    def __init__(
        self,
        *,
        control_dt: float = 0.01,
        ground_friction: float = 0.9,
        sampling_mode: str = "synchronized",
        actuator_channel: ActuatorChannel | None = None,
    ) -> None:
        if sampling_mode != "synchronized":
            raise ValueError(
                "D1FlatPlanePlant requires sampling_mode='synchronized'; "
                f"got {sampling_mode!r}"
            )
        if actuator_channel is not None and not isinstance(actuator_channel, ActuatorChannel):
            raise TypeError(
                "actuator_channel must be an ActuatorChannel or None; "
                f"got {type(actuator_channel).__name__}"
            )
        if not _is_real_scalar(control_dt) or not np.isfinite(float(control_dt)) or float(control_dt) <= 0.0:
            raise ValueError(f"control_dt must be a positive finite real; got {control_dt!r}")
        if not _is_real_scalar(ground_friction) or not np.isfinite(float(ground_friction)):
            raise ValueError(f"ground_friction must be a finite real; got {ground_friction!r}")

        super().__init__(
            control_dt=float(control_dt),
            ground_friction=float(ground_friction),
            sampling_mode="synchronized",
            actuator_channel=actuator_channel,
            arena="flat",
            training_terrain=None,
            locomotion_terrain=None,
        )

    # ------------------------------------------------------------------
    # read-only geometry validation
    # ------------------------------------------------------------------
    def validate_flat_plane(self) -> None:
        """Verify the compiled collision identity.  Read-only; never repairs.

        Raises ``RuntimeError`` if the floor is not an unmutated native plane.
        """
        model = self.model
        floor_id = int(self.floor_geom_id)
        if floor_id < 0 or floor_id >= int(model.ngeom):
            raise RuntimeError(f"invalid floor_geom_id {floor_id} (ngeom={int(model.ngeom)})")

        geom_type = int(model.geom_type[floor_id])
        if geom_type != int(mujoco.mjtGeom.mjGEOM_PLANE):
            raise RuntimeError(
                "floor geom is not mjGEOM_PLANE "
                f"(geom_type={geom_type}, expected {int(mujoco.mjtGeom.mjGEOM_PLANE)})"
            )

        body_id = int(model.geom_bodyid[floor_id])
        if body_id != 0:
            raise RuntimeError(f"floor plane must be attached to the world body 0; got {body_id}")

        quat = np.asarray(model.geom_quat[floor_id], dtype=float)
        if not np.array_equal(quat, np.array([1.0, 0.0, 0.0, 0.0])):
            raise RuntimeError(f"floor plane orientation is not identity; geom_quat={_as_float_list(quat)}")

        pos = np.asarray(model.geom_pos[floor_id], dtype=float)
        if not (pos[0] == 0.0 and pos[1] == 0.0 and pos[2] == 0.0):
            raise RuntimeError(f"floor plane position must be exactly (0, 0, 0); got {_as_float_list(pos)}")

        if int(model.nhfield) != 0:
            raise RuntimeError(f"native plane arena must contain no heightfields; nhfield={int(model.nhfield)}")
        if int(model.geom_dataid[floor_id]) != -1:
            raise RuntimeError("native plane floor must not reference a geometry asset")

        terrain_ids = frozenset(int(g) for g in self.terrain_geom_ids)
        if terrain_ids != frozenset({floor_id}):
            raise RuntimeError(
                "expected exactly one terrain geom equal to the floor plane; "
                f"got {sorted(terrain_ids)} with floor_geom_id={floor_id}"
            )

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------
    @property
    def collision_terrain_metadata(self) -> dict[str, Any]:
        """Fresh, serializable snapshot of the *actual* compiled collision identity.

        A new dict is built on every access; no mutable state is shared with
        callers.  All numbers are read from the compiled model.
        """
        self.validate_flat_plane()

        model = self.model
        floor_id = int(self.floor_geom_id)
        pos = np.asarray(model.geom_pos[floor_id], dtype=float)
        quat = np.asarray(model.geom_quat[floor_id], dtype=float)

        normal = np.zeros(3, dtype=float)
        mujoco.mju_rotVecQuat(normal, np.array([0.0, 0.0, 1.0], dtype=float), quat)

        half_x, half_y = (float(LOCOMOTION_MAP_HALF_SIZE_M[0]), float(LOCOMOTION_MAP_HALF_SIZE_M[1]))

        return {
            "schema": FLAT_PLANE_SCHEMA,
            "plant_schema": FLAT_PLANE_PLANT_SCHEMA,
            "collision_geometry": "native_mujoco_plane",
            "geom_type_name": "mjGEOM_PLANE",
            "geom_type": int(model.geom_type[floor_id]),
            "floor_geom_id": floor_id,
            "floor_body_id": int(model.geom_bodyid[floor_id]),
            "floor_position_m": _as_float_list(pos),
            "floor_height_m": float(pos[2]),
            "floor_quat": _as_float_list(quat),
            "surface_normal": _as_float_list(normal),
            "heightfield_present": False,
            "heightfield_count": int(model.nhfield),
            "terrain_geom_ids": sorted(int(g) for g in self.terrain_geom_ids),
            "geom_friction": _as_float_list(model.geom_friction[floor_id]),
            "geom_condim": int(model.geom_condim[floor_id]),
            "geom_margin_m": float(model.geom_margin[floor_id]),
            "geom_gap_m": float(model.geom_gap[floor_id]),
            "geom_solref": _as_float_list(model.geom_solref[floor_id]),
            "geom_solimp": _as_float_list(model.geom_solimp[floor_id]),
            "physics_timestep_s": float(model.opt.timestep),
            "control_dt_s": float(self.control_dt),
            "physics_steps_per_control": int(self.physics_steps),
            "mujoco_version": str(mujoco.mj_versionString()),
            # The plane itself is unbounded; only the evaluation / ground-reference
            # domain is finite.  This is *not* a finite collision map.
            "physical_geometry_extent": "infinite",
            "evaluation_domain_half_size_m": [half_x, half_y],
            "reference_domain_half_size_m": [half_x, half_y],
            "domain_note": (
                "physical plane geometry is infinite; the finite half sizes bound "
                "only the evaluation and ground-reference domain"
            ),
        }

    # ------------------------------------------------------------------
    # ground reference
    # ------------------------------------------------------------------
    def locomotion_ground_reference(self, x, y) -> TrainingGroundReference:
        """Exact analytic ground reference for the native plane.

        No heightfield interpolation is used, and no clamping or extrapolation
        happens here: coordinates strictly outside the finite reference domain
        raise ``ValueError``.  The base environment's ``_bounded_ground_query``
        remains responsible for clamping solely to produce a terminal-ready
        publication and for terminating on domain exit; the existing safe envelope
        is untouched.
        """
        xf = self._checked_coordinate("x", x)
        yf = self._checked_coordinate("y", y)

        half_x = float(LOCOMOTION_MAP_HALF_SIZE_M[0])
        half_y = float(LOCOMOTION_MAP_HALF_SIZE_M[1])
        if abs(xf) > half_x or abs(yf) > half_y:
            raise ValueError(
                "locomotion_ground_reference query outside the finite reference domain "
                f"|x|<={half_x}, |y|<={half_y}; got (x={xf!r}, y={yf!r})"
            )

        self.validate_flat_plane()
        height = float(np.asarray(self.model.geom_pos[int(self.floor_geom_id)], dtype=float)[2])
        return TrainingGroundReference(height_m=height, pitch_rad=0.0, roll_rad=0.0)

    @staticmethod
    def _checked_coordinate(name: str, value: Any) -> float:
        if not _is_real_scalar(value):
            raise ValueError(f"{name} must be a real scalar (bool/complex rejected); got {value!r}")
        out = float(value)
        if not np.isfinite(out):
            raise ValueError(f"{name} must be finite; got {value!r}")
        return out


class D1FlatPlaneHeadingEnv(D1HeadingTrackingEnv):
    """Heading environment whose collision arena is a native plane.

    The frozen parent constructor builds an unstepped plant plus the wheel_leg
    baseline controller, sets ``loop = None`` and does *not* call ``env.reset`` or
    integrate anything.  After ``super().__init__`` returns, the (never stepped,
    never integrated) heightfield plant is replaced by a ``D1FlatPlanePlant``
    carrying the same control period, floor friction, sampling mode and actuator
    channel.  The original controller object is independent of the plant and is
    retained exactly as constructed: it is not replaced, not reset and its
    ``physical_action_schema`` is untouched.  Nothing global is monkeypatched.

    The discarded plant's construction-time compile and ``reset()`` are model
    initialization only; they are not a trajectory.
    """

    #: Flows through the frozen base ``reset`` bookkeeping.
    task_schema = FLAT_PLANE_TASK_SCHEMA

    def __init__(self, *, terrain: D1LocomotionTerrainConfig | None = None, **kwargs: Any) -> None:
        if terrain is None:
            # D1LocomotionTerrainConfig itself rejects active slope/ripple/step/phase
            # parameters for the flat layout.
            terrain = D1LocomotionTerrainConfig(layout=_REQUIRED_LAYOUT)
        if not isinstance(terrain, D1LocomotionTerrainConfig):
            raise TypeError(
                "terrain must be a D1LocomotionTerrainConfig or None; "
                f"got {type(terrain).__name__}"
            )
        if str(terrain.layout) != _REQUIRED_LAYOUT:
            raise ValueError(
                f"D1FlatPlaneHeadingEnv requires layout={_REQUIRED_LAYOUT!r}; "
                f"got {terrain.layout!r}"
            )
        provider = kwargs.get("provider_config")
        if provider is not None and provider.kind != "oracle":
            raise ValueError("flat-plane diagnostic requires the oracle state provider")
        controller = kwargs.get("wheel_leg_control")
        if controller is not None and controller != D1WheelLegControlConfig():
            raise ValueError("flat-plane diagnostic requires the original default controller")

        super().__init__(terrain=terrain, **kwargs)

        old_plant = self.plant
        if float(old_plant.data.time) != 0.0:
            raise AssertionError(
                f"expected an unstepped plant at time 0.0 before replacement; got {float(old_plant.data.time)!r}"
            )
        if self.loop is not None or self.decision is not None or self._active:
            raise AssertionError("expected an inactive unbound environment before plant replacement")

        floor_friction = float(
            np.asarray(old_plant.model.geom_friction[int(old_plant.floor_geom_id)], dtype=float)[0]
        )
        self.plant = D1FlatPlanePlant(
            control_dt=float(old_plant.control_dt),
            ground_friction=floor_friction,
            sampling_mode="synchronized",
            actuator_channel=old_plant.actuator_channel,
        )
        self.plant.validate_flat_plane()

        # ``self.terrain`` keeps the legacy *requested* layout for base bookkeeping;
        # the new collision identity is recorded explicitly and separately.
        self.collision_terrain_schema = FLAT_PLANE_SCHEMA
        self.requested_terrain_layout = str(terrain.layout)

    # ------------------------------------------------------------------
    @property
    def collision_terrain_metadata(self) -> dict[str, Any]:
        """Fresh actual-geometry metadata from the active plane plant."""
        return self.plant.collision_terrain_metadata

    # ------------------------------------------------------------------
    def _build_heading_task_config(self) -> dict[str, Any]:
        """Add a mandatory, stable collision identity to the heading task config.

        Called from the *parent constructor*, i.e. before the plant replacement,
        so only stable module constants are used here; never runtime plant
        metadata.  The parent config historically excludes terrain, which is why
        this identity block is required to make old sidecars mismatch.
        """
        config = super()._build_heading_task_config()
        if not isinstance(config, dict):
            raise TypeError(
                f"_build_heading_task_config must return a dict; got {type(config).__name__}"
            )
        config["task_schema"] = FLAT_PLANE_TASK_SCHEMA
        config["collision_terrain"] = {
            "schema": FLAT_PLANE_SCHEMA,
            "plant_schema": FLAT_PLANE_PLANT_SCHEMA,
            "collision_geometry": "native_mujoco_plane",
            "geom_type_name": "mjGEOM_PLANE",
            "heightfield_present": False,
            "nominal_floor_height_m": 0.0,
            "nominal_surface_normal": [0.0, 0.0, 1.0],
            "physical_geometry_extent": "infinite",
            "evaluation_domain_half_size_m": [
                float(LOCOMOTION_MAP_HALF_SIZE_M[0]),
                float(LOCOMOTION_MAP_HALF_SIZE_M[1]),
            ],
            "requested_terrain_layout": _REQUIRED_LAYOUT,
            "zero_residual_only": True,
        }
        return config

    # ------------------------------------------------------------------
    def reset(self, *args: Any, **kwargs: Any):
        """Validate the plane, delegate once, then enrich episode metadata."""
        self.plant.validate_flat_plane()
        obs, info = super().reset(*args, **kwargs)

        self._episode_metadata["collision_terrain"] = self.plant.collision_terrain_metadata
        self._episode_metadata["terrain"] = self.plant.collision_terrain_metadata
        self._episode_metadata["requested_terrain_config"] = asdict(self.terrain)
        # Legacy requested layout is recorded separately and keeps its own naming;
        # its historical schema is never relabelled as a plane schema.
        self._episode_metadata["requested_terrain_layout"] = str(self.terrain.layout)

        if isinstance(info, dict):
            info["episode_metadata"] = self.episode_metadata
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action, *args: Any, **kwargs: Any):
        """Zero-residual-only step guard, then a single delegation to the parent."""
        self._require_zero_action(action)
        return super().step(action, *args, **kwargs)

    @staticmethod
    def _require_zero_action(action: Any) -> None:
        if isinstance(action, (bool, np.bool_)):
            raise ValueError("action must be a numeric array of shape (8,), not a bool")  # noqa: TRY004
        array = np.asarray(action)
        if array.dtype == np.bool_ or not (
            np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)
        ):
            raise ValueError(f"action must be a real numeric array; got dtype {array.dtype!r}")
        if array.shape != (_ACTION_DIM,):
            raise ValueError(f"action must have shape ({_ACTION_DIM},); got {array.shape}")
        values = array.astype(float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"action must be all finite; got {values.tolist()!r}")
        if np.any(values != 0.0):
            raise ValueError(
                "D1FlatPlaneHeadingEnv is a zero-residual-only diagnostic environment; "
                f"action must be exactly zero, got {values.tolist()!r}"
            )
