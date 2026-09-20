"""Fixed-schedule command source and read-only collision-geometry utilities for
the D1 stationary height-profile hop readiness diagnostic.

Scope (frozen Astra ultra contract, `d1-fixed-height-hop-readiness-v1`):

* :func:`fixed_height_command` is a stateless lookup of the single permitted
  legacy height schedule.  It owns no clock, no latch, no controller handle and
  no event queue; root calls it once per prepared tick through its existing raw
  callback and logs the returned label.
* :func:`bind_wheel_plane_geometry`, :func:`oriented_cylinder_gaps`,
  :func:`sample_wheel_clearance` and :func:`sampled_true_runs` are pure,
  read-only facilities.  They never construct ``MjData``, never call any
  MuJoCo dynamics/kinematics/collision routine, never mutate a model, plant or
  cache, and never keep a model/data reference.

This module deliberately contains no control law, no gain, no threshold search,
no environment, no runner and no jump mechanism.  Root retains ownership of the
native snapshots, whole-robot COM, score gates, physics recorder, exception
accounting and all experiment execution.

A command phase label never certifies a contact mode; a positive geometric gap
is not, by itself, a flight claim; sampled interval runs describe saved
evaluations only and assert nothing about unsampled continuous time.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# The command dataclass is imported from the accepted combined core.  It is a
# pure data container: no controller, interactive simulation, LQR or MPC class
# is imported or instantiated anywhere in this module.
from wheel_legged_control.d1.control_loop import D1MotionCommand

__all__ = [
    "JUMP_READINESS_CONDITIONS",
    "JUMP_READINESS_PHASES",
    "JUMP_READINESS_SCHEMA",
    "TERMINAL_PREPARED_TICK",
    "WheelPlaneBinding",
    "bind_wheel_plane_geometry",
    "fixed_height_command",
    "oriented_cylinder_gaps",
    "sample_wheel_clearance",
    "sampled_true_runs",
]


JUMP_READINESS_SCHEMA = "d1-fixed-height-hop-readiness-v1"

JUMP_READINESS_CONDITIONS = ("hold", "profile")

JUMP_READINESS_PHASES = (
    "settle",
    "crouch_request",
    "extension_request",
    "return_request",
    "lower_request",
    "hold",
)

#: Highest accepted prepared tick.  Tick 600 returns the already-prepared
#: terminal command of a 600-execution-interval episode; it is never a new
#: execution and root must not step it.
TERMINAL_PREPARED_TICK = 600

_ZERO_FORWARD = 0.0
_ZERO_YAW = 0.0

# (first_tick, last_tick_inclusive, height_m, phase_label)
_PROFILE_SEGMENTS: tuple[tuple[int, int, float, str], ...] = (
    (0, 199, 0.455, "settle"),
    (200, 224, 0.405, "crouch_request"),
    (225, 239, 0.500, "extension_request"),
    (240, 274, 0.455, "return_request"),
    (275, 319, 0.435, "lower_request"),
    (320, 600, 0.455, "hold"),
)

_HOLD_SEGMENTS: tuple[tuple[int, int, float, str], ...] = (
    (0, 199, 0.455, "settle"),
    (200, 600, 0.455, "hold"),
)

_SCHEDULES = {"hold": _HOLD_SEGMENTS, "profile": _PROFILE_SEGMENTS}


# ---------------------------------------------------------------------------
# Fixed command schedule
# ---------------------------------------------------------------------------


def _validate_tick(tick: Any) -> int:
    if isinstance(tick, (bool, np.bool_)):
        raise TypeError("tick must be a non-boolean integer, got a boolean")
    if isinstance(tick, np.integer):
        value = int(tick)
    elif type(tick) is int:
        value = tick
    elif isinstance(tick, int):
        # An int subclass that is not bool is still an exact integer value.
        value = int(tick)
    else:
        raise TypeError(
            "tick must be a Python int or numpy integer (floats, strings, "
            "booleans and integer-valued floats are rejected), got "
            f"{type(tick).__name__}"
        )
    if not math.isfinite(float(value)):
        raise ValueError("tick must be finite")
    if value < 0 or value > TERMINAL_PREPARED_TICK:
        raise ValueError(
            f"tick out of range: expected 0..{TERMINAL_PREPARED_TICK} inclusive, got {value}"
        )
    return value


def _validate_condition(condition: Any) -> str:
    if isinstance(condition, (bool, np.bool_)):
        raise TypeError("condition must be the string 'hold' or 'profile'")
    if not isinstance(condition, str):
        raise TypeError(
            "condition must be the string 'hold' or 'profile', got "
            f"{type(condition).__name__}"
        )
    if condition not in _SCHEDULES:
        raise ValueError(
            "condition must be exactly 'hold' or 'profile', got " + repr(condition)
        )
    return condition


def fixed_height_command(tick: Any, condition: Any) -> tuple[D1MotionCommand, str]:
    """Return ``(D1MotionCommand(0.0, 0.0, height), phase_label)`` for one tick.

    ``tick`` must be a finite integer in ``0..600`` inclusive (booleans, floats
    -- including integer-valued floats -- and strings are rejected).  Tick 600
    is the prepared terminal command of the episode and is never a new
    execution.  ``condition`` must be exactly ``'hold'`` or ``'profile'``.

    Raw forward and yaw are exactly positive float zero for every tick, so the
    stop latch and authority gate remain inactive by construction.  The label is
    a command phase only and certifies no measured contact mode.
    """
    tick_value = _validate_tick(tick)
    condition_value = _validate_condition(condition)

    for first, last, height, label in _SCHEDULES[condition_value]:
        if first <= tick_value <= last:
            command = D1MotionCommand(_ZERO_FORWARD, _ZERO_YAW, float(height))
            return command, label

    # Unreachable for validated ticks; guards against a table edit.
    raise RuntimeError(
        f"fixed schedule does not cover tick {tick_value} for condition {condition_value!r}"
    )


# ---------------------------------------------------------------------------
# Compiled-model geometry binding (read only)
# ---------------------------------------------------------------------------

_MJ_GEOM_PLANE = 0
_MJ_GEOM_CYLINDER = 5
_MJ_JNT_HINGE = 3

_WHEEL_BODY_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
_PLANE_GEOM_NAME = "floor"

_EXPECTED_RADIUS_M = 0.087
_EXPECTED_HALF_LENGTH_M = 0.020
_EXPECTED_WHEEL_MARGIN_M = 0.001
_EXPECTED_PLANE_MARGIN_M = 0.0
_EXPECTED_GAP_M = 0.0

_SIZE_TOL = 1.0e-9
_POS_TOL = 1.0e-12
_QUAT_TOL = 1.0e-9
_SCALAR_TOL = 1.0e-12

_UNIT_TOL = 1.0e-8
_ORTHONORMAL_TOL = 1.0e-6
_PLANE_FRAME_TOL = 1.0e-9
_CONTIGUITY_TOL_S = 1.0e-10
_MONOTONE_TOL_S = 1.0e-12


def _numeric_array(value: Any, name: str) -> np.ndarray:
    """Convert to an owned float64 array, rejecting bool/complex/string/object."""
    arr = value if isinstance(value, np.ndarray) else np.asarray(value)
    if arr.dtype.kind not in ("f", "i", "u"):
        raise TypeError(
            f"{name} must be a real numeric array (bool, complex, string and "
            f"object arrays are rejected), got dtype {arr.dtype!r}"
        )
    return np.array(arr, dtype=np.float64, copy=True)


def _require_shape(arr: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")


def _require_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real finite number, got a boolean")
    if isinstance(value, (complex, np.complexfloating)):
        raise TypeError(f"{name} must be a real finite number, got a complex value")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"{name} must be a real finite number, got {type(value).__name__}"
        )
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _validated_index(value: Any, name: str, limit: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a non-boolean integer id")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} must be a Python int or numpy integer id, got {type(value).__name__}"
        )
    index = int(value)
    if not math.isfinite(float(index)):
        raise ValueError(f"{name} must be a finite integer id")
    if index < 0 or index >= limit:
        raise ValueError(f"{name} out of range: expected 0..{limit - 1}, got {index}")
    return index


def _read_name(model: Any, kind: str, index: int) -> str:
    """Read a compiled name through model name/id read APIs only."""
    accessor = getattr(model, kind, None)
    if accessor is not None:
        try:
            name = accessor(int(index)).name
        except (AttributeError, TypeError, ValueError, IndexError, KeyError):
            name = None
        if name is not None:
            return str(name)
    try:  # pragma: no cover - binding dependent fallback
        import mujoco  # local import: name lookup only, no dynamics call

        obj = getattr(mujoco.mjtObj, "mjOBJ_" + kind.upper())
        name = mujoco.mj_id2name(model, obj, int(index))
    except (AttributeError, TypeError, ValueError, IndexError, KeyError) as exc:  # pragma: no cover
        raise ValueError(f"cannot read compiled {kind} name for id {index}: {exc}") from exc
    return "" if name is None else str(name)


def _model_array(model: Any, attr: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = getattr(model, attr, None)
    if raw is None:
        raise ValueError(f"compiled model is missing required array {attr!r}")
    arr = _numeric_array(raw, f"model.{attr}")
    if arr.ndim == 1 and len(shape) == 2 and shape[1] == 1:
        arr = arr.reshape(shape)
    if arr.ndim == 2 and len(shape) == 1:
        if arr.shape[1] != 1:
            raise ValueError(f"model.{attr} must have shape {shape}, got {arr.shape}")
        arr = arr.reshape(shape)
    _require_shape(arr, shape, f"model.{attr}")
    _require_finite(arr, f"model.{attr}")
    return arr


def _is_identity_quat(quat: np.ndarray) -> bool:
    # q and -q denote the same rotation.
    return bool(
        abs(abs(float(quat[0])) - 1.0) <= _QUAT_TOL
        and float(np.max(np.abs(quat[1:]))) <= _QUAT_TOL
    )


def _tuple_of_floats(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be an immutable tuple, got {type(value).__name__}")
    if len(value) != length:
        raise ValueError(f"{name} must have {length} entries, got {len(value)}")
    out = []
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, float):
            raise TypeError(f"{name} entries must be plain Python floats")
        if not math.isfinite(item):
            raise ValueError(f"{name} entries must be finite")
        out.append(item)
    return tuple(out)


def _tuple_of_ints(value: Any, length: int, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be an immutable tuple, got {type(value).__name__}")
    if len(value) != length:
        raise ValueError(f"{name} must have {length} entries, got {len(value)}")
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, int):
            raise TypeError(f"{name} entries must be plain Python ints")
    return tuple(int(item) for item in value)


@dataclass(frozen=True)
class WheelPlaneBinding:
    """Immutable, self-contained record of the validated wheel/plane geometry.

    Holds only numbers, strings and nested tuples: no model handle, no data
    handle and no numpy view onto compiled arrays.
    """

    schema: str
    plane_geom_id: int
    wheel_geom_ids: tuple[int, int, int, int]
    wheel_body_ids: tuple[int, int, int, int]
    radii_m: tuple[float, float, float, float]
    half_lengths_m: tuple[float, float, float, float]
    contact_margins_m: tuple[float, float, float, float]

    #: Number of geoms in the compiled model the ids above were read from.  Zero
    #: means "unknown" (hand-built bindings); a positive value lets
    #: :func:`sample_wheel_clearance` reject geometry arrays from a different
    #: model, where the same integer ids would silently address other geoms.
    model_ngeom: int = 0

    plane_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    plane_offset_m: float = 0.0
    plane_margin_m: float = 0.0
    plane_gap_m: float = 0.0
    plane_geom_name: str = _PLANE_GEOM_NAME
    plane_local_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    plane_local_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    wheel_body_names: tuple[str, str, str, str] = _WHEEL_BODY_NAMES
    wheel_gaps_m: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    wheel_joint_ids: tuple[int, int, int, int] = (0, 0, 0, 0)
    wheel_local_positions: tuple[tuple[float, float, float], ...] = field(
        default=((0.0, 0.0, 0.0),) * 4
    )
    wheel_local_quaternions: tuple[tuple[float, float, float, float], ...] = field(
        default=((1.0, 0.0, 0.0, 0.0),) * 4
    )

    def __post_init__(self) -> None:
        if self.schema != JUMP_READINESS_SCHEMA:
            raise ValueError(
                f"binding schema must be {JUMP_READINESS_SCHEMA!r}, got {self.schema!r}"
            )
        if not isinstance(self.plane_geom_id, int) or isinstance(self.plane_geom_id, bool):
            raise TypeError("plane_geom_id must be a plain Python int")
        _tuple_of_ints(self.wheel_geom_ids, 4, "wheel_geom_ids")
        _tuple_of_ints(self.wheel_body_ids, 4, "wheel_body_ids")
        _tuple_of_ints(self.wheel_joint_ids, 4, "wheel_joint_ids")
        _tuple_of_floats(self.radii_m, 4, "radii_m")
        _tuple_of_floats(self.half_lengths_m, 4, "half_lengths_m")
        _tuple_of_floats(self.contact_margins_m, 4, "contact_margins_m")
        _tuple_of_floats(self.wheel_gaps_m, 4, "wheel_gaps_m")
        _tuple_of_floats(self.plane_normal, 3, "plane_normal")
        _tuple_of_floats(self.plane_local_position, 3, "plane_local_position")
        _tuple_of_floats(self.plane_local_quaternion, 4, "plane_local_quaternion")
        if any(r <= 0.0 for r in self.radii_m):
            raise ValueError("radii_m must be strictly positive")
        if any(half <= 0.0 for half in self.half_lengths_m):
            raise ValueError("half_lengths_m must be strictly positive")
        _finite_float(self.plane_offset_m, "plane_offset_m")
        _finite_float(self.plane_margin_m, "plane_margin_m")
        _finite_float(self.plane_gap_m, "plane_gap_m")
        if isinstance(self.model_ngeom, bool) or not isinstance(self.model_ngeom, int):
            raise TypeError("model_ngeom must be a plain Python int")
        if self.model_ngeom < 0:
            raise ValueError("model_ngeom must be non-negative")
        if self.model_ngeom:
            for gid in (self.plane_geom_id,) + tuple(self.wheel_geom_ids):
                if gid >= self.model_ngeom:
                    raise ValueError(
                        f"geom id {gid} is outside the declared model_ngeom "
                        f"{self.model_ngeom}"
                    )
        if len(set(self.wheel_geom_ids)) != 4:
            raise ValueError("wheel_geom_ids must be four distinct ids")
        if len(set(self.wheel_body_ids)) != 4:
            raise ValueError("wheel_body_ids must be four distinct ids")
        if not isinstance(self.wheel_body_names, tuple) or len(self.wheel_body_names) != 4:
            raise ValueError("wheel_body_names must be a 4-tuple")
        for nested, length, name in (
            (self.wheel_local_positions, 3, "wheel_local_positions"),
            (self.wheel_local_quaternions, 4, "wheel_local_quaternions"),
        ):
            if not isinstance(nested, tuple) or len(nested) != 4:
                raise ValueError(f"{name} must be a 4-tuple of tuples")
            for row in nested:
                _tuple_of_floats(row, length, name + " entry")

    def as_receipt(self) -> dict[str, Any]:
        """JSON-ready copy of the binding (no arrays, no handles)."""
        return {
            "schema": self.schema,
            "plane_geom_id": int(self.plane_geom_id),
            "plane_geom_name": str(self.plane_geom_name),
            "model_ngeom": int(self.model_ngeom),
            "plane_normal": list(self.plane_normal),
            "plane_offset_m": float(self.plane_offset_m),
            "plane_margin_m": float(self.plane_margin_m),
            "plane_gap_m": float(self.plane_gap_m),
            "plane_local_position": list(self.plane_local_position),
            "plane_local_quaternion": list(self.plane_local_quaternion),
            "wheel_geom_ids": list(self.wheel_geom_ids),
            "wheel_body_ids": list(self.wheel_body_ids),
            "wheel_body_names": list(self.wheel_body_names),
            "wheel_joint_ids": list(self.wheel_joint_ids),
            "wheel_local_positions": [list(row) for row in self.wheel_local_positions],
            "wheel_local_quaternions": [list(row) for row in self.wheel_local_quaternions],
            "radii_m": list(self.radii_m),
            "half_lengths_m": list(self.half_lengths_m),
            "contact_margins_m": list(self.contact_margins_m),
            "wheel_gaps_m": list(self.wheel_gaps_m),
        }


def _collidable_geoms_of_body(
    body_id: int, geom_bodyid: np.ndarray, contype: np.ndarray, conaffinity: np.ndarray
) -> list[int]:
    found = []
    for gid in range(geom_bodyid.shape[0]):
        if round(float(geom_bodyid[gid])) != body_id:
            continue
        if round(float(contype[gid])) == 0 and round(float(conaffinity[gid])) == 0:
            continue  # visual-only geom
        found.append(gid)
    return found


def bind_wheel_plane_geometry(
    model: Any,
    wheel_body_ids: Sequence[Any],
    plane_geom_id: Any = 0,
) -> WheelPlaneBinding:
    """Validate and freeze the compiled wheel-cylinder / floor-plane geometry.

    Only compiled model arrays and name/id read APIs are touched.  No MjData is
    constructed, nothing is reset, stepped, forwarded or collision-solved, and
    the model is never mutated.  Wheel collision geom ids are derived from the
    supplied foot body ids, not hardcoded.  Any unexpected mapping is rejected
    rather than repaired.
    """
    if model is None:
        raise TypeError("model must be a compiled MuJoCo model")

    nbody = _validated_index(getattr(model, "nbody", None), "model.nbody", 1 << 31)
    ngeom = _validated_index(getattr(model, "ngeom", None), "model.ngeom", 1 << 31)
    if nbody <= 0 or ngeom <= 0:
        raise ValueError("compiled model must contain at least one body and one geom")

    geom_bodyid = _model_array(model, "geom_bodyid", (ngeom,))
    geom_type = _model_array(model, "geom_type", (ngeom,))
    geom_size = _model_array(model, "geom_size", (ngeom, 3))
    geom_pos = _model_array(model, "geom_pos", (ngeom, 3))
    geom_quat = _model_array(model, "geom_quat", (ngeom, 4))
    geom_margin = _model_array(model, "geom_margin", (ngeom,))
    geom_gap = _model_array(model, "geom_gap", (ngeom,))
    geom_contype = _model_array(model, "geom_contype", (ngeom,))
    geom_conaffinity = _model_array(model, "geom_conaffinity", (ngeom,))
    body_jntadr = _model_array(model, "body_jntadr", (nbody,))
    body_jntnum = _model_array(model, "body_jntnum", (nbody,))

    njnt = int(getattr(model, "njnt", 0) or 0)
    if njnt <= 0:
        raise ValueError("compiled model must contain joints")
    jnt_type = _model_array(model, "jnt_type", (njnt,))

    nhfield = int(getattr(model, "nhfield", 0) or 0)
    if nhfield != 0:
        raise ValueError(
            f"binding requires a model with no height field, found nhfield={nhfield}"
        )

    # --- plane ------------------------------------------------------------
    plane_id = _validated_index(plane_geom_id, "plane_geom_id", ngeom)
    if round(float(geom_type[plane_id])) != _MJ_GEOM_PLANE:
        raise ValueError(
            f"geom {plane_id} is not a PLANE (type {round(float(geom_type[plane_id]))})"
        )
    if round(float(geom_bodyid[plane_id])) != 0:
        raise ValueError(f"plane geom {plane_id} must belong to world body 0")
    plane_name = _read_name(model, "geom", plane_id)
    if plane_name != _PLANE_GEOM_NAME:
        raise ValueError(
            f"plane geom {plane_id} must be named {_PLANE_GEOM_NAME!r}, got {plane_name!r}"
        )
    if float(np.max(np.abs(geom_pos[plane_id]))) > _POS_TOL:
        raise ValueError(f"plane geom {plane_id} must have zero local position")
    if not _is_identity_quat(geom_quat[plane_id]):
        raise ValueError(f"plane geom {plane_id} must have identity local quaternion")
    if abs(float(geom_margin[plane_id]) - _EXPECTED_PLANE_MARGIN_M) > _SCALAR_TOL:
        raise ValueError(
            f"plane geom {plane_id} margin must be {_EXPECTED_PLANE_MARGIN_M}, got "
            f"{float(geom_margin[plane_id])!r}"
        )
    if abs(float(geom_gap[plane_id]) - _EXPECTED_GAP_M) > _SCALAR_TOL:
        raise ValueError(f"plane geom {plane_id} gap must be 0")
    if round(float(geom_contype[plane_id])) == 0 or round(float(geom_conaffinity[plane_id])) == 0:
        raise ValueError(f"plane geom {plane_id} must have enabled collision masks")

    world_collidable = _collidable_geoms_of_body(
        0, geom_bodyid, geom_contype, geom_conaffinity
    )
    if world_collidable != [plane_id]:
        raise ValueError(
            "world body must carry exactly one collidable geom (the floor plane); "
            f"found collidable world geoms {world_collidable}"
        )

    # --- wheels -----------------------------------------------------------
    if isinstance(wheel_body_ids, (str, bytes, dict)) or not isinstance(
        wheel_body_ids, (Sequence, np.ndarray)
    ):
        raise TypeError("wheel_body_ids must be a sequence of four integer body ids")
    body_ids = list(wheel_body_ids)
    if len(body_ids) != 4:
        raise ValueError(f"wheel_body_ids must contain four ids, got {len(body_ids)}")

    resolved_bodies: list[int] = []
    for position, raw in enumerate(body_ids):
        bid = _validated_index(raw, f"wheel_body_ids[{position}]", nbody)
        resolved_bodies.append(bid)
    if len(set(resolved_bodies)) != 4:
        raise ValueError(f"wheel_body_ids must be four distinct ids, got {resolved_bodies}")

    wheel_geom_ids: list[int] = []
    radii: list[float] = []
    half_lengths: list[float] = []
    margins: list[float] = []
    gaps: list[float] = []
    joint_ids: list[int] = []
    local_positions: list[tuple[float, float, float]] = []
    local_quaternions: list[tuple[float, float, float, float]] = []

    for position, bid in enumerate(resolved_bodies):
        expected_name = _WHEEL_BODY_NAMES[position]
        actual_name = _read_name(model, "body", bid)
        if actual_name != expected_name:
            raise ValueError(
                f"wheel_body_ids[{position}]={bid} must be body {expected_name!r}, "
                f"got {actual_name!r}"
            )

        jnt_num = round(float(body_jntnum[bid]))
        jnt_adr = round(float(body_jntadr[bid]))
        if jnt_num != 1:
            raise ValueError(
                f"body {expected_name} must have exactly one joint, found {jnt_num}"
            )
        if jnt_adr < 0 or jnt_adr >= njnt:
            raise ValueError(f"body {expected_name} has an out-of-range joint address")
        if round(float(jnt_type[jnt_adr])) != _MJ_JNT_HINGE:
            raise ValueError(
                f"body {expected_name} joint {jnt_adr} must be a hinge, got type "
                f"{round(float(jnt_type[jnt_adr]))}"
            )

        collidable = _collidable_geoms_of_body(
            bid, geom_bodyid, geom_contype, geom_conaffinity
        )
        if len(collidable) != 1:
            raise ValueError(
                f"body {expected_name} must carry exactly one collidable geom, "
                f"found {collidable}"
            )
        gid = collidable[0]
        if round(float(geom_type[gid])) != _MJ_GEOM_CYLINDER:
            raise ValueError(
                f"body {expected_name} collision geom {gid} must be a CYLINDER "
                f"(type {_MJ_GEOM_CYLINDER}), got {round(float(geom_type[gid]))}"
            )
        radius = float(geom_size[gid, 0])
        half_length = float(geom_size[gid, 1])
        if abs(radius - _EXPECTED_RADIUS_M) > _SIZE_TOL:
            raise ValueError(
                f"geom {gid} radius must be {_EXPECTED_RADIUS_M} m, got {radius!r}"
            )
        if abs(half_length - _EXPECTED_HALF_LENGTH_M) > _SIZE_TOL:
            raise ValueError(
                f"geom {gid} half-length must be {_EXPECTED_HALF_LENGTH_M} m, got "
                f"{half_length!r}"
            )
        if float(np.max(np.abs(geom_pos[gid]))) > _POS_TOL:
            raise ValueError(f"geom {gid} must have zero local position")
        if not _is_identity_quat(geom_quat[gid]):
            raise ValueError(f"geom {gid} must have identity local quaternion")
        margin = float(geom_margin[gid])
        if abs(margin - _EXPECTED_WHEEL_MARGIN_M) > _SCALAR_TOL:
            raise ValueError(
                f"geom {gid} margin must be {_EXPECTED_WHEEL_MARGIN_M} m, got {margin!r}"
            )
        if abs(float(geom_gap[gid]) - _EXPECTED_GAP_M) > _SCALAR_TOL:
            raise ValueError(f"geom {gid} gap must be 0")

        wheel_geom_ids.append(int(gid))
        radii.append(radius)
        half_lengths.append(half_length)
        # Plane margin is 0 and wheel margin is 0.001, so the bound pair margin
        # is unambiguous: no margin-combination rule is needed.
        margins.append(max(margin, float(geom_margin[plane_id])))
        gaps.append(float(geom_gap[gid]))
        joint_ids.append(int(jnt_adr))
        local_positions.append(tuple(float(v) for v in geom_pos[gid]))
        local_quaternions.append(tuple(float(v) for v in geom_quat[gid]))

    if len(set(wheel_geom_ids)) != 4:
        raise ValueError(f"derived wheel geom ids must be distinct, got {wheel_geom_ids}")
    if plane_id in wheel_geom_ids:
        raise ValueError("plane geom id collides with a derived wheel geom id")

    return WheelPlaneBinding(
        schema=JUMP_READINESS_SCHEMA,
        plane_geom_id=int(plane_id),
        wheel_geom_ids=tuple(wheel_geom_ids),  # type: ignore[arg-type]
        wheel_body_ids=tuple(resolved_bodies),  # type: ignore[arg-type]
        radii_m=tuple(radii),  # type: ignore[arg-type]
        half_lengths_m=tuple(half_lengths),  # type: ignore[arg-type]
        contact_margins_m=tuple(margins),  # type: ignore[arg-type]
        model_ngeom=int(ngeom),
        plane_normal=(0.0, 0.0, 1.0),
        plane_offset_m=0.0,
        plane_margin_m=float(geom_margin[plane_id]),
        plane_gap_m=float(geom_gap[plane_id]),
        plane_geom_name=str(plane_name),
        plane_local_position=tuple(float(v) for v in geom_pos[plane_id]),  # type: ignore[arg-type]
        plane_local_quaternion=tuple(float(v) for v in geom_quat[plane_id]),  # type: ignore[arg-type]
        wheel_body_names=tuple(_WHEEL_BODY_NAMES),  # type: ignore[arg-type]
        wheel_gaps_m=tuple(gaps),  # type: ignore[arg-type]
        wheel_joint_ids=tuple(joint_ids),  # type: ignore[arg-type]
        wheel_local_positions=tuple(local_positions),
        wheel_local_quaternions=tuple(local_quaternions),
    )


# ---------------------------------------------------------------------------
# Pure oriented-cylinder / plane gap geometry
# ---------------------------------------------------------------------------


def oriented_cylinder_gaps(
    centers_world_m: Any,
    axes_world: Any,
    radii_m: Any,
    half_lengths_m: Any,
    *,
    plane_normal: Any = (0.0, 0.0, 1.0),
    plane_offset_m: Any = 0.0,
) -> np.ndarray:
    """Signed lowest-point gaps of four oriented cylinders against one plane.

    ``gap = n.c - plane_offset - [R*sqrt(max(0, 1-(n.a)^2)) + L*|n.a|]``

    Negative values (physical interpenetration/margin overlap) are returned as
    computed and never clamped; positive values are pure geometry and are not a
    flight label.  No model or data is accessed.
    """
    centers = _numeric_array(centers_world_m, "centers_world_m")
    axes = _numeric_array(axes_world, "axes_world")
    radii = _numeric_array(radii_m, "radii_m")
    half_lengths = _numeric_array(half_lengths_m, "half_lengths_m")
    normal = _numeric_array(plane_normal, "plane_normal")
    offset = _finite_float(plane_offset_m, "plane_offset_m")

    _require_shape(centers, (4, 3), "centers_world_m")
    _require_shape(axes, (4, 3), "axes_world")
    _require_shape(radii, (4,), "radii_m")
    _require_shape(half_lengths, (4,), "half_lengths_m")
    _require_shape(normal, (3,), "plane_normal")
    for arr, name in (
        (centers, "centers_world_m"),
        (axes, "axes_world"),
        (radii, "radii_m"),
        (half_lengths, "half_lengths_m"),
        (normal, "plane_normal"),
    ):
        _require_finite(arr, name)

    if not np.all(radii > 0.0):
        raise ValueError("radii_m must be strictly positive")
    if not np.all(half_lengths > 0.0):
        raise ValueError("half_lengths_m must be strictly positive")

    normal_norm = float(np.linalg.norm(normal))
    if abs(normal_norm - 1.0) > _UNIT_TOL:
        raise ValueError(
            f"plane_normal must be a unit vector within {_UNIT_TOL} (it is not "
            f"renormalized), got norm {normal_norm!r}"
        )
    axis_norms = np.linalg.norm(axes, axis=1)
    bad = np.nonzero(np.abs(axis_norms - 1.0) > _UNIT_TOL)[0]
    if bad.size:
        raise ValueError(
            f"axes_world rows {bad.tolist()} are not unit vectors within {_UNIT_TOL} "
            "(axes are not renormalized)"
        )

    n_dot_c = centers @ normal
    n_dot_a = axes @ normal
    # A tiny overshoot inside the accepted unit tolerance is clipped only to
    # keep the sqrt argument real; the physical gap itself is never clamped.
    if float(np.max(np.abs(n_dot_a))) > 1.0 + _UNIT_TOL:
        raise ValueError("axis/normal dot product exceeds the unit tolerance")
    n_dot_a = np.clip(n_dot_a, -1.0, 1.0)

    lateral = radii * np.sqrt(np.maximum(0.0, 1.0 - n_dot_a * n_dot_a))
    axial = half_lengths * np.abs(n_dot_a)
    gaps = n_dot_c - offset - (lateral + axial)

    out = np.array(gaps, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


def _selected_frames(
    geom_xpos: Any,
    geom_xmat: Any,
    indices: Sequence[int],
    expected_ngeom: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    positions = _numeric_array(geom_xpos, "geom_xpos")
    rotations = _numeric_array(geom_xmat, "geom_xmat")

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"geom_xpos must have shape (ngeom, 3), got {positions.shape}"
        )
    ngeom = int(positions.shape[0])
    if expected_ngeom is not None and ngeom != int(expected_ngeom):
        raise ValueError(
            "supplied geometry arrays must cover every geom of the bound model "
            f"({int(expected_ngeom)} geoms) so the compiled geom ids keep their "
            f"meaning, got {ngeom} rows"
        )

    if rotations.ndim == 2 and rotations.shape[1] == 9:
        rotations = rotations.reshape(rotations.shape[0], 3, 3)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(
            "geom_xmat must have shape (ngeom, 9) or (ngeom, 3, 3), got "
            f"{np.asarray(geom_xmat).shape}"
        )
    if int(rotations.shape[0]) != ngeom:
        raise ValueError(
            "geom_xpos and geom_xmat must describe the same number of geoms, got "
            f"{ngeom} and {int(rotations.shape[0])}"
        )

    for index in indices:
        if index < 0 or index >= ngeom:
            raise ValueError(
                f"geom index {index} out of range for supplied arrays of {ngeom} geoms"
            )

    sel = [int(i) for i in indices]
    sel_pos = positions[sel, :]
    sel_rot = rotations[sel, :, :]
    _require_finite(sel_pos, "selected geom_xpos rows")
    _require_finite(sel_rot, "selected geom_xmat rows")

    identity = np.eye(3, dtype=np.float64)
    for local, gid in enumerate(sel):
        rot = sel_rot[local]
        if float(np.max(np.abs(rot.T @ rot - identity))) > _ORTHONORMAL_TOL:
            raise ValueError(f"geom {gid} rotation is not orthonormal")
        if abs(float(np.linalg.det(rot)) - 1.0) > _ORTHONORMAL_TOL:
            raise ValueError(f"geom {gid} rotation is not a proper rotation")

    return sel_pos, sel_rot


def sample_wheel_clearance(
    binding: WheelPlaneBinding, geom_xpos: Any, geom_xmat: Any
) -> dict[str, Any]:
    """Summarise wheel-plane clearance from already synchronized geometry.

    ``geom_xpos``/``geom_xmat`` must already have been produced by root's
    synchronized publication; this function performs no forwarding, no refresh,
    no collision solving and no mutation.  A positive gap is geometry only and
    is never labelled airborne here.
    """
    if not isinstance(binding, WheelPlaneBinding):
        raise TypeError(
            f"binding must be a WheelPlaneBinding, got {type(binding).__name__}"
        )

    wheel_ids = list(binding.wheel_geom_ids)
    plane_id = int(binding.plane_geom_id)
    declared_ngeom = int(binding.model_ngeom)
    positions, rotations = _selected_frames(
        geom_xpos,
        geom_xmat,
        wheel_ids + [plane_id],
        expected_ngeom=declared_ngeom if declared_ngeom > 0 else None,
    )

    wheel_centers = positions[:4, :]
    wheel_axes = rotations[:4, :, 2]  # compiled cylinder axis: third frame column
    plane_center = positions[4, :]
    plane_normal = rotations[4, :, 2]  # plane normal: third frame column

    expected_normal = np.asarray(binding.plane_normal, dtype=np.float64)
    if float(np.max(np.abs(plane_normal - expected_normal))) > _PLANE_FRAME_TOL:
        raise ValueError(
            "supplied plane orientation differs from the validated horizontal "
            f"plane normal {tuple(expected_normal.tolist())}; this binding does not "
            "support a changed plane"
        )
    plane_offset = float(np.dot(plane_normal, plane_center))
    if abs(plane_offset - float(binding.plane_offset_m)) > _PLANE_FRAME_TOL:
        raise ValueError(
            "supplied plane position differs from the validated z=0 plane offset "
            f"{float(binding.plane_offset_m)}; got offset {plane_offset}"
        )

    gaps = oriented_cylinder_gaps(
        wheel_centers,
        wheel_axes,
        np.asarray(binding.radii_m, dtype=np.float64),
        np.asarray(binding.half_lengths_m, dtype=np.float64),
        plane_normal=plane_normal,
        plane_offset_m=plane_offset,
    )

    return {
        "schema": JUMP_READINESS_SCHEMA,
        "wheel_geom_ids": [int(g) for g in wheel_ids],
        "wheel_bottom_gap_m": [float(v) for v in gaps],
        "simultaneous_minimum_gap_m": float(np.min(gaps)),
        "contact_margin_m": float(max(binding.contact_margins_m)),
        "plane_geom_id": plane_id,
        "plane_normal": [float(v) for v in plane_normal],
        "plane_offset_m": float(plane_offset),
    }


# ---------------------------------------------------------------------------
# Pure sampled-run reducer
# ---------------------------------------------------------------------------


def sampled_true_runs(
    mask: Any, start_time_s: Any, end_time_s: Any
) -> list[dict[str, Any]]:
    """Reduce a boolean per-interval mask into contiguous sampled runs.

    The reducer selects no threshold and interprets no contact or force.  Runs
    of consecutive ``True`` entries are joined only when the sampled intervals
    are time-contiguous within 1e-10 s, so a gap in sampling splits a run.  The
    result describes saved sampled intervals only and makes no continuous-time
    flight claim.
    """
    mask_arr = mask if isinstance(mask, np.ndarray) else np.asarray(mask)
    if mask_arr.dtype != np.bool_:
        raise TypeError(
            f"mask must be a strict boolean array, got dtype {mask_arr.dtype!r}"
        )
    if mask_arr.ndim != 1:
        raise ValueError(f"mask must be 1-D, got shape {mask_arr.shape}")
    mask_arr = np.array(mask_arr, dtype=bool, copy=True)

    starts = _numeric_array(start_time_s, "start_time_s")
    ends = _numeric_array(end_time_s, "end_time_s")
    if starts.ndim != 1 or ends.ndim != 1:
        raise ValueError("start_time_s and end_time_s must be 1-D")
    if starts.shape != mask_arr.shape or ends.shape != mask_arr.shape:
        raise ValueError(
            "start_time_s and end_time_s must match the mask shape "
            f"{mask_arr.shape}, got {starts.shape} and {ends.shape}"
        )
    _require_finite(starts, "start_time_s")
    _require_finite(ends, "end_time_s")

    n = int(mask_arr.size)
    if n == 0:
        return []

    if not np.all(ends > starts):
        bad = int(np.nonzero(~(ends > starts))[0][0])
        raise ValueError(f"interval {bad} must have end_time_s > start_time_s")
    if n > 1:
        overlap = np.nonzero(starts[1:] + _MONOTONE_TOL_S < ends[:-1])[0]
        if overlap.size:
            raise ValueError(
                f"intervals must be monotone and non-overlapping; interval "
                f"{int(overlap[0]) + 1} starts before interval {int(overlap[0])} ends"
            )

    runs: list[dict[str, Any]] = []
    i = 0
    while i < n:
        if not bool(mask_arr[i]):
            i += 1
            continue
        j = i + 1
        while (
            j < n
            and bool(mask_arr[j])
            and abs(float(starts[j]) - float(ends[j - 1])) <= _CONTIGUITY_TOL_S
        ):
            j += 1
        runs.append(
            {
                "first_index": int(i),
                "end_index_exclusive": int(j),
                "intervals": int(j - i),
                "start_time_s": float(starts[i]),
                "end_time_s": float(ends[j - 1]),
                "summed_interval_duration_s": float(
                    np.sum(ends[i:j] - starts[i:j])
                ),
            }
        )
        i = j
    return runs
