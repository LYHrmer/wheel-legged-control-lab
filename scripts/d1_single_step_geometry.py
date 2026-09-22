"""Single-step (read-only) geometry qualification for the D1 wheel-legged robot
standing in front of a fixed native MuJoCo plane plus one axis-aligned 15 mm box.

This module is *strictly* geometric and read-only:

* It never calls ``mj_step``, ``mj_forward``, ``mj_kinematics`` or any other
  routine that advances or re-derives state.
* It never mutates ``model`` or ``data`` and never fabricates hidden caches.
* It only reads compiled model fields (``geom_type``, ``geom_size``,
  ``geom_bodyid``, ``geom_contype``, ``geom_conaffinity``, ``geom_margin``,
  ``body_geomadr``, ``body_geomnum``) and the already-populated pose cache
  (``data.geom_xpos``, ``data.geom_xmat``).

Documented limitations
----------------------
1. World AABBs describe the *sampled pose only*.  A box/AABB overlap therefore
   proves nothing about continuous-time motion: it is not a swept-volume test
   and gives no continuous-time penetration or tunnelling guarantee.  It is an
   conservative bound on the true geometry. Disjoint AABBs imply disjoint
   geometry; overlapping AABBs alone do not prove contact.
2. ``box_contact_feature`` *interprets* a contact normal that the caller has
   already corrected for MuJoCo geom ordering (i.e. it must already point from
   the box towards the robot).  It does not solve contacts, does not read the
   solver, and certifies no positive normal load.  Contact positions coming out
   of MuJoCo are the native solver cache for the sampled step, not a
   synchronised endpoint; the position tolerance is caller-owned and should be
   derived from the actual contact ``dist``/``margin``, never from a force
   threshold.
3. Visual decoration geoms (mesh wheels, shells) are deliberately excluded from
   every crossing decision; only geoms that actually take part in collision
   filtering are considered, and mesh/height-field/SDF collision geoms are
   rejected rather than approximated by a guessed radius.

No CLI, and no imports from the flat-plane clearance helpers.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

__all__ = [
    "box_contact_feature",
    "collision_identity",
    "geom_world_bounds",
    "robot_collision_bounds",
    "robot_collision_ids",
]

# Default D1 source-model ordering: legs FL/FR/RL/RR, joints hip/thigh/calf/foot.
# The four wheels are the "*_foot" bodies (compiled collision cylinders,
# radius 0.087 m, half-height 0.02 m; their visual meshes are excluded here).
_WHEEL_BODY_ORDER: tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

# Face naming for the axis-aligned obstacle box, keyed by (axis, sign).
_FACE_NAMES: dict[tuple[int, int], str] = {
    (0, -1): "front",
    (0, +1): "back",
    (1, -1): "right",
    (1, +1): "left",
    (2, -1): "bottom",
    (2, +1): "top",
}

_NORMAL_UNIT_TOL = 1e-6
_AXIS_ACTIVE_TOL = 1e-6
# Interpretation allowance only; never modify collision parameters or forces.
# MuJoCo's multi-run CCD perturbs orientations by 1e-3 radians. The fixed
# 0.0021 cone residual allowance is qualified on saved static contacts, not a
# physical success gate or a claimed universal solver error bound.
NORMAL_CONE_RESIDUAL_LIMIT = 0.0021


def _as_vec3(value: Any, label: str) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float64).reshape(-1)
    if vec.size != 3:
        raise ValueError(f"{label} must have exactly 3 components, got {vec.size}")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{label} must be finite, got {vec.tolist()}")
    return vec


def _check_geom_id(model: Any, geom_id: int) -> int:
    gid = int(geom_id)
    if gid < 0 or gid >= int(model.ngeom):
        raise ValueError(f"geom_id {gid} outside compiled range [0, {int(model.ngeom)})")
    return gid


def collision_identity(model: Any, geom_id: int) -> str:
    """Stable identity string for a geom that survives box-insertion ID shifts.

    Uses the compiled geom name when one exists.  Unnamed imported robot geoms
    (the common case for the D1 URDF/MJCF import) fall back to the canonical
    owning-body name plus the geom's ordinal *within that body*, which is
    invariant under global ID renumbering caused by inserting the obstacle box.
    """
    gid = _check_geom_id(model, geom_id)
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    if name:
        return str(name)

    body_id = int(model.geom_bodyid[gid])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    body_label = str(body_name) if body_name else f"body_{body_id}"
    ordinal = gid - int(model.body_geomadr[body_id])
    if ordinal < 0 or ordinal >= int(model.body_geomnum[body_id]):
        raise ValueError(f"geom {gid} is not consistent with body {body_label} geom block")
    return f"{body_label}#geom{ordinal}"


def robot_collision_ids(model: Any) -> tuple[int, ...]:
    """Physical robot collision geoms: attached to a real body and collidable.

    World terrain (plane, obstacle box owned by worldbody) is excluded via
    ``geom_bodyid > 0``; pure visual decorations are excluded because they carry
    ``contype == 0`` and ``conaffinity == 0``.
    """
    body_ids = np.asarray(model.geom_bodyid, dtype=np.int64)
    contype = np.asarray(model.geom_contype, dtype=np.int64)
    conaff = np.asarray(model.geom_conaffinity, dtype=np.int64)
    mask = (body_ids > 0) & ((contype != 0) | (conaff != 0))
    return tuple(int(i) for i in np.flatnonzero(mask))


def geom_world_bounds(
    model: Any, data: Any, geom_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact world AABB of one primitive collision geom for the sampled pose.

    Reads the already-provided ``data.geom_xpos`` / ``data.geom_xmat`` cache and
    the *compiled* ``model.geom_size`` (never a nominal/expected wheel radius).
    Supported types: sphere, ellipsoid, capsule, cylinder, box.  Any other
    collision type (mesh, height field, SDF, plane, ...) raises ``ValueError``
    instead of guessing a bounding radius.
    """
    gid = _check_geom_id(model, geom_id)
    gtype = int(model.geom_type[gid])
    size = np.asarray(model.geom_size[gid], dtype=np.float64).reshape(3)
    center = np.asarray(data.geom_xpos[gid], dtype=np.float64).reshape(3)
    rot = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)

    if not np.all(np.isfinite(center)):
        raise ValueError(f"geom {gid} world position is not finite: {center.tolist()}")
    if not np.all(np.isfinite(rot)):
        raise ValueError(f"geom {gid} world orientation is not finite")
    if not np.all(np.isfinite(size)):
        raise ValueError(f"geom {gid} compiled size is not finite: {size.tolist()}")

    g = mujoco.mjtGeom
    if gtype == int(g.mjGEOM_SPHERE):
        _require_positive(size[:1], gid, "sphere radius")
        extent = np.full(3, size[0], dtype=np.float64)
    elif gtype == int(g.mjGEOM_ELLIPSOID):
        _require_positive(size[:3], gid, "ellipsoid semi-axes")
        extent = np.sqrt(np.sum((rot * size[None, :]) ** 2, axis=1))
    elif gtype == int(g.mjGEOM_CAPSULE):
        _require_positive(size[:2], gid, "capsule radius/half-length")
        extent = size[0] + size[1] * np.abs(rot[:, 2])
    elif gtype == int(g.mjGEOM_CYLINDER):
        _require_positive(size[:2], gid, "cylinder radius/half-height")
        radial = np.sqrt(rot[:, 0] ** 2 + rot[:, 1] ** 2)
        extent = size[1] * np.abs(rot[:, 2]) + size[0] * radial
    elif gtype == int(g.mjGEOM_BOX):
        _require_positive(size[:3], gid, "box half-sizes")
        extent = np.abs(rot) @ size
    else:
        raise ValueError(
            f"geom {gid} ({collision_identity(model, gid)}) has unsupported collision "
            f"type {gtype}; exact AABB requires sphere/ellipsoid/capsule/cylinder/box"
        )

    extent = np.asarray(extent, dtype=np.float64).reshape(3)
    return center - extent, center + extent


def _require_positive(values: np.ndarray, gid: int, label: str) -> None:
    if not np.all(values > 0.0):
        raise ValueError(f"geom {gid} has non-positive {label}: {values.tolist()}")


def _wheel_index(body_name: str) -> int | None:
    try:
        return _WHEEL_BODY_ORDER.index(body_name)
    except ValueError:
        return None


def robot_collision_bounds(model: Any, data: Any) -> list[dict[str, Any]]:
    """JSON-serialisable world AABB record per physical robot collision geom."""
    records: list[dict[str, Any]] = []
    for gid in robot_collision_ids(model):
        lo, hi = geom_world_bounds(model, data, gid)
        body_id = int(model.geom_bodyid[gid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        body_label = str(body_name) if body_name else f"body_{body_id}"
        records.append(
            {
                "geom_id": int(gid),
                "identity": collision_identity(model, gid),
                "body_name": body_label,
                "geom_type": int(model.geom_type[gid]),
                "minimum_world_m": [float(v) for v in lo],
                "maximum_world_m": [float(v) for v in hi],
                "margin_m": float(model.geom_margin[gid]),
                "wheel_index": _wheel_index(body_label),
            }
        )
    return records


def box_contact_feature(
    position_world_m: Any,
    normal_box_to_robot_world: Any,
    center_world_m: Any,
    half_size_m: Any,
    *,
    tolerance_m: float,
) -> dict[str, Any]:
    """Classify which face/edge/corner of the axis-aligned box supports a contact.

    ``normal_box_to_robot_world`` must already be corrected for geom order and
    normalised (unit length within 1e-6).  Horizontal front normals such as
    ``(-1, 0, 0)`` and front-top diagonals are perfectly valid: no assumption is
    made that terrain normals are vertical.

    ``tolerance_m`` is caller-owned and should come from the actual contact
    ``dist``/``margin`` of the sampled step; it is never a force threshold and
    this helper makes no claim about load.
    """
    tol = float(tolerance_m)
    if not np.isfinite(tol):
        raise ValueError(f"tolerance_m must be finite, got {tolerance_m!r}")
    if tol < 0.0:
        raise ValueError(f"tolerance_m must be non-negative, got {tol}")

    position = _as_vec3(position_world_m, "position_world_m")
    normal = _as_vec3(normal_box_to_robot_world, "normal_box_to_robot_world")
    center = _as_vec3(center_world_m, "center_world_m")
    half = _as_vec3(half_size_m, "half_size_m")
    if not np.all(half > 0.0):
        raise ValueError(f"half_size_m must be strictly positive, got {half.tolist()}")

    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError("normal_box_to_robot_world must be non-zero")
    if abs(norm - 1.0) > _NORMAL_UNIT_TOL:
        raise ValueError(
            f"normal_box_to_robot_world must be normalized within {_NORMAL_UNIT_TOL}, "
            f"got length {norm!r}"
        )

    faces: list[str] = []
    offsets: list[float] = []
    projected = np.zeros(3)
    for axis in range(3):
        if abs(normal[axis]) <= _AXIS_ACTIVE_TOL:
            continue
        sign = 1 if normal[axis] > 0.0 else -1
        support_coord = center[axis] + sign * half[axis]
        offset = abs(float(position[axis]) - float(support_coord))
        if offset > tol:
            continue
        projected[axis] = normal[axis]
        faces.append(_FACE_NAMES[(axis, sign)])
        offsets.append(offset)
    residual = float(np.linalg.norm(normal-projected))
    support_ok = bool(faces) and residual <= NORMAL_CONE_RESIDUAL_LIMIT

    lo = center - half - tol
    hi = center + half + tol
    if not (np.all(position >= lo) and np.all(position <= hi)):
        support_ok = False

    if not faces:
        feature = "unresolved"
    elif len(faces) == 1:
        feature = faces[0]
    elif len(faces) == 2:
        feature = f"{faces[0]}_{faces[1]}_edge"
    else:
        feature = f"{faces[0]}_{faces[1]}_{faces[2]}_corner"

    return {
        "feature": feature,
        "geometric_support_valid": bool(support_ok),
        "supporting_faces": faces,
        "support_offsets_m": [float(v) for v in offsets],
        "normal_box_to_robot_world": [float(v) for v in (normal / norm)],
        "normal_cone_residual": residual,
        "normal_cone_residual_limit": NORMAL_CONE_RESIDUAL_LIMIT,
    }
