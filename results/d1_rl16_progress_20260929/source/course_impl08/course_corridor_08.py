"""Pure conservative reset-footprint check for one nominal flat course path.

Inputs are copied from the actual compiled model and its reset data by the
root-owned construction process.  This is a preflight on a command-integrated
centerline, not a claim that the robot will follow that line or that leg
motion remains inside the reset envelope.  No model, engine or ray is imported.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np
from course_ground_08 import CourseGroundMap

SOURCE_SCHEMA = "actual_compiled_course_reset_geometry_v1"
SAFE_X_M = 11.0
SAFE_Y_M = 6.0


def _number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _point(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu":
        raise TypeError(f"{label} must be a real vector of shape {shape}")
    vector = np.asarray(array, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} must be finite")
    return vector


def _segment_hits_aabb(start: np.ndarray, end: np.ndarray,
                       low: np.ndarray, high: np.ndarray) -> bool:
    """Closed segment versus closed XY box, including touching its boundary."""
    direction = end - start
    enter, exit_ = 0.0, 1.0
    for axis in range(2):
        if direction[axis] == 0.0:
            if start[axis] < low[axis] or start[axis] > high[axis]:
                return False
            continue
        a = (low[axis] - start[axis]) / direction[axis]
        b = (high[axis] - start[axis]) / direction[axis]
        enter = max(enter, min(a, b))
        exit_ = min(exit_, max(a, b))
        if enter > exit_:
            return False
    return True


def assert_nominal_corridor_clearance(
    geometry_manifest: dict[str, Any], nominal_xy: Any, *, margin_m: float = 0.0,
) -> dict[str, Any]:
    """Reject a nominal E1 path colliding with any expanded non-floor box.

    ``geometry_manifest`` requires exactly four keys: ``model_identity``
    (containing source=SOURCE_SCHEMA and model_ngeom),
    ``base_world_position_m``, ``robot_collision_geoms`` and
    ``terrain_world_geoms``. Robot rows carry actual reset-world
    ``world_center_m``, compiled ``rbound_m``, positive ``body_id`` and exact
    ``geom_id``. Terrain rows are the 92 actual compiled world primitive rows
    accepted by CourseGroundMap. ``nominal_xy`` is the 1601 consecutive center
    positions from integrating the raw schedule through the pure servo.
    """
    if not isinstance(geometry_manifest, dict) or set(geometry_manifest) != {
        "model_identity", "base_world_position_m", "robot_collision_geoms",
        "terrain_world_geoms",
    }:
        raise ValueError("footprint preflight needs an actual compiled reset geometry manifest")
    identity = geometry_manifest["model_identity"]
    if not isinstance(identity, dict) or identity.get("source") != SOURCE_SCHEMA:
        raise ValueError("footprint preflight needs actual compiled model provenance")
    model_ngeom = identity.get("model_ngeom")
    if type(model_ngeom) is not int or model_ngeom <= 92:
        raise ValueError("compiled model geom count is missing or inconsistent")
    base = _point(geometry_manifest.get("base_world_position_m"), (3,), "reset base position")
    margin = _number(margin_m, "footprint margin")
    if margin < 0.0:
        raise ValueError("footprint margin cannot be negative")
    path = np.asarray(nominal_xy)
    if path.shape != (1601, 2) or path.dtype.kind not in "fiu":
        raise TypeError("nominal E1 centerline must contain 1601 finite XY points")
    path = np.asarray(path, dtype=np.float64)
    if not np.isfinite(path).all() or not np.array_equal(path[0], base[:2]):
        raise ValueError("nominal centerline must start at the actual reset base XY")

    terrain_rows = geometry_manifest.get("terrain_world_geoms")
    robot_rows = geometry_manifest.get("robot_collision_geoms")
    if not isinstance(terrain_rows, (tuple, list)) or len(terrain_rows) != 92:
        raise ValueError("footprint preflight requires all 92 compiled world collision geoms")
    if not isinstance(robot_rows, (tuple, list)) or not robot_rows:
        raise ValueError("footprint preflight requires actual reset robot collision geoms")
    terrain = CourseGroundMap(terrain_rows)
    terrain_ids = set(terrain.geom_ids)
    robot_ids: set[int] = set()
    radius = 0.0
    for row in robot_rows:
        if not isinstance(row, dict):
            raise TypeError("robot collision geom row must be an object")
        gid = row.get("geom_id")
        body = row.get("body_id")
        if (type(gid) is not int or not 0 <= gid < model_ngeom
                or gid in terrain_ids or gid in robot_ids
                or type(body) is not int or body <= 0
                or row.get("collision") is not True):
            raise ValueError("robot collision geom identity/body is inconsistent")
        center = _point(row.get("world_center_m"), (3,), "reset robot geom center")
        bound = _number(row.get("rbound_m"), "compiled geom rbound")
        if bound < 0.0:
            raise ValueError("compiled geom rbound cannot be negative")
        radius = max(radius, float(np.linalg.norm(center[:2] - base[:2])) + bound)
        robot_ids.add(gid)
    if any(gid < 0 or gid >= model_ngeom for gid in terrain_ids):
        raise ValueError("terrain geom ID exceeds the actual compiled model")
    expanded = radius + margin
    if (np.any(np.abs(path[:, 0]) + expanded >= SAFE_X_M)
            or np.any(np.abs(path[:, 1]) + expanded >= SAFE_Y_M)):
        raise ValueError("nominal centerline plus reset robot envelope touches course safety bounds")

    # The compiled map has already validated world body, type, orientation,
    # unique IDs and closed boxes. Its cached XY box bounds are conservative;
    # adding the robot radius cannot miss a potential footprint intersection.
    for geom, low, high in zip(
        terrain._boxes, terrain._box_xy_low, terrain._box_xy_high, strict=True
    ):
        expanded_low = low - expanded
        expanded_high = high + expanded
        for index in range(len(path) - 1):
            if _segment_hits_aabb(path[index], path[index + 1], expanded_low, expanded_high):
                raise ValueError(
                    f"nominal reset-footprint corridor intersects geom {geom.geom_id} "
                    f"at segment {index}"
                )
    return {
        "schema": "d1-course-nominal-reset-footprint-preflight-v1",
        "model_identity": dict(identity),
        "robot_collision_geom_ids": sorted(robot_ids),
        "terrain_collision_geom_ids": sorted(terrain_ids),
        "reset_robot_radial_bound_m": radius,
        "additional_margin_m": margin,
        "checked_centerline_segments": len(path) - 1,
        "nominal_bounds_m": {
            "x_min": float(np.min(path[:, 0])), "x_max": float(np.max(path[:, 0])),
            "y_min": float(np.min(path[:, 1])), "y_max": float(np.max(path[:, 1])),
        },
        "passed": True,
        "meaning": "command-integrated path against reset geom envelope, not actual trajectory",
    }
