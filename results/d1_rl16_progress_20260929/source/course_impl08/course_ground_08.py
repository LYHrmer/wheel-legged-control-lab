"""Pure vertical ground query over the *compiled* static course primitives.

The caller supplies world-body collision geom rows read from one actual model.
No MuJoCo import, model construction, forward pass, ray API, or contact refresh
occurs here. A vertical world ray intersects each oriented box by its exact
closed slab inequalities, and the highest hit above the infinite plane wins.
This is an oracle for simulation/control bookkeeping, not a deployable sensor.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

GROUND_SCHEMA = "d1-course-compiled-static-primitive-oracle-v1"


def _scalar(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu":
        raise TypeError(f"{label} must be a real array of shape {shape}")
    checked = np.ascontiguousarray(array, dtype=np.float64)
    if not np.isfinite(checked).all():
        raise ValueError(f"{label} must be finite")
    return np.frombuffer(checked.tobytes(), dtype=np.float64).reshape(shape)


def _quat_matrix(quaternion: np.ndarray) -> np.ndarray:
    if abs(float(quaternion @ quaternion) - 1.0) > 1e-12:
        raise ValueError("compiled geom quaternion must be unit length")
    w, x, y, z = quaternion
    return np.asarray((
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    ), dtype=np.float64)


@dataclass(frozen=True)
class CourseGroundHit:
    schema: str
    height_m: float
    normal_world: tuple[float, float, float]
    pitch_rad: float
    roll_rad: float
    geom_id: int
    geom_name: str


@dataclass(frozen=True)
class _Primitive:
    geom_id: int
    name: str
    kind: str
    position: np.ndarray
    size: np.ndarray
    rotation: np.ndarray


def _box_vertical_hit(box: _Primitive, x: float, y: float) -> tuple[float, np.ndarray] | None:
    """Intersect (x,y,*) with a closed oriented box; return upper hit/normal."""
    dx, dy = x - box.position[0], y - box.position[1]
    lower, upper = -math.inf, math.inf
    upper_axis = -1
    upper_sign = 0.0
    for axis in range(3):
        # Local axis value at world z: offset + slope*(z-center_z).
        slope = float(box.rotation[2, axis])
        offset = float(box.rotation[0, axis] * dx + box.rotation[1, axis] * dy)
        half = float(box.size[axis])
        if abs(slope) <= 1e-14:
            if abs(offset) > half:
                return None
            continue
        bound_a = (-half - offset) / slope + box.position[2]
        bound_b = (half - offset) / slope + box.position[2]
        low, high = min(bound_a, bound_b), max(bound_a, bound_b)
        lower = max(lower, low)
        if high < upper:
            upper, upper_axis = high, axis
            upper_sign = 1.0 if slope > 0 else -1.0
    if lower > upper or upper_axis < 0:
        return None
    normal = upper_sign * box.rotation[:, upper_axis]
    if normal[2] <= 0.0:
        raise RuntimeError("vertical box hit did not select an upward-facing boundary")
    return float(upper), normal


class CourseGroundMap:
    """Versioned, immutable snapshot of actual compiled world collision rows."""

    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        parsed: list[_Primitive] = []
        names: set[str] = set()
        ids: set[int] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError("compiled geom row must be an object")
            gid, name, kind = row.get("geom_id"), row.get("name"), row.get("type")
            if type(gid) is not int or gid < 0 or not isinstance(name, str) or not name:
                raise ValueError("compiled geom id/name invalid")
            if gid in ids or name in names or kind not in ("plane", "box"):
                raise ValueError("duplicate or unsupported compiled course geom")
            if row.get("body_id") != 0 or row.get("collision") is not True:
                raise ValueError("course ground row must be a colliding world geom")
            position = _array(row.get("position_m"), (3,), "position")
            size = _array(row.get("size_m"), (3,), "size")
            quat = _array(row.get("quaternion_wxyz"), (4,), "quaternion")
            if kind == "box" and (not name.startswith("terrain_") or np.any(size <= 0)):
                raise ValueError("course boxes must have named, positive compiled extents")
            if kind == "plane" and name != "floor":
                raise ValueError("only the named compiled floor can be a plane")
            rotation = _quat_matrix(quat)
            if kind == "plane" and (not np.array_equal(position, (0.0, 0.0, 0.0))
                                    or not np.array_equal(quat, (1.0, 0.0, 0.0, 0.0))):
                raise ValueError("course floor must remain the original world z=0 plane")
            parsed.append(_Primitive(gid, name, kind, position, size, rotation))
            ids.add(gid)
            names.add(name)
        if sum(geom.kind == "plane" for geom in parsed) != 1 or len(parsed) < 2:
            raise ValueError("course ground needs one plane and compiled terrain boxes")
        self._geoms = tuple(sorted(parsed, key=lambda geom: geom.geom_id))
        self.geom_ids = tuple(geom.geom_id for geom in self._geoms)
        self._floor = next(geom for geom in self._geoms if geom.kind == "plane")
        self._boxes = tuple(geom for geom in self._geoms if geom.kind == "box")
        self._box_position = np.stack([geom.position for geom in self._boxes])
        self._box_size = np.stack([geom.size for geom in self._boxes])
        self._box_rotation = np.stack([geom.rotation for geom in self._boxes])
        # Conservatively reject XY points that cannot intersect an oriented
        # box. Padding only broadens candidates; the exact slab still decides.
        xy_extent = np.einsum("bij,bj->bi", np.abs(self._box_rotation[:, :2, :]),
                              self._box_size) + 1e-12
        self._box_xy_low = self._box_position[:, :2] - xy_extent
        self._box_xy_high = self._box_position[:, :2] + xy_extent

    def query(self, x: Real, y: Real) -> CourseGroundHit:
        px, py = _scalar(x, "ground x"), _scalar(y, "ground y")
        floor = self._floor
        height = float(floor.position[2])
        normal = np.asarray((0.0, 0.0, 1.0))
        selected = floor
        for index, geom in enumerate(self._boxes):
            if (px < self._box_xy_low[index, 0] or px > self._box_xy_high[index, 0]
                    or py < self._box_xy_low[index, 1] or py > self._box_xy_high[index, 1]):
                continue
            hit = _box_vertical_hit(geom, px, py)
            if hit is not None and hit[0] > height:
                height, normal, selected = hit[0], hit[1], geom
        return CourseGroundHit(
            schema=GROUND_SCHEMA, height_m=height,
            normal_world=tuple(float(value) for value in normal),
            pitch_rad=math.atan2(float(normal[0]), float(normal[2])),
            roll_rad=math.atan2(-float(normal[1]), float(normal[2])),
            geom_id=selected.geom_id, geom_name=selected.name,
        )

    def sample_many(self, points_xy_m: Any) -> tuple[CourseGroundHit, ...]:
        """Evaluate many vertical rays with one broadcast slab calculation.

        The ordered result has the same highest-hit/floor/tie semantics as
        ``query``. The compiled geoms and their conservative XY bounds are
        cached at construction; no engine call or dynamic geometry read occurs.
        """
        points = np.asarray(points_xy_m)
        if points.ndim != 2 or points.shape[1] != 2 or points.dtype.kind not in "fiu":
            raise TypeError("batch ground query needs finite real shape (N,2)")
        points = np.asarray(points, dtype=np.float64)
        if not np.isfinite(points).all():
            raise ValueError("batch ground points must be finite")
        if not len(points):
            return ()
        pos, size, rotation = self._box_position, self._box_size, self._box_rotation
        xy = points[:, None, :]
        candidate_xy = np.all((xy >= self._box_xy_low) & (xy <= self._box_xy_high), axis=2)
        dx = points[:, None, 0] - pos[None, :, 0]
        dy = points[:, None, 1] - pos[None, :, 1]
        offset = (dx[:, :, None] * rotation[None, :, 0, :]
                  + dy[:, :, None] * rotation[None, :, 1, :])
        slope = rotation[:, 2, :]
        moving = np.abs(slope) > 1e-14
        horizontal_valid = moving[None, :, :] | (np.abs(offset) <= size[None, :, :])
        safe_slope = np.where(moving, slope, 1.0)
        a = (-size[None, :, :] - offset) / safe_slope[None, :, :] + pos[None, :, 2, None]
        b = (size[None, :, :] - offset) / safe_slope[None, :, :] + pos[None, :, 2, None]
        lower = np.max(np.where(moving[None, :, :], np.minimum(a, b), -np.inf), axis=2)
        upper_by_axis = np.where(moving[None, :, :], np.maximum(a, b), np.inf)
        upper = np.min(upper_by_axis, axis=2)
        valid = (candidate_xy & np.all(horizontal_valid, axis=2)
                 & np.any(moving, axis=1)[None, :] & (lower <= upper)
                 & (upper > self._floor.position[2]))
        elevations = np.where(valid, upper, -np.inf)
        box_index = np.argmax(elevations, axis=1)
        row_index = np.arange(len(points))
        selected_height = elevations[row_index, box_index]
        on_box = np.isfinite(selected_height)
        selected_axis = np.argmin(upper_by_axis[row_index, box_index], axis=1)
        selected_rotation = rotation[box_index]
        selected_normal = np.take_along_axis(
            selected_rotation, selected_axis[:, None, None], axis=2,
        ).squeeze(axis=2)
        selected_slope = slope[box_index, selected_axis]
        selected_normal *= np.where(selected_slope > 0.0, 1.0, -1.0)[:, None]
        hits = []
        for index, is_box in enumerate(on_box):
            if is_box:
                geom = self._boxes[int(box_index[index])]
                normal = selected_normal[index]
                height = float(selected_height[index])
                if normal[2] <= 0.0:
                    raise RuntimeError("batch ray chose a non-upward box boundary")
            else:
                geom = self._floor
                normal = np.asarray((0.0, 0.0, 1.0))
                height = float(geom.position[2])
            hits.append(CourseGroundHit(
                schema=GROUND_SCHEMA, height_m=height,
                normal_world=tuple(float(value) for value in normal),
                pitch_rad=math.atan2(float(normal[0]), float(normal[2])),
                roll_rad=math.atan2(-float(normal[1]), float(normal[2])),
                geom_id=geom.geom_id, geom_name=geom.name,
            ))
        return tuple(hits)
