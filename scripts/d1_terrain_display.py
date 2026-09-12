# scripts/d1_terrain_display.py
"""Terrain wireframe annotation for the D1 MuJoCo height-field floor.

The helpers here draw a world-aligned 0.5 m grid that hugs the real terrain so
an operator can judge relief in the viewer.

VISUAL ANNOTATION ONLY.  Everything produced by this module is appended to an
``mjvScene`` (the render buffer) as ``mjGEOM_CAPSULE`` decoration geoms.  They
are *never* collision bodies and have no physical meaning: no ``mjModel`` or
``mjData`` field is written, no physics is stepped, and removing this module
cannot change simulation results.  Native scene geoms produced by
``mjv_updateScene`` are never overwritten -- annotations are only written at
indices ``[scene.ngeom, scene.maxgeom)``.

Surface evaluation reproduces MuJoCo's height-field triangulation exactly: each
cell is split bottom-left -> top-right, and the two triangles are evaluated with
the formulas below.  Bilinear interpolation is deliberately NOT used; it would
sag below real peaks and invent geometry that the collision engine does not see.
"""

from __future__ import annotations

import numpy as np

try:  # the pure geometry half of this module does not need MuJoCo
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None

FLOOR_GEOM_NAME = "floor"
LINE_WIDTH = 0.0015
LINE_RGBA = (0.35, 0.65, 0.68, 0.85)
MAX_SEGMENTS = 1500
_CANDIDATE_CAP = 40000
_EPS = 1e-9


# --------------------------------------------------------------------- checks
def _finite_vec(name, value, size):
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != size or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be {size} finite numbers, got {value!r}")
    return arr


def _positive(name, value):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not bool")
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return v


def _nonneg(name, value):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not bool")
    v = float(value)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return v


def _count(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int (bool rejected), got {type(value).__name__}")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return int(value)


def _heights(heights):
    grid = np.asarray(heights, dtype=float)
    if grid.ndim != 2 or grid.shape[0] < 2 or grid.shape[1] < 2:
        raise ValueError(f"heights must be (nrow>=2, ncol>=2), got shape {grid.shape}")
    if not np.all(np.isfinite(grid)):
        raise ValueError("heights must be finite")
    return grid


def _half(half_size):
    hs = _finite_vec("half_size", half_size, 2)
    if np.any(hs <= 0.0):
        raise ValueError(f"half_size must be positive, got {half_size!r}")
    return float(hs[0]), float(hs[1])


# ------------------------------------------------------------------- geometry
def _eval(grid, hx, hy, z_scale, geom_pos, pts):
    """Unvalidated surface evaluation; ``pts`` is (..., 2) in world XY."""
    nrow, ncol = grid.shape
    dx, dy = 2.0 * hx / (ncol - 1), 2.0 * hy / (nrow - 1)
    u = (pts[..., 0] - geom_pos[0] + hx) / dx
    v = (pts[..., 1] - geom_pos[1] + hy) / dy
    # Clamp to the final valid cell: MuJoCo does not extrapolate past the field.
    col = np.clip(np.floor(u).astype(int), 0, ncol - 2)
    row = np.clip(np.floor(v).astype(int), 0, nrow - 2)
    fx = np.clip(u - col, 0.0, 1.0)
    fy = np.clip(v - row, 0.0, 1.0)
    h00 = grid[row, col]
    h10 = grid[row, col + 1]
    h01 = grid[row + 1, col]
    h11 = grid[row + 1, col + 1]
    lower = h00 + fx * (h10 - h00) + fy * (h11 - h10)  # fx >= fy triangle
    upper = h00 + fx * (h11 - h01) + fy * (h01 - h00)  # fx <  fy triangle
    return geom_pos[2] + z_scale * np.where(fx >= fy, lower, upper)


def surface_z(heights, half_size, z_scale, geom_pos, points):
    """World Z of the triangulated height field under ``points`` (..., 2)."""
    grid = _heights(heights)
    hx, hy = _half(half_size)
    gpos = _finite_vec("geom_pos", geom_pos, 3)
    points = np.asarray(points, dtype=float)
    if points.ndim == 0 or points.shape[-1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("points must be finite world XY pairs")
    if np.any(np.abs(points - gpos[:2]) > (hx, hy)):
        raise ValueError("surface query lies outside the finite height field")
    return _eval(grid, hx, hy, _positive("z_scale", z_scale), gpos, points)


def _frac(local, step, n):
    """Fractional in-cell coordinate for a local (field-relative) position."""
    t = local / step
    i = min(max(int(np.floor(t)), 0), n - 2)
    return float(np.clip(t - i, 0.0, 1.0))


def _breaks(lo, hi, origin, step, frac):
    """Split coordinates in [lo, hi]: every mesh edge plus every diagonal."""
    k0 = int(np.floor((lo - origin) / step))
    k1 = int(np.ceil((hi - origin) / step))
    edges = origin + np.arange(k0, k1 + 1) * step
    inner = np.concatenate([edges, edges + frac * step])  # diagonal: fx == fy
    inner = inner[(inner > lo + _EPS) & (inner < hi - _EPS)]
    pts = np.concatenate([[lo], np.sort(inner), [hi]])
    return pts[np.concatenate([[True], np.diff(pts) > _EPS])]


def _line_coords(lo, hi, spacing):
    k0 = int(np.ceil(lo / spacing - _EPS))
    k1 = int(np.floor(hi / spacing + _EPS))
    return np.clip(spacing * np.arange(k0, k1 + 1), lo, hi)


def grid_segments(
    heights,
    half_size,
    z_scale,
    geom_pos,
    center_xy,
    spacing=0.5,
    radius=1.5,
    offset=0.002,
    max_segments=MAX_SEGMENTS,
):
    """World-aligned terrain wireframe as an ``(N, 2, 3)`` array of endpoints.

    The grid consists of world-aligned lines at multiples of ``spacing``,
    cropped to the ``+/-radius`` square around ``center_xy`` AND to the actual
    field rectangle.  Every line is split at all mesh row/column boundaries and
    at the triangle diagonal crossing, so each segment lies inside a single
    triangle and the polyline follows the collision surface exactly, lifted by
    ``offset``.  No chord ever cuts a real peak and no obstacle is invented.

    When more than ``max_segments`` candidates exist, the segments nearest to
    ``center_xy`` (by planar midpoint distance, stable order) are kept; retained
    endpoints are still exactly on the true triangle surface.
    """
    grid = _heights(heights)
    hx, hy = _half(half_size)
    z_scale = _positive("z_scale", z_scale)
    gpos = _finite_vec("geom_pos", geom_pos, 3)
    ctr = _finite_vec("center_xy", center_xy, 2)
    spacing = _positive("spacing", spacing)
    radius = _positive("radius", radius)
    offset = _nonneg("offset", offset)
    max_segments = min(_count("max_segments", max_segments), MAX_SEGMENTS)

    nrow, ncol = grid.shape
    dx, dy = 2.0 * hx / (ncol - 1), 2.0 * hy / (nrow - 1)
    x0, x1 = max(ctr[0] - radius, gpos[0] - hx), min(ctr[0] + radius, gpos[0] + hx)
    y0, y1 = max(ctr[1] - radius, gpos[1] - hy), min(ctr[1] + radius, gpos[1] + hy)
    empty = np.zeros((0, 2, 3))
    if x1 - x0 <= _EPS or y1 - y0 <= _EPS:
        return empty

    # Check conservative counts BEFORE np.arange can allocate huge line arrays.
    estimate = ((x1 - x0) / spacing + 2.0) * (2.0 * (y1 - y0) / dy + 4.0) + (
        (y1 - y0) / spacing + 2.0
    ) * (2.0 * (x1 - x0) / dx + 4.0)
    if estimate > _CANDIDATE_CAP:
        raise ValueError(
            f"candidate budget exceeded (~{estimate:.0f} > {_CANDIDATE_CAP}); "
            "reduce radius or increase spacing"
        )
    x_lines, y_lines = _line_coords(x0, x1, spacing), _line_coords(y0, y1, spacing)

    polylines = []
    for yv in y_lines:
        fy = _frac(yv - gpos[1] + hy, dy, nrow)
        xs = _breaks(x0, x1, gpos[0] - hx, dx, fy)
        polylines.append(np.column_stack([xs, np.full(xs.size, yv)]))
    for xv in x_lines:
        fx = _frac(xv - gpos[0] + hx, dx, ncol)
        ys = _breaks(y0, y1, gpos[1] - hy, dy, fx)
        polylines.append(np.column_stack([np.full(ys.size, xv), ys]))

    chunks = []
    for pts in polylines:
        if pts.shape[0] < 2:
            continue
        z = _eval(grid, hx, hy, z_scale, gpos, pts) + offset
        p3 = np.column_stack([pts, z])
        chunks.append(np.stack([p3[:-1], p3[1:]], axis=1))
    if not chunks:
        return empty

    segs = np.concatenate(chunks, axis=0)
    segs = segs[np.linalg.norm(segs[:, 1, :2] - segs[:, 0, :2], axis=1) > _EPS]
    if segs.shape[0] > max_segments:
        dist = np.linalg.norm(segs.mean(axis=1)[:, :2] - ctr, axis=1)
        keep = np.sort(np.argsort(dist, kind="stable")[:max_segments])
        segs = segs[keep]
    return segs


# -------------------------------------------------------------- viewer adapter
def _require_static_axis_aligned(model, gid):
    quat = np.asarray(model.geom_quat[gid], dtype=float)
    if not np.allclose(quat, (1.0, 0.0, 0.0, 0.0), rtol=0, atol=1e-9):
        raise ValueError(
            "rotated height-field geom: terrain annotation assumes an "
            "identity orientation and would render incorrect coordinates"
        )
    if int(model.geom_bodyid[gid]) != 0:
        raise ValueError(
            "height-field geom is not attached to the world body; moving "
            "fields are rejected instead of rendering incorrect coordinates"
        )


def append_terrain_grid(
    model,
    scene,
    center_xy,
    *,
    geom_name=FLOOR_GEOM_NAME,
    spacing=0.5,
    radius=1.5,
    offset=0.002,
    max_segments=MAX_SEGMENTS,
):
    """Append the terrain annotation to ``scene`` and return the geom count.

    Call AFTER ``mjv_updateScene`` and BEFORE ``mjr_render``.  Read-only with
    respect to ``model``/``data``; the appended capsules are decoration only and
    are never collision bodies.  Returns 0 when there is no ``floor`` geom, when
    the floor is not a height field, or when the scene has no spare capacity;
    the count is always bounded by ``max_segments`` (<= 1500) and by the
    remaining ``scene.maxgeom - scene.ngeom`` capacity.  Insufficient capacity
    silently drops the extra annotations rather than raising or overwriting
    native scene geoms.
    """
    if mujoco is None:  # pragma: no cover
        raise RuntimeError("mujoco is required for append_terrain_grid")
    max_segments = min(_count("max_segments", max_segments), MAX_SEGMENTS)

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0 or int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_HFIELD):
        return 0
    _require_static_axis_aligned(model, gid)

    dataid = int(model.geom_dataid[gid])
    if dataid < 0:
        raise ValueError(f"geom '{geom_name}' is a height field without hfield data")
    nrow, ncol = int(model.hfield_nrow[dataid]), int(model.hfield_ncol[dataid])
    adr = int(model.hfield_adr[dataid])
    heights = np.array(model.hfield_data[adr : adr + nrow * ncol], dtype=float).reshape(nrow, ncol)
    size = np.asarray(model.hfield_size[dataid], dtype=float)

    capacity = int(scene.maxgeom) - int(scene.ngeom)
    if capacity <= 0:
        return 0
    segs = grid_segments(
        heights,
        size[:2],
        float(size[2]),
        np.asarray(model.geom_pos[gid], dtype=float),
        center_xy,
        spacing=spacing,
        radius=radius,
        offset=offset,
        max_segments=min(max_segments, capacity),
    )

    rgba = np.array(LINE_RGBA, dtype=np.float32)
    zeros3, identity = np.zeros(3), np.eye(3).ravel()
    appended = 0
    for start, end in segs:
        if scene.ngeom >= scene.maxgeom:  # never clobber native scene geoms
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, int(mujoco.mjtGeom.mjGEOM_CAPSULE), zeros3, zeros3, identity, rgba
        )
        mujoco.mjv_connector(geom, int(mujoco.mjtGeom.mjGEOM_CAPSULE), LINE_WIDTH, start, end)
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        geom.emission = 0.25
        scene.ngeom += 1
        appended += 1
    return appended
