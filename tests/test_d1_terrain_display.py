# tests/test_d1_terrain_display.py
import mujoco
import numpy as np
import pytest

from scripts import d1_terrain_display as dtd

ASYM = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.0]])  # nrow=2 (y), ncol=3 (x)
HALF, POS, ZS = (1.5, 0.5), (1.0, -2.0, 0.3), 2.0
BUMPY = np.array([[0.0, 0.2, 0.4], [0.1, 0.9, 0.3], [0.5, 0.0, 0.7]])
FLAT_KW = {"spacing": 0.5, "radius": 1.0, "offset": 0.002}

HFIELD_XML = """
<mujoco>
  <asset><hfield name="t" nrow="3" ncol="4" size="1 0.6 0.5 0.1"/></asset>
  <worldbody>
    <geom name="floor" type="hfield" hfield="t" pos="0 0 -0.1"/>
    <body pos="0 0 1"><freejoint/><geom name="ball" type="sphere" size="0.1"/></body>
  </worldbody>
</mujoco>
"""
PLANE_XML = """<mujoco><worldbody>
  <geom name="floor" type="plane" size="5 5 0.1"/></worldbody></mujoco>"""
ROTATED_XML = HFIELD_XML.replace('pos="0 0 -0.1"', 'pos="0 0 -0.1" euler="0 0 15"')
CHILD_XML = """
<mujoco>
  <asset><hfield name="t" nrow="3" ncol="4" size="1 0.6 0.5 0.1"/></asset>
  <worldbody><body name="pad" pos="0.3 0 0">
    <geom name="floor" type="hfield" hfield="t" pos="0 0 -0.1"/></body></worldbody>
</mujoco>
"""


def _scene(xml, maxgeom=200):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=maxgeom)
    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    return model, scene


def test_asymmetric_grid_orientation_zscale_and_translation():
    # (1.0, -2.0): col 1, fx=0, fy=0.5 -> upper triangle;  (1.75, -2.5): fx=0.5, fy=0.
    z = dtd.surface_z(ASYM, HALF, ZS, POS, np.array([[1.0, -2.0], [1.75, -2.5]]))
    assert np.allclose(z, [0.3 + 2.0 * 0.625, 0.3 + 2.0 * 0.75])
    # translation only shifts, never rescales
    shifted = dtd.surface_z(ASYM, HALF, ZS, (0.0, 0.0, 0.0), np.array([0.0, 0.0]))
    assert np.isclose(shifted, ZS * 0.625)


def test_triangle_diagonal_beats_bilinear():
    h = np.array([[0.0, 0.0], [0.0, 1.0]])
    below = dtd.surface_z(h, (0.5, 0.5), 1.0, (0, 0, 0), np.array([0.25, -0.25]))
    above = dtd.surface_z(h, (0.5, 0.5), 1.0, (0, 0, 0), np.array([-0.25, 0.25]))
    assert np.isclose(below, 0.25) and np.isclose(above, 0.25)  # two distinct planes
    assert not np.isclose(below, 0.1875)  # the bilinear answer for both points


def test_segments_split_at_edges_and_diagonals():
    segs = dtd.grid_segments(BUMPY, (1.0, 1.0), 1.0, (0, 0, 0), (0.0, 0.0), **FLAT_KW)
    assert segs.shape == (28, 2, 3)  # 2 * (2 + 4 + 2 + 4 + 2)
    ends = segs.reshape(-1, 3)
    exp_ends = dtd.surface_z(BUMPY, (1.0, 1.0), 1.0, (0, 0, 0), ends[:, :2]) + 0.002
    assert np.allclose(ends[:, 2], exp_ends, atol=1e-12)


def test_segment_midpoints_follow_the_triangulated_surface():
    segs = dtd.grid_segments(BUMPY, (1.0, 1.0), 1.0, (0, 0, 0), (0.0, 0.0), **FLAT_KW)
    mid = segs.mean(axis=1)
    exp = dtd.surface_z(BUMPY, (1.0, 1.0), 1.0, (0, 0, 0), mid[:, :2]) + 0.002
    assert np.allclose(mid[:, 2], exp, atol=1e-12)
    # a chord across the diagonal would have missed the ridge by a wide margin
    chord = 0.5 * (0.05 + 0.55)
    assert not np.isclose(chord, 0.45)


def test_crop_to_field_rectangle_and_local_square():
    segs = dtd.grid_segments(ASYM, HALF, ZS, POS, (1.0, -2.0), radius=1.5)
    pts = segs.reshape(-1, 3)
    assert pts[:, 0].min() == pytest.approx(-0.5) and pts[:, 0].max() == pytest.approx(2.5)
    assert pts[:, 1].min() == pytest.approx(-2.5) and pts[:, 1].max() == pytest.approx(-1.5)
    assert dtd.grid_segments(ASYM, HALF, ZS, POS, (100.0, 0.0)).shape == (0, 2, 3)


def test_budget_keeps_nearest_segments_deterministically():
    args = (BUMPY, (1.0, 1.0), 1.0, (0, 0, 0), (0.0, 0.0))
    full = dtd.grid_segments(*args, **FLAT_KW)
    small = dtd.grid_segments(*args, max_segments=6, **FLAT_KW)
    assert small.shape[0] == 6
    d_all = np.sort(np.linalg.norm(full.mean(axis=1)[:, :2], axis=1))
    assert np.linalg.norm(small.mean(axis=1)[:, :2], axis=1).max() <= d_all[5] + 1e-12
    assert np.array_equal(small, dtd.grid_segments(*args, max_segments=6, **FLAT_KW))


def test_d1_sized_field_stays_within_budget():
    heights = np.outer(np.linspace(0, 1, 121), np.linspace(0, 1, 241))
    segs = dtd.grid_segments(heights, (6.0, 3.0), 0.4, (0.0, 0.0, -0.1), (0.0, 0.0))
    assert segs.shape == (840, 2, 3) and segs.shape[0] <= dtd.MAX_SEGMENTS


@pytest.mark.parametrize(
    "kwargs, err",
    [
        ({"max_segments": True}, TypeError),
        ({"max_segments": 4.0}, TypeError),
        ({"max_segments": 0}, ValueError),
        ({"spacing": -0.5}, ValueError),
        ({"radius": 0.0}, ValueError),
        ({"offset": np.nan}, ValueError),
        ({"spacing": True}, TypeError),
        ({"offset": False}, TypeError),
    ],
)
def test_argument_validation(kwargs, err):
    with pytest.raises(err):
        dtd.grid_segments(BUMPY, (1.0, 1.0), 1.0, (0, 0, 0), (0.0, 0.0), **kwargs)


def test_append_preserves_native_geoms_and_model_state():
    model, scene = _scene(HFIELD_XML)
    model.hfield_data[:] = np.linspace(0.1, 0.9, model.hfield_data.size)
    hfield0, pos0 = model.hfield_data.copy(), model.geom_pos.copy()
    native = scene.ngeom
    native_types = [scene.geoms[i].type for i in range(native)]

    count = dtd.append_terrain_grid(model, scene, (0.0, 0.0))

    assert 0 < count <= dtd.MAX_SEGMENTS and scene.ngeom == native + count
    assert [scene.geoms[i].type for i in range(native)] == native_types
    assert np.array_equal(model.hfield_data, hfield0)
    assert np.array_equal(model.geom_pos, pos0)
    geom = scene.geoms[native]
    assert geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE
    assert np.allclose(geom.rgba, dtd.LINE_RGBA, atol=1e-6)
    assert np.isclose(geom.size[0], dtd.LINE_WIDTH)


def test_limited_scene_capacity_skips_extras_without_throwing():
    _, big = _scene(HFIELD_XML)
    model, scene = _scene(HFIELD_XML, maxgeom=big.ngeom + 4)
    count = dtd.append_terrain_grid(model, scene, (0.0, 0.0))
    assert count == 4 and scene.ngeom == scene.maxgeom


def test_plane_or_missing_floor_is_a_noop():
    model, scene = _scene(PLANE_XML)
    before = scene.ngeom
    assert dtd.append_terrain_grid(model, scene, (0.0, 0.0)) == 0
    assert scene.ngeom == before
    other = mujoco.MjModel.from_xml_string(PLANE_XML.replace('name="floor"', 'name="ground"'))
    assert dtd.append_terrain_grid(other, scene, (0.0, 0.0)) == 0
    assert scene.ngeom == before


@pytest.mark.parametrize("xml", [ROTATED_XML, CHILD_XML])
def test_rotated_or_non_world_hfield_is_rejected(xml):
    model, scene = _scene(xml)
    with pytest.raises(ValueError):
        dtd.append_terrain_grid(model, scene, (0.0, 0.0))


@pytest.mark.parametrize("points", [[[1.01, 0.0]], [[np.nan, 0.0]], [0.0, 0.0, 0.0]])
def test_surface_query_does_not_clamp_outside_or_invalid_points(points):
    with pytest.raises(ValueError):
        dtd.surface_z(BUMPY, (1.0, 1.0), 1.0, (0.0, 0.0, 0.0), points)


def test_candidate_line_budget_is_checked_before_allocation(monkeypatch):
    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("candidate budget must be checked before np.arange")

    monkeypatch.setattr(dtd.np, "arange", unexpected_allocation)
    with pytest.raises(ValueError, match="budget"):
        dtd.grid_segments(BUMPY, (1.0, 1.0), 1.0, (0.0, 0.0, 0.0), (0.0, 0.0), spacing=1e-9)


def test_pure_geometry_cannot_exceed_the_hard_scene_budget():
    segments = dtd.grid_segments(
        np.zeros((41, 41)),
        (1.0, 1.0),
        1.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0),
        spacing=0.05,
        radius=1.0,
        max_segments=1501,
    )
    assert len(segments) == 1500
