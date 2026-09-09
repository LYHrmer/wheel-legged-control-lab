import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from itertools import combinations

import mujoco
import numpy as np
import pytest

from wheel_legged_control.d1.controllers import D1Command, D1VMCController
from wheel_legged_control.d1.locomotion_terrain import (
    LOCOMOTION_GRID_SPACING_M,
    LOCOMOTION_MAP_HALF_SIZE_M,
    D1LocomotionTerrainConfig,
    add_locomotion_terrain,
    locomotion_ground_reference,
    locomotion_terrain_configs,
)
from wheel_legged_control.d1.model import D1Plant, build_d1_model
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource
from wheel_legged_control.d1.training_terrain import TrainingTerrainConfig


def _world(config):
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><geom name="floor" type="plane" size="0 0 .1" '
        'friction=".73 .005 .0001"/></worldbody></mujoco>'
    )
    add_locomotion_terrain(spec, config)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _ray(model, data, x, y):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    normal = np.zeros(3)
    distance = mujoco.mj_rayHfield(
        model, data, gid, np.asarray((x, y, 2.0)), np.asarray((0.0, 0.0, -1.0)), normal
    )
    assert distance >= 0.0
    assert normal[2] > 0.0
    return 2.0 - distance, normal


def _normal(reference):
    vector = np.asarray((np.tan(reference.pitch_rad), -np.tan(reference.roll_rad), 1.0))
    return vector / np.linalg.norm(vector)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"layout": "unknown"},
        {"schema": "v0"},
        {"slope_deg": True},
        {"layout": "straight", "slope_deg": 3.001},
        {"layout": "straight", "cross_slope_deg": -2.001},
        {"layout": "straight", "ripple_amplitude_m": 0.0101},
        {"layout": "straight", "ripple_amplitude_m": -0.001},
        {"layout": "straight", "step_height_m": -0.001},
        {"layout": "straight", "step_height_m": 0.0101},
        {"wavelength_x_m": 0.399},
        {"wavelength_y_m": 0.399},
        {"phase_x_rad": np.nan},
        {"phase_y_rad": np.inf},
        {"ripple_amplitude_m": 0.005},
        {"slope_deg": 1.0},
        {"cross_slope_deg": 1.0},
        {"step_height_m": 0.005},
        {"phase_x_rad": 0.1},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        D1LocomotionTerrainConfig(**kwargs)


def test_json_is_complete_canonical_and_reproduces_collision_bytes():
    config = locomotion_terrain_configs("train")[2]
    text = config.to_json()
    restored = D1LocomotionTerrainConfig.from_json(text)
    assert restored == config
    assert restored.to_json() == text
    assert "schema" in json.loads(text)
    with pytest.raises(FrozenInstanceError):
        restored.slope_deg = 0.0
    first, _ = _world(config)
    second, _ = _world(restored)
    for name in ("hfield_data", "hfield_size", "geom_pos"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    with pytest.raises(TypeError):
        D1LocomotionTerrainConfig.from_json("[]")
    with pytest.raises(TypeError):
        D1LocomotionTerrainConfig.from_json('{"unexpected": 0}')


def test_splits_differ_in_physical_layout_and_each_held_parameter():
    sets = [locomotion_terrain_configs(split) for split in ("train", "development", "holdout")]
    for first, second in combinations(sets, 2):
        assert set(first).isdisjoint(second)
        for field in (
            "layout",
            "wavelength_x_m",
            "wavelength_y_m",
            "phase_x_rad",
            "phase_y_rad",
            "slope_deg",
            "cross_slope_deg",
        ):
            assert {getattr(c, field) for c in first}.isdisjoint(getattr(c, field) for c in second)
    fingerprints = []
    for configs in sets:
        for config in configs:
            model, _ = _world(config)
            fingerprints.append(
                hashlib.sha256(
                    model.hfield_data.tobytes()
                    + model.hfield_size.tobytes()
                    + model.geom_pos.tobytes()
                ).hexdigest()
            )
    assert len(set(fingerprints)) == 8
    with pytest.raises(ValueError):
        locomotion_terrain_configs("test")


ALL_ROADS = tuple(
    config
    for split in ("train", "development", "holdout")
    for config in locomotion_terrain_configs(split)
)


@pytest.mark.parametrize("config", ALL_ROADS, ids=lambda c: f"{c.layout}-{c.slope_deg}")
def test_compiled_road_matches_independent_rays_in_both_triangles(config):
    model, data = _world(config)
    max_error = 0.0
    saw_mixed_slope = False
    # Four interior offsets cover both triangles regardless of which diagonal
    # an implementation mistakenly assumes. Nonzero world x/y exercise cross slope.
    for col, row in ((79, 53), (93, 69), (126, 43), (150, 75), (169, 55), (203, 70)):
        for fx, fy in ((0.2, 0.4), (0.4, 0.2), (0.6, 0.8), (0.8, 0.6)):
            x, y = -6.0 + (col + fx) * 0.05, -3.0 + (row + fy) * 0.05
            reference = locomotion_ground_reference(model, x, y)
            height, normal = _ray(model, data, x, y)
            max_error = max(max_error, abs(height - reference.height_m))
            np.testing.assert_allclose(_normal(reference), normal, rtol=0, atol=2e-12)
            saw_mixed_slope |= abs(normal[0]) > 1e-5 and abs(normal[1]) > 1e-5
    assert max_error < 2e-12
    assert saw_mixed_slope
    # Known placement prior covers the robot footprint and the complete start strip.
    for x in (-6.0, -4.35, -3.8, -3.25, -2.4):
        for y in (-3.0, -0.6, 0.0, 0.6, 3.0):
            ref = locomotion_ground_reference(model, x, y)
            assert abs(ref.height_m) < 1e-7
            assert ref.pitch_rad == 0.0
            assert ref.roll_rad == 0.0


def test_non_coplanar_cell_uses_triangles_not_bilinear_or_generating_formula():
    model, data = _world(D1LocomotionTerrainConfig())
    # Deliberately replace four compiled samples, unknown to the generator.
    # The raised top-right vertex distinguishes the two possible diagonals.
    heights = model.hfield_data.reshape(121, 241)
    row, col = 68, 146
    heights[row : row + 2, col : col + 2] = ((0.0, 0.0), (0.0, 0.02))
    for fx, fy in ((0.2, 0.4), (0.4, 0.2), (0.6, 0.8), (0.8, 0.6)):
        x, y = -6.0 + (col + fx) * 0.05, -3.0 + (row + fy) * 0.05
        ref = locomotion_ground_reference(model, x, y)
        height, normal = _ray(model, data, x, y)
        assert ref.height_m == pytest.approx(height, abs=2e-12)
        np.testing.assert_allclose(_normal(ref), normal, rtol=0, atol=2e-12)
        actual_peak = float(heights[row + 1, col + 1])
        assert height == pytest.approx(min(fx, fy) * actual_peak, abs=2e-12)
        assert abs(height - fx * fy * actual_peak) > 1e-3


def test_ripple_amplitude_survives_compile_normalization():
    config = D1LocomotionTerrainConfig(layout="straight", ripple_amplitude_m=0.005)
    low, low_data = _world(config)
    high, high_data = _world(replace(config, ripple_amplitude_m=0.01))
    ray_heights = []
    for x, y in ((0.811, 0.133), (1.017, -0.237), (1.413, 0.341), (1.799, -0.427)):
        h_low, _ = _ray(low, low_data, x, y)
        h_high, _ = _ray(high, high_data, x, y)
        assert h_high == pytest.approx(2.0 * h_low, abs=1e-8)
        ray_heights.append(abs(h_low))
    assert max(ray_heights) > 0.002
    assert max(ray_heights) <= 0.005


def test_small_step_is_explicitly_a_one_cell_bevel():
    model, data = _world(D1LocomotionTerrainConfig(layout="straight", step_height_m=0.005))
    for x, expected in ((2.975, 0.0), (3.025, 0.0025), (3.075, 0.005), (4.175, 0.0025)):
        height, _ = _ray(model, data, x, 0.137)
        assert height == pytest.approx(expected, abs=1e-9)
    assert LOCOMOTION_GRID_SPACING_M == 0.05


def test_edges_creases_and_spatial_limits():
    model, data = _world(ALL_ROADS[0])
    assert tuple(model.hfield_size[0, :2]) == LOCOMOTION_MAP_HALF_SIZE_M
    # A normal on a crease is not unique. Only height is compared at exact seams.
    for x, y in (
        (-6.0, -3.0),
        (6.0, 3.0),
        (6.0, -0.7),
        (0.3, -3.0),
        (-1.05, 0.35),
        (-1.025, 0.375),
    ):
        assert locomotion_ground_reference(model, x, y).height_m == pytest.approx(
            _ray(model, data, x, y)[0], abs=2e-12
        )
    for x, y in (
        (-6.0001, 0.0),
        (6.0001, 0.0),
        (0.0, -3.0001),
        (0.0, 3.0001),
        (np.nan, 0.0),
        (0.0, np.inf),
        (True, 0.0),
    ):
        with pytest.raises(ValueError):
            locomotion_ground_reference(model, x, y)


def test_fixed_geometry_survives_physics_and_data_reset():
    model, data = _world(ALL_ROADS[-1])
    before = (model.hfield_data.copy(), model.hfield_size.copy(), model.geom_pos.copy())
    assert model.geom_friction[0, 0] == pytest.approx(0.73)
    assert model.geom_bodyid[0] == 0
    for _ in range(50):
        mujoco.mj_step(model, data)
    mujoco.mj_resetData(model, data)
    for old, current in zip(
        before, (model.hfield_data, model.hfield_size, model.geom_pos), strict=True
    ):
        np.testing.assert_array_equal(old, current)
    model.geom_pos[0, 0] = 0.1
    with pytest.raises(ValueError, match="world frame"):
        locomotion_ground_reference(model, 0.0, 0.0)


def test_construction_requires_correct_config_and_one_static_floor():
    spec = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
    with pytest.raises(ValueError, match="floor"):
        add_locomotion_terrain(spec, D1LocomotionTerrainConfig())
    with pytest.raises(TypeError):
        add_locomotion_terrain(spec, {})
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.1))
    add_locomotion_terrain(spec, D1LocomotionTerrainConfig())
    with pytest.raises(ValueError, match="already"):
        add_locomotion_terrain(spec, D1LocomotionTerrainConfig())
    bare = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    with pytest.raises(ValueError, match="not built"):
        locomotion_ground_reference(bare, 0.0, 0.0)


def test_real_d1_spawn_contact_and_fixed_geometry_short_loop():
    """Geometry integration only; 0.5 s at spawn is not terrain traversal."""
    config = locomotion_terrain_configs("holdout")[1]
    plant = D1Plant(locomotion_terrain=config, sampling_mode="synchronized")
    plant.reset(base_position=np.asarray((-3.8, 0.0, 0.455)))
    before = plant.model.hfield_data.copy()
    source = D1MujocoTruthStateSource(plant)
    state = source.reset()
    controller = D1VMCController(plant)
    for _ in range(50):
        plant.step(controller.compute(D1Command(), state))
        state = source.read()
    assert plant.data.time == pytest.approx(0.5)
    assert not plant.has_fallen()
    assert plant.undesired_ground_contacts == 0
    # Wheel contact count can change during the transient; no four-wheel
    # steady-state claim is made from this short integration test.
    assert plant.wheel_ground_contacts > 0
    assert plant.base_position[2] > 0.4
    assert plant.floor_geom_id in plant.terrain_geom_ids
    np.testing.assert_array_equal(plant.model.hfield_data, before)
    plant.set_domain()
    plant.reset()
    np.testing.assert_array_equal(plant.model.hfield_data, before)
    assert plant.locomotion_terrain == config
    with pytest.raises(ValueError, match="requires"):
        build_d1_model(locomotion_terrain=config, arena="course")
    with pytest.raises(ValueError, match="requires"):
        build_d1_model(locomotion_terrain=config, training_terrain=TrainingTerrainConfig())
