from dataclasses import FrozenInstanceError

import mujoco
import numpy as np
import pytest

from wheel_legged_control.d1.controllers import D1Command, D1VMCController
from wheel_legged_control.d1.model import D1Plant, build_d1_model
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource
from wheel_legged_control.d1.training_terrain import (
    TRAINING_TERRAIN_X_HALF_SIZE_M,
    TRAINING_TERRAIN_Y_HALF_SIZE_M,
    TrainingTerrainConfig,
    training_ground_reference,
)


@pytest.fixture(scope="module")
def training_plant() -> D1Plant:
    return D1Plant(training_terrain=TrainingTerrainConfig())


def _ray_ground(plant: D1Plant, x: float, y: float) -> tuple[float, np.ndarray]:
    geom_id = np.zeros(1, dtype=np.int32)
    normal = np.zeros(3, dtype=np.float64)
    distance = mujoco.mj_ray(
        plant.model,
        plant.data,
        np.asarray((x, y, 2.0)),
        np.asarray((0.0, 0.0, -1.0)),
        None,
        True,
        -1,
        geom_id,
        normal,
    )
    assert distance >= 0.0
    assert geom_id[0] == plant.floor_geom_id
    return 2.0 - distance, normal


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "steps"},
        {"kind": None},
        {"amplitude_m": -0.001},
        {"kind": "bumps", "amplitude_m": 0.031},
        {"kind": "ramp", "slope_deg": 6.01},
        {"kind": "ramp", "slope_deg": -6.01},
        {"wavelength_m": 0.29},
        {"amplitude_m": 0.001},
        {"slope_deg": 1.0},
        {"kind": "bumps", "slope_deg": 1.0},
        {"kind": "ramp", "amplitude_m": 0.001},
        {"amplitude_m": True},
        {"slope_deg": np.bool_(True)},
        {"wavelength_m": "0.8"},
        {"phase_rad": None},
        {"phase_rad": float("inf")},
        {"slope_deg": float("nan")},
        {"amplitude_m": float("-inf")},
    ],
)
def test_invalid_terrain_configuration_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        TrainingTerrainConfig(**kwargs)


def test_config_is_frozen_and_accepts_numpy_real_scalars() -> None:
    config = TrainingTerrainConfig(kind="bumps", amplitude_m=np.float32(0.01))
    assert isinstance(config.amplitude_m, float)
    with pytest.raises(FrozenInstanceError):
        config.kind = "ramp"


@pytest.mark.parametrize(
    "config",
    [
        TrainingTerrainConfig(),
        TrainingTerrainConfig(kind="ramp", slope_deg=6.0),
        TrainingTerrainConfig(kind="ramp", slope_deg=-6.0),
        TrainingTerrainConfig(kind="bumps", amplitude_m=0.03, wavelength_m=0.3, phase_rad=0.7),
    ],
)
def test_ground_oracle_matches_mujoco_collision_rays(
    training_plant: D1Plant, config: TrainingTerrainConfig
) -> None:
    training_plant.set_training_terrain(config)
    training_plant.reset()
    for x, y in ((-5.987, -2.8), (-1.234, 0.61), (0.003, 1.17), (5.991, 2.7)):
        reference = training_plant.training_ground_reference(x, y)
        height, normal = _ray_ground(training_plant, x, y)
        assert reference.height_m == pytest.approx(height, abs=2e-12)
        np.testing.assert_allclose(
            normal,
            (np.sin(reference.pitch_rad), 0.0, np.cos(reference.pitch_rad)),
            atol=2e-12,
        )
        assert reference.roll_rad == 0.0


@pytest.mark.parametrize("slope_deg", (-6.0, -2.0, 2.0, 6.0))
def test_ramp_height_and_pitch_use_world_convention(
    training_plant: D1Plant, slope_deg: float
) -> None:
    training_plant.set_training_terrain(TrainingTerrainConfig(kind="ramp", slope_deg=slope_deg))
    for x in (-6.0, -1.333, 0.0, 2.113, 6.0):
        reference = training_plant.training_ground_reference(x, 0.0)
        assert reference.height_m == pytest.approx(x * np.tan(np.deg2rad(slope_deg)), abs=9e-8)
        assert reference.pitch_rad == pytest.approx(-np.deg2rad(slope_deg), abs=1e-5)


def test_bump_amplitude_is_not_renormalized_and_phase_shifts_profile(
    training_plant: D1Plant,
) -> None:
    for amplitude in (0.005, 0.01):
        training_plant.set_training_terrain(
            TrainingTerrainConfig(kind="bumps", amplitude_m=amplitude, wavelength_m=0.8)
        )
        assert training_plant.training_ground_reference(0.2, 0.0).height_m == pytest.approx(
            amplitude, abs=9e-8
        )
        assert training_plant.training_ground_reference(-0.2, 0.0).height_m == pytest.approx(
            -amplitude, abs=9e-8
        )
    training_plant.set_training_terrain(
        TrainingTerrainConfig(kind="bumps", amplitude_m=0.01, wavelength_m=0.8, phase_rad=np.pi / 2)
    )
    assert training_plant.training_ground_reference(0.0, 0.0).height_m == pytest.approx(
        0.01, abs=9e-8
    )


def test_oracle_uses_collision_samples_instead_of_analytic_sine(training_plant: D1Plant) -> None:
    config = TrainingTerrainConfig(kind="bumps", amplitude_m=0.03, wavelength_m=0.3)
    training_plant.set_training_terrain(config)
    left = training_plant.training_ground_reference(0.07, 0.0).height_m
    right = training_plant.training_ground_reference(0.08, 0.0).height_m
    middle = training_plant.training_ground_reference(0.075, 0.0).height_m
    assert middle == pytest.approx((left + right) / 2, abs=1e-12)
    assert abs(middle - config.amplitude_m) > 1e-5


def test_reference_includes_edges_and_rejects_outside_and_nonfinite(
    training_plant: D1Plant,
) -> None:
    training_plant.set_training_terrain(TrainingTerrainConfig())
    for x in (-TRAINING_TERRAIN_X_HALF_SIZE_M, TRAINING_TERRAIN_X_HALF_SIZE_M):
        for y in (-TRAINING_TERRAIN_Y_HALF_SIZE_M, TRAINING_TERRAIN_Y_HALF_SIZE_M):
            assert training_plant.training_ground_reference(x, y).height_m == 0.0
    for x, y in (
        (6.001, 0.0),
        (-6.001, 0.0),
        (0.0, 3.001),
        (0.0, -3.001),
        (np.nan, 0),
        (0, np.inf),
    ):
        with pytest.raises(ValueError):
            training_plant.training_ground_reference(x, y)


def test_switching_terrain_keeps_model_identity_and_does_not_change_state(
    training_plant: D1Plant,
) -> None:
    training_plant.reset()
    before_qpos, before_qvel = training_plant.simulation_state()
    model_id = id(training_plant.model)
    floor_id = training_plant.floor_geom_id
    config = TrainingTerrainConfig(kind="bumps", amplitude_m=0.01, phase_rad=0.3)
    training_plant.set_training_terrain(config)
    np.testing.assert_array_equal(training_plant.data.qpos, before_qpos)
    np.testing.assert_array_equal(training_plant.data.qvel, before_qvel)
    assert id(training_plant.model) == model_id
    assert training_plant.floor_geom_id == floor_id
    assert training_plant.training_terrain == config
    first_data = training_plant.model.hfield_data.copy()
    training_plant.set_training_terrain(TrainingTerrainConfig(kind="ramp", slope_deg=-2.0))
    training_plant.set_training_terrain(config)
    np.testing.assert_array_equal(training_plant.model.hfield_data, first_data)
    with pytest.raises(TypeError):
        training_plant.set_training_terrain({"kind": "flat"})
    np.testing.assert_array_equal(training_plant.model.hfield_data, first_data)


def test_training_terrain_is_opt_in_and_incompatible_with_course() -> None:
    legacy = D1Plant()
    assert legacy.training_terrain is None
    assert legacy.model.nhfield == 0
    assert legacy.model.geom_type[legacy.floor_geom_id] == mujoco.mjtGeom.mjGEOM_PLANE
    with pytest.raises(ValueError, match="not constructed"):
        legacy.set_training_terrain(TrainingTerrainConfig())
    with pytest.raises(ValueError, match="not constructed"):
        legacy.training_ground_reference(0.0, 0.0)
    with pytest.raises(ValueError, match="requires arena"):
        build_d1_model(arena="course", training_terrain=TrainingTerrainConfig())
    with pytest.raises(ValueError, match="TrainingTerrainConfig"):
        build_d1_model(training_terrain="flat")


def test_factory_initializes_requested_collision_profile() -> None:
    model = build_d1_model(training_terrain=TrainingTerrainConfig(kind="ramp", slope_deg=3.0))
    assert training_ground_reference(model, 2.0, 0.0).height_m == pytest.approx(
        2.0 * np.tan(np.deg2rad(3.0)), abs=9e-8
    )


def test_flat_heightfield_supports_four_wheels_and_survives_domain_reset(
    training_plant: D1Plant,
) -> None:
    training_plant.set_training_terrain(TrainingTerrainConfig())
    before = training_plant.model.hfield_data.copy()
    training_plant.set_domain()
    np.testing.assert_array_equal(training_plant.model.hfield_data, before)
    training_plant.reset()
    controller = D1VMCController(training_plant)
    states = D1MujocoTruthStateSource(training_plant)
    state = states.reset()
    for _ in range(100):
        training_plant.step(controller.compute(D1Command(), state))
        state = states.read()
    assert training_plant.wheel_ground_contacts == 4
    assert training_plant.undesired_ground_contacts == 0
    assert not training_plant.has_fallen()
    assert training_plant.base_position[2] > 0.40
    assert training_plant.floor_geom_id in training_plant.terrain_geom_ids
