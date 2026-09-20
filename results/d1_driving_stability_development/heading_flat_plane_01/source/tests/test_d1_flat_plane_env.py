"""Native-plane contract checks; every integration entry point is forbidden.

These tests initialize models and refresh collision/kinematic caches only. They
do not qualify a controller or predict the outcome of a plane trajectory.
"""

import json
from dataclasses import asdict, replace

import mujoco
import numpy as np
import pytest

from scripts.d1_flat_plane_env import (
    FLAT_PLANE_SCHEMA,
    FLAT_PLANE_TASK_SCHEMA,
    D1FlatPlaneHeadingEnv,
    D1FlatPlanePlant,
)
from scripts.d1_heading_tracking_env import D1HeadingTrackingEnv, load_heading_policy
from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.locomotion_checkpoint import (
    _environment_contract,
    load_locomotion_policy,
)
from wheel_legged_control.d1.locomotion_env import D1WheelLegControlConfig
from wheel_legged_control.d1.locomotion_terrain import D1LocomotionTerrainConfig
from wheel_legged_control.d1.model import D1Plant, NOMINAL_JOINT_POSITION
from wheel_legged_control.d1.state_estimation import D1EstimatorImpairments
from wheel_legged_control.d1.state_provider import D1StateProviderConfig


@pytest.fixture(autouse=True, scope="module")
def forbid_physics_integration():
    """Guard also covers module-scoped model fixtures, not just test bodies."""
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail("nonphysical flat-plane preflight called a physics integrator")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            patch.setattr(mujoco, name, forbidden)
        yield calls
    assert calls == []


@pytest.fixture(scope="module")
def plant_pair(forbid_physics_integration):
    return (
        D1Plant(sampling_mode="synchronized", locomotion_terrain=D1LocomotionTerrainConfig()),
        D1FlatPlanePlant(),
    )


def assert_compiled_invariants(old, new):
    """Compare physical inputs and identities, excluding floor representation.

    The returned field list lets the standalone preflight record exactly which
    arrays it checked. Robot geometry is compared in full; floor collision
    material is compared explicitly, without equating hfield/plane caches.
    """
    checked = []
    for name in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nmesh", "nmat"):
        assert getattr(old, name) == getattr(new, name), name
        checked.append(name)
    for kind, count in ((mujoco.mjtObj.mjOBJ_BODY, old.nbody),
                        (mujoco.mjtObj.mjOBJ_JOINT, old.njnt),
                        (mujoco.mjtObj.mjOBJ_ACTUATOR, old.nu),
                        (mujoco.mjtObj.mjOBJ_GEOM, old.ngeom)):
        assert [mujoco.mj_id2name(old, kind, i) for i in range(count)] == [
            mujoco.mj_id2name(new, kind, i) for i in range(count)
        ]
    for name in dir(old):
        if name.startswith(("body_", "jnt_", "dof_", "actuator_", "mat_")):
            value = getattr(old, name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, getattr(new, name), err_msg=name)
                checked.append(name)
    for name in ("qpos0", "qpos_spring"):
        np.testing.assert_array_equal(getattr(old, name), getattr(new, name), err_msg=name)
        checked.append(name)
    floor = mujoco.mj_name2id(old, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor == mujoco.mj_name2id(new, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    robot = np.arange(old.ngeom) != floor
    for name in dir(old):
        value = getattr(old, name)
        if name.startswith("geom_") and isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value[robot], getattr(new, name)[robot], err_msg=name)
            checked.append(name + "[robot]")
    for name in ("bodyid", "pos", "quat", "friction", "condim", "margin", "gap",
                 "solref", "solimp", "solmix", "priority", "contype", "conaffinity"):
        name = "geom_" + name
        np.testing.assert_array_equal(getattr(old, name)[floor], getattr(new, name)[floor], err_msg=name)
        checked.append(name + "[floor]")
    for name in dir(old.opt):
        if not name.startswith("_") and not callable(getattr(old.opt, name)):
            np.testing.assert_array_equal(getattr(old.opt, name), getattr(new.opt, name), err_msg="opt." + name)
            checked.append("opt." + name)
    return checked


def assert_zero_clock(plant):
    assert plant.data.time == plant.measurement_data.time == 0.0
    assert plant.data is not plant.measurement_data


def test_robot_solver_material_and_actuator_invariants(plant_pair):
    old, new = plant_pair
    assert_compiled_invariants(old.model, new.model)
    for name in ("joint_ids", "qpos_addresses", "dof_addresses", "actuator_ids",
                 "wheel_body_ids_by_leg", "base_body_id", "floor_geom_id"):
        np.testing.assert_array_equal(getattr(old, name), getattr(new, name), err_msg=name)
    assert old.model.nhfield == 1 and new.model.nhfield == 0
    assert old.model.geom_type[old.floor_geom_id] == mujoco.mjtGeom.mjGEOM_HFIELD
    assert new.model.geom_type[new.floor_geom_id] == mujoco.mjtGeom.mjGEOM_PLANE
    assert new.model.geom_dataid[new.floor_geom_id] == -1
    assert new.terrain_geom_ids == frozenset({new.floor_geom_id})
    assert new.locomotion_terrain is None and new.training_terrain is None
    assert new.control_dt == 0.01 and new.physics_steps == 5
    assert new.model.opt.timestep == 0.002
    assert new.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert new.model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert new.model.opt.iterations == 20 and new.model.opt.ls_iterations == 5
    assert new.model.geom_margin[new.floor_geom_id] == 0.0
    np.testing.assert_array_equal(new.model.geom_friction[new.floor_geom_id], [.9, .005, .0001])
    assert new.reset.__func__ is D1Plant.reset
    assert new.step.__func__ is D1Plant.step
    assert_zero_clock(old)
    assert_zero_clock(new)


@pytest.mark.parametrize("point", [(0, 0), (-6, -3), (6, 3), (-3.8, 0),
                                   (np.float32(1), np.int64(-2))])
def test_finite_analytic_reference_including_boundary(plant_pair, point):
    plant = plant_pair[1]
    ground = plant.locomotion_ground_reference(*point)
    assert (ground.height_m, ground.pitch_rad, ground.roll_rad) == (0., 0., 0.)
    assert_zero_clock(plant)


@pytest.mark.parametrize("point", [(np.nextafter(6., np.inf), 0), (0, -3.0000001),
                                   (np.nan, 0), (0, np.inf), (True, 0),
                                   (0, np.bool_(False)), (1j, 0), ("0", 0),
                                   (np.array(0.), 0), ([0.], 0)])
def test_invalid_reference_coordinates_rejected(plant_pair, point):
    with pytest.raises(ValueError):
        plant_pair[1].locomotion_ground_reference(*point)
    assert_zero_clock(plant_pair[1])


@pytest.mark.parametrize("field,index,value", [
    ("geom_type", None, int(mujoco.mjtGeom.mjGEOM_HFIELD)),
    ("geom_bodyid", None, 1), ("geom_dataid", None, 0),
    ("geom_pos", 2, 1e-15), ("geom_quat", 1, 1e-15),
])
def test_ground_query_and_reset_reject_geometry_drift(field, index, value):
    env = D1FlatPlaneHeadingEnv(episode_seconds=.01)
    array = getattr(env.plant.model, field)
    key = env.plant.floor_geom_id if index is None else (env.plant.floor_geom_id, index)
    original = array[key].copy()
    try:
        array[key] = value  # Temporary isolated test model, never stepped or saved.
        with pytest.raises(RuntimeError):
            env.plant.locomotion_ground_reference(0, 0)
        with pytest.raises(RuntimeError):
            env.reset(seed=55101)
        assert_zero_clock(env.plant)
    finally:
        array[key] = original
        env.close()


def test_constructor_reset_prepare_bind_only_new_plant_without_integrating():
    calls = []

    def command(time_s):
        calls.append(time_s)
        return D1MotionCommand(forward_velocity_mps=0., yaw_rate_rps=0.)

    env = D1FlatPlaneHeadingEnv(episode_seconds=.02, command_source=command)
    try:
        plant = env.plant
        controller = env._controller
        assert isinstance(plant, D1FlatPlanePlant)
        assert env.loop is env.decision is env.last_transition is None
        assert not env._active and calls == []
        assert_zero_clock(plant)
        obs, info = env.reset(seed=55101)
        assert env.loop.plant is plant and env.loop.provider._plant is plant
        assert env.loop.controller is controller and env._controller is controller
        assert env.loop.provider.config.kind == "oracle"
        assert env.decision is not None and env.heading_decision.loop is env.loop
        assert calls == [0.] and env._steps == 0 and env.last_transition is None
        assert obs.dtype == np.float32 and obs.shape == (85,)
        assert np.isfinite(obs).all() and env.observation_space.contains(obs)
        assert env.action_space.shape == (8,) and env.action_mode == "independent8"
        np.testing.assert_array_equal(plant.data.qpos[:7], [-3.8, 0, .455, 1, 0, 0, 0])
        np.testing.assert_array_equal(plant.data.qpos[plant.qpos_addresses], NOMINAL_JOINT_POSITION)
        np.testing.assert_array_equal(plant.data.qvel, np.zeros(plant.model.nv))
        np.testing.assert_array_equal(plant.data.qacc_warmstart, np.zeros(plant.model.nv))
        before = plant.data.qpos.copy(), plant.data.qvel.copy(), plant.data.qacc_warmstart.copy()
        # Parent prepare publishes the unchanged 82-field servo observation;
        # heading reset/step append three fields at their public boundary.
        np.testing.assert_array_equal(env._prepare(), obs[:82])
        np.testing.assert_array_equal(env._prepare(), obs[:82])
        assert calls == [0.]
        for actual, expected in zip((plant.data.qpos, plant.data.qvel, plant.data.qacc_warmstart), before):
            np.testing.assert_array_equal(actual, expected)
        old_loop = env.loop
        repeated, repeated_info = env.reset(seed=55101)
        assert env.plant is plant and env.loop is not old_loop
        assert env.loop.provider._plant is plant and env.loop.plant is plant
        np.testing.assert_array_equal(repeated, obs)
        assert repeated_info == info and calls == [0., 0.]
        assert_zero_clock(plant)
    finally:
        env.close()


def test_actual_plane_metadata_task_identity_and_copy_isolation():
    env = D1FlatPlaneHeadingEnv(episode_seconds=.01)
    old = D1HeadingTrackingEnv(episode_seconds=.01)
    try:
        _, info = env.reset(seed=55101)
        old.reset(seed=55101)
        meta = env.episode_metadata
        terrain = meta["terrain"]
        assert terrain == meta["collision_terrain"] == env.collision_terrain_metadata
        assert terrain["schema"] == FLAT_PLANE_SCHEMA
        assert terrain["plant_schema"] != old.episode_metadata["terrain"]["schema"]
        assert terrain["collision_geometry"] == "native_mujoco_plane"
        assert terrain["physical_geometry_extent"] == "infinite"
        assert terrain["reference_domain_half_size_m"] == [6., 3.]
        assert terrain["surface_normal"] == [0., 0., 1.] and terrain["floor_height_m"] == 0.
        assert terrain["heightfield_present"] is False and terrain["heightfield_count"] == 0
        assert meta["safe_half_size_m"] == (5.3, 2.3)
        assert meta["requested_terrain_config"] == asdict(D1LocomotionTerrainConfig())
        assert meta["task_schema"] == env.task_schema == FLAT_PLANE_TASK_SCHEMA
        assert info["episode_metadata"] == env._episode_metadata == meta
        assert meta["heading_task_config"] == env.heading_task_config
        assert env.heading_task_config["task_schema"] == env.task_schema
        assert env.heading_task_config["collision_terrain"]["zero_residual_only"] is True
        for name in ("source_schema", "controller_schema", "observation_schema", "action_schema", "reward_schema"):
            assert meta[name] == old.episode_metadata[name]
        assert meta["controller_parameters"] == asdict(D1WheelLegControlConfig())
        assert env.heading_kp == 2. and env.heading_kd == .4 and env.heading_limit_rps == 1.
        assert all(value == 1. for value in meta["domain"].values())
        assert "heading_reference_initial" in meta
        json.dumps(meta, allow_nan=False)
        terrain["surface_normal"][2] = 99
        info["episode_metadata"]["terrain"]["floor_height_m"] = 99
        env.collision_terrain_metadata["reference_domain_half_size_m"][0] = 99
        assert env.episode_metadata["terrain"]["surface_normal"] == [0., 0., 1.]
        assert env.episode_metadata["terrain"]["floor_height_m"] == 0.
        assert env.collision_terrain_metadata["reference_domain_half_size_m"] == [6., 3.]
        assert_zero_clock(env.plant)
        assert_zero_clock(old.plant)
    finally:
        env.close()
        old.close()


@pytest.mark.parametrize("loader,match", [(load_heading_policy, "heading_task_config"),
                                          (load_locomotion_policy, "task_schema")])
def test_old_85_checkpoint_rejected_by_each_layer_before_file_or_deserialization(tmp_path, monkeypatch, loader, match):
    from stable_baselines3 import PPO

    def forbidden(*args, **kwargs):
        pytest.fail("incompatible checkpoint reached PPO deserialization")

    monkeypatch.setattr(PPO, "load", forbidden)
    source = D1HeadingTrackingEnv(episode_seconds=.01)
    target = D1FlatPlaneHeadingEnv(episode_seconds=.01)
    try:
        source.reset(seed=55101)
        target.reset(seed=55101)
        sidecar = tmp_path / "old85.json"
        sidecar.write_text(json.dumps({**_environment_contract(source),
                                      "recorded_episode": source.episode_metadata,
                                      "model_sha256": "not-read"}))
        absent_zip = tmp_path / "never_created.zip"
        assert not absent_zip.exists()
        with pytest.raises(ValueError, match=match):
            loader(absent_zip, sidecar, target)
        assert_zero_clock(source.plant)
        assert_zero_clock(target.plant)
    finally:
        source.close()
        target.close()


@pytest.mark.parametrize("kwargs", [
    {"terrain": D1LocomotionTerrainConfig(layout="straight")},
    {"terrain": {}}, {"baseline": "lqr"}, {"baseline": "mpc"},
    {"action_mode": "shared2"}, {"provider_config": D1StateProviderConfig(
        "truth_impairment", impairments=D1EstimatorImpairments())},
    {"wheel_leg_control": replace(D1WheelLegControlConfig(), yaw_feedback_gain=5.)},
    {"heading_kp": 3.}, {"heading_limit_rps": .6},
    {"training_terrain": object()}, {"arena": "course"},
])
def test_unsupported_environment_configuration_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        D1FlatPlaneHeadingEnv(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"sampling_mode": "legacy_mixed"}, {"actuator_channel": object()},
    {"control_dt": True}, {"control_dt": np.inf}, {"control_dt": 0.},
    {"control_dt": .003}, {"ground_friction": np.nan}, {"ground_friction": False},
    {"ground_friction": 0.}, {"arena": "course"}, {"training_terrain": object()},
    {"locomotion_terrain": D1LocomotionTerrainConfig()},
])
def test_unsupported_plant_configuration_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        D1FlatPlanePlant(**kwargs)


@pytest.mark.parametrize("action", [True, np.zeros(8, dtype=bool), np.zeros(7),
                                    np.zeros((1, 8)), np.full(8, np.nan),
                                    np.full(8, np.inf), np.full(8, 1e-300),
                                    np.zeros(8, dtype=complex), ["0"] * 8])
def test_invalid_action_rejected_before_parent_or_any_state_change(monkeypatch, action):
    env = D1FlatPlaneHeadingEnv(episode_seconds=.01)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid action reached the parent step")

    try:
        env.reset(seed=55101)
        state = env.plant.data.qpos.copy(), env.plant.data.qvel.copy()
        monkeypatch.setattr(D1HeadingTrackingEnv, "step", forbidden)
        with pytest.raises(ValueError):
            env.step(action)
        np.testing.assert_array_equal(env.plant.data.qpos, state[0])
        np.testing.assert_array_equal(env.plant.data.qvel, state[1])
        assert env._steps == 0 and env.last_transition is None
        assert_zero_clock(env.plant)
    finally:
        env.close()


@pytest.mark.parametrize("action", [np.zeros(8), [0] * 8, np.full(8, -0., dtype=np.float32)])
def test_exact_zero_delegates_once_without_executing_parent_physics(monkeypatch, action):
    calls = []
    sentinel = object()

    def parent(self, received, *args, **kwargs):
        calls.append((self, received, args, kwargs))
        return sentinel

    env = D1FlatPlaneHeadingEnv(episode_seconds=.01)
    try:
        monkeypatch.setattr(D1HeadingTrackingEnv, "step", parent)
        assert env.step(action) is sentinel
        assert len(calls) == 1 and calls[0][0] is env and calls[0][1] is action
        assert_zero_clock(env.plant)
    finally:
        env.close()
