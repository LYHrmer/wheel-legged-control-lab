from dataclasses import asdict

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from wheel_legged_control.d1.env import D1ResidualEnv
from wheel_legged_control.d1.model import NOMINAL_JOINT_POSITION
from wheel_legged_control.d1.terrain_env import (
    D1_TERRAIN_ACTION_SCHEMA,
    D1_TERRAIN_OBSERVATION_SCHEMA,
    D1_TERRAIN_RESIDUAL_SCALE,
    D1TerrainResidualEnv,
    calculate_d1_terrain_reward,
    sample_training_terrain,
)
from wheel_legged_control.d1.training_terrain import TrainingTerrainConfig


@pytest.fixture
def env():
    instance = D1TerrainResidualEnv(episode_seconds=0.2)
    yield instance
    instance.close()


def test_terrain_environment_follows_gymnasium_contract(env):
    check_env(env, skip_render_check=True)


def test_observation_and_action_contract_is_separate_from_legacy(env):
    observation, info = env.reset(seed=12)
    assert observation.shape == (44,)
    assert observation.dtype == np.float32
    assert env.observation_schema == D1_TERRAIN_OBSERVATION_SCHEMA
    assert env.action_schema == D1_TERRAIN_ACTION_SCHEMA
    assert info["ground_reference_source"] == "simulation_oracle"
    assert info["attitude_target"] == "world_upright"
    np.testing.assert_allclose(env._residual_force(np.ones(2)), (11.25, 20.0))
    legacy = D1ResidualEnv(randomize=False, episode_seconds=0.02)
    try:
        assert legacy.observation_space.shape == (42,)
        assert legacy.observation_schema != env.observation_schema
        np.testing.assert_allclose(legacy._residual_force(np.ones(2)), (45.0, 80.0))
    finally:
        legacy.close()


@pytest.mark.parametrize(
    "terrain",
    [
        TrainingTerrainConfig(),
        TrainingTerrainConfig(kind="ramp", slope_deg=-4.0),
        TrainingTerrainConfig(kind="ramp", slope_deg=4.0),
        TrainingTerrainConfig(kind="bumps", amplitude_m=0.010, wavelength_m=0.8, phase_rad=1.1),
    ],
)
def test_reset_has_no_ground_penetration_or_hidden_physics_steps(env, terrain):
    _, info = env.reset(seed=4, options={"terrain": terrain})
    assert env.plant.data.time == 0.0
    assert info["episode_step"] == 0
    assert not info["policy_applied"]
    assert info["clearance_m"] >= 0.455 - 1e-8
    assert info["initial_height_lift_m"] >= 0.0
    for contact in env.plant.data.contact:
        if (
            int(contact.geom1) in env.plant.terrain_geom_ids
            or int(contact.geom2) in env.plant.terrain_geom_ids
        ):
            assert contact.dist >= -1e-8
    assert info["state_age_ms"] == 0.0
    np.testing.assert_allclose(env._state.base_position, env.plant.base_position)


@pytest.mark.parametrize(
    "terrain",
    [
        {"kind": "flat"},
        {"kind": "ramp", "slope_deg": -4.0},
        {"kind": "ramp", "slope_deg": 4.0},
        {"kind": "bumps", "amplitude_m": 0.010, "wavelength_m": 0.8, "phase_rad": 1.1},
    ],
)
def test_short_rollout_is_finite_and_has_no_hidden_push_or_policy_gate(env, terrain):
    env.reset(seed=3, options={"terrain": terrain, "velocity_mps": 0.35})
    for step in range(20):
        observation, reward, terminated, truncated, info = env.step(np.zeros(2))
        assert np.isfinite(observation).all()
        assert reward == info["reward_terms"]["total"]
        assert not terminated
        assert info["episode_step"] == step + 1
        assert info["push_force_n"] == 0.0
        assert info["policy_applied"]
        assert not info["policy_gated"]
        assert 0.0 <= info["torque_saturation_fraction"] <= 1.0
        assert info["world_height_target_m"] == info["command_height_m"]
    assert truncated
    assert info["termination_reason"] == "time_limit"


def test_reset_seed_reproduces_terrain_command_state_and_step(env):
    env.set_curriculum_stage(3)
    first_observation, first_info = env.reset(seed=18)
    first_transition = env.step(np.asarray((0.1, -0.2)))
    second_observation, second_info = env.reset(seed=18)
    second_transition = env.step(np.asarray((0.1, -0.2)))
    np.testing.assert_array_equal(first_observation, second_observation)
    assert first_info["terrain_config"] == second_info["terrain_config"]
    assert first_info["target_velocity_mps"] == second_info["target_velocity_mps"]
    np.testing.assert_array_equal(first_transition[0], second_transition[0])
    assert first_transition[1:4] == second_transition[1:4]


def test_stage_changes_only_on_future_reset(env):
    _, info = env.reset(seed=1)
    assert info["curriculum_stage"] == 0
    terrain_before = info["terrain_config"]
    env.set_curriculum_stage(3)
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["curriculum_stage"] == 0
    assert info["terrain_config"] == terrain_before
    _, info = env.reset(seed=1)
    assert info["curriculum_stage"] == 3


def test_flat_mode_samples_only_flat_but_explicit_evaluation_terrain_is_allowed():
    env = D1TerrainResidualEnv(training_mode="flat", stage=3, episode_seconds=0.02)
    try:
        for seed in range(5):
            _, info = env.reset(seed=seed)
            assert info["terrain_config"]["kind"] == "flat"
            assert 0.20 <= info["target_velocity_mps"] <= 0.45
        explicit = TrainingTerrainConfig(kind="ramp", slope_deg=3.0)
        _, info = env.reset(options={"terrain": explicit, "velocity_mps": 0.35})
        assert info["terrain_config"] == asdict(explicit)
    finally:
        env.close()


@pytest.mark.parametrize(
    "stage, probabilities",
    [
        (0, (1.0, 0.0, 0.0)),
        (1, (0.5, 0.5, 0.0)),
        (2, (0.3, 0.35, 0.35)),
        (3, (0.25, 0.25, 0.5)),
    ],
)
def test_curriculum_distribution_and_reserved_parameters(stage, probabilities):
    rng = np.random.default_rng(45)
    samples = [sample_training_terrain(rng, stage) for _ in range(2000)]
    counts = [sum(item.kind == kind for item in samples) for kind in ("flat", "bumps", "ramp")]
    np.testing.assert_allclose(np.asarray(counts) / 2000, probabilities, atol=0.04)
    for terrain in samples:
        if terrain.kind == "bumps":
            assert terrain.wavelength_m in (0.8, 1.0, 1.2)
            assert 0.005 <= terrain.amplitude_m <= 0.010
            if stage == 1:
                assert terrain.amplitude_m == 0.005
        if terrain.kind == "ramp":
            assert abs(terrain.slope_deg) in ((2.0,) if stage == 2 else (2.0, 4.0))


def test_relative_height_reward_is_invariant_to_world_translation():
    inputs = {
        "command_clearance_m": 0.455,
        "forward_velocity_mps": 0.32,
        "command_velocity_mps": 0.35,
        "roll_rad": 0.02,
        "pitch_rad": -0.03,
        "joint_position": NOMINAL_JOINT_POSITION,
        "normalized_action": np.asarray((0.1, -0.2)),
        "previous_normalized_action": np.zeros(2),
        "undesired_contacts": 0,
        "terminated": False,
    }
    first = calculate_d1_terrain_reward(ground_height_m=0.0, base_height_m=0.46, **inputs)
    shifted = calculate_d1_terrain_reward(ground_height_m=-0.5, base_height_m=-0.04, **inputs)
    assert first.as_dict() == pytest.approx(shifted.as_dict())


def _move_base(env, *, x, clearance):
    ground = env.plant.training_ground_reference(x, 0.0)
    qpos, qvel = env.plant.simulation_state()
    qpos[:3] = (x, 0.0, ground.height_m + clearance)
    env.plant.set_simulation_state(qpos, qvel)
    env._publish_control_state(env._state_source.read())
    env._command = env._command_at_step()


def test_height_termination_uses_local_ground_not_world_z(env):
    env.reset(options={"terrain": {"kind": "ramp", "slope_deg": 4.0}})
    _move_base(env, x=-4.0, clearance=0.455)
    assert env.plant.base_position[2] < 0.22
    assert not env._has_fallen()
    _move_base(env, x=4.0, clearance=0.21)
    assert env.plant.base_position[2] > 0.22
    assert env._has_fallen()
    assert env._termination_reason == "low_clearance"


def test_observation_height_and_ground_angles_use_current_position(env):
    env.reset(options={"terrain": {"kind": "ramp", "slope_deg": 4.0}})
    _move_base(env, x=0.0, clearance=0.455)
    origin_observation = env._observation()
    _move_base(env, x=2.0, clearance=0.455)
    observation = env._observation()
    np.testing.assert_allclose(observation[:42], origin_observation[:42], atol=1e-6)
    assert observation[10] == pytest.approx(0.0, abs=1e-6)
    assert observation[-2] == pytest.approx(-np.deg2rad(4.0), abs=2e-5)
    assert observation[-1] == 0.0
    assert env._command.pitch_rad == 0.0
    assert env._command.roll_rad == 0.0


def test_transition_reward_and_next_observation_have_explicit_height_timing(env, monkeypatch):
    env.reset(options={"terrain": {"kind": "ramp", "slope_deg": 4.0}})
    previous_target = env._command.base_height_m
    original_step = env.plant.step

    def step_with_known_displacement(*args, **kwargs):
        original_step(*args, **kwargs)
        qpos, qvel = env.plant.simulation_state()
        qpos[0] += 0.2
        env.plant.set_simulation_state(qpos, qvel)

    monkeypatch.setattr(env.plant, "step", step_with_known_displacement)
    observation, _, _, _, info = env.step(np.zeros(2))
    assert info["world_height_target_m"] == previous_target
    assert abs(info["reward_world_height_target_m"] - previous_target) > 0.005
    assert info["reward_world_height_target_m"] == pytest.approx(
        info["ground_height_m"] + info["command_clearance_m"]
    )
    assert info["reward_terms"]["height_tracking"] == pytest.approx(
        np.exp(-((info["clearance_error_m"] / 0.03) ** 2))
    )
    assert observation[10] == pytest.approx(info["clearance_error_m"] / 0.08, abs=1e-6)


def test_command_starts_smoothly_and_never_injects_old_training_push(env):
    env.reset(options={"velocity_mps": 0.4})
    for step, expected in ((0, 0.0), (30, 0.0), (45, 0.2), (60, 0.4), (160, 0.4)):
        env._step_count = step
        command = env._command_at_step()
        assert command.forward_velocity_mps == pytest.approx(expected)
        assert command.yaw_rate_rps == 0.0
    assert env._push_force_n == 0.0
    assert env._push_start == env._push_end == -1


def test_action_delay_and_reset_clear_previous_residual(env):
    env.reset(options={"action_delay_steps": 2})
    for expected in (np.zeros(2), np.zeros(2), np.ones(2)):
        _, _, _, _, info = env.step(np.ones(2))
        np.testing.assert_array_equal(info["residual_action"], expected)
    observation, info = env.reset(options={"action_delay_steps": 0})
    np.testing.assert_array_equal(observation[40:42], (0.0, 0.0))
    assert info["longitudinal_force_n"] == 0.0
    assert info["vertical_residual_n"] == 0.0
    assert len(env._delay_queue) == 0
    np.testing.assert_array_equal(info["residual_scale_n"], D1_TERRAIN_RESIDUAL_SCALE)


@pytest.mark.parametrize("action", [np.zeros(3), [np.nan, 0.0], [0.0, np.inf], 1.0])
def test_invalid_actions_do_not_mutate_environment(env, action):
    env.reset(seed=4)
    before = env.plant.simulation_state()
    with pytest.raises(ValueError, match="two finite"):
        env.step(action)
    after = env.plant.simulation_state()
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    assert env._step_count == 0


@pytest.mark.parametrize(
    "options",
    [
        {"velocity_mps": -0.1},
        {"velocity_mps": np.nan},
        {"velocity_mps": 0.7},
        {"height_m": 0.2},
        {"height_m": np.inf},
        {"height_m": True},
        {"action_delay_steps": -1},
        {"action_delay_steps": 1.5},
        {"initial_pitch": np.nan},
        {"push_force_n": 150.0},
        {"scenario": "training"},
    ],
)
def test_invalid_reset_options_are_rejected(env, options):
    with pytest.raises(ValueError):
        env.reset(options=options)


@pytest.mark.parametrize("stage", [-1, 4, 1.5, True, "2"])
def test_invalid_stage_is_rejected_before_model_creation(stage):
    with pytest.raises(ValueError, match="stage"):
        D1TerrainResidualEnv(stage=stage)


@pytest.mark.parametrize("duration", [0.0, -1.0, 0.001, np.nan, np.inf, True])
def test_invalid_duration_is_rejected_before_model_creation(duration):
    with pytest.raises(ValueError, match="episode_seconds"):
        D1TerrainResidualEnv(episode_seconds=duration)
