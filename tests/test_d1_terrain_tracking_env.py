import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from scipy.spatial.transform import Rotation

from wheel_legged_control.d1.terrain_env import D1TerrainResidualEnv, calculate_d1_terrain_reward
from wheel_legged_control.d1.terrain_tracking_env import (
    D1TerrainTrackingEnv,
    terrain_normal_to_rpy,
)


@pytest.mark.parametrize("slope_deg", [-4.0, 4.0])
def test_zero_residual_tracks_forward_command_on_four_degree_ramps(slope_deg):
    env = D1TerrainTrackingEnv(episode_seconds=4.0, randomize=False)
    try:
        env.reset(
            seed=31,
            options={"terrain": {"kind": "ramp", "slope_deg": slope_deg}, "velocity_mps": 0.35},
        )
        errors = []
        for _ in range(400):
            _, _, terminated, truncated, info = env.step(np.zeros(2))
            errors.append(info["velocity_error_mps"])
            assert not terminated
        assert truncated
        assert 0.7 < info["position_x_m"] < 1.6
        assert np.sqrt(np.mean(np.square(errors))) < 0.16
    finally:
        env.close()


def test_mixed_mode_uses_stage_three_without_allowing_curriculum_regression():
    env = D1TerrainTrackingEnv(training_mode="mixed", episode_seconds=0.02)
    try:
        _, info = env.reset(seed=31)
        assert info["training_mode"] == "mixed"
        assert info["curriculum_stage"] == 3
        env.set_curriculum_stage(3)
        for stage in (0, 1, 2):
            with pytest.raises(ValueError, match="mixed"):
                env.set_curriculum_stage(stage)
        reference = D1TerrainResidualEnv(training_mode="curriculum", stage=3, episode_seconds=0.02)
        try:
            for seed in range(5):
                _, mixed_info = env.reset(seed=seed)
                _, reference_info = reference.reset(seed=seed)
                assert mixed_info["terrain_config"] == reference_info["terrain_config"]
                assert mixed_info["target_velocity_mps"] == reference_info["target_velocity_mps"]
        finally:
            reference.close()
    finally:
        env.close()


def test_v2_rejects_mpc_instead_of_mislabeling_its_control_schema():
    with pytest.raises(ValueError, match="lqr"):
        D1TerrainTrackingEnv(baseline="mpc")


def test_flat_uses_the_identical_physics_path_and_raw_rewards_as_v1():
    old = D1TerrainResidualEnv(training_mode="flat", episode_seconds=1.0)
    new = D1TerrainTrackingEnv(training_mode="flat", episode_seconds=1.0)
    try:
        first_old, old_info = old.reset(seed=31, options={"velocity_mps": 0.35})
        first_new, new_info = new.reset(seed=31, options={"velocity_mps": 0.35})
        np.testing.assert_array_equal(first_old, first_new)
        assert old_info["initial_height_lift_m"] == new_info["initial_height_lift_m"]
        for action in np.random.default_rng(913).uniform(-0.2, 0.2, (100, 2)):
            transition_old = old.step(action)
            transition_new = new.step(action)
            np.testing.assert_array_equal(transition_old[0], transition_new[0])
            assert transition_old[1:4] == transition_new[1:4]
            assert transition_old[4]["reward_terms"] == transition_new[4]["reward_terms"]
            np.testing.assert_array_equal(old.plant.data.qpos, new.plant.data.qpos)
            np.testing.assert_array_equal(old.plant.data.qvel, new.plant.data.qvel)
    finally:
        old.close()
        new.close()


def test_same_observation_shape_does_not_imply_checkpoint_schema_compatibility():
    env = D1TerrainTrackingEnv(episode_seconds=0.01)
    try:
        observation, info = env.reset(seed=31)
        assert observation.shape == (44,)
        assert info["observation_schema"] == "d1-terrain-tracking-oracle-v2"
        assert info["reward_schema"] == "d1-terrain-tracking-v2"
        assert info["control_schema"] == "d1-lqr-vmc-local-tangent-v2"
        assert info["action_schema"] == "d1-terrain-residual-quarter-v1"
        assert info["observation_schema"] != D1TerrainResidualEnv.observation_schema
        assert info["attitude_target"] == "local_collision_tangent"
        assert info["ground_reference_source"] == "simulation_oracle"
        np.testing.assert_array_equal(info["residual_scale_n"], (11.25, 20.0))
    finally:
        env.close()


@pytest.mark.parametrize(
    "slope_x, slope_y, yaw",
    [
        (0.1, 0.0, 0.0),
        (-0.1, 0.0, 0.0),
        (0.12, -0.08, 0.7),
        (0.12, 0.15, -1.2),
        (0.1, 0.0, np.pi / 2),
    ],
)
def test_yaw_rotated_roll_pitch_reconstructs_world_surface_normal(slope_x, slope_y, yaw):
    roll, pitch = terrain_normal_to_rpy(slope_x, slope_y, yaw)
    normal = Rotation.from_euler("xyz", (roll, pitch, yaw)).apply((0, 0, 1))
    expected = np.asarray((-slope_x, -slope_y, 1.0))
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(normal, expected, atol=2e-15)


@pytest.mark.parametrize("yaw", [0.7, np.pi / 2, -1.2])
def test_observation_carries_yaw_relative_target_not_world_pitch(yaw):
    env = D1TerrainTrackingEnv(episode_seconds=0.1)
    try:
        observation, info = env.reset(
            options={"terrain": {"kind": "ramp", "slope_deg": 4.0}, "initial_yaw": yaw}
        )
        normal = Rotation.from_euler("xyz", (observation[-1], observation[-2], yaw)).apply(
            (0, 0, 1)
        )
        expected = np.asarray((np.tan(info["ground_pitch_rad"]), 0.0, 1.0))
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(normal, expected, atol=1e-8)
        assert observation[-2] == pytest.approx(info["command_pitch_rad"], abs=1e-8)
        assert observation[-1] == pytest.approx(info["command_roll_rad"], abs=1e-8)
    finally:
        env.close()


def test_attitude_reward_uses_post_step_target_and_observation_uses_same_target():
    env = D1TerrainTrackingEnv(episode_seconds=1.0)
    try:
        _, previous = env.reset(
            seed=31,
            options={
                "terrain": {"kind": "bumps", "amplitude_m": 0.010, "wavelength_m": 0.8},
                "initial_yaw": 0.4,
                "velocity_mps": 0.35,
            },
        )
        largest_target_change = 0.0
        for _ in range(100):
            observation, reward, terminated, _, info = env.step(np.zeros(2))
            assert not terminated
            assert info["command_roll_rad"] == previous["observation_roll_target_rad"]
            assert info["command_pitch_rad"] == previous["observation_pitch_target_rad"]
            assert observation[-2] == pytest.approx(info["reward_pitch_target_rad"], abs=1e-8)
            assert observation[-1] == pytest.approx(info["reward_roll_target_rad"], abs=1e-8)
            assert info["reward_terms"]["upright"] == pytest.approx(
                np.exp(
                    -((info["roll_error_rad"] / 0.22) ** 2) - (info["pitch_error_rad"] / 0.22) ** 2
                )
            )
            normal = Rotation.from_euler(
                "xyz",
                (info["reward_roll_target_rad"], info["reward_pitch_target_rad"], info["yaw_rad"]),
            ).apply((0, 0, 1))
            expected_normal = np.asarray((np.tan(info["ground_pitch_rad"]), 0.0, 1.0))
            expected_normal /= np.linalg.norm(expected_normal)
            np.testing.assert_allclose(normal, expected_normal, atol=2e-15)
            largest_target_change = max(
                largest_target_change,
                abs(info["reward_pitch_target_rad"] - info["command_pitch_rad"]),
            )
            assert reward == info["reward_terms"]["total"]
            previous = info
        assert largest_target_change > 1e-4
    finally:
        env.close()


def test_only_upright_reward_changes_on_slopes():
    env = D1TerrainTrackingEnv(episode_seconds=0.1)
    try:
        env.reset(
            seed=31,
            options={"terrain": {"kind": "ramp", "slope_deg": 4.0}, "velocity_mps": 0.35},
        )
        previous_action = np.zeros(2)
        for action in np.random.default_rng(913).uniform(-0.2, 0.2, (10, 2)):
            _, reward, terminated, _, info = env.step(action)
            reference = calculate_d1_terrain_reward(
                ground_height_m=info["ground_height_m"],
                command_clearance_m=info["command_clearance_m"],
                base_height_m=float(env.plant.base_position[2]),
                forward_velocity_mps=info["forward_velocity_mps"],
                roll_rad=info["measured_roll_rad"],
                pitch_rad=info["measured_pitch_rad"],
                joint_position=info["joint_position"],
                command_velocity_mps=info["command_velocity_mps"],
                normalized_action=info["residual_action"],
                previous_normalized_action=previous_action,
                undesired_contacts=info["undesired_contacts"],
                terminated=terminated,
            ).as_dict()
            for key, value in reference.items():
                if key not in {"upright", "total"}:
                    assert info["reward_terms"][key] == value
            assert reward > 1.0  # Raw environment reward is not the training-only 0.01 scale.
            previous_action = info["residual_action"]
    finally:
        env.close()


def test_tracking_environment_follows_gymnasium_contract():
    env = D1TerrainTrackingEnv(episode_seconds=0.1, training_mode="mixed")
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


@pytest.mark.parametrize("training_mode", ["flat", "mixed", "curriculum"])
def test_invalid_stage_is_not_silently_accepted_by_mixed_mode(training_mode):
    with pytest.raises(ValueError, match="stage"):
        D1TerrainTrackingEnv(training_mode=training_mode, stage=True)
