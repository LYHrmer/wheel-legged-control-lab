"""Independent short-loop task contracts; not locomotion quality benchmarks."""

from dataclasses import asdict, replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.locomotion_commands import (
    D1CommandSchedule,
    D1CommandSegment,
    schedule_to_dict,
)
from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv, D1LocomotionRandomization
from wheel_legged_control.d1.locomotion_observation import (
    OBSERVATION_SLICES,
    encode_d1_locomotion_observation,
)
from wheel_legged_control.d1.locomotion_terrain import locomotion_terrain_configs
from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.state_estimation import (
    D1EstimatorImpairments,
    D1MujocoTruthStateSource,
    D1NoisyDelayedStateSource,
)
from wheel_legged_control.d1.state_provider import D1StateProviderConfig
from wheel_legged_control.d1.training_terrain import TrainingGroundReference


def _provider(kind):
    if kind == "imu_encoder_fusion":
        return D1StateProviderConfig(
            kind,
            initial_position_m=(-3.8, 0.0, 0.455),
            initial_rpy_rad=(0.0, 0.0, 0.0),
            sensor_delay_steps=1,
        )
    return D1StateProviderConfig(
        kind,
        impairments=D1EstimatorImpairments(delay_steps=1) if kind == "truth_impairment" else None,
    )


def _zero(env):
    return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)


def _part(observation, name):
    start, end = OBSERVATION_SLICES[name]
    return observation[start:end]


@pytest.mark.parametrize("baseline,action_size", (("wheel_leg", 8), ("lqr", 2), ("mpc", 2)))
@pytest.mark.parametrize("source", ("oracle", "truth_impairment", "imu_encoder_fusion"))
def test_controller_provider_combinations_use_next_tick_context(baseline, action_size, source):
    env = D1LocomotionEnv(
        baseline=baseline,
        provider_config=_provider(source),
        episode_seconds=0.06,
        command_source=lambda t: D1MotionCommand(0.10 + t, 0.03 + t),
    )
    try:
        observation, info = env.reset(seed=27)
        assert env.action_space.shape == (action_size,)
        assert env.observation_space.contains(observation)
        assert info["episode_metadata"]["source_schema"] == env.source_schema
        np.testing.assert_array_equal(_part(observation, "previous_applied_action_padded8"), 0.0)
        for tick in range(4):
            previous = env.decision
            action = _zero(env)
            action[0] = 1.2 if tick == 0 else -0.05
            observation, reward, terminated, truncated, info = env.step(action)
            assert not terminated and not truncated
            assert env.observation_space.contains(observation)
            assert env.last_transition.decision is previous
            assert env.decision.context.state is env.loop.provider.read()
            assert env.decision.context.proposal.tick == tick + 1
            assert env.decision.context.proposal.control_time_s == pytest.approx((tick + 1) * 0.01)
            assert env.decision.motion_command != previous.motion_command
            assert info["command"] == asdict(previous.motion_command)
            assert reward == pytest.approx(sum(info["reward_terms"].values()), abs=1e-15)
            assert info["reward_terms"]["termination"] == 0.0
            np.testing.assert_array_equal(info["raw_action"], action)
            np.testing.assert_array_equal(info["applied_action"], np.clip(action, -1, 1))
            previous_observed = _part(observation, "previous_applied_action_padded8")
            np.testing.assert_array_equal(previous_observed[:action_size], np.clip(action, -1, 1))
            np.testing.assert_array_equal(previous_observed[action_size:], 0.0)
            command_observed = _part(observation, "command_vx_yaw_clearance_roll_pitch")
            assert command_observed[0] == pytest.approx(
                env.decision.motion_command.forward_velocity_mps / 0.6
            )
            assert command_observed[1] == pytest.approx(
                env.decision.motion_command.yaw_rate_rps / 0.5
            )
            np.testing.assert_array_equal(
                observation, encode_d1_locomotion_observation(env.decision)
            )
            # Public memories refer to the controller state after the completed
            # compute and before next compute; preparing the observation adds no update.
            current = env.decision.context.proposal.memory
            controller_memory = env.loop.controller.controller.control_memory
            if baseline == "wheel_leg":
                np.testing.assert_array_equal(
                    current.wheel_integral_nm, controller_memory.wheel_integral_nm
                )
            else:
                assert current.distance_m == controller_memory.distance_m
                assert current.distance_reference_m == controller_memory.distance_reference_m
                expected_distance = previous.context.proposal.memory.distance_m + (
                    previous.context.state.base_linear_velocity_body[0] * env.plant.control_dt
                )
                assert current.distance_m == pytest.approx(expected_distance, abs=1e-15)
    finally:
        env.close()


def test_external_command_callback_is_sampled_once_per_prepared_tick():
    queried = []

    def command_source(time):
        queried.append(time)
        return D1MotionCommand(0.2 + time, clearance_m=0.455)

    env = D1LocomotionEnv(episode_seconds=0.05, command_source=command_source)
    try:
        env.reset(seed=0)
        for _ in range(4):
            pending = env.decision
            for _ in range(3):
                assert env.loop.prepare(pending.motion_command) is pending
                encode_d1_locomotion_observation(pending)
                env.loop.provider.read()
            _, _, terminated, truncated, info = env.step(_zero(env))
            assert not terminated and not truncated
            assert info["command"] == asdict(pending.motion_command)
        np.testing.assert_array_equal(queried, np.arange(5) * env.plant.control_dt)
    finally:
        env.close()


def test_time_limit_keeps_terminal_next_observation_without_terminal_penalty():
    env = D1LocomotionEnv(episode_seconds=0.03, command_mode="development")
    try:
        with pytest.raises(RuntimeError, match="reset"):
            env.step(_zero(env))
        env.reset(seed=9)
        for tick in range(3):
            observation, _, terminated, truncated, info = env.step(_zero(env))
            assert not terminated
            assert truncated == (tick == 2)
        assert info["terminal_reason"] == "time_limit"
        assert info["reward_terms"]["termination"] == 0.0
        assert env.decision.context.proposal.tick == 3
        assert env.decision.motion_command == env.schedule.cmd_at(0.03)
        np.testing.assert_array_equal(observation, encode_d1_locomotion_observation(env.decision))
        with pytest.raises(RuntimeError, match="reset"):
            env.step(_zero(env))
        env.reset(seed=9)
        assert not any(env.step(_zero(env))[2:4])
    finally:
        env.close()


@pytest.mark.parametrize("source", ("truth_impairment", "imu_encoder_fusion"))
def test_seeded_reset_rebuilds_domain_noise_delay_and_controller_memory(source):
    env = D1LocomotionEnv(
        episode_seconds=0.05,
        provider_config=_provider(source),
        randomization=D1LocomotionRandomization(
            base_mass_scale=(0.9, 1.1),
            damping_scale=(0.8, 1.2),
            friction_scale=(0.8, 1.1),
            actuator_strength_scale=(0.85, 1.05),
            measurement_delay_steps=(1, 2),
            actuator_delay_steps=(1, 3),
            actuator_time_constant_s=(0.004, 0.008),
            actuator_gain=(0.9, 1.1),
        ),
    )
    try:
        paths = []
        for extra_reads in (False, True):
            observation, info = env.reset(seed=1234)
            path = [observation.tobytes(), info["episode_metadata"]]
            for _ in range(4):
                if extra_reads:
                    for _ in range(4):
                        env.loop.provider.read()
                        env.loop.provider.ground_reference()
                        encode_d1_locomotion_observation(env.decision)
                        env.schedule.cmd_at(env.decision.context.proposal.control_time_s)
                observation, reward, terminated, truncated, info = env.step(_zero(env))
                assert not terminated and not truncated
                path.extend(
                    (
                        observation.tobytes(),
                        reward,
                        info["metrics"],
                        env.last_transition.requested_torque_nm.tobytes(),
                    )
                )
            paths.append(path)
        assert paths[0] == paths[1]
        env.reset(seed=1235)
        assert env.episode_metadata["domain"] != paths[0][1]["domain"]
        copied = env.episode_metadata
        copied["domain"]["base_mass_scale"] = 99.0
        assert env.episode_metadata["domain"]["base_mass_scale"] != 99.0
    finally:
        env.close()


def test_explicit_schedule_override_and_invalid_options():
    env = D1LocomotionEnv(episode_seconds=0.03)
    try:
        with pytest.raises(ValueError, match="option"):
            env.reset(options={"unknown": 1})
        with pytest.raises(ValueError, match="duration"):
            env.reset(
                options={"schedule": D1CommandSchedule((D1CommandSegment(1, D1MotionCommand()),))}
            )
        command = D1MotionCommand(0.13)
        schedule = D1CommandSchedule((D1CommandSegment(0.03, command, 0),))
        _, info = env.reset(seed=1, options={"schedule": schedule})
        assert env.schedule is schedule
        assert env.decision.motion_command == command
        assert info["episode_metadata"]["schedule"] == schedule_to_dict(schedule)
    finally:
        env.close()


@pytest.mark.parametrize(
    "target", ((5.31, 0), (-5.31, 0), (0, 2.31), (0, -2.31), (6.1, 0), (0, -3.1))
)
def test_physical_map_exit_terminates_and_collision_exits_are_explicitly_clamped(target):
    env = D1LocomotionEnv(episode_seconds=0.05)
    try:
        env.reset(seed=2)
        # Inject a physical out-of-bounds placement at t=0, retaining the same
        # nominal pose. This tests termination plumbing, not achievable motion.
        env.plant.reset(base_position=np.asarray((*target, 0.455)))
        observation, _, terminated, truncated, info = env.step(_zero(env))
        assert terminated and not truncated
        assert info["terminal_reason"] == "map_boundary"
        assert info["reward_terms"]["termination"] == -env.reward_config.termination_cost
        assert info["ground_query_clamped"] == (abs(target[0]) > 6 or abs(target[1]) > 3)
        assert np.isfinite(observation).all()
        assert env.decision.context.proposal.tick == 1
        with pytest.raises(RuntimeError):
            env.step(_zero(env))
    finally:
        env.close()


@pytest.mark.parametrize("poison", ("terrain_oracle", "logged_truth"))
def test_fusion_actor_is_invariant_to_reward_only_truth_pollution(monkeypatch, poison):
    env = D1LocomotionEnv(episode_seconds=0.05, provider_config=_provider("imu_encoder_fusion"))

    def sample():
        observation, _ = env.reset(seed=321)
        actor = [observation.tobytes()]
        reward = []
        for _ in range(3):
            observation, scalar, terminated, truncated, _ = env.step(_zero(env))
            assert not terminated and not truncated
            actor.extend((observation.tobytes(), env.last_transition.requested_torque_nm.tobytes()))
            reward.append(scalar)
        return actor, reward

    try:
        clean_actor, clean_reward = sample()
        if poison == "terrain_oracle":
            original = D1Plant.locomotion_ground_reference

            def altered_ground(plant, x, y):
                ground = original(plant, x, y)
                return TrainingGroundReference(
                    ground.height_m + 0.03, ground.pitch_rad + 0.02, ground.roll_rad - 0.01
                )

            monkeypatch.setattr(D1Plant, "locomotion_ground_reference", altered_ground)
        else:
            original = D1MujocoTruthStateSource.read

            def altered_truth(source):
                state = original(source)
                offset = np.asarray((0.0, 0.0, 0.05))
                return replace(
                    state,
                    base_position=state.base_position + offset,
                    foot_position=state.foot_position + offset,
                )

            monkeypatch.setattr(D1MujocoTruthStateSource, "read", altered_truth)
        # Neither branch modifies MjData, sensor collection, encoder positions,
        # IMU measurements or provider state. Only reward/logging truth changes.
        poisoned_actor, poisoned_reward = sample()
        assert poisoned_actor == clean_actor
        assert poisoned_reward != clean_reward
    finally:
        env.close()


def test_estimated_map_exit_marks_clamped_publication_even_if_physics_stays_inside(monkeypatch):
    env = D1LocomotionEnv(episode_seconds=0.05, provider_config=_provider("truth_impairment"))
    try:
        env.reset(seed=2)
        original = D1NoisyDelayedStateSource.read

        def displaced_estimate(source):
            state = original(source)
            offset = np.asarray((20.0, 0.0, 0.0))
            return replace(
                state,
                base_position=state.base_position + offset,
                foot_position=state.foot_position + offset,
            )

        monkeypatch.setattr(D1NoisyDelayedStateSource, "read", displaced_estimate)
        observation, _, terminated, truncated, info = env.step(_zero(env))
        assert terminated and not truncated
        assert info["terminal_reason"] == "ground_query_outside_map"
        assert info["ground_query_clamped"]
        assert abs(info["metrics"]["x_m"]) < 5.3
        assert env.decision.context.state.base_position[0] > 6.0
        assert np.isfinite(observation).all()
    finally:
        env.close()


def test_exposure_uses_real_geometry_not_the_schedule_label():
    env = D1LocomotionEnv(episode_seconds=0.03, terrain=locomotion_terrain_configs("train")[0])
    schedule = D1CommandSchedule(
        (D1CommandSegment(0.03, D1MotionCommand(0.1), 0),),
        label="all-terrain-already-crossed",
    )
    try:
        env.reset(seed=2, options={"schedule": schedule})
        for _ in range(2):
            _, _, _, _, info = env.step(_zero(env))
            assert info["terrain_exposure"]["nonflat_steps"] == 0
            assert info["terrain_exposure"]["nonflat_path_m"] == 0.0
        env.reset(seed=2, options={"schedule": schedule})
        x, y = -1.713, 0.217
        ground = env.plant.locomotion_ground_reference(x, y)
        env.plant.reset(base_position=np.asarray((x, y, 0.455 + ground.height_m)))
        _, _, _, _, info = env.step(_zero(env))
        assert info["terrain_exposure"]["nonflat_now"]
        assert info["terrain_exposure"]["nonflat_steps"] == 1
        assert "not contact or traversal proof" in info["terrain_exposure"]["definition"]
    finally:
        env.close()


def test_invalid_action_does_not_advance_physics_but_requires_reset():
    env = D1LocomotionEnv(episode_seconds=0.03)
    try:
        env.reset(seed=0)
        before = env.plant.data.time
        with pytest.raises(ValueError):
            env.step(np.full(env.action_space.shape, np.nan))
        assert env.plant.data.time == before
        with pytest.raises(RuntimeError, match="reset"):
            env.step(_zero(env))
        env.reset(seed=0)
        env.step(_zero(env))
    finally:
        env.close()


def test_standard_gym_contract():
    env = D1LocomotionEnv(episode_seconds=0.05)
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()
