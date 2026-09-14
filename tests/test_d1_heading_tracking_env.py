"""Physical transition parity, command timing, terminal handling and real PPO use."""
import json
import math
from copy import deepcopy

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from scripts.d1_heading_reference import wrap_angle
from scripts.d1_heading_tracking_env import D1HeadingTrackingEnv, load_heading_policy
from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.locomotion_checkpoint import write_checkpoint_metadata
from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv
from wheel_legged_control.d1.locomotion_observation import encode_d1_locomotion_observation
from wheel_legged_control.d1.state_estimation import D1EstimatorImpairments
from wheel_legged_control.d1.state_provider import D1StateProviderConfig
from wheel_legged_control.d1.wheel_leg_controller import D1ControlEnvelopeError


def command(time_s):
    return D1MotionCommand(
        forward_velocity_mps=.25,
        yaw_rate_rps=.2 if time_s < .03 else -.1 if time_s < .06 else 0.,
    )


def test_real_servo_equivalence_preserves_every_physics_state_and_82_prefix():
    heading = D1HeadingTrackingEnv(episode_seconds=.12, command_source=command)
    base = D1LocomotionEnv(
        episode_seconds=.12,
        command_source=lambda _t: heading.heading_decision.servo_command,
    )
    rng = np.random.default_rng(7)
    try:
        obs, info = heading.reset(seed=8)
        original, _ = base.reset(seed=8)
        assert info['episode_metadata']['heading_task_config'] == heading.heading_task_config
        for tick in range(13):
            assert obs.shape == (85,) and obs.dtype == np.float32
            assert heading.observation_space.contains(obs)
            np.testing.assert_array_equal(obs[:82], original)
            np.testing.assert_array_equal(heading.plant.data.qpos, base.plant.data.qpos)
            np.testing.assert_array_equal(heading.plant.data.qvel, base.plant.data.qvel)
            assert heading.plant.data.time == base.plant.data.time
            if tick == 12:
                break
            action = rng.uniform(-.05, .05, 8)
            obs, reward, done, timeout, info = heading.step(action)
            original, servo_reward, old_done, old_timeout, old_info = base.step(action)
            assert (done, timeout) == (old_done, old_timeout)
            assert info['servo_reward_terms'] == old_info['reward_terms']
            assert info['servo_reward'] == servo_reward
            assert reward == sum(info['reward_terms'].values())
            assert reward <= servo_reward
        assert timeout and not done
        assert info['heading_task']['bootstrap_next_observation']
    finally:
        heading.close()
        base.close()


def test_command_change_left_hold_prepare_cache_and_reset_atomicity():
    calls = []

    def source(time_s):
        calls.append(time_s)
        return command(time_s)

    env = D1HeadingTrackingEnv(episode_seconds=.12, command_source=source)
    try:
        obs, _ = env.reset(seed=9)
        initial = env.heading_decision
        env._prepare()
        env._prepare()
        assert calls == [0.]
        expected = 0.
        for tick in range(8):
            before = env.heading_decision
            expected = wrap_angle(expected + command(tick*.01).yaw_rate_rps*.01)
            obs, _, _, _, info = env.step(np.zeros(8))
            details = info['heading_task']
            assert details['reference_heading_after'] == pytest.approx(expected, abs=1e-14)
            assert env.heading_decision.reference_heading_rad == pytest.approx(expected)
            assert details['user_command_before']['yaw_rate_rps'] == before.user_yaw_rate_rps
            assert obs[84] == pytest.approx(command((tick+1)*.01).yaw_rate_rps/.5)
            env._prepare()
        assert calls == [tick*.01 for tick in range(9)]
        state = env._reference.state
        context = env.heading_decision
        qpos = env.plant.data.qpos.copy()
        random_state = deepcopy(env.np_random.bit_generator.state)
        with pytest.raises(ValueError, match='only an explicit schedule'):
            env.reset(seed=123, options={'bad': True})
        assert env._reference.state is state and env.heading_decision is context
        assert env.np_random.bit_generator.state == random_state
        np.testing.assert_array_equal(qpos, env.plant.data.qpos)
        obs, _ = env.reset(seed=9)
        assert env.heading_decision.loop is not initial.loop
        assert env.heading_decision.reference_heading_rad == initial.reference_heading_rad
        np.testing.assert_array_equal(obs[-3:], np.array([0., 1., .4], dtype=np.float32))
    finally:
        env.close()


def test_appended_heading_uses_published_noisy_measurement_reward_uses_endpoint_truth():
    env = D1HeadingTrackingEnv(
        episode_seconds=.1, command_source=command,
        provider_config=D1StateProviderConfig(
            'truth_impairment', impairments=D1EstimatorImpairments(base_rotation_std_rad=.02),
        ),
    )
    try:
        obs, _ = env.reset(seed=10)
        measured_initial = env.loop.provider.read().base_rpy[2]
        assert env.heading_decision.reference_heading_rad == pytest.approx(measured_initial)
        for _ in range(5):
            obs, reward, _, _, info = env.step(np.zeros(8))
            current = env.heading_decision
            measured = env.loop.provider.read().base_rpy[2]
            error = wrap_angle(current.reference_heading_rad - measured)
            np.testing.assert_allclose(obs[-3:-1], [math.sin(error), math.cos(error)], atol=1e-7)
            np.testing.assert_array_equal(obs[:82], encode_d1_locomotion_observation(env.decision))
            details = info['heading_task']
            truth_error = wrap_angle(
                env.last_transition.truth.base_rpy[2] - details['reference_heading_after']
            )
            assert abs(measured - env.last_transition.truth.base_rpy[2]) > 1e-6
            assert details['heading_error_after'] == pytest.approx(truth_error)
            dt = env.last_transition.receipt.end_time_s-env.last_transition.receipt.start_time_s
            penalty = dt*.5*(math.exp(-(truth_error/math.radians(5))**2)-1.)
            assert reward-info['servo_reward'] == pytest.approx(penalty, abs=1e-15)
            assert -dt*.5 <= penalty <= 0.
            assert info['heading_task']['source_schema'] == env.source_schema
    finally:
        env.close()


def test_reward_weight_changes_training_signal_without_changing_physics_or_observation():
    envs = [D1HeadingTrackingEnv(
        episode_seconds=.1, command_source=command, heading_reward_weight=weight,
    ) for weight in (0., .5)]
    try:
        initial = [env.reset(seed=4)[0] for env in envs]
        np.testing.assert_array_equal(*initial)
        penalties = []
        for _ in range(10):
            zero, weighted = [env.step(np.full(8, .02)) for env in envs]
            np.testing.assert_array_equal(zero[0], weighted[0])
            np.testing.assert_array_equal(envs[0].plant.data.qpos, envs[1].plant.data.qpos)
            assert zero[4]['reward_terms']['heading_goal'] == 0.
            penalty = weighted[4]['reward_terms']['heading_goal']
            assert weighted[1]-zero[1] == pytest.approx(penalty, abs=1e-15)
            penalties.append(penalty)
        assert min(penalties) < -1e-7
    finally:
        for env in envs:
            env.close()


def test_terminal_rejection_returns_old_heading_context_after_actual_physics(monkeypatch):
    env = D1HeadingTrackingEnv(episode_seconds=.1, command_source=command)
    try:
        obs, _ = env.reset(seed=5)
        old = obs.copy()

        def reject(_command):
            raise D1ControlEnvelopeError('finite reference outside test envelope')

        monkeypatch.setattr(env.loop, 'prepare', reject)
        obs, reward, done, timeout, info = env.step(np.zeros(8))
        assert done and not timeout and math.isfinite(reward)
        assert env.plant.data.time == pytest.approx(.01)
        assert env._steps == 1
        np.testing.assert_array_equal(obs, old)
        assert info['terminal_observation_source'] == 'last_executable_decision'
        assert info['heading_task']['appended_observation_decision_tick'] == 0
        assert info['heading_task']['reference_heading_after'] == pytest.approx(.002)
        assert not info['heading_task']['bootstrap_next_observation']
        with pytest.raises(RuntimeError, match='reset'):
            env.step(np.zeros(8))
    finally:
        env.close()


def test_invalid_action_and_numerical_error_do_not_fabricate_transitions(monkeypatch):
    env = D1HeadingTrackingEnv(episode_seconds=.1)
    try:
        env.reset(seed=6)
        for action in (np.zeros(7), np.full(8, np.nan)):
            env.reset(seed=6)
            with pytest.raises(ValueError, match='action'):
                env.step(action)
            assert env._steps == 0 and env.plant.data.time == 0.
            assert env.last_transition is None
        env.reset(seed=6)

        def fail(_action):
            raise FloatingPointError('injected numerical failure')

        monkeypatch.setattr(env.loop, 'step', fail)
        with pytest.raises(FloatingPointError):
            env.step(np.zeros(8))
        assert env.last_transition is None and env._steps == 0
    finally:
        env.close()


@pytest.mark.parametrize('keyword,value', [
    ('heading_reward_weight', True), ('heading_reward_weight', np.bool_(False)),
    ('heading_reward_weight', -1.), ('heading_reward_weight', 1.01),
    ('heading_reward_weight', float('nan')), ('heading_reward_weight', '0.5'),
    ('heading_sigma_rad', False), ('heading_sigma_rad', 0.),
    ('heading_sigma_rad', 1e-4), ('heading_sigma_rad', 3.2),
    ('heading_sigma_rad', float('inf')), ('baseline', 'lqr'), ('action_mode', 'shared2'),
])
def test_invalid_task_configuration_rejected(keyword, value):
    with pytest.raises((TypeError, ValueError)):
        D1HeadingTrackingEnv(**{keyword: value})


def test_public_task_parameters_cannot_drift_from_checkpoint_identity():
    env = D1HeadingTrackingEnv(episode_seconds=.02)
    try:
        env.reset(seed=17)
        original = env.heading_task_config
        for name in ('heading_reward_weight', 'heading_sigma_rad', 'heading_kp',
                     'heading_kd', 'heading_limit_rps'):
            with pytest.raises(AttributeError):
                setattr(env, name, 0.)
        copy = env.heading_task_config
        copy['outer_loop']['kp'] = 99.
        copy['reward']['heading_reward_weight'] = 0.
        assert env.heading_task_config == original
        assert env.episode_metadata['heading_task_config'] == original
        assert env.heading_reward_weight == .5 and env.heading_kp == 2.
    finally:
        env.close()


def test_actual_ppo_update_save_reload_and_execution_in_new_85_task(tmp_path):
    torch.set_num_threads(1)
    train = D1HeadingTrackingEnv(episode_seconds=.08, command_source=command)
    target = D1HeadingTrackingEnv(episode_seconds=.12, command_source=command)
    try:
        model = PPO('MlpPolicy', train, n_steps=8, batch_size=8, n_epochs=1, seed=17, device='cpu')
        before = {key: value.clone() for key, value in model.policy.state_dict().items()}
        model.learn(total_timesteps=16)
        assert model.num_timesteps == 16
        assert any(not torch.equal(before[k], v) for k, v in model.policy.state_dict().items())
        model_path, metadata_path = tmp_path/'model.zip', tmp_path/'metadata.json'
        model.save(model_path)
        metadata = write_checkpoint_metadata(model_path, train, metadata_path)
        assert metadata['observation_dim'] == 85
        assert metadata['recorded_episode']['heading_task_config'] == train.heading_task_config
        obs, _ = target.reset(seed=999)
        loaded = load_heading_policy(model_path, metadata_path, target)
        expected, _ = model.predict(obs, deterministic=True)
        actual, _ = loaded.predict(obs, deterministic=True)
        np.testing.assert_array_equal(actual, expected)
        result = target.step(actual)
        assert result[0].shape == (85,) and target._steps == 1
    finally:
        train.close()
        target.close()


@pytest.mark.parametrize('mismatch', ['old82', 'kp', 'weight', 'normalization', 'timing'])
def test_incompatible_task_rejected_before_model_deserialization(tmp_path, mismatch):
    source = D1LocomotionEnv(episode_seconds=.02) if mismatch == 'old82' else D1HeadingTrackingEnv(episode_seconds=.02)
    target = D1HeadingTrackingEnv(episode_seconds=.02)
    model, sidecar = tmp_path/'invalid.zip', tmp_path/'metadata.json'
    model.write_bytes(b'not a serialized policy: metadata must reject before loading')
    try:
        source.reset(seed=2)
        target.reset(seed=3)
        metadata = write_checkpoint_metadata(model, source, sidecar)
        if mismatch != 'old82':
            config = metadata['recorded_episode']['heading_task_config']
            if mismatch == 'kp':
                config['outer_loop']['kp'] = 3.
            elif mismatch == 'weight':
                config['reward']['heading_reward_weight'] = .2
            elif mismatch == 'normalization':
                config['observation']['user_yaw_rate_scale_rps'] = 1.
            else:
                config['timing_schema'] = 'right-hold'
            sidecar.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match='heading_task_config'):
            load_heading_policy(model, sidecar, target)
    finally:
        source.close()
        target.close()
