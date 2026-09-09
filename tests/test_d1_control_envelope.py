"""A finite controller-domain exit is a terminal event, not a valid next preview."""

from dataclasses import replace

import numpy as np
import pytest

from wheel_legged_control.d1 import wheel_leg_controller
from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv
from wheel_legged_control.d1.locomotion_observation import encode_d1_locomotion_observation
from wheel_legged_control.d1.locomotion_rewards import d1_locomotion_reward_terms
from wheel_legged_control.d1.state_provider import D1StateProviderConfig
from wheel_legged_control.d1.training_terrain import TrainingGroundReference


def zero(env):
    return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)


def reject_next_ground(env, monkeypatch):
    original = env.loop.provider.ground_reference

    def changed_ground():
        ground = original()
        return replace(ground, roll_rad=0.31)

    monkeypatch.setattr(env.loop.provider, "ground_reference", changed_ground)


def test_post_step_finite_ground_exit_returns_true_terminal_cached_decision(monkeypatch):
    env = D1LocomotionEnv(episode_seconds=0.03)
    try:
        initial, _ = env.reset(seed=2)
        executed = env.decision
        reject_next_ground(env, monkeypatch)
        calls = {"physics": 0, "measurement": 0}
        physical_step = env.plant.step
        measurement = env.loop.provider._source.read

        def count_physics(*args, **kwargs):
            calls["physics"] += 1
            return physical_step(*args, **kwargs)

        def count_measurement(*args, **kwargs):
            calls["measurement"] += 1
            return measurement(*args, **kwargs)

        monkeypatch.setattr(env.plant, "step", count_physics)
        monkeypatch.setattr(env.loop.provider._source, "read", count_measurement)
        observation, reward, terminated, truncated, info = env.step(zero(env))
        assert terminated and not truncated
        assert info["terminal_reason"] == "control_reference_out_of_envelope"
        assert info["terminal_observation_source"] == "last_executable_decision"
        assert "attitude commands within .3 rad" in info["control_reference_rejection"]
        assert info["reward_terms"]["termination"] == -env.reward_config.termination_cost
        assert reward == sum(info["reward_terms"].values())
        assert env.last_transition.decision is executed
        assert env.decision is executed
        np.testing.assert_array_equal(observation, initial)
        np.testing.assert_array_equal(observation, encode_d1_locomotion_observation(executed))
        assert env.plant.data.time == pytest.approx(0.01)
        assert env.loop.provider.read().sequence == 1
        assert env._steps == 1
        assert calls == {"physics": 1, "measurement": 1}
        assert not env._active
        with pytest.raises(RuntimeError, match="reset"):
            env.step(zero(env))
        env.reset(seed=2)
        assert not any(env.step(zero(env))[2:4])
    finally:
        env.close()


@pytest.mark.parametrize("source", ("oracle", "imu_encoder_fusion"))
def test_normal_trajectory_matches_the_original_valid_step_order_byte_exactly(source):
    config = (
        D1StateProviderConfig(source)
        if source == "oracle"
        else D1StateProviderConfig(
            source,
            initial_position_m=(-3.8, 0.0, 0.455),
            initial_rpy_rad=(0.0, 0.0, 0.0),
            sensor_delay_steps=2,
        )
    )
    env = D1LocomotionEnv(episode_seconds=0.20, provider_config=config)
    reference = D1LocomotionEnv(episode_seconds=0.20, provider_config=config)
    try:
        observation, _ = env.reset(seed=443)
        expected_observation, _ = reference.reset(seed=443)
        assert observation.tobytes() == expected_observation.tobytes()
        for tick in range(10):
            action = np.sin(np.arange(8) + tick) * 0.01
            # The pre-fix successful order: physics -> metrics/reward -> next
            # preview. Compare in-process, avoiding MuJoCo-version-specific
            # golden hashes in the portable test suite.
            transition = reference.loop.step(action)
            reference._steps += 1
            ground = reference._bounded_ground_query(*transition.truth.base_position[:2])
            expected_metrics = reference._metrics(transition, ground)
            expected_exposure = reference._exposure(transition, ground)
            expected_terms = d1_locomotion_reward_terms(
                transition,
                ground,
                reference.plant.last_control_interval_actuator_traces,
                terminated=False,
                config=reference.reward_config,
            )
            expected_observation = reference._prepare()
            observation, reward, terminated, truncated, info = env.step(action)
            assert not terminated and not truncated
            assert observation.tobytes() == expected_observation.tobytes()
            assert env.plant.data.qpos.tobytes() == reference.plant.data.qpos.tobytes()
            assert (
                env.last_transition.requested_torque_nm.tobytes()
                == transition.requested_torque_nm.tobytes()
            )
            assert reward == sum(expected_terms.values())
            assert info["reward_terms"] == expected_terms
            assert info["metrics"] == expected_metrics
            assert info["terrain_exposure"] == expected_exposure
            assert "terminal_observation_source" not in info
    finally:
        env.close()
        reference.close()


@pytest.mark.parametrize(
    "command",
    (
        D1Command(base_height_m=0.31),
        D1Command(base_height_m=0.57),
        D1Command(roll_rad=0.31),
        D1Command(pitch_rad=-0.31),
    ),
)
def test_only_finite_envelope_violations_have_the_dedicated_type(command):
    controller = wheel_leg_controller.D1WheelLegController()
    with pytest.raises(wheel_leg_controller.D1ControlEnvelopeError):
        controller._extension(command, 0.0)


@pytest.mark.parametrize(
    "command,ground",
    (
        (D1Command(roll_rad=float("nan")), 0.0),
        (D1Command(base_vertical_velocity_mps=0.1), 0.0),
        (D1Command(), float("inf")),
        (D1Command(base_height_m=1e308), -1e308),
    ),
)
def test_nonfinite_and_unsupported_commands_keep_plain_value_errors(command, ground):
    controller = wheel_leg_controller.D1WheelLegController()
    with pytest.raises(ValueError) as caught:
        controller._extension(command, ground)
    assert type(caught.value) is ValueError


@pytest.mark.parametrize("failure", ("ordinary_value_error", "nonfinite_ground"))
def test_next_preview_numerical_and_provider_errors_are_not_terminals(monkeypatch, failure):
    env = D1LocomotionEnv(episode_seconds=0.03)
    try:
        env.reset(seed=2)

        def bad_ground():
            if failure == "ordinary_value_error":
                raise ValueError("broken provider invariant")
            return TrainingGroundReference(0.0, float("nan"), 0.0)

        monkeypatch.setattr(env.loop.provider, "ground_reference", bad_ground)
        with pytest.raises(ValueError) as caught:
            env.step(zero(env))
        assert not isinstance(caught.value, wheel_leg_controller.D1ControlEnvelopeError)
        assert env.plant.data.time == pytest.approx(0.01)
        assert not env._active
    finally:
        env.close()


def test_nonfinite_measurement_state_still_raises_after_physics(monkeypatch):
    env = D1LocomotionEnv(episode_seconds=0.03)
    try:
        env.reset(seed=2)
        original = env.loop.provider._source.read

        def nonfinite_measurement():
            return replace(original(), base_rotation=np.full((3, 3), np.nan))

        monkeypatch.setattr(env.loop.provider._source, "read", nonfinite_measurement)
        with pytest.raises(ValueError) as caught:
            env.step(zero(env))
        assert not isinstance(caught.value, wheel_leg_controller.D1ControlEnvelopeError)
        assert env.plant.data.time == pytest.approx(0.01)
        assert not env._active
    finally:
        env.close()


@pytest.mark.parametrize("history", (1, 4))
def test_vectorized_terminal_history_disables_time_limit_bootstrap(monkeypatch, history):
    from stable_baselines3.common.vec_env import DummyVecEnv

    from wheel_legged_control.d1.observation_history import D1ObservationHistory

    base = D1LocomotionEnv(episode_seconds=0.03)
    vectorized = DummyVecEnv([lambda: D1ObservationHistory(base, history_length=history)])
    try:
        vectorized.seed(2)
        initial = vectorized.reset()
        reject_next_ground(base, monkeypatch)
        observation, _, dones, infos = vectorized.step(zero(base)[None, :])
        assert bool(dones[0])
        # SB3 only bootstraps terminal_observation when this flag is True.
        assert infos[0]["TimeLimit.truncated"] is False
        assert infos[0]["terminal_reason"] == "control_reference_out_of_envelope"
        assert infos[0]["terminal_observation_source"] == "last_executable_decision"
        np.testing.assert_array_equal(infos[0]["terminal_observation"], initial[0])
        assert observation.shape == (1, 82 * history)
        # DummyVecEnv reset already installed a new provider and valid decision.
        assert base._active
        assert not bool(vectorized.step(zero(base)[None, :])[2][0])
    finally:
        vectorized.close()


def test_envelope_error_from_unexecuted_compute_is_not_a_terminal(monkeypatch):
    env = D1LocomotionEnv(episode_seconds=0.03)
    try:
        env.reset(seed=2)

        def fail_compute(*args, **kwargs):
            raise wheel_leg_controller.D1ControlEnvelopeError("cannot execute existing decision")

        monkeypatch.setattr(env.loop.controller, "compute", fail_compute)
        with pytest.raises(wheel_leg_controller.D1ControlEnvelopeError):
            env.step(zero(env))
        assert env.plant.data.time == 0.0
        assert env._steps == 0
        assert not env._active
    finally:
        env.close()


@pytest.mark.parametrize(
    "reason", ("fall_or_body_contact", "map_boundary", "ground_query_outside_map")
)
def test_existing_safety_termination_has_priority_over_preview_rejection(monkeypatch, reason):
    env = D1LocomotionEnv(episode_seconds=0.01)
    try:
        env.reset(seed=2)
        if reason == "map_boundary":
            env.plant.reset(base_position=np.asarray((5.31, 0.0, 0.455)))
        elif reason == "fall_or_body_contact":
            original = env._metrics

            def contact_metrics(*args):
                return {**original(*args), "undesired_ground_contacts": 1}

            monkeypatch.setattr(env, "_metrics", contact_metrics)
        else:
            original = env._exposure

            def clamped_exposure(*args):
                result = original(*args)
                env._ground_query_clamped = True
                return result

            monkeypatch.setattr(env, "_exposure", clamped_exposure)
        reject_next_ground(env, monkeypatch)
        _, _, terminated, truncated, info = env.step(zero(env))
        assert terminated and not truncated
        assert info["terminal_reason"] == reason
        assert info["terminal_observation_source"] == "last_executable_decision"
        assert info["reward_terms"]["termination"] == -env.reward_config.termination_cost
    finally:
        env.close()


def test_envelope_exit_at_time_limit_is_terminated_not_bootstrapped(monkeypatch):
    env = D1LocomotionEnv(episode_seconds=0.01)
    try:
        env.reset(seed=2)
        reject_next_ground(env, monkeypatch)
        _, _, terminated, truncated, info = env.step(zero(env))
        assert terminated and not truncated
        assert info["terminal_reason"] == "control_reference_out_of_envelope"
    finally:
        env.close()


def test_invalid_user_command_at_reset_remains_an_error():
    env = D1LocomotionEnv(command_source=lambda _: D1MotionCommand(clearance_m=0.57))
    try:
        with pytest.raises(ValueError, match="clearance range"):
            env.reset(seed=2)
        assert not env._active
    finally:
        env.close()


def test_reset_reference_out_of_envelope_is_not_swallowed(monkeypatch):
    from wheel_legged_control.d1.state_provider import D1StateProvider

    monkeypatch.setattr(
        D1StateProvider, "ground_reference", lambda _: TrainingGroundReference(0.0, 0.0, 0.31)
    )
    env = D1LocomotionEnv(episode_seconds=0.03)
    try:
        with pytest.raises(wheel_leg_controller.D1ControlEnvelopeError):
            env.reset(seed=2)
        assert env.plant.data.time == 0.0
        assert not env._active
    finally:
        env.close()


@pytest.mark.parametrize("action", (np.full(8, np.nan), np.zeros(7)))
def test_invalid_action_still_raises_before_physics(action):
    env = D1LocomotionEnv(episode_seconds=0.03)
    try:
        env.reset(seed=2)
        with pytest.raises(ValueError, match="finite action schema"):
            env.step(action)
        assert env.plant.data.time == 0.0
        assert not env._active
    finally:
        env.close()
