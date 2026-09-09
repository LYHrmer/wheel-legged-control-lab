"""Short observation history semantics and one real, small PPO update."""

import json

import gymnasium as gym
import numpy as np
import pytest

from wheel_legged_control.d1.observation_history import D1ObservationHistory


class ReusedBufferEnv(gym.Env):
    """Deterministic proprioception-shaped fixture, not a robot simulator."""

    observation_schema = "fixture-proprio-two-v1"
    source_schema = "fixture-counter-source-v1"

    def __init__(self, dtype=np.float32, termination=False):
        self.observation_space = gym.spaces.Box(-100, 100, (2,), dtype=dtype)
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=dtype)
        self.buffer = np.zeros(2, dtype=dtype)
        self.steps = 0
        self.reset_count = 0
        self.termination = termination
        self.bad_observation = None
        self.raise_step = False
        self.raise_reset = False
        self.last_info = {}

    @property
    def plant(self):
        raise AssertionError("history must not read plant")

    @property
    def data(self):
        raise AssertionError("history must not read hidden simulator data")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.raise_reset:
            raise RuntimeError("synthetic reset failure")
        self.steps = 0
        self.reset_count += 1
        self.last_seed, self.last_options = seed, options
        self.buffer[:] = (options or {}).get("initial", (0.0, 0.0))
        self.last_info = {"reset_count": self.reset_count}
        return (
            self.buffer if self.bad_observation is None else self.bad_observation,
            self.last_info,
        )

    def step(self, action):
        self.steps += 1
        if self.raise_step:
            raise RuntimeError("synthetic transition failure")
        self.last_action = np.asarray(action).copy()
        self.buffer[:] = (self.steps, action[0])
        self.last_info = {"step": self.steps}
        done = self.steps == 3
        return (
            self.buffer if self.bad_observation is None else self.bad_observation,
            float(1.0 - np.square(action).sum()),
            done and self.termination,
            done and not self.termination,
            self.last_info,
        )


def test_reset_repeats_initial_frame_and_step_shifts_oldest_to_newest():
    base = ReusedBufferEnv()
    env = D1ObservationHistory(base, 3)
    options = {"initial": (7.0, 8.0)}
    observation, info = env.reset(seed=123, options=options)
    np.testing.assert_array_equal(observation, (7, 8, 7, 8, 7, 8))
    assert base.last_seed == 123 and base.last_options is options
    assert info is base.last_info
    observation, reward, terminated, truncated, info = env.step(np.asarray((0.25,), np.float32))
    np.testing.assert_array_equal(observation, (7, 8, 7, 8, 1, 0.25))
    assert reward == 0.9375 and not terminated and not truncated
    assert info is base.last_info
    observation, *_ = env.step(np.asarray((-0.5,), np.float32))
    np.testing.assert_array_equal(observation, (7, 8, 1, 0.25, 2, -0.5))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_history_one_is_stepwise_identical_to_base_env(dtype):
    base, reference = ReusedBufferEnv(dtype), ReusedBufferEnv(dtype)
    env = D1ObservationHistory(base, 1)
    observed, info = env.reset(seed=7)
    expected, reference_info = reference.reset(seed=7)
    assert observed.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(observed, expected)
    assert info == reference_info
    assert env.observation_schema == base.observation_schema
    for command in (0.125, -0.5, 0.75):
        action = np.asarray((command,), dtype=dtype)
        actual = env.step(action)
        wanted = reference.step(action)
        np.testing.assert_array_equal(actual[0], wanted[0])
        assert actual[1:] == wanted[1:]


def test_returns_and_stored_frames_do_not_alias_env_buffer_or_earlier_outputs():
    base = ReusedBufferEnv()
    env = D1ObservationHistory(base, 3)
    first, _ = env.reset(options={"initial": (7.0, 8.0)})
    first[:] = -99
    second, *_ = env.step(np.asarray((0.25,), np.float32))
    second_snapshot = second.copy()
    assert not np.shares_memory(first, second)
    assert not np.shares_memory(base.buffer, second)
    base.buffer[:] = 66
    third, *_ = env.step(np.asarray((-0.5,), np.float32))
    np.testing.assert_array_equal(third, (7, 8, 1, 0.25, 2, -0.5))
    np.testing.assert_array_equal(second, second_snapshot)
    third[:] = 123
    reset, _ = env.reset(options={"initial": (-1.0, -2.0)})
    np.testing.assert_array_equal(reset, (-1, -2, -1, -2, -1, -2))


@pytest.mark.parametrize("termination", (False, True))
def test_no_hidden_reset_and_no_step_before_reset_or_after_done(termination):
    base = ReusedBufferEnv(termination=termination)
    env = D1ObservationHistory(base, 2)
    action = np.zeros(1, dtype=np.float32)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(action)
    assert base.steps == 0 and base.reset_count == 0
    env.reset()
    for _ in range(3):
        final, _, terminated, truncated, _ = env.step(action)
    assert terminated == termination and truncated != termination
    np.testing.assert_array_equal(final, (2, 0, 3, 0))
    assert base.reset_count == 1
    with pytest.raises(RuntimeError, match="reset"):
        env.step(action)
    assert base.steps == 3 and base.reset_count == 1
    reset, _ = env.reset(options={"initial": (9.0, -9.0)})
    np.testing.assert_array_equal(reset, (9, -9, 9, -9))


def test_schema_and_space_describe_history_without_claiming_a_new_sensor():
    base = ReusedBufferEnv(np.float64)
    base.observation_space = gym.spaces.Box(
        np.asarray((-np.inf, -2.0)), np.asarray((np.inf, 7.0)), dtype=np.float64
    )
    env = D1ObservationHistory(gym.wrappers.TimeLimit(base, max_episode_steps=2), 4)
    assert env.observation_space.shape == (8,)
    assert env.observation_space.dtype == np.float64
    np.testing.assert_array_equal(env.observation_space.low, (-np.inf, -2) * 4)
    np.testing.assert_array_equal(env.observation_space.high, (np.inf, 7) * 4)
    assert env.action_space is base.action_space
    assert env.source_schema == base.source_schema
    assert base.observation_schema in env.observation_schema
    assert env.observation_schema != D1ObservationHistory(ReusedBufferEnv(), 3).observation_schema
    description = env.history_metadata
    assert json.loads(json.dumps(description)) == description
    assert description["history_length"] == 4
    assert description["source_schema"] == base.source_schema
    assert description["base_observation_schema"] == base.observation_schema
    description["history_length"] = 200
    assert env.history_metadata["history_length"] == 4
    with pytest.raises(AttributeError):
        env.history_length = 9


@pytest.mark.parametrize("bad", (0, -1, True, np.bool_(False), 1.5, "3"))
def test_invalid_history_lengths_are_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        D1ObservationHistory(ReusedBufferEnv(), bad)


@pytest.mark.parametrize("field", ("observation_space", "action_space"))
@pytest.mark.parametrize(
    "space",
    (
        gym.spaces.Discrete(3),
        gym.spaces.Box(-1, 1, (2, 2), np.float32),
        gym.spaces.Box(0, 10, (2,), np.int32),
    ),
)
def test_non_vector_or_non_float_box_spaces_are_rejected(field, space):
    base = ReusedBufferEnv()
    setattr(base, field, space)
    with pytest.raises((ValueError, TypeError)):
        D1ObservationHistory(base, 2)


@pytest.mark.parametrize("field", ("observation_schema", "source_schema"))
@pytest.mark.parametrize("value", (None, "", 3))
def test_missing_or_invalid_schema_is_not_silently_fabricated(field, value):
    base = ReusedBufferEnv()
    setattr(base, field, value)
    with pytest.raises((ValueError, TypeError, AttributeError)):
        D1ObservationHistory(base, 2)


@pytest.mark.parametrize(
    "bad", (0.0, [], [0.0, 0.0], [[0.0]], [True], [1j], ["0"], [np.inf], [np.nan])
)
def test_invalid_action_does_not_advance_base_or_history(bad):
    base = ReusedBufferEnv()
    env = D1ObservationHistory(base, 2)
    env.reset(options={"initial": (7.0, 8.0)})
    with pytest.raises((TypeError, ValueError)):
        env.step(bad)
    assert base.steps == 0
    observation, *_ = env.step(np.asarray((0.25,), np.float32))
    np.testing.assert_array_equal(observation, (7, 8, 1, 0.25))


def test_action_values_and_dtype_are_not_silently_clipped_or_normalized():
    base = ReusedBufferEnv()
    env = D1ObservationHistory(base, 2)
    env.reset()
    action = np.asarray((2.5,), dtype=np.float64)
    env.step(action)
    assert base.last_action.dtype == action.dtype
    np.testing.assert_array_equal(base.last_action, action)


def test_mixed_boolean_action_is_rejected_before_numpy_coercion():
    base = ReusedBufferEnv()
    base.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
    env = D1ObservationHistory(base)
    env.reset()
    with pytest.raises(TypeError, match="boolean"):
        env.step([0.25, True])
    assert base.steps == 0


@pytest.mark.parametrize("operation", ("reset", "step"))
@pytest.mark.parametrize(
    "bad",
    (
        np.zeros(3, np.float32),
        np.zeros((1, 2), np.float32),
        np.zeros(2, np.float64),
        np.asarray((0.0, np.nan), np.float32),
        np.asarray((0.0, np.inf), np.float32),
        [0.0, 0.0],
    ),
)
def test_invalid_emitted_observation_requires_reset_to_avoid_stale_history(bad, operation):
    base = ReusedBufferEnv()
    env = D1ObservationHistory(base, 2)
    env.reset()
    base.bad_observation = bad
    with pytest.raises((TypeError, ValueError)):
        if operation == "reset":
            env.reset()
        else:
            env.step(np.zeros(1, np.float32))
    expected_steps = int(operation == "step")
    assert base.steps == expected_steps
    base.bad_observation = None
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(1, np.float32))
    assert base.steps == expected_steps
    env.reset()
    observation, *_ = env.step(np.zeros(1, np.float32))
    np.testing.assert_array_equal(observation, (0, 0, 1, 0))


@pytest.mark.parametrize("operation", ("reset", "step"))
def test_underlying_failure_requires_explicit_reset(operation):
    base = ReusedBufferEnv()
    env = D1ObservationHistory(base, 2)
    env.reset()
    setattr(base, f"raise_{operation}", True)
    with pytest.raises(RuntimeError, match="synthetic"):
        if operation == "reset":
            env.reset()
        else:
            env.step(np.zeros(1, np.float32))
    setattr(base, f"raise_{operation}", False)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(1, np.float32))


def test_sb3_terminal_observation_survives_its_separate_auto_reset():
    sb3 = pytest.importorskip("stable_baselines3.common.vec_env")
    vec = sb3.DummyVecEnv([lambda: D1ObservationHistory(ReusedBufferEnv(), 2)])
    try:
        vec.reset()
        for _ in range(3):
            observation, _, done, infos = vec.step(np.zeros((1, 1), np.float32))
        assert done[0]
        np.testing.assert_array_equal(infos[0]["terminal_observation"], (2, 0, 3, 0))
        np.testing.assert_array_equal(observation[0], (0, 0, 0, 0))
    finally:
        vec.close()


def test_real_ppo_update_save_and_reload_on_history_observations(tmp_path):
    torch = pytest.importorskip("torch")
    PPO = pytest.importorskip("stable_baselines3").PPO
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    env = D1ObservationHistory(ReusedBufferEnv(), 3)
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=8,
            n_epochs=2,
            seed=7,
            device="cpu",
            policy_kwargs={"net_arch": [16, 16]},
        )
        before = {key: value.clone() for key, value in model.policy.state_dict().items()}
        model.learn(total_timesteps=32)
        assert model.num_timesteps == 32 and model._n_updates == 8
        assert any(
            not torch.equal(before[key], value) for key, value in model.policy.state_dict().items()
        )
        observation, _ = env.reset(seed=23)
        prediction = model.predict(observation, deterministic=True)[0]
        path = tmp_path / "history_fixture_ppo.zip"
        model.save(path)
        loaded = PPO.load(path, device="cpu")
        assert loaded.observation_space.shape == (6,)
        assert loaded.observation_space.dtype == np.float32
        np.testing.assert_array_equal(
            loaded.predict(observation, deterministic=True)[0], prediction
        )
    finally:
        env.close()
        torch.set_num_threads(threads)
