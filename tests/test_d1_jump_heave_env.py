"""Interface and strict checkpoint qualification with integration forbidden."""

import copy
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import scripts.d1_jump_heave_env as module
from scripts.d1_jump_ppo_task import JumpEpisodeSpec


@pytest.fixture(autouse=True)
def no_integration(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("interface tests must not integrate physics")
    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, fail)


@pytest.mark.parametrize("tick,start,active", [(174,175,False),(175,175,True),
                                               (294,175,True),(295,175,False),(200,None,False),
                                               (224,225,False),(225,225,True),
                                               (344,225,True),(345,225,False)])
def test_parent_once_physical_history_and_execute_not_prepare(tick, start, active):
    class Inner:
        _active = True
        jump_episode_spec = SimpleNamespace(request_tick=start)
        calls = 0
        previous = np.array([1.,1.,1.,1.,0.,0.,0.,0.])
        integral = np.array([.1,.2,.3,.4])

        def _require_context(self, method):
            return SimpleNamespace(tick=tick)

        def step(self, physical):
            self.calls += 1
            self.observation = np.arange(95, dtype=np.float32)
            self.received = physical.copy()
            self.previous_before = self.previous.copy()
            reward = -float(np.sum((physical-self.previous)**2))
            self.previous = physical.copy()
            return self.observation, reward, False, False, {
                "applied_action": physical.tolist(),
                "heading_task": {"appended_observation_decision_tick": tick+1},
            }

    env = object.__new__(module.D1JumpHeaveEnv)
    env._inner = inner = Inner()
    old = inner.previous.copy()
    obs, reward, terminated, truncated, info = env.step([.5])
    expected = np.array([.5]*4+[0.]*4) if active else np.zeros(8)
    assert inner.calls == 1 and obs is inner.observation
    np.testing.assert_array_equal(inner.received, expected)
    np.testing.assert_array_equal(inner.previous_before, old)
    np.testing.assert_array_equal(inner.integral, [.1,.2,.3,.4])
    assert reward == -float(np.sum((expected-old)**2))
    assert not terminated and not truncated
    assert info["heave_action"]["executed_tick"] == tick
    assert info["heave_action"]["active"] is active
    assert info["policy_action"] == [.5]
    assert info["heave_action"]["applied_physical_action_8"] == expected.tolist()


def test_invalid_action_never_calls_parent():
    class Inner:
        _active = True
        jump_episode_spec = SimpleNamespace(request_tick=None)
        def _require_context(self, method):
            return SimpleNamespace(tick=200)
        def step(self, action):
            pytest.fail("invalid action reached parent")
    env = object.__new__(module.D1JumpHeaveEnv)
    env._inner = Inner()
    with pytest.raises(ValueError, match="finite"):
        env.step([float("nan")])


@pytest.fixture
def real_env():
    env = module.D1JumpHeaveEnv(episode_spec=JumpEpisodeSpec(175,.02))
    observation, info = env.reset(seed=77202)
    assert observation.shape == (95,)
    assert env.action_space.shape == (1,)
    assert env._inner.action_space.shape == (8,)
    assert env.unwrapped is env
    assert info["episode_metadata"] == env.episode_metadata
    assert "policy_to_physical_indices" not in env.episode_metadata
    assert env._inner.episode_metadata["policy_action_size"] == 8
    assert env.episode_metadata["policy_action_size"] == 1
    assert not hasattr(env,"policy_to_physical_indices")
    env._test_reset_observation = observation.copy()
    yield env
    env.close()


def test_owned_metadata_does_not_relabel_inner(real_env):
    original = copy.deepcopy(real_env._inner.episode_metadata)
    changed = real_env.episode_metadata
    changed["action_mapping"]["active_embedding_matrix"][0][0] = 9
    changed["jump_task_config"]["action_mapping"]["gate"]["window_ticks"] = 1
    assert real_env._inner.episode_metadata == original
    assert real_env.jump_task_config["action_mapping"]["gate"]["window_ticks"] == 120
    assert real_env.episode_metadata["action_mapping"]["active_embedding_matrix"][0][0] == 1


def test_real_guarded_reset_preserves_original_observation(real_env):
    original = module.D1JumpPPOEnv(episode_spec=JumpEpisodeSpec(175,.02))
    try:
        observation, _ = original.reset(seed=77202)
        assert observation.tobytes() == real_env._test_reset_observation.tobytes()
        assert original.plant.data.qpos.tobytes() == real_env.plant.data.qpos.tobytes()
        assert original.plant.data.time == real_env.plant.data.time == 0.
    finally:
        original.close()


def test_fresh_model_roundtrip_zero_training_and_physics(real_env, tmp_path, monkeypatch):
    from stable_baselines3 import PPO

    from wheel_legged_control.d1.ppo_update_audit import parameter_sha256
    learner = PPO("MlpPolicy", real_env, n_steps=128, batch_size=128, seed=62888,
                  device="cpu", policy_kwargs={"net_arch":[64,64],"log_std_init":-2.})
    path = tmp_path/"fresh.zip"
    learner.save(path)
    sidecar = tmp_path/"fresh.json"
    module.write_heave_checkpoint_metadata(path, real_env, sidecar)
    def forbidden(*args, **kwargs):
        pytest.fail("reload must not reset or learn")
    monkeypatch.setattr(real_env,"reset",forbidden)
    monkeypatch.setattr(PPO,"learn",forbidden)
    loaded = module.load_heave_policy(path, sidecar, real_env)
    obs = np.stack([real_env._test_reset_observation,real_env._test_reset_observation])
    np.savez(tmp_path/"observations.npz", observations=obs)
    obs = np.load(tmp_path/"observations.npz")["observations"]
    np.testing.assert_array_equal(learner.predict(obs, deterministic=True)[0],
                                  loaded.predict(obs, deterministic=True)[0])
    assert parameter_sha256(learner.policy) == parameter_sha256(loaded.policy)
    assert loaded.num_timesteps == learner.num_timesteps == 0
    assert real_env.plant.data.time == 0.


def test_eight_action_model_with_forged_one_action_sidecar_rejected(real_env, tmp_path):
    from stable_baselines3 import PPO
    learner = PPO("MlpPolicy", real_env._inner, n_steps=128, batch_size=128,
                  seed=62889, device="cpu")
    path, sidecar = tmp_path/"eight.zip", tmp_path/"forged.json"
    learner.save(path)
    module.write_heave_checkpoint_metadata(path, real_env, sidecar)
    with pytest.raises(ValueError, match="action_space"):
        module.load_heave_policy(path, sidecar, real_env)


@pytest.mark.parametrize("mutation", ["old_schema", "old_action", "wheel_mapping", "gate",
                                       "recorded_identity", "hash", "dimension_boolean",
                                       "parent_config", "gather", "duplicate", "nonfinite", "overflow"])
def test_incompatible_sidecar_rejected_before_deserialization(real_env, tmp_path, monkeypatch, mutation):
    from stable_baselines3 import PPO
    path = tmp_path/"not_a_serialized_model.zip"
    path.write_bytes(b"metadata-only rejection fixture")
    sidecar = tmp_path/"model.json"
    payload = module.write_heave_checkpoint_metadata(path, real_env, sidecar)
    if mutation == "old_schema":
        payload["schema"] = "d1-locomotion-checkpoint-v1"
    elif mutation == "old_action":
        payload["action_dim"] = 8
    elif mutation == "wheel_mapping":
        payload["action_mapping"]["active_embedding_matrix"][4][0] = 1.
    elif mutation == "gate":
        payload["action_mapping"]["gate"]["window_ticks"] = 121
    elif mutation == "recorded_identity":
        payload["recorded_episode"]["task_schema"] = "old-task"
    elif mutation == "hash":
        payload["model_sha256"] = "0"*64
    elif mutation == "parent_config":
        payload["jump_task_config"]["parent_jump_task_config"]["action_size"] = 7
    elif mutation == "gather":
        payload["policy_to_physical_indices"] = [0]*8
    elif mutation == "nonfinite":
        payload["extra"]["nonfinite"] = float("nan")
    elif mutation == "dimension_boolean":
        payload["action_dim"] = True
    text = json.dumps(payload)
    if mutation == "duplicate":
        text = '{"action_dim":1,'+text[1:]
    if mutation == "overflow":
        text = text.replace('"extra": {}', '"extra": {"nested": [1e309]}')
    sidecar.write_text(text)
    def forbidden(*args, **kwargs):
        pytest.fail("incompatible sidecar reached deserialization")
    monkeypatch.setattr(PPO,"load",forbidden)
    with pytest.raises(ValueError):
        module.load_heave_policy(path, sidecar, real_env)


def test_inner_exception_is_never_retried():
    class Inner:
        _active = True
        calls = 0
        jump_episode_spec = SimpleNamespace(request_tick=175)
        def _require_context(self, method):
            return SimpleNamespace(tick=175)
        def step(self, action):
            self.calls += 1
            self._active = False
            raise RuntimeError("retained partial failure")
    env = object.__new__(module.D1JumpHeaveEnv)
    env._inner = Inner()
    with pytest.raises(RuntimeError, match="retained partial"):
        env.step([1.])
    assert env._inner.calls == 1 and not env._inner._active
