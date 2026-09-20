"""Fixed curriculum, native counter and checkpoint checks with integration forbidden."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from scripts.d1_jump_ppo_env import D1JumpPPOEnv, load_jump_policy
from scripts.run_d1_heading_study import AfterUpdatePPO
from scripts.run_d1_jump_ppo_study import NativeCounter, episode_spec
from scripts.run_d1_locomotion_experiment import PPO_SETTINGS
from wheel_legged_control.d1.locomotion_checkpoint import write_checkpoint_metadata
from wheel_legged_control.d1.ppo_update_audit import parameter_sha256


@pytest.mark.parametrize("transitions,expected", [(0,.005),(32767,.005),(32768,.01),(65535,.01),(65536,.02)])
def test_stage_uses_completed_transitions_at_reset(transitions, expected):
    assert episode_spec("formal", 0, transitions).net_clearance_m == expected


def test_episode_cycle_and_smoke_are_fixed():
    assert [episode_spec("formal", k, 0).request_tick for k in range(10)] == [
        150,200,250,150,None,200,250,150,200,None]
    assert episode_spec("smoke",0,0).request_tick == 200
    assert episode_spec("smoke",1,600).is_hold
    assert episode_spec("smoke",3,599).is_hold


@pytest.mark.parametrize("fail", [False, True])
def test_native_counter_counts_partial_and_restores_binding(monkeypatch, fail):
    data = SimpleNamespace(time=0.)
    model = object()
    calls = []
    def original(m, d):
        calls.append(1)
        if fail:
            raise ValueError("partial")
        d.time += .002
    monkeypatch.setattr(mujoco, "mj_step", original)
    counter = NativeCounter(SimpleNamespace(model=model,data=data), 1)
    with counter:
        if fail:
            with pytest.raises(ValueError):
                mujoco.mj_step(model,data)
        else:
            mujoco.mj_step(model,data)
        with pytest.raises(RuntimeError,match="budget"):
            mujoco.mj_step(model,data)
    assert len(calls) == counter.attempted == 1
    assert counter.returned == int(not fail) and counter.failed == int(fail)
    assert mujoco.mj_step is original


def test_fresh_policy_save_reload_needs_no_physics(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("physical integration forbidden")
    for key in ("mj_step","mj_step1","mj_step2"):
        monkeypatch.setattr(mujoco,key,forbidden)
    env = D1JumpPPOEnv()
    try:
        obs,_ = env.reset(seed=123)
        model = AfterUpdatePPO("MlpPolicy",env,seed=62001,device="cpu",**PPO_SETTINGS)
        assert model.num_timesteps == model._n_updates == 0
        assert not model.policy.optimizer.state
        target = tmp_path / "model.zip"
        metadata = tmp_path / "metadata.json"
        model.save(target)
        write_checkpoint_metadata(target,env,metadata)
        loaded = load_jump_policy(target,metadata,env)
        assert parameter_sha256(loaded.policy) == parameter_sha256(model.policy)
        assert np.array_equal(model.predict(obs,deterministic=True)[0],
                              loaded.predict(obs,deterministic=True)[0])
        assert env.plant.data.time == 0.
    finally:
        env.close()
