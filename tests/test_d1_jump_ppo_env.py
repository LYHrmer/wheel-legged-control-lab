"""Integration seams checked without advancing the MuJoCo integrator."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import scripts.d1_jump_ppo_env as mod
from scripts.d1_jump_ppo_task import JumpEpisodeSpec, JumpProgress


@pytest.fixture(autouse=True)
def no_integration(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("physical integration is forbidden in this test")
    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, forbidden)


@pytest.mark.parametrize("scale", [.95, 1., 1.05])
def test_real_reset_has_same_actor_prefix_actual_friction_and_persistent_metadata(scale):
    env = mod.D1JumpPPOEnv(episode_spec=JumpEpisodeSpec(200, .005, scale))
    try:
        obs, info = env.reset(seed=629990000)
        context = env._require_context("test")
        from wheel_legged_control.d1.locomotion_observation import (
            encode_d1_locomotion_observation,
        )
        prefix = env._extend(encode_d1_locomotion_observation(context.decision), context)
        assert obs.shape == (95,) and obs.dtype == np.float32
        assert obs[:85].tobytes() == prefix.tobytes()
        assert obs[85:87].tolist() == [-1., 0.]
        assert np.isfinite(obs).all()
        assert env.max_steps == 600 and env.plant.data.time == 0
        assert env.plant.model.geom_friction[env.plant.floor_geom_id, 0] == pytest.approx(.9 * scale)
        assert info["episode_metadata"] == env.episode_metadata
        assert env.episode_metadata["jump_task_config"]["observation"]["total_size"] == 95
        assert env.episode_metadata["task_schema"] == mod.JUMP_TASK_SCHEMA
        assert env.episode_metadata["terrain"]["heightfield_present"] is False
        with pytest.raises(RuntimeError, match="active episode"):
            env.configure_next_episode(JumpEpisodeSpec(None, 0.))
        env.close()
        env.configure_next_episode(JumpEpisodeSpec(None, 0.))
    finally:
        env.close()


@pytest.mark.parametrize("kwargs", [
    {"baseline": "lqr"}, {"action_mode": "shared2"}, {"episode_seconds": 1.},
    {"ground_friction": .95}, {"randomization": None}, {"provider_config": None},
    {"heading_reward_weight": .1}, {"command_source": lambda t: None},
])
def test_noncontract_configuration_rejected_before_parent(monkeypatch, kwargs):
    def forbidden(*args, **values):
        raise AssertionError("invalid configuration reached the parent constructor")
    monkeypatch.setattr(mod.D1HeadingTrackingEnv, "__init__", forbidden)
    with pytest.raises((TypeError, ValueError)):
        mod.D1JumpPPOEnv(**kwargs)


def test_contact_wrapper_once_and_reverse_order_cleanup():
    calls = []
    def original(torque, **kwargs):
        calls.append((torque, kwargs))
        return 12
    env = object.__new__(mod.D1JumpPPOEnv)
    env.plant = SimpleNamespace(step=original)
    env._install_contact_readout()
    inner = env.plant.step
    assert inner([1], measure_contact_wrench=False) == 12
    assert calls == [([1], {"measure_contact_wrench": True})]
    env.plant.step = lambda *a, **kw: inner(*a, **kw)
    with pytest.raises(RuntimeError, match="outer"):
        env._restore_contact_readout()
    assert env._plant_step_original is original
    env.plant.step = inner
    env._restore_contact_readout()
    assert env.plant.step is original
    env._restore_contact_readout()


@pytest.mark.parametrize("time", [-.01, 6.01, .005, float("nan")])
def test_callback_refuses_invalid_clock(time):
    env = object.__new__(mod.D1JumpPPOEnv)
    env.plant = SimpleNamespace(control_dt=.01)
    env._jump_spec = JumpEpisodeSpec(200, .005)
    with pytest.raises(ValueError):
        env._raw_command_callback(time)


def test_endpoint_uses_truth_not_stale_actor_state():
    env = object.__new__(mod.D1JumpPPOEnv)
    env.plant = SimpleNamespace(last_control_interval_contact_wrench=SimpleNamespace(
        active_sample_fraction_by_wheel=np.ones(4),
        wheel_force_world_n=np.tile([0., 0., 10.], (4, 1)), physics_sample_count=5))
    env.last_transition = SimpleNamespace(truth=SimpleNamespace(
        base_position=np.array([.01, 0., .455]), base_rpy=np.array([.01, -.02, .03]),
        base_linear_velocity_body=np.array([.025, 0., 0.])))
    env._initial_origin = np.zeros(2)
    env._whole_robot_com = lambda: (np.array([0., 0., .3]), np.array([0., 0., .02]))
    geometry = {"simultaneous_minimum_gap_m": -.001, "contact_margin_m": .001,
                "wheel_bottom_gap_m": [-.001] * 4}
    metrics = env._endpoint_metrics(geometry, {"heading_task": {"heading_error_after": .04}})
    assert metrics.body_forward_velocity_mps == .025
    assert metrics.heading_error_rad == .04
    assert metrics.com_velocity_mps[2] == .02
    with pytest.raises(KeyError):
        env._endpoint_metrics(geometry, {"heading_task": {}})
    env.plant.last_control_interval_contact_wrench = None
    with pytest.raises(RuntimeError, match="missing"):
        env._endpoint_metrics(geometry, {})


def test_post_step_exception_invalidates_episode_without_retry():
    env = object.__new__(mod.D1JumpPPOEnv)
    env._active = True
    env._progress = JumpProgress()
    calls = []
    def failed(action):
        calls.append(action)
        raise ValueError("invalid geometry")
    env._step_once = failed
    with pytest.raises(ValueError, match="geometry"):
        env.step(np.zeros(8))
    assert not env._active and len(calls) == 1
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(8))
    assert len(calls) == 1
