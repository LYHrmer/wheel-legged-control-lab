"""Check the two reviewed adapters compose before any physics is allowed."""
import json

import mujoco
import numpy as np
import pytest

from scripts.d1_flat_plane_env import D1FlatPlaneHeadingEnv, D1FlatPlanePlant
from scripts.d1_heading_tracking_env import D1HeadingTrackingEnv, load_heading_policy
from scripts.d1_stop_leg_damping import STOP_LEG_DAMPING_SCHEMA, StopLegDampingController
from scripts.probe_d1_heading_plane_stop_damping import PlaneStopDampingEnv
from scripts.probe_d1_heading_stop_damping import StopDampingEnv
from wheel_legged_control.d1.locomotion_checkpoint import (
    _environment_contract,
    load_locomotion_policy,
)


@pytest.fixture(autouse=True)
def forbid_integration(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("composition test must not integrate")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, forbidden)


def test_composition_binds_plane_and_fixed_damper_once_at_zero_time():
    env = PlaneStopDampingEnv(diagnostic_output=None, episode_seconds=.01)
    try:
        assert isinstance(env.plant, D1FlatPlanePlant)
        assert isinstance(env._controller.controller, StopLegDampingController)
        assert env.loop is None
        obs, info = env.reset(seed=55101)
        assert env.plant.data.time == 0. and env.loop.plant is env.plant
        assert obs.shape == (85,) and obs.dtype == np.float32
        config = info["episode_metadata"]["heading_task_config"]
        assert config["task_schema"] == env.task_schema
        assert config["collision_terrain"]["collision_geometry"] == "native_mujoco_plane"
        assert config["stop_leg_damping"]["enabled"] is True
        assert info["episode_metadata"]["controller_schema"] == STOP_LEG_DAMPING_SCHEMA
        assert not env._controller.controller.last_damping
        assert env._controller.controller.yaw_request_limit_rps == .6
    finally:
        env.close()


@pytest.mark.parametrize("source_type", [D1HeadingTrackingEnv, D1FlatPlaneHeadingEnv])
@pytest.mark.parametrize("loader,match", [(load_heading_policy, "heading_task_config"),
                                          (load_locomotion_policy, "task_schema")])
def test_old_checkpoint_rejected_before_deserialization(tmp_path, monkeypatch, source_type, loader, match):
    from stable_baselines3 import PPO

    def forbidden(*args, **kwargs):
        pytest.fail("old checkpoint reached model deserialization")

    monkeypatch.setattr(PPO, "load", forbidden)
    source = source_type(episode_seconds=.01)
    target = PlaneStopDampingEnv(diagnostic_output=None, episode_seconds=.01)
    try:
        source.reset(seed=55101)
        target.reset(seed=55101)
        sidecar = tmp_path / "old.json"
        sidecar.write_text(json.dumps({**_environment_contract(source),
            "recorded_episode": source.episode_metadata, "model_sha256": "not-read"}))
        with pytest.raises(ValueError, match=match):
            loader(tmp_path / "absent.zip", sidecar, target)
        assert source.plant.data.time == target.plant.data.time == 0.
        assert target._controller.controller.last_damping is None
    finally:
        source.close()
        target.close()


@pytest.mark.parametrize("action", [np.full(8, 1e-300), np.zeros(8, dtype=bool),
                                    np.full(8, np.nan), np.zeros(7)])
def test_invalid_action_does_not_consume_command_or_integrate(action):
    env = PlaneStopDampingEnv(diagnostic_output=None, episode_seconds=.01)
    try:
        env.reset(seed=55101)
        with pytest.raises(ValueError):
            env.step(action)
        assert env.plant.data.time == 0.
        assert env._controller.controller.last_damping is None
    finally:
        env.close()


def test_detail_close_failure_still_reaches_parent_archive(monkeypatch):
    env = PlaneStopDampingEnv(diagnostic_output=None, episode_seconds=.01)
    calls = []
    parent_close = StopDampingEnv.close

    def parent(self):
        calls.append(self)
        return parent_close(self)

    def fail():
        raise OSError("injected detail close failure")

    monkeypatch.setattr(StopDampingEnv, "close", parent)
    monkeypatch.setattr(env._details, "close", fail)
    with pytest.raises(OSError, match="injected"):
        env.close()
    assert calls == [env]
    assert env.plant.data.time == 0.
    env.close()
    assert calls == [env]
