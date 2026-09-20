"""Composition identity rejects previous task checkpoints before deserialization."""
import json

import mujoco
import pytest

from scripts.d1_flat_plane_env import D1FlatPlaneHeadingEnv
from scripts.d1_heading_tracking_env import D1HeadingTrackingEnv, load_heading_policy
from scripts.d1_stop_turn_env import StopTurnEnv
from scripts.probe_d1_heading_plane_stop_damping import PlaneStopDampingEnv
from scripts.probe_d1_heading_turn_yaw_authority import TurnAuthorityEnv
from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.locomotion_checkpoint import (
    _environment_contract,
    load_locomotion_policy,
)


@pytest.mark.parametrize("source_type", [D1HeadingTrackingEnv, D1FlatPlaneHeadingEnv,
                                         PlaneStopDampingEnv, TurnAuthorityEnv])
@pytest.mark.parametrize("loader,match", [(load_heading_policy, "heading_task_config"),
                                         (load_locomotion_policy, "task_schema")])
def test_old_checkpoint_rejected_before_load(tmp_path, monkeypatch, source_type, loader, match):
    from stable_baselines3 import PPO

    def forbidden(*args, **kwargs):
        pytest.fail("incompatible checkpoint deserialization or physics integration")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, forbidden)
    monkeypatch.setattr(PPO, "load", forbidden)
    source_folder, target_folder = tmp_path/"source", tmp_path/"target"
    source_folder.mkdir()
    target_folder.mkdir()
    kwargs = {"diagnostic_output": None} if source_type is PlaneStopDampingEnv else {}
    if source_type is TurnAuthorityEnv:
        kwargs = {"diagnostic_output": source_folder, "command_source": lambda t: D1MotionCommand()}
    source = source_type(episode_seconds=.01, **kwargs)
    target = StopTurnEnv(command_source=lambda t: D1MotionCommand(), diagnostic_output=target_folder,
                         episode_seconds=.01)
    try:
        source.reset(seed=55101)
        target.reset(seed=55101)
        sidecar = tmp_path/"old.json"
        sidecar.write_text(json.dumps({**_environment_contract(source),
            "recorded_episode": source.episode_metadata, "model_sha256": "not-read"}))
        with pytest.raises(ValueError, match=match):
            loader(tmp_path/"absent.zip", sidecar, target)
        config = target.heading_task_config
        assert config["stop_turn_composition"]["fixed_damping_nspm"] == 126.4374005337902
        assert config["turn_yaw_authority"]["leg_damping"] == "fixed post-stop latch only"
        assert source.plant.data.time == target.plant.data.time == 0.
    finally:
        source.close()
        target.close()
