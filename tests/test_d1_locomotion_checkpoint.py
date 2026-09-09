import json

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from wheel_legged_control.d1.locomotion_checkpoint import (
    load_locomotion_policy,
    write_checkpoint_metadata,
)
from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv
from wheel_legged_control.d1.observation_history import D1ObservationHistory


def test_real_untrained_checkpoint_roundtrip_uses_wrapped_observation(tmp_path):
    torch.set_num_threads(1)
    env = D1ObservationHistory(D1LocomotionEnv(episode_seconds=0.02), 2)
    try:
        observation, _ = env.reset(seed=17)
        model = PPO("MlpPolicy", env, n_steps=2, batch_size=2, seed=73, device="cpu")
        model_path = tmp_path / "model.zip"
        model.save(model_path)
        metadata_path = tmp_path / "metadata.json"
        metadata = write_checkpoint_metadata(
            model_path, env, metadata_path, extra={"training_steps": 0}
        )
        assert metadata["observation_dim"] == 164
        assert metadata["history_length"] == 2
        assert metadata["action_dim"] == 8
        assert metadata["extra"] == {"training_steps": 0}
        assert json.loads(metadata_path.read_text()) == metadata
        loaded = load_locomotion_policy(model_path, metadata_path, env)
        before, _ = model.predict(observation, deterministic=True)
        after, _ = loaded.predict(observation, deterministic=True)
        np.testing.assert_array_equal(before, after)
    finally:
        env.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("baseline", "lqr"),
        ("source_schema", "d1-synchronized-imu-encoder-fusion-v1"),
        ("observation_schema", "d1-continuous-sensor-command45-v1"),
        ("action_schema", "d1-terrain-residual-quarter-v1"),
        ("history_length", True),
        ("task_schema", "different-task"),
        ("control_schema", "different-loop"),
        ("controller_schema", "different-controller"),
        ("reward_schema", "different-reward"),
        ("model_sha256", "0" * 64),
    ],
)
def test_metadata_mismatch_rejected_before_loading_model(tmp_path, field, value):
    env = D1LocomotionEnv(episode_seconds=0.01)
    try:
        env.reset(seed=17)
        model_path, metadata_path = tmp_path / "not-a-model.zip", tmp_path / "metadata.json"
        model_path.write_bytes(b"deliberately invalid model: compatibility must be checked first")
        fields = write_checkpoint_metadata(model_path, env, metadata_path)
        fields[field] = value
        metadata_path.write_text(json.dumps(fields))
        with pytest.raises(ValueError, match=field):
            load_locomotion_policy(model_path, metadata_path, env)
    finally:
        env.close()


def test_metadata_requires_reset_and_never_overwrites(tmp_path):
    env = D1LocomotionEnv(episode_seconds=0.01)
    path = tmp_path / "model.zip"
    path.write_bytes(b"fixture")
    sidecar = tmp_path / "metadata.json"
    try:
        with pytest.raises(RuntimeError, match="reset"):
            write_checkpoint_metadata(path, env, sidecar)
        env.reset(seed=17)
        write_checkpoint_metadata(path, env, sidecar)
        before = sidecar.read_bytes()
        with pytest.raises(FileExistsError):
            write_checkpoint_metadata(path, env, sidecar)
        assert sidecar.read_bytes() == before
    finally:
        env.close()


def test_same_dimensions_wrong_source_and_baseline_do_not_load(tmp_path):
    env = D1LocomotionEnv(baseline="lqr", episode_seconds=0.01)
    target = D1LocomotionEnv(baseline="mpc", episode_seconds=0.01)
    path, sidecar = tmp_path / "model.zip", tmp_path / "metadata.json"
    path.write_bytes(b"invalid SB3 fixture")
    try:
        env.reset(seed=17)
        target.reset(seed=17)
        assert env.observation_space.shape == target.observation_space.shape == (82,)
        assert env.action_space.shape == target.action_space.shape == (2,)
        write_checkpoint_metadata(path, env, sidecar)
        with pytest.raises(ValueError, match="baseline"):
            load_locomotion_policy(path, sidecar, target)
    finally:
        env.close()
        target.close()


def test_sidecar_cannot_disguise_actual_saved_model_observation_shape(tmp_path):
    torch.set_num_threads(1)
    env = D1LocomotionEnv(episode_seconds=0.01)
    wrapped = D1ObservationHistory(env, 2)
    try:
        wrapped.reset(seed=17)
        model = PPO("MlpPolicy", env, n_steps=2, batch_size=2, device="cpu")
        path, sidecar = tmp_path / "model.zip", tmp_path / "metadata.json"
        model.save(path)
        write_checkpoint_metadata(path, wrapped, sidecar)
        with pytest.raises(ValueError, match="saved model observation_space"):
            load_locomotion_policy(path, sidecar, wrapped)
    finally:
        wrapped.close()
