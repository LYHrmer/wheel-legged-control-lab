"""One final shared-leg checkpoint with an explicit constant-wheel action map."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.d1_rolling_residual_task import (
    HEADING_OBSERVATION_SCHEMA,
    PHYSICAL_ACTION_SIZE,
    POLICY_ACTION_SIZE,
    ROLLING_ACTION_SCHEMA,
    ROLLING_CONTROL_SCHEMA,
    ROLLING_TASK_SCHEMA,
    rolling_task_definition,
)

CHECKPOINT_SCHEMA = "d1-rolling-shared-leg-final-ppo-v1"
FINAL_TIMESTEPS = 65536
FINAL_TRAIN_CALLS = 512
FINAL_OPTIMIZATION_EPOCHS = 2048


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()


def _check_environment(env: Any) -> None:
    if (env.task_schema != ROLLING_TASK_SCHEMA
            or env.observation_schema != HEADING_OBSERVATION_SCHEMA
            or env.action_schema != ROLLING_ACTION_SCHEMA
            or env.policy_action_size != POLICY_ACTION_SIZE
            or env.physical_action_size != PHYSICAL_ACTION_SIZE
            or env.unwrapped._controller.controller.control_schema != ROLLING_CONTROL_SCHEMA
            or env.observation_space.shape != (85,)
            or env.action_space.shape != (POLICY_ACTION_SIZE,)
            or tuple(env.policy_to_physical_indices) != (0, 0, 0, 0, None, None, None, None)):
        raise ValueError("environment is not the frozen shared-leg residual actor")


def save_final_checkpoint(
    model: Any,
    env: Any,
    directory: Path,
    *,
    initial_parameter_sha256: str,
    final_parameter_sha256: str,
    probe_observations: np.ndarray,
    dso_sha256: str,
    training_protocol_sha256: str,
) -> dict[str, Any]:
    """Save exactly one checkpoint after the final completed PPO update."""
    _check_environment(env)
    if (model.num_timesteps != FINAL_TIMESTEPS
            or model.completed_train_calls != FINAL_TRAIN_CALLS
            or model._n_updates != FINAL_OPTIMIZATION_EPOCHS
            or initial_parameter_sha256 == final_parameter_sha256):
        raise ValueError("PPO final update or parameter-change proof is missing")
    observations = np.asarray(probe_observations, dtype=np.float32)
    if observations.shape != (2, 85) or not np.isfinite(observations).all():
        raise ValueError("checkpoint probe needs two finite 85D observations")
    target = Path(directory)
    target.mkdir(exist_ok=False)
    model_path = target / "model.zip"
    model.save(model_path)
    actions, _ = model.predict(observations, deterministic=True)
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (2, POLICY_ACTION_SIZE) or not np.isfinite(actions).all():
        raise ValueError("final policy prediction has wrong shape or nonfinite values")
    np.savez_compressed(target / "reload_probe.npz", observations=observations, actions=actions)
    sidecar = {
        "schema": CHECKPOINT_SCHEMA,
        "task_definition": rolling_task_definition(),
        "task_schema": ROLLING_TASK_SCHEMA,
        "observation_schema": HEADING_OBSERVATION_SCHEMA,
        "action_schema": ROLLING_ACTION_SCHEMA,
        "controller_schema": ROLLING_CONTROL_SCHEMA,
        "observation_shape": [85],
        "action_shape": [POLICY_ACTION_SIZE],
        "action_expansion": {
            "policy_to_physical_indices": [0, 0, 0, 0, None, None, None, None],
            "leg_normalized_scale": 0.25,
            "wheel_constant": 0.0,
            "gate": "raw_forward_velocity_mps != 0.0",
        },
        "num_timesteps": FINAL_TIMESTEPS,
        "completed_train_calls": FINAL_TRAIN_CALLS,
        "optimization_epochs": FINAL_OPTIMIZATION_EPOCHS,
        "initial_parameter_sha256": initial_parameter_sha256,
        "final_parameter_sha256": final_parameter_sha256,
        "model_sha256": _sha256(model_path),
        "reload_probe_sha256": _sha256(target / "reload_probe.npz"),
        "dso_sha256": dso_sha256,
        "training_protocol_sha256": training_protocol_sha256,
        "recorded_episode": env.episode_metadata,
    }
    _write_json_exclusive(target / "model.metadata.json", sidecar)
    return sidecar


def load_final_checkpoint(
    model_path: Path,
    metadata_path: Path,
    env: Any,
    *,
    expected_dso_sha256: str,
    expected_training_protocol_sha256: str,
) -> Any:
    """Validate static identity before the trusted local PPO deserialization."""
    _check_environment(env)
    model_path = Path(model_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint metadata must be an object")
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "task_definition": rolling_task_definition(),
        "task_schema": ROLLING_TASK_SCHEMA,
        "observation_schema": HEADING_OBSERVATION_SCHEMA,
        "action_schema": ROLLING_ACTION_SCHEMA,
        "controller_schema": ROLLING_CONTROL_SCHEMA,
        "observation_shape": [85],
        "action_shape": [POLICY_ACTION_SIZE],
        "action_expansion": {
            "policy_to_physical_indices": [0, 0, 0, 0, None, None, None, None],
            "leg_normalized_scale": 0.25,
            "wheel_constant": 0.0,
            "gate": "raw_forward_velocity_mps != 0.0",
        },
        "num_timesteps": FINAL_TIMESTEPS,
        "completed_train_calls": FINAL_TRAIN_CALLS,
        "optimization_epochs": FINAL_OPTIMIZATION_EPOCHS,
    }
    for key, want in expected.items():
        if type(metadata.get(key)) is not type(want) or metadata[key] != want:
            raise ValueError(f"checkpoint {key} is incompatible")
    if (metadata.get("dso_sha256") != expected_dso_sha256
            or metadata.get("training_protocol_sha256") != expected_training_protocol_sha256):
        raise ValueError("checkpoint engine or training protocol identity differs")
    if metadata.get("model_sha256") != _sha256(model_path):
        raise ValueError("checkpoint model hash mismatch")
    probe_path = Path(metadata_path).parent / "reload_probe.npz"
    if metadata.get("reload_probe_sha256") != _sha256(probe_path):
        raise ValueError("checkpoint reload-probe hash mismatch")
    from stable_baselines3 import PPO

    from wheel_legged_control.d1.ppo_update_audit import parameter_sha256

    model = PPO.load(model_path, device="cpu")
    if (model.observation_space.shape != (85,)
            or model.action_space.shape != (POLICY_ACTION_SIZE,)
            or model.num_timesteps != FINAL_TIMESTEPS
            or getattr(model, "completed_train_calls", None) != FINAL_TRAIN_CALLS
            or model._n_updates != FINAL_OPTIMIZATION_EPOCHS
            or parameter_sha256(model.policy) != metadata.get("final_parameter_sha256")):
        raise ValueError("loaded policy does not match the final checkpoint sidecar")
    with np.load(probe_path, allow_pickle=False) as probe:
        observations = probe["observations"]
        actions = probe["actions"]
    if (observations.shape != (2, 85) or actions.shape != (2, POLICY_ACTION_SIZE)
            or not np.isfinite(observations).all() or not np.isfinite(actions).all()):
        raise ValueError("checkpoint reload-probe values are invalid")
    predicted, _ = model.predict(observations, deterministic=True)
    if not np.array_equal(np.asarray(predicted, dtype=np.float32), actions):
        raise ValueError("reloaded policy prediction differs from saved final policy")
    return model
