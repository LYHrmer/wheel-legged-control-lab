"""One-coordinate hop task around the unchanged eight-action jump environment."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from scripts.d1_jump_heave_action import (
    HEAVE_ACTION_SCHEMA,
    heave_action_metadata,
    map_heave_action,
)
from scripts.d1_jump_ppo_env import D1JumpPPOEnv
from wheel_legged_control.d1.locomotion_checkpoint import _sha256, _space_record

HEAVE_TASK_SCHEMA = "d1-native-plane-shared-heave-hop-task-v1"
HEAVE_OBSERVATION_SCHEMA = "d1-jump-oracle-task95-v1"
HEAVE_CHECKPOINT_SCHEMA = "d1-jump-shared-heave-checkpoint-v1"


class D1JumpHeaveEnv(gym.Env):
    """Keep the physical parent's action history and all control memory intact.

    Composition keeps its eight-dimensional input contract unchanged. PPO sees
    one dimension; observations and action-change reward retain eight physical
    action coordinates. No legacy gather indices describe this gated matrix.
    """

    task_schema = HEAVE_TASK_SCHEMA
    observation_schema = HEAVE_OBSERVATION_SCHEMA
    action_schema = HEAVE_ACTION_SCHEMA
    action_mode = "request_gated_shared_heave1"
    policy_action_size = 1
    physical_action_size = 8

    def __init__(self, **kwargs):
        self._inner = D1JumpPPOEnv(**kwargs)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = self._inner.observation_space

    def __getattr__(self, name):
        # Existing readout/recording helpers access the actual inner plant and
        # prepared decision. Assignment never silently changes the inner task.
        if name == "policy_to_physical_indices":
            raise AttributeError("shared heave uses an explicit gated matrix, not a gather")
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    @property
    def jump_task_config(self):
        return {
            "schema": "d1-shared-heave-hop-config-v1",
            "task_schema": self.task_schema,
            "observation_schema": self.observation_schema,
            "observation_size": 95,
            "policy_action_size": 1,
            "physical_action_size": 8,
            "action_mapping": heave_action_metadata(),
            "parent_jump_task_config": self._inner.jump_task_config,
        }

    @property
    def episode_metadata(self):
        parent = self._inner.episode_metadata
        if not parent:
            return {}
        metadata = copy.deepcopy(parent)
        metadata["parent_action_contract"] = {
            key: metadata[key] for key in (
                "task_schema", "observation_schema", "action_schema", "action_mode",
                "policy_action_size", "physical_action_size", "policy_to_physical_indices",
            ) if key in metadata
        }
        metadata.pop("policy_to_physical_indices", None)
        metadata.update({
            "task_schema": self.task_schema,
            "observation_schema": self.observation_schema,
            "action_schema": self.action_schema,
            "action_mode": self.action_mode,
            "policy_action_size": 1,
            "physical_action_size": 8,
            "action_mapping": heave_action_metadata(),
            "jump_task_config": self.jump_task_config,
        })
        return metadata

    def reset(self, *, seed=None, options=None):
        observation, info = self._inner.reset(seed=seed, options=options)
        info = dict(info)
        info["episode_metadata"] = self.episode_metadata
        info["heave_action_mapping"] = heave_action_metadata()
        return observation, info

    def step(self, action):
        if not self._inner._active:
            raise RuntimeError("reset is required before a shared-heave step")
        tick = int(self._inner._require_context("step").tick)
        start = self._inner.jump_episode_spec.request_tick
        physical = map_heave_action(action, executed_tick=tick, request_tick=start)
        observation, reward, terminated, truncated, info = self._inner.step(physical)
        info = dict(info)
        if not np.array_equal(info["applied_action"], physical):
            raise RuntimeError("inner jump task changed the mapped physical action")
        raw = np.asarray(action, dtype=np.float64).tolist()
        info["policy_action"] = raw
        info["heave_action"] = {
            "schema": self.action_schema,
            "executed_tick": tick,
            "request_tick": start,
            "active": start is not None and start <= tick < start + 120,
            "policy_action_1": raw,
            "clipped_policy_action_1": np.clip(np.asarray(raw), -1., 1.).tolist(),
            "applied_physical_action_8": physical.tolist(),
            "parent_reward_uses": "applied_and_previous_physical_action_8",
        }
        return observation, reward, terminated, truncated, info

    def close(self):
        self._inner.close()


def _checkpoint_contract(env):
    task = env.unwrapped
    if not isinstance(task, D1JumpHeaveEnv):
        raise TypeError("a shared-heave checkpoint requires its own environment")
    episode = task.episode_metadata
    if not episode:
        raise RuntimeError("reset before using checkpoint metadata")
    if env.observation_space.shape != (95,) or env.action_space.shape != (1,):
        raise ValueError("shared-heave spaces must be 95 observations and one policy action")
    keys = ("baseline", "task_schema", "control_schema", "controller_schema", "reward_schema",
            "action_schema", "source_schema", "controller_parameters", "physical_action_schema")
    return {
        "schema": HEAVE_CHECKPOINT_SCHEMA,
        **{key: episode[key] for key in keys},
        "observation_schema": HEAVE_OBSERVATION_SCHEMA,
        "observation_dim": 95,
        "action_dim": 1,
        "policy_action_size": 1,
        "physical_action_size": 8,
        "action_mode": task.action_mode,
        "action_mapping": heave_action_metadata(),
        "jump_task_config": task.jump_task_config,
        "observation_space": _space_record(env.observation_space),
        "action_space": _space_record(env.action_space),
    }


def write_heave_checkpoint_metadata(model_path, env, output_path, extra=None):
    """Write a separate exact matrix/gate contract, never a fictitious gather."""
    payload = {
        **_checkpoint_contract(env),
        "model_sha256": _sha256(Path(model_path)),
        "recorded_episode": env.unwrapped.episode_metadata,
        "extra": {} if extra is None else extra,
    }
    if not isinstance(payload["extra"], dict):
        raise TypeError("extra must be an object")
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with Path(output_path).open("x") as stream:
        stream.write(text)
    return json.loads(text)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint identity key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite checkpoint number: {value}")


def load_heave_policy(model_path, metadata_path, env):
    """Reject old/task-mismatched metadata before trusted local deserialization."""
    payload = json.loads(Path(metadata_path).read_text(), object_pairs_hook=_unique_object,
                         parse_constant=_reject_constant)
    # JSON exponents such as 1e309 overflow without invoking parse_constant.
    json.dumps(payload, allow_nan=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint metadata must be an object")
    if "policy_to_physical_indices" in payload:
        raise ValueError("legacy gather fields cannot describe the shared-heave mapping")
    expected = _checkpoint_contract(env)
    for key, value in expected.items():
        actual = payload.get(key)
        if type(actual) is not type(value) or json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(value, sort_keys=True, allow_nan=False):
            raise ValueError(f"incompatible shared-heave checkpoint {key}")
    recorded = payload.get("recorded_episode")
    if not isinstance(recorded, dict):
        raise TypeError("recorded shared-heave task contract must be an object")
    for key in (
        "baseline", "task_schema", "observation_schema", "control_schema", "controller_schema",
        "reward_schema", "action_schema", "source_schema", "controller_parameters", "action_mode",
        "physical_action_schema", "policy_action_size", "physical_action_size", "action_mapping",
        "jump_task_config",
    ):
        if json.dumps(recorded.get(key), sort_keys=True, allow_nan=False) != json.dumps(expected[key], sort_keys=True, allow_nan=False):
            raise ValueError(f"incompatible recorded shared-heave {key}")
    if payload.get("model_sha256") != _sha256(Path(model_path)):
        raise ValueError("checkpoint model SHA256 mismatch")
    from stable_baselines3 import PPO

    model = PPO.load(model_path, env=None, device="cpu")
    for name in ("observation_space", "action_space"):
        if _space_record(getattr(model, name)) != _space_record(getattr(env, name)):
            raise ValueError(f"saved model {name} differs from the shared-heave environment")
    return model
