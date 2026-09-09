"""Checkpoint compatibility for the same v3 Gym/keyboard observation stream.

Load only trusted local SB3 checkpoints: their serialization can execute Python.
Schemas establish field meanings, not a claim that a policy works on new terrain
or sensor/actuator parameters. Those episode settings are recorded separately.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from gymnasium.spaces import Box

CHECKPOINT_SCHEMA = "d1-locomotion-checkpoint-v1"
_EPISODE_KEYS = (
    "baseline",
    "task_schema",
    "control_schema",
    "controller_schema",
    "reward_schema",
    "action_schema",
    "source_schema",
)
_DEFAULT_CONTROLLER_SCHEMA = "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-v2"
# Schemas recorded before feedback bandwidth was configurable: their sidecars
# carry three gains and mean unscaled leg/attitude feedback.
_UNSCALED_CONTROLLER_SCHEMAS = (
    _DEFAULT_CONTROLLER_SCHEMA,
    "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-configured-v3",
)
_UNSCALED_GAIN_FIELDS = ("wheel_kp", "wheel_ki", "yaw_feedback_gain")
_FEEDBACK_SCALE_FIELDS = ("leg_feedback_scale", "attitude_feedback_scale")


def _controller_parameters(metadata: dict, expected: dict) -> dict:
    """Normalize a sidecar gain record onto the current five-field contract.

    A three-gain record is accepted only for the pre-scale schemas and only when
    the runtime feedback scales are both one, in which case the missing scales
    are normalized to one. Everything else - a scaled runtime, a newer schema,
    unknown fields or a malformed scalar - is rejected here, before any model
    deserialization. Differing gain values are then caught by the caller.
    """
    from dataclasses import asdict

    from .wheel_leg_controller import D1WheelLegControlConfig

    actual = metadata.get("controller_parameters")
    schema = metadata.get("controller_schema")
    unscaled_runtime = all(expected[name] == 1.0 for name in _FEEDBACK_SCALE_FIELDS)
    try:
        if "controller_parameters" not in metadata:
            # Historical v1 sidecars had only the fixed default controller.
            # A configured controller must supply its exact parameters.
            if schema != _DEFAULT_CONTROLLER_SCHEMA or expected != asdict(
                D1WheelLegControlConfig()
            ):
                raise ValueError("a configured controller must record its gains")
            return expected
        if type(actual) is not dict:
            raise TypeError("gain record must be an object")
        if set(actual) == set(_UNSCALED_GAIN_FIELDS) != set(expected):
            if schema not in _UNSCALED_CONTROLLER_SCHEMAS or not unscaled_runtime:
                raise ValueError("three-gain records require unscaled feedback")
            actual = {**actual, **dict.fromkeys(_FEEDBACK_SCALE_FIELDS, 1.0)}
        if set(actual) != set(expected):
            raise ValueError("missing/extra gain fields")
        D1WheelLegControlConfig(**actual)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint controller_parameters are invalid") from exc
    return actual


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _space_record(space) -> dict:
    if not isinstance(space, Box) or len(space.shape) != 1 or space.dtype.kind != "f":
        raise ValueError("locomotion checkpoint requires a flat floating-point Box")
    if not np.isfinite(space.low).all() or not np.isfinite(space.high).all():
        raise ValueError("locomotion checkpoint requires finite Box limits")
    return {
        "shape": list(space.shape),
        "dtype": space.dtype.name,
        "low": space.low.tolist(),
        "high": space.high.tolist(),
    }


def _environment_contract(env) -> dict:
    episode = env.unwrapped.episode_metadata
    if not episode:
        raise RuntimeError("reset the locomotion environment before checkpoint metadata use")
    fields = {name: episode[name] for name in _EPISODE_KEYS}
    if any(not isinstance(value, str) or not value for value in fields.values()):
        raise ValueError("checkpoint schemas and baseline must be nonempty strings")
    for name in ("action_schema", "source_schema"):
        if env.get_wrapper_attr(name) != fields[name]:
            raise ValueError(f"wrapper {name} differs from the task")
    observation_schema = env.get_wrapper_attr("observation_schema")
    if not isinstance(observation_schema, str) or not observation_schema:
        raise ValueError("observation_schema must be a nonempty string")
    try:
        history_length = env.get_wrapper_attr("history_length")
    except AttributeError:
        history_length = 1
    if (
        isinstance(history_length, bool)
        or not isinstance(history_length, int)
        or history_length < 1
    ):
        raise ValueError("history_length must be a positive integer")
    observation = _space_record(env.observation_space)
    action = _space_record(env.action_space)
    return {
        **(
            {"controller_parameters": episode["controller_parameters"]}
            if "controller_parameters" in episode
            else {}
        ),
        "schema": CHECKPOINT_SCHEMA,
        **fields,
        "observation_schema": observation_schema,
        "history_length": history_length,
        "observation_dim": observation["shape"][0],
        "action_dim": action["shape"][0],
        "observation_space": observation,
        "action_space": action,
    }


def write_checkpoint_metadata(model_path, env, output_path, extra=None) -> dict:
    """Write one new sidecar for an existing model and already-reset environment.

    Extra training notes are nested, so they cannot replace compatibility fields.
    Existing sidecars are never overwritten; the caller owns checkpoint naming.
    """
    model_path, output_path = Path(model_path), Path(output_path)
    metadata = {
        **_environment_contract(env),
        "model_sha256": _sha256(model_path),
        "recorded_episode": env.unwrapped.episode_metadata,
        "extra": {} if extra is None else extra,
    }
    if not isinstance(metadata["extra"], dict):
        raise TypeError("extra must be a JSON-serializable dict")
    payload = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with output_path.open("x") as stream:
        stream.write(payload)
    return json.loads(payload)


def load_locomotion_policy(model_path, metadata_path, env):
    """Validate sidecar/hash first, then load a trusted CPU PPO and check spaces.

    Baseline, source, field schemas and history must match even if dimensions
    happen to agree. Terrain, episode duration and synthetic dynamics may differ:
    evaluation is allowed there, but compatibility does not establish robustness.
    """
    model_path = Path(model_path)
    metadata = json.loads(Path(metadata_path).read_text())
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint metadata must be an object")
    for name, expected in _environment_contract(env).items():
        # JSON equality alone would accept True as history_length=1.
        actual = (
            _controller_parameters(metadata, expected)
            if name == "controller_parameters"
            else metadata.get(name)
        )
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"checkpoint {name} is incompatible with the environment")
    if metadata.get("model_sha256") != _sha256(model_path):
        raise ValueError("checkpoint model_sha256 mismatch")
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    for name in ("observation_space", "action_space"):
        if _space_record(getattr(model, name)) != _space_record(getattr(env, name)):
            raise ValueError(f"saved model {name} differs from the environment")
    return model
