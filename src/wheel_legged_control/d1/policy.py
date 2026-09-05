"""Checkpoint compatibility checks and lazy PPO loading for D1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .env import D1_OBSERVATION_SCHEMA
from .rewards import D1_REWARD_SCHEMA


def load_compatible_d1_policy(
    path: Path,
    *,
    expected_baseline: str = "lqr",
    expected_state_mode: str = "oracle",
    expected_latency_compensation: str = "none",
    expected_contact_allocation: str = "legacy",
) -> Any:
    """Load a PPO checkpoint only when its recorded contracts match the code."""

    policy_path = Path(path)
    if not policy_path.exists():
        raise FileNotFoundError(policy_path)
    config_path = policy_path.parent / "training_config.json"
    if not config_path.exists():
        raise ValueError(f"missing policy metadata: {config_path}")
    training_config = json.loads(config_path.read_text(encoding="utf-8"))
    if training_config.get("robot") != "d1":
        raise ValueError("checkpoint metadata does not describe a D1 policy")
    if training_config.get("baseline") != expected_baseline:
        raise ValueError(
            f"policy was trained over {training_config.get('baseline')!r}, "
            f"not {expected_baseline!r}"
        )
    if training_config.get("observation_schema") != D1_OBSERVATION_SCHEMA:
        raise ValueError("policy observation schema does not match this code")
    if training_config.get("reward_schema") != D1_REWARD_SCHEMA:
        raise ValueError("policy reward schema does not match this code")
    recorded_state_mode = training_config.get("state_mode", "oracle")
    if recorded_state_mode != expected_state_mode:
        raise ValueError(
            f"policy was trained with {recorded_state_mode!r} state, "
            f"not {expected_state_mode!r} state"
        )
    recorded_compensation = training_config.get("latency_compensation", "none")
    if recorded_compensation != expected_latency_compensation:
        raise ValueError(
            f"policy was trained with {recorded_compensation!r} latency compensation, "
            f"not {expected_latency_compensation!r}"
        )
    recorded_contact_allocation = training_config.get("contact_allocation", "legacy")
    if recorded_contact_allocation != expected_contact_allocation:
        raise ValueError(
            f"policy was trained with {recorded_contact_allocation!r} contact allocation, "
            f"not {expected_contact_allocation!r}"
        )
    recorded_digest = training_config.get("model_sha256")
    actual_digest = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    if recorded_digest != actual_digest:
        raise ValueError("policy checksum does not match training metadata")

    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise RuntimeError('install RL dependencies with: pip install -e ".[rl]"') from error
    return PPO.load(policy_path, device="cpu")
