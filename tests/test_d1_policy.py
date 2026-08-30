import hashlib
import json

import pytest

from wheel_legged_control.d1.env import D1_OBSERVATION_SCHEMA
from wheel_legged_control.d1.policy import load_compatible_d1_policy
from wheel_legged_control.d1.rewards import D1_REWARD_SCHEMA


def test_checkpoint_state_mode_must_match_runtime(tmp_path) -> None:
    model = tmp_path / "model.zip"
    model.write_bytes(b"checkpoint")
    (tmp_path / "training_config.json").write_text(
        json.dumps(
            {
                "robot": "d1",
                "baseline": "lqr",
                "state_mode": "oracle",
                "observation_schema": D1_OBSERVATION_SCHEMA,
                "reward_schema": D1_REWARD_SCHEMA,
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="trained with 'oracle' state"):
        load_compatible_d1_policy(model, expected_state_mode="estimated")
