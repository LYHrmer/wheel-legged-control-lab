"""Tracking-v2 profile, training reward units and predeclared task criteria."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "tracking_experiment_cli", ROOT / "scripts" / "run_d1_terrain_curriculum.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_defaults_preserve_legacy_and_define_tracking(runner, tmp_path):
    legacy = runner.build_parser().parse_args(["--output", str(tmp_path / "legacy")])
    tracking = runner.build_parser().parse_args(
        ["--output", str(tmp_path / "tracking"), "--profile", "tracking-v2"]
    )
    assert runner.reward_scale(legacy) == 1.0
    assert runner.training_conditions(legacy) == ["flat", "curriculum"]
    assert runner.reward_scale(tracking) == 0.01
    assert runner.training_conditions(tracking) == ["flat", "mixed", "curriculum"]
    assert len(runner.build_protocol(legacy)["cases"]) == 16
    protocol = runner.build_protocol(tracking)
    assert len(protocol["cases"]) == 24
    assert protocol["control_schema"] == "d1-lqr-vmc-local-tangent-v2"
    assert protocol["quality_criteria"] == runner.TRACKING_QUALITY_CRITERIA


@pytest.mark.parametrize("scale", ["0", "-1", "nan", "inf"])
def test_invalid_reward_scale_rejected_before_output(runner, tmp_path, scale):
    output = tmp_path / "invalid"
    args = runner.build_parser().parse_args(
        ["--output", str(output), "--profile", "tracking-v2", "--reward-scale", scale]
    )
    with pytest.raises(ValueError, match="reward-scale"):
        runner.validate_args(args)
    assert not output.exists()


def test_legacy_mixed_and_duplicate_conditions_are_rejected(runner, tmp_path):
    for extra in (["--conditions", "mixed"], ["--conditions", "flat", "flat"]):
        args = runner.build_parser().parse_args(["--output", str(tmp_path / "new"), *extra])
        with pytest.raises(ValueError):
            runner.validate_args(args)


class RewardFixture(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-5.0, 5.0, (2,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, np.float32), {}

    def step(self, action):
        return np.asarray(action), 7.5, False, False, {"reward_terms": {"tracking": 7.5}}


def test_reward_wrapper_changes_only_returned_scalar(runner):
    raw = RewardFixture()
    wrapped = runner.TrainingRewardScale(RewardFixture(), 0.01)
    action = np.array([0.2, -0.3], np.float32)
    left, right = raw.step(action), wrapped.step(action)
    np.testing.assert_array_equal(left[0], right[0])
    assert right[1] == pytest.approx(0.075)
    assert left[2:] == right[2:]


def test_monitor_stays_raw_and_timeout_bootstrap_uses_scaled_value_units(runner):
    torch = pytest.importorskip("torch")
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    class OneStepFixture(RewardFixture):
        def step(self, action):
            observation, reward, terminated, _, info = super().step(action)
            return observation, reward, terminated, True, info

    class Capture(BaseCallback):
        def _on_step(self):
            self.last_info = self.locals["infos"][0]
            return True

    env = DummyVecEnv([lambda: runner.TrainingRewardScale(Monitor(OneStepFixture()), 0.01)])
    try:
        model = sb3.PPO("MlpPolicy", env, n_steps=2, batch_size=2, gamma=0.9, device="cpu")
        with torch.no_grad():
            model.policy.value_net.weight.zero_()
            model.policy.value_net.bias.fill_(5.0)  # Five units of scaled return.
        capture = Capture()
        _, callback = model._setup_learn(2, callback=capture)
        model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=2)
        assert capture.last_info["episode"]["r"] == 7.5
        assert capture.last_info["reward_terms"] == {"tracking": 7.5}
        # The wrapper scales r, not r + gamma * V. SB3 adds V in scaled units.
        np.testing.assert_allclose(model.rollout_buffer.rewards, 0.01 * 7.5 + 0.9 * 5.0)
        np.testing.assert_allclose(model.rollout_buffer.returns, model.rollout_buffer.rewards)
    finally:
        env.close()


@pytest.mark.parametrize("scale", [False, 0, -1, np.nan, np.inf])
def test_reward_wrapper_direct_interface_validates_scale(runner, scale):
    with pytest.raises(ValueError):
        runner.TrainingRewardScale(RewardFixture(), scale)


@pytest.mark.parametrize("split", ["development", "holdout"])
def test_tracking_cases_cover_complete_geometry_grid(runner, split):
    cases = runner.evaluation_cases(split, "tracking-v2")
    assert len(cases) == len({case["case_id"] for case in cases}) == 24
    assert len([case for case in cases if case["terrain"]["kind"] == "flat"]) == 4
    bumps = [case for case in cases if case["terrain"]["kind"] == "bumps"]
    assert len(bumps) == 12
    assert len({tuple(sorted(case["terrain"].items())) for case in bumps}) == 12
    ramps = [case for case in cases if case["terrain"]["kind"] == "ramp"]
    expected = {-3.5, -2.5, 2.5, 3.5} if split == "holdout" else {-4, -2, 2, 4}
    assert {case["terrain"]["slope_deg"] for case in ramps} == expected
    if split == "holdout":
        assert {case["terrain"]["wavelength_m"] for case in bumps} == {0.75, 0.95, 1.15}
        assert {case["terrain"]["phase_rad"] for case in bumps} == {0.53, 2.17}
        assert [case["environment_seed"] for case in cases] == list(range(44000, 44024))


def test_mixed_callback_never_reverts_pending_stage(runner):
    callback = runner._curriculum_callback(object, "mixed", 1024)
    callback.training_env = SimpleNamespace(env_method=lambda *args: pytest.fail("stage changed"))
    callback.locals = {
        "infos": [
            {
                "curriculum_stage": 3,
                "terrain_config": {"kind": "ramp", "slope_deg": 4.0},
                "episode_step": 1,
                "target_velocity_mps": 0.35,
            }
        ],
        "dones": [False],
    }
    for timestep in (1, 256, 512, 1024):
        callback.num_timesteps = timestep
        assert callback._on_step()
    assert callback.pending_stage == 3
    assert callback.statistics()["stage_steps"] == {"0": 0, "1": 0, "2": 0, "3": 4}


@pytest.mark.parametrize(
    "field", ["observation_schema", "reward_schema", "control_schema", "action_schema"]
)
def test_tracking_checkpoint_checks_all_semantic_schemas(runner, field):
    from wheel_legged_control.d1.terrain_tracking_env import D1TerrainTrackingEnv

    metadata = {
        name: getattr(D1TerrainTrackingEnv, name)
        for name in ("observation_schema", "reward_schema", "control_schema", "action_schema")
    }
    model = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(44,)), action_space=SimpleNamespace(shape=(2,))
    )
    runner.validate_checkpoint(metadata, model, D1TerrainTrackingEnv)
    metadata[field] = "old_or_wrong_semantics"
    with pytest.raises(ValueError, match=field):
        runner.validate_checkpoint(metadata, model, D1TerrainTrackingEnv)


def quality_rows():
    rows, distance = [], 0.0
    for step in range(400):
        command = 0.0 if step < 30 else 0.35
        distance += command * 0.01
        rows.append(
            {
                "position_x_m": distance,
                "initial_position_x_m": 0.0,
                "command_velocity_mps": command,
                "forward_velocity_mps": command,
                "velocity_error_mps": 0.0,
                "clearance_error_m": 0.0,
                "yaw_rad": 0.0,
                "torque_saturation_fraction": 0.0,
                "policy_enabled": 1,
                "policy_gated": 0,
                "reward": 1.0,
                "terminated": 0,
                "truncated": int(step == 399),
                "termination_reason": "time_limit" if step == 399 else "ongoing",
            }
        )
    return rows


def test_task_quality_uses_command_integral_and_last_second(runner):
    rows = quality_rows()
    for row in rows[:30]:
        row["velocity_error_mps"] = 0.5
    metrics = runner.summarize_episode(
        rows,
        quality_criteria=runner.TRACKING_QUALITY_CRITERIA,
        requested_duration_s=4.0,
        target_velocity_mps=0.35,
    )
    assert metrics["completed"] == metrics["quality_success"] == 1
    assert metrics["progress_fraction"] == pytest.approx(1.0)
    assert metrics["velocity_rmse_mps"] > 0.1
    assert metrics["tail_velocity_rmse_mps"] == 0
    assert metrics["tail_observed_seconds"] == 1.0
    assert "quality_success" not in runner.summarize_episode(rows)


@pytest.mark.parametrize(
    "failure",
    ["progress", "tail_velocity", "clearance", "yaw", "torque_saturation", "incomplete_episode"],
)
def test_survival_alone_cannot_pass_quality_criteria(runner, failure):
    rows = quality_rows()
    if failure == "progress":
        rows[-1]["position_x_m"] *= 0.5
    elif failure == "tail_velocity":
        for row in rows[-100:]:
            row["velocity_error_mps"] = 0.1
    elif failure == "clearance":
        for row in rows:
            row["clearance_error_m"] = 0.031
    elif failure == "yaw":
        rows[-1]["yaw_rad"] = 0.151
    elif failure == "torque_saturation":
        for row in rows:
            row["torque_saturation_fraction"] = 0.051
    else:
        rows = rows[:300]
        rows[-1].update(truncated=0, terminated=1, termination_reason="excess_pitch")
    metrics = runner.summarize_episode(
        rows,
        quality_criteria=runner.TRACKING_QUALITY_CRITERIA,
        requested_duration_s=4.0,
        target_velocity_mps=0.35,
    )
    assert metrics["quality_success"] == 0
    assert failure in metrics["quality_failure_reasons"]


def test_short_smoke_quality_is_undefined(runner):
    result = runner.summarize_episode(
        quality_rows()[:50],
        quality_criteria=runner.TRACKING_QUALITY_CRITERIA,
        requested_duration_s=0.5,
        target_velocity_mps=0.35,
    )
    assert result["quality_success"] is None


def test_actual_v2_training_scales_reward_and_keeps_mixed_stage(runner, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from wheel_legged_control.d1.terrain_tracking_env import D1TerrainTrackingEnv

    args = runner.build_parser().parse_args(
        [
            "--output",
            str(tmp_path / "experiment"),
            "--profile",
            "tracking-v2",
            "--steps",
            "256",
            "--envs",
            "1",
            "--seeds",
            "7",
            "--episode-seconds",
            "0.1",
        ]
    )
    output = tmp_path / "mixed_policy"
    _, metadata = runner.train_policy(
        args,
        condition="mixed",
        seed=7,
        output=output,
        environment=D1TerrainTrackingEnv,
        dependencies=runner._load_rl_dependencies(),
        source_hashes={"test": "integration"},
    )
    assert metadata["training_reward_scale"] == 0.01
    assert metadata["actual_timesteps"] == 256
    assert metadata["control_schema"] == D1TerrainTrackingEnv.control_schema
    assert metadata["exposure"]["stage_steps"] == {"0": 0, "1": 0, "2": 0, "3": 256}
    assert json.loads((output / "episode_starts.json").read_text())[0]["curriculum_stage"] == 3
