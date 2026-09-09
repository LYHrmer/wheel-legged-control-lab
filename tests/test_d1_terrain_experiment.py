"""Frozen-budget and paired-evaluation checks, not assertions that PPO wins."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_d1_terrain_curriculum.py"


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    specification = importlib.util.spec_from_file_location("terrain_curriculum_cli", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("requested", "envs", "expected"), [(128, 1, 128), (129, 1, 256), (513, 4, 1024)]
)
def test_budget_matches_whole_sb3_rollouts(runner, requested, envs, expected):
    assert runner.actual_budget(requested, envs) == expected


@pytest.mark.parametrize(("steps", "envs"), [(0, 1), (-1, 1), (100, 3)])
def test_budget_rejects_invalid_configuration(runner, steps, envs):
    with pytest.raises(ValueError):
        runner.actual_budget(steps, envs)


@pytest.mark.parametrize(
    ("step", "stage"), [(0, 0), (255, 0), (256, 1), (512, 2), (768, 3), (1024, 3)]
)
def test_curriculum_schedule_is_fixed_before_scores_exist(runner, step, stage):
    assert runner.curriculum_stage(step, 1024) == stage


def test_evaluation_geometry_is_reproducible_and_holdout_is_disjoint(runner):
    development = runner.evaluation_cases("development")
    holdout = runner.evaluation_cases("holdout")
    assert len(holdout) == len(development) == 16
    assert holdout == runner.evaluation_cases("holdout")
    assert len({case["case_id"] for case in holdout}) == 16
    assert len({case["environment_seed"] for case in holdout}) == 16
    for case in holdout:
        terrain = case["terrain"]
        if terrain["kind"] == "bumps":
            assert terrain["wavelength_m"] not in (0.8, 1.0, 1.2)
            assert terrain["amplitude_m"] in (0.005, 0.010)
        if terrain["kind"] == "ramp":
            assert abs(terrain["slope_deg"]) == 3.0
    assert {case["environment_seed"] for case in development}.isdisjoint(
        case["environment_seed"] for case in holdout
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--steps", "0"],
        ["--steps", "128", "--envs", "1"],
        ["--seeds", "1", "1"],
        ["--seeds", "-1"],
        ["--episode-seconds", "nan"],
        ["--episode-seconds", "0.01"],
    ],
)
def test_invalid_run_has_no_output_side_effect(runner, tmp_path, extra):
    output = tmp_path / "new"
    args = runner.build_parser().parse_args(["--output", str(output), *extra])
    with pytest.raises(ValueError):
        runner.validate_args(args)
    assert not output.exists()


def test_existing_output_is_rejected_before_environment_or_rl_import(runner, tmp_path):
    args = runner.build_parser().parse_args(["--output", str(tmp_path)])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.run(args)


class ToyTerrainEnv(gym.Env):
    """A small deterministic transition model for runner plumbing, not D1 physics."""

    observation_schema = "d1-terrain-oracle-v1"
    reward_schema = "d1-terrain-clearance-v1"
    action_schema = "d1-terrain-residual-quarter-v1"
    resets: ClassVar[list[dict]] = []
    closes = 0

    def __init__(self, episode_seconds=0.1, training_mode="flat", stage=0, **kwargs):
        self.observation_space = gym.spaces.Box(-5.0, 5.0, (44,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.max_steps = round(episode_seconds / 0.01)
        self.training_mode = training_mode
        self.pending_stage = stage
        self.stage = stage

    def set_curriculum_stage(self, stage):
        self.pending_stage = stage

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.stage = self.pending_stage
        self.options = options or {
            "terrain": {"kind": "flat"},
            "velocity_mps": 0.35,
            "height_m": 0.455,
        }
        self.resets.append({"seed": seed, "options": self.options})
        return np.zeros(44, dtype=np.float32), {"position_x_m": 0.0}

    def step(self, action):
        self.step_count += 1
        velocity = 0.2 + float(action[0]) * 0.01
        info = {
            "episode_step": self.step_count,
            "curriculum_stage": self.stage,
            "terrain_config": self.options["terrain"],
            "position_x_m": self.step_count * 0.002,
            "position_y_m": 0.0,
            "yaw_rad": 0.01,
            "forward_velocity_mps": velocity,
            "command_velocity_mps": self.options["velocity_mps"],
            "target_velocity_mps": self.options["velocity_mps"],
            "velocity_error_mps": velocity - self.options["velocity_mps"],
            "clearance_m": 0.450,
            "command_clearance_m": self.options["height_m"],
            "clearance_error_m": 0.450 - self.options["height_m"],
            "ground_height_m": 0.0,
            "ground_pitch_rad": 0.0,
            "ground_roll_rad": 0.0,
            "torque_saturation_fraction": 1 / 16,
            "initial_height_lift_m": 0.002,
            "policy_applied": True,
            "termination_reason": "time_limit" if self.step_count >= self.max_steps else "ongoing",
        }
        return (
            np.full(44, self.step_count / 100, dtype=np.float32),
            1.0 - float(np.sum(np.square(action))),
            False,
            self.step_count >= self.max_steps,
            info,
        )

    def close(self):
        type(self).closes += 1


class ZeroPolicy:
    def predict(self, observation, deterministic):
        assert deterministic
        return np.zeros(2, dtype=np.float32), None


def test_eval_resets_identical_case_and_distinguishes_enabled_zero_policy(runner):
    case = runner.evaluation_cases("holdout")[4]
    ToyTerrainEnv.resets.clear()
    baseline_rows, baseline = runner.evaluate_case(ToyTerrainEnv, case, 0.1)
    policy_rows, policy = runner.evaluate_case(ToyTerrainEnv, case, 0.1, model=ZeroPolicy())
    assert ToyTerrainEnv.resets[0] == ToyTerrainEnv.resets[1]
    assert len(baseline_rows) == len(policy_rows) == 10
    assert baseline["velocity_rmse_mps"] == policy["velocity_rmse_mps"]
    assert baseline["policy_enabled_fraction"] == 0.0
    assert policy["policy_enabled_fraction"] == 1.0
    assert policy["policy_gated_fraction"] == 0.0
    assert baseline["completed"] == 1
    assert baseline_rows[0]["time_s"] == 0.01
    assert baseline_rows[-1]["time_s"] == 0.1


def test_metrics_are_recomputed_from_raw_transition_rows(runner):
    case = runner.evaluation_cases("development")[0]
    rows, metrics = runner.evaluate_case(ToyTerrainEnv, case, 0.1)
    assert metrics["velocity_rmse_mps"] == pytest.approx(
        np.sqrt(np.mean([row["velocity_error_mps"] ** 2 for row in rows]))
    )
    assert metrics["clearance_rmse_m"] == pytest.approx(0.005)
    assert metrics["mean_torque_saturation_fraction"] == pytest.approx(1 / 16)
    assert metrics["episode_return"] == 10
    # Reaching a low-error or high-return episode is not the completion definition.
    rows[-1].update(terminated=1, truncated=0, termination_reason="low_clearance")
    failed = runner.summarize_episode(rows)
    assert failed["completed"] == 0
    assert failed["termination_reason"] == "low_clearance"


def test_invalid_policy_action_closes_environment(runner):
    policy = SimpleNamespace(predict=lambda *args, **kwargs: (np.array([np.nan, 0.0]), None))
    before = ToyTerrainEnv.closes
    with pytest.raises(ValueError, match="invalid action"):
        runner.evaluate_case(
            ToyTerrainEnv, runner.evaluation_cases("development")[0], 0.1, model=policy
        )
    assert ToyTerrainEnv.closes == before + 1


@pytest.mark.parametrize("split", ["development", "holdout"])
def test_all_declared_cases_can_reset_and_step_real_d1(runner, split):
    from wheel_legged_control.d1.terrain_env import D1TerrainResidualEnv

    env = D1TerrainResidualEnv(episode_seconds=0.1, randomize=False)
    try:
        for case in runner.evaluation_cases(split):
            observation, info = env.reset(
                seed=case["environment_seed"],
                options={key: case[key] for key in ("terrain", "velocity_mps", "height_m")},
            )
            assert observation.shape == (44,)
            assert info["initial_height_lift_m"] >= 0.0
            _, reward, terminated, _, step_info = env.step(np.zeros(2, np.float32))
            assert np.isfinite(reward)
            assert not terminated
            assert set(runner.INFO_FIELDS).issubset(step_info)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("schema", "action_schema", "shape"),
    [
        ("d1-base-link-velocity-v2", "d1-terrain-residual-quarter-v1", (42,)),
        ("d1-terrain-oracle-v1", "legacy", (44,)),
        ("d1-terrain-oracle-v1", "d1-terrain-residual-quarter-v1", (42,)),
    ],
)
def test_checkpoint_rejects_old_or_mislabeled_policy(runner, schema, action_schema, shape):
    model = SimpleNamespace(
        observation_space=SimpleNamespace(shape=shape), action_space=SimpleNamespace(shape=(2,))
    )
    with pytest.raises(ValueError):
        runner.validate_checkpoint(
            {"observation_schema": schema, "action_schema": action_schema}, model
        )


def test_callback_reports_effective_stage_not_just_pending_stage(runner):
    class FakeBase:
        pass

    callback = runner._curriculum_callback(FakeBase, "curriculum", 1024)
    changed_stages = []
    callback.training_env = SimpleNamespace(
        env_method=lambda method, stage: changed_stages.append((method, stage))
    )
    callback.num_timesteps = 800
    callback.locals = {
        "infos": [
            {
                "curriculum_stage": 1,
                "terrain_config": {"kind": "bumps"},
                "episode_step": 1,
                "target_velocity_mps": 0.35,
            }
        ],
        "dones": [False],
    }
    assert callback._on_step()
    assert changed_stages == [("set_curriculum_stage", 3)]
    stats = callback.statistics()
    assert stats["stage_steps"]["1"] == 1
    assert stats["stage_episodes_started"]["1"] == 1
    assert stats["final_stage_sampled"] is False
    callback.locals["infos"][0].update(
        curriculum_stage=3, terrain_config={"kind": "ramp", "slope_deg": -4.0}
    )
    callback.locals["dones"][0] = True
    callback._on_step()
    assert callback.statistics()["final_stage_sampled"] is True
    assert callback.statistics()["stage_episodes_completed"]["3"] == 1
    assert callback.statistics()["stage_kind_steps"] == {"1:bumps": 1, "3:ramp": 1}
    assert callback.statistics()["ramp_slope_steps"] == {"-4": 1}
    assert callback.episode_starts[-1] == {
        "global_timesteps_after_vector_step": 800,
        "env_index": 0,
        "curriculum_stage": 3,
        "terrain_config": {"kind": "ramp", "slope_deg": -4.0},
        "target_velocity_mps": 0.35,
    }


def test_baseline_artifact_pairing_and_hashes_without_loading_rl(runner, tmp_path, monkeypatch):
    from wheel_legged_control.d1 import terrain_env

    monkeypatch.setattr(terrain_env, "D1TerrainResidualEnv", ToyTerrainEnv)
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"test_fixture": "frozen"})
    monkeypatch.setattr(
        runner, "_load_rl_dependencies", lambda: pytest.fail("baseline must not load Torch/SB3")
    )
    output = tmp_path / "baseline"
    args = runner.build_parser().parse_args(
        ["--output", str(output), "--mode", "baseline", "--episode-seconds", "0.1"]
    )
    runner.run(args)
    protocol = json.loads((output / "protocol.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert protocol["training_seeds"] == []
    assert summary["baseline_episodes"] == 16
    assert summary["training_runs"] == 0
    with (output / "metrics.csv").open(newline="") as stream:
        metrics = list(csv.DictReader(stream))
    assert len(metrics) == 16
    assert {row["training_seed"] for row in metrics} == {""}
    manifest = json.loads((output / "manifest.json").read_text())
    for relative, digest in manifest["sha256"].items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == digest


def test_real_cpu_ppo_updates_saves_and_reloads_on_tiny_transition_model(runner, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    args = runner.build_parser().parse_args(
        ["--output", str(tmp_path / "experiment"), "--steps", "256", "--envs", "1", "--seeds", "3"]
    )
    args.episode_seconds = 0.1
    output = tmp_path / "one_policy"
    loaded, metadata = runner.train_policy(
        args,
        condition="curriculum",
        seed=3,
        output=output,
        environment=ToyTerrainEnv,
        dependencies=runner._load_rl_dependencies(),
        source_hashes={"test_fixture": "not_a_physical_D1_experiment"},
    )
    assert metadata["actual_timesteps"] == 256
    assert metadata["completed_rollouts"] == 2
    assert metadata["ppo_epochs_completed"] == 8
    assert sum(metadata["exposure"]["stage_steps"].values()) == 256
    assert metadata["exposure"]["final_stage_sampled"] is True
    starts = json.loads((output / "episode_starts.json").read_text())
    assert len(starts) == sum(metadata["exposure"]["stage_episodes_started"].values())
    assert (
        metadata["episode_starts_sha256"]
        == hashlib.sha256((output / "episode_starts.json").read_bytes()).hexdigest()
    )
    assert (
        metadata["model_sha256"] == hashlib.sha256((output / "model.zip").read_bytes()).hexdigest()
    )
    action, _ = loaded.predict(np.zeros(44, np.float32), deterministic=True)
    assert action.shape == (2,)
    assert np.isfinite(action).all()
    with (output / "learning_metrics" / "progress.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[-1]["train/value_loss"] != ""
    assert rows[-1]["train/n_updates"] == "8"
