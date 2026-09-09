"""Fixed-budget sensor-command45 PPO, with frozen development/holdout cases."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import random
import sys
import tarfile
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from wheel_legged_control.d1.continuous_task import (
    SENSOR_CONTROL_SCHEMA,
    STAGES,
    TASK_SCHEMA,
    ContinuousTaskConfig,
    D1ContinuousTask,
    summarize_task,
)
from wheel_legged_control.d1.sensor_estimation import D1SensorNoise, D1SensorStateSource

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "continuous_rollout_helpers", ROOT / "scripts/run_d1_continuous_task.py"
)
ROLLOUT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ROLLOUT)
SEEDS = (19000, 20000, 21000)
DEVELOPMENT_SEEDS = (17, 29, 43)
HOLDOUT_SEEDS = (617, 629, 643)
BUDGETS = (32768, 65536, 131072)
REWARD_SCALE = 0.01
NOISE = D1SensorNoise(
    gyro_std_rad_s=0.002,
    accelerometer_std_m_s2=0.03,
    encoder_position_std_rad=0.0005,
    encoder_velocity_std_rad_s=0.005,
)
PPO_SETTINGS = {
    "learning_rate": 3e-4,
    "n_steps": 128,
    "batch_size": 128,
    "n_epochs": 4,
    "gamma": math.exp(-0.01 / 2),
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "policy_kwargs": {"net_arch": [64, 64]},
}


@contextmanager
def preserve_training_rng():
    """Model loading/dev work must not change the next training Gaussian draw."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def task_config(duration_s=45.0) -> ContinuousTaskConfig:
    return ContinuousTaskConfig(
        duration_s=duration_s,
        ground_reference_mode="estimated",
        observation_layout="command45",
        yaw_controller="pi",
        yaw_kp=2.0,
        yaw_ki=3.0,
    )


def make_task(duration_s=45.0) -> D1ContinuousTask:
    factory = partial(
        D1SensorStateSource,
        noise=NOISE,
        initial_position=(-3.8, 0, 0.455),
        initial_rpy=(0, 0, 0),
        delay_steps=0,
    )
    task = D1ContinuousTask(task_config(duration_s), state_source_factory=factory)
    assert_sensor_task(task)
    return task


def assert_sensor_task(task) -> None:
    if not isinstance(task.source, D1SensorStateSource):
        raise TypeError("actor/control state must be D1SensorStateSource")
    if task.config.ground_reference_mode != "estimated" or task.source.delay_steps != 0:
        raise ValueError("training requires measured ground and zero measurement delay")
    if task.observation_schema != "d1-continuous-sensor-command45-v1":
        raise ValueError("training requires the sensor command45 schema")
    if task.observation_space.shape != (45,) or task.action_space.shape != (2,):
        raise ValueError("training requires 45 observations and two force residuals")
    if getattr(task._estimated_ground, "__self__", None) is not task.source:
        raise ValueError("control ground must be bound to the same sensor source")


class EpisodeSeedStream(gym.Wrapper):
    """Deterministic independent noise seeds, including SB3's seed=None resets."""

    def __init__(self, env, training_seed: int, worker: int, output: Path):
        super().__init__(env)
        self.training_seed, self.worker, self.output = training_seed, worker, output
        self.rng = np.random.default_rng(np.random.SeedSequence([training_seed, worker]))
        self.episode = 0

    def reset(self, *, seed=None, options=None):
        # SB3 sets a worker seed on the first reset. All later None resets
        # consume this reproducible stream instead of default_rng(None).
        if seed is not None:
            self.rng = np.random.default_rng(np.random.SeedSequence([seed, self.worker]))
        self.sensor_seed = int(self.rng.integers(1_000_000, 2**31 - 1))
        obs, info = self.env.reset(seed=self.sensor_seed, options=options)
        with self.output.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "training_seed": self.training_seed,
                        "worker": self.worker,
                        "episode": self.episode,
                        "sensor_seed": self.sensor_seed,
                    }
                )
                + "\n"
            )
        self.episode += 1
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, {**info, "sensor_episode_seed": self.sensor_seed}


class TrainingRewardScale(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            obs,
            reward * REWARD_SCALE,
            terminated,
            truncated,
            {**info, "raw_task_reward": reward},
        )


def make_training_env(training_seed: int, worker: int, directory: Path):
    torch.set_num_threads(1)
    task = make_task()
    stream = EpisodeSeedStream(
        task, training_seed, worker, directory / f"episode_seeds_{worker}.jsonl"
    )
    # Monitor sees raw rewards; PPO sees scaled rewards outside it.
    return TrainingRewardScale(Monitor(stream, str(directory / f"worker_{worker}.monitor.csv")))


def frozen_cases() -> dict:
    return {
        "development": [{"seed": seed, "duration_s": 45.0} for seed in DEVELOPMENT_SEEDS],
        "holdout": [
            {"seed": seed, "duration_s": duration}
            for duration in (45.0, 50.0)
            for seed in HOLDOUT_SEEDS
        ],
    }


def build_protocol(smoke=False) -> dict:
    return {
        "schema": "d1-continuous-sensor-ppo-experiment-v1",
        "smoke_only": smoke,
        "training_seeds": [SEEDS[0]] if smoke else list(SEEDS),
        "checkpoint_budgets": [1024] if smoke else list(BUDGETS),
        "n_envs": 4,
        "training_config": asdict(task_config()),
        "sensor_noise": asdict(NOISE),
        "noise_status": "illustrative per-sample noise, not identified hardware measurements",
        "sensor_delay_steps": 0,
        "episode_seed_rule": "worker SeedSequence -> integers [1000000,2147483647), including automatic resets",
        "state_source_schema": D1SensorStateSource.schema,
        "observation_schema": "d1-continuous-sensor-command45-v1",
        "action_schema": "d1-terrain-residual-quarter-v1",
        "control_schema": SENSOR_CONTROL_SCHEMA,
        "reward_schema": "d1-continuous-task-reward-v1",
        "raw_reward": "exp(-(vx_error/0.2)^2)+exp(-(yaw_rate_error/0.15)^2)-(true_clearance_error/0.08)^2-5*terminated",
        "training_reward_scale": REWARD_SCALE,
        "monitor_reward_units": "raw",
        "ppo_settings": PPO_SETTINGS,
        "policy_distribution": "standard_diagonal_gaussian",
        "residual_force_scale_n": [11.25, 20.0],
        "physical_action_mapping": "[ax,az]",
        "policy_gating": False,
        "stages_45_second_template": STAGES,
        "evaluation_cases": frozen_cases(),
        "checkpoint_rule": "fixed budget endpoints; no best-model selection",
        "development_rule": "32768:seed17;65536:seed29;131072:seeds17,29,43, all45s",
        "final_rule": "only final checkpoints; all independent seeds617,629,643 x durations45,50s",
        "replication_unit": "training seed, never control steps; evaluation noise seeds are paired",
        "smoke_evaluation_steps": 16 if smoke else None,
        "versions": {
            name: version(name)
            for name in ("mujoco", "torch", "stable_baselines3", "gymnasium", "numpy")
        },
        "python": platform.python_version(),
    }


def source_snapshot(output: Path) -> dict:
    # Record imported project dependencies, plus geometry assets. Excluding
    # unrelated in-progress scripts avoids false claims about their use.
    paths = {
        Path(__file__),
        ROOT / "scripts/run_d1_continuous_task.py",
        ROOT / "scripts/render_d1_continuous_task.py",
        ROOT / "pyproject.toml",
    }
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if (
            file
            and str(ROOT / "src/wheel_legged_control") in str(file)
            and str(file).endswith(".py")
        ):
            paths.add(Path(file).resolve())
    paths.update(
        path
        for path in (ROOT / "src/wheel_legged_control/d1/assets").rglob("*")
        if path.is_file() and path.suffix.lower() in {".xml", ".urdf", ".stl"}
    )
    hashes = {str(path.relative_to(ROOT)): ROLLOUT.sha256(path) for path in sorted(paths)}
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for path in sorted(paths):
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    ROLLOUT.write_json(
        output / "source.json",
        {"sha256": hashes, "archive_sha256": ROLLOUT.sha256(output / "source.tar.gz")},
    )
    return hashes


def verify_source(hashes: dict) -> None:
    changed = [name for name, expected in hashes.items() if ROLLOUT.sha256(ROOT / name) != expected]
    if changed:
        raise RuntimeError(f"frozen source changed during experiment: {changed}")


class TrainingAudit(BaseCallback):
    def __init__(self, directory: Path):
        super().__init__()
        self.directory = directory
        self.raw_actions, self.clipped_actions, self.raw_rewards, self.scaled_rewards = (
            [],
            [],
            [],
            [],
        )
        self.exposure, self.terrain_exposure = Counter(), Counter()
        self.rollouts = []

    def _on_step(self) -> bool:
        raw = np.asarray(self.locals["actions"], dtype=np.float32)
        clipped = np.asarray(self.locals["clipped_actions"], dtype=np.float32)
        np.testing.assert_array_equal(clipped, np.clip(raw, -1, 1))
        self.raw_actions.append(raw.copy())
        self.clipped_actions.append(clipped.copy())
        infos = self.locals["infos"]
        raw_reward = np.asarray([info["raw_task_reward"] for info in infos])
        # This callback executes before SB3 adds timeout value bootstrapping.
        np.testing.assert_allclose(
            self.locals["rewards"], raw_reward * REWARD_SCALE, rtol=1e-5, atol=1e-6
        )
        self.raw_rewards.append(raw_reward)
        self.scaled_rewards.append(np.asarray(self.locals["rewards"]).copy())
        self.exposure.update(info["stage"] for info in infos)
        self.terrain_exposure.update(info["terrain_section"] for info in infos)
        return True

    def _on_rollout_end(self) -> None:
        self.rollouts.append(
            {
                "timesteps": self.num_timesteps,
                "completed_ppo_updates_before_this_rollout": self.model._n_updates,
                "raw_reward_mean": float(np.mean(self.raw_rewards[-128:])),
                "clip_fraction": float(np.mean(np.abs(self.raw_actions[-128:]) > 1)),
            }
        )

    def save(self) -> dict:
        raw, clipped = np.asarray(self.raw_actions), np.asarray(self.clipped_actions)
        np.savez_compressed(
            self.directory / "training_samples.npz",
            raw_gaussian=raw,
            clipped_action=clipped,
            raw_reward=self.raw_rewards,
            scaled_reward=self.scaled_rewards,
        )
        summary = {
            "raw_shape": list(raw.shape),
            "physical_samples": int(raw.shape[0] * raw.shape[1]),
            "raw_mean": raw.mean(axis=(0, 1)).tolist(),
            "raw_std": raw.std(axis=(0, 1)).tolist(),
            "raw_min": raw.min(axis=(0, 1)).tolist(),
            "raw_max": raw.max(axis=(0, 1)).tolist(),
            "clipped_fraction_per_action": (np.abs(raw) > 1).mean(axis=(0, 1)).tolist(),
            "stage_steps": dict(self.exposure),
            "terrain_steps": dict(self.terrain_exposure),
        }
        ROLLOUT.write_json(self.directory / "action_exposure_summary.json", summary)
        ROLLOUT.write_json(self.directory / "rollout_progress.json", self.rollouts)
        return summary


@preserve_training_rng()
def checkpoint(
    model, directory: Path, seed: int, budget: int, hashes: dict, protocol_sha: str
) -> tuple[Path, dict]:
    directory.mkdir(parents=True)
    path = directory / "model.zip"
    model.save(path)
    metadata = {
        "training_seed": seed,
        "num_timesteps": int(model.num_timesteps),
        "requested_budget": budget,
        "checkpoint_rule": "fixed final endpoint of this training budget",
        "observation_schema": "d1-continuous-sensor-command45-v1",
        "action_schema": "d1-terrain-residual-quarter-v1",
        "control_schema": SENSOR_CONTROL_SCHEMA,
        "reward_schema": "d1-continuous-task-reward-v1",
        "state_source_schema": D1SensorStateSource.schema,
        "sensor_noise": asdict(NOISE),
        "sensor_delay_steps": 0,
        "residual_force_scale_n": [11.25, 20.0],
        "physical_action_mapping": "[ax,az]",
        "policy_distribution": "standard_diagonal_gaussian",
        "model_sha256": ROLLOUT.sha256(path),
        "source_sha256": hashes,
        "protocol_sha256": protocol_sha,
        "training_reward_scale": REWARD_SCALE,
    }
    ROLLOUT.write_json(directory / "metadata.json", metadata)
    loaded = PPO.load(path, device="cpu")
    validation_task = make_task()
    ROLLOUT.validate_checkpoint(metadata, loaded, validation_task, path)
    for name, value in model.policy.state_dict().items():
        torch.testing.assert_close(value, loaded.policy.state_dict()[name], rtol=0, atol=0)
    return path, metadata


@preserve_training_rng()
def evaluate(
    model,
    metadata: dict | None,
    directory: Path,
    case: dict,
    hashes: dict,
    *,
    smoke_steps: int | None = None,
) -> dict:
    directory.mkdir(parents=True)
    task = make_task(case["duration_s"])
    obs, _ = task.reset(seed=case["seed"])
    compiled_model = ROLLOUT.store_compiled_model(task.plant.model, directory)
    protocol = {
        "task_schema": TASK_SCHEMA,
        "config": asdict(task.config),
        "mode": "policy" if model else "zero",
        "seed": case["seed"],
        "control_dt_s": task.plant.control_dt,
        "road": task.road,
        "observation_schema": task.observation_schema,
        "control_schema": task.control_schema,
        "action_schema": task.action_schema,
        "sensor_noise": asdict(NOISE),
        "sensor_delay_steps": 0,
        "checkpoint_sha256": metadata["model_sha256"] if metadata else None,
        "checkpoint_metadata": metadata,
        "compiled_model_sha256": compiled_model["uncompressed_sha256"],
        "episode_reset_count": 1,
        "interpretation": "actual matching sensor-command45 policy rollout"
        if model
        else "actual paired sensor/noise zero-residual baseline",
        "smoke_steps": smoke_steps,
    }
    ROLLOUT.write_json(directory / "protocol.json", protocol)
    qpos, qvel = task.plant.simulation_state()
    positions, velocities, times, rows = [qpos.copy()], [qvel.copy()], [0.0], []
    for _ in range(min(task.max_steps, smoke_steps or task.max_steps)):
        action = np.zeros(2) if model is None else model.predict(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, info = task.step(action)
        rows.append({**info, "reward": reward})
        qpos, qvel = task.plant.simulation_state()
        positions.append(qpos.copy())
        velocities.append(qvel.copy())
        times.append(info["time_s"])
        if terminated or truncated:
            break
    with (directory / "telemetry.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(directory / "states.npz", qpos=positions, qvel=velocities, time_s=times)
    summary = summarize_task(rows)
    summary.update(
        {
            "evaluation_seed": case["seed"],
            "planned_duration_s": case["duration_s"],
            "checkpoint_sha256": metadata["model_sha256"] if metadata else None,
            "interpretation": "one fixed-road sensor/noise rollout; use paired cases and training seeds for comparisons",
            "mean_abs_action": np.mean(
                [[abs(row["action_longitudinal"]), abs(row["action_vertical"])] for row in rows],
                axis=0,
            ).tolist(),
        }
    )
    ROLLOUT.write_json(directory / "summary.json", summary)
    ROLLOUT.write_json(
        directory / "manifest.json",
        {
            "schema": "d1-continuous-rollout-artifact-v1",
            "compiled_model": compiled_model,
            "sha256": {
                path.name: ROLLOUT.sha256(path) for path in directory.iterdir() if path.is_file()
            },
            "source_sha256": hashes,
        },
    )
    return summary


def write_manifest(output: Path) -> None:
    ROLLOUT.write_json(
        output / "manifest.json",
        {
            "schema": "d1-continuous-sensor-ppo-artifacts-v1",
            "sha256": {
                str(path.relative_to(output)): ROLLOUT.sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )


def run(args) -> dict:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    protocol = build_protocol(args.smoke)
    # Protocol and independent case list are persisted before any training.
    ROLLOUT.write_json(args.output / "protocol.json", protocol)
    probe = make_task()
    del probe
    hashes = source_snapshot(args.output)
    protocol_sha = ROLLOUT.sha256(args.output / "protocol.json")
    results, zero_results = [], {}
    evaluation_cases = frozen_cases()
    baseline_cases = (
        [evaluation_cases["development"][0]]
        if args.smoke
        else [case for split in evaluation_cases.values() for case in split]
    )
    for case in baseline_cases:
        key = f"seed{case['seed']}_{case['duration_s']:g}s"
        zero_results[key] = evaluate(
            None,
            None,
            args.output / "rollouts" / f"zero_{key}",
            case,
            hashes,
            smoke_steps=16 if args.smoke else None,
        )
    for seed in protocol["training_seeds"]:
        directory = args.output / f"training_seed{seed}"
        directory.mkdir()
        vec = SubprocVecEnv(
            [partial(make_training_env, seed, worker, directory) for worker in range(4)],
            start_method="spawn",
        )
        audit = TrainingAudit(directory)
        try:
            model = PPO("MlpPolicy", vec, seed=seed, device="cpu", verbose=0, **PPO_SETTINGS)
            start = time.perf_counter()
            for budget in protocol["checkpoint_budgets"]:
                verify_source(hashes)
                model.learn(
                    total_timesteps=budget - model.num_timesteps,
                    reset_num_timesteps=model.num_timesteps == 0,
                    callback=audit,
                )
                path, metadata = checkpoint(
                    model,
                    args.output / "checkpoints" / f"seed{seed}" / f"step{budget}",
                    seed,
                    budget,
                    hashes,
                    protocol_sha,
                )
                print(
                    json.dumps(
                        {
                            "checkpoint": str(path),
                            "seed": seed,
                            "timesteps": model.num_timesteps,
                            "elapsed_s": time.perf_counter() - start,
                        }
                    ),
                    flush=True,
                )
                cases = (
                    [evaluation_cases["development"][0]]
                    if args.smoke
                    else evaluation_cases["development"]
                    if budget == BUDGETS[-1]
                    else [evaluation_cases["development"][0 if budget == BUDGETS[0] else 1]]
                )
                for case in cases:
                    key = f"seed{case['seed']}_{case['duration_s']:g}s"
                    report = evaluate(
                        model,
                        metadata,
                        args.output / "rollouts" / f"dev_train{seed}_step{budget}_{key}",
                        case,
                        hashes,
                        smoke_steps=16 if args.smoke else None,
                    )
                    results.append(
                        {
                            "split": "development",
                            "training_seed": seed,
                            "budget": budget,
                            "case": key,
                            **report,
                        }
                    )
            if not args.smoke:
                for case in evaluation_cases["holdout"]:
                    key = f"seed{case['seed']}_{case['duration_s']:g}s"
                    report = evaluate(
                        model,
                        metadata,
                        args.output / "rollouts" / f"holdout_train{seed}_{key}",
                        case,
                        hashes,
                    )
                    results.append(
                        {
                            "split": "holdout",
                            "training_seed": seed,
                            "budget": budget,
                            "case": key,
                            **report,
                        }
                    )
            audit.save()
        finally:
            vec.close()
    verify_source(hashes)
    summary = {
        "smoke_only": args.smoke,
        "zero_baselines": zero_results,
        "policy_evaluations": results,
        "selection": "all fixed-budget checkpoints retained; no evaluation-based policy/case selection",
    }
    ROLLOUT.write_json(args.output / "summary.json", summary)
    write_manifest(args.output)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1024 real training samples; 16-step dev plumbing only, no holdout",
    )
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "smoke_only": result["smoke_only"],
                "policy_evaluations": len(result["policy_evaluations"]),
            }
        )
    )
