"""Fixed-budget CPU experiments for the shared command-conditioned D1 task.

Benchmark first, then train fixed endpoints and evaluate paired independent
episodes. Evaluation reports tracking and actual geometric exposure, not only
episode completion. No holdout-driven checkpoint selection is implemented.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import tarfile
from dataclasses import asdict
from functools import partial
from importlib.metadata import version
from numbers import Integral
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import psutil
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from wheel_legged_control.d1.locomotion_env import (
    LOCOMOTION_SPAWN_POSITION_M,
    D1LocomotionEnv,
    D1LocomotionRandomization,
)
from wheel_legged_control.d1.locomotion_terrain import (
    D1LocomotionTerrainConfig,
    locomotion_terrain_configs,
)
from wheel_legged_control.d1.observation_history import D1ObservationHistory
from wheel_legged_control.d1.sensor_estimation import D1SensorNoise
from wheel_legged_control.d1.state_estimation import D1EstimatorImpairments
from wheel_legged_control.d1.state_provider import D1StateProviderConfig
from wheel_legged_control.d1.wheel_leg_controller import D1WheelLegControlConfig

ROOT = Path(__file__).resolve().parents[1]
NOISE = D1SensorNoise(
    gyro_std_rad_s=0.002,
    accelerometer_std_m_s2=0.03,
    encoder_position_std_rad=0.0005,
    encoder_velocity_std_rad_s=0.005,
)
QUALITY = {
    "velocity_rmse_mps": 0.08,
    "yaw_rmse_rps": 0.08,
    "height_rmse_m": 0.025,
    "attitude_rmse_rad": 0.15,
}
PPO_SETTINGS = {
    "learning_rate": 3e-4,
    "n_steps": 128,
    "batch_size": 128,
    "n_epochs": 4,
    "gamma": math.exp(-0.01 / 2.0),
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "policy_kwargs": {"net_arch": [64, 64], "log_std_init": -2.0},
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(directory):
    paths = [Path(__file__), ROOT / "pyproject.toml", *sorted((ROOT / "src").rglob("*.py"))]
    paths += sorted(
        p
        for p in (ROOT / "src/wheel_legged_control/d1/assets").rglob("*")
        if p.is_file() and p.suffix.lower() in {".xml", ".urdf", ".stl"}
    )
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    with tarfile.open(directory / "source.tar.gz", "w:gz") as archive:
        for p in paths:
            archive.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
    write_json(
        directory / "source.json",
        {"sha256": hashes, "archive_sha256": sha256(directory / "source.tar.gz")},
    )
    return hashes


def verify_source(directory, hashes):
    changed = [
        name
        for name, expected in hashes.items()
        if not (ROOT / name).is_file() or sha256(ROOT / name) != expected
    ]
    write_json(
        directory / "source_consistency.json", {"unchanged": not changed, "changed": changed}
    )
    if changed:
        raise RuntimeError(f"source changed during experiment: {changed}")


def provider_config(kind, delay=0):
    if isinstance(delay, (bool, np.bool_)) or not isinstance(delay, Integral) or delay < 0:
        raise ValueError("measurement delay must be a nonnegative integer")
    if kind == "oracle" and delay:
        raise ValueError("oracle cannot have measurement delay")
    if kind == "imu_encoder_fusion":
        return D1StateProviderConfig(
            kind,
            sensor_noise=NOISE,
            sensor_delay_steps=delay,
            initial_position_m=LOCOMOTION_SPAWN_POSITION_M,
            initial_rpy_rad=(0, 0, 0),
        )
    return D1StateProviderConfig(
        kind,
        impairments=D1EstimatorImpairments(delay_steps=delay)
        if kind == "truth_impairment"
        else None,
    )


def wheel_control_config(args):
    overrides = {
        name: getattr(args, name, None)
        for name in (
            "wheel_kp",
            "wheel_ki",
            "yaw_feedback_gain",
            "leg_feedback_scale",
            "attitude_feedback_scale",
        )
    }
    overrides = {name: value for name, value in overrides.items() if value is not None}
    if overrides and args.baseline != "wheel_leg":
        raise ValueError("wheel gains and feedback scales require --baseline wheel_leg")
    return D1WheelLegControlConfig(**overrides) if overrides else None


class EpisodeRecord(gym.Wrapper):
    """One reset seed stream owned by Gym; None resets advance it reproducibly."""

    def __init__(self, env, path):
        super().__init__(env)
        self.path = path
        self.episode = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        with self.path.open("a") as stream:
            stream.write(
                json.dumps({"episode": self.episode, **info["episode_metadata"]}, allow_nan=False)
                + "\n"
            )
        self.episode += 1
        return observation, info


def make_env(
    baseline,
    worker,
    duration_s,
    source,
    history,
    delay_randomization=False,
    directory=None,
    wheel_control=None,
):
    torch.set_num_threads(1)
    randomization = (
        D1LocomotionRandomization(
            measurement_delay_steps=(0, 2),
            actuator_delay_steps=(0, 3),
            actuator_time_constant_s=(0.0, 0.006),
            actuator_gain=(0.95, 1.05),
        )
        if delay_randomization
        else None
    )
    env = D1LocomotionEnv(
        baseline=baseline,
        episode_seconds=duration_s,
        terrain=locomotion_terrain_configs("train")[worker % 4],
        provider_config=provider_config(source),
        randomization=randomization,
        wheel_leg_control=wheel_control,
    )
    if directory is not None:
        env = EpisodeRecord(env, directory / f"worker{worker}_episodes.jsonl")
        env = Monitor(env, str(directory / f"worker{worker}"))
    return D1ObservationHistory(env, history_length=history)


def make_vec(args, workers, directory=None):
    factories = [
        partial(
            make_env,
            args.baseline,
            worker,
            args.duration,
            args.source,
            args.history,
            args.delay_randomization,
            directory,
            wheel_control=wheel_control_config(args),
        )
        for worker in range(workers)
    ]
    return (
        DummyVecEnv(factories) if workers == 1 else SubprocVecEnv(factories, start_method="spawn")
    )


def process_memory():
    process = psutil.Process()
    items = [process, *process.children(recursive=True)]
    rss = sum(p.memory_info().rss for p in items if p.is_running())
    # Shared pages can be counted more than once in RSS; USS excludes them.
    uss = sum(p.memory_full_info().uss for p in items if p.is_running())
    return {
        "process_tree_rss_mb": rss / 2**20,
        "process_tree_uss_mb": uss / 2**20,
        "system_available_mb": psutil.virtual_memory().available / 2**20,
    }


def benchmark(args, directory):
    results = []
    for workers in (1, 2, 4):
        vector = make_vec(args, workers)
        try:
            vector.seed(args.seed)
            vector.reset()
            action = np.zeros((workers, *vector.action_space.shape), np.float32)
            for _ in range(25):
                vector.step(action)
            samples, timings, memory = 0, [], []
            start = perf_counter()
            while samples < args.steps:
                before = perf_counter()
                vector.step(action)
                timings.append((perf_counter() - before) * 1e3)
                samples += workers
                if samples % (workers * 100) == 0:
                    memory.append(process_memory())
            elapsed = perf_counter() - start
            memory.append(process_memory())
            result = {
                "workers": workers,
                "samples": samples,
                "wall_s": elapsed,
                "samples_per_second": samples / elapsed,
                "vector_step_median_ms": float(np.median(timings)),
                "vector_step_p99_ms": float(np.percentile(timings, 99)),
                "max_observed_tree_uss_mb": max(m["process_tree_uss_mb"] for m in memory),
                "max_observed_tree_rss_mb": max(m["process_tree_rss_mb"] for m in memory),
                "min_observed_system_available_mb": min(m["system_available_mb"] for m in memory),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            write_json(directory / "benchmark.json", results)
        finally:
            vector.close()
    return results


class RolloutAudit(BaseCallback):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.records = []
        self.rows = []

    def _on_step(self):
        raw, applied = (
            np.asarray(self.locals["actions"]),
            np.asarray(self.locals["clipped_actions"]),
        )
        for worker, info in enumerate(self.locals["infos"]):
            self.rows.append(
                {
                    "worker": worker,
                    "sample": self.num_timesteps,
                    "reward": float(self.locals["rewards"][worker]),
                    "raw_action_rms": float(np.sqrt(np.mean(raw[worker] ** 2))),
                    "applied_action_rms": float(np.sqrt(np.mean(applied[worker] ** 2))),
                    "raw_action_clipped_fraction": float(np.mean(raw[worker] != applied[worker])),
                    "nonflat_now": info["terrain_exposure"]["nonflat_now"],
                    "terminated": info["terminal_reason"] not in (None, "time_limit"),
                    **{
                        f"raw_action_{axis}": float(value) for axis, value in enumerate(raw[worker])
                    },
                    **{
                        f"applied_action_{axis}": float(value)
                        for axis, value in enumerate(applied[worker])
                    },
                    **info["metrics"],
                    **{f"reward_{name}": value for name, value in info["reward_terms"].items()},
                }
            )
        return True

    def _on_rollout_end(self):
        keys = tuple(self.rows[0])
        path = self.directory / "training_samples.csv"
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerows(self.rows)
        self.records.append(
            {
                "num_timesteps": self.num_timesteps,
                "samples": len(self.rows),
                "nonflat_fraction": float(np.mean([r["nonflat_now"] for r in self.rows])),
                "terminated_episodes": sum(r["terminated"] for r in self.rows),
            }
        )
        write_json(self.directory / "rollout_exposure.json", self.records)
        self.rows.clear()


def train(args, directory):
    from wheel_legged_control.d1.locomotion_checkpoint import write_checkpoint_metadata
    from wheel_legged_control.d1.ppo_update_audit import AuditedPPO

    vector = make_vec(args, args.workers, directory)
    try:
        learner = AuditedPPO(
            "MlpPolicy",
            vector,
            seed=args.seed,
            device="cpu",
            verbose=0,
            audit_directory=directory / "updates",
            **PPO_SETTINGS,
        )
        before = perf_counter()
        learner.learn(total_timesteps=args.steps, callback=RolloutAudit(directory))
        wall = perf_counter() - before
        learner.save(directory / "checkpoint.zip")
        reference = make_env(
            args.baseline,
            0,
            args.duration,
            args.source,
            args.history,
            args.delay_randomization,
            wheel_control=wheel_control_config(args),
        )
        try:
            reference.reset(seed=args.seed)
            write_checkpoint_metadata(
                directory / "checkpoint.zip",
                reference,
                directory / "checkpoint.json",
                extra={"training_seed": args.seed, "num_timesteps": learner.num_timesteps},
            )
        finally:
            reference.close()
        result = {
            "num_timesteps": learner.num_timesteps,
            "wall_s": wall,
            "samples_per_second": learner.num_timesteps / wall,
            "checkpoint_sha256": sha256(directory / "checkpoint.zip"),
            "updates": learner._n_updates,
        }
        write_json(directory / "training.json", result)
        print(json.dumps(result), flush=True)
        return result
    finally:
        vector.close()


def summarize(rows, info):
    rms = lambda key: float(np.sqrt(np.mean([row[key] ** 2 for row in rows])))
    summary = {
        "executed_steps": len(rows),
        "duration_s": rows[-1]["time_s"],
        "completed": info["terminal_reason"] == "time_limit",
        "terminal_reason": info["terminal_reason"],
        "velocity_rmse_mps": rms("velocity_error_mps"),
        "yaw_rmse_rps": rms("yaw_rate_error_rps"),
        "height_rmse_m": rms("height_error_m"),
        "attitude_rmse_rad": math.sqrt(rms("roll_error_rad") ** 2 + rms("pitch_error_rad") ** 2),
        "mean_mechanical_power_w": float(np.mean([r["mechanical_power_w"] for r in rows])),
        "action_rms": float(np.sqrt(np.mean([r["action_mean_square"] for r in rows]))),
        "terrain_exposure": info["terrain_exposure"],
    }
    summary["quality_pass"] = bool(
        summary["completed"] and all(summary[name] <= limit for name, limit in QUALITY.items())
    )
    return summary


def evaluate(args, directory):
    from wheel_legged_control.d1.locomotion_checkpoint import load_locomotion_policy

    results = []
    configs = (
        (D1LocomotionTerrainConfig(),)
        if args.split == "flat"
        else locomotion_terrain_configs(args.split)
    )
    seeds = (17, 29) if args.split != "holdout" else (617, 629)
    for terrain_index, terrain in enumerate(configs):
        for seed in seeds:
            name = f"road{terrain_index}_seed{seed}"
            base = D1LocomotionEnv(
                baseline=args.baseline,
                episode_seconds=args.duration,
                terrain=terrain,
                command_mode="holdout" if args.split == "holdout" else "development",
                provider_config=provider_config(args.source, args.measurement_delay),
                wheel_leg_control=wheel_control_config(args),
            )
            env = D1ObservationHistory(base, args.history)
            try:
                obs, reset_info = env.reset(seed=seed)
                model = (
                    load_locomotion_policy(args.policy, args.metadata, env) if args.policy else None
                )
                # Loader validation must not reset the environment or draw action samples.
                rows, states, actions = [], [base.plant.data.qpos.copy()], []
                while True:
                    action = (
                        model.predict(obs, deterministic=True)[0]
                        if model is not None
                        else np.zeros(env.action_space.shape, np.float32)
                    )
                    obs, reward, terminated, truncated, info = env.step(action)
                    rows.append(
                        {
                            **info["metrics"],
                            **{f"command_{k}": v for k, v in info["command"].items()},
                            "reward": reward,
                            "action_mean_square": float(np.mean(np.square(action))),
                            "nonflat_now": info["terrain_exposure"]["nonflat_now"],
                            **{f"reward_{k}": v for k, v in info["reward_terms"].items()},
                        }
                    )
                    states.append(base.plant.data.qpos.copy())
                    actions.append(action.copy())
                    if terminated or truncated:
                        break
                with (directory / f"{name}.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(rows)
                np.savez_compressed(
                    directory / f"{name}.npz", qpos=np.asarray(states), actions=np.asarray(actions)
                )
                result = {
                    "case": name,
                    "seed": seed,
                    "terrain": asdict(terrain),
                    **summarize(rows, info),
                }
                results.append(result)
                write_json(directory / f"{name}_episode.json", reset_info["episode_metadata"])
                write_json(directory / "evaluation.json", results)
                print(json.dumps(result), flush=True)
            finally:
                env.close()
    return results


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("benchmark", "train", "evaluate"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", choices=("wheel_leg", "lqr", "mpc"), default="wheel_leg")
    p.add_argument("--wheel-kp", type=float)
    p.add_argument("--wheel-ki", type=float)
    p.add_argument("--yaw-feedback-gain", type=float)
    p.add_argument(
        "--leg-feedback-scale", type=float, help="multiplies the nominal leg PD pair (80/3)"
    )
    p.add_argument(
        "--attitude-feedback-scale",
        type=float,
        help="multiplies the nominal body attitude pair (180/24)",
    )
    p.add_argument(
        "--source", choices=("oracle", "truth_impairment", "imu_encoder_fusion"), default="oracle"
    )
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--history", type=int, default=1)
    p.add_argument("--seed", type=int, default=24000)
    p.add_argument("--steps", type=int, default=32768)
    p.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    p.add_argument("--delay-randomization", action="store_true")
    p.add_argument("--measurement-delay", type=int, default=0)
    p.add_argument("--split", choices=("flat", "development", "holdout"), default="development")
    p.add_argument("--policy", type=Path)
    p.add_argument("--metadata", type=Path)
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        wheel_control_config(args)
    except (TypeError, ValueError) as exc:
        p.error(str(exc))
    if (
        args.steps <= 0
        or args.history < 1
        or not math.isfinite(args.duration)
        or args.duration < 0.01
        or args.seed < 0
    ):
        p.error("steps/history/duration must be positive and seed nonnegative")
    if not math.isclose(args.duration / 0.01, round(args.duration / 0.01), rel_tol=0, abs_tol=1e-8):
        p.error("duration must be an integer number of 10 ms ticks")
    if not 0 <= args.measurement_delay <= 20:
        p.error("measurement delay must be between 0 and 20 decision steps")
    if args.mode != "evaluate" and (args.measurement_delay or args.policy or args.metadata):
        p.error("measurement-delay/policy/metadata are evaluation-only")
    if args.mode == "evaluate" and args.delay_randomization:
        p.error("evaluation uses fixed delay, not training delay randomization")
    if args.mode != "evaluate" and args.split != "development":
        p.error("split selects evaluation roads only; training always uses its four fixed roads")
    if args.delay_randomization and args.source == "oracle":
        p.error("measurement delay randomization requires an impaired or sensor source")
    if args.measurement_delay and args.source == "oracle":
        p.error("oracle cannot have measurement delay")
    if bool(args.policy) != bool(args.metadata):
        p.error("policy and metadata must be supplied together")
    if args.policy and (not args.policy.is_file() or not args.metadata.is_file()):
        p.error("policy and metadata files must exist")
    if args.mode == "train" and args.steps % (args.workers * PPO_SETTINGS["n_steps"]):
        p.error("training steps must be an exact rollout-budget multiple")
    if args.output.exists():
        p.error("output must be a new directory; failed runs are retained")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    protocol = {
        "schema": "d1-command-locomotion-experiment-v1",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "ppo_settings": PPO_SETTINGS,
        "quality_thresholds": QUALITY,
        "selection_rule": "fixed final budget; no holdout checkpoint selection",
        "reward_scale": 1.0,
        "discount_horizon_s": 2.0,
        "replication_unit": "training seed, not episode steps",
        "sensor_noise": asdict(NOISE),
        "noise_origin": "synthetic, not calibrated D1 data",
        "terrain_splits": {
            split: [asdict(config) for config in locomotion_terrain_configs(split)]
            for split in ("train", "development", "holdout")
        },
        "evaluation_seeds": {"development": [17, 29], "holdout": [617, 629]},
        "command_splits": "development and holdout schedules in frozen locomotion_commands.py; full targets logged per episode",
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in ("mujoco", "torch", "stable_baselines3", "gymnasium", "numpy")
        },
    }
    write_json(args.output / "protocol.json", protocol)
    hashes = snapshot(args.output)
    try:
        {"benchmark": benchmark, "train": train, "evaluate": evaluate}[args.mode](args, args.output)
        verify_source(args.output, hashes)
    except Exception as exc:
        write_json(args.output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
