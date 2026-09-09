"""Fresh, paired PPO runs for full Gaussian, bounded mean, and Fx-only actions.

All controls, rewards, samplers and budgets are shared. The mean ablation
changes only mu=tanh(logits); the Fx ablation removes the vertical action.
Intermediate checkpoints see development cases only. Final holdout evaluation
is a separate operation after all candidates have finished training.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import tarfile
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from wheel_legged_control.d1.ppo_action_policies import BoundedMeanActorCriticPolicy
from wheel_legged_control.d1.residual_action_env import D1LongitudinalResidualEnv
from wheel_legged_control.d1.terrain_tracking_env import D1TerrainTrackingEnv

if __package__:
    from . import run_d1_terrain_curriculum as shared
else:
    import run_d1_terrain_curriculum as shared

VARIANTS = ("full_gaussian", "bounded_mean", "fx_only")
INTERMEDIATE_CASES = (0, 4, 5, 10, 11, 22)
ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def preserve_training_rng():
    """Checkpoint loading/evaluation must not reset the CPU training stream."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


class InitializationSpaceEnv(gym.Env):
    """Only supplies spaces for the common two-action initialization, never stepped."""

    observation_space = spaces.Box(-5.0, 5.0, (44,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(44, dtype=np.float32), {}

    def step(self, action):
        raise RuntimeError("initialization-only environment must never be stepped")


def parameter_sha256(state_dict):
    digest = hashlib.sha256()
    for key, tensor in sorted(state_dict.items()):
        array = tensor.detach().cpu().numpy()
        digest.update(key.encode() + str(array.shape).encode() + str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def initialize_from_common_policy(model, seed):
    common = PPO(
        "MlpPolicy", InitializationSpaceEnv(), **shared.PPO_SETTINGS, seed=seed, device="cpu"
    )
    common_state = common.policy.state_dict()
    target = model.policy.state_dict()
    mapped = {}
    for key, value in common_state.items():
        if target[key].shape == value.shape:
            mapped[key] = value.clone()
        elif (
            key in {"action_net.weight", "action_net.bias", "log_std"} and target[key].shape[0] == 1
        ):
            mapped[key] = value[:1].clone()
        else:
            raise ValueError(f"unregistered initialization projection: {key}")
    model.policy.load_state_dict(mapped, strict=True)
    model.set_random_seed(seed)
    return {
        "common_two_action_parameter_sha256": parameter_sha256(common_state),
        "initialized_parameter_sha256": parameter_sha256(mapped),
        "projection": "identity"
        if model.action_space.shape == (2,)
        else "copy all shared tensors; first action-head row and log_std only",
    }


def frozen_holdout_cases():
    """New parameters frozen before this experiment; never used for development."""
    cases = shared.evaluation_cases("development", "tracking-v2")
    for i, case in enumerate(cases):
        case["case_id"] = f"action_v3_holdout_{i:02d}_{case['terrain']['kind']}"
        case["environment_seed"] = 55000 + i
        if case["terrain"]["kind"] == "bumps":
            case["terrain"]["wavelength_m"] = (0.85, 1.05, 1.25)[((i - 4) % 6) // 2]
            case["terrain"]["phase_rad"] = (0.91, 2.61)[(i - 4) % 2]
        elif case["terrain"]["kind"] == "ramp":
            case["terrain"]["slope_deg"] = (-3.75, -2.75, 2.75, 3.75)[(i - 16) // 2]
    return cases


class ActionDiagnostics(BaseCallback):
    """Record actual sampled pre-clip actions and their distributions during training."""

    def __init__(self):
        super().__init__()
        self.samples = {
            key: []
            for key in ("timesteps", "raw_mean", "std", "raw_action", "executed_action", "log_prob")
        }
        self.rollout_rows = []
        self._start = 0
        self.max_unchanged_log_prob_error = 0.0

    def _on_step(self):
        raw = np.asarray(self.locals["actions"]).copy()
        clipped = np.asarray(self.locals["clipped_actions"]).copy()
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(self.locals["obs_tensor"])
            mu = distribution.distribution.mean.cpu().numpy().copy()
            std = distribution.distribution.stddev.cpu().numpy().copy()
            lp = distribution.log_prob(torch.as_tensor(raw, device=self.model.device)).cpu().numpy()
        old_lp = self.locals["log_probs"].cpu().numpy()
        error = float(np.max(np.abs(lp - old_lp)))
        self.max_unchanged_log_prob_error = max(self.max_unchanged_log_prob_error, error)
        if error > 1e-5:
            raise RuntimeError("unchanged policy does not reproduce sampled-action log_prob")
        for key, value in (
            ("timesteps", np.full((len(raw),), self.num_timesteps, dtype=np.int64)),
            ("raw_mean", mu),
            ("std", std),
            ("raw_action", raw),
            ("executed_action", clipped),
            ("log_prob", old_lp.copy()),
        ):
            self.samples[key].append(value)
        return True

    def _on_rollout_end(self):
        raw = np.concatenate(self.samples["raw_action"][self._start :])
        clipped = np.concatenate(self.samples["executed_action"][self._start :])
        mu = np.concatenate(self.samples["raw_mean"][self._start :])
        std = np.concatenate(self.samples["std"][self._start :])
        self._start = len(self.samples["raw_action"])
        row = {
            "timesteps": self.num_timesteps,
            "samples": len(raw),
            "max_unchanged_log_prob_error": self.max_unchanged_log_prob_error,
        }
        for axis in range(raw.shape[1]):
            row.update(
                {
                    f"mu_mean_{axis}": float(mu[:, axis].mean()),
                    f"mu_abs_max_{axis}": float(np.abs(mu[:, axis]).max()),
                    f"std_mean_{axis}": float(std[:, axis].mean()),
                    f"sample_clip_fraction_{axis}": float(
                        np.mean(raw[:, axis] != clipped[:, axis])
                    ),
                    f"deterministic_upper_fraction_{axis}": float(np.mean(mu[:, axis] >= 1)),
                    f"executed_action_mean_{axis}": float(clipped[:, axis].mean()),
                }
            )
        self.rollout_rows.append(row)

    def save(self, output):
        np.savez_compressed(
            output / "training_action_samples.npz",
            **{k: np.concatenate(v) for k, v in self.samples.items()},
        )
        shared._write_csv(
            output / "training_action_diagnostics.csv",
            self.rollout_rows,
            tuple(self.rollout_rows[0]),
        )


class PhysicalActionPolicy:
    """Explicit Fx-only embedding into the identical two-force evaluation plant."""

    def __init__(self, model, variant):
        self.model, self.variant = model, variant
        self.diagnostics = []

    def predict(self, observation, deterministic=True):
        action, state = self.model.predict(observation, deterministic=deterministic)
        with torch.no_grad():
            tensor, _ = self.model.policy.obs_to_tensor(observation)
            distribution = self.model.policy.get_distribution(tensor).distribution
            mu = distribution.mean.cpu().numpy()[0]
            std = distribution.stddev.cpu().numpy()[0]
        self.diagnostics.append(
            {
                "raw_mean_x": float(mu[0]),
                "std_x": float(std[0]),
                "raw_mean_z": float(mu[1]) if len(mu) == 2 else 0.0,
                "std_z": float(std[1]) if len(std) == 2 else 0.0,
                "vertical_channel_active": int(len(mu) == 2),
            }
        )
        if self.variant == "fx_only":
            if action.shape != (1,):
                raise ValueError("Fx-only model must have exactly one action")
            action = np.asarray((action[0], 0.0), dtype=np.float32)
        elif action.shape != (2,):
            raise ValueError("full residual model must have exactly two actions")
        return action, state


def evaluate(model, variant, cases, output):
    output.mkdir(parents=True, exist_ok=False)
    metrics = []
    for case in cases:
        policy = None if model is None else PhysicalActionPolicy(model, variant)
        rows, summary = shared.evaluate_case(
            D1TerrainTrackingEnv,
            case,
            4.0,
            model=policy,
            quality_criteria=shared.TRACKING_QUALITY_CRITERIA,
        )
        if policy is not None:
            for row, diagnostics in zip(rows, policy.diagnostics, strict=True):
                row.update(diagnostics)
        shared._write_csv(output / f"{case['case_id']}.csv", rows, tuple(rows[0]))
        metrics.append({"variant": variant, "case_id": case["case_id"], **summary})
    shared._write_csv(output / "metrics.csv", metrics, tuple(metrics[0]))
    return metrics


def loaded_sources():
    loaded = {
        str(Path(module.__file__).resolve())
        for module in list(sys.modules.values())
        if getattr(module, "__file__", None)
    }
    files = {
        name: digest
        for name, digest in shared._source_hashes().items()
        if not name.endswith(".py") or str((ROOT / name).resolve()) in loaded
    }
    files[str(Path(__file__).resolve().relative_to(ROOT))] = shared._sha256(Path(__file__))
    return files


def snapshot_sources(output, sources):
    with tarfile.open(output / "runtime_source.tar.gz", "w:gz") as archive:
        for relative in sorted(sources):
            archive.add(ROOT / relative, arcname=relative, recursive=False)


def train_variant(args, variant, seed, sources):
    output = args.output / f"{variant}_seed_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    environment = D1LongitudinalResidualEnv if variant == "fx_only" else D1TerrainTrackingEnv
    vec = make_vec_env(
        environment,
        n_envs=args.envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv if args.envs > 1 else DummyVecEnv,
        vec_env_kwargs={"start_method": "spawn"} if args.envs > 1 else None,
        env_kwargs={
            "training_mode": "mixed",
            "episode_seconds": 4.0,
            "randomize": False,
            "stage": 3,
        },
        monitor_dir=str(output / "episodes"),
        wrapper_class=partial(shared.TrainingRewardScale, scale=0.01),
    )
    logger = None
    try:
        policy = BoundedMeanActorCriticPolicy if variant == "bounded_mean" else "MlpPolicy"
        model = PPO(policy, vec, **shared.PPO_SETTINGS, seed=seed, device="cpu", verbose=0)
        initialization = initialize_from_common_policy(model, seed)
        logger = configure(str(output / "learning_metrics"), ["csv"])
        model.set_logger(logger)
        exposure = shared._curriculum_callback(BaseCallback, "mixed", args.budgets[-1])
        action_log = ActionDiagnostics()
        callbacks = CallbackList([exposure, action_log])
        started = time.perf_counter()
        development = shared.evaluation_cases("development", "tracking-v2")
        development_scores = []
        for target in args.budgets:
            model.learn(
                total_timesteps=target - model.num_timesteps,
                callback=callbacks,
                reset_num_timesteps=False,
            )
            logger.dump(step=model.num_timesteps)
            if model.num_timesteps != target:
                raise RuntimeError("budget did not end on the declared PPO rollout")
            checkpoint = output / f"checkpoint_{target}.zip"
            model.save(checkpoint)
            cases = (
                development
                if target == args.budgets[-1]
                else [development[i] for i in INTERMEDIATE_CASES]
            )
            with preserve_training_rng():
                loaded = PPO.load(checkpoint, device="cpu")
                scores = evaluate(loaded, variant, cases, output / f"development_{target}")
            development_scores.extend(
                {"training_seed": seed, "timesteps": target, **score} for score in scores
            )
            print(
                f"{variant} seed={seed} steps={target} development_quality={sum(x['quality_success'] for x in scores)}/{len(scores)}",
                flush=True,
            )
        final = output / "model.zip"
        model.save(final)
        action_log.save(output)
        shared._write_json(output / "episode_starts.json", exposure.episode_starts)
        shared._write_csv(
            output / "development_curve.csv", development_scores, tuple(development_scores[0])
        )
        metadata = {
            "variant": variant,
            "training_seed": seed,
            "actual_timesteps": model.num_timesteps,
            "ppo_epochs_completed": model._n_updates,
            "ppo": shared.PPO_SETTINGS,
            "observation_schema": environment.observation_schema,
            "action_schema": environment.action_schema,
            "reward_schema": environment.reward_schema,
            "control_schema": environment.control_schema,
            "residual_force_scale_n": [11.25] if variant == "fx_only" else [11.25, 20.0],
            "physical_action_mapping": "[ax,0]" if variant == "fx_only" else "[ax,az]",
            "distribution": "Normal(tanh(logits), std), samples still clipped"
            if variant == "bounded_mean"
            else "Normal(logits, std), samples clipped",
            "training_reward_scale": 0.01,
            "checkpoint_selection": "final fixed budget; no holdout used",
            "initialization": initialization,
            "model_sha256": shared._sha256(final),
            "source_sha256": sources,
            "wall_training_and_development_s": time.perf_counter() - started,
            "max_unchanged_log_prob_error": action_log.max_unchanged_log_prob_error,
            "exposure": exposure.statistics(),
            "episode_starts_sha256": shared._sha256(output / "episode_starts.json"),
        }
        shared._write_json(output / "metadata.json", metadata)
        return metadata
    finally:
        vec.close()
        if logger is not None:
            logger.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[9000, 10000, 11000])
    parser.add_argument("--budgets", type=int, nargs="+", default=[32768, 65536, 131072])
    parser.add_argument("--envs", type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must not exist")
    rollout = shared.PPO_SETTINGS["n_steps"] * args.envs
    if args.budgets != sorted(set(args.budgets)) or any(
        x <= 0 or x % rollout for x in args.budgets
    ):
        parser.error("budgets must increase, positive, exact multiples of rollout size")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be unique and nonnegative")
    if len(set(args.variants)) != len(args.variants):
        parser.error("variants must be unique")
    if any(b - a < args.envs for a, b in zip(sorted(args.seeds), sorted(args.seeds)[1:])):
        parser.error("seed ranges must not overlap across vector workers")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = loaded_sources()
    snapshot_sources(args.output, sources)
    shared._write_json(
        args.output / "protocol.json",
        {
            "scope": "development single-variable ablations; new holdout frozen but NOT evaluated here",
            "variants": args.variants,
            "seeds": args.seeds,
            "budgets": args.budgets,
            "envs": args.envs,
            "ppo": shared.PPO_SETTINGS,
            "training_reward_scale": 0.01,
            "training_mode": "mixed",
            "episode_seconds": 4.0,
            "development_cases": shared.evaluation_cases("development", "tracking-v2"),
            "intermediate_development_indices": INTERMEDIATE_CASES,
            "new_holdout_cases": frozen_holdout_cases(),
            "quality_criteria": shared.TRACKING_QUALITY_CRITERIA,
            "source_sha256": sources,
            "parameterization_caveat": "tanh mean also rescales gradients; identical weights do not imply identical initial action distributions",
            "development_rng_isolation": "restore Python, numpy and torch CPU streams after checkpoint load and evaluation",
        },
    )
    records = []
    for seed in args.seeds:
        for variant in args.variants:
            print(f"Training {variant}, seed {seed}", flush=True)
            records.append(train_variant(args, variant, seed, sources))
    unchanged = all(shared._sha256(ROOT / key) == value for key, value in sources.items())
    shared._write_json(
        args.output / "summary.json",
        {
            "training_runs": len(records),
            "source_unchanged_during_run": unchanged,
            "models": records,
        },
    )
    shared._write_json(
        args.output / "manifest.json",
        {
            str(p.relative_to(args.output)): shared._sha256(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    )
    if not unchanged:
        raise SystemExit("loaded runtime changed during training")


if __name__ == "__main__":
    main()
