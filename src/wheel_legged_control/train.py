"""Train a PPO residual policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from .provenance import capture_git_provenance

_DEPENDENCIES = (
    "gymnasium",
    "mujoco",
    "numpy",
    "scipy",
    "stable-baselines3",
    "torch",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("planar", "d1"), default="planar")
    parser.add_argument("--baseline", choices=("lqr", "mpc"), default="lqr")
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256, help="rollout steps per environment")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument(
        "--state-mode",
        choices=("oracle", "estimated"),
        default="oracle",
        help="D1 controller state source; estimated mode is unavailable for the planar model",
    )
    parser.add_argument(
        "--latency-compensation",
        choices=("none", "constant_velocity"),
        default="none",
        help="short-horizon D1 state extrapolation; requires estimated state mode",
    )
    parser.add_argument(
        "--contact-allocation",
        choices=("legacy", "constrained"),
        default="legacy",
        help="D1 contact-force allocation path; unavailable for the planar model",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--log-training-metrics",
        action="store_true",
        help="write SB3 training metrics to output/learning_metrics/progress.csv",
    )
    parser.add_argument("--output", type=Path)
    return parser


def build_ppo_hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    """Validate run sizes and build the explicit PPO constructor settings.

    Call before creating an output directory or environment. A minibatch need
    not divide the rollout size; SB3 can use a smaller final minibatch.
    """

    for name in ("steps", "envs", "runs", "n_steps", "batch_size", "n_epochs"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            if name in {"steps", "envs", "runs"}:
                raise ValueError("--steps, --envs, and --runs must be positive integers")
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer")
    if args.batch_size <= 1 or args.batch_size > args.n_steps * args.envs:
        raise ValueError("--batch-size must be > 1 and <= --n-steps * --envs")
    for name in ("learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if not 0.0 <= args.gamma <= 1.0 or not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("--gamma and --gae-lambda must be in [0, 1]")
    if not 0.0 < args.clip_range < 1.0:
        raise ValueError("--clip-range must be in (0, 1)")
    if args.ent_coef < 0.0:
        raise ValueError("--ent-coef must be nonnegative")
    return {
        "learning_rate": float(args.learning_rate),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "n_epochs": int(args.n_epochs),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "clip_range": float(args.clip_range),
        "ent_coef": float(args.ent_coef),
        "policy_kwargs": {"net_arch": [128, 128]},
    }


def _configure_training_logger(log_directory: Path, verbose: bool) -> Any:
    from stable_baselines3.common.logger import configure

    formats = ["stdout", "csv"] if verbose else ["csv"]
    return configure(str(log_directory), format_strings=formats)


def _git_provenance() -> dict[str, str | bool | None]:
    """Capture source provenance before a long training run starts."""

    return capture_git_provenance(Path(__file__).resolve().parent)


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for dependency in _DEPENDENCIES:
        try:
            versions[dependency] = version(dependency)
        except PackageNotFoundError:
            versions[dependency] = None
    return versions


def _load_rl_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as error:
        raise SystemExit('install RL dependencies with: pip install -e ".[rl]"') from error
    return PPO, make_vec_env, DummyVecEnv, SubprocVecEnv


def train_once(
    args: argparse.Namespace,
    *,
    output: Path,
    seed: int,
    environment: type,
    ppo_class: Any,
    make_vec_env: Any,
    dummy_vec_env: type,
    subproc_vec_env: type,
    logger_factory: Callable[[Path, bool], Any] | None = None,
) -> Path:
    """Train and persist one seeded run.

    Training dependencies are injected so metadata and output behaviour can be
    tested without constructing a real PPO model.
    """

    hyperparameters = build_ppo_hyperparameters(args)
    provenance = _git_provenance()
    output.mkdir(parents=True, exist_ok=True)
    vector_class = subproc_vec_env if args.envs > 1 else dummy_vec_env
    vector_kwargs = {"start_method": "fork"} if args.envs > 1 else None
    env_kwargs = {"baseline": args.baseline, "randomize": True}
    if args.robot == "d1":
        env_kwargs["state_mode"] = args.state_mode
        env_kwargs["latency_compensation"] = args.latency_compensation
        env_kwargs["contact_allocation"] = args.contact_allocation
    vector_env = make_vec_env(
        environment,
        n_envs=args.envs,
        seed=seed,
        vec_env_cls=vector_class,
        vec_env_kwargs=vector_kwargs,
        env_kwargs=env_kwargs,
    )
    training_logger = None
    try:
        model = ppo_class(
            "MlpPolicy",
            vector_env,
            **hyperparameters,
            verbose=int(args.verbose),
            seed=seed,
            device=args.device,
        )
        if args.log_training_metrics:
            configure_logger = logger_factory or _configure_training_logger
            training_logger = configure_logger(output / "learning_metrics", args.verbose)
            model.set_logger(training_logger)
        model.learn(total_timesteps=args.steps)
        model_path = output / "model"
        model.save(model_path)
        model_archive = model_path.with_suffix(".zip")
        model_sha = hashlib.sha256(model_archive.read_bytes()).hexdigest()
        training_config = (
            vars(args)
            | provenance
            | {
                "seed": seed,
                "output": str(output),
                "actual_timesteps": int(model.num_timesteps),
                "python_version": platform.python_version(),
                "dependency_versions": _dependency_versions(),
                "actual_device": str(getattr(model, "device", args.device)),
                "ppo_hyperparameters": hyperparameters,
                "learning_metrics_csv": (
                    "learning_metrics/progress.csv" if args.log_training_metrics else None
                ),
                "observation_schema": getattr(environment, "observation_schema", None),
                "reward_schema": getattr(environment, "reward_schema", None),
                "model_sha": model_sha,
                # Retain the existing checkpoint compatibility contract.
                "model_sha256": model_sha,
            }
        )
        (output / "training_config.json").write_text(
            json.dumps(training_config, indent=2), encoding="utf-8"
        )
    finally:
        try:
            if training_logger is not None:
                training_logger.close()
        finally:
            vector_env.close()
    print(f"saved policy to {model_path}.zip")
    return model_archive


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        build_ppo_hyperparameters(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.robot == "planar" and args.state_mode != "oracle":
        raise SystemExit("--state-mode estimated is only supported with --robot d1")
    if args.robot == "planar" and args.latency_compensation != "none":
        raise SystemExit("latency compensation is only supported with --robot d1")
    if args.robot == "planar" and args.contact_allocation != "legacy":
        raise SystemExit("contact allocation is only supported with --robot d1")
    if args.state_mode != "estimated" and args.latency_compensation != "none":
        raise SystemExit("latency compensation requires --state-mode estimated")

    PPO, make_vec_env, DummyVecEnv, SubprocVecEnv = _load_rl_dependencies()

    if args.output is None:
        args.output = Path(
            "results/residual_ppo" if args.robot == "planar" else "results/d1_residual_ppo"
        )
    if args.robot == "planar":
        from .env import WheelLeggedResidualEnv as Environment
    else:
        from .d1.env import D1ResidualEnv as Environment

    run_seeds = [args.seed + run_offset * args.envs for run_offset in range(args.runs)]
    if args.runs > 1:
        args.output.mkdir(parents=True, exist_ok=True)
        expected_directories = {f"seed_{seed:04d}" for seed in run_seeds}
        stale_directories = sorted(
            path.name
            for path in args.output.glob("seed_*")
            if path.is_dir() and path.name not in expected_directories
        )
        if stale_directories:
            raise SystemExit(
                "output contains seed directories from another run set: "
                + ", ".join(stale_directories)
            )

    completed_runs: list[dict[str, str | int]] = []
    for seed in run_seeds:
        output = args.output if args.runs == 1 else args.output / f"seed_{seed:04d}"
        model_archive = train_once(
            args,
            output=output,
            seed=seed,
            environment=Environment,
            ppo_class=PPO,
            make_vec_env=make_vec_env,
            dummy_vec_env=DummyVecEnv,
            subproc_vec_env=SubprocVecEnv,
        )
        completed_runs.append({"seed": seed, "model": str(model_archive)})

    if args.runs > 1:
        manifest = {
            "robot": args.robot,
            "baseline": args.baseline,
            "state_mode": args.state_mode,
            "latency_compensation": args.latency_compensation,
            "environment_seed_stride": args.envs,
            "runs": completed_runs,
        }
        (args.output / "training_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
