"""Train a PPO residual policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .env import WheelLeggedResidualEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/residual_ppo"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.steps <= 0 or args.envs <= 0:
        raise SystemExit("--steps and --envs must be positive")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as error:
        raise SystemExit('install RL dependencies with: pip install -e ".[rl]"') from error

    args.output.mkdir(parents=True, exist_ok=True)
    vector_class = SubprocVecEnv if args.envs > 1 else DummyVecEnv
    vector_kwargs = {"start_method": "fork"} if args.envs > 1 else None
    vector_env = make_vec_env(
        WheelLeggedResidualEnv,
        n_envs=args.envs,
        seed=args.seed,
        vec_env_cls=vector_class,
        vec_env_kwargs=vector_kwargs,
        env_kwargs={"baseline": "lqr", "randomize": True},
    )
    model = PPO(
        "MlpPolicy",
        vector_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=int(args.verbose),
        seed=args.seed,
        device=args.device,
    )
    model.learn(total_timesteps=args.steps)
    model_path = args.output / "model"
    model.save(model_path)
    (args.output / "training_config.json").write_text(
        json.dumps(vars(args) | {"output": str(args.output)}, indent=2), encoding="utf-8"
    )
    vector_env.close()
    print(f"saved policy to {model_path}.zip")


if __name__ == "__main__":
    main()
