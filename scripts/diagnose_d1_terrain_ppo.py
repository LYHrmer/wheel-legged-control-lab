"""Read a frozen D1 PPO checkpoint and probe its critic on one real rollout.

Only in-memory copies are updated. Fixed-batch fitting diagnoses optimization;
it does not retrain the controller or measure policy performance.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np


def lambda_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_starts: np.ndarray,
    last_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    """Independent NumPy transcription of SB3's bootstrapped rollout targets.

    Rewards must already contain SB3's timeout bootstrap correction. This
    function must not add that correction a second time.
    """

    rewards, values = np.asarray(rewards), np.asarray(values)
    if rewards.shape != values.shape or rewards.ndim != 2:
        raise ValueError("rewards and values must have identical (time, env) shape")
    if episode_starts.shape != values.shape:
        raise ValueError("episode_starts must match values")
    if last_values.shape != (values.shape[1],) or dones.shape != last_values.shape:
        raise ValueError("bootstrap arrays must have shape (env,)")
    advantages = np.zeros_like(values)
    carry = np.zeros(values.shape[1])
    for step in reversed(range(len(values))):
        next_values = last_values if step == len(values) - 1 else values[step + 1]
        mask = 1.0 - (dones if step == len(values) - 1 else episode_starts[step + 1])
        delta = rewards[step] + gamma * next_values * mask - values[step]
        carry = delta + gamma * gae_lambda * mask * carry
        advantages[step] = carry
    return values + advantages


def explained_variance(values: np.ndarray, targets: np.ndarray) -> float | None:
    variance = float(np.var(targets))
    return None if variance <= 1e-12 else 1.0 - float(np.var(targets - values)) / variance


def value_statistics(values: np.ndarray, targets: np.ndarray) -> dict[str, float | None]:
    return {
        "value_mean": float(values.mean()),
        "value_std": float(values.std()),
        "return_mean": float(targets.mean()),
        "return_std": float(targets.std()),
        "mse": float(np.mean(np.square(targets - values))),
        "explained_variance": explained_variance(values, targets),
    }


def collect_rollout(checkpoint: Path, seed: int, steps: int, n_envs: int) -> tuple[Any, dict]:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from wheel_legged_control.d1.terrain_env import D1TerrainResidualEnv

    class CaptureCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.raw_rewards = []

        def _on_step(self):
            self.raw_rewards.append(self.locals["rewards"].copy())
            return True

    env = DummyVecEnv(
        [lambda: D1TerrainResidualEnv(training_mode="flat", randomize=False) for _ in range(n_envs)]
    )
    try:
        model = PPO.load(checkpoint, env=env, device="cpu")
        if model.observation_space.shape != (44,) or model.action_space.shape != (2,):
            raise ValueError("probe requires the new 44-observation/2-action terrain checkpoint")
        metadata = json.loads(checkpoint.with_name("metadata.json").read_text())
        if metadata["observation_schema"] != "d1-terrain-oracle-v1":
            raise ValueError("incompatible checkpoint observation schema")
        if metadata["model_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError("checkpoint SHA differs from its metadata")
        model.rollout_buffer = RolloutBuffer(
            steps,
            model.observation_space,
            model.action_space,
            n_envs=n_envs,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
        )
        model.set_random_seed(seed)
        capture = CaptureCallback()
        _, callback = model._setup_learn(steps * n_envs, callback=capture)
        model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=steps)
        buffer = model.rollout_buffer
        data = {
            name: getattr(buffer, name).copy()
            for name in (
                "observations",
                "actions",
                "rewards",
                "values",
                "returns",
                "advantages",
                "episode_starts",
                "log_probs",
            )
        }
        with torch.no_grad():
            data["last_values"] = (
                model.policy.predict_values(torch.as_tensor(model._last_obs)).numpy().flatten()
            )
        data["dones"] = model._last_episode_starts.copy()
        data["raw_rewards"] = np.asarray(capture.raw_rewards)
        return model, data
    finally:
        env.close()


def _tensors(data: dict) -> dict:
    import torch

    return {
        name: torch.as_tensor(value.reshape((-1, *value.shape[2:])), dtype=torch.float32)
        for name, value in data.items()
        if name in {"observations", "actions", "values", "returns", "advantages", "log_probs"}
    }


def _probe_minibatch(data: dict) -> dict:
    tensors = _tensors(data)
    return {name: values[data["probe_indices"]] for name, values in tensors.items()}


def _losses(policy: Any, batch: dict, scale: float, model: Any) -> tuple[Any, Any]:
    import torch

    values, log_prob, entropy = policy.evaluate_actions(batch["observations"], batch["actions"])
    advantages = batch["advantages"] * scale
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    ratio = torch.exp(log_prob - batch["log_probs"])
    epsilon = float(model.clip_range(1.0))
    actor = (
        -torch.minimum(
            ratio * advantages, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        ).mean()
        - model.ent_coef * entropy.mean()
    )
    critic = model.vf_coef * torch.mean((values.flatten() - batch["returns"] * scale) ** 2)
    return actor, critic


def _norm(tensors: list[Any]) -> float:
    import torch

    return float(torch.sqrt(sum(torch.sum(tensor.detach() ** 2) for tensor in tensors)))


def gradient_probe(model: Any, data: dict, scale: float) -> dict:
    import torch

    policy = copy.deepcopy(model.policy)
    with torch.no_grad():
        policy.value_net.weight.mul_(scale)
        policy.value_net.bias.mul_(scale)
    actor_loss, critic_loss = _losses(policy, _probe_minibatch(data), scale, model)
    parameters = list(policy.parameters())
    actor_grads = torch.autograd.grad(actor_loss, parameters, allow_unused=True, retain_graph=True)
    critic_grads = torch.autograd.grad(critic_loss, parameters, allow_unused=True)
    actor_norm = _norm([gradient for gradient in actor_grads if gradient is not None])
    critic_norm = _norm([gradient for gradient in critic_grads if gradient is not None])
    shared = sum(
        a is not None and c is not None for a, c in zip(actor_grads, critic_grads, strict=True)
    )
    total = np.sqrt(actor_norm**2 + critic_norm**2)
    factor = min(1.0, model.max_grad_norm / (total + 1e-6))
    return {
        "consistent_value_reward_scale": scale,
        "actor_gradient_norm": actor_norm,
        "weighted_critic_gradient_norm": critic_norm,
        "shared_parameter_tensors": shared,
        "global_clip_factor": factor,
        "actor_gradient_norm_after_global_clip": actor_norm * factor,
    }


def optimizer_probe(model: Any, data: dict, clip_mode: str) -> dict:
    import torch

    policy = copy.deepcopy(model.policy)
    batch = _probe_minibatch(data)
    actor_parameters = [
        parameter
        for name, parameter in policy.named_parameters()
        if "policy_net" in name or "action_net" in name or name == "log_std"
    ]
    critic_parameters = [
        parameter for name, parameter in policy.named_parameters() if "value_net" in name
    ]
    actor_before = [parameter.detach().clone() for parameter in actor_parameters]
    with torch.no_grad():
        before_mean = policy.get_distribution(batch["observations"]).distribution.mean.clone()
    actor, critic = _losses(policy, batch, 1.0, model)
    policy.optimizer.zero_grad()
    (actor + critic).backward()
    if clip_mode == "global":
        torch.nn.utils.clip_grad_norm_(policy.parameters(), model.max_grad_norm)
    elif clip_mode == "separate":
        torch.nn.utils.clip_grad_norm_(actor_parameters, model.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(critic_parameters, model.max_grad_norm)
    elif clip_mode != "none":
        raise ValueError("clip_mode must be global, separate, or none")
    policy.optimizer.step()
    with torch.no_grad():
        after_mean = policy.get_distribution(batch["observations"]).distribution.mean
    return {
        "clip_mode": clip_mode,
        "actor_parameter_step_norm": _norm(
            [
                parameter - before
                for parameter, before in zip(actor_parameters, actor_before, strict=True)
            ]
        ),
        "mean_absolute_action_mean_change": float(torch.mean(torch.abs(after_mean - before_mean))),
        "optimizer": "checkpoint Adam state; exactly one frozen-minibatch step",
    }


def fit_critic(
    model: Any,
    data: dict,
    *,
    scale: float,
    fresh: bool,
    steps: int,
    seed: int,
    scale_initial_output: bool = True,
) -> tuple[dict, list]:
    import torch

    torch.manual_seed(seed)
    trunk = copy.deepcopy(model.policy.mlp_extractor.value_net)
    head = copy.deepcopy(model.policy.value_net)
    if fresh:
        for layer in trunk:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, np.sqrt(2))
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.orthogonal_(head.weight, 1.0)
        torch.nn.init.zeros_(head.bias)
    if scale_initial_output:
        with torch.no_grad():
            head.weight.mul_(scale)
            head.bias.mul_(scale)
    critic = torch.nn.Sequential(trunk, head)
    optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4, eps=1e-5)
    batch = _tensors(data)
    targets = batch["returns"] * scale
    rows = []
    for step in range(steps + 1):
        predictions = critic(batch["observations"]).flatten()
        loss = torch.mean((predictions - targets) ** 2)
        if step == 0 or step == steps or step % 20 == 0:
            stats = value_statistics(
                predictions.detach().numpy() / scale, data["returns"].flatten()
            )
            rows.append({"update": step, **stats})
        if step < steps:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), model.max_grad_norm)
            optimizer.step()
    return {
        "fresh_critic": fresh,
        "target_scale": scale,
        "initial_output_scale": scale if scale_initial_output else 1.0,
        "optimizer": "fresh Adam lr=3e-4 eps=1e-5, critic-only max_grad_norm=0.5",
        "fixed_batch_updates": steps,
        "initial": rows[0],
        "final": rows[-1],
        "scope": "in-sample frozen-target fit only; no control performance claim",
    }, rows


def activation_probe(model: Any, data: dict) -> list[dict]:
    import torch

    activation = _tensors(data)["observations"]
    rows = []
    with torch.no_grad():
        for index, layer in enumerate(model.policy.mlp_extractor.value_net):
            activation = layer(activation)
            if isinstance(layer, torch.nn.Tanh):
                rows.append(
                    {
                        "layer": index,
                        "fraction_abs_activation_gt_095": float(
                            (activation.abs() > 0.95).float().mean()
                        ),
                        "mean_unit_std_across_samples": float(activation.std(dim=0).mean()),
                    }
                )
    return rows


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not 2 <= args.rollout_steps <= 400 or args.envs not in (1, 2, 4):
        raise ValueError("rollout-steps must be 2..400; envs must be 1, 2, or 4")
    if args.fit_steps < 1 or args.seed < 0 or args.batch_size < 2:
        raise ValueError("fit-steps must be positive, seed nonnegative, batch-size at least 2")
    import torch
    from stable_baselines3.common import buffers, on_policy_algorithm, policies

    torch.set_num_threads(1)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    model, data = collect_rollout(args.checkpoint, args.seed, args.rollout_steps, args.envs)
    root = Path(__file__).resolve().parents[1]
    source_paths = list((root / "src" / "wheel_legged_control").rglob("*.py"))
    source_paths.extend(
        Path(module.__file__) for module in (buffers, on_policy_algorithm, policies)
    )
    source_paths.append(Path(inspect.getfile(type(model))))
    data["probe_indices"] = np.random.default_rng(args.seed + 1).permutation(
        args.rollout_steps * args.envs
    )[: args.batch_size]
    recomputed = lambda_returns(
        data["rewards"],
        data["values"],
        data["episode_starts"],
        data["last_values"],
        data["dones"],
        model.gamma,
        model.gae_lambda,
    )
    np.testing.assert_allclose(recomputed, data["returns"], rtol=2e-6, atol=2e-5)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "rollout.npz", **data)
    initial = value_statistics(data["values"], data["returns"])
    gradients = [gradient_probe(model, data, scale) for scale in (1.0, 0.1, 0.01)]
    optimizer_steps = [
        optimizer_probe(model, data, mode) for mode in ("global", "separate", "none")
    ]
    fits = []
    for fresh in (False, True):
        for scale in (1.0, 0.01):
            result, rows = fit_critic(
                model, data, scale=scale, fresh=fresh, steps=args.fit_steps, seed=args.seed
            )
            fits.append(result)
            path = args.output / f"critic_fit_{'fresh' if fresh else 'loaded'}_scale_{scale:g}.csv"
            with path.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    result, rows = fit_critic(
        model,
        data,
        scale=0.01,
        fresh=True,
        steps=args.fit_steps,
        seed=args.seed,
        scale_initial_output=False,
    )
    fits.append(result)
    with (args.output / "critic_fit_fresh_reward_only_scale_0.01.csv").open(
        "x", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    offset = float(data["raw_rewards"].mean() / (1 - model.gamma) - data["values"].mean())
    shifted = lambda_returns(
        data["rewards"],
        data["values"] + offset,
        data["episode_starts"],
        data["last_values"] + offset,
        data["dones"],
        model.gamma,
        model.gae_lambda,
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_metadata_sha256": hashlib.sha256(
            args.checkpoint.with_name("metadata.json").read_bytes()
        ).hexdigest(),
        "seed": args.seed,
        "rollout_steps": args.rollout_steps,
        "n_envs": args.envs,
        "gradient_optimizer_minibatch_size": len(data["probe_indices"]),
        "data": "new seeded real flat D1 rollout, stochastic frozen policy; development only",
        "actions": "stored unsquashed samples; environment applied bounded copies as in SB3",
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "vf_coef": model.vf_coef,
        "max_grad_norm": model.max_grad_norm,
        "normalize_advantage": model.normalize_advantage,
        "reward_mean": float(data["raw_rewards"].mean()),
        "timeout_bootstrap_corrections": int(
            np.count_nonzero(data["rewards"] - data["raw_rewards"])
        ),
        "numpy_sb3_return_max_abs_difference": float(np.max(np.abs(recomputed - data["returns"]))),
        "initial": initial,
        "gradient_probes": gradients,
        "optimizer_probes": optimizer_steps,
        "critic_activations": activation_probe(model, data),
        "critic_fits": fits,
        "constant_bootstrap_probe": {
            "offset": offset,
            "return_time_correlation_before": float(
                np.corrcoef(np.arange(args.rollout_steps), data["returns"].mean(axis=1))[0, 1]
            ),
            "return_time_correlation_after": float(
                np.corrcoef(np.arange(args.rollout_steps), shifted.mean(axis=1))[0, 1]
            ),
            "statistics_after": value_statistics(data["values"] + offset, shifted),
            "scope": "algebraic sensitivity only; mean_reward/(1-gamma) is not a known true value",
            "valid_without_timeout_only": not bool(
                np.any(data["dones"]) or np.any(data["episode_starts"][1:])
            ),
        },
        "checkpoint_unchanged": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        == checkpoint_hash,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_source_sha256": {
            str(path.relative_to(root)) if path.is_relative_to(root) else str(path): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(set(source_paths))
        },
        "package_versions": {
            name: version(name) for name in ("torch", "stable-baselines3", "mujoco", "numpy")
        },
    }
    with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (args.output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(args.output.iterdir())
                if path.name != "manifest.json"
            },
            stream,
            indent=2,
        )
    print(
        json.dumps(
            {"initial": initial, "gradient_probes": gradients, "critic_fits": fits}, indent=2
        )
    )
    if args.require_ev is not None and initial["explained_variance"] < args.require_ev:
        raise SystemExit(
            f"[DEBUG-terrain-ppo] EV below diagnostic threshold {args.require_ev}; artifacts retained"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--envs", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--fit-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--require-ev",
        type=float,
        help="optional red/green diagnostic threshold, not a quality guarantee",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
