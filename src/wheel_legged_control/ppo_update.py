"""One inspectable CPU/float64 PPO minibatch update; requires optional Torch.

This teaching module is separate from SB3 training. There is no action squash,
advantage normalization, value clipping, gradient clipping, or optimizer state.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral, Real

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal

from .ppo_learning import clipped_surrogate, generalized_advantage_estimate


def _scalar(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


@dataclass(frozen=True)
class UpdateConfig:
    learning_rate: float = 0.01
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = _scalar(
                field.name, getattr(self, field.name), positive=field.name == "learning_rate"
            )
            if field.name == "clip_range" and value >= 1:
                raise ValueError("clip_range must be less than 1")
            object.__setattr__(self, field.name, value)


@dataclass(frozen=True)
class FrozenBatch:
    """Owned, detached snapshots; callers must not modify tensor storage in place.

    Observations/actions have shape (T, obs_dim)/(T, action_dim); all other
    fields have shape (T,). Construction copies storage and disconnects caller
    graphs. Loss evaluation also detaches targets; update never writes them.
    """

    observations: Tensor
    actions: Tensor
    old_log_prob: Tensor
    advantages: Tensor
    returns: Tensor

    def __post_init__(self) -> None:
        for field in fields(self):
            raw = torch.as_tensor(getattr(self, field.name))
            if raw.is_complex() or raw.dtype == torch.bool:
                raise ValueError(f"{field.name} must contain real numbers")
            value = (
                torch.as_tensor(getattr(self, field.name), device="cpu", dtype=torch.float64)
                .detach()
                .clone()
            )
            expected_ndim = 2 if field.name in ("observations", "actions") else 1
            if value.ndim != expected_ndim or 0 in value.shape:
                raise ValueError(f"{field.name} must have nonempty rank {expected_ndim}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{field.name} must contain finite values")
            object.__setattr__(self, field.name, value)
        if any(len(getattr(self, field.name)) != len(self.actions) for field in fields(self)):
            raise ValueError("all batch fields must have the same sample count")


class GaussianActor(nn.Module):
    """Two observations, one raw Gaussian action, four inspectable parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[0.1, 0.15]], dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor([-0.05], dtype=torch.float64))
        self.log_std = nn.Parameter(torch.tensor([-0.4], dtype=torch.float64))

    def forward(self, observations: Tensor) -> Normal:
        return Normal(observations @ self.weight.T + self.bias, self.log_std.exp())


class ValueCritic(nn.Module):
    """Independent linear value function; no shared actor features."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.2, 0.1], dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor(0.05, dtype=torch.float64))

    def forward(self, observations: Tensor) -> Tensor:
        return observations @ self.weight + self.bias


def ppo_terms(
    actor: nn.Module, critic: nn.Module, batch: FrozenBatch, config: UpdateConfig
) -> dict[str, Tensor]:
    """Differentiable losses and per-sample terms, without changing parameters.

    Actor returns an independent-coordinate Normal of shape (T, action_dim).
    Critic returns (T,). Both must use CPU/float64 parameters. Targets and raw
    sampled actions are constants, even if callers enable requires_grad on them.
    Value loss is MSE, WITHOUT an implicit 1/2; vf_coef supplies its coefficient.
    """

    observations = batch.observations.detach()
    distribution = actor(observations)
    if distribution.loc.shape != batch.actions.shape:
        raise ValueError("actor distribution must match batch action shape")
    log_prob = distribution.log_prob(batch.actions.detach()).sum(dim=-1)
    value = critic(observations)
    if value.shape != batch.returns.shape:
        raise ValueError("critic output must have shape (T,)")
    ratio = (log_prob - batch.old_log_prob.detach()).exp()
    advantage = batch.advantages.detach()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - config.clip_range, 1 + config.clip_range) * advantage
    minimum = torch.minimum(unclipped, clipped)
    actor_loss = -minimum.mean()
    value_loss = (value - batch.returns.detach()).square().mean()
    entropy = distribution.entropy().sum(dim=-1).mean()
    terms = {
        "log_prob": log_prob,
        "value": value,
        "ratio": ratio,
        "unclipped": unclipped,
        "clipped": clipped,
        "minimum": minimum,
        "actor_loss": actor_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "total_loss": actor_loss + config.vf_coef * value_loss - config.ent_coef * entropy,
    }
    if not all(torch.isfinite(value).all() for value in terms.values()) or (ratio == 0).any():
        raise ValueError("PPO arithmetic exceeded the finite float64 range")
    return terms


def one_sgd_update(
    actor: nn.Module, critic: nn.Module, batch: FrozenBatch, config: UpdateConfig | None = None
) -> dict:
    """Mutate actor/critic with exactly one plain SGD step and return its trace.

    Existing parameter gradients are cleared. No optimizer state is reused.
    Nonfinite losses/gradients or proposed parameters are rejected before step.
    This function never collects new data or changes the frozen batch.
    """

    config = UpdateConfig() if config is None else config
    named = [(f"actor.{n}", p) for n, p in actor.named_parameters()]
    named += [(f"critic.{n}", p) for n, p in critic.named_parameters()]
    if len({id(parameter) for _, parameter in named}) != len(named):
        raise ValueError("actor and critic must not share parameters")
    if not named or any(
        p.device.type != "cpu" or p.dtype != torch.float64 or not p.requires_grad for _, p in named
    ):
        raise ValueError("all parameters must be trainable CPU float64 tensors")
    optimizer = torch.optim.SGD([p for _, p in named], lr=config.learning_rate)
    before_parameters = {name: parameter.detach().clone() for name, parameter in named}
    optimizer.zero_grad(set_to_none=True)
    before = ppo_terms(actor, critic, batch, config)
    before["total_loss"].backward()
    gradients = {}
    for name, parameter in named:
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise ValueError(f"missing or nonfinite gradient: {name}")
        gradients[name] = gradient.detach().clone()
        if not torch.isfinite(parameter.detach() - config.learning_rate * gradient).all():
            raise ValueError(f"SGD would produce nonfinite parameters: {name}")
    optimizer.step()  # The only parameter update in this experiment.
    try:
        with torch.no_grad():
            after = ppo_terms(actor, critic, batch, config)
    except (ValueError, RuntimeError):
        with torch.no_grad():
            for name, parameter in named:
                parameter.copy_(before_parameters[name])
        raise ValueError("SGD produced invalid policy outputs; parameters restored") from None

    def plain(terms: dict[str, Tensor]) -> dict:
        return {name: value.detach().tolist() for name, value in terms.items()}

    parameter_rows = []
    for name, parameter in named:
        old = before_parameters[name].flatten()
        grad = gradients[name].flatten()
        new = parameter.detach().flatten()
        for index in range(parameter.numel()):
            parameter_rows.append(
                {
                    "parameter": name,
                    "flat_index": index,
                    "before": float(old[index]),
                    "gradient": float(grad[index]),
                    "after": float(new[index]),
                    "delta": float(new[index] - old[index]),
                }
            )
    numpy_before = clipped_surrogate(
        batch.old_log_prob.detach().numpy(),
        before["log_prob"].detach().numpy(),
        batch.advantages.detach().numpy(),
        config.clip_range,
    )
    return {
        "optimizer_steps": 1,
        "before": plain(before),
        "after": plain(after),
        "parameters": parameter_rows,
        "gradient_norms": {
            group: float(
                torch.cat([g.flatten() for n, g in gradients.items() if n.startswith(group)]).norm()
            )
            for group in ("actor", "critic")
        },
        "numpy_actor_loss_before": numpy_before.loss,
    }


def synthetic_rollout(seed: int = 7) -> tuple[GaussianActor, ValueCritic, FrozenBatch, dict]:
    """Sample eight transitions on-policy in a dimensionless first-order toy.

    v_next=0.85*v+0.25*a, reference=0.8. Reward is
    1-(v_next-reference)^2-0.05*a^2. Final time cutoff bootstraps V(final_obs).
    Initial weights are fixed; seed controls action noise only, using a local RNG.
    """

    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    actor, critic = GaussianActor(), ValueCritic()
    velocity, reference = -0.4, 0.8
    observations, next_observations, actions, rewards = [], [], [], []
    with torch.no_grad():
        for _ in range(8):
            observation = torch.tensor([velocity, reference], dtype=torch.float64)
            distribution = actor(observation)
            action = distribution.loc + distribution.scale * torch.randn(
                (1,), generator=generator, dtype=torch.float64
            )
            velocity = 0.85 * velocity + 0.25 * float(action[0])
            observations.append(observation)
            next_observations.append(torch.tensor([velocity, reference], dtype=torch.float64))
            actions.append(action)
            rewards.append(1 - (velocity - reference) ** 2 - 0.05 * float(action[0]) ** 2)
        observations = torch.stack(observations)
        actions = torch.stack(actions)
        values = critic(observations)
        next_values = critic(torch.stack(next_observations))
        old_log_prob = actor(observations).log_prob(actions).sum(dim=-1)
    terminated = np.zeros(8, dtype=bool)
    truncated = np.array([False] * 7 + [True])
    advantages, returns = generalized_advantage_estimate(
        rewards,
        values.numpy(),
        next_values.numpy(),
        terminated,
        truncated,
        gamma=0.9,
        gae_lambda=0.8,
    )
    batch = FrozenBatch(observations, actions, old_log_prob, advantages, returns)
    rollout = {
        "seed": int(seed),
        "gamma": 0.9,
        "gae_lambda": 0.8,
        "dynamics": "v_next=0.85*v+0.25*a; reference=0.8; dimensionless",
        "reward_definition": "1-(v_next-reference)^2-0.05*a^2",
        "rewards": rewards,
        "values": values.tolist(),
        "next_observations": torch.stack(next_observations).tolist(),
        "next_values": next_values.tolist(),
        "terminated": terminated.tolist(),
        "truncated": truncated.tolist(),
    }
    return actor, critic, batch, rollout


def clipping_fixture(clip_range: float = 0.2) -> list[dict]:
    """Separate algebra fixture, NOT samples from the synthetic rollout."""

    epsilon = UpdateConfig(clip_range=clip_range).clip_range
    ratios = np.array([1.5, 0.5, 1.5, 0.5])
    advantages = np.array([2.0, 2.0, -2.0, -2.0])
    old = np.zeros(4)
    new = torch.tensor(np.log(ratios), requires_grad=True)
    ratio = new.exp()
    advantage = torch.tensor(advantages)
    objective = torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)
    derivative = torch.autograd.grad(objective.sum(), new)[0]
    numpy_terms = clipped_surrogate(old, new.detach().numpy(), advantages, epsilon)
    return [
        {
            "ratio": float(ratios[index]),
            "advantage": float(advantages[index]),
            "unclipped": float(numpy_terms.unclipped[index]),
            "clipped": float(numpy_terms.clipped[index]),
            "minimum": float(numpy_terms.minimum[index]),
            "objective_derivative_wrt_log_prob": float(derivative[index]),
        }
        for index in range(4)
    ]
