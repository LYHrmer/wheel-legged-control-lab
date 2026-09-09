"""PPO action-parameterization ablations; no changes to SB3 likelihood semantics."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class BoundedMeanActorCriticPolicy(ActorCriticPolicy):
    """Bound the Gaussian mean, not its samples, with the original PPO buffer.

    ``mu = tanh(action_net(latent))``. Samples remain Normal(mu, sigma) on R^n,
    and the environment clips them exactly as in the baseline. No tanh Jacobian
    belongs in log_prob: the random variable itself has not been transformed.
    This also changes the parameter gradient by tanh', so a performance change
    cannot be attributed solely to avoiding deterministic action saturation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_sde or not isinstance(self.action_dist, DiagGaussianDistribution):
            raise ValueError("bounded-mean ablation requires the diagonal Gaussian, without gSDE")
        if not isinstance(self.action_space, spaces.Box) or not (
            np.all(self.action_space.low == -1) and np.all(self.action_space.high == 1)
        ):
            raise ValueError("bounded-mean ablation requires a normalized [-1, 1] Box")

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        mean = torch.tanh(self.action_net(latent_pi))
        return self.action_dist.proba_distribution(mean, self.log_std)
