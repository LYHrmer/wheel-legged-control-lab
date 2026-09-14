"""Residual PPO policy that bounds the *common* (mean) wheel-speed residual.

Context
-------
The D1 wheeled-leg controller consumes an 8-dimensional residual action on top of a
baseline (nominal) controller:

* ``action[0:4]`` -- per-leg extension residuals (left untouched by this module),
* ``action[4:8]`` -- per-wheel speed residuals, where one action unit equals 4 rad/s.

Closed-loop interventions in MuJoCo showed that a large *common* (equal on all four
wheels) negative wheel residual fights the baseline controller and slows the robot.
This module therefore freezes a single-variable policy parameterization: the mean of the
Gaussian action distribution has its wheel-common component squashed through ``tanh``,
while the *differential* wheel mean components and all four leg means are preserved
exactly.

Parameterization
----------------
With ``raw = action_net(latent_pi)`` and limit ``c > 0``::

    m          = raw[..., 4:].mean(-1, keepdim=True)     # common wheel mean
    bounded_m  = c * tanh(m / c)                         # |bounded_m| <= c numerically
    mean       = cat((raw[..., :4], raw[..., 4:] - m + bounded_m), dim=-1)

The distribution is then ``self.action_dist.proba_distribution(mean, self.log_std)``,
i.e. an unchanged diagonal Gaussian whose *mean* has been reparameterized.

Important semantics / caveats
-----------------------------
* This is **not** a squashed distribution. Only the mean is transformed; the
  log-probability of a sample under a normal distribution with the transformed mean is
  exact, so **no tanh-Jacobian correction is applied or needed**. Stochastic samples are
  unconstrained Gaussian draws: an individual sampled action may have a wheel-common
  component larger than ``c``. There is no sample projection and no inference-only
  post-processing here.
* Because the environment's action space is ``Box(-1, 1)`` and Stable-Baselines3 clips
  actions to the Box before stepping the environment, the *executed* action can differ
  from the sampled action. Clipping is applied element-wise and is **not** aware of the
  wheel-common decomposition, so **native clipping can invalidate the wheel-common bound
  on the executed action** (and can also perturb the differential wheel components).
  The bound described here is a property of the distribution *mean* prior to sampling
  and prior to clipping.
* The clipping caveat also applies to deterministic actions. For example, wheel
  means ``[3, -1, -1, -1]`` have common mean zero, but clipping to ``[-1, 1]``
  gives common mean ``-0.5``. Evaluations must record clipping and executed means.
* Unit reading of the default ``c = 0.05``: the common wheel residual mean stays within
  ``+/- 0.05 * 4 rad/s = +/- 0.2 rad/s``. With the nominal 0.087 m wheel radius that is
  an *ideal, no-slip* wheel-ground speed of about ``+/- 0.0174 m/s``. This is a bound on
  the commanded residual only -- **not** a guarantee on actual robot velocity.
* ``wheel_common_mean_limit=None`` disables the reparameterization and defers to the
  stock SB3 implementation, preserving mean, distribution object, RNG consumption and
  deterministic actions bit-for-bit for an identical seed.

The policy never mutates weights, samples, or standard deviations, adds no parameters,
and does not re-initialize (or zero-initialize) anything.
"""

from __future__ import annotations

import math
import numbers
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.distributions import DiagGaussianDistribution, Distribution
from stable_baselines3.common.policies import ActorCriticPolicy

__all__ = ["ACTION_DIM", "LEG_DIM", "OBS_DIM", "D1WheelCommonMeanPolicy", "bounded_wheel_mean"]

#: Flat observation dimension of the D1 residual task.
OBS_DIM = 82
#: Physical action dimension: 4 leg extension residuals + 4 wheel speed residuals.
ACTION_DIM = 8
#: Number of leading leg-extension components (wheels are ``[LEG_DIM:ACTION_DIM]``).
LEG_DIM = 4


def _validate_limit(limit: float | None) -> float | None:
    """Validate the wheel-common mean limit.

    :param limit: ``None`` (feature disabled) or a finite, strictly positive real number.
        ``bool`` is rejected explicitly, as are complex numbers, strings and arrays.
    :return: ``None`` or the limit as a Python ``float``.
    :raises TypeError: if ``limit`` is not ``None`` and not a real scalar.
    :raises ValueError: if ``limit`` is not finite and strictly positive.
    """
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, numbers.Real):
        raise TypeError(
            "wheel_common_mean_limit must be None or a finite strictly positive real "
            f"number (bool is not accepted), got {limit!r} of type {type(limit).__name__}."
        )
    value = float(limit)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            "wheel_common_mean_limit must be finite and strictly positive, "
            f"got {limit!r}."
        )
    return value


def _check_dtype_limit(limit: float | None, dtype: th.dtype) -> None:
    """Avoid scalar overflow/underflow and nonfinite gradients in tensor arithmetic."""
    if limit is not None:
        bounds = th.finfo(dtype)
        if not bounds.tiny <= limit <= bounds.max:
            raise ValueError(
                f"wheel_common_mean_limit must be a positive normal value for {dtype}"
            )


def bounded_wheel_mean(raw_mean: th.Tensor, limit: float | None) -> th.Tensor:
    """Bound the common (across-wheel average) component of an 8-D action mean.

    Pure tensor helper -- no module state, no in-place mutation of ``raw_mean``.
    Leading dimensions, floating dtype and device are preserved. The last dimension
    must already be exactly :data:`ACTION_DIM`; the input is never reshaped.

    :param raw_mean: Float tensor of shape ``(..., 8)``; ``[..., :4]`` are leg
        extension residual means, ``[..., 4:]`` are wheel speed residual means.
    :param limit: ``None`` to return ``raw_mean`` unchanged, else the finite strictly
        positive bound ``c`` on the common wheel mean.
    :return: Tensor of the same shape/dtype/device with wheel means shifted so that
        their average is ``c * tanh(average / c)``; leg means are unchanged and
        pairwise wheel differences are preserved to floating-point precision.
    :raises TypeError: for non-tensor or non-floating input, or a malformed ``limit``.
    :raises ValueError: for a wrong trailing dimension or a malformed ``limit``.
    """
    if not isinstance(raw_mean, th.Tensor):
        raise TypeError(f"raw_mean must be a torch.Tensor, got {type(raw_mean).__name__}.")
    if raw_mean.ndim < 1 or raw_mean.shape[-1] != ACTION_DIM:
        raise ValueError(
            "raw_mean must have last dimension exactly "
            f"{ACTION_DIM}, got shape {tuple(raw_mean.shape)}."
        )
    if not raw_mean.is_floating_point():
        raise TypeError(f"raw_mean must have a floating dtype, got {raw_mean.dtype}.")

    c = _validate_limit(limit)
    if c is None:
        return raw_mean
    _check_dtype_limit(c, raw_mean.dtype)

    wheels = raw_mean[..., LEG_DIM:]
    common = wheels.mean(dim=-1, keepdim=True)
    bounded_common = c * th.tanh(common / c)
    return th.cat((raw_mean[..., :LEG_DIM], wheels - common + bounded_common), dim=-1)


def _check_spaces(observation_space: gym.Space, action_space: gym.Space) -> None:
    """Reject every space layout this frozen parameterization does not support.

    :param observation_space: must be a flat ``Box`` of shape ``(82,)``.
    :param action_space: must be a floating ``Box`` of shape ``(8,)`` with all
        ``low == -1`` and all ``high == +1``.
    :raises ValueError: with a readable message for any unsupported space.
    """
    if not isinstance(observation_space, gym.spaces.Box):
        raise TypeError(
            "D1WheelCommonMeanPolicy requires a flat gymnasium Box observation space, "
            f"got {type(observation_space).__name__}."
        )
    if observation_space.shape != (OBS_DIM,):
        raise ValueError(
            "D1WheelCommonMeanPolicy requires observation space of shape "
            f"({OBS_DIM},), got {tuple(observation_space.shape)}."
        )

    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(
            "D1WheelCommonMeanPolicy requires a continuous gymnasium Box action space, "
            f"got {type(action_space).__name__}."
        )
    if action_space.shape != (ACTION_DIM,):
        raise ValueError(
            "D1WheelCommonMeanPolicy requires action space of shape "
            f"({ACTION_DIM},) (4 leg + 4 wheel residuals), got "
            f"{tuple(action_space.shape)}."
        )
    if not np.issubdtype(action_space.dtype, np.floating):
        raise ValueError(
            "D1WheelCommonMeanPolicy requires a floating-point action space dtype, "
            f"got {action_space.dtype}."
        )
    low = np.asarray(action_space.low, dtype=np.float64)
    high = np.asarray(action_space.high, dtype=np.float64)
    if not (np.all(low == -1.0) and np.all(high == 1.0)):
        raise ValueError(
            "D1WheelCommonMeanPolicy requires the normalized residual action box with "
            f"all low = -1 and all high = +1, got low={low.tolist()}, "
            f"high={high.tolist()}."
        )


class D1WheelCommonMeanPolicy(ActorCriticPolicy):
    """``ActorCriticPolicy`` whose Gaussian mean has a bounded wheel-common component.

    Drop-in replacement for the stock SB3 PPO policy on the D1 residual task: the
    environment, reward and PPO algorithm are untouched, only the mapping from the actor
    latent to the distribution mean changes.

    :param observation_space: flat ``Box`` of shape ``(82,)``.
    :param action_space: floating ``Box`` of shape ``(8,)`` with ``low = -1``, ``high = 1``.
    :param lr_schedule: standard SB3 learning-rate schedule.
    :param wheel_common_mean_limit: ``None`` to behave exactly like the stock policy
        (identical mean, distribution, RNG usage and deterministic actions for the same
        seed), or a finite strictly positive ``c`` bounding the common wheel mean to
        ``(-c, c)`` action units (``1`` unit = 4 rad/s). Default ``0.05`` -> ``+/- 0.2
        rad/s`` commanded common wheel residual.
    :raises ValueError: for unsupported spaces, ``use_sde=True``, ``squash_output=True``
        (i.e. any action distribution other than the original
        :class:`DiagGaussianDistribution`), or a non-positive / non-finite limit.

    See the module docstring for the exact clipping caveat: SB3's native Box clipping is
    element-wise and can invalidate the wheel-common bound of the *executed* action.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Any,
        *args: Any,
        wheel_common_mean_limit: float | None = 0.05,
        **kwargs: Any,
    ) -> None:
        limit = _validate_limit(wheel_common_mean_limit)
        _check_spaces(observation_space, action_space)

        if bool(kwargs.get("use_sde", False)):
            raise ValueError(
                "D1WheelCommonMeanPolicy only supports use_sde=False; gSDE "
                "(StateDependentNoiseDistribution) is not part of this frozen "
                "parameterization."
            )
        if bool(kwargs.get("squash_output", False)):
            raise ValueError(
                "D1WheelCommonMeanPolicy only supports squash_output=False; a squashed "
                "distribution would change the action log-probabilities."
            )

        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        if type(self.action_dist) is not DiagGaussianDistribution:
            raise ValueError(
                "D1WheelCommonMeanPolicy only supports the original "
                "DiagGaussianDistribution, got "
                f"{type(self.action_dist).__name__}."
            )

        _check_dtype_limit(limit, self.action_net.weight.dtype)
        self.wheel_common_mean_limit: float | None = limit

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """Build the diagonal Gaussian, bounding the wheel-common mean component.

        :param latent_pi: actor latent code, shape ``(..., latent_dim_pi)``.
        :return: ``DiagGaussianDistribution`` parameterized by the (possibly
            reparameterized) mean and the unchanged ``self.log_std``.
        """
        if self.wheel_common_mean_limit is None:
            # Stock SB3 behaviour, bit-for-bit.
            return super()._get_action_dist_from_latent(latent_pi)

        mean_actions = bounded_wheel_mean(self.action_net(latent_pi), self.wheel_common_mean_limit)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        """Add ``wheel_common_mean_limit`` so ``policy.save``/``load`` round-trips.

        :return: constructor kwargs, extending the SB3 defaults.
        """
        data = super()._get_constructor_parameters()
        data.update(wheel_common_mean_limit=self.wheel_common_mean_limit)
        return data
