"""NumPy exercises for GAE and PPO clipping, independent of the training path.

These functions expose intermediate arrays for inspection. They do not update a
policy, normalize advantages, or replace Stable-Baselines3's rollout buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _samples(name: str, value: ArrayLike) -> FloatArray:
    array = np.asarray(value)
    if array.ndim not in (1, 2) or 0 in array.shape:
        raise ValueError(f"{name} must have nonempty shape (T,) or (T, N)")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real numbers")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _unit_interval(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar in [0, 1]")
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a real scalar in [0, 1]")
    return float(value)


def generalized_advantage_estimate(
    rewards: ArrayLike,
    values: ArrayLike,
    next_values: ArrayLike,
    terminated: ArrayLike,
    truncated: ArrayLike,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[FloatArray, FloatArray]:
    """Return unnormalized advantages and lambda-return value targets.

    All inputs have identical shape (T,) or (T, N), with time on axis zero.
    ``values[t]`` is V(observation_t); ``next_values[t]`` is V of the actual
    next/final observation reached by transition t, BEFORE any autoreset.
    Both boundary masks must have boolean dtype.

    Termination removes bootstrapping. Truncation preserves bootstrapping but
    cuts the recursive advantage carry, so a reset episode cannot leak into the
    preceding one. If both flags are true, termination takes precedence.
    At a nonterminal rollout tail, next_values[-1] provides the bootstrap, while
    the unobserved future advantage carry starts at zero.
    """

    reward = _samples("rewards", rewards)
    value = _samples("values", values)
    next_value = _samples("next_values", next_values)
    terminal = np.asarray(terminated)
    timeout = np.asarray(truncated)
    for name, array in (
        ("values", value),
        ("next_values", next_value),
        ("terminated", terminal),
        ("truncated", timeout),
    ):
        if array.shape != reward.shape:
            raise ValueError(f"{name} must have the same shape as rewards")
    if terminal.dtype != np.bool_ or timeout.dtype != np.bool_:
        raise ValueError("terminated and truncated must have boolean dtype")
    discount = _unit_interval("gamma", gamma)
    trace_decay = _unit_interval("gae_lambda", gae_lambda)

    advantages = np.empty_like(reward)
    carry = np.zeros(reward.shape[1:], dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise"):
            delta = reward + discount * np.where(terminal, 0.0, next_value) - value
            for step in range(len(reward) - 1, -1, -1):
                continues = ~(terminal[step] | timeout[step])
                carry = delta[step] + discount * trace_decay * np.where(continues, carry, 0.0)
                advantages[step] = carry
            returns = advantages + value
    except FloatingPointError as error:
        raise ValueError("GAE arithmetic produced non-finite values") from error
    return advantages, returns


@dataclass(frozen=True)
class ClippedSurrogate:
    """Per-sample PPO objective terms and the loss minimized by an optimizer."""

    ratio: FloatArray
    unclipped: FloatArray
    clipped: FloatArray
    minimum: FloatArray
    loss: float


def clipped_surrogate(
    old_log_prob: ArrayLike,
    new_log_prob: ArrayLike,
    advantages: ArrayLike,
    clip_range: float = 0.2,
) -> ClippedSurrogate:
    """Evaluate PPO's clipped actor objective without advantage normalization.

    Log probabilities refer to the SAME sampled actions and observations. Sum
    across action coordinates first; each entry is one joint-action log density.
    Inputs have equal shape (T,) or (T, N); no broadcasting is allowed.

    ``minimum`` is min(ratio * A, clip(ratio) * A), and loss is its negative
    mean. This does not constrain every ratio to the clipping interval: for
    A < 0 and ratio > 1 + clip_range, the unclipped branch remains active.
    Value-function loss and entropy are deliberately outside this exercise.
    """

    old = _samples("old_log_prob", old_log_prob)
    new = _samples("new_log_prob", new_log_prob)
    advantage = _samples("advantages", advantages)
    if new.shape != old.shape or advantage.shape != old.shape:
        raise ValueError("log probabilities and advantages must have the same shape")
    epsilon = _unit_interval("clip_range", clip_range)
    if epsilon == 1.0:
        raise ValueError("clip_range must be less than 1")
    try:
        with np.errstate(over="raise", under="raise", invalid="raise"):
            ratio = np.exp(new - old)
            unclipped = ratio * advantage
            clipped = np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage
            minimum = np.minimum(unclipped, clipped)
            # Divide before summing to avoid overflowing a representable mean.
            loss = -float(np.sum(minimum / minimum.size))
    except FloatingPointError as error:
        raise ValueError("PPO arithmetic exceeded the finite float64 range") from error
    return ClippedSurrogate(ratio, unclipped, clipped, minimum, loss)
