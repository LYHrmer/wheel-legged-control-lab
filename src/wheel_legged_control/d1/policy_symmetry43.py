"""Lateral reflection of B22 course99 observations and residual16 actions.

This is a candidate policy prior, not a claim of exact plant symmetry. The
released CAD has small left/right differences. It applies no task-dependent
gain and adds no privileged input. The wrapped actor is evaluated twice.
"""
from __future__ import annotations

import numpy as np

LEG_SWAP = (1, 0, 3, 2)  # FL, FR, RL, RR
LEG_SIGN = np.asarray((-1., 1., 1.), dtype=np.float32)
JOINT_SIGN = np.asarray((-1., 1., 1., 1.), dtype=np.float32)


def _vector(value, size, limit):
    array = np.asarray(value)
    if (array.shape != (size,) or array.dtype != np.float32 or
            not np.isfinite(array).all() or np.any(np.abs(array) > limit)):
        raise ValueError(f'expected finite bounded float32 vector of size {size}')
    return array


def mirror_action(action):
    """Action order is twelve leg joints then four wheels, not interleaved."""
    value = _vector(action, 16, 1.)
    result = np.empty(16, dtype=np.float32)
    result[:12] = (value[:12].reshape(4, 3)[list(LEG_SWAP)] * LEG_SIGN).reshape(12)
    result[12:] = value[12:][list(LEG_SWAP)]
    return result


def mirror_observation(observation):
    """Reflect body Y; polar Y and axial X/Z change sign."""
    value = _vector(observation, 99, 5.)
    result = value.copy()
    result[1] *= -1  # linear velocity, polar vector
    result[[3, 5]] *= -1  # angular velocity, axial vector
    result[7] *= -1  # projected gravity, polar vector
    for start in (10, 43):  # encoded leg position and nominal target; hip offset is zero
        result[start:start+12] = (
            value[start:start+12].reshape(4, 3)[list(LEG_SWAP)] * LEG_SIGN).reshape(12)
    result[22:38] = (value[22:38].reshape(4, 4)[list(LEG_SWAP)] * JOINT_SIGN).reshape(16)
    result[[39, 40]] *= -1  # applied lateral and yaw commands
    for start in (55, 59, 79):  # nominal wheel speed, PI integral, measured contact
        result[start:start+4] = value[start:start+4][list(LEG_SWAP)]
    # The previous effective action occupies the wider observation box, but
    # its physical contract still requires original action bounds.
    result[63:79] = mirror_action(value[63:79])
    result[83:98] = value[83:98].reshape(5, 3)[:, ::-1].reshape(15)
    return result


class SymmetricActor:
    """Project a deterministic clipped actor onto lateral equivariance.

    Input is copied before either callback, preventing an actor from mutating
    its counterpart. Average in float64 then issue float32; boundedness is
    inherited from both actions. Counts include failed callback attempts.
    """
    def __init__(self, predict):
        self._predict = predict
        self.predict_calls = self.actor_attempts = self.actor_returns = 0

    def __call__(self, observation):
        value = _vector(observation, 99, 5.).copy()
        reflected = mirror_observation(value)
        self.predict_calls += 1
        actions = []
        for obs in (value, reflected):
            self.actor_attempts += 1
            action = _vector(self._predict(obs), 16, 1.).copy()
            self.actor_returns += 1
            actions.append(action)
        return ((actions[0].astype(np.float64) +
                 mirror_action(actions[1]).astype(np.float64)) * .5).astype(np.float32)
