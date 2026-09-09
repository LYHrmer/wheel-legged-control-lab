"""A one-dimensional residual ablation on the unchanged D1 v2 controller."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .terrain_tracking_env import D1TerrainTrackingEnv


class D1LongitudinalResidualEnv(D1TerrainTrackingEnv):
    """Learn Fx only; the vertical residual is exactly zero at every step.

    The 44-value observation layout, terrain sampler and reward equation are
    unchanged. The previous-action observation still has both physical force
    channels; its vertical field is zero. This needs a new one-action model.
    """

    action_schema = "d1-terrain-residual-longitudinal-only-v1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError("longitudinal residual action must contain one finite value")
        return super().step(np.asarray((action[0], 0.0), dtype=np.float64))
