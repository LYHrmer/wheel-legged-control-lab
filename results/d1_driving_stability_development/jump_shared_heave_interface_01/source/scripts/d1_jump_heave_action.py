"""Pure, request-gated shared extension action; no controller or simulator."""

from numbers import Integral, Real

import numpy as np

HEAVE_ACTION_SCHEMA = "d1-jump-shared-heave-window-v1"
HEAVE_WINDOW_TICKS = 120


def _tick(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-Boolean integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(value)


def map_heave_action(action, *, executed_tick: int, request_tick: int | None) -> np.ndarray:
    """Map one finite real coordinate to the original eight physical channels.

    Input validation also applies outside the request window. This pure mapping
    has no memory and does not reset previous action, controller PI or targets.
    """
    raw = np.asarray(action)
    if raw.shape != (1,):
        raise ValueError("heave action must have exact shape (1,)")
    value = raw[0]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("heave action must contain one real non-Boolean value")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("heave action must be finite")
    tick = _tick(executed_tick, "executed_tick")
    start = None if request_tick is None else _tick(request_tick, "request_tick")
    physical = np.zeros(8, dtype=np.float64)
    if start is not None and start <= tick < start + HEAVE_WINDOW_TICKS:
        physical[:4] = np.clip(value, -1.0, 1.0)
    return physical


def heave_action_metadata() -> dict:
    """Return an owned JSON-compatible description, not a legacy gather map."""
    legs = ("FL", "FR", "RL", "RR")
    return {
        "action_schema": HEAVE_ACTION_SCHEMA,
        "policy_action_size": 1,
        "physical_action_size": 8,
        "physical_action_schema": "d1-wheel-leg-extension-speed-v1",
        "physical_channels": [f"{leg}_extension_m" for leg in legs]
        + [f"{leg}_wheel_speed_rad_s" for leg in legs],
        "active_embedding_matrix": [[1.0] for _ in legs] + [[0.0] for _ in legs],
        "leg_action_scale_m": 0.04,
        "wheel_action_scale_rad_s": 4.0,
        "wheel_residual_forced_zero": True,
        "gate": {
            "window_ticks": HEAVE_WINDOW_TICKS,
            "index": "current_executed_control_interval",
            "active": "request_tick <= executed_tick < request_tick + 120",
            "no_request": "physical_positive_zero8",
            "outside_window": "physical_positive_zero8",
            "clears_controller_or_action_memory": False,
        },
    }
