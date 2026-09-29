"""Pure NumPy calculations for the bounded 64-step value diagnostic."""

from __future__ import annotations

import numpy as np


def discounted_sum(rewards, gamma: float = 0.99):
    """Return sum_k gamma**k * reward[k] for a 1D or batched 2D array."""
    a = np.asarray(rewards, dtype=np.float64)
    if a.ndim not in (1, 2) or not a.shape[-1] or not np.isfinite(a).all():
        raise ValueError("rewards must be a nonempty finite 1D/2D array")
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and in [0, 1]")
    weights = np.power(float(gamma), np.arange(a.shape[-1], dtype=np.float64))
    result = a @ weights
    if not np.isfinite(result).all():
        raise ValueError("discounted reward sum is nonfinite")
    return result


def bootstrap_targets(rewards, endpoint_values, gamma: float = 0.99):
    """64-step saved-behavior rewards plus gamma**64 * current endpoint value."""
    a = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(endpoint_values, dtype=np.float64)
    if a.ndim != 2 or v.shape != (len(a),) or not np.isfinite(v).all():
        raise ValueError("expected (N, H) rewards and finite (N,) endpoint values")
    out = discounted_sum(a, gamma) + float(gamma) ** a.shape[1] * v
    if not np.isfinite(out).all():
        raise ValueError("bootstrap target is nonfinite")
    return out


def diagnostic_stats(start_values, targets):
    """Bias is target minus current start value; EV is null for constant targets."""
    value = np.asarray(start_values, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if value.ndim != 1 or target.shape != value.shape or not len(value):
        raise ValueError("expected equal nonempty 1D arrays")
    if not np.isfinite(value).all() or not np.isfinite(target).all():
        raise ValueError("value and target must be finite")
    error = target - value
    target_var = float(np.var(target))
    result = {
        "n": int(len(value)),
        "bias_target_minus_value": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "median_error_target_minus_value": float(np.median(error)),
        "p05_error": float(np.quantile(error, 0.05)),
        "p95_error": float(np.quantile(error, 0.95)),
        "value_mean": float(np.mean(value)),
        "value_std": float(np.std(value)),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.sqrt(target_var)),
        "explained_variance": None if target_var <= 1e-12 else
        float(1.0 - np.var(error) / target_var),
    }
    if not all(v is None or np.isfinite(v) for v in result.values()):
        raise ValueError("diagnostic statistic is nonfinite")
    return result
