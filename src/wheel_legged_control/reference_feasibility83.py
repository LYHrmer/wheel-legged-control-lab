"""Necessary reference torque condition for a known scalar implicit-fast plant.

This module imports only the Python standard library. It neither simulates a
plant nor modifies a controller, reference or torque limit. For a stdlib-only
readback, load this file directly with importlib.util: the existing package
__init__ imports other modules with numerical dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from math import exp, isfinite, tanh
from numbers import Real


def _scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        # One ValueError contract covers malformed references and parameters.
        raise ValueError(f"{name} must be a finite real scalar")  # noqa: TRY004
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must fit finite binary64") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_result(value: float, label: str, index: int) -> float:
    if not isfinite(value):
        raise ValueError(f"interval {index}: {label} overflow/nonfinite result")
    return value


def check_reference_torque(
    reference: Sequence[float],
    *,
    dt: float,
    total_inertia: float,
    damping: float,
    coulomb: float,
    torque_limit: float,
    stribeck: float = 0.0,
    friction_smoothing: float = 0.05,
    stribeck_velocity: float = 0.3,
) -> dict:
    """Check all N intervals of an explicitly supplied r[0..N] in rad/s.

    The assumed discrete balance is
    ``tau[k] = M*(r[k+1]-r[k])/dt + b*r[k+1] + friction(r[k])``,
    with ``friction(v) = (c + s*exp(-(v/vs)**2))*tanh(v/vsmooth)``.
    ``M`` is the *total* load-plus-armature inertia, not armature alone.
    Damping is implicit at the next speed; smooth friction is held at the
    beginning of the interval. Setting ``stribeck=0`` assumes that term absent.

    Strict ``abs(tau)>torque_limit`` is a violation; equality satisfies this
    necessary condition. A violation rules out exact reference endpoint
    tracking under this stated plant model and applied-torque bound. Passing
    does not establish closed-loop tracking: delay, initial state, available
    prior commands, unmodelled dynamics and identification error are unchecked.
    The result is not an RMSE lower bound. No terminal sample is extrapolated.
    All quantities use joint-side SI units; every arithmetic overflow rejects
    the input rather than treating a nonfinite comparison as a pass.
    """
    parameters = {
        name: _scalar(value, name)
        for name, value in {
            "dt": dt,
            "total_inertia": total_inertia,
            "damping": damping,
            "coulomb": coulomb,
            "torque_limit": torque_limit,
            "stribeck": stribeck,
            "friction_smoothing": friction_smoothing,
            "stribeck_velocity": stribeck_velocity,
        }.items()
    }
    for name in ("dt", "total_inertia", "torque_limit", "friction_smoothing", "stribeck_velocity"):
        if parameters[name] <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("damping", "coulomb", "stribeck"):
        if parameters[name] < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    if (
        not isinstance(reference, Sequence)
        or isinstance(reference, (str, bytes, bytearray))
        or len(reference) < 2
    ):
        raise ValueError("reference must contain all N+1 samples for at least one interval")
    samples = [_scalar(value, f"reference[{index}]") for index, value in enumerate(reference)]
    dt, inertia = parameters["dt"], parameters["total_inertia"]
    damping, coulomb = parameters["damping"], parameters["coulomb"]
    stribeck, limit = parameters["stribeck"], parameters["torque_limit"]
    intervals = []
    for index, (now, following) in enumerate(pairwise(samples)):
        delta = _finite_result(following - now, "reference increment", index)
        # Preserve the frozen82 expression order: M*delta first, then /dt.
        momentum = _finite_result(inertia * delta, "inertia times increment", index)
        inertial_torque = _finite_result(momentum / dt, "inertial torque", index)
        damping_torque = _finite_result(damping * following, "implicit damping torque", index)
        friction_torque = 0.0
        if coulomb != 0.0 or stribeck != 0.0:
            amplitude = coulomb
            if stribeck != 0.0:
                ratio = _finite_result(
                    now / parameters["stribeck_velocity"], "Stribeck ratio", index
                )
                squared = _finite_result(ratio * ratio, "Stribeck squared ratio", index)
                extra = _finite_result(stribeck * exp(-squared), "Stribeck amplitude", index)
                amplitude = _finite_result(amplitude + extra, "friction amplitude", index)
            ratio = _finite_result(now / parameters["friction_smoothing"], "friction ratio", index)
            friction_torque = _finite_result(amplitude * tanh(ratio), "friction torque", index)
        subtotal = _finite_result(inertial_torque + damping_torque, "inertia plus damping", index)
        required = _finite_result(subtotal + friction_torque, "required torque", index)
        violation = abs(required) > limit
        intervals.append(
            {
                "index": index,
                "reference_now_rad_s": now,
                "reference_next_rad_s": following,
                "inertial_torque_nm": inertial_torque,
                "implicit_damping_torque_nm": damping_torque,
                "preinterval_friction_torque_nm": friction_torque,
                "required_torque_nm": required,
                "violates_torque_limit": violation,
                "excess_abs_torque_nm": max(0.0, abs(required) - limit),
            }
        )
    peak = max(intervals, key=lambda row: abs(row["required_torque_nm"]))
    count = sum(row["violates_torque_limit"] for row in intervals)
    return {
        "schema": "reference-necessary-torque83-v1",
        "sample_count": len(samples),
        "interval_count": len(intervals),
        "parameters": parameters,
        "intervals": intervals,
        "peak_abs_required_torque_nm": abs(peak["required_torque_nm"]),
        "peak_signed_required_torque_nm": peak["required_torque_nm"],
        "peak_interval_index": peak["index"],
        "violation_count": count,
        "violation_fraction": count / len(intervals),
        "necessary_torque_condition": count == 0,
        "closed_loop_tracking": "NOT_EVALUATED",
        "rmse_lower_bound": None,
        "interpretation": (
            "Exact reference endpoints violate the applied-torque bound under the stated model."
            if count
            else "Only the necessary torque condition is satisfied under the stated model."
        ),
        "unchecked": [
            "initial state equals reference[0]",
            "command delay and available prehistory",
            "closed-loop stability and tracking",
            "model identification error and omitted dynamics",
        ],
        "reference_or_torque_limit_modified": False,
    }
