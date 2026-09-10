"""Causal encoder velocity estimation from sampled joint positions.

The observer consumes exactly one signal: the unwrapped encoder position and
the sampling instant it was measured at.  It never reads the plant, a
ground-truth or independently synthesised velocity, the commanded torque, or
any future sample.  The timestamp of an estimate is the current sampling
instant, so this module performs no actuator or measurement delay
compensation.  Positions are treated as continuous real numbers and are never
wrapped into a bounded interval.

Standard library only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

__all__ = [
    "METHODS",
    "EncoderVelocityConfig",
    "EncoderVelocityEstimate",
    "EncoderVelocityObserver",
]

METHODS = ("difference_lowpass", "alpha_beta")

#: Maximum absolute clock tolerance, also capped at one quarter of config.dt.
CLOCK_ATOL_S = 1e-10


def _finite_scalar(value: object, name: str) -> float:
    """Return ``value`` as a finite float, rejecting non-real-scalar inputs."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"{name} must be a real finite scalar, got {type(value).__name__}"
        )
    try:
        number = float(value)
    except OverflowError as exc:  # e.g. an int larger than the float range
        raise ValueError(f"{name} is not representable as a finite float") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _sample_time(value: object) -> float:
    time_s = _finite_scalar(value, "time_s")
    if time_s < 0.0:
        raise ValueError(f"time_s must be non-negative, got {time_s!r}")
    return time_s


@dataclass(frozen=True)
class EncoderVelocityConfig:
    """Immutable observer configuration; every field is validated eagerly."""

    method: str = "difference_lowpass"
    dt: float = 0.002
    cutoff_hz: float = 20.0
    alpha: float = 0.25
    beta: float = 0.04

    def __post_init__(self) -> None:
        if not isinstance(self.method, str):
            raise TypeError(f"method must be a str, got {type(self.method).__name__}")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")

        # Both methods validate all numeric fields, so a config is always a
        # legal config for either estimator.
        dt = _finite_scalar(self.dt, "dt")
        cutoff_hz = _finite_scalar(self.cutoff_hz, "cutoff_hz")
        alpha = _finite_scalar(self.alpha, "alpha")
        beta = _finite_scalar(self.beta, "beta")

        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt!r}")
        if cutoff_hz <= 0.0:
            raise ValueError(f"cutoff_hz must be > 0, got {cutoff_hz!r}")
        nyquist_hz = 0.5 / dt
        if not cutoff_hz < nyquist_hz:
            raise ValueError(
                f"cutoff_hz must be below the Nyquist frequency {nyquist_hz!r} Hz, "
                f"got {cutoff_hz!r}"
            )
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must satisfy 0 < alpha <= 1, got {alpha!r}")
        beta_limit = 4.0 - 2.0 * alpha
        if not 0.0 < beta < beta_limit:
            raise ValueError(
                f"beta must satisfy 0 < beta < 4 - 2*alpha = {beta_limit!r}, got {beta!r}"
            )

        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "cutoff_hz", cutoff_hz)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)


@dataclass(frozen=True)
class EncoderVelocityEstimate:
    """Observer output at one sampling instant."""

    time_s: float
    position_rad: float
    velocity_rad_s: float


class EncoderVelocityObserver:
    """Fixed-rate causal velocity observer driven by encoder positions only."""

    def __init__(self, config: EncoderVelocityConfig) -> None:
        if not isinstance(config, EncoderVelocityConfig):
            raise TypeError(
                "config must be an EncoderVelocityConfig, got "
                f"{type(config).__name__}"
            )
        self._config = config
        # One-pole discrete decay of the low-pass stage: r = exp(-2*pi*fc*dt).
        self._decay = math.exp(-2.0 * math.pi * (config.cutoff_hz * config.dt))
        self._state: EncoderVelocityEstimate | None = None

    @property
    def config(self) -> EncoderVelocityConfig:
        return self._config

    @property
    def state(self) -> EncoderVelocityEstimate:
        """Current estimate; reading it never advances the observer."""
        if self._state is None:
            raise RuntimeError("observer has no state yet: call reset() first")
        return self._state

    def reset(
        self,
        time_s: float,
        position_rad: float,
        *,
        known_initial_rest: bool,
    ) -> EncoderVelocityEstimate:
        """Drop all history and restart from an explicitly known rest state.

        ``known_initial_rest`` must be exactly ``True``: the zero initial
        velocity is a collection precondition the caller asserts, never
        something inferred from position differences.
        """
        if not isinstance(known_initial_rest, bool):
            raise TypeError(
                "known_initial_rest must be the bool True, got "
                f"{type(known_initial_rest).__name__}"
            )
        if known_initial_rest is not True:
            raise ValueError(
                "reset() requires known_initial_rest=True; initial rest must be an "
                "explicit measurement precondition"
            )
        time_s = _sample_time(time_s)
        position_rad = _finite_scalar(position_rad, "position_rad")
        self._state = EncoderVelocityEstimate(time_s, position_rad, 0.0)
        return self._state

    def update(self, time_s: float, position_rad: float) -> EncoderVelocityEstimate:
        """Consume one encoder sample and return the estimate at that instant.

        Repeated, out-of-order and skipped samples are rejected and leave the
        observer state untouched. Clock tolerance is min(1e-10, 0.25*dt)
        seconds, with no relative component, so sub-nanosecond sample periods
        cannot turn the absolute tolerance into permission to skip a sample.
        """
        previous = self._state
        if previous is None:
            raise RuntimeError("update() requires a preceding reset()")

        time_s = _sample_time(time_s)
        position_rad = _finite_scalar(position_rad, "position_rad")

        if time_s <= previous.time_s:
            raise ValueError("sample time must strictly increase")
        dt = self._config.dt
        expected_s = previous.time_s + dt
        clock_tolerance_s = min(CLOCK_ATOL_S, 0.25 * dt)
        if not abs(time_s - expected_s) <= clock_tolerance_s:
            raise ValueError(
                f"expected the next fixed-rate sample at t={expected_s!r} s "
                f"(atol={clock_tolerance_s!r}), got t={time_s!r} s"
            )

        try:
            if self._config.method == "difference_lowpass":
                raw_velocity = (position_rad - previous.position_rad) / dt
                velocity = (
                    self._decay * previous.velocity_rad_s
                    + (1.0 - self._decay) * raw_velocity
                )
                position = position_rad  # the measured position is reported as-is
            else:  # "alpha_beta"
                predicted_position = (
                    previous.position_rad + dt * previous.velocity_rad_s
                )
                residual = position_rad - predicted_position
                position = predicted_position + self._config.alpha * residual
                velocity = (
                    previous.velocity_rad_s + self._config.beta * residual / dt
                )
        except OverflowError as exc:
            raise ValueError(
                "encoder velocity update overflowed; observer state unchanged"
            ) from exc

        if not (math.isfinite(position) and math.isfinite(velocity)):
            raise ValueError(
                "encoder velocity update produced a non-finite result; observer "
                "state unchanged"
            )

        # Commit only once every arithmetic result is known to be finite.
        self._state = EncoderVelocityEstimate(time_s, position, velocity)
        return self._state
