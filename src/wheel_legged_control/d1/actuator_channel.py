"""Synthetic motor-torque channel, advanced once per physics step (normally 2 ms).

Request -> input limit -> integer delay -> exact-ZOH first-order response ->
gain -> independent final limit. All torque values are joint-side N m; joint
velocities are rad/s, not revolutions/s. Passive friction remains in the plant:
the recorded velocity never adds or subtracts motor torque here.

Nonideal parameters are teaching assumptions, not identified D1 hardware data.
The implementation uses float64 and commits history only after a valid step.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

ScalarOrPerAxis = float | Sequence[float]


def _as_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not bool or complex")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _as_index(value: object, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not bool")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _check_sign(value: float, name: str, *, positive: bool = False) -> float:
    if value < 0 or (positive and value == 0):
        requirement = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {requirement}")
    return value


def _normalise(
    value: object, name: str, n_axes: int, *, positive: bool = False
) -> float | tuple[float, ...]:
    if isinstance(value, (tuple, list, np.ndarray)):
        if isinstance(value, np.ndarray) and value.ndim != 1:
            raise ValueError(f"{name} must be scalar or a 1-D per-axis sequence")
        if len(value) != n_axes:
            raise ValueError(f"{name} must provide exactly {n_axes} per-axis values")
        return tuple(
            _check_sign(_as_real(item, f"{name}[{index}]"), name, positive=positive)
            for index, item in enumerate(value)
        )
    return _check_sign(_as_real(value, name), name, positive=positive)


def _vector(value: object, name: str, n_axes: int | None = None) -> np.ndarray:
    # Inspect Python list elements before numpy can coerce a mixed bool/float
    # sequence to floats. Existing numeric ndarrays retain their input dtype.
    if isinstance(value, (list, tuple)):
        array = np.asarray(value, dtype=object)
        if array.ndim == 1:
            array = np.asarray([_as_real(item, name) for item in value], dtype=np.float64)
    else:
        array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be a real numeric vector, not bool or complex")
    if array.ndim != 1 or array.size == 0 or (n_axes is not None and array.shape != (n_axes,)):
        raise ValueError(f"{name} must be a nonempty vector of shape ({n_axes},)")
    with np.errstate(over="ignore", invalid="ignore"):
        result = array.astype(np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite in float64")
    return result


def _broadcast(value: float | tuple[float, ...], n_axes: int) -> np.ndarray:
    result = np.empty(n_axes, dtype=np.float64)
    result[:] = np.asarray(value, dtype=np.float64)
    return result


@dataclass(frozen=True)
class ActuatorChannelConfig:
    """Synthetic drive settings; sequence fields are copied into immutable tuples.

    ``physics_dt_s`` and ``time_constant_s`` use seconds. ``delay_steps`` counts
    physics steps; zero means the current command. ``gain`` is dimensionless
    and never rescales ``torque_limit_nm``. ``None`` disables both torque clips.
    Defaults give exact, same-step float64 passthrough on all 16 D1 axes.
    """

    n_axes: int = 16
    physics_dt_s: float = 0.002
    delay_steps: int = 0
    time_constant_s: ScalarOrPerAxis = 0.0
    gain: ScalarOrPerAxis = 1.0
    torque_limit_nm: ScalarOrPerAxis | None = None

    def __post_init__(self) -> None:
        n_axes = _as_index(self.n_axes, "n_axes", 1)
        object.__setattr__(self, "n_axes", n_axes)
        object.__setattr__(
            self,
            "physics_dt_s",
            _check_sign(_as_real(self.physics_dt_s, "physics_dt_s"), "physics_dt_s", positive=True),
        )
        object.__setattr__(self, "delay_steps", _as_index(self.delay_steps, "delay_steps", 0))
        for name in ("time_constant_s", "gain"):
            object.__setattr__(self, name, _normalise(getattr(self, name), name, n_axes))
        if self.torque_limit_nm is not None:
            object.__setattr__(
                self,
                "torque_limit_nm",
                _normalise(self.torque_limit_nm, "torque_limit_nm", n_axes, positive=True),
            )


@dataclass(frozen=True, eq=False)
class ActuatorTrace:
    """Immutable trace of one successful physics step.

    ``requested_nm`` is the controller request, ``limited_nm`` is its input
    clip, ``delayed_nm`` enters the response model, and ``applied_nm`` is the
    post-gain, final-limited motor torque to write to the plant. It excludes
    passive friction/gravity. ``joint_velocity_rps`` records rad/s only.
    Each field owns an immutable bytes-backed float64 vector.
    """

    requested_nm: np.ndarray
    limited_nm: np.ndarray
    delayed_nm: np.ndarray
    applied_nm: np.ndarray
    joint_velocity_rps: np.ndarray

    def __post_init__(self) -> None:
        n_axes = None
        for name in (
            "requested_nm",
            "limited_nm",
            "delayed_nm",
            "applied_nm",
            "joint_velocity_rps",
        ):
            value = _vector(getattr(self, name), name, n_axes)
            n_axes = value.size
            object.__setattr__(self, name, np.frombuffer(value.tobytes(), dtype=np.float64))


class ActuatorChannel:
    """A small stateful motor-torque Interface, independent of MuJoCo or policy.

    Each ``step`` consumes exactly one ``config.physics_dt_s`` interval. The
    caller must invoke it at the physics rate, not once per slower RL action.
    Invalid inputs or nonfinite intermediate torque leave state unchanged.
    """

    def __init__(self, config: ActuatorChannelConfig | None = None) -> None:
        if config is None:
            config = ActuatorChannelConfig()
        if not isinstance(config, ActuatorChannelConfig):
            raise TypeError("config must be ActuatorChannelConfig")
        self._config = config
        tau = _broadcast(config.time_constant_s, config.n_axes)
        self._gain = _broadcast(config.gain, config.n_axes)
        self._limit = (
            None
            if config.torque_limit_nm is None
            else _broadcast(config.torque_limit_nm, config.n_axes)
        )
        self._instant = tau == 0
        self._alpha = np.ones(config.n_axes, dtype=np.float64)
        # Finite dt/tiny tau may overflow to +inf: its exact ZOH limit is alpha=1.
        with np.errstate(over="ignore"):
            self._alpha[~self._instant] = -np.expm1(-config.physics_dt_s / tau[~self._instant])
        self.reset()

    @property
    def config(self) -> ActuatorChannelConfig:
        return self._config

    def reset(self) -> None:
        """Clear the response state and set every pending command to zero."""
        self._lag_state = np.zeros(self.config.n_axes, dtype=np.float64)
        self._history = np.zeros((self.config.delay_steps, self.config.n_axes), dtype=np.float64)
        self._cursor = 0

    def step(self, torque_nm: object, joint_velocity_rps: object) -> ActuatorTrace:
        """Advance one physics step; inputs are exact-shape vectors in N m/rad/s.

        Malformed inputs raise TypeError/ValueError. Nonfinite torque produced
        by a stage raises OverflowError before history or response is changed.
        """
        requested = _vector(torque_nm, "torque_nm", self.config.n_axes)
        velocity = _vector(joint_velocity_rps, "joint_velocity_rps", self.config.n_axes)
        with np.errstate(over="ignore", invalid="ignore"):
            limited = self._clip(requested)
            delayed = (
                limited.copy()
                if self.config.delay_steps == 0
                else self._history[self._cursor].copy()
            )
            lagged = np.where(
                self._instant,
                delayed,
                self._lag_state + self._alpha * (delayed - self._lag_state),
            )
            commanded = lagged * self._gain
            applied = self._clip(commanded)
        for name, vector in (("lag", lagged), ("gain", commanded), ("output limit", applied)):
            if not np.isfinite(vector).all():
                raise OverflowError(f"nonfinite {name} torque; state unchanged")
        trace = ActuatorTrace(requested, limited, delayed, applied, velocity)
        self._lag_state = lagged
        if self.config.delay_steps:
            self._history[self._cursor] = limited
            self._cursor = (self._cursor + 1) % self.config.delay_steps
        return trace

    def _clip(self, values: np.ndarray) -> np.ndarray:
        if self._limit is None:
            return values.copy()
        return np.clip(values, -self._limit, self._limit)
