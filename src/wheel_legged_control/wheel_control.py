"""Single-wheel PI speed control with optional reference-model feedforward.

This controller accepts measured speed, not an ``ActuatorBench`` or its hidden
parameters. All torques are joint-side N m. The feedforward model does not
compensate command delay, even when its parameters contain ``delay_steps``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, tanh
from numbers import Real

from wheel_legged_control.actuator_bench import (
    FRICTION_SMOOTHING_RAD_S,
    WHEEL_INERTIA_KG_M2,
    ActuatorParameters,
)


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


@dataclass(frozen=True)
class WheelControlConfig:
    """PI gains and sampling: kp [N m s/rad], ki [N m/rad], dt [s]."""

    kp: float = 0.12
    ki: float = 0.6
    dt: float = 0.002
    torque_limit_nm: float = 2.0

    def __post_init__(self) -> None:
        for name in ("kp", "ki", "dt", "torque_limit_nm"):
            value = _finite_scalar(getattr(self, name), name)
            if value < 0.0 or (name in ("dt", "torque_limit_nm") and value == 0.0):
                requirement = "non-negative" if name in ("kp", "ki") else "positive"
                raise ValueError(f"{name} must be {requirement}")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class WheelControlOutput:
    """One control sample, with the accepted integral already included.

    ``saturated`` means the final requested torque exceeds the limit.
    ``integration_frozen`` means a candidate integral increment was rejected;
    this can be true even if the final torque is not saturated.
    """

    error_rad_s: float
    proportional_torque_nm: float
    integral_torque_nm: float
    feedforward_torque_nm: float
    requested_torque_nm: float
    command_torque_nm: float
    saturated: bool
    integration_frozen: bool


class WheelVelocityController:
    """Compute one command per ``config.dt``, then advance the plant once.

    At sample k, calculate error from the current measurement and propose
    ``I_candidate = I_previous + ki * error * dt``. Reject this increment only
    when the candidate total torque is outside the limit and the increment
    pushes farther out. Otherwise accept it, including increments that unwind
    an already saturated command. Recompute total torque with the accepted
    integral, then clip. The integral is stored in N m, not radian-seconds.

    Optional feedforward is ``(J_load + armature) * reference_acceleration +
    damping * reference_velocity + coulomb * tanh(reference_velocity / 0.05)``.
    Reference acceleration must come from the command generator, not future
    measured states. No delay inverse or low-speed Stribeck compensation is
    performed. ``reset`` clears only this controller's integral state.
    """

    def __init__(
        self,
        config: WheelControlConfig,
        feedforward_parameters: ActuatorParameters | None = None,
        load_inertia_kg_m2: float = WHEEL_INERTIA_KG_M2,
    ) -> None:
        if not isinstance(config, WheelControlConfig):
            raise TypeError("config must be WheelControlConfig")
        if feedforward_parameters is not None and not isinstance(
            feedforward_parameters, ActuatorParameters
        ):
            raise TypeError("feedforward_parameters must be ActuatorParameters or None")
        inertia = _finite_scalar(load_inertia_kg_m2, "load_inertia_kg_m2")
        if inertia <= 0.0:
            raise ValueError("load_inertia_kg_m2 must be positive")
        self.config = config
        self.feedforward_parameters = feedforward_parameters
        self.load_inertia_kg_m2 = inertia
        self.reset()

    def reset(self) -> None:
        self._integral_torque_nm = 0.0

    def compute(
        self,
        reference_velocity_rad_s: float,
        reference_acceleration_rad_s2: float,
        measured_velocity_rad_s: float,
    ) -> WheelControlOutput:
        reference = _finite_scalar(reference_velocity_rad_s, "reference_velocity_rad_s")
        acceleration = _finite_scalar(
            reference_acceleration_rad_s2, "reference_acceleration_rad_s2"
        )
        measured = _finite_scalar(measured_velocity_rad_s, "measured_velocity_rad_s")
        error = reference - measured
        proportional = self.config.kp * error
        increment = self.config.ki * error * self.config.dt
        candidate_integral = self._integral_torque_nm + increment
        feedforward = 0.0
        parameters = self.feedforward_parameters
        if parameters is not None:
            feedforward = (
                (self.load_inertia_kg_m2 + parameters.armature) * acceleration
                + parameters.damping * reference
                + parameters.coulomb_friction * tanh(reference / FRICTION_SMOOTHING_RAD_S)
            )
        candidate = proportional + candidate_integral + feedforward
        if not all(
            isfinite(value)
            for value in (
                error,
                proportional,
                increment,
                candidate_integral,
                feedforward,
                candidate,
            )
        ):
            raise ValueError("control arithmetic produced a non-finite value")
        limit = self.config.torque_limit_nm
        frozen = (candidate > limit and increment > 0.0) or (candidate < -limit and increment < 0.0)
        integral = self._integral_torque_nm if frozen else candidate_integral
        requested = proportional + integral + feedforward
        if not isfinite(requested):
            raise ValueError("control arithmetic produced a non-finite value")
        self._integral_torque_nm = integral
        return WheelControlOutput(
            error_rad_s=error,
            proportional_torque_nm=proportional,
            integral_torque_nm=integral,
            feedforward_torque_nm=feedforward,
            requested_torque_nm=requested,
            command_torque_nm=max(-limit, min(limit, requested)),
            saturated=abs(requested) > limit,
            integration_frozen=frozen,
        )
