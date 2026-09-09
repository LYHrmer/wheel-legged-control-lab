"""Immutable current-tick proposals, control memory and previous action receipts.

Controllers create a proposal through a pure preview. The runtime joins that
proposal to the exact state publication before asking the actor for an action.
No function here calls a controller or reads its private state. A controller's
last executed result is historical telemetry, never an implicit current proposal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from .state_estimation import D1StateEstimate, _immutable_array


def _tick(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("tick must be a non-negative integer")


def _finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class D1ControllerMemory:
    """Only integrals actually owned by the selected controller.

    ``None`` means the controller has no such state, not that its value is zero.
    Distances describe the state before this tick's single controller update.
    A controller with no integrators publishes the default empty memory.
    """

    distance_m: float | None = None
    distance_reference_m: float | None = None
    yaw_integral_nm: float | None = None
    wheel_integral_nm: np.ndarray | None = None

    def __post_init__(self) -> None:
        if (self.distance_m is None) != (self.distance_reference_m is None):
            raise ValueError("distance and distance_reference must be published together")
        for name in ("distance_m", "distance_reference_m", "yaw_integral_nm"):
            if (value := getattr(self, name)) is not None:
                _finite(value, name)
        if self.wheel_integral_nm is not None:
            object.__setattr__(
                self, "wheel_integral_nm", _immutable_array(self.wheel_integral_nm, shape=(4,))
            )


@dataclass(frozen=True, slots=True)
class D1ForceBaseline:
    """Current nominal force requests, before adding policy residuals.

    Fx is heading-longitudinal and precedes the combined baseline+residual
    limiter. Support Fz is the total nominal upward support request, not a
    delta; feedforward Fz is a separate additional request. Neither is an
    achieved contact force. Keep all three distinct in observations/logging.
    """

    longitudinal_force_n: float
    support_vertical_force_n: float
    vertical_feedforward_force_n: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "longitudinal_force_n",
            "support_vertical_force_n",
            "vertical_feedforward_force_n",
        ):
            _finite(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class D1JointTargetBaseline:
    """Current action-free targets; no fictitious whole-body Fx/Fz mapping."""

    nominal_joint_target_rad: np.ndarray
    nominal_wheel_speed_rad_s: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "nominal_joint_target_rad",
            _immutable_array(self.nominal_joint_target_rad, shape=(16,)),
        )
        object.__setattr__(
            self,
            "nominal_wheel_speed_rad_s",
            _immutable_array(self.nominal_wheel_speed_rad_s, shape=(4,)),
        )


@dataclass(frozen=True, slots=True)
class D1ControlProposal:
    """A controller's pure preview for one specific current state publication."""

    tick: int
    control_time_s: float
    baseline: D1ForceBaseline | D1JointTargetBaseline
    memory: D1ControllerMemory = field(default_factory=D1ControllerMemory)

    def __post_init__(self) -> None:
        _tick(self.tick)
        _finite(self.control_time_s, "control_time_s")
        if not isinstance(self.baseline, (D1ForceBaseline, D1JointTargetBaseline)):
            raise TypeError("baseline must explicitly name force or joint-target semantics")
        if not isinstance(self.memory, D1ControllerMemory):
            raise TypeError("memory must be D1ControllerMemory")


@dataclass(frozen=True, slots=True)
class D1AppliedAction:
    """Normalized policy command actually supplied to the controller last tick.

    The receipt refers to the completed interval [start_time_s, end_time_s].
    It is after policy clipping/delay, but is not a measured actuator torque.
    Actuator-channel requested/applied torques have their own physical record.
    """

    tick: int
    start_time_s: float
    end_time_s: float
    normalized_action: np.ndarray

    def __post_init__(self) -> None:
        _tick(self.tick)
        _finite(self.start_time_s, "start_time_s")
        _finite(self.end_time_s, "end_time_s")
        if self.end_time_s <= self.start_time_s:
            raise ValueError("action receipt requires a positive completed interval")
        array = np.asarray(self.normalized_action, dtype=np.float64)
        if array.ndim != 1 or not array.size:
            raise ValueError("normalized_action must be a nonempty vector")
        if not np.isfinite(array).all() or np.any(np.abs(array) > 1.0):
            raise ValueError("applied normalized action must be finite and clipped to [-1,1]")
        object.__setattr__(self, "normalized_action", _immutable_array(array, shape=array.shape))


@dataclass(frozen=True, slots=True)
class D1ControlContext:
    """The actor's current decision context, joined without copying state arrays.

    The proposal must describe the same publication as ``state``. At reset
    there is no previous receipt; thereafter exactly the previous completed
    tick is required. The state keeps its original measurement age, including
    delayed measurements, independently of the current proposal timestamp.
    """

    state: D1StateEstimate
    proposal: D1ControlProposal
    action_schema: str
    action_size: int
    previous_applied_action: D1AppliedAction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, D1StateEstimate) or not isinstance(
            self.proposal, D1ControlProposal
        ):
            raise TypeError("context requires a state publication and an explicit control proposal")
        if not isinstance(self.action_schema, str) or not self.action_schema.strip():
            raise ValueError("action_schema must be a nonempty name")
        if (
            isinstance(self.action_size, bool)
            or not isinstance(self.action_size, Integral)
            or self.action_size <= 0
        ):
            raise ValueError("action_size must be a positive integer")
        if self.state.sequence != self.proposal.tick or not np.isclose(
            self.state.control_time_s, self.proposal.control_time_s, rtol=0, atol=1e-10
        ):
            raise ValueError("proposal must describe the current state tick and publication time")
        previous = self.previous_applied_action
        if self.proposal.tick == 0:
            if previous is not None:
                raise ValueError("reset context cannot inherit a previous action receipt")
        elif previous is None:
            raise ValueError("noninitial context requires the previous completed action receipt")
        else:
            if not isinstance(previous, D1AppliedAction):
                raise TypeError("previous_applied_action must be D1AppliedAction")
            if previous.tick != self.proposal.tick - 1 or not np.isclose(
                previous.end_time_s, self.proposal.control_time_s, rtol=0, atol=1e-10
            ):
                raise ValueError("previous action must end at the current state publication")
            if previous.normalized_action.shape != (self.action_size,):
                raise ValueError("previous action size differs from the current action schema")

    @property
    def previous_normalized_action(self) -> np.ndarray:
        if self.previous_applied_action is None:
            return _immutable_array(np.zeros(self.action_size), shape=(self.action_size,))
        return self.previous_applied_action.normalized_action
