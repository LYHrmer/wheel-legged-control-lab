"""Yaw command feedforward for the nominal wheel-leg yaw law.

The frozen law is ``effective = clip(cmd + K * (cmd - measured), +-limit)``. With the four
wheels in a line, the measured closed-loop yaw gain on the original B22 course is only about
0.63 of the command, because ground scrubbing absorbs most of the differential wheel speed.
This module adds one command term, ``effective = clip(ff * cmd + K * (cmd - measured), ...)``.

``ff == 1`` is the frozen law bit for bit, and any tick whose yaw command is zero is unchanged
for every ``ff``; legs, extension targets and the whole wheel/leg torque chain are untouched.
It is a subclass in a new module on purpose: the frozen controller file and its source
fingerprints stay valid for every earlier run.
"""

from __future__ import annotations

import numpy as np

from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.state_estimation import D1StateEstimate
from wheel_legged_control.d1.wheel_leg_controller import (
    D1WheelLegController,
    D1WheelLegTargets,
)

WHEEL_RADIUS_M = 0.087
WHEEL_SPEED_LIMIT_RAD_S = 30.0


class D1WheelLegYawFeedforwardController(D1WheelLegController):
    control_schema = "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedforward-v5"

    def __init__(self, *, yaw_command_feedforward: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.yaw_command_feedforward = _checked(yaw_command_feedforward)

    def nominal_targets(
        self, command: D1Command, state: D1StateEstimate, *, ground_height_m: float = 0.0
    ) -> D1WheelLegTargets:
        base = super().nominal_targets(command, state, ground_height_m=ground_height_m)
        if self.yaw_command_feedforward == 1.0:
            return base
        yaw = float(state.base_rpy[2])
        lateral = state.foot_offset_world @ np.asarray((-np.sin(yaw), np.cos(yaw), 0.0))
        effective = float(
            np.clip(
                self.yaw_command_feedforward * command.yaw_rate_rps
                + self.yaw_feedback_gain
                * (command.yaw_rate_rps - state.base_angular_velocity_body[2]),
                -self.yaw_request_limit_rps,
                self.yaw_request_limit_rps,
            )
        )
        wheels = np.clip(
            (command.forward_velocity_mps - effective * lateral) / WHEEL_RADIUS_M,
            -WHEEL_SPEED_LIMIT_RAD_S,
            WHEEL_SPEED_LIMIT_RAD_S,
        )
        return D1WheelLegTargets(
            base.nominal_joint_target_rad, wheels, base.leg_extension_target_m, effective
        )


def _checked(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 8.0:
        raise ValueError("yaw_command_feedforward must be finite and in [0, 8]")
    return value


def adopt_yaw_feedforward(
    nominal: D1WheelLegController, feedforward: float
) -> D1WheelLegYawFeedforwardController:
    """Switch an already built frozen controller to the feedforward law in place.

    The frozen B22 adapters construct and hold their own nominal controller, so a built
    instance is converted rather than rebuilt; only the exact frozen class is accepted and its
    state (wheel integral, gains, geometry tables) is kept as is.
    """
    if type(nominal) is not D1WheelLegController:
        raise TypeError("only the exact frozen D1WheelLegController can adopt the feedforward")
    nominal.__class__ = D1WheelLegYawFeedforwardController
    nominal.yaw_command_feedforward = _checked(feedforward)
    return nominal


__all__ = ("D1WheelLegYawFeedforwardController", "adopt_yaw_feedforward")
