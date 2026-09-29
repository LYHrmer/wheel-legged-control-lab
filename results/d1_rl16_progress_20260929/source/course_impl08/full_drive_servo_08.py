"""Pure one-tick rate-limited request publication for the new RL task.

The operator/training target is never silently replaced by the servo value.
Each tick publishes both. A product-limited intermediate state delays an
increasing component while the other decays, retaining both rate limits and
an explicit status. This is command shaping, not observed robot acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass

from full_drive_command_08 import FullDriveCommand, QualifiedCommandCaps

SERVO_SCHEMA = "d1-full-drive-request-servo-v1"
CONTROL_DT_S = 0.01
FORWARD_RATE_MPS2 = 0.5
YAW_RATE_RPS2 = 0.6
COMBINED_REQUEST_LIMIT_MPS2 = 0.8


def _toward(previous: float, target: float, maximum_step: float) -> float:
    return previous + max(-maximum_step, min(maximum_step, target - previous))


@dataclass(frozen=True, slots=True)
class ServoReceipt:
    schema: str
    tick: int
    raw_target: FullDriveCommand
    independently_rate_limited: FullDriveCommand
    applied: FullDriveCommand
    product_delayed_axis: str | None
    forward_increment_mps: float
    yaw_increment_rps: float


class FullDriveCommandServo:
    """Advance a qualified request exactly once for each control tick."""

    def __init__(self, caps: QualifiedCommandCaps) -> None:
        if not isinstance(caps, QualifiedCommandCaps):
            raise TypeError("command servo requires explicit qualified caps")
        self.caps = caps
        self.reset()

    def reset(self) -> None:
        self._last_tick = -1
        self._applied = FullDriveCommand()
        self._receipt: ServoReceipt | None = None

    @property
    def last_receipt(self) -> ServoReceipt | None:
        return self._receipt

    def advance(self, tick: int, raw_target: FullDriveCommand) -> ServoReceipt:
        if type(tick) is not int or tick < 0:
            raise ValueError("servo tick must be a non-negative integer")
        if not isinstance(raw_target, FullDriveCommand):
            raise TypeError("servo raw target must be FullDriveCommand")
        if tick == self._last_tick:
            if self._receipt is None or raw_target != self._receipt.raw_target:
                raise RuntimeError("same-tick servo command cannot change")
            return self._receipt
        if tick != self._last_tick + 1:
            raise RuntimeError("servo tick cannot skip or replay a control interval")
        self.caps.check(raw_target)
        if raw_target.lateral_velocity_mps != 0.0 or raw_target.jump_requested:
            raise ValueError("first command-servo pilot permits no lateral or jump request")
        if abs(raw_target.forward_velocity_mps * raw_target.yaw_rate_rps) > COMBINED_REQUEST_LIMIT_MPS2:
            raise ValueError("raw forward/yaw combination exceeds the pilot request limit")

        old = self._applied
        forward = _toward(old.forward_velocity_mps, raw_target.forward_velocity_mps,
                         FORWARD_RATE_MPS2 * CONTROL_DT_S)
        yaw = _toward(old.yaw_rate_rps, raw_target.yaw_rate_rps,
                      YAW_RATE_RPS2 * CONTROL_DT_S)
        independent = FullDriveCommand(forward, 0.0, yaw, raw_target.clearance_m, False)
        delayed_axis: str | None = None
        if abs(forward * yaw) > COMBINED_REQUEST_LIMIT_MPS2:
            forward_increasing = abs(forward) > abs(old.forward_velocity_mps)
            yaw_increasing = abs(yaw) > abs(old.yaw_rate_rps)
            if forward_increasing and not yaw_increasing:
                forward = old.forward_velocity_mps
                delayed_axis = "forward"
            elif yaw_increasing and not forward_increasing:
                yaw = old.yaw_rate_rps
                delayed_axis = "yaw"
            else:
                # Feasible endpoints and monotone steps normally rule this
                # out; stay put and record it rather than violating either
                # the product or an axis-rate limit through a sudden clip.
                forward, yaw = old.forward_velocity_mps, old.yaw_rate_rps
                delayed_axis = "both"
        if (abs(forward * yaw) > COMBINED_REQUEST_LIMIT_MPS2
                or abs(forward - old.forward_velocity_mps) > FORWARD_RATE_MPS2 * CONTROL_DT_S + 1e-15
                or abs(yaw - old.yaw_rate_rps) > YAW_RATE_RPS2 * CONTROL_DT_S + 1e-15):
            raise AssertionError("command shaping violated its physical request envelope")
        applied = FullDriveCommand(forward, 0.0, yaw, raw_target.clearance_m, False)
        receipt = ServoReceipt(
            SERVO_SCHEMA, tick, raw_target, independent, applied, delayed_axis,
            forward - old.forward_velocity_mps, yaw - old.yaw_rate_rps,
        )
        self._last_tick, self._applied, self._receipt = tick, applied, receipt
        return receipt
