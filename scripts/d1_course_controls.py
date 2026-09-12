"""Course-only input and heading feedback; does not modify simulator state.

The formal locomotion policy and its frozen low-level gains are unaffected.
R resets and Space jump requests are dispatched by the course runner.
"""

from __future__ import annotations

import math
import time

from wheel_legged_control.d1.control_loop import D1MotionCommand


def wrap_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


class CourseKeyboardCommands:
    key_codes = frozenset(map(ord, "WASDQERXTG")) | {32, 256}
    heading_rate_rps = 0.60
    heading_gain = 2.0
    heading_damping = 0.4
    yaw_request_limit_rps = 1.00
    wheel_velocity_gain = 0.55

    def __init__(self, clock=time.monotonic, timeout_s=0.8):
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._finite(timeout_s)
        if timeout_s <= 0:
            raise ValueError("timeout must be positive")
        self._clock, self._timeout = clock, timeout_s
        self._stopped = False
        self._last_update = None
        self._focused = False
        self._state_time = self._control_time = 0.0
        self._actual_yaw = self._actual_velocity = 0.0
        self._actual_yaw_rate = 0.0
        self._fallen = False
        self.position_xy = (0.0, 0.0)
        self.goal_yaw_rad = 0.0
        self.requested_forward_mps = 0.0
        self._clearance = 0.455
        self._command = D1MotionCommand(0.0, 0.0, self._clearance)
        self.message = "Q/E set heading; Space jump; R simulation reset"

    @staticmethod
    def _finite(*values):
        if any(isinstance(v, bool) or not math.isfinite(v) for v in values):
            raise ValueError("finite numeric values required")

    def set_state(self, *, yaw_rad, position_xy, forward_velocity_mps=0.0,
                  yaw_rate_rps=0.0, fallen=False, simulation_time_s):
        position = tuple(position_xy)
        if len(position) != 2 or type(fallen) is not bool:
            raise ValueError("state needs XY and a boolean fallen flag")
        self._finite(yaw_rad, *position, forward_velocity_mps, yaw_rate_rps, simulation_time_s)
        if simulation_time_s < self._state_time:
            raise ValueError("state time moved backwards; reset the segment first")
        self._actual_yaw, self.position_xy = wrap_angle(yaw_rad), position
        self._actual_velocity, self._fallen = forward_velocity_mps, fallen
        self._actual_yaw_rate = yaw_rate_rps
        self._state_time = simulation_time_s

    def reset(self, yaw_rad, position_xy, simulation_time_s=0.0):
        position = tuple(position_xy)
        if len(position) != 2:
            raise ValueError("reset needs XY")
        self._finite(yaw_rad, *position, simulation_time_s)
        if simulation_time_s < 0:
            raise ValueError("negative reset time")
        self._state_time = self._control_time = simulation_time_s
        self._actual_yaw = wrap_angle(yaw_rad)
        self.position_xy = position
        self._actual_velocity, self._fallen = 0.0, False
        self._actual_yaw_rate = 0.0
        self._clearance = 0.455
        self._last_update = None
        self._focused = False
        self._cancel()
        self.message = "Simulation reset: upright at this zone's start"

    def _cancel(self):
        self.requested_forward_mps = 0.0
        self.goal_yaw_rad = self._actual_yaw
        self._command = D1MotionCommand(0.0, 0.0, self._clearance)

    def update_pressed(self, keys, focused=True):
        if not isinstance(keys, (set, frozenset)) or any(type(k) is not int for k in keys):
            raise TypeError("integer key set required")
        if type(focused) is not bool:
            raise TypeError("focused must be boolean")
        now = self._clock()
        self._finite(now)
        if self._last_update is not None and now < self._last_update:
            raise ValueError("input clock moved backwards")
        stale = self._last_update is not None and now-self._last_update >= self._timeout
        dt = 0.01 if self._last_update is None else min(0.1, now-self._last_update)
        self._control_time = self._state_time
        self._last_update = now
        if stale or focused != self._focused:
            self._cancel()
        self._focused = focused
        if focused and 256 in keys:
            self._stopped = True
        if self._stopped or not focused or self._fallen or ord("X") in keys or ord("R") in keys:
            self._cancel()
            self.message = "Fallen: press R for simulation reset" if self._fallen else "Motion cancelled"
            return
        forward = int(ord("W") in keys)-int(ord("S") in keys)
        target = 0.30 if forward > 0 else -0.20 if forward < 0 else 0.0
        if target == 0:
            self.requested_forward_mps = 0.0
        else:
            difference = target-self.requested_forward_mps
            self.requested_forward_mps += math.copysign(min(abs(difference), 0.5*dt), difference)
        turn = int(ord("Q") in keys)-int(ord("E") in keys)
        if turn:
            desired = wrap_angle(self.goal_yaw_rad+turn*self.heading_rate_rps*dt)
            error = max(-1.2, min(1.2, wrap_angle(desired-self._actual_yaw)))
            self.goal_yaw_rad = wrap_angle(self._actual_yaw+error)
        height = int(ord("T") in keys)-int(ord("G") in keys)
        self._clearance = max(0.43, min(0.48, self._clearance+0.025*height*dt))
        yaw_request = max(-self.yaw_request_limit_rps, min(self.yaw_request_limit_rps,
                          self.heading_gain*wrap_angle(self.goal_yaw_rad-self._actual_yaw)
                          -self.heading_damping*self._actual_yaw_rate))
        self._command = D1MotionCommand(self.requested_forward_mps, yaw_request, self._clearance)
        self.message = ("A/D side stepping is not available with this controller"
                        if ord("A") in keys or ord("D") in keys
                        else "Heading target retained after releasing Q/E")

    def __call__(self, simulation_time_s):
        self._finite(simulation_time_s)
        if simulation_time_s < 0:
            raise ValueError("negative simulation time")
        now = self._clock()
        self._finite(now)
        if self._last_update is not None:
            if now < self._last_update:
                raise ValueError("input clock moved backwards")
            if now-self._last_update >= self._timeout:
                self._cancel()
        return self._command

    @property
    def stopped(self):
        return self._stopped
