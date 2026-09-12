"""Held-key commands; no simulator access and no dependency on keyboard repeat."""

from __future__ import annotations

import math
import time

from wheel_legged_control.d1.control_loop import D1MotionCommand


class HeldKeyboardCommands:
    def __init__(self, clock=time.monotonic, timeout_s=0.8):
        if not callable(clock):
            raise TypeError("clock must be callable")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self._clock, self._timeout = clock, float(timeout_s)
        self._last_update = None
        self._vx = self._yaw = 0.0
        self._clearance = 0.455
        self._stopped = False

    @staticmethod
    def _towards(value, target, amount):
        if target == 0:
            return 0.0
        return value + math.copysign(min(abs(target - value), amount), target - value)

    def _now(self):
        now = self._clock()
        if isinstance(now, bool) or not math.isfinite(now):
            raise ValueError("clock must return a finite time")
        if self._last_update is not None and now < self._last_update:
            raise ValueError("input clock moved backwards")
        return now

    def update_pressed(self, keys: set[int], focused: bool = True):
        if not isinstance(keys, (set, frozenset)) or any(type(key) is not int for key in keys):
            raise TypeError("pressed keys must be a set of integer GLFW key codes")
        if type(focused) is not bool:
            raise TypeError("focused must be bool")
        now = self._now()
        dt = 0.01 if self._last_update is None else min(0.1, now - self._last_update)
        if self._last_update is not None and now - self._last_update >= self._timeout:
            self._vx = self._yaw = 0.0
        self._last_update = now
        if focused and 256 in keys:
            self._stopped = True
        if self._stopped or not focused or 32 in keys:
            self._vx = self._yaw = 0.0
            return
        forward = int(ord("W") in keys) - int(ord("S") in keys)
        turn = int(ord("A") in keys) - int(ord("D") in keys)
        target_vx = 0.30 if forward > 0 else -0.20 if forward < 0 else 0.0
        self._vx = self._towards(self._vx, target_vx, 0.5 * dt)
        self._yaw = self._towards(self._yaw, 0.25 * turn, 0.8 * dt)
        height = int(ord("R") in keys) - int(ord("F") in keys)
        self._clearance = min(0.48, max(0.43, self._clearance + 0.025 * height * dt))

    def __call__(self, simulation_time_s):
        if (
            isinstance(simulation_time_s, bool)
            or not math.isfinite(simulation_time_s)
            or simulation_time_s < 0
        ):
            raise ValueError("simulation_time_s must be finite and non-negative")
        now = self._now()
        if self._last_update is not None and now - self._last_update >= self._timeout:
            self._vx = self._yaw = 0.0
        return D1MotionCommand(self._vx, self._yaw, self._clearance)

    @property
    def stopped(self):
        return self._stopped
