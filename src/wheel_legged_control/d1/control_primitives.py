"""Control math shared by historical experiments and the unified D1 loop."""

from __future__ import annotations

import math

import numpy as np


def terrain_normal_to_rpy(slope_x: float, slope_y: float, yaw_rad: float) -> tuple[float, float]:
    """Align body z to [-dh/dx, -dh/dy, 1], using Rz(yaw) Ry(pitch) Rx(roll)."""
    forward = slope_x * math.cos(yaw_rad) + slope_y * math.sin(yaw_rad)
    lateral = -slope_x * math.sin(yaw_rad) + slope_y * math.cos(yaw_rad)
    return math.atan2(lateral, math.sqrt(1.0 + forward * forward)), -math.atan(forward)


class YawRatePI:
    """Differential torque in N m with the historical bounded anti-windup rule."""

    def __init__(self, kp: float, ki: float):
        self.kp, self.ki = kp, ki
        self.integral_nm = 0.0

    def reset(self):
        self.integral_nm = 0.0

    def compute(self, error_rps: float, dt: float) -> float:
        candidate = float(np.clip(self.integral_nm + self.ki * error_rps * dt, -2.0, 2.0))
        request = self.kp * error_rps + candidate
        if abs(request) <= 4.0 or request * error_rps < 0:
            self.integral_nm = candidate
        return float(np.clip(self.kp * error_rps + self.integral_nm, -4.0, 4.0))
