"""Body-mounted terrain preview for a straight rolling D1, from acquired ranges only."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

# Body axes x forward, y left, z up. Each track's beams are ordered near to far.
DEPRESSION_RAD = np.deg2rad([74.0, 56.0, 43.0, 33.0])
ORIGINS = np.array([[0.195, 0.190, 0.055], [0.195, -0.190, 0.055]])
DIRECTIONS = np.column_stack((np.cos(DEPRESSION_RAD), np.zeros(4), -np.sin(DEPRESSION_RAD)))
RATE_HZ = 6.0
LATENCY_S = 0.040
MAX_AGE_S = 0.250
LOOKAHEAD_S = 0.80
SLEW_S = 0.10
FORCE_RESERVE_SHARE = 0.20
REFERENCE_GRADE_RAD = math.radians(4.0)
MAX_GRADE_RAD = math.radians(12.0)
TRACK_AGREEMENT_RAD = math.radians(3.0)
DEADBAND_RAD = math.radians(0.20)


def read_ranges(model, data, base_body_id: int) -> np.ndarray:
    """Sensor adapter. Only first-return distance leaves the simulated rangefinder."""
    rotation = data.xmat[base_body_id].reshape(3, 3)
    origins = ORIGINS @ rotation.T + data.xpos[base_body_id]
    directions = DIRECTIONS @ rotation.T
    ranges = np.full((2, 4), np.nan)
    geom_id = np.empty(1, dtype=np.int32)
    for lane, origin in enumerate(origins):
        for beam, direction in enumerate(directions):
            distance = mujoco.mj_ray(model, data, origin, direction, None, True, -1, geom_id)
            if 0.14 <= distance <= 1.45:
                ranges[lane, beam] = distance
    return ranges


def _json_grid(grid):
    return [[float(v) if np.isfinite(v) else None for v in lane] for lane in grid]


@dataclass(frozen=True)
class RangeFrame:
    sequence: int
    captured_s: float
    available_s: float
    travelled_m: float
    ranges: np.ndarray
    x_m: np.ndarray
    z_m: np.ndarray

    @classmethod
    def from_ranges(cls, sequence, captured_s, available_s, travelled_m, ranges, pitch, roll):
        """Level the returns using the attitude estimate available at capture time."""
        ranges = np.asarray(ranges, dtype=float)
        if ranges.shape != (2, 4):
            raise ValueError("the rangefinder has two tracks of four beams")
        distances = np.where((ranges >= 0.14) & (ranges <= 1.45), ranges, np.nan)
        xyz = ORIGINS[:, None, :] + distances[:, :, None] * DIRECTIONS
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        # R_y(pitch) R_x(roll), retaining the capture frame's origin and heading.
        z_after_roll = sr * xyz[..., 1] + cr * xyz[..., 2]
        x = cp * xyz[..., 0] + sp * z_after_roll
        z = -sp * xyz[..., 0] + cp * z_after_roll
        return cls(sequence, captured_s, available_s, travelled_m, distances, x, z)


@dataclass(frozen=True)
class PreviewReading:
    frame: RangeFrame | None
    query_x_m: float
    reason: str
    left_slope_rad: float | None = None
    right_slope_rad: float | None = None

    @property
    def valid(self):
        return self.reason == "valid"

    @property
    def slope_rad(self):
        if not self.valid:
            return None
        return (self.left_slope_rad + self.right_slope_rad) * 0.5

    @property
    def sequence(self):
        return self.frame.sequence if self.frame else None

    @property
    def captured_s(self):
        return self.frame.captured_s if self.frame else None

    @property
    def available_s(self):
        return self.frame.available_s if self.frame else None

    def as_dict(self):
        frame = self.frame
        return {
            "sequence": self.sequence, "captured_s": self.captured_s,
            "available_s": self.available_s, "query_x_m": self.query_x_m,
            "valid": self.valid, "reason": self.reason, "slope_rad": self.slope_rad,
            "left_slope_rad": self.left_slope_rad, "right_slope_rad": self.right_slope_rad,
            "range_m": _json_grid(frame.ranges) if frame else None,
            "x_m": _json_grid(frame.x_m) if frame else None,
            "z_m": _json_grid(frame.z_m) if frame else None,
        }


def _lane_grade(x, z, query):
    # Mask each original adjacent pair. A missing beam must not bridge a gap.
    widths = np.diff(x)
    inside = (x[:-1] <= query) & (query < x[1:]) & (widths >= 0.035)
    indices = np.flatnonzero(inside & np.isfinite(z[:-1]) & np.isfinite(z[1:]))
    if indices.size != 1:
        return None
    i = indices.item()
    return math.atan2(z[i + 1] - z[i], widths[i])


def query_frame(frame, now_s, travelled_m, forward_track_x_m, speed_mps):
    """Query the expected wheel track in the *original measured* local profile."""
    query = forward_track_x_m + speed_mps * LOOKAHEAD_S
    if frame is None:
        return PreviewReading(None, query, "waiting")
    query += travelled_m - frame.travelled_m
    times = frame.captured_s, frame.available_s, now_s
    if not all(math.isfinite(v) for v in (*times, query)):
        return PreviewReading(frame, query, "nonfinite")
    if not frame.captured_s <= frame.available_s <= now_s + 1e-12:
        return PreviewReading(frame, query, "time_order")
    if now_s - frame.captured_s > MAX_AGE_S + 1e-12:
        return PreviewReading(frame, query, "expired")
    grades = [_lane_grade(x, z, query) for x, z in zip(frame.x_m, frame.z_m)]
    if any(g is None for g in grades):
        return PreviewReading(frame, query, "unobserved", *grades)
    if max(abs(v) for v in grades) > MAX_GRADE_RAD or abs(grades[0]-grades[1]) > TRACK_AGREEMENT_RAD:
        return PreviewReading(frame, query, "inconsistent", *grades)
    return PreviewReading(frame, query, "valid", *grades)


class PreviewChannel:
    def __init__(self):
        self._pending = deque()
        self._frame = None
        self._next = 0
        self._last_s = -math.inf
        self._delta = 0.0

    def observe(self, plant, now_s, *, measured_body_pitch_rad, measured_body_roll_rad,
                travelled_m, forward_track_x_m, speed_mps):
        if now_s < self._last_s or not math.isfinite(now_s):
            raise ValueError("preview clock must be finite and monotone in one episode")
        self._last_s = now_s
        if now_s + 1e-12 >= self._next / RATE_HZ:
            ranges = read_ranges(plant.model, plant.measurement_data, plant.base_body_id)
            self._pending.append(RangeFrame.from_ranges(
                self._next, now_s, now_s + LATENCY_S, travelled_m, ranges,
                measured_body_pitch_rad, measured_body_roll_rad,
            ))
            # Sampling starts at the first available control tick; missed exposures stay missed.
            self._next = math.floor((now_s + 1e-12) * RATE_HZ) + 1
        while self._pending and self._pending[0].available_s <= now_s + 1e-12:
            self._frame = self._pending.popleft()
        return query_frame(self._frame, now_s, travelled_m, forward_track_x_m, speed_mps)

    def correction(self, reading, arm, *, support_pitch_rad, baseline_force_n,
                   pitch_gain_n_per_rad, force_limit_n, dt_s):
        """Return an additive pitch reference. The rest of LQR/VMC is shared."""
        if arm not in ("on", "off", "inverted"):
            raise ValueError("unknown preview arm")
        if not all(math.isfinite(v) for v in (support_pitch_rad, baseline_force_n,
                   pitch_gain_n_per_rad, force_limit_n, dt_s)) or (
            force_limit_n <= 0 or abs(pitch_gain_n_per_rad) < 1e-12 or dt_s <= 0
        ):
            raise ValueError("invalid control gain, step or force bound")
        full_scale = FORCE_RESERVE_SHARE * force_limit_n / abs(pitch_gain_n_per_rad)
        available = full_scale * max(0.0, 1.0 - abs(baseline_force_n) / force_limit_n)
        fraction = 0.0
        if reading.valid and arm != "off":
            # Geometry slope is positive uphill; simulated D1 nose-up pitch is negative.
            difference = -reading.slope_rad - support_pitch_rad
            if abs(difference) > DEADBAND_RAD:
                fraction = float(np.clip(difference / REFERENCE_GRADE_RAD, -1, 1))
            if arm == "inverted":
                fraction = -fraction
        target = available * fraction
        step = full_scale * dt_s / SLEW_S
        delta = self._delta + float(np.clip(target - self._delta, -step, step))
        self._delta = float(np.clip(delta, -available, available))
        return self._delta
