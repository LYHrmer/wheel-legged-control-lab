"""C33 body-only timing adapter around the same fixed C31 side instance.

Importing this module constructs no model or MjData. The factory upgrades the
one C31/C27 side object in place; the original phase gates, foot references,
IK, torque allocation, and safe handoff remain on their existing paths.
"""
from __future__ import annotations

import numpy as np

from body_curve33 import (ALPHAS33, PEAK_ACCEL33, RAMP_FRACTION33, SHAPE33,
                          curve33, duration33)


def make_side33(plant, *, alpha, geometry_binding, control_index_provider,
                cycle_index_provider):
    """Use C31 fixed XY/overlap with only its body shift/recenter shape changed."""
    from side_skill31 import make_side31

    if float(alpha) not in ALPHAS33:
        raise ValueError("C33 alpha must be a frozen pilot coefficient")
    side = make_side31(
        plant, mode='fixed', geometry_binding=geometry_binding,
        control_index_provider=control_index_provider,
        cycle_index_provider=cycle_index_provider)
    if (not side.cfg.adaptive_timing or side.cfg.zmp_budget_m != 0.015
            or side.cfg.shift_time_bounds[0] != 0.30):
        raise RuntimeError("C33 requires the frozen Fast body timing budget")

    class BodyTiming33(type(side)):
        def set_alpha33(self, alpha):
            if float(alpha) not in ALPHAS33:
                raise ValueError("C33 alpha must be a frozen pilot coefficient")
            if self.phase != 'idle':
                raise RuntimeError("C33 alpha may change only between reset cases")
            self._body_alpha33 = float(alpha)

        def _move_time(self, span):
            span = np.asarray(span, dtype=float)
            if span.shape != (3,):
                raise ValueError("C33 body span must have three components")
            if not np.isfinite(span[:2]).all():
                return float(self.cfg.shift_time_bounds[1])
            return duration33(span[:2], self.com_height,
                              self.cfg.zmp_budget_m, self._body_alpha33)

        def _advance(self):
            # C31/Fast still computes every stability predicate and transition.
            # Fast reads the prior control's acceleration for its dynamic-margin
            # gate; after its transition checks, supply this control's S reference
            # before IK and torque consume pose and body_accel.
            before_phase = self.phase
            self._body_reference_tick33 = None
            if before_phase in ('shift', 'recenter'):
                origin = np.asarray(self.body_from, dtype=float).copy()
                span = np.asarray(self.body_to, dtype=float).copy() - origin
                duration = float(self.shift_time)
                elapsed = float(self.phase_time + self.dt)
            super()._advance()
            if before_phase in ('shift', 'recenter'):
                position, _, acceleration = curve33(elapsed, duration)
                self._body_reference_tick33 = {
                    'source_phase': before_phase,
                    'body_from_m': origin.tolist(), 'body_span_m': span.tolist(),
                    'duration_s': duration,
                    'elapsed_s': elapsed,
                    'position_fraction': position,
                    'acceleration_fraction_per_s2': acceleration}
                if self.phase == before_phase:
                    self.pose = origin + span * position
                    self.body_accel = span * acceleration

        def compute(self):
            torque = super().compute()
            record = self.last_skill30_record
            if record is not None:
                velocity = np.zeros(3)
                if self.phase in ('shift', 'recenter'):
                    tick = self._body_reference_tick33
                    if tick is not None and tick['source_phase'] == self.phase:
                        _, derivative, _ = curve33(tick['elapsed_s'],
                                                    tick['duration_s'])
                        velocity = np.asarray(tick['body_span_m']) * derivative
                record['body_reference_velocity_mps'] = velocity.tolist()
                record['body_curve33'] = {
                    'shape': SHAPE33, 'ramp_fraction_of_total_duration': RAMP_FRACTION33,
                    'peak_normalized_acceleration': PEAK_ACCEL33,
                    'alpha': self._body_alpha33,
                    'effective_zmp_budget_m': self.cfg.zmp_budget_m*self._body_alpha33,
                    'reference_tick': self._body_reference_tick33}
            return torque

    side.__class__ = BodyTiming33
    side._body_alpha33 = float(alpha)
    return side
