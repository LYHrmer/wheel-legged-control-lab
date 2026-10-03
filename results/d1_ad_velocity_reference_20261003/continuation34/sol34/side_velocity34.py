"""C34 body-velocity wrench term on the one C33/C31/Fast side instance.

Only the XY wrench supplied to the existing allocator changes. Importing this
module creates no plant, scratch data, model, controller, or physics step.
"""
from __future__ import annotations

import numpy as np

from body_curve33 import curve33

BETA34 = (0.0, 0.5, 1.0)
DISTANCE34 = (0.03, 0.04)
BODY_PHASES = ('shift', 'recenter')


def _coefficient(value, allowed, name):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
        raise ValueError(f'C34 {name} must be a finite pilot value')
    value = float(value)
    if value not in allowed:
        raise ValueError(f'C34 {name} must be a frozen pilot value')
    return value


def reference_velocity34(tick, *, phase, failure, cancelled):
    """Use the *current* C33 tick; never revive velocity after a transition.

    Returns (world velocity, active reference flag). Endpoint and non-body
    controls have zero velocity, including the exact duration boundary.
    """
    zero = np.zeros(3, dtype=float)
    if (phase not in BODY_PHASES or failure is not None or cancelled
            or tick is None or tick.get('source_phase') != phase):
        return zero, False
    duration = float(tick['duration_s'])
    elapsed = float(tick['elapsed_s'])
    span = np.asarray(tick['body_span_m'], dtype=float)
    if (span.shape != (3,) or not np.isfinite(span).all()
            or not np.isfinite(duration) or duration <= 0
            or not np.isfinite(elapsed)):
        raise ValueError('C34 C33 reference tick is malformed')
    if elapsed <= 0.0 or elapsed >= duration:
        return zero, False
    _, derivative, _ = curve33(elapsed, duration)
    velocity = span*derivative
    if not np.isfinite(velocity).all():
        raise ValueError('C34 reference velocity is nonfinite')
    return velocity, True


def compensated_wrench34(base_wrench, tick, *, phase, failure, cancelled,
                          beta, mass, body_kd):
    """Return base, vref, XY-only delta, desired, active as finite arrays.

    The 0.015 m budget still constrains the *reference acceleration* only;
    adding this wrench term changes the commanded force authority.
    """
    beta = _coefficient(beta, BETA34, 'beta')
    base = np.asarray(base_wrench, dtype=float)
    if base.shape != (6,) or not np.isfinite(base).all():
        raise ValueError('C34 base wrench must be finite six-vector')
    if not np.isfinite(mass) or mass <= 0 or not np.isfinite(body_kd) or body_kd < 0:
        raise ValueError('C34 mass/damping must be finite physical values')
    velocity, active = reference_velocity34(
        tick, phase=phase, failure=failure, cancelled=cancelled)
    delta = np.zeros(6, dtype=float)
    if beta == 0.0 or not active:
        desired = base.copy()  # exact Fast byte identity, including signed zero
    else:
        delta[:2] = float(mass)*float(body_kd)*beta*velocity[:2]
        desired = base + delta
    return base.copy(), velocity.copy(), delta, desired, active


def install_side34(side, *, beta, configured_distance_m):
    """Upgrade the existing C33 object; preserve its inherited allocation.

    The frozen C27 Hybrid calls side.start(..., distance_m=.03). C34 checks
    that bridge explicitly and forwards the configured .03/.04 to Fast.start.
    """
    beta = _coefficient(beta, BETA34, 'beta')
    distance = _coefficient(configured_distance_m, DISTANCE34, 'distance')
    if (side.phase != 'idle' or float(side._body_alpha33) != 1.0
            or side.side_arm != 'teacher'):
        raise RuntimeError('C34 requires idle C33 alpha=1 teacher side')

    class VelocitySide34(type(side)):
        def set_case34(self, *, beta, configured_distance_m):
            chosen_beta = _coefficient(beta, BETA34, 'beta')
            chosen_distance = _coefficient(configured_distance_m, DISTANCE34,
                                           'distance')
            if self.phase != 'idle' or float(self._body_alpha33) != 1.0:
                raise RuntimeError('C34 case may change only at idle alpha=1')
            self._body_beta34 = chosen_beta
            self._configured_distance34 = chosen_distance

        def set_alpha33(self, alpha):
            if float(alpha) != 1.0:
                raise ValueError('C34 holds the C33 body alpha at 1')
            return super().set_alpha33(alpha)

        def start(self, direction, distance_m=.03):
            if float(distance_m) != .03:
                raise RuntimeError('C34 requires the frozen Hybrid .03 start bridge')
            if float(self._body_alpha33) != 1.0:
                raise RuntimeError('C34 alpha changed before start')
            requested = float(self._configured_distance34)
            accepted = super().start(direction, distance_m=requested)
            self._last_start_distance34 = {
                'incoming_hybrid_distance_m': float(distance_m),
                'configured_distance_m': requested,
                'forwarded_fast_distance_m': requested,
                'accepted': bool(accepted)}
            return accepted

        def _allocate(self, points, com, wrench, weights):
            base, velocity, delta, desired, active = compensated_wrench34(
                wrench, getattr(self, '_body_reference_tick33', None),
                phase=self.phase, failure=self.failure,
                cancelled=bool(self.fixed_cancelled or self._skill30_cancel_requested),
                beta=self._body_beta34, mass=self.mass,
                body_kd=self.cfg.body_kd)
            forces = super()._allocate(points, com, desired, weights)
            if not np.array_equal(self.last_allocation['desired_wrench'], desired):
                raise RuntimeError('C34 inherited teacher allocator changed wrench')
            # C27's audited teacher allocation records the received wrench as
            # its original. Expose Fast's true pre-C34 wrench and the exact
            # term, while keeping C27's final desired/forces unaltered.
            self.last_allocation['original_wrench'] = base.copy()
            self.last_allocation['c34_base_wrench'] = base.copy()
            self.last_allocation['c34_reference_velocity_mps'] = velocity.copy()
            self.last_allocation['c34_beta'] = float(self._body_beta34)
            self.last_allocation['c34_wrench_delta_n'] = delta.copy()
            self.last_allocation['c34_desired_wrench'] = desired.copy()
            self.last_allocation['c34_velocity_reference_active'] = bool(active)
            self.last_allocation['c34_configured_distance_m'] = float(self._configured_distance34)
            tick = getattr(self, '_body_reference_tick33', None)
            self.last_allocation['c34_reference_tick'] = (None if tick is None else dict(tick))
            self.last_allocation['c34_start_distance'] = getattr(
                self, '_last_start_distance34', None)
            self.last_allocation['c34_force_semantics'] = (
                'original_wrench is Fast pre-beta; desired_wrench is final allocator input; '
                '0.015m bounds reference acceleration only')
            return forces

    side.__class__ = VelocitySide34
    side._body_beta34 = beta
    side._configured_distance34 = distance
    return side


def make_side34(plant, *, beta, configured_distance_m, geometry_binding,
                control_index_provider, cycle_index_provider):
    """Construct exactly the C33 fixed-reference side and upgrade in place."""
    from side_timing33 import make_side33

    side = make_side33(
        plant, alpha=1.0, geometry_binding=geometry_binding,
        control_index_provider=control_index_provider,
        cycle_index_provider=cycle_index_provider)
    return install_side34(side, beta=beta,
                          configured_distance_m=configured_distance_m)
