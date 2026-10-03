"""Independent C35 command integral and actual displacement arithmetic."""
from __future__ import annotations

import math
import numpy as np


DT35 = .01
RAMP35 = .25
SPEED35 = .025


def hermite_velocity35(elapsed_s: float, duration_s: float, start_velocity_mps: float,
                       start_accel_mps2: float, start_jerk_mps3: float,
                       target_velocity_mps: float) -> tuple[float, float, float, float]:
    """Independent C2 quintic velocity segment and exact displacement integral.

    The target acceleration and jerk are zero.  Nonzero start derivatives are
    essential when cancel/expiry arrives during an unfinished command ramp.
    Returns displacement, velocity, acceleration and jerk at elapsed time.
    """
    args = (elapsed_s, duration_s, start_velocity_mps, start_accel_mps2,
            start_jerk_mps3, target_velocity_mps)
    need(all(math.isfinite(x) for x in args) and duration_s > 0.
         and 0. <= elapsed_s <= duration_s, 'invalid C35 Hermite segment')
    coefficients = hermite_coefficients35(duration_s, start_velocity_mps,
                                          start_accel_mps2, start_jerk_mps3,
                                          target_velocity_mps)
    return sample_coefficients35(coefficients, elapsed_s, duration_s)


def hermite_coefficients35(duration_s: float, start_velocity_mps: float,
                           start_accel_mps2: float, start_jerk_mps3: float,
                           target_velocity_mps: float) -> tuple[float, ...]:
    args = (duration_s, start_velocity_mps, start_accel_mps2,
            start_jerk_mps3, target_velocity_mps)
    need(all(math.isfinite(x) for x in args) and duration_s > 0.,
         'invalid C35 Hermite coefficient inputs')
    T = duration_s
    c0 = start_velocity_mps
    c1 = start_accel_mps2*T
    c2 = .5*start_jerk_mps3*T*T
    delta = target_velocity_mps-c0-c1-c2
    slope = -c1-2.*c2
    curve = -2.*c2
    c3 = 10.*delta-4.*slope+.5*curve
    c4 = -15.*delta+7.*slope-curve
    c5 = 6.*delta-3.*slope+.5*curve
    return (c0, c1, c2, c3, c4, c5)


def sample_coefficients35(coefficients, elapsed_s: float,
                          duration_s: float = RAMP35) -> tuple[float, float, float, float]:
    need(len(coefficients) == 6 and math.isfinite(elapsed_s)
         and 0. <= elapsed_s <= duration_s,
         'invalid C35 Hermite coefficient sample clock')
    T = duration_s
    coefficients = tuple(float(c) for c in coefficients)
    u = elapsed_s/T
    position = T*sum(c*u**(i+1)/(i+1) for i, c in enumerate(coefficients))
    velocity = sum(c*u**i for i, c in enumerate(coefficients))
    acceleration = sum(i*c*u**(i-1) for i, c in enumerate(coefficients) if i)/T
    jerk = sum(i*(i-1)*c*u**(i-2) for i, c in enumerate(coefficients) if i >= 2)/(T*T)
    return position, velocity, acceleration, jerk


def integrate_coefficients35(coefficients, elapsed_start_s: float,
                             elapsed_end_s: float,
                             duration_s: float = RAMP35) -> float:
    need(len(coefficients) == 6 and 0. <= elapsed_start_s <= elapsed_end_s
         and math.isfinite(elapsed_end_s) and duration_s > 0.,
         'invalid C35 reference integration interval')
    a = min(1., elapsed_start_s/duration_s)
    b = min(1., elapsed_end_s/duration_s)
    polynomial = duration_s*sum(float(c)*(b**(i+1)-a**(i+1))/(i+1)
                                for i, c in enumerate(coefficients))
    plateau = sum(float(c) for c in coefficients)
    return polynomial+plateau*max(0., elapsed_end_s-max(elapsed_start_s, duration_s))


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise AssertionError(reason)


def _q(u: float) -> float:
    return u*u*u*(10. + u*(-15. + 6.*u))


def _dq(u: float) -> float:
    return 30.*u*u*(1.-u)*(1.-u)


def _iq(u: float) -> float:
    return u**4*(2.5 + u*(-3. + u))


def commanded35(t_s: float, direction: int, active_controls: int) -> tuple[float, float, float]:
    """Signed lateral position, velocity, acceleration at a true control clock.

    The stop ramp begins at the raw active-control boundary.  Its integral
    completes the symmetric start/stop ramps, so a six-second ±.025 pulse
    commands exactly ±.150 m, with no credit for physical overshoot.
    """
    need(direction in (-1, 0, 1) and type(active_controls) is int
         and active_controls > 0 and math.isfinite(t_s) and t_s >= 0.,
         'invalid actual command clock or direction')
    active_s = active_controls*DT35
    need(active_s >= RAMP35, 'active command shorter than ramp')
    speed = direction*SPEED35
    if t_s < RAMP35:
        u = t_s/RAMP35
        return speed*RAMP35*_iq(u), speed*_q(u), speed/RAMP35*_dq(u)
    if t_s < active_s:
        return speed*(RAMP35*.5 + t_s-RAMP35), speed, 0.
    if t_s < active_s+RAMP35:
        u = (t_s-active_s)/RAMP35
        position = speed*(active_s-RAMP35*.5 + RAMP35*(u-_iq(u)))
        return position, speed*(1.-_q(u)), -speed/RAMP35*_dq(u)
    return speed*active_s, 0., 0.


def entry_heading35(entry_qpos) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(entry_qpos, dtype=np.float64)
    need(q.ndim == 1 and q.size >= 7 and np.isfinite(q).all(),
         'missing actual entry pose')
    w, x, y, z = q[3:7]
    need(abs(float(w*w+x*x+y*y+z*z)-1.) <= 1e-6,
         'invalid actual entry quaternion')
    yaw = math.atan2(2.*(w*z+x*y), 1.-2.*(y*y+z*z))
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-math.sin(yaw), math.cos(yaw)])
    return forward, left


def continuous_metrics35(side_qpos, direction: int,
                         active_controls: int = 600) -> dict:
    """Reconstruct held-window speed and tracking from actual origin states.

    `side_qpos` contains actual side pre-state then every side post-state,
    including the stop ramp.  The steady window is half-open controls
    [100,500), always from real pre/post XY differences, never reference metadata.
    """
    q = np.asarray(side_qpos, dtype=np.float64)
    need(q.ndim == 2 and q.shape[1] >= 7 and np.isfinite(q).all()
         and direction in (-1, 1) and active_controls == 600
         and len(q) >= active_controls+1,
         'missing six-second actual side origin states')
    forward, left = entry_heading35(q[0])
    origin_vxy = np.diff(q[:, :2], axis=0)/DT35
    signed_vy = direction*(origin_vxy @ left)
    steady = signed_vy[100:500]
    need(len(steady) == 400, 'incomplete held steady control window')
    tracking = steady-SPEED35
    signed_active_displacement = float(direction*(q[active_controls, :2]-q[0, :2])@left)
    reference_goal = commanded35(active_controls*DT35+RAMP35, direction,
                                 active_controls)[0]
    steady_net = float(direction*(q[500, :2]-q[100, :2])@left)
    return {'steady_control_interval_half_open': [100, 500],
            'mean_signed_origin_vy_mps': float(np.mean(steady)),
            'mean_signed_origin_vy_from_boundary_states_mps': steady_net/4.,
            'rms_steady_tracking_error_mps': float(np.sqrt(np.mean(tracking**2))),
            'fraction_steady_signed_vy_ge_010': float(np.mean(steady >= .010)),
            'signed_active_displacement_m': signed_active_displacement,
            'max_abs_longitudinal_excursion_active_m': float(np.max(np.abs(
                (q[:active_controls+1, :2]-q[0, :2]) @ forward))),
            'commanded_complete_goal_m': reference_goal,
            'commanded_complete_goal_abs_m': abs(reference_goal)}
