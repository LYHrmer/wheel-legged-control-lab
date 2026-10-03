"""Independent exact C35 command-to-body Hermite reference replay."""
from __future__ import annotations

import math
import numpy as np

from task_math35 import (entry_heading35, hermite_coefficients35,
                         sample_coefficients35, integrate_coefficients35)


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise AssertionError(reason)


def close(a, b, atol=2e-10) -> bool:
    x, y = np.asarray(a), np.asarray(b)
    return bool(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
                and np.allclose(x, y, atol=atol, rtol=0))


def replay_body35(side_rows: list[dict], entry_qpos) -> dict:
    """Derive every consumed p/v/a/j from actual fresh raw side commands.

    This intentionally does not import Core35.  Segment replacement preserves
    the current v/a/jerk, including release during an unfinished start ramp.
    """
    need(bool(side_rows), 'C35 body replay lacks actual side controls')
    _, left = entry_heading35(entry_qpos)
    position = np.asarray(entry_qpos[:2], dtype=float).copy()
    velocity = acceleration = jerk = target = 0.
    coefficients = (0.,)*6
    elapsed = .25
    stopped = False
    first_stop = None
    samples = []
    prior_index = None
    for row in side_rows:
        index = row['control_index']
        need(prior_index is None or index == prior_index+1,
             'C35 body command control clocks are not contiguous')
        prior_index = index
        owner = row['info']['controller_record']
        controller = owner['controller_record35']
        command = owner['command35']
        need(command == controller['command']
             and command['vy_mps'] in (-.025, 0., .025)
             and command['issued_control_index'] == index
             and command['expires_control_index'] == index+20
             and type(command['gait_enabled']) is bool
             and command['stop_kind'] in ('none', 'release', 'cancel', 'conflict', 'expired'),
             'C35 fresh bounded raw velocity command differs')
        if (command['stop_kind'] != 'none'
                or command['vy_mps'] == 0. and not command['gait_enabled']):
            stopped = True
            if first_stop is None:
                first_stop = index
        desired = 0. if stopped else float(command['vy_mps'])
        if desired != target:
            coefficients = hermite_coefficients35(.25, velocity, acceleration,
                                                   jerk, desired)
            target, elapsed = desired, 0.
        next_elapsed = elapsed+.01
        distance = integrate_coefficients35(coefficients, elapsed, next_elapsed)
        position += left*distance
        elapsed = next_elapsed
        if elapsed >= .25:
            velocity, acceleration, jerk = target, 0., 0.
        else:
            _, velocity, acceleration, jerk = sample_coefficients35(
                coefficients, elapsed)
        reference = controller['reference']
        need(close(reference['ramp_coefficients_mps'], coefficients)
             and close(reference['ramp_elapsed_s'], elapsed)
             and close(reference['body_xy_m'], position)
             and close(reference['body_vxy_mps'], left*velocity)
             and close(reference['body_axy_mps2'], left*acceleration)
             and close(reference['body_jerk_xy_mps3'], left*jerk),
             'C35 consumed body pose/velocity/acceleration/jerk not one Hermite integral')
        sensed = owner['sensed35']
        contact_points = np.asarray(sensed['wheel_contact_points_world_m'], dtype=float)
        need(contact_points.shape == (4, 3) and np.isfinite(contact_points).all(),
             'C35 reference acceleration budget lacks actual contact-point geometry')
        com_height = float(sensed['whole_com_z_m'])-float(np.min(contact_points[:, 2]))
        need(com_height > 0. and abs(velocity) <= .030+1e-12
             and abs(acceleration)*com_height/9.81 <= .015+1e-12,
             'C35 smoothed velocity/reference acceleration budget exceeded')
        samples.append({'control_index': index, 'position_xy_m': position.copy(),
                        'signed_lateral_velocity_mps': velocity,
                        'signed_lateral_acceleration_mps2': acceleration,
                        'signed_lateral_jerk_mps3': jerk,
                        'requested_raw_vy_mps': command['vy_mps'],
                        'stop_latched': stopped})
    return {'controls': len(samples), 'first_stop_control_index': first_stop,
            'final_body_reference_xy_m': position,
            'final_signed_command_integral_m': float((position-entry_qpos[:2])@left),
            'max_abs_signed_reference_velocity_mps': max(
                abs(row['signed_lateral_velocity_mps']) for row in samples),
            'samples': samples}
