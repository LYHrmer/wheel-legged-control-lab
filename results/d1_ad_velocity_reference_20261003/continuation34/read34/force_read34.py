"""Saved-only C34 force/torque proof, retaining the C27 complete arithmetic.

The final wrench is independently derived from pre-state, current C33 S tick,
and beta; all old allocation, Jacobian, PD, brake, inertia, and clip checks run.
"""
from __future__ import annotations

import numpy as np

from reader26 import require, close
from side_math27 import (LIMITS, SWING, compiled_side_geometry, scalar_state)
from reference_read33 import integrated_curve33


def c34_velocity34(before, after, *, cancelled):
    """Reconstruct world vref without importing runtime C33/C34 modules."""
    velocity = np.zeros(3)
    phase = before['phase']
    if (phase not in ('shift', 'recenter') or after['phase'] != phase
            or after['failure'] is not None or cancelled):
        return velocity, False
    span = np.asarray(before['body_to'], dtype=float)-np.asarray(before['body_from'], dtype=float)
    elapsed = float(before['phase_time_s'])+.01
    duration = float(before['shift_time_s'])
    require(span.shape == (3,) and np.isfinite(span).all(), 'C34 invalid saved body span')
    if elapsed >= duration:
        return velocity, False
    _, derivative, _ = integrated_curve33(elapsed, duration)
    return span*derivative, True


def recompute_side_torque34(record, qpos, qvel, binding, kin, profile,
                            *, cancelled_history=False, beta34, distance_m):
    """C27 physical chain with one exact beta wrench term and full torque proof."""
    require(record['schema'] == 'd1-c27-hybrid-traditional-side-v1'
            and profile == 'teacher', 'C34 side controller/profile differs')
    before, after = record['diagnostic_before'], record['diagnostic_after']
    calc = record['side_calculation']
    q, v = np.asarray(qpos), np.asarray(qvel)
    qa = np.asarray(binding['qpos_addresses'], dtype=int)
    da = np.asarray(binding['dof_addresses'], dtype=int)
    geometry = compiled_side_geometry(kin, q, v, binding['wheel_map'])
    require(close(calc['contact_points_m'], geometry['contact_points_m'], atol=2e-9)
            and close(calc['whole_com_m'], geometry['whole_com_m'], atol=2e-9),
            'C34 side geometry/COM input differs from compiled state')
    target = np.asarray(calc['target_position_rad'], dtype=float)
    require(target.shape == (16,) and np.isfinite(target).all(), 'C34 invalid saved IK target')
    vt = np.clip((target-np.asarray(before['previous_target']))/.01, -4., 4.)
    at = np.clip((vt-np.asarray(before['previous_velocity_target']))/.01, -60., 60.)
    require(close(calc['velocity_target_rad_s'], vt)
            and close(calc['accel_target_rad_s2'], at)
            and close(after['previous_target'], target)
            and close(after['previous_velocity_target'], vt),
            'C34 old joint target derivative/clip/state chain differs')
    state = scalar_state(q, v, binding)
    phase, leg = after['phase'], after['leg']
    require(close(calc['damping_multiplier_xy'], [1., 1.])
            and close(calc['xy_damping_delta_n'], [0., 0.]),
            'C34 old C27 damping multiplier was changed')
    mass = float(np.sum(kin['body_mass']))
    force = mass*(np.array([0., 0., 9.81])+np.asarray(after['body_accel'])
                  +26.*(np.asarray(after['pose'])-q[:3])-9.*state['world_com_velocity'])
    yaw = float(after['start_yaw_rad'])
    error_yaw = np.arctan2(np.sin(yaw-state['rpy'][2]), np.cos(yaw-state['rpy'][2]))
    c, s = np.cos(yaw), np.sin(yaw)
    heading = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    moment = 180.*(heading@np.array([-state['rpy'][0], -state['rpy'][1], error_yaw]))-25.*state['world_angular_velocity']
    base = np.r_[force, moment]
    beta = float(calc['c34_beta'])
    start = calc['c34_start_distance']
    require(beta == beta34 and distance_m in (.03, .04)
            and calc['c34_configured_distance_m'] == distance_m
            and start == dict(incoming_hybrid_distance_m=.03,
                              configured_distance_m=distance_m,
                              forwarded_fast_distance_m=distance_m,
                              accepted=True)
            and calc['c34_force_semantics'] == (
        'original_wrench is Fast pre-beta; desired_wrench is final allocator input; '
        '0.015m bounds reference acceleration only'),
        'C34 beta/distance/start bridge/force semantics differ')
    cancelled = bool(cancelled_history or record['cancel_requested'])
    vref, active = c34_velocity34(before, after, cancelled=cancelled)
    tick = calc['c34_reference_tick']
    if before['phase'] in ('shift', 'recenter'):
        origin = np.asarray(before['body_from'], dtype=float)
        span = np.asarray(before['body_to'], dtype=float)-origin
        elapsed = float(before['phase_time_s'])+.01
        duration = float(before['shift_time_s'])
        position, _, acceleration = integrated_curve33(elapsed, duration)
        require(tick is not None and tick['source_phase'] == before['phase']
                and close(tick['body_from_m'], origin)
                and close(tick['body_span_m'], span)
                and close(tick['elapsed_s'], elapsed)
                and close(tick['duration_s'], duration)
                and close(tick['position_fraction'], position)
                and close(tick['acceleration_fraction_per_s2'], acceleration),
                'C34 force used a different C33 body reference tick')
    else:
        require(tick is None, 'C34 nonbody force invented a body tick')
    delta = np.zeros(6)
    if beta == 0. or not active:
        desired = base.copy()
    else:
        delta[:2] = mass*9.*beta*vref[:2]
        desired = base+delta
    require(close(calc['original_wrench'], base, atol=2e-8)
            and close(calc['c34_base_wrench'], base, atol=2e-8)
            and close(calc['c34_reference_velocity_mps'], vref, atol=2e-10)
            and close(calc['c34_wrench_delta_n'], delta, atol=2e-8)
            and close(calc['c34_desired_wrench'], desired, atol=2e-8)
            and close(calc['desired_wrench'], desired, atol=2e-8)
            and calc['c34_velocity_reference_active'] is active,
            'C34 actual beta force differs from independently rebuilt Fast/C33 input')
    require(close(calc['original_wrench'][2:], calc['desired_wrench'][2:]),
            'C34 altered Fz or angular wrench')
    if beta == 0. or not active:
        require(np.asarray(calc['original_wrench'], dtype=float).tobytes()
                == np.asarray(calc['desired_wrench'], dtype=float).tobytes(),
                'C34 inactive/beta0 did not preserve exact old wrench bytes')
    weights = np.ones(4)
    if leg is not None and phase in SWING:
        weights[leg] = after['load_weight']
    elif after['loading_leg'] is not None:
        weights[after['loading_leg']] = after['load_weight']
    weights = np.clip(weights, 0., 1.)
    require(close(calc['weights'], weights), 'C34 old support weights differ')
    points, com = geometry['contact_points_m'], geometry['whole_com_m']
    allocation = np.zeros((6, 12))
    for i in range(4):
        x, y, z = points[i]-com
        allocation[:3, 3*i:3*i+3] = np.eye(3)
        allocation[3:, 3*i:3*i+3] = [[0., -z, y], [z, 0., -x], [-y, x, 0.]]
    penalty = np.repeat(.01/np.maximum(weights, 1e-3)**2, 3)
    try:
        allocated = np.linalg.solve(allocation.T@allocation+np.diag(penalty), allocation.T@desired)
    except np.linalg.LinAlgError:
        allocated = np.linalg.lstsq(allocation, desired, rcond=None)[0]
    allocated = allocated.reshape(4, 3)
    for i in range(4):
        allocated[i, 2] = np.clip(allocated[i, 2], 0., weights[i]*.85*mass*9.81)
        allocated[i, :2] = np.clip(allocated[i, :2], -.60*allocated[i, 2], .60*allocated[i, 2])
    require(close(calc['allocated_forces'], allocated, atol=2e-7),
            'C34 independent force allocation differs')
    tau = np.asarray(calc['bias_torque_nm'], dtype=float).copy()
    require(tau.shape == (16,) and np.isfinite(tau).all(), 'C34 invalid bound native bias torque')
    jacobians = calc['jacobians_by_leg']
    require(len(jacobians) == 4, 'C34 missing side Jacobian slots')
    for i in range(4):
        if weights[i] <= 1e-3:
            require(jacobians[i] is None, 'C34 unconsumed force Jacobian invented')
            continue
        jac = np.asarray(jacobians[i], dtype=float)
        require(jac.shape == (3, 16) and np.isfinite(jac).all(), 'C34 invalid force Jacobian')
        tau -= jac.T@allocated[i]
    kp, kd = np.full(16, 80.), np.full(16, 3.)
    swinging = leg is not None and phase in SWING
    if swinging:
        kp[4*leg:4*leg+3], kd[4*leg:4*leg+3] = 150., 5.
    velocity = v[da]
    tau += kp*(target-q[qa])+kd*(vt-velocity)
    inertia = calc['swing_inertia_3x3']
    if swinging and phase != 'unload':
        inertia = np.asarray(inertia, dtype=float)
        require(inertia.shape == (3, 3) and np.isfinite(inertia).all(),
                'C34 missing actual swing inertia cache')
        tau[4*leg:4*leg+3] += inertia@at[4*leg:4*leg+3]
    else:
        require(inertia is None, 'C34 unused swing inertia cache invented')
    wheel = np.arange(3, 16, 4)
    tau[wheel] += 12.*(np.asarray(after['wheel_angles'])-q[qa[wheel]])-2.*velocity[wheel]
    tau[wheel] -= kp[wheel]*(target[wheel]-q[qa[wheel]])+kd[wheel]*(vt[wheel]-velocity[wheel])
    safe = np.clip(tau, -LIMITS, LIMITS)
    require(close(record['returned_side_torque_nm'], safe, atol=2e-7),
            'C34 actual return differs from independent full torque arithmetic')
    return {'computed': {'safe_torque_nm': safe}, 'recomputed_preclip_torque_nm': tau,
            'geometry': geometry, 'state': state,
            'scope': 'independent saved force/allocation/torque arithmetic including beta XY term'}
