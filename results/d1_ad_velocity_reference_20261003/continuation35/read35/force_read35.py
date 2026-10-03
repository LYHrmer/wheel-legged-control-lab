"""Saved-only C35 paired force/allocation/PD/inertia/brake/clip reconstruction."""
from __future__ import annotations

import math
import numpy as np

from side_math27 import LIMITS, compiled_side_geometry, scalar_state


AIR35 = ('lift', 'horizontal', 'lower', 'probe', 'dwell')


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise AssertionError(reason)


def close(a, b, atol=2e-8) -> bool:
    x, y = np.asarray(a), np.asarray(b)
    return bool(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
                and np.allclose(x, y, atol=atol, rtol=0))


def smooth35(u: float) -> float:
    u = min(1., max(0., u))
    return u**3*(10.-15.*u+6.*u*u)


def recompute_paired_torque35(record: dict, qpos, qvel, binding: dict,
                              kin: dict, *, entry_z_m: float,
                              entry_yaw_rad: float,
                              previous_reference: dict | None = None,
                              joint_ids, joint_ranges, joint_limited) -> dict:
    """Rebuild each consumed 16-torque from old compiled geometry and new ref.

    Native reader26 must separately tie the returned torque to all five actual
    requested/limited/delayed/applied actuator traces.  Saved Jacobian/fullM
    caches remain explicit inputs, shape/finite/phase-checked here.
    """
    need(record['schema'] == 'd1-c35-continuous-side-control-v1'
         and math.isfinite(entry_z_m) and math.isfinite(entry_yaw_rad),
         'C35 side force input/schema differs')
    controller = record['controller_record35']
    need(controller['schema'] == 'd1-c35-continuous-side-control-v1',
         'C35 paired controller record schema differs')
    reference, calc = controller['reference'], controller['calculation']
    q, v = np.asarray(qpos, dtype=float), np.asarray(qvel, dtype=float)
    qa = np.asarray(binding['qpos_addresses'], dtype=int)
    da = np.asarray(binding['dof_addresses'], dtype=int)
    need(q.shape == (23,) and v.shape == (22,) and qa.shape == da.shape == (16,)
         and close(calc['pre_qpos'], q) and close(calc['pre_qvel'], v),
         'C35 force pre-state detached from actual saved control state')
    target = np.asarray(calc['target16'], dtype=float)
    previous = np.asarray(calc['previous_target16'], dtype=float)
    previous_v = np.asarray(calc['previous_velocity_target16'], dtype=float)
    need(target.shape == previous.shape == previous_v.shape == (16,)
         and np.isfinite(target).all() and np.isfinite(previous).all()
         and np.isfinite(previous_v).all(), 'C35 paired joint target invalid')
    ids = np.asarray(joint_ids, dtype=int)
    ranges = np.asarray(joint_ranges, dtype=float)
    limited = np.asarray(joint_limited, dtype=bool)
    need(ids.shape == (16,) and len(set(ids.tolist())) == 16
         and ranges.ndim == 2 and ranges.shape[1] == 2
         and limited.shape == (len(ranges),)
         and all(ranges[j, 0]-1e-12 <= target[i] <= ranges[j, 1]+1e-12
                 for i, j in enumerate(ids) if limited[j]),
         'C35 IK joint target exceeded unchanged compiled bounds')
    target_v = np.clip((target-previous)/.01, -4., 4.)
    target_a = np.clip((target_v-previous_v)/.01, -60., 60.)
    need(close(calc['target_velocity16'], target_v)
         and close(calc['target_accel16'], target_a),
         'C35 joint derivative/clip arithmetic differs')
    geometry = compiled_side_geometry(kin, q, v, binding['wheel_map'])
    points = np.asarray(geometry['contact_points_m'])
    com = np.asarray(geometry['whole_com_m'])
    need(close(calc['contact_points4'], points)
         and close(calc['whole_com_world_m'], com),
         'C35 force geometry/COM differs from compiled actual state')
    state = scalar_state(q, v, binding)
    linear = np.asarray(state['world_com_velocity'])
    angular = np.asarray(state['world_angular_velocity'])
    need(close(calc['base_linear_velocity_world_mps'], linear)
         and close(calc['base_angular_velocity_world_rps'], angular),
         'C35 actual COM/angular velocity force input differs')
    mass = float(np.sum(np.asarray(kin['body_mass'], dtype=float)))
    pose = np.asarray(reference['body_pose_world_m'], dtype=float)
    body_v = np.array((*reference['body_vxy_mps'], 0.), dtype=float)
    body_a = np.array((*reference['body_axy_mps2'], 0.), dtype=float)
    need(pose.shape == (3,) and close(pose[:2], reference['body_xy_m'])
         and close(pose[2], entry_z_m)
         and close(reference['body_yaw_rad'], entry_yaw_rad),
         'C35 body reference changed entry Z/yaw or split XY clocks')
    target_q = q.copy()
    target_q[:3] = pose
    target_q[3:7] = (math.cos(entry_yaw_rad/2), 0., 0.,
                     math.sin(entry_yaw_rad/2))
    target_q[qa] = target
    target_fk = compiled_side_geometry(kin, target_q, np.zeros_like(v),
                                       binding['wheel_map'])
    centers = np.asarray(target_fk['wheel_body_positions_m'])
    ik_world = np.asarray(calc['ik_world_targets4'], dtype=float)
    final_ik_error = float(np.linalg.norm(centers-ik_world, axis=1).max())
    need(ik_world.shape == (4, 3) and np.isfinite(ik_world).all()
         and final_ik_error <= .008+1e-10
         and final_ik_error <= float(calc['ik_error_m'])+1e-8
         and float(calc['ik_error_m']) <= .008,
         'C35 target23 pure FK or paired IK acceptance differs')
    force = mass*(np.array((0., 0., 9.81))+body_a
                  +26.*(pose-q[:3])+9.*(body_v-linear))
    rpy = np.asarray(state['rpy'])
    yaw_error = math.atan2(math.sin(entry_yaw_rad-rpy[2]),
                           math.cos(entry_yaw_rad-rpy[2]))
    cy, sy = math.cos(entry_yaw_rad), math.sin(entry_yaw_rad)
    heading = np.array(((cy, -sy, 0.), (sy, cy, 0.), (0., 0., 1.)))
    moment = 180.*(heading@np.array((-rpy[0], -rpy[1], yaw_error)))-25.*angular
    wrench = np.r_[force, moment]
    need(close(calc['desired_wrench6'], wrench),
         'C35 requested six-dimensional wrench differs')
    phase, pair = reference['phase'], reference['pair']
    weights = np.ones(4)
    swing = tuple(pair) if phase in AIR35 else ()
    need(not swing or tuple(sorted(swing)) in ((0, 3), (1, 2)),
         'C35 force used non-diagonal swing legs')
    if swing:
        weights[list(swing)] = 0.
    elif phase == 'transfer':
        need(pair is not None and tuple(sorted(pair)) in ((0, 3), (1, 2)),
             'C35 transfer lost intended pair')
        weights[list(pair)] = 1.-smooth35(float(reference['phase_elapsed_s'])/.06)
    elif phase == 'transfer_restore':
        need(pair is not None and tuple(sorted(pair)) in ((0, 3), (1, 2))
             and previous_reference is not None,
             'C35 transfer restore lacks actual prior pair')
        start = float(reference['restore_start_weight'])
        old = np.asarray(previous_reference['force_weights'], dtype=float)
        prior_phase = previous_reference['phase']
        prior_expected = (start if prior_phase == 'transfer' else
                          start+(1.-start)*smooth35(
                              float(previous_reference['phase_elapsed_s'])/.06))
        elapsed = float(reference['phase_elapsed_s'])
        clock_ok = (close(elapsed, 0.) if prior_phase == 'transfer' else
                    close(elapsed, float(previous_reference['phase_elapsed_s'])+.01))
        need(old.shape == (4,) and close(old[list(pair)], [prior_expected]*2)
             and prior_phase in ('transfer', 'transfer_restore')
             and previous_reference['pair'] == pair
             and (prior_phase != 'transfer_restore' or
                  close(previous_reference['restore_start_weight'], start))
             and clock_ok,
             'C35 restore weight not bound to prior consumed transfer')
        weights[list(pair)] = start+(1.-start)*smooth35(elapsed/.06)
    elif phase == 'load':
        need(pair is not None and tuple(sorted(pair)) in ((0, 3), (1, 2)),
             'C35 load lost returning pair')
        weights[list(pair)] = smooth35(float(reference['phase_elapsed_s'])/.06)
    need(close(reference['force_weights'], weights),
         'C35 pair allocation weights differ from consumed phase')
    allocation = np.zeros((6, 12))
    for leg in range(4):
        x, y, z = points[leg]-com
        allocation[:3, 3*leg:3*leg+3] = np.eye(3)
        allocation[3:, 3*leg:3*leg+3] = ((0., -z, y), (z, 0., -x), (-y, x, 0.))
    penalty = np.repeat(.01/np.maximum(weights, 1e-3)**2, 3)
    try:
        allocated = np.linalg.solve(allocation.T@allocation+np.diag(penalty),
                                    allocation.T@wrench)
    except np.linalg.LinAlgError:
        allocated = np.linalg.lstsq(allocation, wrench, rcond=None)[0]
    allocated = allocated.reshape(4, 3)
    for leg in range(4):
        allocated[leg, 2] = np.clip(allocated[leg, 2], 0.,
                                    weights[leg]*.85*mass*9.81)
        allocated[leg, :2] = np.clip(allocated[leg, :2],
                                      -.60*allocated[leg, 2], .60*allocated[leg, 2])
    need(close(calc['allocation_forces4'], allocated, atol=2e-7),
         'C35 original finite allocation/caps differ')
    tau = np.asarray(calc['bias16'], dtype=float).copy()
    need(tau.shape == (16,) and np.isfinite(tau).all(),
         'C35 native bias torque cache missing')
    jacobians = calc['jacobians4']
    need(len(jacobians) == 4, 'C35 Jacobian slots missing')
    for leg in range(4):
        jac = jacobians[leg]
        if weights[leg] <= 1e-3:
            need(jac is None, 'C35 unloaded leg invented force Jacobian')
            continue
        jac = np.asarray(jac, dtype=float)
        need(jac.shape == (3, 16) and np.isfinite(jac).all(),
             'C35 actual force Jacobian cache invalid')
        tau -= jac.T@allocated[leg]
    kp, kd = np.full(16, 80.), np.full(16, 3.)
    for leg in swing:
        kp[4*leg:4*leg+3], kd[4*leg:4*leg+3] = 150., 5.
    need(close(calc['kp16'], kp) and close(calc['kd16'], kd),
         'C35 pair PD gains differ from original actuator permissions')
    velocity = v[da]
    tau += kp*(target-q[qa])+kd*(target_v-velocity)
    inertia = calc['paired_inertia6']
    if swing:
        inertia = np.asarray(inertia, dtype=float)
        need(inertia.shape == (6, 6) and np.isfinite(inertia).all()
             and close(inertia, inertia.T, atol=1e-8)
             and np.linalg.eigvalsh(inertia).min() > 0.,
             'C35 coupled paired inertia cache invalid')
        positions = np.array([4*leg+j for leg in swing for j in range(3)])
        tau[positions] += inertia@target_a[positions]
    else:
        need(inertia is None, 'C35 non-swing phase invented coupled inertia')
    wheel = np.arange(3, 16, 4)
    need('wheel_angles4' in calc, 'C35 actual wheel brake anchors missing')
    anchors = np.asarray(calc['wheel_angles4'], dtype=float)
    need(anchors.shape == (4,) and np.isfinite(anchors).all(),
         'C35 actual wheel brake anchors invalid')
    tau[wheel] += 12.*(anchors-q[qa[wheel]])-2.*velocity[wheel]
    tau[wheel] -= kp[wheel]*(target[wheel]-q[qa[wheel]])
    tau[wheel] -= kd[wheel]*(target_v[wheel]-velocity[wheel])
    need(close(calc['preclip_torque16'], tau, atol=2e-7),
         'C35 paired preclip torque differs from full arithmetic')
    limits = np.asarray(calc['torque_limits16'], dtype=float)
    need(limits.shape == (16,) and close(limits, LIMITS)
         and np.all(limits > 0.), 'C35 original actuator limits changed')
    safe = np.clip(tau, -limits, limits)
    need(close(calc['safe_torque16'], safe, atol=2e-7)
         and close(record['returned_side_torque_nm'], safe, atol=2e-7),
         'C35 returned paired torque/clip differs')
    return {'computed': {'safe_torque_nm': safe}, 'geometry': geometry,
            'state': state, 'preclip_torque_nm': tau,
            'scope': 'independent paired wrench/allocation/PD/inertia/brake/clip'}
