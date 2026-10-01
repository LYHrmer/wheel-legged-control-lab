"""Fixed 54-D event observation from pre-integration state and compiled indices."""
from __future__ import annotations

import math
import numpy as np


OBS_SCHEMA31 = 'd1-c31-lateral-observation54-v1'
CONTINUOUS = np.ones(54, dtype=bool)
CONTINUOUS[:5] = False  # direction and leg one-hot are categorical
VELOCITY_SCALES = np.tile([4., 4., 4., 20.], 4)


def compiled_joint_map31(plant, geometry_binding: dict) -> dict:
    """Bind actuator order to actual joint IDs, addresses, leg ancestry, and ranges."""
    model = plant.model
    ids = np.asarray(geometry_binding['joint_ids'], dtype=int)
    qadr = np.asarray(geometry_binding['joint_qpos_addresses'], dtype=int)
    dadr = np.asarray(geometry_binding['joint_dof_addresses'], dtype=int)
    actuators = np.asarray(geometry_binding['actuator_ids'], dtype=int)
    trnid = np.asarray(geometry_binding['actuator_trnid'], dtype=int)
    trntype = np.asarray(geometry_binding['actuator_trntype'], dtype=int)
    if (ids.shape != (16,) or qadr.shape != (16,) or dadr.shape != (16,)
            or actuators.shape != (16,) or len(set(ids.tolist())) != 16
            or len(set(qadr.tolist())) != 16 or len(set(dadr.tolist())) != 16
            or np.min(ids) < 0 or np.max(ids) >= model.njnt
            or np.min(actuators) < 0 or np.max(actuators) >= model.nu
            or not np.array_equal(ids, trnid[actuators, 0])
            or not np.array_equal(ids, np.asarray(model.actuator_trnid)[actuators, 0])
            or not np.array_equal(trntype[actuators], np.zeros(16, dtype=int))
            or not np.array_equal(qadr, np.asarray(model.jnt_qposadr)[ids])
            or not np.array_equal(dadr, np.asarray(model.jnt_dofadr)[ids])
            or not np.array_equal(qadr, np.asarray(plant.qpos_addresses))
            or not np.array_equal(dadr, np.asarray(plant.dof_addresses))
            or not np.array_equal(ids, np.asarray(plant.joint_ids))
            or not np.array_equal(actuators, np.asarray(plant.actuator_ids))):
        raise ValueError('C31 actuator/joint/address identity differs from compiled robot')
    free = geometry_binding['kinematics24']
    if (int(free['jnt_type'][0]) != 0 or int(free['jnt_qposadr'][0]) != 0
            or int(free['jnt_dofadr'][0]) != 0 or qadr.min() < 7 or dadr.min() < 6):
        raise ValueError('C31 first joint is not the actual free base')
    body_owner = {}
    for body, (begin, count) in enumerate(zip(model.body_jntadr, model.body_jntnum)):
        for joint in range(int(begin), int(begin + count)):
            body_owner[joint] = body
    wheel_bodies = [int(x) for x in plant.wheel_body_ids_by_leg]
    if len(wheel_bodies) != 4 or len(set(wheel_bodies)) != 4:
        raise ValueError('C31 needs four distinct compiled wheel bodies')
    for leg, wheel in enumerate(wheel_bodies):
        lineage = set()
        body = wheel
        while body != 0:
            if body in lineage:
                raise ValueError('compiled body tree has a cycle')
            lineage.add(body)
            body = int(model.body_parentid[body])
        if any(body_owner.get(int(joint)) not in lineage for joint in ids[4*leg:4*leg+4]):
            raise ValueError('C31 actuator group is not the compiled wheel leg')
    ranges = np.asarray(model.jnt_range, dtype=np.float64)[ids]
    limited = np.asarray(model.jnt_limited, dtype=bool)[ids]
    leg = np.arange(16).reshape(4, 4)[:, :3].reshape(12)
    lo, hi = ranges[leg].T
    if (not limited[leg].all() or not np.isfinite(ranges).all()
            or np.any(hi <= lo) or np.any(qadr >= model.nq)
            or np.any(dadr >= model.nv)):
        raise ValueError('C31 actual leg joint limits/addresses are invalid')
    return {
        'joint_ids': ids.tolist(), 'joint_qpos_addresses': qadr.tolist(),
        'joint_dof_addresses': dadr.tolist(), 'actuator_ids': actuators.tolist(),
        'actuator_joint_ids': trnid[actuators, 0].tolist(),
        'jnt_range': ranges.tolist(), 'jnt_limited': limited.tolist(),
        'wheel_body_ids_by_leg': wheel_bodies,
        'leg_qpos_addresses': qadr[leg].tolist(),
        'leg_range_mid': ((lo + hi)/2).tolist(),
        'leg_range_half': ((hi - lo)/2).tolist(),
    }


def _rpy(quaternion: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(float(np.clip(2*(w*y-z*x), -1., 1.)))
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return roll, pitch, yaw


def observe31(*, qpos, qvel, initial_xy_m, initial_yaw_rad, direction,
              leg_index, macro_index, teacher_body_from_m, teacher_body_to_m,
              teacher_shift_time_s, plan_com_height_m, previous_consumed_action,
              completed_side_controls, joint_map: dict) -> dict:
    """No model query; all inputs are actual pre-state or original teacher plan."""
    q = np.asarray(qpos, dtype=np.float64)
    v = np.asarray(qvel, dtype=np.float64)
    start = np.asarray(initial_xy_m, dtype=np.float64)
    body_from = np.asarray(teacher_body_from_m, dtype=np.float64)
    body_to = np.asarray(teacher_body_to_m, dtype=np.float64)
    previous = np.asarray(previous_consumed_action, dtype=np.float64)
    if (q.shape != (23,) or v.shape != (22,) or start.shape != (2,)
            or body_from.shape != (3,) or body_to.shape != (3,)
            or previous.shape != (3,) or direction not in (-1, 1)
            or type(leg_index) is not int or leg_index not in range(4)
            or type(macro_index) is not int or macro_index not in range(4)
            or type(completed_side_controls) is not int or completed_side_controls < 0
            or not all(np.isfinite(a).all() for a in (q, v, start, body_from, body_to, previous))
            or not all(math.isfinite(float(x)) for x in (
                initial_yaw_rad, teacher_shift_time_s, plan_com_height_m))
            or abs(float(np.linalg.norm(q[3:7])) - 1.) > 1e-3):
        raise ValueError('C31 event observation has invalid actual inputs')
    cy, sy = math.cos(initial_yaw_rad), math.sin(initial_yaw_rad)
    rotation = np.array([[cy, sy], [-sy, cy]])
    xy = rotation @ (q[:2] - start)
    velocity_xy = rotation @ v[:2]
    teacher_xy = rotation @ (body_to[:2] - body_from[:2])
    roll, pitch, yaw = _rpy(q[3:7])
    yaw_error = math.atan2(math.sin(yaw-initial_yaw_rad),
                           math.cos(yaw-initial_yaw_rad))
    qadr = np.asarray(joint_map['leg_qpos_addresses'], dtype=int)
    dadr = np.asarray(joint_map['joint_dof_addresses'], dtype=int)
    middle = np.asarray(joint_map['leg_range_mid'], dtype=float)
    half = np.asarray(joint_map['leg_range_half'], dtype=float)
    if (qadr.shape != (12,) or dadr.shape != (16,) or middle.shape != (12,)
            or half.shape != (12,) or np.any(half <= 0)
            or np.any(qadr < 0) or np.any(qadr >= 23)
            or np.any(dadr < 0) or np.any(dadr >= 22)):
        raise ValueError('C31 compiled joint map differs from 54-D contract')
    raw = np.zeros(54, dtype=np.float64)
    raw[0] = direction
    raw[1 + leg_index] = 1.
    raw[5] = macro_index
    raw[6:8] = xy
    raw[8] = q[2] - .455
    raw[9:12] = (roll, pitch, yaw_error)
    raw[12:15] = (velocity_xy[0], velocity_xy[1], v[2])
    raw[15:18] = v[3:6]
    raw[18:30] = q[qadr]
    raw[30:46] = v[dadr]
    raw[46:48] = teacher_xy
    raw[48] = teacher_shift_time_s
    raw[49] = plan_com_height_m
    raw[50:53] = previous
    raw[53] = completed_side_controls * .01
    normalized = raw.copy()
    normalized[5] /= 3.
    normalized[6:8] /= .12
    normalized[8] /= .05
    normalized[9:12] /= .12
    normalized[12:15] /= [.2, .2, .1]
    normalized[15:18] /= .5
    normalized[18:30] = (raw[18:30]-middle)/half
    normalized[30:46] /= VELOCITY_SCALES
    normalized[46:48] /= .12
    normalized[48] /= 1.5
    normalized[49] /= .5
    normalized[50:53] /= [.012, .012, 1.]
    normalized[53] /= 15.
    if not np.isfinite(normalized).all():
        raise ValueError('C31 normalized observation is nonfinite')
    clipped = CONTINUOUS & (np.abs(normalized) > 5.)
    normalized[CONTINUOUS] = np.clip(normalized[CONTINUOUS], -5., 5.)
    return {
        'schema': OBS_SCHEMA31,
        'pre_qpos23': q.tolist(), 'pre_qvel22': v.tolist(),
        'initial_xy_m': start.tolist(), 'initial_yaw_rad': float(initial_yaw_rad),
        'direction': int(direction), 'leg_index': leg_index,
        'macro_index': macro_index,
        'teacher_body_from_m': body_from.tolist(),
        'teacher_body_to_m': body_to.tolist(),
        'teacher_shift_time_s': float(teacher_shift_time_s),
        'plan_com_height_m': float(plan_com_height_m),
        'previous_consumed_action': previous.tolist(),
        'completed_side_controls': completed_side_controls,
        'side_elapsed_s': float(raw[53]),
        'raw54': raw.tolist(), 'normalized54': normalized.tolist(),
        'clip_mask54': clipped.tolist(),
        'compiled_joint_map31': joint_map,
    }
