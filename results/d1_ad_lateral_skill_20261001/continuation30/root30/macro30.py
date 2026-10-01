"""Saved event transitions for C30; no policy, weighted reward, or engine calls."""
from __future__ import annotations

import numpy as np


def components30(qpos, applied_native_torques, torque_limits, *, origin_xy, yaw0, direction):
    q = np.asarray(qpos, dtype=float)
    tau = np.asarray(applied_native_torques, dtype=float)
    limits = np.asarray(torque_limits, dtype=float)
    n = len(q) - 1
    if (q.shape != (n+1, 23) or tau.shape != (n, 5, 16)
            or limits.shape != (16,) or np.any(limits <= 0)
            or direction not in (-1, 1) or n < 1
            or not all(np.isfinite(v).all() for v in (q, tau, limits))):
        raise ValueError('C30 macro requires complete actual endpoint/5T torque arrays')
    c, s = np.cos(yaw0), np.sin(yaw0)
    xy = q[:, :2] - np.asarray(origin_xy)
    x = xy @ np.array([c, s])
    lateral = direction * (xy @ np.array([-s, c]))
    quat = q[:, 3:7]
    w, a, b, z = quat.T
    yaw = np.arctan2(2*(w*z+a*b), 1-2*(b*b+z*z))
    yaw_error = np.arctan2(np.sin(yaw-yaw0), np.cos(yaw-yaw0))
    longitudinal = .01 * float(np.dot(x[1:], x[1:]))
    goal = .01 * float(np.sum((lateral[1:]-.03)**2))
    heading = .01 * float(np.dot(yaw_error[1:], yaw_error[1:]))
    return dict(progress=float(np.minimum(lateral[-1]/.03, 1.) - np.minimum(lateral[0]/.03, 1.)),
                backtrack=float(np.maximum(-np.diff(lateral), 0.).sum()/.03),
                elapsed_s=n*.01,
                longitudinal_square_m2_s=longitudinal,
                longitudinal_square_normalized_s=longitudinal/.03**2,
                lateral_goal_square_m2_s=goal,
                lateral_goal_square_normalized_s=goal/.03**2,
                yaw_square_rad2_s=heading,
                yaw_square_normalized_s=heading/.12**2,
                actual_torque_square_normalized_s=.01*float(np.mean((tau/limits)**2, axis=(1, 2)).sum()))


def raw_observation30(states, boundary, *, macro_index, direction):
    """Named, unnormalised state record; deliberately not B22's observation99."""
    return dict(schema='d1-c30-event-observation-raw-v1',
                control_boundary=int(boundary), macro_index=int(macro_index),
                direction=int(direction), time_s=float(states['time'][boundary]),
                qpos=np.asarray(states['qpos'][boundary]).tolist(),
                qvel=np.asarray(states['qvel'][boundary]).tolist(),
                source='actual_control_endpoint', learning_normalization_defined=False)


def build_macros30(states, rows, latches, *, torque_limits, origin_xy, yaw0, direction, end_reason):
    """A true leg latch begins a transition; 100 Hz rows are never events."""
    if not latches:
        return []
    starts = [int(item['control_index']) for item in latches]
    n = len(rows)
    if (len(starts) > 4 or not 0 <= starts[0] < n
            or any(a >= b for a, b in zip(starts, starts[1:])) or starts[-1] >= n):
        raise ValueError('C30 leg latches must be ordered within actual consumed controls')
    native = np.asarray([[t['applied_nm'] for t in row['native_actuator_traces']]
                         for row in rows], dtype=float)
    result = []
    for index, (start, finish) in enumerate(zip(starts, starts[1:]+[n])):
        latch = latches[index]
        reason = 'next_leg' if index+1 < len(starts) else end_reason
        result.append(dict(
            schema='d1-c30-event-transition-v1', macro_index=index,
            control_start=start, control_end=finish, elapsed_controls=finish-start,
            observation=raw_observation30(states, start, macro_index=index, direction=direction),
            next_observation=raw_observation30(states, finish, macro_index=index+1, direction=direction),
            proposed_action=list(latch['proposed_action']),
            projected_action=list(latch['projected_action']),
            consumed_action=list(latch['consumed_action']),
            projection_reason=latch['reason'],
            reward_components=components30(np.asarray(states['qpos'])[start:finish+1],
                native[start:finish], torque_limits, origin_xy=origin_xy, yaw0=yaw0, direction=direction),
            reward_weights_defined=False, weighted_reward=None,
            terminal_reason=reason,
            terminated=reason in ('success_after_retention', 'safe_cancel', 'physical_failure'),
            truncated=reason in ('budget_truncated', 'software_failure'),
        ))
    return result
