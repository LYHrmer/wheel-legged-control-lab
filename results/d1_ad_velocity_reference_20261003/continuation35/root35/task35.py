"""Online continuous-task classification; cold C35 reader recomputes it."""
from __future__ import annotations

import math
import numpy as np

from task31 import GATES31, _state31, _geometry31, _tail31, _native_loads31


def actual_margin35(evidence):
    """Inward distance to convex hull of four load-weighted real contacts."""
    points = evidence['positive_load_weighted_wheel_contact_world_m']
    if any(p is None for p in points):
        return -1.
    points = sorted(set(tuple(p[:2]) for p in points))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1]+upper[:-1]
    if len(hull) < 3:
        return -1.
    com = evidence['whole_com_world_m'][:2]
    return min(cross(a, b, com)/math.hypot(b[0]-a[0], b[1]-a[1])
               for a, b in zip(hull, hull[1:]+hull[:1]))


def stopped_post35(native, binding):
    e = native['contact_evidence35']
    state = _state31(native['after']['qpos'], native['after']['qvel'], binding)
    return bool(e['consumed_phase'] in ('settle', 'done') and not e['swing_legs']
        and all(e['active_wheel_terrain_contact'])
        and min(e['positive_wheel_normal_load_sum_n']) > 1e-8
        and actual_margin35(e) >= .005
        and np.linalg.norm(state['origin_velocity']) <= .010
        and np.linalg.norm(state['world_angular_velocity']) <= .05)


def sensed35(binding, qpos, qvel, last_five, control_index):
    from core35 import Sensed35
    from kinematics24 import reconstruct24, rotation
    if len(last_five) != 5:
        raise RuntimeError('C35 sensing needs five actual previous native returns')
    wheel_map = {int(k): int(v) for k, v in binding['wheel_index_by_body_id'].items()}
    geometry = _geometry31(binding['kinematics24'], qpos, qvel, wheel_map)
    kin = binding['kinematics24']
    fk = reconstruct24(kin, qpos, qvel)
    centers = np.zeros((4, 3))
    for body, leg in wheel_map.items():
        gid = next(i for i, b in enumerate(kin['geom_bodyid']) if int(b) == body)
        body_rotation = fk['geom_xmat'][gid]@rotation(kin['geom_quat'][gid]).T
        centers[leg] = fk['geom_xpos'][gid]-body_rotation@np.asarray(kin['geom_pos'][gid])
    state = _state31(qpos, qvel, binding)
    evidence = [r['contact_evidence35'] for r in last_five]
    latest = evidence[-1]
    mass = binding['nominal_total_mass_kg']
    pairs = ((0, 3), (1, 2))
    loaded = tuple(all(all(e['active_wheel_terrain_contact'][j]
        and e['positive_wheel_normal_load_sum_n'][j] > 1e-8 for j in range(4) if j not in pair)
        and sum(e['positive_wheel_normal_load_sum_n'][j] for j in range(4) if j not in pair)
            >= .5*mass*9.81 for e in evidence) for pair in pairs)
    rpy = state['rpy']
    # COM velocity uses the exact saved base inertia offset, in world axes.
    world_com_velocity = rotation(np.asarray(qpos)[3:7])@state['body_com_velocity']
    return Sensed35(control_index=int(control_index), body_xy_m=tuple(qpos[:2]),
        body_com_vxy_mps=tuple(world_com_velocity[:2]), yaw_rad=float(rpy[2]),
        body_origin_vxyz_mps=tuple(state['origin_velocity']),
        foot_world_m=tuple(tuple(p) for p in centers),
        wheel_contact_points_world_m=tuple(tuple(p) for p in geometry['contact_points_m']),
        actual_whole_wheel_min_z_m=tuple(geometry['wheel_shape_min_z_m']),
        previous_five_contact_free=tuple(all(not e['active_wheel_terrain_contact'][j]
            for e in evidence) for j in range(4)), previous_five_pair_loaded=loaded,
        foot_contact=tuple(latest['active_wheel_terrain_contact']),
        normal_load_n=tuple(latest['positive_wheel_normal_load_sum_n']), mass_kg=float(mass),
        whole_com_z_m=float(geometry['whole_com_m'][2]), base_z_m=float(qpos[2]),
        static_support_margin_m=actual_margin35(latest),
        world_angular_speed_rps=float(np.linalg.norm(state['world_angular_velocity'])),
        native_safety_ok=True)


def entry35(sensed, qpos, qvel, binding):
    state = _state31(qpos, qvel, binding)
    checks = dict(exact_200_preparation=sensed.control_index == 200,
        roll_pitch=bool(np.max(np.abs(state['rpy'][:2])) <= .12),
        origin_speed=bool(np.linalg.norm(state['origin_velocity']) < .04),
        angular_speed=sensed.world_angular_speed_rps < .08,
        foot_plane=bool(np.max(np.abs(np.asarray(sensed.wheel_contact_points_world_m)[:, 2])) <= .01),
        four_actual_positive_contacts=all(sensed.foot_contact)
            and min(sensed.normal_load_n) > 1e-8)
    return dict(checks=checks, passed=all(checks.values()))


def score35(rows, states, native, *, binding, geometry, torque_limits, definition,
            side_end, stop_index, pair_events, baseline_integral, entry):
    n = len(rows)
    if (side_end is None or n-side_end != 400 or stop_index is None
            or len(native) != 5*n or states['qpos'].shape != (n+1, 23)):
        raise RuntimeError('C35 task requires complete side and retention arrays')
    q, v = states['qpos'], states['qvel']
    initial = _state31(q[200], v[200], binding)
    yaw = initial['rpy'][2]
    left = np.array([-math.sin(yaw), math.cos(yaw)])
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    displacement = (q[:, :2]-q[200, :2])@left
    direction = definition['direction']
    command_goal = np.asarray(rows[side_end-1]['info']['controller_record']
                              ['controller_record35']['reference']['body_xy_m'])
    final_xy = float(np.linalg.norm(q[side_end, :2]-command_goal))
    final_yaw = _state31(q[side_end], v[side_end], binding)['rpy'][2]
    yaw_error = abs(math.atan2(math.sin(final_yaw-yaw), math.cos(final_yaw-yaw)))
    # Every actual native actuator force, in the original 16 actuator order.
    torques = np.asarray([r['after']['actuator_force'] for r in native])
    norm2 = np.mean((torques/np.asarray(torque_limits))**2, axis=1)
    integral = float(norm2.sum()*.002)
    mean = float(norm2.mean())
    loads, _ = _native_loads31(native, 1e-8)
    tail = _tail31(rows, states, loads, n-100, n, binding, geometry, GATES31)
    side_q = q[200:side_end+1]
    pair_count = len(pair_events)
    signed_handoff = float(direction*displacement[side_end])
    signed_retained = float(direction*displacement[-1])
    rollback = (float(np.max(direction*(displacement[side_end]-displacement[side_end:])))
                if direction else float(np.max(np.linalg.norm(q[side_end:, :2]-q[side_end, :2], axis=1))))
    stop_distance = float(np.max(np.abs(displacement[stop_index:side_end+1]-displacement[stop_index])))
    metrics = dict(full_tau2_mean=mean, full_tau2_integral_s=integral,
        full_elapsed_s=n*.01, final_xy_error_m=final_xy, final_yaw_error_rad=yaw_error,
        pair_exchanges=pair_count, stop_to_handoff_controls=side_end-stop_index,
        max_distance_after_stop_m=stop_distance, max_rollback_m=rollback,
        signed_handoff_displacement_m=signed_handoff, signed_retained_displacement_m=signed_retained,
        commanded_distance_cost_denominator_m=None, tau2_integral_per_commanded_meter=None,
        distance_efficiency_gate=None, tail=tail, entry=entry)
    checks = dict(entry=entry['passed'], original_tail=tail['passed'],
        mean_effort=mean <= .050, final_yaw=yaw_error < .12,
        stop_time=side_end-stop_index <= 250, stop_distance=stop_distance <= .03,
        rollback=rollback <= .005)
    if definition['kind'] != 'cancel':
        checks.update(final_xy=final_xy < .012,
                      pair_exchanges=pair_count >= (2 if direction == 0 else 4))
    if definition['kind'] == 'continuous':
        signed_velocity = direction*(np.diff(side_q[:, :2], axis=0)@left)/.01
        steady = signed_velocity[100:500]
        if len(steady) != 400:
            raise RuntimeError('C35 incomplete steady window')
        actual_mean = float(steady.mean())
        rms = float(np.sqrt(np.mean((steady-.025)**2)))
        fraction = float(np.mean(steady >= .010))
        longitudinal = float(np.max(np.abs((side_q[:601, :2]-side_q[0, :2])@forward)))
        denominator = abs(float((command_goal-q[200, :2])@left))
        ratio = signed_retained/signed_handoff if signed_handoff > 0 else None
        efficiency = integral/denominator if denominator > 0 else None
        efficiency_gate = efficiency is not None and efficiency <= baseline_integral/.03
        metrics.update(mean_signed_origin_vy_mps=actual_mean, rms_steady_tracking_error_mps=rms,
            fraction_steady_signed_vy_ge_010=fraction, max_abs_longitudinal_excursion_m=longitudinal,
            retained_fraction=ratio, commanded_distance_cost_denominator_m=denominator,
            tau2_integral_per_commanded_meter=efficiency, distance_efficiency_gate=efficiency_gate,
            baseline_tau2_integral_per_commanded_meter=baseline_integral/.03)
        checks.update(actual_speed=actual_mean >= .020, tracking=rms <= .010,
            sustained=fraction >= .8, longitudinal=longitudinal <= .03,
            retention=ratio is not None and ratio >= .9, distance_efficiency=efficiency_gate,
            command_integral=abs(denominator-.150) <= 1e-10)
    metrics.update(checks=checks, eligible=all(checks.values()),
                   independent_readback_pending=True, RL_speed_benefit_proven=False)
    return metrics
