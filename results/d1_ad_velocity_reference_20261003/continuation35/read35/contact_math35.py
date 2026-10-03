"""Saved-contact arithmetic for C35; no simulator or controller imports.

The caller must first use the frozen compiled-geometry contact verifier on each
contact.  This module separately reconstructs the world wrench from the saved
local six-vector and reports wheel support from actual positive normal load.
"""
from __future__ import annotations

import math
import numpy as np


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise AssertionError(reason)


def vector35(value, length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    need(result.shape == (length,) and np.isfinite(result).all(), label)
    return result


def contact_wrench35(contact: dict, terrain_geoms: set[int], com_world_m) -> dict | None:
    """Rebuild one robot-terrain reaction, including the contact's local torque.

    MuJoCo's saved contact frame stores basis vectors as rows.  The local
    six-vector is a force/moment on geom2 when geom1 is terrain, and has the
    opposite sign when geom2 is terrain.  Non-terrain contacts do not enter the
    support resultant.
    """
    g1, g2 = int(contact['geom1']), int(contact['geom2'])
    t1, t2 = g1 in terrain_geoms, g2 in terrain_geoms
    need(not (t1 and t2), 'terrain-terrain contact')
    if not (t1 or t2) or int(contact['efc_address']) < 0:
        return None
    frame = np.asarray(contact['frame_world'], dtype=np.float64)
    need(frame.shape == (3, 3) and np.isfinite(frame).all()
         and np.allclose(frame @ frame.T, np.eye(3), atol=1e-10, rtol=0)
         and abs(float(np.linalg.det(frame))-1.) <= 1e-10,
         'invalid actual contact frame')
    local = vector35(contact['contact_force_local'], 6, 'missing local contact wrench')
    position = vector35(contact['position_world_m'], 3, 'missing contact position')
    com = vector35(com_world_m, 3, 'missing actual whole COM')
    sign = 1. if t1 else -1.
    force = sign * (frame.T @ local[:3])
    couple = sign * (frame.T @ local[3:])
    moment = np.cross(position-com, force) + couple
    outward = frame[0] if t1 else -frame[0]
    normal = float(force @ outward)
    wheel = contact['robot_wheel_index']
    need(wheel is None or type(wheel) is int and 0 <= wheel < 4,
         'invalid actual wheel index')
    need(np.allclose(force, vector35(contact['force_on_robot_world_n'], 3,
                                       'missing saved world force'), atol=1e-8, rtol=0)
         and math.isclose(normal, float(contact['normal_load_on_robot_n']),
                          abs_tol=1e-8, rel_tol=0),
         'saved contact force/normal differs from local six-vector')
    return {'wheel_index': wheel, 'force_world_n': force,
            'moment_about_com_world_nm': moment, 'normal_load_n': normal,
            'position_world_m': position,
            'positive_wheel_load': bool(wheel is not None and normal > 0.)}


def native_resultant35(native: dict, terrain_geoms: set[int], com_world_m) -> dict:
    """Sum every robot-terrain reaction, not only contacts on the stance pair."""
    rows = [r for c in native['contacts']
            if (r := contact_wrench35(c, terrain_geoms, com_world_m)) is not None]
    force = sum((r['force_world_n'] for r in rows), np.zeros(3))
    moment = sum((r['moment_about_com_world_nm'] for r in rows), np.zeros(3))
    normal = np.zeros(4)
    mask = [False]*4
    effective = [False]*4
    for contact in native['contacts']:
        if (int(contact['efc_address']) >= 0
                and ((int(contact['geom1']) in terrain_geoms)
                     != (int(contact['geom2']) in terrain_geoms))):
            wheel = contact['robot_wheel_index']
            if wheel is not None:
                need(type(wheel) is int and 0 <= wheel < 4,
                     'invalid active terrain wheel index')
                effective[wheel] = True
    for row in rows:
        wheel = row['wheel_index']
        if wheel is not None and row['normal_load_n'] > 0.:
            mask[wheel] = True
            normal[wheel] += row['normal_load_n']
    return {'contact_count': len(rows), 'world_force_sum_n': force,
            'world_moment_about_com_sum_nm': moment,
            'positive_wheel_support_mask': tuple(mask),
            'effective_terrain_wheel_contact_mask': tuple(effective),
            'positive_wheel_normal_load_sum_n': normal,
            'per_contact': rows}


def pair_support35(result: dict, stance_pair: tuple[int, int], mass_kg: float,
                   gravity_mps2: float = 9.81) -> dict:
    """Per-wheel positive support plus a *pair-total* normal-load threshold."""
    need(tuple(sorted(stance_pair)) in ((0, 3), (1, 2)),
         'unqualified diagonal stance pair')
    need(math.isfinite(mass_kg) and mass_kg > 0.
         and math.isfinite(gravity_mps2) and gravity_mps2 > 0.,
         'invalid mass/gravity binding')
    loads = vector35(result['positive_wheel_normal_load_sum_n'], 4,
                     'missing actual wheel normal loads')
    mask = tuple(bool(x) for x in result['positive_wheel_support_mask'])
    need(len(mask) == 4, 'missing actual support mask')
    total = float(sum(loads[i] for i in stance_pair))
    passed = bool(all(mask[i] and loads[i] > 1e-8 for i in stance_pair)
                  and total >= .5*mass_kg*gravity_mps2)
    return {'stance_pair': stance_pair, 'each_stance_wheel_positive':
            bool(all(mask[i] and loads[i] > 1e-8 for i in stance_pair)),
            'stance_pair_normal_sum_n': total,
            'required_pair_normal_sum_n': .5*mass_kg*gravity_mps2,
            'passed': passed}


def all_four_support35(result: dict, mass_kg: float,
                       gravity_mps2: float = 9.81) -> dict:
    """Four real loaded wheels at transfer/load/settle, with total ≥0.5mg."""
    need(math.isfinite(mass_kg) and mass_kg > 0.
         and math.isfinite(gravity_mps2) and gravity_mps2 > 0.,
         'invalid mass/gravity binding')
    loads = vector35(result['positive_wheel_normal_load_sum_n'], 4,
                     'missing actual four-wheel loads')
    mask = tuple(bool(x) for x in result['positive_wheel_support_mask'])
    need(len(mask) == 4, 'missing actual support mask')
    total = float(np.sum(loads))
    each = bool(all(mask[i] and loads[i] > 1e-8 for i in range(4)))
    return {'each_wheel_positive': each, 'all_wheel_normal_sum_n': total,
            'required_all_wheel_normal_sum_n': .5*mass_kg*gravity_mps2,
            'passed': bool(each and total >= .5*mass_kg*gravity_mps2)}


def native_phase_support35(result: dict, phase: str, swing_pair: tuple[int, int] | None,
                           mass_kg: float, gravity_mps2: float = 9.81) -> dict:
    """Select the frozen gate from the actual consumed phase, not a gate flag."""
    if phase in ('idle', 'transfer', 'transfer_restore', 'load', 'settle', 'done'):
        proof = all_four_support35(result, mass_kg, gravity_mps2)
        return {'phase': phase, 'gate': 'all_four', **proof}
    if phase in ('lift', 'horizontal', 'lower', 'probe', 'dwell'):
        need(swing_pair is not None
             and tuple(sorted(swing_pair)) in ((0, 3), (1, 2)),
             'air/probe phase lacks named opposite swing pair')
        stance_pair = tuple(i for i in range(4) if i not in swing_pair)
        proof = pair_support35(result, stance_pair, mass_kg, gravity_mps2)
        return {'phase': phase, 'gate': 'stance_pair', **proof}
    raise AssertionError('unqualified C35 consumed phase')


def static_contact_margin35(result: dict, com_world_m) -> dict:
    """Four weighted actual contact representatives, convex hull and real COM.

    Only true active terrain contact rows with positive normal force enter a
    wheel representative.  This is a stop/handoff margin, not the old reference
    acceleration `dynamic_margin` proxy and not a two-foot air stability claim.
    """
    com = vector35(com_world_m, 3, 'missing actual whole COM for stop margin')
    representatives = []
    for wheel in range(4):
        contacts = [row for row in result['per_contact']
                    if row['wheel_index'] == wheel and row['normal_load_n'] > 0.]
        loads = np.array([row['normal_load_n'] for row in contacts], dtype=float)
        need(bool(contacts) and float(np.sum(loads)) > 1e-8,
             'stop margin lacks all four loaded actual wheels')
        xy = np.array([row['position_world_m'][:2] for row in contacts])
        representatives.append(np.sum(xy*loads[:, None], axis=0)/np.sum(loads))
    points = np.asarray(representatives)
    sorted_points = sorted(map(tuple, points.tolist()))

    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])

    lower: list[tuple[float, float]] = []
    for point in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(sorted_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1]+upper[:-1], dtype=float)
    need(hull.ndim == 2 and hull.shape[1] == 2 and len(hull) >= 3,
         'actual four-contact support hull degenerate')
    margins = []
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        edge = b-a
        length = float(np.linalg.norm(edge))
        need(length > 1e-8, 'actual support hull edge degenerate')
        relative = com[:2]-a
        margins.append(float((edge[0]*relative[1]-edge[1]*relative[0])/length))
    return {'wheel_contact_representative_xy_m': points,
            'hull_xy_m': hull, 'whole_com_xy_m': com[:2],
            'static_margin_m': min(margins),
            'stop_margin_ge_005': bool(min(margins) >= .005)}


def horizontal_admission35(*, pair: tuple[int, int], pre_shape_min_z_m,
                           previous_native_indices, previous_effective_masks,
                           first_horizontal_control_index: int,
                           stance_support: dict) -> dict:
    """Only measured current geometry and the immediately previous 5T interval."""
    need(tuple(sorted(pair)) in ((0, 3), (1, 2)), 'invalid intended swing pair')
    need(type(first_horizontal_control_index) is int
         and first_horizontal_control_index >= 1,
         'horizontal has no complete preceding native interval')
    pre = vector35(pre_shape_min_z_m, 4, 'missing measured pre-state wheel shape')
    indices = list(previous_native_indices)
    expected = list(range(5*(first_horizontal_control_index-1),
                          5*first_horizontal_control_index))
    need(indices == expected and len(previous_effective_masks) == 5
         and all(len(row) == 4 for row in previous_effective_masks),
         'horizontal admission used stale/future/incomplete native evidence')
    clear = bool(all(pre[leg] > .012 for leg in pair)
                 and all(not row[leg] for row in previous_effective_masks
                         for leg in pair))
    return {'previous_native_indices': indices,
            'measured_pre_shape_min_z_m': pre,
            'prior_five_swing_contact_free': bool(all(
                not row[leg] for row in previous_effective_masks for leg in pair)),
            'current_stance_support_passed': stance_support['passed'] is True,
            'passed': bool(clear and stance_support['passed'] is True)}


def horizontal_native35(*, pair: tuple[int, int], control_index: int,
                        current_native_indices, current_shape_min_z_m,
                        current_effective_masks) -> dict:
    """After-step evidence for every native substep with horizontal motion."""
    need(tuple(sorted(pair)) in ((0, 3), (1, 2))
         and type(control_index) is int and control_index >= 0,
         'invalid horizontal pair/control')
    indices = list(current_native_indices)
    minima = np.asarray(current_shape_min_z_m, dtype=np.float64)
    masks = np.asarray(current_effective_masks, dtype=bool)
    need(indices == list(range(5*control_index, 5*(control_index+1)))
         and minima.shape == masks.shape == (5, 4)
         and np.isfinite(minima).all(),
         'horizontal native interval/masks/shape differ')
    above = bool(np.all(minima[:, pair] > .012))
    free = bool(not np.any(masks[:, pair]))
    return {'native_indices': indices, 'all_shape_above_12mm': above,
            'all_swing_active_terrain_contact_free': free,
            'passed': bool(above and free),
            'between_native_samples_certified': False}
