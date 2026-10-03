"""Independent C35 landing and two-foot horizontal reference arithmetic."""
from __future__ import annotations

import math
import numpy as np


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise AssertionError(reason)


def vec2(value, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    need(result.shape == (2,) and np.isfinite(result).all(), label)
    return result


def landing35(body_xy, nominal_foot_offset_xy, vref_xy, actual_com_vxy,
              liftoff_anchor_xy) -> dict:
    """Frozen raw proposal and radial 60 mm projection from the same anchor."""
    body = vec2(body_xy, 'missing body XY reference')
    offset = vec2(nominal_foot_offset_xy, 'missing entry nominal foot offset')
    vref = vec2(vref_xy, 'missing body reference velocity')
    vcom = vec2(actual_com_vxy, 'missing actual COM velocity')
    anchor = vec2(liftoff_anchor_xy, 'missing liftoff anchor')
    raw = body+offset+.975*vref+.08*(vcom-vref)
    delta = raw-anchor
    norm = float(np.linalg.norm(delta))
    saturated = norm > .060
    projected = anchor+delta*(.060/norm) if saturated else raw.copy()
    return {'raw_target_xy_m': raw, 'projected_target_xy_m': projected,
            'raw_distance_from_anchor_m': norm,
            'consumed_distance_from_anchor_m': float(np.linalg.norm(projected-anchor)),
            'saturated': bool(saturated)}


def horizontal35(elapsed_s: float, start_xy, target_xy) -> dict:
    """Quintic XY trajectory with actual projected span and matching clock."""
    start = vec2(start_xy, 'missing horizontal start')
    target = vec2(target_xy, 'missing projected target')
    need(math.isfinite(elapsed_s) and elapsed_s >= 0., 'invalid horizontal elapsed')
    span = target-start
    duration = max(.08, 1.875*float(np.linalg.norm(span))/.16)
    u = min(1., elapsed_s/duration)
    q = u**3*(10.-15.*u+6.*u*u)
    dq = 30.*u*u*(1.-u)*(1.-u)
    ddq = 60.*u*(1.-u)*(1.-2.*u)
    if elapsed_s >= duration:
        dq = ddq = 0.
    return {'duration_s': duration, 'position_xy_m': start+span*q,
            'velocity_xy_mps': span*(dq/duration),
            'acceleration_xy_mps2': span*(ddq/duration**2),
            'position_fraction': q}


def two_swing35(pair, elapsed_s: float, starts_xy, targets_xy) -> dict:
    need(tuple(sorted(pair)) in ((0, 3), (1, 2)), 'unqualified swing pair')
    need(len(starts_xy) == len(targets_xy) == 4, 'missing four-foot reference')
    trajectories = {leg: horizontal35(elapsed_s, starts_xy[leg], targets_xy[leg])
                    for leg in pair}
    duration = max(row['duration_s'] for row in trajectories.values())
    # A simultaneous pair must share one clock; slowest projected span sets T.
    shared = {}
    for leg in pair:
        start = vec2(starts_xy[leg], 'missing paired start')
        target = vec2(targets_xy[leg], 'missing paired target')
        u = min(1., elapsed_s/duration)
        q = u**3*(10.-15.*u+6.*u*u)
        dq = 0. if elapsed_s >= duration else 30.*u*u*(1.-u)*(1.-u)/duration
        ddq = 0. if elapsed_s >= duration else 60.*u*(1.-u)*(1.-2.*u)/duration**2
        shared[leg] = {'position_xy_m': start+(target-start)*q,
                       'velocity_xy_mps': (target-start)*dq,
                       'acceleration_xy_mps2': (target-start)*ddq}
    return {'duration_s': duration, 'swing_by_leg': shared}


def verify_horizontal_tick35(reference: dict, previous_reference: dict | None,
                             entry_nominal_foot_offset_xy,
                             actual_com_vxy_mps) -> dict:
    """Rebuild each saved pair XY position/velocity from one actual shared clock."""
    need(reference['phase'] == 'horizontal', 'not a horizontal consumed reference')
    pair = tuple(reference['pair'])
    need(tuple(sorted(pair)) in ((0, 3), (1, 2)),
         'horizontal reference lacks diagonal pair')
    origin = np.asarray(reference['horizontal_origin_world_m'], dtype=float)
    target = np.asarray(reference['horizontal_target_world_m'], dtype=float)
    span = np.asarray(reference['horizontal_span_xy_m'], dtype=float)
    feet = np.asarray(reference['feet_world_m'], dtype=float)
    velocity = np.asarray(reference['feet_velocity_world_mps'], dtype=float)
    need(origin.shape == target.shape == feet.shape == velocity.shape == (4, 3)
         and span.shape == (4, 2) and np.isfinite(origin).all()
         and np.isfinite(target).all() and np.isfinite(span).all()
         and np.isfinite(feet).all() and np.isfinite(velocity).all()
         and np.allclose(span, target[:, :2]-origin[:, :2], atol=1e-10, rtol=0),
         'horizontal origin/target/span evidence differs')
    elapsed = float(reference['phase_elapsed_s'])
    expected = two_swing35(pair, elapsed, origin[:, :2], target[:, :2])
    T = expected['duration_s']
    need(math.isclose(reference['horizontal_common_duration_s'], T, abs_tol=1e-10)
         and all(np.allclose(feet[leg, :2], expected['swing_by_leg'][leg]['position_xy_m'],
                             atol=1e-10, rtol=0)
                 and np.allclose(velocity[leg, :2],
                                 expected['swing_by_leg'][leg]['velocity_xy_mps'],
                                 atol=1e-10, rtol=0)
                 and math.isclose(feet[leg, 2], origin[leg, 2]+.035, abs_tol=1e-10)
                 and math.isclose(velocity[leg, 2], 0., abs_tol=1e-10)
                 for leg in pair),
         'horizontal paired XY/Z curve or derivative differs')
    if previous_reference is not None and previous_reference['phase'] == 'horizontal':
        need(previous_reference['pair'] == reference['pair']
             and np.allclose(previous_reference['horizontal_origin_world_m'], origin,
                             atol=1e-10, rtol=0)
             and np.allclose(previous_reference['horizontal_target_world_m'], target,
                             atol=1e-10, rtol=0)
             and math.isclose(elapsed,
                              float(previous_reference['phase_elapsed_s'])+.01,
                              abs_tol=1e-10),
             'horizontal clock/landing proposal changed mid-swing')
    elif previous_reference is not None and previous_reference['phase'] == 'lift':
        need(math.isclose(elapsed, 0., abs_tol=1e-10),
             'first horizontal reference did not start at curve origin')
        offsets = np.asarray(entry_nominal_foot_offset_xy, dtype=float)
        need(offsets.shape == (4, 2) and np.isfinite(offsets).all(),
             'missing entry nominal foot offsets')
        raw = reference['landing_raw_xy_m']
        projected = reference['landing_projected_xy_m']
        for leg in pair:
            proposal = landing35(reference['body_xy_m'], offsets[leg],
                                 reference['body_vxy_mps'], actual_com_vxy_mps,
                                 origin[leg, :2])
            need(np.allclose(raw[leg], proposal['raw_target_xy_m'], atol=1e-10, rtol=0)
                 and np.allclose(projected[leg], proposal['projected_target_xy_m'],
                                 atol=1e-10, rtol=0)
                 and np.allclose(target[leg, :2], proposal['projected_target_xy_m'],
                                 atol=1e-10, rtol=0)
                 and reference['foot_target_clipped'][leg] is proposal['saturated'],
                 'first horizontal landing proposal/projection differs')
    else:
        need(previous_reference is None or previous_reference['phase'] == 'lift',
             'horizontal phase began without measured lift predecessor')
    current_nonzero = bool(any(np.linalg.norm(velocity[leg, :2]) > 1e-12
                               for leg in pair))
    need(reference['horizontal_reference_velocity_nonzero'] is current_nonzero,
         'horizontal current velocity flag differs from verified quintic')
    return {'shared_duration_s': T, 'pair': pair, 'current_velocity_nonzero': current_nonzero}


def verify_horizontal_to_lower35(previous_reference: dict, current_reference: dict) -> None:
    """Lower starts only after the same paired XY curve reaches its endpoint."""
    need(previous_reference['phase'] == 'horizontal'
         and current_reference['phase'] == 'lower'
         and previous_reference['pair'] == current_reference['pair'],
         'invalid horizontal-to-lower pair transition')
    T = float(previous_reference['horizontal_common_duration_s'])
    need(float(previous_reference['phase_elapsed_s'])+.01 >= T-1e-10
         and math.isclose(float(current_reference['phase_elapsed_s']), 0., abs_tol=1e-10),
         'pair lowered before horizontal zero-derivative endpoint')
    pair = tuple(current_reference['pair'])
    target = np.asarray(previous_reference['horizontal_target_world_m'], dtype=float)
    feet = np.asarray(current_reference['feet_world_m'], dtype=float)
    velocity = np.asarray(current_reference['feet_velocity_world_mps'], dtype=float)
    need(all(np.allclose(feet[leg, :2], target[leg, :2], atol=1e-10, rtol=0)
             and np.allclose(velocity[leg, :2], 0., atol=1e-10, rtol=0)
             for leg in pair), 'lower XY did not hold completed horizontal target')


def verify_foot_reference35(reference: dict, previous: dict | None,
                            anchors_world_m, sensed: dict,
                            touchdown_world_m) -> np.ndarray | None:
    """Rebuild consumed vertical swing curves and unchanged stance anchors."""
    anchors = np.asarray(anchors_world_m, dtype=float)
    feet = np.asarray(reference['feet_world_m'], dtype=float)
    velocity = np.asarray(reference['feet_velocity_world_mps'], dtype=float)
    need(anchors.shape == feet.shape == velocity.shape == (4, 3)
         and np.isfinite(anchors).all() and np.isfinite(feet).all()
         and np.isfinite(velocity).all(), 'missing finite C35 four-foot reference')
    phase, pair = reference['phase'], reference['pair']
    named = tuple(pair) if pair is not None else ()
    air = phase in ('lift', 'horizontal', 'lower', 'probe', 'dwell')
    stance = tuple(i for i in range(4) if i not in named) if air or phase == 'load' else tuple(range(4))
    need(all(np.allclose(feet[i], anchors[i], atol=1e-10, rtol=0)
             and np.allclose(velocity[i], 0., atol=1e-10, rtol=0)
             for i in stance), 'C35 non-swing foot anchor/velocity changed')
    if phase not in ('lift', 'horizontal', 'lower', 'probe', 'dwell', 'load'):
        return None
    need(tuple(sorted(named)) in ((0, 3), (1, 2)),
         'C35 moving foot reference has no diagonal pair')
    origin = np.asarray(reference['horizontal_origin_world_m'], dtype=float)
    target = np.asarray(reference['horizontal_target_world_m'], dtype=float)
    need(origin.shape == target.shape == (4, 3)
         and np.isfinite(origin).all() and np.isfinite(target).all()
         and np.allclose(origin[list(named)], anchors[list(named)], atol=1e-10, rtol=0),
         'C35 lift origin differs from last actual anchor')
    if phase in ('horizontal', 'lower', 'probe'):
        need(np.allclose(target[list(named), 2], origin[list(named), 2],
                         atol=1e-10, rtol=0),
             'C35 swing target Z differs from original liftoff anchor')
    t = float(reference['phase_elapsed_s'])
    need(math.isfinite(t) and t >= 0., 'C35 foot phase clock invalid')
    u = min(1., t/.12)
    shape = u**3*(10.-15.*u+6.*u*u)
    rate = 0. if t >= .12 else 30.*u*u*(1.-u)*(1.-u)/.12
    if phase == 'dwell' and previous is not None and previous['phase'] == 'probe':
        measured = np.asarray(sensed['foot_world_m'], dtype=float)
        touchdown_world_m = target.copy()
        for i in named:
            touchdown_world_m[i, :2] = np.asarray(previous['horizontal_target_world_m'])[i, :2]
            touchdown_world_m[i, 2] = measured[i, 2]
    if phase in ('dwell', 'load'):
        need(touchdown_world_m is not None and
             np.allclose(target[list(named)], touchdown_world_m[list(named)], atol=1e-10, rtol=0),
             'C35 touchdown anchor differs from actual measured wheel center')
    for i in named:
        base = origin[i]
        if phase == 'lift':
            expected = base + (0., 0., .035*shape)
            vexpected = (0., 0., .035*rate)
        elif phase == 'horizontal':
            expected = None  # Independently checked by verify_horizontal_tick35.
            vexpected = None
            need(math.isclose(feet[i, 2], base[2]+.035, abs_tol=1e-10)
                 and math.isclose(velocity[i, 2], 0., abs_tol=1e-10),
                 'C35 horizontal lift-height reference differs')
        elif phase == 'lower':
            expected = np.r_[target[i, :2], base[2]+.035-.038*shape]
            vexpected = (0., 0., -.038*rate)
        elif phase == 'probe':
            expected = np.r_[target[i, :2], base[2]-.003-min(.012,.03*t)]
            vexpected = (0., 0., -.03 if t < .4 else 0.)
        else:
            expected = target[i]
            vexpected = (0., 0., 0.)
        if expected is not None:
            need(np.allclose(feet[i], expected, atol=1e-10, rtol=0)
                 and np.allclose(velocity[i], vexpected, atol=1e-10, rtol=0),
                 'C35 swing Z/XY position or derivative differs from frozen phase curve')
    return touchdown_world_m
