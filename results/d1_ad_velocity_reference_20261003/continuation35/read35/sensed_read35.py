"""Bind each saved C35 sensor input to pre-state FK and prior native contacts."""
from __future__ import annotations

import numpy as np

from kinematics24 import reconstruct24, rotation
from task31 import _geometry31, _state31
from contact_math35 import pair_support35, static_contact_margin35, horizontal_admission35


def need(ok, why):
    if not ok:
        raise AssertionError(why)


def near(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    return (a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=2e-8, rtol=0))


def verify_sensed35(sensed, q, v, prior_five, control_index,
                    global_control_index, binding, kin, wheel_map):
    """No controller/runtime import: all values originate in saved physical rows."""
    need(len(prior_five) == 5 and
         [r['native_index'] for r in prior_five]
            == list(range(5*(global_control_index-1), 5*global_control_index)),
         'C35 sensed gate lacks immediately preceding five native samples')
    q, v = np.asarray(q), np.asarray(v)
    fk = reconstruct24(kin, q, v)
    geom = _geometry31(kin, q, v, wheel_map)
    state = _state31(q, v, binding)
    centers = np.zeros((4, 3))
    for body, leg in wheel_map.items():
        gid = next(i for i, owner in enumerate(kin['geom_bodyid']) if int(owner) == body)
        frame = fk['geom_xmat'][gid] @ rotation(kin['geom_quat'][gid]).T
        centers[leg] = fk['geom_xpos'][gid]-frame@np.asarray(kin['geom_pos'][gid])
    last = prior_five[-1]['contact']
    masks = [p['contact']['effective_terrain_wheel_contact_mask'] for p in prior_five]
    mass = float(sum(kin['body_mass']))
    pairs = ((0, 3), (1, 2))
    pair_loaded = [all(pair_support35(p['contact'], tuple(i for i in range(4)
                   if i not in pair), mass)['passed'] for p in prior_five)
                   for pair in pairs]
    margin = (static_contact_margin35(last, fk['whole_com_position'])['static_margin_m']
              if all(last['positive_wheel_support_mask']) else -1.)
    world_com_vxy = rotation(q[3:7])@state['body_com_velocity']
    numeric = dict(body_xy_m=q[:2], body_com_vxy_mps=world_com_vxy[:2],
                   body_origin_vxyz_mps=state['origin_velocity'], yaw_rad=state['rpy'][2],
                   foot_world_m=centers,
                   wheel_contact_points_world_m=geom['contact_points_m'],
                   actual_whole_wheel_min_z_m=geom['wheel_shape_min_z_m'],
                   normal_load_n=last['positive_wheel_normal_load_sum_n'],
                   mass_kg=mass, whole_com_z_m=fk['whole_com_position'][2],
                   base_z_m=q[2], static_support_margin_m=margin,
                   world_angular_speed_rps=np.linalg.norm(state['world_angular_velocity']))
    booleans = dict(previous_five_contact_free=[all(not mask[i] for mask in masks)
                                               for i in range(4)],
                    previous_five_pair_loaded=pair_loaded,
                    foot_contact=last['effective_terrain_wheel_contact_mask'])
    need(sensed['control_index'] == control_index and sensed['native_safety_ok'] is True
         and all(near(sensed[k], value) for k, value in numeric.items())
         and all(list(sensed[k]) == list(value) for k, value in booleans.items()),
         'C35 sensed input differs from actual pre-state or previous five contacts')
    return {'geometry': geom, 'masks': masks, 'last_contact': last,
            'mass_kg': mass, 'pair_loaded': pair_loaded}


def verify_admission35(reference, previous_reference, sensed_proof, prior_five,
                       global_control_index):
    """The first lift/horizontal tick must use current and previous-5 evidence."""
    pair = reference['pair']
    if pair is None:
        return
    pair = tuple(pair)
    prior_phase = None if previous_reference is None else previous_reference['phase']
    if reference['phase'] == 'lift' and prior_phase == 'transfer':
        stance = tuple(i for i in range(4) if i not in pair)
        now = pair_support35(sensed_proof['last_contact'], stance,
                             sensed_proof['mass_kg'])
        need(now['passed'] and all(pair_support35(p['contact'], stance,
             sensed_proof['mass_kg'])['passed'] for p in prior_five),
             'C35 first lift lacked current and previous-five actual stance load')
    if reference['phase'] == 'horizontal' and prior_phase == 'lift':
        stance = tuple(i for i in range(4) if i not in pair)
        now = pair_support35(sensed_proof['last_contact'], stance,
                             sensed_proof['mass_kg'])
        admission = horizontal_admission35(pair=pair,
            pre_shape_min_z_m=sensed_proof['geometry']['wheel_shape_min_z_m'],
            previous_native_indices=[r['native_index'] for r in prior_five],
            previous_effective_masks=sensed_proof['masks'],
            first_horizontal_control_index=global_control_index,
            stance_support=now)
        need(admission['passed'],
             'C35 first horizontal lacked measured >12mm shape/contact-free pair admission')
