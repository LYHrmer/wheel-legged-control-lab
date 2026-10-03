"""Independent C35 native contact/mask proof over the frozen lower reader."""
from __future__ import annotations

import math
import numpy as np

from geometry_read30 import wheel_envelope30
from kinematics24 import reconstruct24
from contact_math35 import (native_resultant35, native_phase_support35,
                            static_contact_margin35)


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise AssertionError(reason)


def close(a, b, atol=2e-8) -> bool:
    x, y = np.asarray(a), np.asarray(b)
    return bool(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
                and np.allclose(x, y, atol=atol, rtol=0))


def verify_native35(native: dict, reference: dict | None, *, kin: dict,
                    wheel_map: dict, terrain_geoms: set[int], mass_kg: float,
                    leg_dof_addresses, previous_reference: dict | None = None) -> dict:
    """Call after reader26.check_native's state/clock/actuator/contact proof.

    `reference` must be the controller reference actually consumed for this
    interval.  C35 never trusts a saved support-gate boolean as qualification.
    """
    evidence = native['contact_evidence35']
    need(evidence['schema'] == 'd1-c35-native-contact-evidence-v1'
         and evidence['native_index'] == native['native_index']
         and evidence['global_control_index'] == native['native_index']//5,
         'C35 native contact evidence detached from actual substep')
    q, v = native['after']['qpos'], native['after']['qvel']
    state = reconstruct24(kin, q, v)
    com = state['whole_com_position']
    shape = wheel_envelope30(kin, q, v, wheel_map)['wheel_shape_min_z_m']
    result = native_resultant35(native, terrain_geoms, com)
    need(close(evidence['whole_com_world_m'], com)
         and close(evidence['world_force_sum_n'], result['world_force_sum_n'])
         and close(evidence['world_moment_about_com_sum_nm'],
                   result['world_moment_about_com_sum_nm'])
         and close(evidence['positive_wheel_normal_load_sum_n'],
                   result['positive_wheel_normal_load_sum_n'])
         and evidence['active_wheel_terrain_contact']
             == list(result['effective_terrain_wheel_contact_mask'])
         and close(evidence['whole_wheel_min_z_m'], shape)
         and close(native['skill30_geometry']['whole_wheel_min_z_m'], shape),
         'C35 saved world contact resultant/whole wheel shape differs')
    if reference is None:
        need(evidence['stance_legs'] == evidence['swing_legs'] == []
             and evidence['horizontal_active'] is False
             and evidence['support_gate_required'] is False,
             'B22 native falsely claimed side contact permission')
        need(max(abs(float(native['roll_deg'])), abs(float(native['pitch_deg']))) <= 10.
             and float(q[2]) >= .28 and abs(float(q[0])) <= 11.
             and abs(float(q[1])) <= 6.
             and native['nonwheel_contact_count'] == 0,
             'C35 original B22 rolling native world/pose/collision bound failed')
        return {'native_index': native['native_index'], 'contact': result,
                'support': None, 'shape_min_z_m': shape}
    phase = reference['phase']
    pair = reference['pair']
    if phase in ('idle', 'transfer', 'transfer_restore', 'load', 'settle', 'done'):
        expected_stance, expected_swing = list(range(4)), []
    else:
        need(pair is not None and tuple(sorted(pair)) in ((0, 3), (1, 2)),
             'consumed air reference lacks true diagonal pair')
        expected_swing = list(pair)
        expected_stance = [i for i in range(4) if i not in pair]
    need(evidence['stance_legs'] == expected_stance
         and evidence['swing_legs'] == expected_swing
         and evidence['support_gate_required'] is True
         and evidence['consumed_phase'] == phase,
         'C35 saved native support masks/phase differ from consumed reference')
    feet_velocity = np.asarray(reference['feet_velocity_world_mps'], dtype=float)
    need(feet_velocity.shape == (4, 3) and np.isfinite(feet_velocity).all(),
         'missing consumed four-foot reference velocity')
    current_horizontal = bool(phase == 'horizontal' and any(
        np.linalg.norm(feet_velocity[leg, :2]) > 1e-12 for leg in expected_swing))
    previous_horizontal = False
    if (previous_reference is not None and phase in ('lift', 'horizontal', 'lower', 'probe', 'dwell')
            and previous_reference['phase'] == 'horizontal'
            and previous_reference['pair'] == pair):
        previous_velocity = np.asarray(previous_reference['feet_velocity_world_mps'], dtype=float)
        need(previous_velocity.shape == (4, 3) and np.isfinite(previous_velocity).all(),
             'missing prior horizontal reference velocity')
        previous_horizontal = bool(any(np.linalg.norm(previous_velocity[leg, :2]) > 1e-12
                                       for leg in expected_swing))
    horizontal = bool(current_horizontal or previous_horizontal)
    need(reference['horizontal_reference_velocity_nonzero'] is current_horizontal
         and reference['horizontal_previous_velocity_nonzero'] is previous_horizontal
         and evidence['horizontal_active'] is horizontal,
         'C35 horizontal native gate flag differs from consumed foot motion')
    support = native_phase_support35(result, phase, tuple(expected_swing) if expected_swing else None,
                                     mass_kg)
    need(support['passed'], 'C35 actual native pair/four-foot contact/load gate failed')
    if horizontal:
        need(all(shape[leg] > .012
                 and not result['effective_terrain_wheel_contact_mask'][leg]
                 for leg in expected_swing),
             'C35 actual moving swing whole-wheel shape/contact gate failed')
    leg_dof = np.asarray(leg_dof_addresses, dtype=int)
    need(leg_dof.shape == (12,) and len(set(leg_dof.tolist())) == 12,
         'C35 actual leg velocity binding missing')
    need(max(abs(float(native['roll_deg'])), abs(float(native['pitch_deg'])))
             <= math.degrees(.32)
         and float(q[2]) >= .32 and np.max(np.abs(np.asarray(v)[leg_dof])) <= 18.
         and native['nonwheel_contact_count'] == 0,
         'C35 original side native pose/joint velocity/nonwheel bounds failed')
    margin = (static_contact_margin35(result, com)
              if all(result['positive_wheel_support_mask']) else None)
    return {'native_index': native['native_index'], 'contact': result,
            'support': support, 'static_margin': margin,
            'shape_min_z_m': shape, 'horizontal_active': horizontal,
            'horizontal_current': current_horizontal,
            'horizontal_previous': previous_horizontal}
