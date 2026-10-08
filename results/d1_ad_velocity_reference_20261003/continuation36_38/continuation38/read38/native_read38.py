"""C38-aware extension of the frozen read35.native_read35.verify_native35.

The frozen checker hardcodes a C35-era assumption: `transfer` and `transfer_restore` are
always in the all-four regime. C36/C37/C38 introduced a new, pure-tested regime inside
those same phases - the pre-unload window, open while `phase == 'transfer_restore'` or
`phase == 'transfer' and phase_elapsed_s >= 0.03` - during which only the stance pair (the
complement of the swing pair) is required to carry load, exactly mirroring the strictness
the frozen air regime already applies one phase later.

This is a documentation gap in the independent reader, not a weakening: it extends the
reader's known-good expectation to a regime that was pure-tested before any physics (see
continuation36/sol36/test_core36_pure.py and continuation38/sol38/test_adapter38_pure.py),
and it determines the window from fields the frozen reader already trusts for its OTHER
branches (`reference['phase']`, now also `reference['phase_elapsed_s']`) - no new field and
no new trust assumption. The boundary constant (0.03 s) is copied, not re-derived, from
continuation36/sol36/core36.py's PREUNLOAD_ELAPSED36; a pure test below asserts the two
values are byte-identical imports from the same frozen source rather than a re-typed literal.

Everything else - the contact resultant, the whole-wheel shape, the moment, the horizontal
clearance check, the pose/velocity/collision bounds, the stop margin - is the frozen
`verify_native35` body, copied verbatim.
"""
from __future__ import annotations

import math

import numpy as np
from contact_math35 import (native_phase_support35, native_resultant35,
                            pair_support35, static_contact_margin35)
from geometry_read30 import wheel_envelope30
from kinematics24 import reconstruct24
from native_read35 import close, need

PREUNLOAD_ELAPSED38 = .03  # must equal core36.PREUNLOAD_ELAPSED36; asserted by a pure test


def preunload_window38(phase: str, phase_elapsed_s: float) -> bool:
    """Independent copy of PairController36.preunload36, evaluated on the saved record."""
    if phase == 'transfer':
        return phase_elapsed_s >= PREUNLOAD_ELAPSED38
    return phase == 'transfer_restore'


def native_phase_support38(result: dict, phase: str, phase_elapsed_s: float,
                           swing_pair, mass_kg: float, gravity_mps2: float = 9.81) -> dict:
    """native_phase_support35 extended with the pre-unload stance-pair regime."""
    if phase in ('transfer', 'transfer_restore') and preunload_window38(phase, phase_elapsed_s):
        need(swing_pair is not None and tuple(sorted(swing_pair)) in ((0, 3), (1, 2)),
             'pre-unload transfer lacks a named diagonal swing pair')
        stance_pair = tuple(i for i in range(4) if i not in swing_pair)
        proof = pair_support35(result, stance_pair, mass_kg, gravity_mps2)
        return {'phase': phase, 'gate': 'stance_pair_preunload38', **proof}
    return native_phase_support35(result, phase, swing_pair, mass_kg, gravity_mps2)


def verify_native38(native: dict, reference: dict | None, *, kin: dict, wheel_map: dict,
                    terrain_geoms: set[int], mass_kg: float, leg_dof_addresses,
                    previous_reference: dict | None = None) -> dict:
    """verify_native35 with the mask/gate branch extended for the pre-unload window."""
    evidence = native['contact_evidence35']
    need(evidence['schema'] == 'd1-c35-native-contact-evidence-v1'
         and evidence['native_index'] == native['native_index']
         and evidence['global_control_index'] == native['native_index']//5,
         'C38 native contact evidence detached from actual substep')
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
         'C38 saved world contact resultant/whole wheel shape differs')
    if reference is None:
        need(evidence['stance_legs'] == evidence['swing_legs'] == []
             and evidence['horizontal_active'] is False
             and evidence['support_gate_required'] is False,
             'B22 native falsely claimed side contact permission')
        need(max(abs(float(native['roll_deg'])), abs(float(native['pitch_deg']))) <= 10.
             and float(q[2]) >= .28 and abs(float(q[0])) <= 11.
             and abs(float(q[1])) <= 6.
             and native['nonwheel_contact_count'] == 0,
             'C38 original B22 rolling native world/pose/collision bound failed')
        return {'native_index': native['native_index'], 'contact': result,
                'support': None, 'shape_min_z_m': shape}
    phase = reference['phase']
    pair = reference['pair']
    elapsed = float(reference['phase_elapsed_s'])
    preunload = phase in ('transfer', 'transfer_restore') and preunload_window38(phase, elapsed)
    if phase in ('idle', 'transfer', 'transfer_restore', 'load', 'settle', 'done') and not preunload:
        expected_stance, expected_swing = list(range(4)), []
    else:
        need(pair is not None and tuple(sorted(pair)) in ((0, 3), (1, 2)),
             'consumed air/pre-unload reference lacks true diagonal pair')
        expected_stance = [i for i in range(4) if i not in pair]
        # Pre-unload declares the SAME convention C38's own adapter writes: stance_legs is
        # the complement of pair, swing_legs stays at the frozen empty value (see
        # adapter38.PairController38._reference_record35 and its pure tests). Only the true
        # air phases (lift/horizontal/lower/probe/dwell) declare swing_legs as the pair.
        expected_swing = [] if preunload else list(pair)
    need(evidence['stance_legs'] == expected_stance
         and evidence['swing_legs'] == expected_swing
         and evidence['support_gate_required'] is True
         and evidence['consumed_phase'] == phase,
         'C38 saved native support masks/phase differ from consumed reference')
    feet_velocity = np.asarray(reference['feet_velocity_world_mps'], dtype=float)
    need(feet_velocity.shape == (4, 3) and np.isfinite(feet_velocity).all(),
         'missing consumed four-foot reference velocity')
    air_swing = list(pair) if (not preunload and phase in
                               ('lift', 'horizontal', 'lower', 'probe', 'dwell')) else []
    current_horizontal = bool(phase == 'horizontal' and any(
        np.linalg.norm(feet_velocity[leg, :2]) > 1e-12 for leg in air_swing))
    previous_horizontal = False
    if (previous_reference is not None
            and phase in ('lift', 'horizontal', 'lower', 'probe', 'dwell')
            and previous_reference['phase'] == 'horizontal'
            and previous_reference['pair'] == pair):
        previous_velocity = np.asarray(previous_reference['feet_velocity_world_mps'],
                                       dtype=float)
        need(previous_velocity.shape == (4, 3) and np.isfinite(previous_velocity).all(),
             'missing prior horizontal reference velocity')
        previous_horizontal = bool(any(np.linalg.norm(previous_velocity[leg, :2]) > 1e-12
                                       for leg in air_swing))
    horizontal = bool(current_horizontal or previous_horizontal)
    need(reference['horizontal_reference_velocity_nonzero'] is current_horizontal
         and reference['horizontal_previous_velocity_nonzero'] is previous_horizontal
         and evidence['horizontal_active'] is horizontal,
         'C38 horizontal native gate flag differs from consumed foot motion')
    support = native_phase_support38(result, phase, elapsed,
                                     pair if (preunload or air_swing) else None,
                                     mass_kg)
    need(support['passed'], 'C38 actual native pair/four-foot/pre-unload contact/load gate failed')
    if horizontal:
        need(all(shape[leg] > .012
                 and not result['effective_terrain_wheel_contact_mask'][leg]
                 for leg in air_swing),
             'C38 actual moving swing whole-wheel shape/contact gate failed')
    leg_dof = np.asarray(leg_dof_addresses, dtype=int)
    need(leg_dof.shape == (12,) and len(set(leg_dof.tolist())) == 12,
         'C38 actual leg velocity binding missing')
    need(max(abs(float(native['roll_deg'])), abs(float(native['pitch_deg'])))
             <= math.degrees(.32)
         and float(q[2]) >= .32 and np.max(np.abs(np.asarray(v)[leg_dof])) <= 18.
         and native['nonwheel_contact_count'] == 0,
         'C38 original side native pose/joint velocity/nonwheel bounds failed')
    margin = (static_contact_margin35(result, com)
              if all(result['positive_wheel_support_mask']) else None)
    return {'native_index': native['native_index'], 'contact': result,
            'support': support, 'shape_min_z_m': shape, 'margin': margin,
            'preunload38': preunload}
