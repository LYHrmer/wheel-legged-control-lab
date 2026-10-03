"""C35 saved control/state/native bridge below the old single-leg classifier."""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np

from reader26 import (STATE_KEYS, MODEL_SHA, require, close, equal, byte_equal,
                      saved_document, saved_arrays, saved_rows, check_reset,
                      check_controller, check_native, _body_com_velocity, _ground_hit)
from force_read35 import recompute_paired_torque35
from native_read35 import verify_native35
from reference_math35 import (verify_horizontal_tick35, verify_horizontal_to_lower35,
                              verify_foot_reference35)
from body_read35 import replay_body35
from sensed_read35 import verify_sensed35, verify_admission35


def read_episode35(run, case_index: int, offset: int, session: dict,
                   binding: dict, geometry: dict, kin: dict,
                   side_binding: dict, joint_limits: dict) -> dict:
    """Every completed control is fully checked even when a later case aborts."""
    folder = Path(run)/f'episode_{case_index}'
    reset = saved_document(folder/'reset.json')
    initial = saved_arrays(folder/'initial_state.npz')
    states = saved_arrays(folder/'states.npz')
    segment = saved_document(folder/'segment_receipt.json')
    cycle = saved_document(folder/'cycle_receipt35.json')
    rows = list(saved_rows(folder/'controls.jsonl.gz'))
    n = len(rows)
    definition = session['cases'][case_index]
    require(cycle['schema'] == 'd1-c35-cycle-receipt-v1'
            and cycle['case_id'] == definition['id']
            and cycle['case_index'] == case_index
            and cycle['direction'] == definition['direction']
            and cycle['kind'] == definition['kind']
            and cycle['control_range'] == [offset, offset+n]
            and segment['completed_controls'] == n
            and segment['global_control_begin'] == offset
            and segment['global_control_end'] == offset+n
            and 0 <= n <= 1800,
            'C35 actual episode identity/control range differs')
    require(reset['case_index'] == case_index and reset['case_id'] == definition['id']
            and reset['global_control_offset'] == offset
            and reset['model_address'] > 0 and reset['data_address'] > 0,
            'C35 actual reset/case binding differs')
    core = {key: initial[key] for key in STATE_KEYS}
    check_reset(reset['control_state'], core)
    require(set(initial) in (set(STATE_KEYS), set(STATE_KEYS)|{'act'})
            and set(states) == set(initial)
            and all(states[k].shape == (n+1, *initial[k].shape)
                    and byte_equal(states[k][0], initial[k]) for k in initial)
            and close(states['time'], np.arange(n+1)*.01),
            'C35 complete state/initial/control clock join differs')
    require(close(initial['qpos'][:3], session['spawn_position_m'], atol=1e-12)
            and close(initial['qpos'][3:7], [1., 0., 0., 0.], atol=1e-12),
            'C35 reset pose/yaw differs from frozen case')
    guard = segment['native_segment']
    names = guard['native_files']
    require(len(names) == len(set(names))
            and guard['archive_block_manifests'] == [x+'.manifest.json' for x in names],
            'C35 native archive block list differs')
    native = []
    for name in names:
        require(Path(name).name == name, 'C35 unsafe native archive name')
        native.extend(saved_rows(folder/name))
    require(cycle['native_range'] == [5*offset, 5*offset+len(native)]
            and segment['global_native_begin'] == 5*offset
            and segment['global_native_end'] == 5*offset+len(native)
            and all(row['native_index'] == 5*offset+i for i, row in enumerate(native)),
            'C35 actual global native archive gap/duplicate/range differs')
    full = cycle['terminal_kind'] != 'fatal_incomplete_or_unsafe'
    require((len(native) == 5*n and guard['record_valid'] is True
             and not guard['partial_native_interval']) if full else
            (5*n <= len(native) <= 5*(n+1) and cycle['result']['failure'] is not None),
            'C35 native closed prefix or declared fatal suffix differs')
    require(cycle['result']['completed_controls'] == n
            and cycle['result']['actual_normal_native'] == len(native)
            and cycle['result']['terminal_kind'] == cycle['terminal_kind'],
            'C35 saved result/native count differs')
    terrain = set(geometry)
    mass = float(sum(kin['body_mass']))
    leg_dof = np.asarray(side_binding['joint_dof_addresses'], dtype=int).reshape(4, 4)[:, :3].ravel()
    wheel_map = binding['wheel_map']
    predictions = side_controls = 0
    previous_reference = previous_calc = None
    foot_anchors = touchdown_world = None
    side_rows = []
    side_native = []
    all_native_proofs = []
    entry_z = entry_yaw = None
    entry_offsets = None
    for tick, row in enumerate(rows):
        info = row['info']
        actor = row['actor']
        require(row['schema'] == 'd1-c35-control-row-v1'
                and row['control_index'] == tick
                and row['cycle_index'] == case_index
                and row['global_control_index'] == offset+tick
                and row['native_index_range'] == [5*(offset+tick), 5*(offset+tick+1)]
                and row['pre_state_index'] == tick and row['post_state_index'] == tick+1
                and row['control_stages18'] == info['controller_record']
                and info['completed_control_intervals'] == tick+1
                and actor in ('final_policy', 'continuous_pair35'),
                'C35 control row owner/index/preview join differs')
        for key in initial:
            require(equal(row['pre_state'][key], states[key][tick])
                    and equal(row['post_state'][key], states[key][tick+1]),
                    'C35 control/state endpoint differs: '+key)
        action = np.asarray(row['action16'], dtype=float)
        require(action.shape == (16,) and np.isfinite(action).all()
                and equal(row['input_observation99'], states['observation'][tick])
                and equal(row['raw_action16'], action)
                and equal(row['policy_input_action'], action)
                and row['policy_predict_called'] is (actor == 'final_policy')
                and row['checkpoint_sha256'] == (MODEL_SHA if actor == 'final_policy' else None),
                'C35 B22 action/observation/model-call identity differs')
        raw = row['raw_command']
        require(raw == info['raw_operator_command']
                and raw['forward_velocity_mps'] == raw['lateral_velocity_mps'] == raw['yaw_rate_rps'] == 0.
                and info['consumed_command']['forward_velocity_mps'] == 0.
                and info['consumed_command']['lateral_velocity_mps'] == 0.
                and info['consumed_command']['yaw_rate_rps'] == 0.,
                'C35 independent vy contaminated frozen B22 command channel')
        if actor == 'continuous_pair35':
            require(equal(action, np.zeros(16)) and row['side_requested'] is True
                    and tick >= 200 and entry_z is not None,
                    'C35 side owner/action entered outside explicit session')
            side_controls += 1
            owner = info['controller_record']
            if side_controls == 1:
                observed_feet = np.asarray(owner['sensed35']['foot_world_m'], dtype=float)
                require(observed_feet.shape == (4, 3) and np.isfinite(observed_feet).all(),
                        'C35 side entry lacks four actual foot centers')
                entry_offsets = observed_feet[:, :2]-np.asarray(states['qpos'][tick][:2])
                foot_anchors = observed_feet.copy()
                first_calc = owner['controller_record35']['calculation']
                qa = np.asarray(binding['qpos_addresses'], dtype=int)
                require(close(first_calc['previous_target16'],
                              np.asarray(states['qpos'][tick])[qa])
                        and close(first_calc['previous_velocity_target16'], np.zeros(16)),
                        'C35 Fast.reset target/derivative state not bound to side entry')
            require(owner['schema'] == 'd1-c35-continuous-side-control-v1'
                    and owner['side_compute_index'] == side_controls
                    and owner['prepared_preview_consumed'] is True
                    and owner['requested_policy_action'] == [0.]*16
                    and owner['physical_policy_residual_nm'] == [0.]*16,
                    'C35 side owner/prepared exact-zero action differs')
            prior_five = all_native_proofs[-5:]
            sensed_proof = verify_sensed35(owner['sensed35'], states['qpos'][tick],
                states['qvel'][tick], prior_five, tick, offset+tick,
                binding, kin, wheel_map)
            reference = owner['controller_record35']['reference']
            for event in owner['controller_record35']['pair_events']:
                before = np.asarray(event['anchors_before_world_m'], dtype=float)
                after = np.asarray(event['anchors_after_world_m'], dtype=float)
                pair = tuple(event['pair_legs'])
                sensed_feet = np.asarray(owner['sensed35']['foot_world_m'], dtype=float)
                require(event['event'] == 'pair_completed'
                        and tuple(sorted(pair)) in ((0, 3), (1, 2))
                        and close(before, foot_anchors)
                        and all(close(after[j], sensed_feet[j]) for j in pair)
                        and all(close(after[j], before[j]) for j in range(4) if j not in pair),
                        'C35 pair anchor update differs from actual measured touchdown pair')
                foot_anchors = after
                touchdown_world = None
            touchdown_world = verify_foot_reference35(reference, previous_reference,
                foot_anchors, owner['sensed35'], touchdown_world)
            verify_admission35(reference, previous_reference, sensed_proof,
                               prior_five, offset+tick)
            calc = recompute_paired_torque35(owner, states['qpos'][tick], states['qvel'][tick],
                binding, kin, entry_z_m=entry_z, entry_yaw_rad=entry_yaw,
                previous_reference=previous_reference,
                joint_ids=side_binding['joint_ids'],
                joint_ranges=joint_limits['jnt_range'],
                joint_limited=joint_limits['jnt_limited'])
            if previous_calc is not None:
                current_calc = owner['controller_record35']['calculation']
                require(close(current_calc['previous_target16'], previous_calc['target16'])
                        and close(current_calc['previous_velocity_target16'],
                                  previous_calc['target_velocity16']),
                        'C35 paired target derivative state chain differs')
            if reference['phase'] == 'horizontal':
                actual_vcom = owner['sensed35']['body_com_vxy_mps']
                verify_horizontal_tick35(reference, previous_reference, entry_offsets,
                                         actual_vcom)
            if previous_reference is not None and (previous_reference['phase'], reference['phase']) == ('horizontal', 'lower'):
                verify_horizontal_to_lower35(previous_reference, reference)
            scope = owner['access_scope_receipt']
            weights = np.asarray(reference['force_weights'], dtype=float)
            require(scope['kind'] == 'compute'
                    and scope['physical_arrays_unchanged'] is True
                    and scope['native_boundary_unchanged'] is True
                    and scope['failure'] is None
                    and scope['calls']['jac'] == int(np.sum(weights > 1e-3))
                    and scope['calls']['fullM'] == int(bool(reference['swing_legs']))
                    and scope['calls']['objectVelocity'] == 1,
                    'C35 counted static dynamics cache does not match paired compute')
            side_rows.append(row)
            previous_calc = owner['controller_record35']['calculation']
            qa = np.asarray(binding['qpos_addresses'], dtype=int)
            wheel = np.arange(3, 16, 4)
            require(close(previous_calc['wheel_angles4'],
                          np.asarray(states['qpos'][200])[qa[wheel]]),
                    'C35 wheel brake anchors drifted from actual side entry')
        else:
            require(row['side_requested'] is False
                    and row['checkpoint_sha256'] == MODEL_SHA,
                    'C35 B22 rolling tick masked as side')
            predictions += 1
            calc = check_controller(row, tick, pre_qpos=states['qpos'][tick],
                pre_qvel=states['qvel'][tick],
                pre_body_forward_mps=float(_body_com_velocity(
                    states['qpos'][tick], states['qvel'][tick], binding)[0]),
                servo_yaw_rps=0.)
            reference = None
        five = native[5*tick:5*tick+5]
        alias = dict(row, episode_tick=tick)
        check_native(alias, five, calc, binding, geometry, 5*(offset+tick))
        require(row['side_interval35']['contact_evidence35']
                == [item['contact_evidence35'] for item in five],
                'C35 saved five-native evidence detached from native slots')
        for item in five:
            proof = verify_native35(item, reference, kin=kin, wheel_map=wheel_map,
                terrain_geoms=terrain, mass_kg=mass,
                leg_dof_addresses=leg_dof,
                previous_reference=previous_reference if reference is not None else None)
            all_native_proofs.append(proof)
            if reference is not None:
                side_native.append(proof)
        if reference is not None:
            previous_reference = reference
        q, v = states['qpos'][tick+1], states['qvel'][tick+1]
        metrics = info['metrics']
        ground, _, gid = _ground_hit(geometry, q[0], q[1])
        velocity = _body_com_velocity(q, v, binding)
        require(close(metrics['body_com_vx_mps'], velocity[0], atol=1e-6)
                and close(metrics['body_com_vy_mps'], velocity[1], atol=1e-6)
                and close(metrics['body_yaw_rate_rps'], v[5], atol=1e-6)
                and close(metrics['clearance_m'], q[2]-ground, atol=1e-6)
                and metrics['ground_geom_id'] == gid
                and close(row['reward'], sum(info['reward_terms'].values())),
                'C35 actual endpoint/reward geometry differs')
        require(not row['terminated'] and not row['truncated'],
                'C35 continued after environment termination')
        if tick == 199:
            entry_z = float(states['qpos'][200][2])
            qentry = np.asarray(states['qpos'][200])
            w, x, y, z = qentry[3:7]
            entry_yaw = math.atan2(2.*(w*z+x*y), 1.-2.*(y*y+z*z))
    pending = cycle['result']['pending_control']
    expected_predictions = predictions+int(pending is not None and not pending['side_requested'])
    require(side_controls <= 1200 and predictions == n-side_controls
            and cycle['result']['policy_predictions'] == expected_predictions
            and cycle['result']['pair_exchanges'] == len(cycle['pair_events']),
            'C35 side/model/pair count differs')
    body = replay_body35(side_rows, states['qpos'][200]) if side_rows else None
    if body is not None:
        require(body['first_stop_control_index'] == cycle['command']['stop_control_index'],
                'C35 actual raw stop boundary differs from independent command replay')
    partial_suffix = native[5*n:]
    require(not partial_suffix or not full,
            'C35 complete episode contains an unreturned native suffix')
    pending_reference = None if pending is None else (
        (pending.get('actual_last_side_record') or {}).get('reference'))
    partial_proofs = []
    for item in partial_suffix:
        if 'contact_evidence35' in item:
            try:
                proof = verify_native35(item, pending_reference,
                    kin=kin, wheel_map=wheel_map, terrain_geoms=terrain,
                    mass_kg=mass, leg_dof_addresses=leg_dof,
                    previous_reference=previous_reference)
                partial_proofs.append({'native_index': item['native_index'],
                                       'saved_native_proof_passed': True,
                                       'actual_support_gate': proof['support']})
            except (AssertionError, ValueError) as error:
                partial_proofs.append({'native_index': item['native_index'],
                                       'saved_native_proof_passed': False,
                                       'failure': str(error), 'qualification': False})
        else:
            partial_proofs.append({'native_index': item['native_index'],
                                   'contact_evidence_complete': False,
                                   'qualification': False})
    return {'cycle': cycle, 'segment': segment, 'reset': reset, 'states': states,
            'initial': initial, 'rows': rows, 'native': native, 'side_rows': side_rows,
            'side_native': side_native, 'native_proofs': all_native_proofs,
            'side_controls': side_controls, 'predictions': predictions,
            'closed_prefix_controls': n, 'fatal_partial': not full,
            'body_reference': body, 'partial_native_proofs': partial_proofs,
            'partial_native_rows': len(partial_suffix)}
