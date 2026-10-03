"""Cold saved-only C35 continuous diagonal-pair independent readback."""
from __future__ import annotations

import argparse
import importlib.abc
import json
import math
import os
from pathlib import Path
import sys
import traceback

FORBIDDEN = {'mujoco', 'glfw', 'torch', 'stable_baselines3', 'gym',
             'gymnasium', 'engine_binding', 'wheel_legged_control'}


class NoExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] in FORBIDDEN:
            raise RuntimeError('C35 saved reader forbids execution import: '+fullname)


if any(name.partition('.')[0] in FORBIDDEN for name in sys.modules):
    raise RuntimeError('C35 reader must be launched cold')
sys.meta_path.insert(0, NoExecution())
sys.dont_write_bytecode = True

W = Path(__file__).resolve().parents[2]
R = Path('/home/lyh/wheel-legged-control-lab')
for directory in (W, W/'course_impl08', W/'rl11', W/'continuation18',
                  W/'continuation24', W/'continuation26/root26',
                  W/'continuation27/read27', W/'continuation30/read30',
                  W/'continuation31/root31', W/'continuation35/read35'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np
from reader26 import (require, close, document, saved_document, identity,
                      check_binding, geometry_map, MODEL_SHA)
from episode_read35 import read_episode35
from task_math35 import continuous_metrics35, entry_heading35
from stop_cost_math35 import stop_metrics35, torque_cost35
from side_math27 import scalar_state, compiled_side_geometry, LIMITS
from reader27 import frozen_profile
from task31 import _tail31, _native_loads31, GATES31
from ledger_read35 import check_ledger35


CONTRACT = 'C35_continuous_diagonal_pair_fixed_feasibility_v1'


def validate_cycle35_shape(cycle: dict) -> None:
    """Explicit top-level receipt contract, also used by pure schema fixture."""
    require(cycle['schema'] == 'd1-c35-cycle-receipt-v1'
            and isinstance(cycle['pair_events'], list)
            and isinstance(cycle['result'], dict)
            and isinstance(cycle['command'], dict)
            and isinstance(cycle['control_range'], list)
            and isinstance(cycle['native_range'], list)
            and isinstance(cycle['side_access'], dict)
            and type(cycle['archive_completed']) is bool,
            'C35 cycle receipt top-level schema differs')
    result, command = cycle['result'], cycle['command']
    require(all(key in result for key in ('completed_controls', 'actual_normal_native',
                                        'policy_predictions', 'pair_exchanges',
                                        'pending_control', 'terminal_kind', 'task', 'failure'))
            and all(key in command for key in ('definition', 'side_begin_control',
                                              'side_end_control', 'stop_control_index',
                                              'cancel_control_index')),
            'C35 cycle receipt result/command paths differ')


def same_optional35(actual, expected) -> bool:
    return (actual is None and expected is None) or (
        actual is not None and expected is not None and close(actual, expected))


def verify_case_command35(side: list[dict], definition: dict, cycle: dict) -> dict:
    """Bind fresh per-control command to the frozen active pulse/cancel trigger."""
    command = cycle['command']
    require(command['definition'] == definition and command['side_begin_control'] == 200
            and definition['active_controls'] in (200, 600)
            and definition['vy_mps'] == definition['direction']*.025,
            'C35 actual case command differs from preregistration')
    moving = [row['control_index'] for row in side if row['info']['controller_record']
              ['controller_record35']['reference']['horizontal_reference_velocity_nonzero']]
    cancel_index = (moving[0]+1 if definition['kind'] == 'cancel' and moving else None)
    require(definition['kind'] != 'cancel' or cancel_index is not None,
            'C35 complete cancel case lacked first actual horizontal movement')
    require(command['cancel_control_index'] == cancel_index,
            'C35 cancel did not fire on next control after first real horizontal movement')
    stop_index = min(200+definition['active_controls'], cancel_index) if cancel_index else 200+definition['active_controls']
    require(command['stop_control_index'] == stop_index,
            'C35 raw stop boundary differs from frozen pulse/cancel plan')
    for row in side:
        index = row['control_index']
        stopping = index >= stop_index
        expected = dict(vy_mps=0. if stopping else definition['vy_mps'],
                        gait_enabled=not stopping, issued_control_index=index,
                        expires_control_index=index+20,
                        stop_kind=('cancel' if cancel_index is not None and index >= cancel_index
                                   else 'release' if index >= 200+definition['active_controls']
                                   else 'none'))
        require(row['info']['controller_record']['command35'] == expected,
                'C35 actual fresh raw command differs from signed case schedule')
    return {'first_actual_horizontal_control': moving[0] if moving else None,
            'cancel_control': cancel_index, 'stop_control': stop_index}


def sources35(run: Path, go_path: Path) -> tuple[dict, dict, dict]:
    session = document(run/'session.json')
    go = document(go_path)
    worker = document(run/'worker_receipt.json')
    spec = document(session['spec35_path'])
    require(go['decision'] == 'GO' and go['execution_contract_id'] == CONTRACT
            and session['execution_contract_id'] == CONTRACT
            and session['arm'] == 'development' and session['mode'] == 'fixed'
            and session['render'] is False and session['retry_permitted'] is False
            and Path(session['output_directory']).resolve() == run
            and all(session.get(key) == value for key, value in spec.items())
            and session['control_limit'] == 9000 and session['cycles_limit'] == 5
            and session['normal_native_cap'] == 45000,
            'C35 frozen source GO/session/spec differs')
    hashes = session['source_hashes']
    require(all(hashes.get(path) == identity(path) for path in hashes)
            and all(hashes.get(path) == value for path, value in go['inputs'].items())
            and hashes.get(str(go_path)) == identity(go_path),
            'C35 actual sources differ from reviewed bundle')
    require(worker['schema'] == 'd1-c35-headless-worker-v1'
            and worker['execution_contract_id'] == CONTRACT
            and worker['session_identity'] == identity(run/'session.json')
            and worker['archive_failed'] is False
            and worker['cleanup_errors'] == []
            and worker['warnings'] == []
            and worker['retry_permitted'] is False,
            'C35 worker/archive provenance differs')
    host = document(run/'host_receipt.json')
    supervisor = document(run/'supervisor_receipt.json')
    require(host['source_mismatches'] == [] and host['owned_no_orphans'] is True
            and host['reservation_closed'] is True
            and supervisor['cleanup']['remaining'] == {}
            and supervisor['outer_limit_s'] == 1200,
            'C35 owned host/source/deadline did not close')
    for name in ('runtime35_module_origins_before.json', 'runtime35_module_origins.json',
                 'loaded_origins_final.json', 'mapped_libraries_before_load.json',
                 'mapped_libraries_final.json'):
        path = run/name
        if path.exists():
            value = document(path)
            if name.startswith('runtime35_module'):
                require(all(hashes.get(row['path']) == identity(row['path'])
                            for row in value.values()),
                        'C35 runtime module origin left reviewed source closure')
            if name.startswith('mapped_libraries'):
                require(all(path in hashes for path in value['paths']),
                        'C35 mapped numerical library unsealed')
    require(session['checkpoint_sha256'] == MODEL_SHA,
            'C35 B22 checkpoint identity differs')
    return session, worker, go


def construction35(run: Path, session: dict) -> tuple[dict, dict, dict, dict, dict]:
    construction = saved_document(run/'construction_receipt.json')
    proof = construction['proof']
    require(proof['passed'] is True
            and proof['dso_path'] == session['library']
            and proof['dso_sha256'] == session['source_hashes'][session['library']]['sha256']
            and len(proof['jump_slots']) == 4
            and all(slot['passed'] is True for slot in proof['jump_slots'])
            and construction['C_state']['construction_attempts']
                == construction['C_state']['construction_returns'] == 2
            and construction['nominal_cache']['misses'] == 1,
            'C35 original cold engine/construction proof differs')
    binding = check_binding(construction['geometry_binding'])
    geometry = geometry_map(construction['compiled_geometry'], binding)
    reference = document(session['reference_construction_path'])
    require(construction['geometry_binding'] == reference['geometry_binding']
            and construction['compiled_geometry'] == reference['compiled_geometry'],
            'C35 actual compiled course/robot differs from frozen construction')
    side = construction['side_kinematic_binding']
    require({key: value for key, value in side.items()
             if key not in ('kinematics24', 'dynamics24', 'solver24')}
            == construction['geometry_binding'],
            'C35 side FK/dynamics binding differs from compiled robot')
    kin = side['kinematics24']
    for key in ('geom_bodyid', 'geom_type', 'geom_size', 'geom_contype',
                'geom_conaffinity', 'body_mass'):
        require(close(kin[key], construction['geometry_binding'][key]),
                'C35 compiled FK array differs: '+key)
    require(close(sum(kin['body_mass']), side['nominal_total_mass_kg'])
            and close(np.asarray(side['dynamics24']['actuator_ctrlrange'])[binding['actuator_ids']],
                      np.stack((-LIMITS, LIMITS), axis=1))
            and construction['side_profile']
                == frozen_profile(R/'scripts/d1_fast_side_step.py'),
            'C35 compiled mass/original sixteen torque limits differ')
    strict = saved_document(run/'strict_load.json')
    require(strict['probe_rows'] == 32 and strict['probe_actions_byte_exact'] is True
            and strict['matches_training_final_policy_state'] is True
            and strict['verified_without_engine_or_reset'] is True,
            'C35 strict B22 load/probe differs')
    return construction, binding, geometry, kin, side


def case_evidence35(episode: dict, definition: dict, baseline_integral: float,
                    binding: dict, geometry: dict, kin: dict) -> dict:
    cycle = episode['cycle']
    validate_cycle35_shape(cycle)
    if episode['fatal_partial']:
        return {'case_id': definition['id'], 'record_auditable': True,
                'physical_task_qualified': False, 'partial_record': True,
                'closed_prefix_controls': episode['closed_prefix_controls'],
                'partial_native_rows': episode['partial_native_rows'],
                'partial_native_proofs': episode['partial_native_proofs'],
                'first_failure': cycle['result']['failure'],
                'uncompleted_requirements': ['all4_stop', '400_B22_retention',
                                             'complete_native_case'],
                'source_scope': 'all returned native slots and completed-control prefix archived; unsafe pending control does not qualify'}
    states = episode['states']
    q, v = states['qpos'], states['qvel']
    n = len(episode['rows'])
    side_end = cycle['command']['side_end_control']
    stop_index = cycle['command']['stop_control_index']
    require(side_end is not None and stop_index is not None
            and 200 < stop_index < side_end and n-side_end == 400
            and episode['side_controls'] == side_end-200
            and cycle['archive_completed'] is True,
            'C35 complete side/stop/400-retention archive differs')
    side = episode['side_rows']
    body = episode['body_reference']
    command_proof = verify_case_command35(side, definition, cycle)
    require(body is not None and body['controls'] == len(side)
            and body['first_stop_control_index'] == stop_index,
            'C35 body Hermite command/stop receipt differs')
    actual_events = [event for row in side for event in
                     row['info']['controller_record']['controller_record35']['pair_events']]
    require(actual_events == cycle['pair_events']
            and len(actual_events) <= 16
            and all(event['event'] == 'pair_completed' for event in actual_events),
            'C35 pair exchanges differ from consumed controller references')
    margin, loaded, pending = [], [], []
    for i, row in enumerate(side):
        five_proofs = episode['side_native'][5*i:5*i+5]
        require(len(five_proofs) == 5, 'C35 stop/control native evidence incomplete')
        proof = five_proofs[-1]
        static = proof.get('static_margin')
        margin.append(np.nan if static is None else static['static_margin_m'])
        loaded.append(bool(all(np.all(p['contact']['positive_wheel_normal_load_sum_n'] > 1e-8)
                               and all(p['contact']['effective_terrain_wheel_contact_mask'])
                               for p in five_proofs)))
        ref = row['info']['controller_record']['controller_record35']['reference']
        pending.append(bool(ref['swing_legs']))
    stop = stop_metrics35(q[200:side_end+1], v[200:side_end+1],
        stop_request_control=stop_index-200, handoff_control=side_end-200,
        support_margin_by_control=margin,
        all_four_loaded_by_control=loaded, pending_swing_by_control=pending)
    torques = np.asarray([r['after']['actuator_force'] for r in episode['native']], dtype=float)
    need_limits = LIMITS
    require(torques.shape == (5*n, 16), 'C35 full native actuator torque archive incomplete')
    norm2 = np.mean((torques/need_limits)**2, axis=1)
    mean = float(np.mean(norm2))
    integral = float(np.sum(norm2)*.002)
    kind, direction = definition['kind'], definition['direction']
    denominator = (abs(body['final_signed_command_integral_m'])
                   if kind == 'continuous' else None)
    cost = torque_cost35(full_scene_mean=mean,
        full_scene_integral_s=integral, kind=kind,
        commanded_distance_m=denominator,
        c33_same_direction_integral_s=(baseline_integral if kind == 'continuous' else None))
    _, left = entry_heading35(q[200])
    signed = direction*((q[:, :2]-q[200, :2])@left)
    final_xy_error = float(np.linalg.norm(q[side_end, :2]-body['final_body_reference_xy_m']))
    rpy0 = scalar_state(q[200], v[200], binding)['rpy']
    rpy1 = scalar_state(q[side_end], v[side_end], binding)['rpy']
    yaw_error = abs(math.atan2(math.sin(rpy1[2]-rpy0[2]),
                               math.cos(rpy1[2]-rpy0[2])))
    rollback = (float(np.max(signed[side_end]-signed[side_end:]))
                if direction else float(np.max(np.linalg.norm(
                    q[side_end:, :2]-q[side_end, :2], axis=1))))
    retention = (float(signed[-1]/signed[side_end])
                 if kind == 'continuous' and signed[side_end] > 0. else None)
    steady = continuous_metrics35(q[200:side_end+1], direction) if kind == 'continuous' else None
    online = cycle['result']['task']
    require(isinstance(online, dict), 'C35 complete case lacks online task receipt')
    positive_native, _ = _native_loads31(episode['native'], 1e-8)
    tail = _tail31(episode['rows'], states, positive_native,
                   n-100, n, binding, geometry, GATES31)
    entry_state = scalar_state(q[200], v[200], binding)
    entry_geometry = compiled_side_geometry(kin, q[200], v[200], binding['wheel_map'])
    entry_loads = episode['native_proofs'][5*200-1]['contact']['positive_wheel_normal_load_sum_n']
    entry_checks = {'exact_200_preparation': episode['side_rows'][0]['control_index'] == 200,
        'roll_pitch': bool(np.max(np.abs(entry_state['rpy'][:2])) <= .12),
        'origin_speed': bool(np.linalg.norm(entry_state['origin_velocity']) < .04),
        'angular_speed': bool(np.linalg.norm(entry_state['world_angular_velocity']) < .08),
        'foot_plane': bool(np.max(np.abs(entry_geometry['contact_points_m'][:, 2])) <= .01),
        'four_actual_positive_contacts': bool(np.all(entry_loads > 1e-8))}
    entry_passed = bool(all(entry_checks.values()))
    require(online['tail'] == tail and online['entry']['checks'] == entry_checks
            and online['entry']['passed'] is entry_passed,
            'C35 original B22 retention/entry gates differ from saved states/native')
    checks = {'stop': stop['passed'], 'mean_effort': cost['mean_gate_passed'],
              'rollback': rollback <= .005, 'final_yaw': yaw_error < .12,
              'original_tail': tail['passed'], 'entry': entry_passed}
    if kind != 'cancel':
        checks.update(final_xy=final_xy_error < .012,
                      pair_exchanges=len(actual_events) >= (2 if direction == 0 else 4))
    if kind == 'continuous':
        checks.update(actual_speed=steady['mean_signed_origin_vy_mps'] >= .020,
            tracking=steady['rms_steady_tracking_error_mps'] <= .010,
            sustained=steady['fraction_steady_signed_vy_ge_010'] >= .8,
            longitudinal=steady['max_abs_longitudinal_excursion_active_m'] <= .03,
            retention=retention is not None and retention >= .9,
            command_integral=abs(denominator-.150) <= 1e-10,
            distance_efficiency=cost['distance_efficiency_gate'] is True)
    online_checks = {key: value for key, value in checks.items() if key != 'stop'}
    online_checks['stop_time'] = stop['stop_delay_controls'] <= 250
    online_checks['stop_distance'] = stop['maximum_lateral_excursion_after_stop_request_m'] <= .03
    require(online['checks'] == online_checks
            and close(online['full_tau2_mean'], mean)
            and close(online['full_tau2_integral_s'], integral)
            and close(online['full_elapsed_s'], n*.01)
            and close(online['final_xy_error_m'], final_xy_error)
            and close(online['final_yaw_error_rad'], yaw_error)
            and close(online['max_rollback_m'], rollback)
            and online['stop_to_handoff_controls'] == stop['stop_delay_controls']
            and close(online['max_distance_after_stop_m'],
                      stop['maximum_lateral_excursion_after_stop_request_m'])
            and same_optional35(online['commanded_distance_cost_denominator_m'], denominator)
            and online['distance_efficiency_gate'] == cost['distance_efficiency_gate']
            and online['pair_exchanges'] == len(actual_events),
            'C35 online task/cost differs from saved physical reconstruction')
    if steady is not None:
        require(close(online['mean_signed_origin_vy_mps'], steady['mean_signed_origin_vy_mps'])
                and close(online['rms_steady_tracking_error_mps'], steady['rms_steady_tracking_error_mps'])
                and close(online['fraction_steady_signed_vy_ge_010'],
                          steady['fraction_steady_signed_vy_ge_010'])
                and close(online['max_abs_longitudinal_excursion_m'],
                          steady['max_abs_longitudinal_excursion_active_m'])
                and close(online['retained_fraction'], retention)
                and close(online['tau2_integral_per_commanded_meter'],
                          cost['tau2_integral_per_commanded_meter']),
                'C35 online steady speed/quality differs from saved origin states')
    require(online['eligible'] is all(online_checks.values()),
            'C35 online eligibility differs from independently reconstructed gates')
    passed = bool(all(checks.values()) and online['eligible'] is True
                  and cycle['terminal_kind'] == 'success')
    return {'case_id': definition['id'], 'record_auditable': True,
            'physical_task_qualified': passed, 'partial_record': False,
            'checks': checks, 'steady': steady, 'stop': stop, 'cost': cost,
            'final_xy_error_m': final_xy_error, 'final_yaw_error_rad': yaw_error,
            'retention_fraction': retention, 'max_rollback_m': rollback,
            'pair_exchanges': len(actual_events),
            'command_proof': command_proof,
            'scope': 'new two-support fixed pilot; no inherited C34 tripod/RL/GUI qualification'}


def read_run35(run: Path, go_path: Path) -> dict:
    run, go_path = run.resolve(strict=True), go_path.resolve(strict=True)
    session, worker, _go = sources35(run, go_path)
    construction, binding, geometry, kin, side = construction35(run, session)
    definitions = saved_document(run/'case_definition35.json')
    require(definitions['schema'] == 'd1-c35-case-definition-v1'
            and definitions['cases'] == session['cases'],
            'C35 preregistered case definitions differ')
    attempted = worker['result']['attempted_cases']
    require(1 <= attempted <= 5, 'C35 actual campaign attempt count differs')
    baseline = saved_document(run/'baseline35.json')
    reports = []
    episodes = []
    offset = 0
    for index in range(attempted):
        definition = session['cases'][index]
        episode = read_episode35(run, index, offset, session,
            binding, geometry, kin, side, construction['skill30_joint_limits'])
        episodes.append(episode)
        cycle = episode['cycle']
        validate_cycle35_shape(cycle)
        case = case_evidence35(episode, definition,
            baseline['right' if definition['direction'] < 0 else 'left'],
            binding, geometry, kin)
        reports.append(case)
        offset += episode['closed_prefix_controls']
        require(saved_document(run/f'case35_{index:02d}.json') == cycle['result'],
                'C35 case online result differs from cycle receipt')
        if not episode['fatal_partial']:
            require(saved_document(run/f'episode_{index}/entry35.json')
                    == cycle['result']['task']['entry'],
                    'C35 saved entry evidence differs from independently checked task')
        if not case['physical_task_qualified']:
            require(index == attempted-1,
                    'C35 continued physics after first failed case')
    ledger = check_ledger35(worker, episodes, session, construction)
    require(offset == worker['result']['controls']
            and sum(case['pair_exchanges'] for case in worker['result']['cases'])
                == worker['result']['pair_exchanges']
            and worker['result']['normal_native'] == worker['C_final']['control_returns'],
            'C35 global control/native ledger differs')
    qualification = len(reports) == 5 and all(r['physical_task_qualified'] for r in reports)
    return {'schema': 'd1-c35-independent-saved-readback-v1',
            'run': str(run), 'execution_contract_id': CONTRACT,
            'source_GO_identity': identity(go_path),
            'worker_receipt_identity': identity(run/'worker_receipt.json'),
            'record_auditable': True, 'pilot_fixed_feasible': qualification,
            'qualification': qualification, 'cases': reports,
            'ledger': ledger,
            'attempted_cases': attempted, 'completed_controls': offset,
            'actual_normal_native': worker['result']['normal_native'],
            'partial_record': any(r['partial_record'] for r in reports),
            'new_contact_semantics': 'per-native named pair or all4 actual positive loads; no flight; full contact wrench/stop margin independent',
            'dynamics_scope': 'independent FK/COM, joint target FK error, reference, wrench/allocation, PD/cache multiplication/brake/clip and native actuator chain; sealed Jacobian/bias/fullM caches not independently recomputed',
            'reader_model_calls': 0, 'reader_physics_calls': 0,
            'RL_speed_benefit_proven': False, 'user_goal_complete': False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source-go', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = read_run35(args.run, args.source_go)
        code = 0
    except BaseException as error:
        report = {'schema': 'd1-c35-independent-saved-readback-v1',
                  'record_auditable': False, 'qualification': False,
                  'reader_error': {'type': type(error).__name__, 'message': str(error),
                                   'traceback': traceback.format_exc()},
                  'reader_model_calls': 0, 'reader_physics_calls': 0}
        code = 2
    with args.output.open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False,
                  default=lambda value: value.tolist() if isinstance(value, np.ndarray)
                  else value.item() if isinstance(value, np.generic)
                  else (_ for _ in ()).throw(TypeError(type(value).__name__)))
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return code


if __name__ == '__main__':
    raise SystemExit(main())
