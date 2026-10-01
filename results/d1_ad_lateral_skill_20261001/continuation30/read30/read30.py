"""C30 independent saved headless skill readback; no model/physics imports."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

W = Path(__file__).resolve().parents[2]
P = W/'continuation30'
R = Path('/home/lyh/wheel-legged-control-lab')
BUNDLE = R/'runtime/d1_b22/research'
sys.path.insert(0, str(W/'continuation26/root26'))
from reader26 import require, close, identity, document, saved_document, check_binding, geometry_map, MODEL_SHA
sys.path.insert(0, str(W/'continuation27/read27'))
from side_math27 import LIMITS
from records27 import check_episode
from reader27 import frozen_profile, check_model_calls
from reference_read30 import verify_latches30, native_geometry30, verify_reference30
from reward_read30 import verify_macro_receipts30
from score_read30 import score_skill30
import numpy as np

CONTRACT = 'C30_lateral_skill_feasibility_v1'
INITIAL_KEYS = ('qpos','qvel','act','ctrl','qacc_warmstart','observation','time')

def construction_and_records(run, session, result):
    construction = saved_document(run/'construction_receipt.json')
    proof = construction['proof']
    require(proof['passed'] is True and proof['dso_path'] == session['library']
            and proof['dso_sha256'] == session['source_hashes'][session['library']]['sha256']
            and len(proof['jump_slots']) == 4 and all(r['passed'] is True for r in proof['jump_slots'])
            and construction['nominal_cache']['misses'] == 1
            and construction['C_state']['construction_attempts'] == construction['C_state']['construction_returns'] == 2,
            'cold two-compiler/engine binding proof differs')
    binding = check_binding(construction['geometry_binding'])
    geometry = geometry_map(construction['compiled_geometry'], binding)
    reference_path = Path(session['reference_construction_path'])
    require(str(reference_path) in session['source_hashes'], 'C22 reference construction absent from source closure')
    reference = document(reference_path)
    require(construction['geometry_binding'] == reference['geometry_binding']
            and construction['compiled_geometry'] == reference['compiled_geometry'],
            'C27 altered the frozen C22 compiled robot/course geometry')
    side_binding = construction['side_kinematic_binding']
    require({k: v for k, v in side_binding.items() if k not in ('kinematics24', 'dynamics24', 'solver24')}
            == construction['geometry_binding'], 'side FK binding differs from actual compiled course binding')
    kin = side_binding['kinematics24']
    for name in ('geom_bodyid', 'geom_type', 'geom_size', 'geom_contype', 'geom_conaffinity', 'body_mass'):
        require(close(kin[name], construction['geometry_binding'][name]), 'side compiled-tree array differs: '+name)
    controls_range = np.asarray(side_binding['dynamics24']['actuator_ctrlrange'])[binding['actuator_ids']]
    require(close(controls_range, np.stack((-LIMITS, LIMITS), axis=1)), 'actual side motor limits differ')
    require(construction['side_profile'] == frozen_profile(R/'scripts/d1_fast_side_step.py'),
            'actual teacher configuration differs from frozen FAST_PROFILE')
    episode = check_episode(run, session, binding, geometry, kin)
    require(set(episode['initial']) == set(INITIAL_KEYS)
            and episode['controls'] == result['completed_controls']
            and {p.name for p in run.glob('episode_*') if p.is_dir()} == {'episode_0'}, 'C30 reset/segment/actual control set differs')
    return episode, binding, geometry, kin, construction


def sources30(run, session, worker):
    require(session['execution_contract_id'] == CONTRACT and session['render'] is False
            and session['control_limit'] == 2200 and session['side_arm'] == 'teacher'
            and session['spawn_position_m'] == [-8., -4.7, .455]
            and (session['soft_s'], session['close_s'], session['hard_s']) == (240, 270, 300)
            and session['retry_permitted'] is False,
            'C30 finite headless session differs')
    go_path = Path(session['source_go_path'])
    go = document(go_path)
    require(identity(go_path) == session['source_go_identity'] and go['decision'] == 'GO'
            and go['execution_contract_id'] == CONTRACT
            and all(session.get(k) == v for k, v in go['shared_session'].items())
            and all(session.get(k) == v for k, v in go['arms'][session['arm']].items()),
            'session is not its frozen single-case source GO')
    require(worker['schema'] == 'd1-c30-headless-worker-v1' and worker['execution_contract_id'] == CONTRACT
            and worker['session_identity'] == identity(run/'session.json')
            and worker['failure'] is None and worker['cleanup_errors'] == [] and worker['warnings'] == []
            and worker['archive_failed'] is False and worker['execution_complete'] is True
            and worker['training_controls'] == worker['optimizer_steps'] == 0,
            'software/source/archive failure prevents complete qualification readback')
    host, supervisor = document(run/'host_receipt.json'), document(run/'supervisor_receipt.json')
    require(host['exit_code'] == 0 and host['failure'] is None and host['source_mismatches'] == []
            and host['owned_no_orphans'] is True and host['reservation_closed'] is True
            and host['reserved_controls'] == 2200 and host['retry_permitted'] is False
            and supervisor['exit_code'] == 0 and supervisor['failure'] is None
            and supervisor['cleanup']['remaining'] == {} and supervisor['outer_limit_s'] <= 900
            and supervisor['elapsed_s'] <= supervisor['outer_limit_s'], 'actual source host did not close successfully')
    sources = session['source_hashes']
    require(all(sources.get(path) == expected for path, expected in go['inputs'].items()), 'session source closure differs')
    before = document(run/'runtime_module_origins_before.json')
    final = saved_document(run/'runtime_module_origins.json')
    require(before == final and {'worker30','runtime30','record30','macro30','side_skill30',
            'hybrid_adapter27','side_runtime27','d1_fast_side_step','d1_side_step','engine_binding',
            'controller18','kinematics24','world_upright_course_11'} <= set(final),
            'actual model-before and final production origin chain differs')
    checked = {}
    for name, value in final.items():
        path = Path(value['path'])
        expected = {k: value[k] for k in ('bytes','sha256')}
        require(sources.get(str(path)) == identity(path) == expected, 'actual runtime source changed: '+name)
        checked[name] = expected
    modules = ('reader26','reader26_deferred','reader27','records27','side_math27','score27',
               'kinematics24','verify_control18','verify_course_e_08_03','verify_rl16_training_08',
               'reference_read30','geometry_read30','reward_read30','score_read30')
    for name in modules:
        actual = Path(sys.modules[name].__file__).resolve()
        bound = actual if str(actual) in sources else BUNDLE/actual.relative_to(W)
        require(sources.get(str(bound)) == identity(actual), 'actual independent reader differs: '+name)
        checked[name] = identity(actual)
    require(sources.get(str(Path(__file__).resolve())) == identity(Path(__file__).resolve()), 'actual C30 reader not frozen')
    loaded = saved_document(run/'loaded_origins_final.json')
    require(all(path in sources for path in loaded['module_origins'].values())
            and all(v['identity_verified'] is True and v['defining_source'] in sources
                    for v in loaded['synthetic_module_aliases'].values()), 'numerical import identity unbound')
    for name in ('mapped_libraries_before_load.json', 'mapped_libraries_final.json'):
        paths = saved_document(run/name)['paths']
        require(paths and all(path in sources for path in paths), 'actual mapped DSO lacks frozen identity')
    relocation, manifest = session['checkpoint_relocation'], session['checkpoint_manifest']
    path = Path(relocation['original_manifest_path'])
    original = document(path)
    require(identity(path) == relocation['original_manifest_identity'] == sources[str(path)]
            and relocation['changed_fields'] == ['folder'] and original['folder'] == relocation['original_folder']
            and relocation['current_folder'] == session['checkpoint_folder']
            and manifest == {**original, 'folder': session['checkpoint_folder']}, 'checkpoint relocation changed numeric identity')
    for name, expected in manifest['files'].items():
        require(identity(Path(session['checkpoint_folder'])/name) == expected, 'checkpoint payload differs')
    load = saved_document(run/'strict_load.json')
    require(session['checkpoint_sha256'] == manifest['files']['final_model.zip']['sha256'] == MODEL_SHA
            and load['probe_rows'] == 32 and load['probe_actions_byte_exact'] is True
            and load['matches_training_final_policy_state'] is True
            and load['verified_without_engine_or_reset'] is True
            and load['observation_space']['shape'] == [99] and load['action_space']['shape'] == [16],
            'strict B22 probe/99D16D identity differs')
    return {'passed': True, 'runtime_and_reader_current_identities': checked,
            'full_source_pre_and_postcheck_evidence': 'successful saved host with source_mismatches=[]',
            'reader_full_historical_closure_rehash_performed': False}


def ledgers30(worker, session, episode, construction, events):
    result = worker['result']
    n = episode['controls']
    c, py, native = worker['C_final'], worker['python'], worker['native_guard']
    require(c['construction_attempts'] == c['construction_returns'] == 2
            and c['control_attempts'] == c['control_returns'] == 5*n
            and c['phase'] == c['target_model'] == c['target_data'] == c['violations'] == 0
            and c['native_construction_caller_verified'] is True
            and c['ccd_attempts'] == c['ccd_returns']
            and (c['ccd_attempts'] == 0 or c['native_ccd_caller_verified'] is True)
            and py['control_attempted'] == py['control_completed'] == n
            and py['native_attempted'] == py['native_returned'] == py['clock_advanced_substeps'] == 5*n
            and py['forbidden_entries'] == 0 and py['fatal_latched'] is False
            and native['native_attempted'] == native['native_returned'] == native['native_checked'] == 5*n
            and native['failure'] is None and result['control_step_caller_verified'] is True,
            'actual counted native/model construction chain does not close')
    check_model_calls({'model_calls': worker['model_calls'], 'policy_predictions': result['policy_predictions']},
                      session, episode['policy_predictions'])
    access, edges = worker['side_access'], worker['copy_edges']
    t, p, starts = access['computes'], access['prepares'], access['starts']
    require(t == episode['side_controls'] <= 1500 and p == len(events) <= 4
            and starts == 1 and access['rejected'] == 0 and access['within_actual_bounds'] is True
            and access['static_contact_is_not_native_load'] is True, 'actual side scopes exceed C30')
    bounds = dict(copy=t+4*p, forward=112*t+236*p, jacBody=96*t+192*p,
                  jac=4*t, fullM=t, objectVelocity=2*t+starts)
    hard = dict(copy=1516, forward=168944, jacBody=144768, jac=6000, fullM=1500, objectVelocity=3001)
    require(access['actual_T_P_bounds'] == bounds and access['limits'] == hard
            and set(access['counts']) == set(bounds), 'C30 extra-IK API bound schema differs')
    scopes = access['scopes']
    require(len(scopes) == starts+t and scopes[0]['kind'] == 'start'
            and all(s['kind'] == 'compute' for s in scopes[1:]), 'actual start/compute scopes differ')
    prior_p = 0
    for i, scope in enumerate(scopes):
        require(scope['start_count'] == 1 and scope['compute_count'] == i
                and prior_p <= scope['prepare_count'] <= p
                and scope['physical_arrays_unchanged'] is True and scope['native_boundary_unchanged'] is True
                and scope['static_ccd_returned'] is True and scope['failure'] is None
                and scope['static_ccd_delta']['ccd_attempts'] == scope['static_ccd_delta']['ccd_returns'] >= 0
                and scope['calls']['objectVelocity'] <= (1 if i == 0 else 2),
                'scratch API scope changed live arrays/integration or source identity')
        prior_p = scope['prepare_count']
    require(prior_p == p and all(access['counts'][k] == sum(s['calls'][k] for s in scopes)
                                and access['counts'][k] <= bounds[k] <= hard[k] for k in bounds),
            'actual static-query receipts do not add up to the registered bounds')
    side_rows = [r for r in episode['rows'] if r['actor'] == 'traditional_side']
    require([r['info']['controller_record']['access_scope_receipt'] for r in side_rows] == scopes[1:],
            'actual consumed side scope differs from native API ledger')
    addresses = construction['data_addresses']
    require(set(addresses) == {'live','measurement','side_scratch'} and len(set(addresses.values())) == 3
            and all(type(v) is int and v > 0 for v in addresses.values())
            and addresses['live'] == episode['reset']['data_address']
            and addresses['side_scratch'] == access['scratch_address']
            and edges['rejected'] == 0 and edges['live_to_measurement'] >= n
            and edges['measurement_to_side_scratch'] == access['counts']['copy']
            and set(edges) == {'live_to_measurement','measurement_to_side_scratch','rejected'}
            and result['pure_fk_native_calls'] == 5*n, 'three-data copy/actual native FK ledger differs')
    return {'passed': True, 'controls': n, 'normal_native': 5*n, 'compiler_native': 2,
            'side_compute_controls': t, 'leg_prepares': p, 'actual_static_API': access['counts'],
            'static_queries_are_not_integration_or_native_contact_load': True,
            'actual_native_FK_calls': result['pure_fk_native_calls'], 'actual_data_objects': 3}


def read_run(run):
    run = Path(run).resolve(strict=True)
    session, worker = document(run/'session.json'), document(run/'worker_receipt.json')
    source = sources30(run, session, worker)
    result = worker['result']
    episode, binding, geometry, kin, construction = construction_and_records(run, session, result)
    saved = saved_document(run/'episode_0/macro_transitions30.json')
    require(saved['schema'] == 'd1-c30-macros-v1' and saved['training'] is False
            and saved['elapsed_controls'] == episode['controls']
            and saved['direction'] == session['direction'], 'macro archive identity differs')
    events = saved['events']
    ledger = ledgers30(worker, session, episode, construction, events)
    latches = verify_latches30(episode, events, kin, binding, construction['skill30_joint_limits'], session['arm'])
    native_minima, native_contacts = native_geometry30(episode, kin, binding)
    reference = verify_reference30(episode, events, kin, binding, session['arm'], native_minima, native_contacts)
    task = score_skill30(episode, binding, geometry, kin, session['arm'], session['direction'],
                        document(W/'continuation27/spec27.json'))
    side_rows = [r for r in episode['rows'] if r['actor'] == 'traditional_side']
    finish = side_rows[-1]['info']['controller_record']['diagnostic_after']
    side_end = side_rows[-1]['control_index']+1
    retained = finish['done'] and episode['controls'] == side_end+400
    if retained and finish['failure'] == 'cancelled':
        reason = 'safe_cancel'
    elif retained and finish['status']['success'] and finish['failure'] is None:
        reason = 'success_after_retention'
    elif finish['failure'] is not None or any(r['terminated'] for r in episode['rows']) or not reference['horizontal_native_geometry_passed']:
        reason = 'physical_failure'
    else:
        reason = 'budget_truncated'
    require(saved['end_reason'] == reason, 'macro terminal reason differs from actual finish and retention')
    macros = verify_macro_receipts30(episode, events, saved['transitions'], session['direction'],
                                     binding['actuator_ids'], LIMITS, reason)
    require(saved['event_transition_count'] == result['actual_event_transitions'] == macros['actual_macro_transitions'],
            'actual macro transition count differs')
    full_keys = dict(progress='progress_potential_difference', backtrack='backtrack_normalized', elapsed_s='elapsed_s',
        longitudinal_square_m2_s='longitudinal_square_integral_m2_s',
        longitudinal_square_normalized_s='longitudinal_normalized_square_integral_s',
        lateral_goal_square_m2_s='lateral_goal_error_square_integral_m2_s',
        lateral_goal_square_normalized_s='lateral_goal_error_normalized_square_integral_s',
        yaw_square_rad2_s='yaw_error_square_integral_rad2_s',
        yaw_square_normalized_s='yaw_error_normalized_square_integral_s',
        actual_torque_square_normalized_s='torque_normalized_square_integral_s')
    require(set(saved['full_case_components']) == set(full_keys)
            and all(close(saved['full_case_components'][key], macros['full_scene_components'][value])
                    for key, value in full_keys.items())
            and close(saved['origin_xy'], episode['states']['qpos'][side_rows[0]['control_index'], :2])
            and close(saved['yaw0'], side_rows[0]['info']['controller_record']['diagnostic_before']['start_yaw_rad']),
            'full-scene reward or initial-heading normalization differs')
    is_cancel = session['arm'].startswith('cancel_')
    overlap_rows = [r['control_index'] for r in side_rows
                    if np.linalg.norm(r['skill30']['foot_reference_velocity_mps'][:2]) > 1e-10
                    and abs(r['skill30']['foot_reference_velocity_mps'][2]) > 1e-10]
    cancel_requests = [r['control_index'] for r in episode['rows'] if r['cancel_requested30']]
    cancel_gate = (bool(overlap_rows and cancel_requests == [overlap_rows[0]+1]
                        and result['cancel_control_index'] == cancel_requests[0]
                        and all(e['control_index'] <= cancel_requests[0] for e in events))
                   if is_cancel else not cancel_requests and result['cancel_control_index'] is None)
    active_nonzero = any(np.any(np.abs(e['consumed_action'][:2]) > 1e-12) for e in events)
    coverage = session['arm'].startswith('zero_') or active_nonzero or reference['actual_overlap_legs'] >= 1
    passed = bool(task['base_task_gates_passed'] and reference['horizontal_native_geometry_passed'] and coverage and cancel_gate)
    return {'schema': 'd1-c30-independent-saved-skill-readback-v1', 'execution_contract_id': CONTRACT,
            'run': str(run), 'case_id': session['arm'], 'record_valid': True, 'source': source,
            'ledger': ledger, 'task': task, 'reference': reference, 'latches': latches, 'macros': macros,
            'nonzero_body_permission_consumed': active_nonzero, 'effective_permission_coverage': bool(coverage),
            'future_training_overlap_threshold_only': reference['actual_overlap_legs'] >= 2,
            'cancel_trigger_or_absence_correct': cancel_gate,
            'qualification_passed': passed, 'GUI_qualified': False, 'RL_trained': False,
            'faster_than_zero_claim': False, 'hardware_evidence': False,
            'end_reason': reason, 'reader_model_calls': 0, 'reader_physics_calls': 0,
            'reader_source_identity': identity(Path(__file__).resolve()),
            'session_identity': identity(run/'session.json'), 'worker_receipt_identity': identity(run/'worker_receipt.json'),
            'scope': 'one headless 30 mm skill case; cross-case speed benefit and zero trajectory pairing require their separately authorized records'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'exclusive C30 readback already exists')
    report = read_run(args.run)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == '__main__':
    main()
