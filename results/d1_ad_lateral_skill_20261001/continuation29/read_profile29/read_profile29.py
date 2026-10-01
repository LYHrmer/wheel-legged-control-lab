"""C29 saved-only profile readback; no complete-cycle or GUI qualification claim."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

W = Path(__file__).resolve().parents[2]
P = W/'continuation29'
P27 = W/'continuation27'
R = Path('/home/lyh/wheel-legged-control-lab')
BUNDLE = R/'runtime/d1_b22/research'
sys.path.insert(0, str(W/'continuation26/root26'))
from reader26 import (require, identity, document, saved_document, saved_rows,
                      check_binding, geometry_map, close, MODEL_SHA)
sys.path.insert(0, str(P27/'read27'))
from side_math27 import LIMITS
from records29 import check_episode
from ledger29 import frozen_profile, check_ledgers
from archive29 import check_archive29
from timing29 import verify_timing
from input_reader27 import check_preparation
import numpy as np

INNER = 'C27_AD_teacher_diagnostic_v1'
OUTER = 'C29_AD_pipeline_profile_v1'
INITIAL_KEYS = ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart', 'observation', 'time')

def check_origins(run, session):
    sources = session['source_hashes']
    origins = saved_document(run/'loaded_origins_final.json')
    require(all(name in origins['module_origins'] for name in ('torch', 'numpy', 'stable_baselines3', 'gymnasium'))
            and all(path in sources for path in origins['module_origins'].values())
            and all(row['identity_verified'] is True and row['defining_source'] in sources
                    for row in origins['synthetic_module_aliases'].values()),
            'actual numerical import origins unbound')
    relative = {'input23': 'continuation23/gui23/input23.py', 'controller18': 'continuation18/controller18.py',
        'residual18': 'continuation18/residual18.py', 'world_upright_course_11': 'upright11/world_upright_course_11.py',
        'full_drive_env_08': 'course_impl08/full_drive_env_08.py', 'gui13_bridge': 'continuation13/gui13_revision02/gui13_bridge.py',
        'async_course_renderer_12': 'gui12/async_course_renderer_12.py', 'latest_frame_mailbox_12': 'gui12/latest_frame_mailbox_12.py',
        'rl16_learning_11': 'rl11/rl16_learning_11.py',
        'run_rl16_training_08': 'run_rl16_training_08.py', 'state21': 'continuation21/state21.py',
        'worker22': 'continuation22/worker22.py', 'kinematics24': 'continuation24/kinematics24.py'}
    expected = {k: str(BUNDLE/v) for k, v in relative.items()}
    expected.update(run_gui29=str(P/'root29/run_gui29.py'), input27=str(P27/'sol27/input27.py'),
        hybrid_adapter27=str(P27/'sol27/hybrid_adapter27.py'), side_runtime27=str(P27/'root27/side_runtime27.py'),
        d1_fast_side_step=str(R/'scripts/d1_fast_side_step.py'), d1_side_step=str(R/'scripts/d1_side_step.py'),
        d1_course_side_intent=str(R/'scripts/d1_course_side_intent.py'),
        engine_binding=session['binding_module'])
    runtime = saved_document(run/'runtime_module_origins.json')
    require(document(run/'runtime_module_origins_before.json') == runtime == expected and all(path in sources for path in runtime.values()),
            'actual side/rolling modules left reviewed frozen origins')
    for name in ('mapped_libraries_before_load.json', 'mapped_libraries_final.json'):
        paths = saved_document(run/name)['paths']
        require(paths and all(path in sources for path in paths), 'actual DSO mapping unbound')



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
            and result['segment_receipts'] == [episode['guard']]
            and {p.name for p in run.glob('episode_*') if p.is_dir()} == {'episode_0'}
            and result['input_reset_used'] is False, 'C27 reset/segment/actual control set differs')
    return episode


def check_sources(run, session, worker, spec):
    require(session['execution_contract_id'] == spec['inner_execution_contract_id'] == INNER
            and session['outer_execution_contract_id'] == spec['outer_execution_contract_id'] == OUTER
            and session['schema'] == spec['worker_numeric_schema'] == 'd1-c23-gui-worker-v1'
            and session['arm'] == spec['case_id'] == 'profile_left'
            and session['qualification'] is spec['qualification'] is False
            and session['retry_permitted'] is False and session['actor'] == 'B'
            and session['controller_variant'] == 'combined' and session['profile'] == 'yaw_1p2'
            and session['side_arm'] == spec['side_arm'] == 'teacher'
            and session['direction'] == spec['direction'] == 1
            and session['seed'] == spec['seed'] == 271001
            and session['mode'] == spec['mode'] == 'keyboard' and session['render'] is True,
            'C29 source/profile/identity differs')
    expected = {'control_limit': 500, 'frame_limit': 500, 'soft_s': 30, 'close_s': 60, 'hard_s': 75}
    require(all(session[k] == spec[k] == v for k, v in expected.items()), 'C29 fixed reservation differs')
    require(worker['schema'] == 'd1-c23-gui-worker-receipt-v1'
            and worker['execution_contract_id'] == INNER and worker['arm'] == session['arm']
            and worker['session_identity'] == identity(run/'session.json')
            and worker['qualification'] is False and worker['diagnostic_only'] is True
            and worker['failure'] is None and worker['warnings'] == []
            and worker['archive_failed'] is False and worker['writer_partial'] == [],
            'worker/source/archive error prevents a valid profiling record')
    host, supervisor = document(run/'host_receipt.json'), document(run/'supervisor_receipt.json')
    require(host['arm'] == session['arm'] and host['exit_code'] == 0 and host['failure'] is None
            and host['source_mismatches'] == [] and host['owned_no_orphans'] is True
            and host['reservation_closed'] is True and host['reserved_controls'] == 500
            and host['retry_permitted'] is False and supervisor['exit_code'] == 0
            and supervisor['failure'] is None and supervisor['cleanup']['remaining'] == {}
            and supervisor['outer_limit_s'] == 600 and supervisor['elapsed_s'] <= 600,
            'profile host/supervisor did not close its actual reservation')
    go_path = Path(session['source_go_path'])
    go = document(go_path)
    require(identity(go_path) == session['source_go_identity'] and go['decision'] == 'GO'
            and go['outer_execution_contract_id'] == OUTER
            and all(session.get(k) == v for k, v in go['arms'][session['arm']].items())
            and all(session.get(k) == v for k, v in go['shared_session'].items())
            and go['worker'] == session['argv'][0] == str(P/'root29/run_gui29.py'),
            'C29 session differs from its pre-execution source GO')
    sources = session['source_hashes']
    require(all(sources.get(k) == v for k, v in go['inputs'].items()), 'GO closure missing from session')
    for name, expected in sources.items():
        path = Path(name)
        require(path.is_absolute() and not path.is_symlink() and identity(path) == expected,
                'frozen source/payload differs: '+name)
    critical = [P/'spec29.json', P/'profile_contract_29.md', P/'root29/run_gui29.py',
                P27/'root27/side_runtime27.py', P27/'sol27/hybrid_adapter27.py',
                P27/'sol27/input27.py', R/'scripts/d1_fast_side_step.py', R/'scripts/d1_side_step.py']
    critical += list((P/'read_profile29').glob('*.py'))
    require(all(str(p) in sources for p in critical), 'profile reader/physical source absent from GO')
    pure = {}
    for name in ('reader26', 'reader26_deferred', 'side_math27', 'input_reader27', 'kinematics24',
                 'verify_control18', 'verify_course_e_08_03', 'verify_rl16_training_08',
                 'records29', 'ledger29', 'archive29', 'timing29'):
        path = Path(sys.modules[name].__file__).resolve()
        bound = path if str(path) in sources else BUNDLE/path.relative_to(W)
        require(sources.get(str(bound)) == identity(path), 'actual pure dependency differs: '+name)
        pure[name] = {'actual_path': str(path), 'frozen_path': str(bound), 'identity': identity(path)}
    original_path = Path(session['checkpoint_relocation']['original_manifest_path'])
    original = document(original_path)
    relocation, manifest = session['checkpoint_relocation'], session['checkpoint_manifest']
    require(identity(original_path) == relocation['original_manifest_identity'] == sources.get(str(original_path))
            and relocation['changed_fields'] == ['folder'] and original['folder'] == relocation['original_folder']
            and relocation['current_folder'] == session['checkpoint_folder']
            and manifest == {**original, 'folder': session['checkpoint_folder']}, 'B22 folder-only relocation differs')
    for name, expected in manifest['files'].items():
        require(identity(Path(session['checkpoint_folder'])/name) == expected, 'B22 payload differs: '+name)
    require(manifest['files']['final_model.zip']['sha256'] == session['checkpoint_sha256'] == MODEL_SHA,
            'frozen B22 identity differs')
    load = saved_document(run/'strict_load.json')
    require(load['probe_rows'] == 32 and load['probe_actions_byte_exact'] is True
            and load['matches_training_final_policy_state'] is True
            and load['verified_without_engine_or_reset'] is True
            and load['observation_space']['shape'] == [99] and load['action_space']['shape'] == [16],
            'strict B22 load/probe differs')
    return go, pure


def read_run(run):
    run = Path(run).resolve(strict=True)
    session = document(run/'session.json')
    worker = saved_document(run/'worker_receipt.json')
    spec = document(P/'spec29.json')
    go, pure = check_sources(run, session, worker, spec)
    result = worker['result']
    episode = construction_and_records(run, session, result)
    side = check_ledgers(result, session, episode, spec)
    check_origins(run, session)
    archive = check_archive29(run, session, worker, episode)
    timing_path = run/'timing29.json'
    require(worker['timing29_identity'] == identity(timing_path), 'worker/timing payload differs')
    timing = saved_document(timing_path)
    spans = verify_timing(timing, result, episode['rows'])
    inputs = list(saved_rows(run/'input_snapshots.jsonl.gz'))
    decisions = list(saved_rows(run/'owner_decisions.jsonl.gz'))
    require(worker['input_polls'] == len(inputs) and worker['owner_decisions'] == len(decisions),
            'actual input/preparation archive count differs')
    preparation = check_preparation(session, inputs, decisions, episode['rows'])
    performance = saved_document(run/'performance.json')
    require(performance['active_start_ns'] == result['active_start_ns']
            and performance['active_end_ns'] == result['active_end_ns']
            and 0 < performance['render_api_calls'] <= 500
            and len([s for s in timing['main_spans'] if s['stage'] == 'render']) == performance['render_api_calls']
            and len([s for s in timing['main_spans'] if s['stage'] == 'poll']) == len(inputs)
            and performance['poll_timestamps_ns'] == [r['poll']['wall_ns'] for r in inputs],
            'actual main render/poll call counts differ from timed spans')
    # This profile deliberately ends before landing/handoff/retention/W/stop.
    # Preserve the full prefix evidence without applying or weakening those gates.
    n = episode['controls']
    return {'schema': 'd1-c29-independent-profile-readback-v1', 'execution_contract_id': OUTER,
            'numerical_protocol_contract_id': INNER, 'run': str(run), 'qualification': False,
            'record_valid': True, 'profile_record_valid': True, 'full_profile_budget_completed': n == 500,
            'task_qualified': False, 'GUI_qualified': False, 'complete_cycle_or_safe_stop_claim': False,
            'source_and_actual_origins_verified': True,
            'session_identity': identity(run/'session.json'), 'worker_receipt_identity': identity(run/'worker_receipt.json'),
            'source_go_identity': session['source_go_identity'], 'reader_source_identity': identity(Path(__file__).resolve()),
            'timing_identity': identity(timing_path), 'pure_reader_dependencies': pure,
            'completed_controls': n, 'normal_native_returns': 5*n, 'compiler_native_returns': 2,
            'actual_B22_predictions': episode['policy_predictions'], 'actual_traditional_side_controls': episode['side_controls'],
            'side_access': side, 'model_calls': result['model_calls'], 'archive_closure': archive,
            'side_math_summary': episode['side_math_summary'], 'input_preparation': preparation,
            'input_event_plan': go['x11_events_by_arm'][session['arm']],
            'timing': spans, 'worker_stop_reason': result['stop_reason'],
            'last_actor': episode['rows'][-1]['actor'],
            'last_side_status': result['side_status'],
            'training_controls': 0, 'optimizer_steps': 0, 'reader_model_calls': 0, 'reader_physics_calls': 0,
            'render_visual_fullcycle_gates_not_applied': True,
            'scope': 'one bounded profiling prefix; saved state/native/torque arithmetic verified; not full-step performance, complete skill, learned improvement, or hardware evidence'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'exclusive profile output already exists')
    report = read_run(args.run)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == '__main__':
    main()
