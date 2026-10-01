"""Saved-only diagnosis of failed C28; this cannot confer run qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

W = Path(__file__).resolve().parents[2]
R = Path('/home/lyh/wheel-legged-control-lab')
BUNDLE = R/'runtime/d1_b22/research'
sys.path[:0] = [str(W/'continuation27/read27'), str(W/'continuation26/root26')]
from reader26 import (require, close, identity, document, saved_document, saved_value,
                      saved_rows, committed, check_binding, geometry_map)
from records27 import check_episode
from score27 import score_case
from reader27 import check_model_calls, check_side_access, frozen_profile, MODEL_SHA
from reader26_deferred import check_owner_join_metadata
from input_reader27 import check_preparation, check_events, check_performance_data, _contexts
import numpy as np


def source_subset(session):
    sources = session['source_hashes']
    names = ('reader26', 'reader26_deferred', 'records27', 'score27', 'side_math27',
             'reader27', 'input_reader27', 'archive_reader27', 'kinematics24',
             'verify_control18', 'verify_course_e_08_03', 'verify_rl16_training_08')
    used = {}
    for name in names:
        path = Path(sys.modules[name].__file__).resolve()
        bound = path if str(path) in sources else BUNDLE/path.relative_to(W)
        actual = identity(path)
        require(sources.get(str(bound)) == actual, 'frozen diagnostic dependency differs: '+name)
        used[str(path)] = {'identity': actual, 'frozen_counterpart': str(bound)}
    for path in (W/'continuation27/spec27.json', R/'scripts/d1_fast_side_step.py',
                 Path(session['reference_construction_path'])):
        actual = identity(path)
        require(sources.get(str(path)) == actual, 'critical frozen input differs: '+str(path))
        used[str(path)] = {'identity': actual, 'frozen_counterpart': str(path)}
    return used


def render_diagnostic(run, session, worker, episode, deferred, go):
    rows = episode['rows']
    inputs = list(saved_rows(run/'input_snapshots.jsonl.gz'))
    decisions = list(saved_rows(run/'owner_decisions.jsonl.gz'))
    perf = saved_document(run/'performance.json')
    frames = saved_value(run/'render_frames.json')
    require(worker['input_polls'] == len(inputs) and worker['owner_decisions'] == len(decisions)
            and perf['render_frames'] == frames and worker['render_frames'] == len(frames),
            'saved input/render counts do not join')
    preparation = check_preparation(session, inputs, decisions, rows)
    events = check_events(run, session, go, inputs, decisions, rows)
    published = [{**row['metadata'], 'publication': row['publication']}
                 for row in perf['published_frame_snapshots']]
    require([r['publication']['frame_seq'] for r in published] == list(range(1,len(published)+1)),
            'saved frame publications are not contiguous')
    contexts = _contexts(rows)
    for meta in published:
        tick = meta['episode_tick']
        require(meta['episode_id'] == 0 and meta['control_index'] == tick
                and 0 <= tick <= episode['controls']
                and close(meta['sim_time_s'], episode['states']['time'][tick]),
                'published frame detached from actual state endpoint')
        h = hashlib.sha256()
        for name in ('qpos','qvel','act','ctrl','qacc_warmstart','observation','time'):
            value = np.asarray(episode['states'][name][tick])
            h.update(name.encode()+b'\0'+str(value.dtype).encode()+b'\0'
                     +str(value.shape).encode()+b'\0'+value.tobytes(order='C'))
        require(meta['state_sha256'] == h.hexdigest()
                and meta['state_hash_protocol'] == 'fields+NUL,dtype+NUL,shape+NUL,C_bytes'
                and meta['side_active'] is contexts[tick]['side_active'],
                'saved frame state/hash/side owner differs')
    require(0 < perf['render_api_calls'] <= session['frame_limit'] == 2200
            and perf['poll_timestamps_ns'] == [r['poll']['wall_ns'] for r in inputs],
            'actual draw/poll counts differ')
    start, end = deferred['active_start_ns'], deferred['active_end_ns']
    result = check_performance_data(run, episode['controls'], start, end,
                                   perf['poll_timestamps_ns'], frames, published)
    return {'preparation': preparation, 'events': events, 'performance': result,
        'render_api_calls': perf['render_api_calls'], 'published_frames': len(published),
        'actual_active_wall_s': (end-start)/1e9,
        'active_window_source': 'deferred_archive_receipt.json:active_start_ns/active_end_ns',
        'original_performance_active_start_ns': perf['active_start_ns'],
        'original_performance_active_end_ns': perf['active_end_ns'],
        'original_null_fields_preserved': True,
        'not_verified_without_worker_result': ['mailbox total/copy edges', 'pause_render_proofs',
                                              'owner stop_reason and final timeline'],
        'formal_input_render_qualification': False}


def archive_diagnostic(run, session, worker, episode, receipt):
    ready = document(run/'preflight_ready.json')
    require(ready['session_sha256'] == identity(run/'session.json')['sha256'], 'preflight/session differs')
    owner_join = check_owner_join_metadata(worker['owner_join'], ready, session, receipt)
    n = episode['controls']
    require(receipt['failure'] is None and receipt['pending_count'] == 0
            and receipt['control_rows'] == n and receipt['native_rows'] == 5*n
            and receipt['episode_count'] == 1
            and receipt['active_start_ns'] < receipt['active_end_ns']
                <= receipt['flush_start_ns'] <= receipt['flush_end_ns'],
            'actual deferred archive boundary is incomplete')
    queued, flushed = receipt['queued_jobs'], receipt['flushed_jobs']
    require(1 <= len(queued) == len(flushed) == receipt['peak_jobs'] <= receipt['max_jobs'] == 64,
            'deferred transaction count differs')
    require([r['id'] for r in queued] == sorted({r['id'] for r in queued})
            == [r['id'] for r in flushed], 'deferred jobs reordered/duplicated')
    payloads = set()
    for a,b in zip(queued,flushed):
        paths = [Path(p) for p in a['payload_paths']]
        require(paths and all(p.is_absolute() and p.is_relative_to(run) and not p.is_symlink() for p in paths),
                'unsafe deferred payload')
        manifest = paths[0].with_name(paths[0].name+'.manifest.json')
        require(a['manifest_name'] == b['manifest_name'] == manifest.name
                and identity(manifest) == b['actual_manifest_identity'], 'deferred manifest identity differs')
        actual = committed(paths[0])
        require([r['file'] for r in actual['payloads']] == [p.name for p in paths]
                and not payloads.intersection(paths), 'deferred payload membership differs')
        payloads.update(paths)
        require(all(str(p) in worker['writer_committed'] for p in (*paths,manifest)),
                'payload not present in actual committed writer ledger')
    required = {run/'episode_0'/name for name in ('reset.json','initial_state.npz','states.npz',
                                                   'controls.jsonl.gz','segment_receipt.json')}
    required.update(run/'episode_0'/name for name in episode['guard']['native_files'])
    require(required <= payloads, 'required numeric payload did not pass deferred archive')
    for section, keys in (('C_state', ('control_attempts','control_returns','construction_attempts','construction_returns')),
                          ('python', ('control_attempted','control_completed','native_attempted','native_returned'))):
        require(all(receipt['boundary_before_flush'][section][k] == receipt['boundary_after_flush'][section][k]
                    for k in keys), 'archive flush advanced physical state')
    return {'owner_join':owner_join,'all_required_numeric_payloads_durable':True,
            'transaction_count':len(queued),'payload_count':len(payloads),
            'flush_wall_s':(receipt['flush_end_ns']-receipt['flush_start_ns'])/1e9}


def diagnose(run):
    started = time.monotonic()
    run = Path(run).resolve(strict=True)
    session, worker = document(run/'session.json'), saved_document(run/'worker_receipt.json')
    require(worker['failure'] is not None and worker['result'] == {}
            and worker['accounting_failure'] is None and worker['final_accounting'] is not None,
            'this entry is only for the recorded failed C28 with complete final accounting')
    require(worker['session_identity'] == identity(run/'session.json')
            and worker['archive_failed'] is False and worker['writer_partial'] == [],
            'failed worker session/archive identity differs')
    report = {'schema':'d1-c28-failed-run-saved-diagnostic-v1','run':str(run),
        'record_qualification':False,'arm_qualification_passed':False,
        'original_worker_failure':worker['failure'],'original_worker_result':worker['result'],
        'formal_run_status':'failed; diagnosis cannot upgrade or authorize later arms',
        'reader_model_calls':0,'reader_physics_calls':0,'sections':{},'diagnostic_errors':[],
        'missing_upper_layer_files':[name for name in ('runtime_module_origins.json','mapped_libraries_final.json')
                                     if not (run/name).exists()],
        'missing_original_result_fields':['thread_violations','copy_edges','data_addresses','mailbox',
            'pause_render_proofs','control_step_caller_verified','stop_reason','timeline','native_guard'],
        'source_note':'critical executed-reader inputs are checked here; missing actual final runtime origins remain missing'}
    def section(name, function):
        try:
            value = function()
            report['sections'][name] = {'checked':True,'value':value}
            return value
        except Exception as error:
            report['sections'][name] = {'checked':False,'error':type(error).__name__+': '+str(error)}
            report['diagnostic_errors'].append({'section':name,'traceback':traceback.format_exc()})
            return None
    dependencies = section('critical_source_subset', lambda: source_subset(session))
    go_path = Path(session['source_go_path'])
    go = document(go_path)
    require(identity(go_path) == session['source_go_identity'] and go['decision'] == 'GO', 'source GO identity differs')
    spec = document(W/'continuation27/spec27.json')
    construction = saved_document(run/'construction_receipt.json')
    binding = check_binding(construction['geometry_binding'])
    geometry = geometry_map(construction['compiled_geometry'], binding)
    kin = construction['side_kinematic_binding']['kinematics24']
    reference = document(Path(session['reference_construction_path']))
    require(construction['geometry_binding'] == reference['geometry_binding']
            and construction['compiled_geometry'] == reference['compiled_geometry']
            and construction['side_profile'] == frozen_profile(R/'scripts/d1_fast_side_step.py'),
            'actual compiled geometry or teacher profile changed')
    require(dependencies is not None, 'cannot run even a diagnostic through changed pure readers')
    episode = section('complete_control_state_native_and_torque_chain',
                      lambda: check_episode(run,session,binding,geometry,kin))
    if episode is not None:
        n = episode['controls']
        # Do not serialize large internal arrays as if they were a new recording.
        report['sections']['complete_control_state_native_and_torque_chain']['value'] = {
            'completed_controls':n,'native_returns':len(episode['native']),
            'side_controls':episode['side_controls'],'actual_B22_predictions':episode['policy_predictions'],
            'complete_initial_and_endpoint_fields':list(episode['initial']),
            'handoff_count':len(episode['handoffs']), 'all_saved_native_rows_checked':True}
        case = next(c for c in spec['cases'] if c['id'] == session['arm'])
        task = section('physical_task_recomputation', lambda: score_case(episode,spec,case,binding,geometry,kin))
        if task is not None:
            task['interpretation'] = 'saved physical task gates only; failed source/finalization run remains unqualified'
        final = worker['final_accounting']
        c,p = final['C_state'], final['python']
        def accounting():
            require(final['completed_controls_in_rows'] == n and final['policy_predictions_in_loop'] == episode['policy_predictions']
                    and c['control_attempts'] == c['control_returns'] == p['native_attempted'] == p['native_returned'] == 5*n
                    and c['construction_attempts'] == c['construction_returns'] == 2
                    and p['control_attempted'] == p['control_completed'] == n
                    and p['clock_advanced_substeps'] == 5*n and c['violations'] == p['forbidden_entries'] == 0
                    and p['fatal_latched'] is False and c['phase'] == c['target_model'] == c['target_data'] == 0
                    and c['ccd_attempts'] == c['ccd_returns'], 'actual final accounting does not join saved rows')
            # This is an explicit view of existing final_accounting fields for a
            # pure checker, never a replacement/reconstruction of worker.result.
            view = {'model_calls':final['model_calls'],'policy_predictions':final['policy_predictions_in_loop']}
            check_model_calls(view, session, episode['policy_predictions'])
            return {'controls':n,'normal_native':5*n,'compiler_native':2,
                    'policy_predictions':episode['policy_predictions'],'model_calls':final['model_calls'],
                    'source':'worker_receipt.json.final_accounting, cross-joined to actual rows'}
        section('final_accounting', accounting)
        deferred = saved_document(run/'deferred_archive_receipt.json')
        def side():
            require(final['side_access']['owner_ident'] == deferred['owner_thread_ident'], 'actual side/archive owner differs')
            view = {'side_access':final['side_access'],'owner_thread_ident':deferred['owner_thread_ident']}
            return check_side_access(view, episode, spec)
        section('side_scoped_API_counts', side)
        section('durable_archive_and_owner_close', lambda: archive_diagnostic(run,session,worker,episode,deferred))
        section('input_and_render_from_saved_boundaries',
                lambda: render_diagnostic(run,session,worker,episode,deferred,go))
    report['input_identities'] = {str(p):identity(p) for p in run.rglob('*')
                                 if p.is_file() and p.name not in ('worker_stdout.log',)}
    report['diagnostic_source_identity'] = identity(Path(__file__).resolve())
    report['diagnostic_completed_without_check_errors'] = not report['diagnostic_errors']
    report['wall_s'] = time.monotonic()-started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'exclusive diagnostic output already exists')
    value = diagnose(args.run)
    with args.output.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'output':str(args.output),'qualification':False,
                      'diagnostic_errors':len(value['diagnostic_errors']),'wall_s':value['wall_s']}))


if __name__ == '__main__':
    main()
