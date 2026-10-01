"""Read the failed C29 host's saved prefix; never promote its qualification."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P/'read_profile29'))
import read_profile29 as frozen


def diagnose(run):
    run = Path(run).resolve(strict=True)
    started = time.monotonic()
    session = frozen.document(run/'session.json')
    worker = frozen.saved_document(run/'worker_receipt.json')
    result = worker['result']
    host = frozen.document(run/'host_receipt.json')
    supervisor = frozen.document(run/'supervisor_receipt.json')
    spec = frozen.document(P/'spec29.json')
    sections = {}

    def capture(name, function):
        begin = time.monotonic()
        try:
            value = function()
        except Exception as error:
            sections[name] = {'passed': False, 'error_type': type(error).__name__,
                              'error': str(error), 'reader_wall_s': time.monotonic()-begin}
            return None
        sections[name] = {'passed': True, 'reader_wall_s': time.monotonic()-begin}
        return value

    def check_host():
        # Preserve the original rejection. No altered session, worker, host or GO view.
        frozen.require(host['exit_code'] == 0 and host['failure'] is None
                       and supervisor['exit_code'] == 0 and supervisor['failure'] is None,
                       'original C29 host/supervisor success requirement failed: host exit='
                       +str(host['exit_code'])+', supervisor exit='+str(supervisor['exit_code']))

    capture('original_host_success_gate', check_host)

    def source_scope():
        frozen.require(session['execution_contract_id'] == frozen.INNER
                       and session['outer_execution_contract_id'] == frozen.OUTER
                       and session['arm'] == 'profile_left' and session['qualification'] is False
                       and session['control_limit'] == 500 and session['close_s'] == 60
                       and session['hard_s'] == 75 and session['side_arm'] == 'teacher'
                       and spec['outer_execution_contract_id'] == frozen.OUTER,
                       'actual diagnostic session differs from C29')
        frozen.require(worker['session_identity'] == frozen.identity(run/'session.json')
                       and worker['failure'] is None and worker['archive_failed'] is False
                       and worker['writer_partial'] == [] and worker['accounting_failure'] is None,
                       'saved worker is not the stated normally archived prefix')
        path = Path(session['source_go_path'])
        go = frozen.document(path)
        frozen.require(frozen.identity(path) == session['source_go_identity']
                       and go['decision'] == 'GO'
                       and all(session.get(k) == v for k, v in go['shared_session'].items())
                       and all(session.get(k) == v for k, v in go['arms']['profile_left'].items()),
                       'saved session/GO identity differs')
        # Recheck only the code actually used by this new pure reader. Do not
        # repeat the host's thousands-file pre/post source scan.
        dependencies = {}
        names = ('read_profile29', 'records29', 'archive29', 'ledger29', 'timing29',
                 'reader26', 'reader26_deferred', 'side_math27', 'input_reader27',
                 'kinematics24', 'verify_control18', 'verify_course_e_08_03',
                 'verify_rl16_training_08')
        for name in names:
            actual = Path(sys.modules[name].__file__).resolve()
            bound = actual if str(actual) in session['source_hashes'] else frozen.BUNDLE/actual.relative_to(frozen.W)
            identity = frozen.identity(actual)
            frozen.require(session['source_hashes'].get(str(bound)) == identity,
                           'actual frozen reader dependency differs: '+name)
            dependencies[name] = {'actual_path': str(actual), 'frozen_path': str(bound), 'identity': identity}
        frozen.require(session['source_hashes'].get(str(P/'spec29.json')) == frozen.identity(P/'spec29.json'),
                       'C29 spec changed')
        return {'dependencies': dependencies,
                'host_recorded_source_mismatches': host.get('source_mismatches'),
                'full_source_closure_rehashed_by_this_reader': False,
                'scope': 'session/GO and actual pure-reader bytes checked; full historical source drift result is the saved host report, not a new independent scan'}

    source = capture('limited_source_binding', source_scope)
    capture('saved_actual_module_origins', lambda: frozen.check_origins(run, session))
    episode = capture('construction_state_control_native_and_side_arithmetic',
                      lambda: frozen.construction_and_records(run, session, result))
    rows = episode['rows'] if episode is not None else capture(
        'atomic_rows_only_fallback', lambda: list(frozen.saved_rows(run/'episode_0/controls.jsonl.gz')))
    ledger = archive = preparation = timing_result = None
    if episode is not None:
        ledger = capture('actual_model_native_side_ledgers',
                         lambda: frozen.check_ledgers(result, session, episode, spec))
        archive = capture('saved_owner_join_durable_archive_and_final_accounting',
                          lambda: frozen.check_archive29(run, session, worker, episode))
    else:
        for key in ('actual_model_native_side_ledgers', 'saved_owner_join_durable_archive_and_final_accounting'):
            sections[key] = {'passed': False, 'not_run': 'independent episode verification failed'}

    def read_timing():
        path = run/'timing29.json'
        frozen.require(worker['timing29_identity'] == frozen.identity(path), 'worker/timing identity differs')
        return frozen.saved_document(path)

    timing = capture('atomic_timing_payload', read_timing)
    if rows is not None and timing is not None:
        timing_result = capture('exclusive_wall_threadCPU_and_parent_containment',
                                lambda: frozen.verify_timing(timing, result, rows))

    def input_check():
        inputs = list(frozen.saved_rows(run/'input_snapshots.jsonl.gz'))
        decisions = list(frozen.saved_rows(run/'owner_decisions.jsonl.gz'))
        frozen.require(worker['input_polls'] == len(inputs) and worker['owner_decisions'] == len(decisions),
                       'saved input counts differ')
        value = frozen.check_preparation(session, inputs, decisions, rows)
        performance = frozen.saved_document(run/'performance.json')
        frozen.require(performance['active_start_ns'] == result['active_start_ns']
                       and performance['active_end_ns'] == result['active_end_ns']
                       and performance['poll_timestamps_ns'] == [r['poll']['wall_ns'] for r in inputs]
                       and 0 < performance['render_api_calls'] <= 500
                       and len([s for s in timing['main_spans'] if s['stage'] == 'render']) == performance['render_api_calls']
                       and len([s for s in timing['main_spans'] if s['stage'] == 'poll']) == len(inputs),
                       'saved input/render/timing counts differ')
        return value

    if rows is not None and timing is not None:
        preparation = capture('saved_input_preparation_and_timed_main_call_counts', input_check)
    required = ('construction_state_control_native_and_side_arithmetic', 'actual_model_native_side_ledgers',
                'saved_owner_join_durable_archive_and_final_accounting')
    numerical_verified = all(sections.get(k, {}).get('passed') is True for k in required)
    n = result['completed_controls']
    return {'schema': 'd1-c29-failed-host-saved-prefix-diagnostic-v1',
            'execution_contract_id': frozen.OUTER, 'run': str(run),
            'qualification': False, 'record_qualification': False, 'profile_record_valid': False,
            'GUI_qualified': False, 'task_qualified': False,
            'original_host_success_gate_overridden': False,
            'full_profile_budget_completed': n == 500, 'scope_is_incomplete_prefix': True,
            'observed_controls_in_worker_result': n,
            'numerical_prefix_independently_verified': numerical_verified,
            'verified_controls': n if numerical_verified else None,
            'verified_normal_native_returns': 5*n if numerical_verified else None,
            'verified_compiler_native_returns': 2 if numerical_verified else None,
            'worker_reported_policy_predictions': result['policy_predictions'],
            'worker_reported_stop_reason': result['stop_reason'],
            'actual_host_receipt': host, 'actual_supervisor_receipt': supervisor,
            'worker_failure': worker['failure'], 'worker_archive_failed': worker['archive_failed'],
            'sections': sections, 'source_scope': source, 'actual_side_API_ledger': ledger,
            'saved_archive_closure': archive, 'input_preparation': preparation, 'timing': timing_result,
            'timing_numeric_row_basis_independently_verified': episode is not None,
            'identities': {name: frozen.identity(run/name) for name in (
                'session.json', 'worker_receipt.json', 'host_receipt.json', 'supervisor_receipt.json', 'timing29.json')},
            'reader_source_identity': frozen.identity(Path(__file__).resolve()),
            'reader_wall_s': time.monotonic()-started, 'reader_model_calls': 0, 'reader_physics_calls': 0,
            'scope': 'saved prefix of a failed host; nested wall/CPU are not summed; no full-cycle speed, safe stop, qualification, causal attribution to other processes, or new experiment claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frozen.require(not args.output.exists(), 'exclusive diagnostic output already exists')
    report = diagnose(args.run)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == '__main__':
    main()
