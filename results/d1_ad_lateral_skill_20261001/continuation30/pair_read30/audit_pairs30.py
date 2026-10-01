"""Saved-only C30 six-case pairing and pilot trigger; never authorizes training.

Full 5T/contact/reference arithmetic is delegated to the already closed, hashed
independent per-case readbacks. This increment checks their evidence bindings,
recomputes pairing/aggregate counts/comparison metrics, and retains failures.
No worker, controller, learning library or physics module is imported.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

P = Path(__file__).resolve().parents[1]
CONTRACT = 'C30_lateral_skill_feasibility_v1'
CASES = ('fixed_nonzero_left', 'fixed_nonzero_right', 'zero_left', 'zero_right',
         'cancel_left', 'cancel_right')
ARRAYS = ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart', 'observation', 'time')
TRIGGER = dict(all_six_records_valid=True,
               both_full_directions_and_both_cancels_qualified=True,
               each_direction_cycle_fixed_over_zero_max=.95,
               effective_overlap_controls_per_leg_min=1,
               effective_overlap_legs_per_direction_min=2,
               full_torque_square_mean_over_zero_max=1.2,
               goal_error_delta_over_zero_max_m=.002,
               retention_absolute_min=.9, retention_delta_over_zero_min=-.02)


def need(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identity(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
            size += len(block)
    return dict(bytes=size, sha256=digest.hexdigest())


def reject_constant(value):
    raise ValueError('nonfinite JSON: '+value)


class Evidence:
    def __init__(self):
        self.inputs = {}

    def bind(self, path):
        path = Path(path).absolute()
        need(path.is_file() and not path.is_symlink(), 'missing/symlink input: '+str(path))
        key = str(path)
        if key not in self.inputs:
            self.inputs[key] = identity(path)
        return self.inputs[key]

    def document(self, path):
        self.bind(path)
        result = json.loads(Path(path).read_text(), parse_constant=reject_constant)
        need(isinstance(result, dict), 'nonobject document: '+str(path))
        return result

    def committed(self, path):
        path = Path(path)
        manifest = self.document(path.with_name(path.name+'.manifest.json'))
        items = manifest['payloads']
        names = [row['file'] for row in items]
        need(manifest['schema'] == 'd1-archive-transaction-13-v1'
             and len(names) == len(set(names)) and path.name in names,
             'invalid atomic manifest: '+str(path))
        for row in items:
            need(Path(row['file']).name == row['file'], 'unsafe atomic member')
            need(self.bind(path.parent/row['file']) == {k: row[k] for k in ('bytes', 'sha256')},
                 'atomic payload changed: '+str(path.parent/row['file']))

    def saved(self, path):
        self.committed(path)
        return self.document(path)

    def arrays(self, path):
        self.committed(path)
        with np.load(path, allow_pickle=False) as archive:
            need(len(archive.files) == len(set(archive.files)), 'duplicate NPZ members')
            result = {key: archive[key].copy() for key in archive.files}
        need(set(result) == set(ARRAYS)
             and all(a.dtype.kind in 'fibu' and np.isfinite(a).all() for a in result.values()),
             'state schema/nonfinite array')
        return result


def byte_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def close(a, b):
    return bool(np.isfinite([a, b]).all() and abs(a-b) <= 1e-10)


def actual_ledger(worker):
    """Independent integer closure; actual executions remain visible on failure."""
    n = worker['result']['completed_controls']
    c, py, native = worker['C_final'], worker['python'], worker['native_guard']
    compiler = c['construction_returns']
    need(type(n) is int and 0 <= n <= 2200 and type(compiler) is int,
         'bad actual control/construction count')
    need(py['control_attempted'] == py['control_completed'] == n
         and c['control_attempts'] == c['control_returns'] == 5*n
         and py['native_attempted'] == py['native_returned'] == py['clock_advanced_substeps'] == 5*n
         and native['native_attempted'] == native['native_returned'] == native['native_checked'] == 5*n
         and c['construction_attempts'] == compiler == 2,
         'actual C/Python/native/construction counts disagree')
    calls = worker['model_calls']
    counts = calls['counts']
    for name in ('load', 'torch_load', 'predict'):
        need(counts[name]['attempted'] == counts[name]['returned'], 'incomplete model call')
    need(counts['load']['returned'] == 1 and counts['torch_load']['returned'] == 3
         and worker['training_controls'] == worker['optimizer_steps'] == calls['critic_rows'] == 0
         and all(counts.get(k, {}).get('attempted', 0) == counts.get(k, {}).get('returned', 0) == 0
                 for k in ('train', 'learn', 'save', 'backward', 'forward', 'evaluate_actions', 'predict_values')),
         'unexpected training/model calls')
    return dict(controls=n, normal_native=5*n, compiler_native=compiler,
                load=1, torch_load=3, predict=counts['predict']['returned'],
                actor_rows=calls['actor_rows'], training_controls=0, optimizer_steps=0)


def read_case(base, case, evidence, item):
    run = base/(case+'_01')
    report_path = base/('independent_'+case+'_01.json')
    session = evidence.document(run/'session.json')
    worker = evidence.document(run/'worker_receipt.json')
    item['actual_ledger'] = actual_ledger(worker)
    report = evidence.document(report_path)
    item['reported_qualification'] = report.get('qualification_passed')
    item['reported_record_valid'] = report.get('record_valid')
    execution = evidence.document(base/('reader_'+case+'_execution_30.json'))
    host = evidence.document(run/'host_receipt.json')
    supervisor = evidence.document(run/'supervisor_receipt.json')
    need(report['case_id'] == execution['case'] == session['arm'] == case
         and Path(report['run']).resolve() == run.resolve()
         and report['execution_contract_id'] == session['execution_contract_id'] == CONTRACT
         and report['session_identity'] == worker['session_identity'] == evidence.bind(run/'session.json')
         and report['worker_receipt_identity'] == evidence.bind(run/'worker_receipt.json')
         and execution['output']['path'] == str(report_path)
         and {k: execution['output'][k] for k in ('bytes', 'sha256')} == evidence.bind(report_path)
         and execution['exit_code'] == 0 and execution['timed_out'] is False
         and execution['qualification_passed'] is report['qualification_passed']
         and report['record_valid'] is True and report['source']['passed'] is True,
         'independent readback/session/execution binding failed')
    need(host['exit_code'] == 0 and host['failure'] is None and host['source_mismatches'] == []
         and host['owned_no_orphans'] is True and host['reservation_closed'] is True
         and host['reserved_controls'] == 2200 and host['retry_permitted'] is False
         and supervisor['exit_code'] == 0 and supervisor['failure'] is None
         and supervisor['cleanup']['remaining'] == {} and supervisor['outer_limit_s'] <= 900
         and supervisor['elapsed_s'] <= supervisor['outer_limit_s']
         and worker['failure'] is None and worker['cleanup_errors'] == []
         and worker['warnings'] == [] and worker['archive_failed'] is False
         and worker['execution_complete'] is True, 'host/worker/archive did not close')
    go = evidence.document(session['source_go_path'])
    need(evidence.bind(session['source_go_path']) == session['source_go_identity']
         and go['decision'] == 'GO' and go['execution_contract_id'] == CONTRACT
         and all(session.get(k) == v for k, v in go['shared_session'].items())
         and all(session.get(k) == v for k, v in go['arms'][case].items()), 'case is not its source GO')
    for path in (base/'spec30_draft.json', base/'skill_contract_30.md', base/'read30/read30.py'):
        need(session['source_hashes'][str(path)] == evidence.bind(path), 'contract/reader source drift')
    need(report['reader_source_identity'] == evidence.bind(base/'read30/read30.py'), 'reader identity differs')
    origins = evidence.saved(run/'runtime_module_origins.json')
    need(origins == evidence.document(run/'runtime_module_origins_before.json'), 'loaded sources changed')
    for name, source in origins.items():
        expected = {k: source[k] for k in ('bytes', 'sha256')}
        need(evidence.bind(source['path']) == expected == session['source_hashes'][source['path']]
             == report['source']['runtime_and_reader_current_identities'][name], 'runtime source drift: '+name)
    loaded = evidence.saved(run/'loaded_origins_final.json')
    loaded_paths = set(loaded['module_origins'].values())
    loaded_paths.update(v['defining_source'] for v in loaded['synthetic_module_aliases'].values())
    loaded_sources = {path: session['source_hashes'][path] for path in sorted(loaded_paths)}
    folder = run/'episode_0'
    initial = evidence.arrays(folder/'initial_state.npz')
    states = evidence.arrays(folder/'states.npz')
    reset = evidence.saved(folder/'reset.json')
    segment = evidence.saved(folder/'segment_receipt.json')
    macro = evidence.saved(folder/'macro_transitions30.json')
    construction = evidence.saved(run/'construction_receipt.json')
    for name in segment['native_segment']['native_files']:
        need(Path(name).name == name, 'unsafe native block path')
        evidence.committed(folder/name)
    n = item['actual_ledger']['controls']
    need(n > 200 and segment['completed_controls'] == n
         and all(states[k].shape == (n+1, *initial[k].shape)
                 and byte_equal(states[k][0], initial[k]) for k in ARRAYS), 'actual initial/state/count join')
    need(all(report['ledger'][k] == item['actual_ledger'][k]
             for k in ('controls', 'normal_native', 'compiler_native')), 'report count differs')
    compiled = construction['side_kinematic_binding']
    ids = np.asarray(construction['geometry_binding']['actuator_ids'], dtype=int)
    ranges = np.asarray(compiled['dynamics24']['actuator_ctrlrange'])[ids]
    limits = ranges[:, 1]
    need(limits.shape == (16,) and np.all(limits > 0) and np.array_equal(ranges[:, 0], -limits),
         'compiled nominal motor limits differ')
    prefix = hashlib.sha256()
    side_indices, overlap, cancels = [], {}, []
    summed_cost = 0.
    entry = None
    row_count = 0
    evidence.committed(folder/'controls.jsonl.gz')
    with gzip.open(folder/'controls.jsonl.gz', 'rt') as stream:
        for i, line in enumerate(stream):
            row = json.loads(line, parse_constant=reject_constant)
            need(row['control_index'] == i, 'raw control index gap')
            row_count += 1
            torque = np.asarray([t['applied_nm'] for t in row['native_actuator_traces']], dtype=float)
            need(torque.shape == (5, 16) and np.isfinite(torque).all(), 'actual 5T torque shape')
            summed_cost += float(np.mean(np.square(torque/limits)))
            if i < 200:
                need(row['actor'] == 'B' and row['policy_predict_called'] is True
                     and row['skill30'] is None and row['macro_events30'] == [], 'preparation is not B22 only')
                selected = {key: row[key] for key in ('input_observation99', 'raw_action16', 'action16',
                            'policy_input_action', 'raw_command', 'controller_handoff_receipt', 'side_start_receipt')}
                selected['info'] = {key: row['info'][key] for key in
                                    ('controller_record', 'raw_operator_command', 'consumed_command', 'servo_receipt')}
                prefix.update(canonical(selected)+b'\n')
            if i == 200:
                entry = row
                need(all(np.array_equal(row['pre_state'][k], states[k][200]) for k in ARRAYS),
                     'actual control-200 pre endpoint differs from saved array')
            if row['actor'] == 'traditional_side':
                side_indices.append(i)
                skill = row['skill30']
                velocity = np.asarray(skill['foot_reference_velocity_mps'], dtype=float)
                if np.linalg.norm(velocity[:2]) > 1e-10 and abs(velocity[2]) > 1e-10:
                    index = str(skill['macro_index'])
                    overlap[index] = overlap.get(index, 0)+1
            if row['cancel_requested30']:
                cancels.append(i)
    need(row_count == n and side_indices and side_indices == list(range(200, side_indices[-1]+1)),
         'raw control count/one side interval differs')
    end = side_indices[-1]+1
    need(report['task']['side_control_window'] == [200, end]
         and close(report['task']['side_duration_s'], (end-200)*.01), 'actual cycle boundary differs')
    q0, qe, qfinal = states['qpos'][200], states['qpos'][end], states['qpos'][-1]
    w, x, y, z = q0[3:7]
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    left = np.array([-math.sin(yaw), math.cos(yaw)])
    direction = 1 if case.endswith('_left') else -1
    need(session['direction'] == direction, 'case direction differs')
    signed = float(direction*np.dot(qe[:2]-q0[:2], left))
    retained = float(direction*np.dot(qfinal[:2]-q0[:2], left))
    ratio = retained/signed if signed != 0 and end+400 == n else None
    metrics = dict(cycle_controls=end-200, cycle_s=(end-200)*.01,
                   goal_error_m=float(np.linalg.norm(qe[:2]-(q0[:2]+direction*.03*left))),
                   retention_ratio=ratio, full_torque_square_mean=summed_cost/n,
                   signed_lateral_m=signed, retained_signed_lateral_m=retained,
                   cancel_to_handoff_controls=end-cancels[0] if len(cancels) == 1 else None,
                   overlap_legs=sum(value >= 1 for value in overlap.values()),
                   nonzero_xy=any(any(abs(v) > 1e-12 for v in e['consumed_action'][:2]) for e in macro['events']),
                   nonzero_action=any(any(abs(v) > 1e-12 for v in e['consumed_action']) for e in macro['events']))
    task, full = report['task'], report['macros']['full_scene_components']
    need(close(metrics['goal_error_m'], task['final_xy_error_m'])
         and (ratio is None and task['retention_ratio'] is None or
              ratio is not None and close(ratio, task['retention_ratio']))
         and close(metrics['full_torque_square_mean'], full['torque_normalized_square_mean'])
         and close(.01*summed_cost, full['torque_normalized_square_integral_s'])
         and metrics['overlap_legs'] == report['reference']['actual_overlap_legs']
         and metrics['cancel_to_handoff_controls'] == task['cancel_to_handoff_controls'],
         'independent comparison metrics differ from closed per-case reader')
    first_event = macro['events'][0]
    need(first_event['control_index'] == 200 and entry['info']['controller_record']['accepted_start'] is True,
         'first real side latch is not control 200')
    item.update(evidence_valid=True, qualification_passed=report['qualification_passed'], metrics=metrics,
                end_reason=report['end_reason'], overlap_controls_per_macro=overlap,
                full_retention_complete=end+400 == n,
                cancel_trigger_correct=report['cancel_trigger_or_absence_correct'])
    return dict(initial=initial, prefix_arrays={k: states[k][:201] for k in ARRAYS},
                reset_memory=reset['control_reset_state'], reset_metadata=reset['episode_metadata'],
                prefix_sha256=prefix.hexdigest(), entry=entry, first_event=first_event,
                compiled=compiled, profile=construction['side_profile'],
                origins=origins, reader_identity=report['reader_source_identity'],
                checkpoint_sha256=session['checkpoint_sha256'], loaded_sources=loaded_sources,
                frozen_sources=session['source_hashes'])


def pair_states(fixed, zero):
    checks = {}
    for key in ARRAYS:
        checks['initial_'+key+'_byte_equal'] = byte_equal(fixed['initial'][key], zero['initial'][key])
        checks['preparation_through_entry_'+key+'_byte_equal'] = byte_equal(
            fixed['prefix_arrays'][key], zero['prefix_arrays'][key])
    for key in ('reset_memory', 'reset_metadata', 'prefix_sha256', 'compiled', 'profile',
                'origins', 'reader_identity', 'checkpoint_sha256', 'loaded_sources'):
        checks[key+'_equal'] = canonical(fixed[key]) == canonical(zero[key])
    shared = set(fixed['frozen_sources']) & set(zero['frozen_sources'])
    checks['all_shared_frozen_identities_equal'] = all(
        fixed['frozen_sources'][path] == zero['frozen_sources'][path] for path in shared)
    a, b = fixed['entry'], zero['entry']
    for key in ('input_observation99', 'raw_command', 'side_start_receipt'):
        checks['entry_'+key+'_equal'] = canonical(a[key]) == canonical(b[key])
    for key in ('preview_consumed', 'wheel_memory_before'):
        checks['entry_'+key+'_equal'] = canonical(a['info']['controller_record'][key]) == canonical(b['info']['controller_record'][key])
    for key in ('raw_operator_command', 'consumed_command', 'servo_receipt'):
        checks['entry_'+key+'_equal'] = canonical(a['info'][key]) == canonical(b['info'][key])
    for key in ('teacher_body_from_m', 'teacher_body_to_m', 'teacher_shift_time_s',
                'plan_com_height_m', 'initial_body_yaw_rad'):
        checks['pre_action_'+key+'_equal'] = canonical(fixed['first_event'][key]) == canonical(zero['first_event'][key])
    return dict(passed=all(checks.values()), checks=checks, shared_frozen_input_count=len(shared),
                proof_kind='direct_saved_prefix_and_source_derived_entry_memory_equivalence',
                direct_evidence='reset full control state; 7 initial/prefix arrays through tick 200; 200 B22 records/actions/servos; actual entry provider/PI/z/servo/input; teacher plan before action',
                source_derived_scope='unsaved complete tick-200 context and actuator/provider hidden state follow identical frozen reset/transition sources and identical full 200-control prefix',
                post_latch_body_to_and_shift_time_compared=False,
                seed_alone_used_as_pairing_proof=False)


def compare_direction(fixed, zero, paired):
    f, z = fixed['metrics'], zero['metrics']
    gates = dict(paired_actual_entry=paired,
                 both_evidence_valid=fixed['evidence_valid'] and zero['evidence_valid'],
                 both_qualified=fixed['qualification_passed'] and zero['qualification_passed'],
                 cycle_at_least_5pct_faster=f['cycle_s'] <= .95*z['cycle_s'],
                 goal_error_le_zero_plus_2mm=f['goal_error_m'] <= z['goal_error_m']+.002,
                 retention_ge_max_absolute_and_zero_minus_02=f['retention_ratio'] is not None and
                     z['retention_ratio'] is not None and f['retention_ratio'] >= max(.9, z['retention_ratio']-.02),
                 full_torque_square_mean_le_1p20zero=f['full_torque_square_mean'] <= 1.2*z['full_torque_square_mean'],
                 actual_nonzero_action_consumed=f['nonzero_action'], actual_overlap_at_least_two_legs=f['overlap_legs'] >= 2)
    return dict(passed=all(gates.values()), gates=gates,
                cycle_fixed_over_zero=f['cycle_s']/z['cycle_s'],
                cycle_reduction_fraction=1-f['cycle_s']/z['cycle_s'],
                goal_error_delta_m=f['goal_error_m']-z['goal_error_m'],
                full_torque_square_mean_fixed_over_zero=(f['full_torque_square_mean']/z['full_torque_square_mean']
                                                         if z['full_torque_square_mean'] else None),
                fixed=f, zero=z, cost_is_energy=False)


def run_audit(base):
    need(not any(name.split('.')[0] in ('mujoco', 'torch', 'stable_baselines3') for name in sys.modules),
         'saved-only process already contains model/physics modules')
    evidence = Evidence()
    evidence.bind(Path(__file__))
    spec = evidence.document(base/'spec30_draft.json')
    evidence.bind(base/'skill_contract_30.md')
    need(spec['execution_contract_id'] == CONTRACT and spec['training_trigger'] == TRIGGER
         and (spec['all_case_controls'], spec['all_case_normal_native'], spec['all_case_compiler_native']) == (13200, 66000, 12),
         'training trigger/budgets differ from frozen C30 contract')
    cases, raw, pairs, comparisons = {}, {}, {}, {}
    for case in CASES:
        item = cases[case] = dict(evidence_valid=False, qualification_passed=False, errors=[])
        try:
            raw[case] = read_case(base, case, evidence, item)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            item['errors'].append(type(error).__name__+': '+str(error))
    for direction in ('left', 'right'):
        fixed, zero = 'fixed_nonzero_'+direction, 'zero_'+direction
        if fixed in raw and zero in raw:
            pairs[direction] = pair_states(raw[fixed], raw[zero])
            comparisons[direction] = compare_direction(cases[fixed], cases[zero], pairs[direction]['passed'])
        else:
            pairs[direction] = dict(passed=False, reason='paired source/records unavailable or invalid')
            comparisons[direction] = dict(passed=False, reason='paired comparison not established')
    cancel_gates = {}
    for direction in ('left', 'right'):
        item = cases['cancel_'+direction]
        delay = item.get('metrics', {}).get('cancel_to_handoff_controls')
        cancel_gates[direction] = bool(item['evidence_valid'] and item['qualification_passed']
            and item.get('end_reason') == 'safe_cancel' and item.get('full_retention_complete')
            and item.get('cancel_trigger_correct') and delay is not None and 0 < delay <= 300)
    known = [c['actual_ledger'] for c in cases.values() if 'actual_ledger' in c]
    totals = {key: sum(row[key] for row in known) for key in
              ('controls', 'normal_native', 'compiler_native', 'load', 'torch_load', 'predict',
               'actor_rows', 'training_controls', 'optimizer_steps')}
    gates = dict(all_six_actual_ledgers_available=len(known) == 6,
                 all_six_records_valid=all(c['evidence_valid'] for c in cases.values()),
                 all_six_qualified=all(c['qualification_passed'] for c in cases.values()),
                 both_actual_entry_pairs=all(p['passed'] for p in pairs.values()),
                 both_direction_trigger_gates=all(c['passed'] for c in comparisons.values()),
                 both_cancels_safe_and_within_300=all(cancel_gates.values()),
                 total_resource_caps=totals['controls'] <= 13200 and totals['normal_native'] <= 66000
                    and totals['compiler_native'] <= 12 and totals['training_controls'] == totals['optimizer_steps'] == 0)
    return dict(schema='d1-c30-six-case-pairing-pilot-trigger-v1', execution_contract_id=CONTRACT,
                pilot_trigger=all(gates.values()), gates=gates, cases=cases, pairs=pairs,
                direction_comparisons=comparisons, cancellation_gates=cancel_gates,
                actual_totals=totals, actual_ledger_case_count=len(known),
                counts_for_missing_or_invalid_ledgers_not_invented=True,
                training_authorized=False, training_GO_required_separately=True,
                GUI_qualified=False, RL_trained=False, learned_speedup_claim=False,
                deterministic_pairing_not_statistical_independence=True,
                control_native_model_counts_are_actual_only=True,
                pure_FK_static_API_not_added_to_normal_native=True,
                scope='per-case frozen independent reader retains full native/contact/reference proof; this increment rehashes bound actual payloads and recomputes paired entry/comparison/counts, not all historical GO inputs',
                inputs=evidence.inputs, reader_model_calls=0, reader_physics_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=P)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), 'exclusive audit output already exists')
    report = run_audit(args.base.resolve(strict=True))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
