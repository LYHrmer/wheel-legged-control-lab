"""Cold saved-only C33 fixed S-curve campaign readback; no model or physics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
W = Path(__file__).resolve().parents[2]
R = Path('/home/lyh/wheel-legged-control-lab')
for path in (W/'continuation26/root26', W/'continuation24',
             W/'continuation27/read27', W/'continuation30/read30',
             W/'continuation30/pair_read30', W/'continuation31/read31'):
    sys.path.insert(0, str(path))

import numpy as np
from reader26 import require, close, identity, document, saved_document
from episode_read31 import construction31
from ledger_read31 import case_ledger31
from pairing_read31 import entry_snapshot31, metrics31
from pairing_read33 import pair_entry33
from episode_read33 import physical_episode33
from source_read33 import sources33, CONTRACT
from comparison_read33 import prior_baselines33, qualification33, select_alpha33

LABELS = {0.85: '085', 0.925: '0925', 1.0: '100'}


def _case_plan33():
    return [dict(case_id=f'alpha_{label}_{side}', alpha=alpha,
                 direction=direction, kind='development')
            for alpha, label in LABELS.items()
            for side, direction in (('left', 1), ('right', -1))]


def _metric_checks33(saved, report, *, cancel):
    task, full = report['task'], report['macros']['full_scene_components']
    gates = task['gates']
    mandatory_safety = all((gates['entry'], gates['side_physical_safety'],
        gates['rolling_physical_safety'], gates['all_raw_servo_motion_zero'],
        gates['side_controls_within_1500'], gates['safe_landed_handoff'],
        gates['exact_400_retention_complete'],
        gates['original_tail_stopped_and_loaded'],
        gates['B22_predict_restored_in_retention'],
        gates['no_environment_termination']))
    expected = dict(full_tau2_mean=full['torque_normalized_square_mean'],
        full_tau2_integral_s=full['torque_normalized_square_integral_s'],
        full_elapsed_s=full['elapsed_s'], side_duration_s=task['side_duration_s'],
        final_xy_error_m=task['final_xy_error_m'],
        retention_ratio=task['retention_ratio'],
        safety_complete=mandatory_safety,
        entry_gates=task['entry_gates'])
    for key, value in expected.items():
        if isinstance(value, (dict, bool)) or value is None:
            require(saved[key] == value, 'C33 online metric differs from saved task: '+key)
        else:
            require(close(saved[key], value), 'C33 online metric differs from native full case: '+key)
    require(saved['full_tau2_mean'] == saved['full_tau2_integral_s']/saved['full_elapsed_s']
            or close(saved['full_tau2_mean'],
                     saved['full_tau2_integral_s']/saved['full_elapsed_s']),
            'C33 online tau² mean differs from actual integral/duration')
    if cancel:
        delay = task['cancel_to_handoff_controls']
        require(saved['eligible'] is report['qualification_passed']
                and saved['cancel_to_handoff_controls'] == delay
                and delay is not None and 0 < delay <= 300
                and report['independent_terminal_kind'] == 'safe_cancel',
                'C33 selected-alpha cancel/handoff differs from old safe gate')


def global_ledger33(run, session, worker, construction, audits):
    result = worker['result']
    n = sum(row['controls'] for row in audits)
    t = sum(row['side_controls'] for row in audits)
    p = sum(row['macros'] for row in audits)
    predictions = sum(row['policy_predictions'] for row in audits)
    count = len(audits)
    require(1 <= count <= 8 and count == result['closed_cases']
            and n == result['controls'] <= 17600
            and 5*n == result['normal_native'] <= 88000
            and p == result['true_macros'] <= 32
            and predictions == result['policy_predictions'] == 600*count <= 4800
            and result['budget']['limits'] == dict(cycles=8, controls=17600, macros=32)
            and len(result['budget']['closed']) == count
            and result['budget']['macros'] == p
            and len({row['model_address'] for row in audits})
                == len({row['data_address'] for row in audits}) == 1,
            'C33 actual case/control/native/prediction budget or single-model identity differs')
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
            and native['failure'] is None,
            'C33 actual C/Python/native/compiler ledger differs')
    caller = c['first_control_step_caller_dladdr']
    lib = caller['path']
    require(caller['address'] == c['first_control_step_caller']
            and caller['offset'] == 777003
            and caller['address']-caller['base'] == caller['offset']
            and session['source_hashes'][lib]['sha256'] == construction['proof']['elf_hashes'][lib]
            and any(row['elf'] == lib and row['symbol'] == 'mj_step' and row['passed'] is True
                    for row in construction['proof']['jump_slots']),
            'C33 normal native step caller is not the frozen E return site')
    access = worker['side_access']
    bounds = dict(copy=t+4*p, forward=112*t+236*p, jacBody=96*t+192*p,
                  jac=4*t, fullM=t, objectVelocity=2*t+count)
    hard = {key: value*8 for key, value in dict(copy=1516, forward=168944,
        jacBody=144768, jac=6000, fullM=1500, objectVelocity=3001).items()}
    require(access['starts'] == count and access['computes'] == t
            and access['prepares'] == p and access['rejected'] == 0
            and access['within_actual_bounds'] is True and access['open_cycle'] is None
            and len(access['closed_cycles']) == count
            and access['actual_T_P_bounds'] == bounds and access['limits'] == hard
            and audits[0]['scope_range'][0] == 0
            and audits[-1]['scope_range'][1] == len(access['scopes'])
            and all(a['scope_range'][1] == b['scope_range'][0]
                    for a, b in zip(audits, audits[1:])),
            'C33 actual static API scope/ceiling differs')
    for key in bounds:
        require(access['counts'][key] == sum(row['static_counts'][key] for row in audits)
                <= bounds[key] <= hard[key], 'C33 actual static API count differs: '+key)
    addresses, edges = construction['data_addresses'], worker['copy_edges']
    require(set(addresses) == {'live', 'measurement', 'side_scratch'}
            and len(set(addresses.values())) == 3
            and addresses['live'] == audits[0]['data_address']
            and addresses['side_scratch'] == access['scratch_address']
            and set(edges) == {'live_to_measurement', 'measurement_to_side_scratch', 'rejected'}
            and edges['rejected'] == 0 and edges['live_to_measurement'] >= n
            and edges['measurement_to_side_scratch'] == access['counts']['copy'],
            'C33 one-plant/three-data copy fence differs')
    calls = worker['model_calls']
    counts = calls['counts']
    expected = dict(load=1, torch_load=3, predict=predictions+1,
        forward=0, evaluate_actions=0, predict_values=0, backward=0,
        learn=0, train=0, save=0)
    require(calls['limits'] == session['model_limits']
            and set(counts) <= set(expected)
            and all(counts.get(key, {}).get('attempted', 0)
                    == counts.get(key, {}).get('returned', 0) == value
                    for key, value in expected.items())
            and counts['predict']['rows_attempted'] == counts['predict']['rows_returned'] == predictions+32
            and calls['actor_rows'] == predictions+32 and calls['critic_rows'] == 0
            and worker.get('lateral_model_calls') is None,
            'C33 B22-only model/no-training ledger differs')
    reset = saved_document(Path(run)/'reset_yaw_receipt31.json')
    require(reset['calls'] == len(reset['receipts']) == count,
            'C33 extra or missing reset calls')
    for i, row in enumerate(reset['receipts']):
        require(row == dict(call_index=i, requested_initial_yaw_rad=0.,
            forwarded_base_quaternion_wxyz=None, original_reset_called_once=True,
            phase='plant_provider_controller_reset_before_prepare'),
            'C33 reset/yaw forwarding differs')
    return dict(passed=True, cases=count, controls=n, normal_native=5*n,
                compiler_native=2, macros=p, side_controls=t,
                B22_control_predictions=predictions, B22_probe_rows=32,
                new_lateral_actor_calls=0, new_lateral_value_calls=0,
                new_optimizer_steps=0, data_objects=3,
                actual_static_API=access['counts'])


def read_run33(run, *, reader_review_path):
    run = Path(run).resolve(strict=True)
    session = document(run/'session.json')
    worker = document(run/'worker_receipt.json')
    source = sources33(run, session, worker, reader_review_path=reader_review_path)
    definition = saved_document(run/'case_definition33.json')
    selection = saved_document(run/'campaign_selection33.json')
    require(definition['schema'] == 'd1-c33-case-definition-v1'
            and selection['schema'] == 'd1-c33-campaign-selection-v1'
            and selection['qualification'] is False
            and selection['independent_readback_pending'] is True,
            'C33 attempted-case/selection archive is missing')
    cases = definition['cases']
    require(1 <= len(cases) <= 8
            and {path.name for path in run.glob('episode_*') if path.is_dir()}
                == {f'episode_{i}' for i in range(len(cases))},
            'C33 actual episode directory count differs from attempted ledger')
    construction, binding, geometry, kin = construction31(run, session, R)
    spec = document(W/'continuation27/spec27.json')
    prior, evidence = prior_baselines33(session)
    online_baseline = saved_document(run/'baseline33.json')
    for side in ('left', 'right'):
        for mode, old_name in (('zero', 'zero'), ('fixed', 'fixed_nonzero')):
            row = online_baseline[side][mode]
            item = prior[f'{old_name}_{side}'][0]['metrics']
            old_path = W/'continuation30'/f'independent_{old_name}_{side}_01.json'
            require(row['source_path'] == str(old_path)
                    and row['source_identity'] == identity(old_path)
                    and close(row['side_duration_s'], item['cycle_s'])
                    and close(row['final_xy_error_m'], item['goal_error_m'])
                    and close(row['retention_ratio'], item['retention_ratio'])
                    and close(row['full_tau2_mean'], item['full_torque_square_mean']),
                    'C33 online baseline did not use the exact independently saved C30 metrics')
    reports, comparisons, audits, online_rows = {}, {}, [], []
    development = {}
    offset = 0
    previous_access = None
    plan = _case_plan33()
    dev_count = sum(row['kind'] == 'development' for row in cases)
    require([dict((k, row[k]) for k in ('case_id', 'alpha', 'direction', 'kind'))
             for row in cases[:dev_count]] == plan[:dev_count]
            and all(row['kind'] == 'cancel' for row in cases[dev_count:])
            and dev_count <= 6 and len(cases)-dev_count in (0, 2),
            'C33 development/cancel order differs from frozen list')
    for index, item in enumerate(cases):
        require(item['cycle_index'] == index
                and item['cycle_receipt_identity'] == identity(
                    run/f'episode_{index}'/'cycle_receipt.json'),
                'C33 case definition is not bound to its saved original cycle')
        online = saved_document(run/f'case33_{index:02d}.json')
        require(all(online[k] == item[k] for k in
                    ('cycle_index', 'case_id', 'alpha', 'direction', 'kind',
                     'cycle_receipt_identity'))
                and online.get('body_alpha33', item['alpha']) == item['alpha'],
                'C33 online case row differs from bound definition')
        report, cycle, episode = physical_episode33(run, session, construction,
            binding, geometry, kin, spec, index, offset, alpha33=item['alpha'])
        require(report['case_id'] == item['case_id'] and report['mode'] == 'fixed'
                and report['direction'] == item['direction']
                and report['requested_initial_yaw_rad'] == 0.
                and (cycle['terminal_kind'] == 'safe_cancel') is (item['kind'] == 'cancel')
                and online['terminal_kind'] == cycle['terminal_kind']
                and online['actual_controls'] == episode['controls']
                and online['policy_predictions'] == episode['policy_predictions'],
                'C33 actual fixed episode condition/terminal state differs')
        access = episode['cycle_receipt31']['side_access']
        before = access['before']
        if previous_access is None:
            require(all(before[k] == 0 for k in ('starts', 'computes', 'prepares', 'scopes'))
                    and all(value == 0 for value in before['counts'].values()),
                    'C33 hidden side work before first case')
        else:
            require(all(before[key] == previous_access['after'][key]
                        for key in ('starts', 'computes', 'prepares', 'counts'))
                    and before['scopes'] == previous_access['scope_range'][1],
                    'C33 side API work was reset/refunded between cases')
        previous_access = access
        require(all(event['skill31_latch']['policy_receipt'] is None
                    for event in cycle['macros']),
                'C33 fixed mode invoked lateral actor/value')
        _metric_checks33(online['metrics'], report, cancel=item['kind'] == 'cancel')
        audits.append(case_ledger31(run, session, worker, episode, report, index, offset))
        offset += episode['controls']
        progress = saved_document(run/f'progress33_{index:02d}.json')
        require(progress == dict(closed_cases=index+1, controls=offset,
            normal_native=5*offset, macros=sum(a['macros'] for a in audits),
            policy_predictions=sum(a['policy_predictions'] for a in audits),
            last_case=online),
            'C33 cumulative progress differs from actual completed saved cases')
        snapshot = entry_snapshot31(run, session, construction, episode)
        if item['kind'] == 'development':
            check = qualification33(report, snapshot, item['direction'], item['alpha'], prior)
            require(online['metrics']['eligible'] is check['eligible']
                    and online['metrics']['nominal_fixed_engineering_gain'] is
                        check['engineering_speed_passed']
                    and all(online['metrics']['gates'].values()) is check['eligible'],
                    'C33 online candidate qualification differs from independent old gates')
            development.setdefault(item['alpha'], {})[item['direction']] = check
            comparisons[item['case_id']] = check
        else:
            label = 'left' if item['direction'] == 1 else 'right'
            pair_zero = pair_entry33(snapshot, prior['zero_'+label][1],
                                     alpha33=item['alpha'])
            pair_fixed = pair_entry33(snapshot, prior['fixed_nonzero_'+label][1],
                                      alpha33=item['alpha'])
            require(pair_zero['passed'] and pair_fixed['passed'],
                    'C33 cancel entry failed strict old reset/prefix/teacher pair')
            check = dict(eligible=report['qualification_passed'],
                         cancel_to_handoff_controls=report['task']['cancel_to_handoff_controls'],
                         safe_cancel=report['independent_terminal_kind'] == 'safe_cancel',
                         pair_zero=pair_zero, pair_fixed=pair_fixed)
            comparisons[item['case_id']] = check
        reports[item['case_id']] = report
        online_rows.append(online)
        del episode
    ledger = global_ledger33(run, session, worker, construction, audits)
    independent = select_alpha33(development)
    selected = independent['selected_alpha33']
    require(definition['selected_alpha33'] == selection['selected_alpha33']
            == worker['result']['selected_alpha33'] == selected,
            'C33 online winner differs from independently ranked eligible fixed pairs')
    expected_stop = 'development_complete'
    first_ineligible = next((index for index, item in enumerate(cases[:dev_count])
        if not comparisons[item['case_id']]['eligible']), None)
    if first_ineligible is None:
        require(dev_count == 6,
                'C33 stopped development early while every attempted case passed')
    else:
        require(first_ineligible == dev_count-1,
                'C33 continued development after first failed candidate')
        item = cases[first_ineligible]
        if item['case_id'] in comparisons:
            expected_stop = ('controlled_task_failure_closes_higher_alpha'
                if reports[item['case_id']]['independent_terminal_kind'] == 'controlled_failure'
                else 'development_gate_failure_closes_higher_alpha')
    require(selection['stop_reason'] == definition['stop_reason']
            == worker['result']['stop_reason'] == expected_stop
            and selection['all_attempted_cases'] == online_rows
            and worker['last_case'] == online_rows[-1],
            'C33 attempted-case ledger, stop point or selection history differs')
    labels = LABELS
    if selected is None:
        require(len(cases) == dev_count and selection['cancel_conditions']['directions'] == [],
                'C33 no-winner campaign invented cancellation')
    else:
        suffix = labels[selected]
        expected = [f'cancel_{suffix}_left', f'cancel_{suffix}_right']
        require([row['case_id'] for row in cases[dev_count:]] == expected
                and [row['direction'] for row in cases[dev_count:]] == [1, -1]
                and all(row['alpha'] == selected for row in cases[dev_count:])
                and selection['cancel_conditions']['selected_alpha'] == selected
                and selection['cancel_conditions']['directions'] == [1, -1]
                and selection['cancel_conditions']['next_control_after_first_dual_velocity_overlap'] is True,
                'C33 selected-alpha cancellation conditions differ')
    expected_candidates = []
    for alpha, label in labels.items():
        sides = development.get(alpha, {})
        if set(sides) == {1, -1} and all(row['eligible'] for row in sides.values()):
            left, right = sides[1]['actual'], sides[-1]['actual']
            expected_candidates.append(dict(alpha=alpha,
                case_ids=[f'alpha_{label}_left', f'alpha_{label}_right'], eligible=True,
                max_side_duration_s=max(left['cycle_s'], right['cycle_s']),
                mean_full_tau2_mean=(left['full_torque_square_mean']+
                                     right['full_torque_square_mean'])/2.))
    require(len(selection['candidates']) == len(expected_candidates),
            'C33 online candidate list omitted an eligible pair')
    for online, expected in zip(selection['candidates'], expected_candidates):
        require(all(close(online[key], value) if isinstance(value, float)
                    else online[key] == value for key, value in expected.items()),
                'C33 online candidate ranking inputs differ from independent full metrics')
    return dict(schema='d1-c33-independent-fixed-scurve-readback-v1',
        execution_contract_id=CONTRACT, run=str(run), record_valid=True,
        session_identity=identity(run/'session.json'),
        worker_identity=identity(run/'worker_receipt.json'), source=source,
        ledger=ledger, attempted_cases=reports, comparisons=comparisons,
        old_baseline_input_identities=evidence, independent_selection=independent,
        online_selection_reproduced=True,
        selected_cancel_pair_qualified=bool(selected is not None and
            all(comparisons[name]['eligible'] for name in
                (f'cancel_{labels[selected]}_left', f'cancel_{labels[selected]}_right'))),
        fixed_engineering_gain=independent['fixed_engineering_gain'],
        RL_speed_benefit_proven=False, reader_model_calls=0, reader_physics_steps=0,
        torque_cost_is_energy=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reader-review', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'C33 independent output already exists')
    report = read_run33(args.run, reader_review_path=args.reader_review)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False,
                  default=lambda value: value.tolist() if isinstance(value, np.ndarray)
                  else value.item() if isinstance(value, np.generic)
                  else (_ for _ in ()).throw(TypeError(type(value).__name__)))
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == '__main__':
    main()
