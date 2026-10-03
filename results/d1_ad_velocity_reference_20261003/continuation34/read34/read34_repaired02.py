"""Cold saved-only C34 actual-distance and velocity-wrench campaign readback."""
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
             W/'continuation30/pair_read30', W/'continuation31/read31',
             W/'continuation33/read33', W/'continuation34/read34'):
    sys.path.insert(0, str(path))

import numpy as np
from reader26 import require, close, identity, document, saved_document
from episode_read31 import construction31, local_episode31
from ledger_read31 import case_ledger31
from pairing_read31 import entry_snapshot31
from comparison_read33 import prior_baselines33
from episode_read34 import physical_episode34
from pairing_read34 import pair_entry34
from ledger_read34 import global_ledger34
from source_read34 import sources34, CONTRACT


def _metrics(report):
    task = report['task']
    full = report['macros']['full_scene_components']
    return dict(full_tau2_mean=full['torque_normalized_square_mean'],
        full_tau2_integral_s=full['torque_normalized_square_integral_s'],
        full_elapsed_s=full['elapsed_s'], side_duration_s=task['side_duration_s'],
        final_xy_error_m=task['final_xy_error_m'],
        retention_ratio=task['retention_ratio'],
        safety_complete=all(report['online_mandatory_gates'].values()),
        entry_gates=task['entry_gates'])


def _quality(report, old, *, distance_m, same_task_zero=None, cancel=False):
    task = report['task']
    metrics = _metrics(report)
    retained = (task['signed_lateral_m']*task['retention_ratio']
                if task['retention_ratio'] is not None else None)
    goal_finished = bool(not cancel and report['qualification_passed'])
    metrics.update(signed_lateral_m=task['signed_lateral_m'],
        retained_signed_lateral_m=retained,
        nominal_commanded_speed_mps=(distance_m/task['side_duration_s']
            if goal_finished and task['side_duration_s'] else None),
        actual_retained_progress_per_cycle_mps=(retained/task['side_duration_s']
            if goal_finished and retained is not None and task['side_duration_s'] else None))
    if cancel:
        metrics['eligible'] = bool(report['qualification_passed'] and metrics['safety_complete'])
        metrics['cancel_to_handoff_controls'] = task['cancel_to_handoff_controls']
        return metrics
    zero, fixed = old['zero'], old['fixed']
    cost = metrics['full_tau2_mean']
    retention = metrics['retention_ratio']
    gates = dict(original_safety_and_task=bool(report['qualification_passed']
                                              and metrics['safety_complete']),
        entry=all(metrics['entry_gates'].values()),
        goal_error_vs_zero=metrics['final_xy_error_m'] <= zero['final_xy_error_m']+.002,
        retention_vs_zero=retention is not None and
            retention >= max(.90, zero['retention_ratio']-.02),
        tau2_vs_fixed=cost <= 1.10*fixed['full_tau2_mean'],
        tau2_vs_zero=cost <= 1.20*zero['full_tau2_mean'])
    if same_task_zero is not None:
        gates.update(goal_error_vs_d40_beta0=metrics['final_xy_error_m'] <=
                same_task_zero['final_xy_error_m']+.002,
            retention_vs_d40_beta0=retention is not None and
                retention >= max(.90, same_task_zero['retention_ratio']-.02),
            tau2_vs_d40_beta0=cost <= 1.10*same_task_zero['full_tau2_mean'])
    metrics['gates'] = gates
    metrics['eligible'] = all(gates.values())
    return metrics


def _compare_online(saved, expected, *, distance_m, cancel):
    require(set(saved) == set(expected),
            'C34 online metrics keys differ from independent full case')
    for key, value in expected.items():
        require(saved[key] == value if isinstance(value, (bool, dict)) or value is None
                else close(saved[key], value),
                'C34 online metric differs from independent full case: '+key)
    require(expected['nominal_commanded_speed_mps'] is None if cancel else True,
            'C34 cancellation cannot claim commanded-goal speed')
    require(close(expected['full_tau2_mean'],
                  expected['full_tau2_integral_s']/expected['full_elapsed_s']),
            'C34 cost is not the original full-scene native mean')


def _pair(left, right, *, distance_m, beta, archived=False):
    require((left['direction'], right['direction']) == (1, -1)
            and left['distance_m'] == right['distance_m'] == distance_m
            and left['beta'] == right['beta'] == beta,
            'C34 complete setting pair direction/target differs')
    lm, rm = left['metrics'], right['metrics']
    return dict(distance_m=distance_m, beta=beta,
        case_ids=[left['case_id'], right['case_id']], archived=archived,
        complete=True,
        eligible=bool(lm['eligible'] and rm['eligible']),
        max_side_duration_s=max(lm['side_duration_s'], rm['side_duration_s']),
        mean_full_tau2_mean=(lm['full_tau2_mean']+rm['full_tau2_mean'])/2.)


def _best(pairs):
    allowed = [pair for pair in pairs if pair['eligible']]
    return min(allowed, key=lambda row: (row['max_side_duration_s'],
        row['mean_full_tau2_mean'], row['beta'])) if allowed else None


def _archive33(session):
    run = Path(session['archived_c33_run_path']).resolve(strict=True)
    report_path = Path(session['archived_c33_report_path']).resolve(strict=True)
    report = document(report_path)
    selection = saved_document(run/'campaign_selection33.json')
    require(session['source_hashes'][str(report_path)] == identity(report_path)
            and session['source_hashes'][str(run/'campaign_selection33.json')]
                == identity(run/'campaign_selection33.json')
            and report['record_valid'] is True and report['run'] == str(run)
            and report['selected_cancel_pair_qualified'] is True
            and selection['selected_alpha33'] == 1.,
            'C34 archived independently qualified C33 beta0 anchor differs')
    c33_session = document(run/'session.json')
    construction, binding, geometry, kin = construction31(run, c33_session, R)
    by_id = {row['case_id']: row for row in selection['all_attempted_cases']}
    snapshots, rows = {}, []
    for side in ('left', 'right'):
        case_id = session['archived_c33_cases'][side]
        cancel_id = session['archived_c33_cancel_cases'][side]
        online = by_id[case_id]
        require(case_id == f'alpha_100_{side}'
                and cancel_id == f'cancel_100_{side}'
                and online['metrics']['eligible'] is True
                and report['attempted_cases'][case_id]['record_valid'] is True
                and report['attempted_cases'][cancel_id]['record_valid'] is True,
                'C34 archived C33 case/cancel pair differs')
        index = online['cycle_index']
        receipt_path = run/f'episode_{index}'/'cycle_receipt.json'
        receipt = saved_document(receipt_path)
        require(session['source_hashes'][str(receipt_path)] == identity(receipt_path)
                and online['cycle_receipt_identity'] == identity(receipt_path),
                'C34 archived C33 cycle receipt escaped source closure')
        offset = receipt['result']['global_control_begin']
        episode = local_episode31(run, c33_session, construction, binding,
                                  geometry, kin, index, offset)
        snapshots[side] = entry_snapshot31(run, c33_session, construction, episode)
        rows.append(online)
        del episode
    archive = dict(report=str(report_path), report_identity=identity(report_path),
        selection=str(run/'campaign_selection33.json'),
        selection_identity=identity(run/'campaign_selection33.json'),
        rows=rows, cancel_case_ids=session['archived_c33_cancel_cases'])
    return archive, snapshots


def _plan(selected_a):
    first = [(f'd030_beta{tag}_{side}', .03, beta, direction)
        for beta, tag in ((.5, '050'), (1., '100'))
        for side, direction in (('left', 1), ('right', -1))]
    second = [(f'd040_beta000_{side}', .04, 0., direction)
        for side, direction in (('left', 1), ('right', -1))]
    if selected_a != 0.:
        tag = {0.5: '050', 1.: '100'}[selected_a]
        second += [(f'd040_beta{tag}_{side}', .04, selected_a, direction)
            for side, direction in (('left', 1), ('right', -1))]
    return first, second


def read_run34(run, *, reader_review_path):
    run = Path(run).resolve(strict=True)
    session = document(run/'session.json')
    worker = document(run/'worker_receipt.json')
    source = sources34(run, session, worker,
                       reader_review_path=reader_review_path)
    definition = saved_document(run/'case_definition34.json')
    selection = saved_document(run/'campaign_selection34.json')
    require(definition['schema'] == 'd1-c34-case-definition-v1'
            and selection['schema'] == 'd1-c34-campaign-selection-v1'
            and selection['qualification'] is False
            and selection['independent_readback_pending'] is True,
            'C34 attempted-case/selection archive differs')
    cases = definition['cases']
    require(1 <= len(cases) <= 10
            and {path.name for path in run.glob('episode_*') if path.is_dir()}
                == {f'episode_{i}' for i in range(len(cases))},
            'C34 attempted episode count or directories differ')
    construction, binding, geometry, kin = construction31(run, session, R)
    spec = document(W/'continuation27/spec27.json')
    prior, old_evidence = prior_baselines33(session)
    old_online = saved_document(run/'baseline34.json')
    for side in ('left', 'right'):
        for name, mode in (('zero', 'zero'), ('fixed_nonzero', 'fixed')):
            saved = old_online[side][mode]
            expected = prior[f'{name}_{side}'][0]['metrics']
            path = W/'continuation30'/f'independent_{name}_{side}_01.json'
            require(saved['source_path'] == str(path)
                    and saved['source_identity'] == identity(path)
                    and close(saved['side_duration_s'], expected['cycle_s'])
                    and close(saved['final_xy_error_m'], expected['goal_error_m'])
                    and close(saved['retention_ratio'], expected['retention_ratio'])
                    and close(saved['full_tau2_mean'], expected['full_torque_square_mean']),
                    'C34 old zero/fixed anchor differs from accepted C30')
    archive, c33_snapshots = _archive33(session)
    require(saved_document(run/'archived_c33_reference34.json') == archive
            and selection['archived_C33_reference'] == archive,
            'C34 archived beta0 selection anchor differs')
    reports, comparisons, audits, online_rows = {}, {}, [], []
    development = []
    d40_zero = {}
    offset = 0
    previous_access = None
    dev_count = sum(row['kind'] == 'development' for row in cases)
    require(all(row['kind'] == 'development' for row in cases[:dev_count])
            and all(row['kind'] == 'cancel' for row in cases[dev_count:])
            and dev_count <= 8 and len(cases)-dev_count in (0, 2),
            'C34 development/cancel ordering differs')
    first_plan, _ = _plan(0.)
    for index, item in enumerate(cases):
        require(item['cycle_index'] == index
                and item['incoming_hybrid_distance_m'] == .03
                and item['requested_distance_m'] == item['distance_m']
                and item['applied_fast_distance_m'] == item['distance_m']
                and item['cycle_receipt_identity'] == identity(
                    run/f'episode_{index}'/'cycle_receipt.json'),
                'C34 case definition is not bound to real episode')
        online = saved_document(run/f'case34_{index:02d}.json')
        require(all(online[key] == item[key] for key in
                    ('cycle_index', 'case_id', 'distance_m', 'beta',
                     'direction', 'kind', 'cycle_receipt_identity'))
                and online['incoming_hybrid_distance_m'] == .03
                and online['requested_distance_m'] == item['distance_m']
                and online['applied_fast_distance_m'] == item['distance_m']
                and online['body_beta34'] == item['beta']
                and online['body_alpha33'] == 1.,
                'C34 online case differs from preregistered D/beta definition')
        report, cycle, episode = physical_episode34(run, session, construction,
            binding, geometry, kin, spec, index, offset,
            beta34=item['beta'], distance_m=item['distance_m'])
        require(report['case_id'] == item['case_id'] and report['mode'] == 'fixed'
                and report['direction'] == item['direction']
                and report['requested_initial_yaw_rad'] == 0.
                and (cycle['terminal_kind'] == 'safe_cancel') is (item['kind'] == 'cancel')
                and online['terminal_kind'] == cycle['terminal_kind']
                and online['actual_controls'] == episode['controls']
                and online['policy_predictions'] == episode['policy_predictions'],
                'C34 actual fixed episode/terminal/D case differs')
        access = episode['cycle_receipt31']['side_access']
        before = access['before']
        if previous_access is None:
            require(all(before[key] == 0 for key in
                    ('starts', 'computes', 'prepares', 'scopes'))
                    and all(value == 0 for value in before['counts'].values()),
                    'C34 hidden side work before first case')
        else:
            require(all(before[key] == previous_access['after'][key] for key in
                    ('starts', 'computes', 'prepares', 'counts'))
                    and before['scopes'] == previous_access['scope_range'][1],
                    'C34 side API work was reset/refunded between cases')
        previous_access = access
        require(all(event['skill31_latch']['policy_receipt'] is None
                    for event in cycle['learning_cycle']['macros']),
                'C34 fixed mode invoked lateral actor/value')
        side = 'left' if item['direction'] == 1 else 'right'
        snapshot = entry_snapshot31(run, session, construction, episode)
        pair_c33 = pair_entry34(snapshot, c33_snapshots[side],
            direction=item['direction'], distance_m=item['distance_m'],
            archived_c33=True)
        pair_zero = pair_entry34(snapshot, prior['zero_'+side][1],
            direction=item['direction'], distance_m=item['distance_m'])
        pair_fixed = pair_entry34(snapshot, prior['fixed_nonzero_'+side][1],
            direction=item['direction'], distance_m=item['distance_m'])
        require(pair_c33['passed'] and pair_zero['passed'] and pair_fixed['passed'],
                'C34 entry failed strict archived/C30 reset/B22/teacher pairing')
        same_task = d40_zero.get(side) if item['distance_m'] == .04 and item['beta'] else None
        expected = _quality(report, old_online[side], distance_m=item['distance_m'],
                            same_task_zero=same_task,
                            cancel=item['kind'] == 'cancel')
        _compare_online(online['metrics'], expected,
                        distance_m=item['distance_m'], cancel=item['kind'] == 'cancel')
        audits.append(case_ledger31(run, session, worker, episode, report, index, offset))
        offset += episode['controls']
        progress = saved_document(run/f'progress34_{index:02d}.json')
        require(progress == dict(closed_cases=index+1, controls=offset,
            normal_native=5*offset, macros=sum(a['macros'] for a in audits),
            policy_predictions=sum(a['policy_predictions'] for a in audits),
            last_case=online),
            'C34 cumulative progress differs from saved cases')
        comparisons[item['case_id']] = dict(metrics=expected,
            pair_archived_C33=pair_c33, pair_C30_zero=pair_zero,
            pair_C30_fixed=pair_fixed)
        reports[item['case_id']] = report
        online_rows.append(online)
        if item['kind'] == 'development':
            development.append(dict(online, metrics=expected))
            if item['distance_m'] == .04 and item['beta'] == 0. and expected['eligible']:
                d40_zero[side] = expected
        del episode
    ledger = global_ledger34(run, session, worker, construction, audits)
    require(selection['all_attempted_cases'] == online_rows
            and worker['last_case'] == online_rows[-1],
            'C34 attempted-case ledger differs from actual archives')
    archived_rows = [dict(row, distance_m=.03, beta=0.) for row in archive['rows']]
    archived_pair = _pair(*archived_rows, distance_m=.03, beta=0., archived=True)
    pairs = [archived_pair]
    first_ineligible = next((i for i, row in enumerate(development)
        if not row['metrics']['eligible'] or row['terminal_kind'] == 'controlled_failure'), None)
    require(first_ineligible is None or first_ineligible == dev_count-1,
            'C34 continued development after first controlled quality failure')
    expected_stop = ('controlled_task_or_quality_failure' if first_ineligible is not None
                     else 'development_complete')
    require(selection['stop_reason'] == definition['stop_reason']
            == worker['result']['stop_reason'] == expected_stop,
            'C34 actual stop reason differs')
    for j in range(0, min(dev_count, 4), 2):
        if j+1 < dev_count:
            left, right = development[j:j+2]
            if left['metrics']['eligible'] and right['metrics']['eligible']:
                pairs.append(_pair(left, right, distance_m=.03,
                                   beta=left['beta']))
    selected_a = _best([pair for pair in pairs if pair['distance_m'] == .03])
    require(selected_a is not None and selection['selected_stage_A_beta'] == selected_a['beta'],
            'C34 Stage A selected beta differs from independent pair ranking')
    _, second_plan = _plan(selected_a['beta'])
    expected_dev = first_plan+second_plan
    require([(row['case_id'], row['distance_m'], row['beta'], row['direction'])
             for row in development] == expected_dev[:dev_count]
            and (first_ineligible is not None or dev_count == len(expected_dev))
            and (dev_count <= 4 or all(row['metrics']['eligible'] for row in development[:4])),
            'C34 development did not follow fixed stage order/stop rule')
    for j in range(4, dev_count, 2):
        if j+1 < dev_count:
            left, right = development[j:j+2]
            if left['metrics']['eligible'] and right['metrics']['eligible']:
                pairs.append(_pair(left, right, distance_m=.04,
                                   beta=left['beta']))
    selected_distance = max(pair['distance_m'] for pair in pairs if pair['eligible'])
    selected = _best([pair for pair in pairs if pair['distance_m'] == selected_distance])
    require(selection['selected_setting34'] == definition['selected_setting34']
            == worker['result']['selected_setting34'] == selected,
            'C34 online selected setting differs from independent complete pairs')
    by_id = {row['case_id']: row for row in archived_rows+development}
    zero40 = next((pair for pair in pairs if pair['distance_m'] == .04 and pair['beta'] == 0.), None)
    candidate_rows = []
    for pair in pairs:
        left, right = (by_id[name] for name in pair['case_ids'])
        times = dict(left=left['metrics']['side_duration_s'],
                     right=right['metrics']['side_duration_s'])
        c33 = dict(left=archive['rows'][0]['metrics']['side_duration_s'],
                   right=archive['rows'][1]['metrics']['side_duration_s'])
        milestone = (all(times[side] <= .95*c33[side] for side in ('left', 'right'))
            if pair['distance_m'] == .03 else
            all(.04/times[side] >= 1.25*(.03/c33[side]) for side in ('left', 'right')))
        gain = None
        if pair['distance_m'] == .04 and pair['beta'] and zero40:
            ztime = dict(left=by_id[zero40['case_ids'][0]]['metrics']['side_duration_s'],
                         right=by_id[zero40['case_ids'][1]]['metrics']['side_duration_s'])
            gain = all(times[side] <= .95*ztime[side] for side in ('left', 'right'))
        candidate_rows.append(dict(pair, side_cycle_s=times,
            nominal_D_over_T_mps={side: pair['distance_m']/times[side]
                                   for side in ('left', 'right')},
            distance_specific_milestone=milestone,
            d40_vs_beta0_five_percent_faster=gain))
    require(len(selection['candidates']) == len(candidate_rows)
            and all(all(close(saved[key], expected) if isinstance(expected, float)
                        else saved[key] == expected for key, expected in candidate.items())
                    for saved, candidate in zip(selection['candidates'], candidate_rows)),
            'C34 online candidate ranking/milestones differ from saved full cases')
    if selected['archived']:
        require(len(cases) == dev_count and selection['cancel_conditions']['reused_archived_cancel'] is True,
                'C34 archived beta0 selection re-ran a cancellation')
    else:
        tag = {0.: '000', .5: '050', 1.: '100'}[selected['beta']]
        dist = '030' if selected['distance_m'] == .03 else '040'
        require([row['case_id'] for row in cases[dev_count:]] ==
                [f'cancel_d{dist}_beta{tag}_left', f'cancel_d{dist}_beta{tag}_right']
                and all(comparisons[row['case_id']]['metrics']['eligible'] for row in cases[dev_count:])
                and selection['cancel_conditions']['reused_archived_cancel'] is False,
                'C34 selected new setting cancellation pair differs')
    require(selection['cancel_conditions']['directions'] == [1, -1]
            and selection['cancel_conditions']['next_control_after_first_dual_velocity_overlap'] is True,
            'C34 cancellation trigger/directions differ')
    return dict(schema='d1-c34-independent-fixed-tracking-readback-v1',
        execution_contract_id=CONTRACT, run=str(run), record_valid=True,
        session_identity=identity(run/'session.json'),
        worker_identity=identity(run/'worker_receipt.json'), source=source,
        ledger=ledger, attempted_cases=reports, comparisons=comparisons,
        old_baseline_input_identities=old_evidence,
        archived_C33_reference_identity=archive['report_identity'],
        independent_selection=selected, online_selection_reproduced=True,
        selected_cancel_pair_qualified=True,
        C34_40mm_throughput_milestone=bool(any(row['distance_m'] == .04
            and row['eligible'] and row['distance_specific_milestone'] for row in candidate_rows)),
        RL_speed_benefit_proven=False, user_goal_complete=False,
        reader_model_calls=0, reader_physics_steps=0, torque_cost_is_energy=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reader-review', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'C34 independent output already exists')
    report = read_run34(args.run, reader_review_path=args.reader_review)
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
