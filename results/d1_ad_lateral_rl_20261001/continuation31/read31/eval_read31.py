"""Independent C31 final-only ten-case evaluation and paired learned gains."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from read31 import (W, P, R, CONTRACT, require, document, saved_document, identity,
                    sources31, construction31, physical_episode31, case_ledger31,
                    close_ledgers31, checked_parameters, checkpoint_parameters31,
                    linear_outputs, same, json_value, np)
from pairing_read31 import entry_snapshot31, comparison31


def expected_cases31():
    result = []
    for direction, name in ((1, 'left'), (-1, 'right')):
        result.append(dict(case_id='nominal_learned_'+name, direction=direction,
            initial_yaw_rad=0., mode='learned', cancel=False))
    for direction, name, yaw in ((1, 'left', .04), (-1, 'right', -.04)):
        for mode in ('zero', 'fixed', 'learned'):
            result.append(dict(case_id='heldout_'+name+'_'+mode, direction=direction,
                initial_yaw_rad=yaw, mode=mode, cancel=False))
    for direction, name in ((1, 'left'), (-1, 'right')):
        result.append(dict(case_id='cancel_learned_'+name, direction=direction,
            initial_yaw_rad=0., mode='learned', cancel=True))
    return result


def training_bridge31(session, initial_parameters):
    report_path = Path(session['training_readback_path']).resolve(strict=True)
    train_run = Path(session['training_run']).resolve(strict=True)
    report = document(report_path)
    require(identity(report_path) == session['training_readback_identity']
            == session['source_hashes'].get(str(report_path))
            and report['execution_contract_id'] == CONTRACT and report['training_valid'] is True
            and report['record_valid'] is True and report['real_actor_update'] is True
            and Path(report['run']).resolve() == train_run
            and report['worker_identity'] == identity(train_run/'worker_receipt.json')
            and report['session_identity'] == identity(train_run/'session.json'),
            'evaluation lacks the valid unique independent pilot readback')
    final = saved_document(train_run/'final_parameters31.json')
    folder = Path(session['lateral_checkpoint_folder']).resolve(strict=True)
    manifest = document(folder/'manifest.json')
    expected = session['lateral_checkpoint_manifest']
    require(folder == train_run/'final_checkpoint'
            and expected == {**manifest, 'manifest_sha256': identity(folder/'manifest.json')['sha256']}
            and report['checkpoint']['checkpoint_identity'] == identity(folder/'lateral_final.pt')
            and Path(report['checkpoint']['checkpoint_path']).resolve() == folder/'lateral_final.pt',
            'evaluation selected a checkpoint other than the audited final')
    for name, value in manifest['files'].items():
        require(identity(folder/name) == value, 'evaluation final payload drift: '+name)
    parameters = checkpoint_parameters31(folder/'lateral_final.pt')
    require(all(np.array_equal(parameters[k], final[k]) and np.array_equal(parameters[k], initial_parameters[k])
                for k in parameters), 'evaluation actual parameters differ from all four final trained tensors')
    probe = document(folder/'probe_after.json')
    mean, value = linear_outputs(parameters, probe['observation54'])
    same(probe['actor_mean_z3'], mean, 'evaluation saved actor probe differs')
    same(probe['value'], value, 'evaluation saved value probe differs')
    execution_path = session.get('training_readback_execution_path')
    if execution_path is not None:
        require(session['source_hashes'].get(str(Path(execution_path).resolve())) == identity(execution_path),
                'training reader execution receipt lacks frozen identity')
    return parameters, dict(passed=True, training_readback_path=str(report_path),
        training_readback_identity=identity(report_path),
        final_checkpoint_identity=identity(folder/'lateral_final.pt'), four_full_tensors_equal=True,
        reader_model_loads=0, earlier_checkpoint_selection=False)


def read_evaluation31(run):
    run = Path(run).resolve(strict=True)
    session, worker = document(run/'session.json'), document(run/'worker_receipt.json')
    require(session['arm'] == 'evaluation' and session['evaluation_cases'] == expected_cases31(),
            'final evaluation list differs from fixed preregistered ten actual cases')
    source = sources31(run, session, worker, reader_review_path=P/'eval_reader_source_review_31.json')
    spec = document(P/'spec31_draft.json')['evaluation']
    expected_gains = dict(learned_over_zero_cycle_max=.90, learned_over_fixed_cycle_max=.95,
        goal_error_over_zero_delta_max_m=.002, retention_absolute_min=.9,
        retention_over_zero_delta_min=-.02, full_tau2_over_zero_max=1.20, full_tau2_over_fixed_max=1.10)
    require(all(spec[k] == value for k, value in expected_gains.items()), 'final scientific gates drifted')
    result = saved_document(run/'evaluation_receipt31.json')
    require(result == worker['result'] and result['status'] == 'complete' and result['training'] is False
            and result['actual_batches'] == result['actual_optimizer_steps'] == result['actual_minibatch_attempts'] == 0
            and result['cycles'] == 10 and result['pending_cycles'] == 0
            and {p.name for p in run.glob('episode_*') if p.is_dir()} == {f'episode_{i}' for i in range(10)},
            'final evaluation omitted a case or updated the lateral policy')
    initial = checked_parameters(saved_document(run/'initial_parameters31.json'))
    parameters, bridge = training_bridge31(session, initial)
    final = checked_parameters(saved_document(run/'final_parameters31.json'))
    require(all(np.array_equal(initial[k], final[k]) for k in final), 'evaluation changed actor/value parameters')
    construction, binding, geometry, kin = construction31(run, session, R)
    original_spec = document(W/'continuation27/spec27.json')
    reports, snapshots, audits = {}, {}, []
    offset = sampled = 0
    prior_access = None
    for index, case in enumerate(session['evaluation_cases']):
        report, cycle, episode = physical_episode31(run, session, construction, binding, geometry, kin,
            original_spec, index, offset, parameters)
        require(all(report[k] == case[k] for k in ('case_id', 'mode', 'direction'))
                and report['requested_initial_yaw_rad'] == case['initial_yaw_rad']
                and (episode['cycle_receipt31']['result']['cancel_control_index'] is not None) is case['cancel'],
                'actual final evaluation condition/actor/cancellation differs')
        access = episode['cycle_receipt31']['side_access']
        before = access['before']
        if prior_access is None:
            require(before['starts'] == before['computes'] == before['prepares'] == before['scopes'] == 0
                    and all(v == 0 for v in before['counts'].values()), 'hidden work before evaluation')
        else:
            require(all(before[k] == prior_access['after'][k] for k in ('starts', 'computes', 'prepares', 'counts'))
                    and before['scopes'] == prior_access['scope_range'][1], 'evaluation reset refunded side API work')
        prior_access = access
        for macro in cycle['macros']:
            latch = macro['skill31_latch']
            if case['mode'] == 'learned':
                sampled += 1
                receipt = latch['policy_receipt']
                require(receipt['actor_rows_cumulative'] == sampled
                        and receipt['value_rows_cumulative'] == receipt['rng_samples_cumulative'] == 0,
                        'learned evaluation sampled RNG/value or invented event actor rows')
            else:
                require(latch['policy_receipt'] is None, 'zero/fixed invoked lateral actor')
        audits.append(case_ledger31(run, session, worker, episode, report, index, offset))
        offset += episode['controls']
        snapshots[case['case_id']] = entry_snapshot31(run, session, construction, episode)
        reports[case['case_id']] = report
        del episode
    ledger = close_ledgers31(run, session, worker, construction, audits, {}, sampled)
    comparison = comparison31(session, reports, snapshots)
    physical = all(report['qualification_passed'] for report in reports.values())
    return dict(schema='d1-c31-independent-final-evaluation-v1', execution_contract_id=CONTRACT,
        run=str(run), session_identity=identity(run/'session.json'), worker_identity=identity(run/'worker_receipt.json'),
        record_valid=True, source=source, training_to_final_bridge=bridge, ledger=ledger,
        cases=reports, paired_gain_evaluation=comparison, physical_qualification_passed=physical,
        learned_research_candidate_qualified=bool(physical and comparison['passed']),
        RL_speed_benefit_gate_passed=bool(comparison['passed']), no_retraining_or_checkpoint_selection=True,
        reused_nominal_not_called_new_heldout=True, GUI_qualified=False, hardware_evidence=False,
        torque_cost_is_energy=False, reader_model_calls=0, reader_physics_steps=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'exclusive evaluation readback already exists')
    report = read_evaluation31(args.run)
    with args.output.open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False, default=json_value)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == '__main__':
    main()
