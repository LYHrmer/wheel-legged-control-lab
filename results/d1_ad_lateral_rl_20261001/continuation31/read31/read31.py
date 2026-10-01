"""Cold NumPy-only C31 physical/event/learning readback from saved records."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
W = Path(__file__).resolve().parents[2]
P = W/'continuation31'
R = Path('/home/lyh/wheel-legged-control-lab')
sys.path.insert(0, str(W/'continuation26/root26'))
from reader26 import require, document, saved_document, identity
for path in (W/'continuation24', W/'continuation27/read27', W/'continuation30/read30'):
    sys.path.insert(0, str(path))
import numpy as np
from source_read31 import sources31
from episode_read31 import construction31, physical_episode31
from ledger_read31 import case_ledger31, close_ledgers31
from learning_read31 import zero_state31, verify_rollout31, verify_update31, same
from math_read31 import checked_parameters, linear_outputs
from checkpoint_read31 import checkpoint_parameters31

CONTRACT = 'C31_event_lateral_RL_pilot_v1'


def final_checkpoint31(run, session, result, parameters):
    folder = Path(run)/'final_checkpoint'
    manifest = document(folder/'manifest.json')
    receipt = saved_document(Path(run)/'final_checkpoint_receipt31.json')
    require(receipt == {**manifest, 'manifest_sha256': identity(folder/'manifest.json')['sha256']}
            and manifest['schema'] == 'd1-c31-final-lateral-linear-v1'
            and manifest['independent_reload'] is True
            and manifest['probe_observation_rows'] == 16
            and manifest['probe_actor_rows'] == manifest['probe_value_rows'] == 32
            and manifest['checkpoint_save_calls'] == manifest['checkpoint_reload_calls'] == 1,
            'final-only checkpoint identity/probe/reload differs')
    require(manifest['source_metadata'] == dict(execution_contract_id=CONTRACT,
                source_hashes=session['source_hashes'], policy_seed=session['policy_seed'])
            and all(result.get(key) == value for key, value in manifest['training_receipt'].items())
            and set(manifest['files']) == {'lateral_final.pt', 'probe_before.json', 'probe_after.json'},
            'checkpoint is not this complete frozen-source pilot')
    for name, expected in manifest['files'].items():
        require(identity(folder/name) == expected, 'final checkpoint payload changed: '+name)
    actual = checkpoint_parameters31(folder/'lateral_final.pt')
    for key in parameters:
        require(np.array_equal(actual[key], parameters[key]), 'saved .pt tensor differs from actual last update: '+key)
    before, after = document(folder/'probe_before.json'), document(folder/'probe_after.json')
    require(before == after and before['observation54'] == session['final_probe_observations54'],
            'probe changed rows or independent reload output')
    mean, value = linear_outputs(parameters, before['observation54'])
    same(before['actor_mean_z3'], mean, 'saved actor probe differs from all four decoded final tensors')
    same(before['value'], value, 'saved value probe differs from all four decoded final tensors')
    return dict(passed=True, checkpoint_path=str(folder/'lateral_final.pt'),
        checkpoint_identity=identity(folder/'lateral_final.pt'),
        four_full_tensors_equal_last_Adam_state=True, independent_linear_probe_verified=True,
        reader_torch_loads=0, reader_model_forwards=0)


def read_training31(run):
    run = Path(run).resolve(strict=True)
    session, worker = document(run/'session.json'), document(run/'worker_receipt.json')
    require(session['arm'] == 'train', 'this entrypoint requires the unique C31 training worker')
    source = sources31(run, session, worker)
    result = saved_document(run/'training_receipt31.json')
    require(result == worker['result'] and result['status'] == 'complete'
            and result['training'] is True and result['pending_cycles'] == 0
            and result['actual_batches'] == 8 and result['cycles'] == 64
            and result['independent_readback_pending'] is True,
            'training did not complete its fixed 64 cycles/eight batches')
    require({path.name for path in run.glob('episode_*') if path.is_dir()}
            == {f'episode_{i}' for i in range(64)}
            and {path.name for path in run.glob('batch_??.json')} == {f'batch_{i:02d}.json' for i in range(8)},
            'missing/extra actual episodes or learning batches')
    construction, binding, geometry, kin = construction31(run, session, R)
    original_spec = document(W/'continuation27/spec27.json')
    parameters, adam = zero_state31()
    initial = checked_parameters(saved_document(run/'initial_parameters31.json'))
    require(all(np.array_equal(initial[k], v) for k, v in parameters.items()), 'actor/value initialization is not exact zero')
    reports, audits, batch_reports = [], [], []
    counts = {}
    offset = sampled = controlled = consecutive = 0
    real_actor_change = False
    previous_access = None
    for batch_index in range(8):
        cycles = []
        for index in range(8*batch_index, 8*batch_index+8):
            report, cycle, episode = physical_episode31(run, session, construction, binding, geometry, kin,
                original_spec, index, offset, parameters)
            require(report['mode'] == 'train' and report['direction'] == (1 if index % 2 == 0 else -1)
                    and report['case_id'] == f'train_{index:02d}' and report['requested_initial_yaw_rad'] == 0.
                    and cycle['independent_learning_eligible'] is True,
                    'training list changed or unsafe/incomplete case entered learning')
            side = episode['cycle_receipt31']['side_access']
            before = side['before']
            if previous_access is None:
                require(before['starts'] == before['computes'] == before['prepares'] == before['scopes'] == 0
                        and all(v == 0 for v in before['counts'].values()), 'hidden side work before first cycle')
            else:
                require(all(before[k] == previous_access['after'][k]
                            for k in ('starts', 'computes', 'prepares', 'counts'))
                        and before['scopes'] == previous_access['scope_range'][1], 'side API reset at episode boundary')
            previous_access = side
            for macro in cycle['macros']:
                sampled += 1
                receipt = macro['skill31_latch']['policy_receipt']
                require(receipt['actor_rows_cumulative'] == receipt['value_rows_cumulative']
                        == receipt['rng_samples_cumulative'] == sampled, 'latch sampler skipped/reused a model row')
            audits.append(case_ledger31(run, session, worker, episode, report, index, offset))
            offset += episode['controls']
            failed = cycle['terminal_kind'] == 'controlled_failure'
            controlled += int(failed)
            consecutive = consecutive+1 if failed else 0
            require(consecutive < 2 and controlled < 8, 'training continued beyond preregistered task failure stop')
            progress = saved_document(run/f'progress_{index:02d}.json')
            require(progress == dict(closed_cycles=index+1, actual_controls=offset, actual_normal_native=5*offset,
                actual_macros=sampled, actual_optimizer_steps=counts.get('actual_optimizer_steps', 0),
                actual_attempts=counts.get('actual_evaluated_minibatches', 0), current_case=f'train_{index:02d}',
                terminal_kind=cycle['terminal_kind']), 'saved progress differs from actual pre-update ledger')
            reports.append(report)
            cycles.append(cycle)
            del episode
        path = run/f'batch_{batch_index:02d}.json'
        batch = saved_document(path)
        require(result['batches'][batch_index] == dict(path=str(path), **identity(path))
                and batch['cycle_receipts'] == {str(c['cycle_index']): identity(run/f"episode_{c['cycle_index']}"/'cycle_receipt.json')
                                               for c in cycles}, 'batch is not bound to these actual eight cycles')
        rollout = verify_rollout31(cycles, batch['prepared_batch'], parameters)
        update = verify_update31(batch['prepared_batch'], batch['update_receipt'], parameters, adam,
            batch_index=batch_index, permutation_seed=session['policy_seed']+batch_index)
        require(update['valid_train_call'] is True, 'invalid/no-step batch cannot qualify final pilot')
        parameters, adam = update['parameters'], update['adam']
        for key, value in update['counts'].items():
            counts[key] = counts.get(key, 0)+value
        require(counts['actual_evaluated_minibatches'] <= 64 and counts['actual_optimizer_steps'] <= 64,
                'global actual PPO attempts/steps exceeded fixed cap')
        real_actor_change |= update['actor_nonzero_gradient'] and update['actor_parameter_changed']
        batch_reports.append(dict(batch_index=batch_index, actual_cycles=rollout['actual_cycles'],
            actual_macro_samples=rollout['actual_macros'], counts=update['counts'],
            raw_gradients_group_clipping_Adam_parameters_moments_verified=True,
            normalized_advantage_and_real_duration_GAE_verified=True,
            permutation_partition_verified=True, torch_random_permutation_replayed=False))
    final = checked_parameters(saved_document(run/'final_parameters31.json'))
    require(all(np.array_equal(final[k], parameters[k]) for k in parameters)
            and real_actor_change and result['actor_nonzero_gradient_and_parameter_change'] is True
            and result['controlled_failures'] == controlled, 'final tensors/real actor change/failure count differs')
    checkpoint = final_checkpoint31(run, session, result, final)
    ledger = close_ledgers31(run, session, worker, construction, audits, counts, sampled)
    return dict(schema='d1-c31-independent-training-readback-v1', execution_contract_id=CONTRACT,
        run=str(run), session_identity=identity(run/'session.json'), worker_identity=identity(run/'worker_receipt.json'),
        record_valid=True, training_valid=True, source=source, ledger=ledger, cycles=reports, batches=batch_reports,
        checkpoint=checkpoint, controlled_task_failures=controlled, real_actor_update=True,
        reward_terminal_uses_independent_task_result_not_teacher_string=True,
        final_evaluation_qualified=False, RL_speed_benefit_proven=False, GUI_qualified=False,
        hardware_evidence=False, new_reader_model_calls=0, new_reader_physics_steps=0)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'exclusive C31 saved reader output already exists')
    report = read_training31(args.run)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False, default=json_value)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == '__main__':
    main()
