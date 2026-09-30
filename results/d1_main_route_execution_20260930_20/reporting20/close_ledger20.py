"""Read completed worker ledgers and hash B's partial archives, without replay."""
from pathlib import Path
import hashlib
import json

C = Path(__file__).resolve().parent.parent


def read(path):
    return json.loads(path.read_text())


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def main():
    output = C / 'execution_ledger_20.json'
    if output.exists():
        raise FileExistsError(output)
    rows = {}
    inputs = {}
    totals = {'physical_controls': 0, 'normal_native': 0, 'compiler_native': 0,
              'actor_rows': 0, 'critic_rows': 0, 'optimizer_steps': 0}
    for arm in ('A', 'B'):
        folder = C / f'train_{arm}_1'
        host = read(folder / 'host_receipt.json')
        worker = read(folder / 'worker_receipt.json')
        clip = read(folder / 'clip_audit.json')
        for name in ('host_receipt.json', 'worker_receipt.json', 'clip_audit.json'):
            inputs[str(folder / name)] = identity(folder / name)
        assert host['postcheck_complete'] and host['no_live_owned_processes']
        assert host['fully_reserved_budget_closed'] and not host['changed_sources']
        assert not host['cleanup_errors'] and not worker['archive_failed']
        physics = worker['python']
        count = physics['control_completed']
        assert count == physics['control_attempted']
        assert physics['native_attempted'] == physics['native_returned'] == count * 5
        assert worker['C_final']['control_attempts'] == worker['C_final']['control_returns'] == count * 5
        assert worker['C_final']['construction_attempts'] == worker['C_final']['construction_returns'] == 2
        assert clip['optimizer_attempted'] == clip['optimizer_returned']
        assert clip['native_clip_attempted'] == clip['native_clip_returned'] == 2 * clip['optimizer_returned']
        calls = worker['model_calls']
        assert calls['counts']['backward']['returned'] == clip['optimizer_returned']
        blocks = read(folder / 'training/training_blocks_manifest.json')
        row = {
            'physical_controls': count, 'normal_native': count * 5, 'compiler_native': 2,
            'actor_rows': calls['actor_rows'], 'critic_rows': calls['critic_rows'],
            'optimizer_steps': clip['optimizer_returned'],
            'model_api_counts': calls['counts'],
            'evaluated_batches': len(clip['evaluated_batches']),
            'max_actual_batch_kl': max(x['approx_kl'] for x in clip['evaluated_batches']),
            'hard_kl_stop_batches': sum(x['hard_stop'] for x in clip['evaluated_batches']),
            'normal_kl_stop_batches': sum(x['normal_kl_stop'] for x in clip['evaluated_batches']),
            'worker_execution_complete': worker['execution_complete'],
            'worker_failure': worker['failure'], 'host_elapsed_s': host['elapsed_s'],
            'numeric_rows': sum(x['rows'] for x in blocks['numeric_blocks']),
            'full_control_rows': sum(x['rows'] for x in blocks['full_control_blocks20']),
            'gaussian_rows': blocks['gaussian_records'],
            'pending_gaussian_control_index': blocks['pending_gaussian_control_index'],
        }
        for key in totals:
            totals[key] += row[key]
        rows[arm] = row
    assert rows['A']['physical_controls'] == 32768 and rows['A']['worker_execution_complete']
    assert rows['B']['physical_controls'] == 18000 and not rows['B']['worker_execution_complete']
    assert rows['B']['numeric_rows'] == rows['B']['full_control_rows'] == 18000
    assert rows['B']['gaussian_rows'] == 17999 and rows['B']['pending_gaussian_control_index'] == 17999
    a = read(C / 'train_A_1_readback.json')
    assert a['training_valid'] and a['coverage_valid']
    assert read(C / 'first1024_pair_20.json')['passed']
    for name in ('train_A_1_readback.json', 'train_A_1_readback_host_receipt.json',
                 'first1024_pair_20.json', 'failure_geometry_snapshot_20.json'):
        inputs[str(C / name)] = identity(C / name)
    checked = []
    for manifest in sorted((C / 'train_B_1').rglob('*.manifest.json')):
        value = read(manifest)
        for item in value['payloads']:
            payload = manifest.parent / item['file']
            expected = {key: item[key] for key in ('bytes', 'sha256')}
            assert identity(payload) == expected, str(payload)
            checked.append(str(payload.relative_to(C)))
    result = {
        'schema': 'd1-c20-closed-execution-ledger-v1', 'arms': rows, 'actual_totals': totals,
        'reserved_training_controls': 65536, 'unused_reserved_controls_are_not_refunded': True,
        'source_inputs_and_hosts_closed': True, 'inputs': inputs,
        'B_atomic_payload_hashes_checked': checked,
        'B_full_independent_training_readback_performed': False,
        'B_partial_evidence_is_not_a_valid_training_record': True,
        'B_callback_gap_explanation': 'The 18000th physical step completed, then DummyVecEnv reset raised before returning that transition to the callback. Gaussian/learning transitions stop at 17999; this discrepancy is retained, not repaired.',
        'development_controls': 0, 'final_controls': 0, 'stage2_controls': 0,
        'complete_AB_comparison': False, 'continue_training': False,
        'interpretation': 'inconclusive_execution_contract_geometry_failure',
        'new_model_calls_by_this_script': 0, 'new_physics_calls_by_this_script': 0,
    }
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(output), 'totals': totals,
                      'B_payload_hash_checks': len(checked), 'arms': {
                          k: {n: v[n] for n in ('max_actual_batch_kl', 'hard_kl_stop_batches',
                                                'normal_kl_stop_batches')} for k, v in rows.items()}}))


if __name__ == '__main__':
    main()
