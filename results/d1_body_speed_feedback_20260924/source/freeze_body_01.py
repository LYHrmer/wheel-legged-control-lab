"""Freeze a separate body-COM common-P process after the closed drive experiment."""
import argparse
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
P = D.parent / 'stability_20260924_drive_damping02'
D1 = D.parent / 'stability_20260924_drive_damping01'
L = D.parent / 'stability_20260923_rl01'
E = D.parent / 'stability_20260923_engine01'
R = Path('/home/lyh/wheel-legged-control-lab')
CONTRACT = P / 'next_body_speed_feedback_plan_06/next_contract.md'
CONTRACT_SHA = '13343fc40b726c0a8db6f17bbd475d44d4c303f0c5b7a5a2be2f0257b1af92e7'


def record(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    assert record(CONTRACT)['sha256'] == CONTRACT_SHA
    snapshot = load(args.snapshot)
    assert snapshot['decision'] == 'GO'
    previous = P / 'root_execution_freeze_01.json'
    files = dict(load(previous)['files'])
    assert len(files) == 662
    assert record(previous)['sha256'] == '3de560e0a46e456b72b30661ec497ec1a1cc497abf49fcf33ab46553a8bc12b9'
    for name, row in files.items():
        assert record(name) == row, name
    archive = P / 'run_01_archive_manifest_01.json'
    assert record(archive)['sha256'] == 'dc547afa31d65b099a6de93d50d796eb1f7956a2f9ac1dc26806c70d4712912c'
    assert len(load(archive)['files']) == 77
    for relative, row in load(archive)['files'].items():
        path = P / 'run_01' / relative
        assert record(path) == row, str(path)
        files[str(path)] = row
    for name, row in snapshot['files'].items():
        assert record(name) == row, name
        files[name] = row
    output = D / 'run_01'
    budget = D / 'budget_activation_01.json'
    freeze = D / 'root_execution_freeze_01.json'
    worker = D / 'run_body_speed_evaluation_01.py'
    library = E / 'build/libepa01_engine.so'
    assert not output.exists() and not budget.exists() and not freeze.exists()
    python_argv = [str(worker), '--library', str(library), '--original-run', str(L / 'run_02'),
                   '--output', str(output), '--contract', str(CONTRACT), '--freeze', str(freeze)]
    argv = ['rtk', 'proxy', 'env', '-u', 'LD_LIBRARY_PATH', f'LD_PRELOAD={library}',
            'LD_BIND_NOW=1', 'PYTHONDONTWRITEBYTECODE=1', 'OPENBLAS_NUM_THREADS=1',
            'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
            f'PYTHONPATH=/home/lyh/.local/lib/python3.10/site-packages:{R}/.local-deps:{R}/src:{R}:{E}:{D1}:{P}',
            'python3', '-B', *python_argv]
    additions = [Path(__file__), worker, D / 'launch_body_once_01.py', CONTRACT,
                 args.review, args.snapshot, previous, archive,
                 P / 'run_01_boundary_closure_01.json']
    additions.extend(D / name for name in ('body_speed_math_06.py', 'body_speed_controller_06.py',
        'body_speed_validator_06.py', 'body_speed_transfer_env_06.py',
        'test_body_common_p_opus_01.py', 'test_body_chain_root_01.py',
        'run_body_pure_tests_01.py', 'root_readback_01.py', 'body_validator_fixture_02.json.gz'))
    for path in additions:
        files[str(path.resolve())] = record(path)
    value = {'schema': 'body-common-p-evaluation-freeze-v1', 'astra_decision': 'GO',
             'process_limit': 1, 'training_control_limit': 0, 'evaluation_control_limit': 4800,
             'normal_native_limit': 24000, 'compiler_native_limit': 5, 'wallclock_limit_s': 600,
             'contract_path': str(CONTRACT), 'contract_sha256': CONTRACT_SHA,
             'prior_boundary_closure': str(P / 'run_01_boundary_closure_01.json'),
             'original_run': str(L / 'run_02'), 'baseline_case_map': str(L / 'complete_cases_map_03.json'),
             'output_directory': str(output), 'budget_path': str(budget),
             'original_training_protocol_sha256': '494eb82ca80a842c190bbbff9bc75fa10c2e7832789a2d9bce0785acc3684733',
             'checkpoint_sha256': '4f59d795afbdec256ec17bb7256ea05a3ddd04561e9ba52beffb1de375196b2e',
             'python_argv': python_argv, 'execution_argv': argv, 'files': files,
             'drive_baseline_directory': str(P / 'run_01'),
             'closed_predecessor_control_returns': 4800, 'closed_predecessor_compiler_returns': 5,
             'old_original_run_02_qualification': False, 'old_final_native_C_counts': None}
    with freeze.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'files': len(files), 'freeze_sha256': record(freeze)['sha256']}))


if __name__ == '__main__':
    main()
