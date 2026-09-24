"""Freeze a separate lifecycle-corrected process after the closed startup failure."""
import argparse
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
P = D.parent / 'stability_20260924_drive_damping01'
L = D.parent / 'stability_20260923_rl01'
E = D.parent / 'stability_20260923_engine01'
R = Path('/home/lyh/wheel-legged-control-lab')
CONTRACT = P / 'next_transfer_lifecycle_plan_05/next_contract.md'
CONTRACT_SHA = '489497bb23ed1ff14dbf70544ef7bf10be7f0a3e72c28760a2af18e93aa8bfde'


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
    assert len(files) == 635
    for name, row in files.items():
        assert record(name) == row, name
    archive = P / 'run_01_failed_archive_manifest_01.json'
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
    worker = D / 'run_drive_damping_evaluation_02.py'
    library = E / 'build/libepa01_engine.so'
    assert not output.exists() and not budget.exists() and not freeze.exists()
    python_argv = [str(worker), '--library', str(library), '--original-run', str(L / 'run_02'),
                   '--output', str(output), '--contract', str(CONTRACT), '--freeze', str(freeze)]
    argv = ['rtk', 'proxy', 'env', '-u', 'LD_LIBRARY_PATH', f'LD_PRELOAD={library}',
            'LD_BIND_NOW=1', 'PYTHONDONTWRITEBYTECODE=1', 'OPENBLAS_NUM_THREADS=1',
            'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
            f'PYTHONPATH=/home/lyh/.local/lib/python3.10/site-packages:{R}/.local-deps:{R}/src:{R}:{E}:{P}',
            'python3', '-B', *python_argv]
    additions = [Path(__file__), worker, D / 'launch_drive_once_02.py', CONTRACT,
                 args.review, args.snapshot, previous, archive,
                 P / 'run_01_boundary_closure_01.json']
    additions.extend(D / name for name in ('drive_fresh_lifecycle_05.py',
        'drive_damping_transfer_env_05.py', 'test_drive_fresh_lifecycle_05.py',
        'run_lifecycle_pure_tests_01.py'))
    for path in additions:
        files[str(path.resolve())] = record(path)
    value = {'schema': 'drive-damping-evaluation-freeze-v1', 'astra_decision': 'GO',
             'process_limit': 1, 'training_control_limit': 0, 'evaluation_control_limit': 4800,
             'normal_native_limit': 24000, 'compiler_native_limit': 5, 'wallclock_limit_s': 600,
             'contract_path': str(CONTRACT), 'contract_sha256': CONTRACT_SHA,
             'prior_boundary_closure': str(P / 'run_01_boundary_closure_01.json'),
             'original_run': str(L / 'run_02'), 'baseline_case_map': str(L / 'complete_cases_map_03.json'),
             'output_directory': str(output), 'budget_path': str(budget),
             'original_training_protocol_sha256': '494eb82ca80a842c190bbbff9bc75fa10c2e7832789a2d9bce0785acc3684733',
             'checkpoint_sha256': '4f59d795afbdec256ec17bb7256ea05a3ddd04561e9ba52beffb1de375196b2e',
             'python_argv': python_argv, 'execution_argv': argv, 'files': files,
             'closed_startup_control_returns': 0, 'closed_startup_compiler_returns': 5,
             'old_original_run_02_qualification': False, 'old_final_native_C_counts': None}
    with freeze.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'files': len(files), 'freeze_sha256': record(freeze)['sha256']}))


if __name__ == '__main__':
    main()
