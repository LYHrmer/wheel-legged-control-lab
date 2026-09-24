"""Create a portable full-evidence release asset from closed runs only."""
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

D = Path(__file__).resolve().parent
B = D.parent
L = B / 'stability_20260923_rl01'
P = B / 'stability_20260924_drive_damping01'
E = B / 'stability_20260923_epa01'
W = B / 'stability_20260923_engine01'


def record(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


def main():
    assert read(D / 'root_run_01_readback_01.json')['passed']
    assert read(D / 'run_01/drive_evaluation_receipt.json')['execution_valid']
    assert read(D / 'run_01_boundary_closure_01.json')['reservation_closed']
    mapping = {}
    runs = [('run_02', L / 'run_02', L / 'run_02_interrupted_archive_manifest_01.json'),
            ('run_03', L / 'run_03', L / 'run_03_archive_manifest_01.json'),
            ('drive_failed_04', P / 'run_01', P / 'run_01_failed_archive_manifest_01.json'),
            ('drive_run_01', D / 'run_01', D / 'run_01_archive_manifest_01.json')]
    for label, root, manifest in runs:
        expected = read(manifest)['files']
        actual = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}
        assert actual == set(expected), label
        for relative, row in expected.items():
            source = root / relative
            assert source.resolve().is_relative_to(root.resolve())
            assert record(source) == row, str(source)
            mapping[f'{label}/{relative}'] = source
        mapping[f'manifests/{label}.json'] = manifest
    for label, root in [('engine_run_d', W / 'run_d'), ('epa01', E)]:
        for source in sorted(root.rglob('*')):
            if source.is_file() and '__pycache__' not in source.parts:
                assert not source.is_symlink()
                mapping[f'{label}/{source.relative_to(root).as_posix()}'] = source
    extras = {
        'readbacks/original_interrupted.json': L / 'root_run_02_interrupted_readback_01.json',
        'readbacks/continuation.json': L / 'root_run_03_readback_01.json',
        'readbacks/drive_intervention.json': D / 'root_run_01_readback_01.json',
        'readbacks/engine_d.json': W / 'root_d_readback_01.json',
        'closures/original_interrupted.json': L / 'run_02_boundary_closure_01.json',
        'closures/continuation.json': L / 'run_03_boundary_closure_01.json',
        'closures/drive_failed_04.json': P / 'run_01_boundary_closure_01.json',
        'closures/drive_intervention.json': D / 'run_01_boundary_closure_01.json',
        'contracts/rolling_rl.md': W / 'next_rolling_rl_speed_plan_01/next_contract.md',
        'contracts/drive_04.md': L / 'next_drive_damping_speed_plan_04/next_contract.md',
        'contracts/lifecycle_05.md': P / 'next_transfer_lifecycle_plan_05/next_contract.md',
    }
    mapping.update(extras)
    identities = {name: record(path) for name, path in sorted(mapping.items())}
    asset = D / 'wheel-legged-rolling-rl-speed-20260924-raw.tar.gz'
    with (asset.open('xb') as raw,
          gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz,
          tarfile.open(fileobj=gz, mode='w|') as tar):
        for name, source in sorted(mapping.items()):
            info = tarfile.TarInfo(name)
            info.size = identities[name]['bytes']
            info.mode = 0o644
            with source.open('rb') as stream:
                tar.addfile(info, stream)
        payload = (json.dumps({'schema': 'closed-rolling-full-evidence-archive-v1',
                              'files': identities}, indent=2, sort_keys=True) + '\n').encode()
        info = tarfile.TarInfo('archive_manifest.json')
        info.size = len(payload)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(payload))
    assert all(record(path) == identities[name] for name, path in mapping.items())
    receipt = {'asset': str(asset), **record(asset), 'file_count': len(mapping) + 1,
               'source_files_unchanged': True, 'contains_interrupted_prefix': True,
               'contains_failed04_startup': True, 'old_run02_final_C_counts': None,
               'scope': 'full raw sources for eight baseline cases, four fixed interventions and prior engine/EPA evidence'}
    with (D / 'raw_archive_receipt_01.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
