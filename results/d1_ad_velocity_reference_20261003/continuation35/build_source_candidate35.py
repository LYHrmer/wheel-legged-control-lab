"""Bind unchanged runtime, C35 source and actual cold import closure; no models."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
REPOSITORY = Path('/home/lyh/wheel-legged-control-lab')


def identity(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE/'source_candidate35_01.json')
    parser.add_argument('--origins', type=Path, default=HERE/'reader_origins35.json')
    args = parser.parse_args()
    destination = args.output
    if destination.exists():
        raise FileExistsError(destination)
    parent_path = WORK / 'continuation34/source_go34.json'
    parent = json.loads(parent_path.read_text())
    inputs = dict(parent['inputs'])
    mismatches = [path for path, expected in inputs.items()
                  if identity(path) != expected]
    if mismatches:
        raise RuntimeError('Inherited sources changed: ' + repr(mismatches))
    shared_path = HERE / 'astra_plan/shared_session35.json'
    shared = json.loads(shared_path.read_text())
    spec_path = HERE / 'astra_plan/spec35.json'
    spec = json.loads(spec_path.read_text())
    contract_id = 'C35_continuous_diagonal_pair_fixed_feasibility_v1'
    for name in ('execution_contract_id', 'numerical_protocol_contract_id',
                 'outer_execution_contract_id'):
        if shared[name] != contract_id:
            raise RuntimeError('Shared C35 execution identity differs: ' + name)
    if spec['execution_contract_id'] != contract_id:
        raise RuntimeError('Spec C35 execution identity differs')
    origins_path = args.origins
    origins = json.loads(origins_path.read_text())
    if not origins['forbidden_import_guard_active']:
        raise RuntimeError('Cold reader guard was absent')
    additions = {Path(__file__), parent_path, shared_path, spec_path, origins_path}
    for directory in (HERE / 'root35', HERE / 'sol35', HERE / 'read35'):
        additions.update(directory.glob('*.py'))
    additions.update(HERE.glob('*.py'))
    additions.update((HERE / 'astra_plan').glob('*.json'))
    additions.update((HERE / 'astra_plan').glob('*.md'))
    for row in origins['loaded_sources'].values():
        path = Path(row['path'])
        expected = {key: row[key] for key in ('bytes', 'sha256')}
        if identity(path) != expected:
            raise RuntimeError('Cold imported source changed: ' + str(path))
        additions.add(path)
    for path in shared['baseline_c33_integral_paths35'].values():
        additions.add(Path(path))
    for folder in HERE.glob('pure_*'):
        if folder.is_dir():
            additions.update(p for p in folder.iterdir() if p.is_file())
    for path in sorted(additions):
        resolved = str(path.resolve(strict=True))
        actual = identity(path)
        if resolved in inputs and inputs[resolved] != actual:
            raise RuntimeError('Attempted inherited identity replacement: ' + resolved)
        inputs[resolved] = actual
    environment = dict(parent['runtime_environment'])
    old_paths = environment['PYTHONPATH'].split(':')
    site = '/home/lyh/.local/lib/python3.10/site-packages'
    if old_paths[0] != site:
        raise RuntimeError('Inherited bootstrap does not keep site-packages first')
    additions_path = list(map(str, (HERE / 'root35', HERE / 'sol35', HERE / 'read35')))
    environment['PYTHONPATH'] = ':'.join(dict.fromkeys(
        [site, *additions_path, *old_paths[1:]]))
    head = subprocess.check_output(
        ['rtk', 'proxy', 'git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, text=True).strip()
    candidate = dict(
        schema='d1-c35-source-candidate-v1', decision='NEEDS_ASTRA_REVIEW',
        execution_contract_id=contract_id, repository=str(REPOSITORY),
        repository_head=head, worker=str(HERE / 'root35/worker35.py'),
        host=str(HERE / 'root35/host35.py'), inputs=inputs,
        runtime_environment=environment, shared_session=shared,
        arms={'development': spec}, x11_events_by_arm={'development': None},
        training_authorized=False, automatic_retry=False,
        parent_source_go_identity=dict(path=str(parent_path), **identity(parent_path)),
        cold_reader_origins_identity=dict(path=str(origins_path), **identity(origins_path)),
        inventory=dict(inherited_entries=len(parent['inputs']),
                       added_or_existing_paths=len(additions), total_entries=len(inputs)),
        builder_elapsed_s=time.monotonic() - started,
    )
    with destination.open('x') as stream:
        json.dump(candidate, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(destination), **identity(destination),
                         inventory=candidate['inventory'],
                         elapsed_s=candidate['builder_elapsed_s'])))


if __name__ == '__main__':
    main()
