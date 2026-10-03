"""Bind the existing runtime and new C33 sources; never run models or physics."""
import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
WORK = HERE.parent
REPOSITORY = Path('/home/lyh/wheel-legged-control-lab')


def identity(path):
    data = Path(path).read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    destination = HERE / 'source_candidate33.json'
    if destination.exists():
        raise FileExistsError(destination)
    parent_path = WORK / 'continuation30/source_go_30.json'
    parent = json.loads(parent_path.read_text())
    inputs = dict(parent['inputs'])
    mismatches = [path for path, record in inputs.items()
                  if identity(path) != record]
    if mismatches:
        raise RuntimeError('Inherited source changed: '+repr(mismatches))
    shared_path = HERE / 'astra_plan/shared_session33.json'
    shared = json.loads(shared_path.read_text())
    spec_path = HERE / 'astra_plan/spec33.json'
    spec = json.loads(spec_path.read_text())
    additions = {Path(__file__), parent_path, shared_path, spec_path}
    for directory in (WORK / 'continuation31/root31',
                      WORK / 'continuation31/sol31',
                      WORK / 'continuation31/read31', HERE / 'root33',
                      HERE / 'sol33', HERE / 'read33'):
        additions.update(directory.glob('*.py'))
    additions.update(HERE.glob('*.py'))
    additions.update((HERE / 'astra_plan').glob('*.json'))
    additions.update((HERE / 'astra_plan').glob('*.md'))
    for group in shared['baseline_paths33'].values():
        for path in group.values():
            report_path = Path(path)
            additions.add(report_path)
            run = Path(json.loads(report_path.read_text())['run'])
            # Existing reports certify the old native payloads. Pairing needs
            # the old states, controls, reset/geometry and source receipts only.
            for old in run.rglob('*'):
                if (old.is_file() and not old.is_symlink()
                        and not ('native' in old.name and old.name.endswith('.jsonl.gz'))):
                    additions.add(old)
    for path in sorted(additions):
        inputs[str(path.resolve(strict=True))] = identity(path)
    environment = dict(parent['runtime_environment'])
    prepend = [HERE / 'root33', HERE / 'sol33', HERE / 'read33',
               WORK / 'continuation31/root31', WORK / 'continuation31/sol31',
               WORK / 'continuation31/read31']
    environment['PYTHONPATH'] = ':'.join(map(str, prepend))+':'+environment['PYTHONPATH']
    head = subprocess.check_output(['rtk', 'proxy', 'git', 'rev-parse', 'HEAD'],
                                   cwd=REPOSITORY, text=True).strip()
    candidate = {
        'schema': 'd1-c33-source-candidate-v1', 'decision': 'NEEDS_ASTRA_REVIEW',
        'execution_contract_id': 'C33_fixed_S_curve_feasibility_v1',
        'repository': str(REPOSITORY), 'repository_head': head,
        'worker': str(HERE / 'root33/worker33.py'),
        'host': str(HERE / 'root33/host33.py'), 'inputs': inputs,
        'runtime_environment': environment, 'shared_session': shared,
        'arms': {'development': spec}, 'x11_events_by_arm': {'development': None},
        'training_authorized': False,
        'parent_source_go_identity': {'path': str(parent_path), **identity(parent_path)},
        'inventory': {'inherited_entries': len(parent['inputs']),
                      'added_or_rebound_paths': len(additions),
                      'total_entries': len(inputs)},
    }
    with destination.open('x') as stream:
        json.dump(candidate, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'path': str(destination), **identity(destination),
                      'inventory': candidate['inventory']}))


if __name__ == '__main__':
    main()
