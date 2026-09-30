"""Root-only bundle/candidate assembly; never executes a model or simulator."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/home/lyh/wheel-legged-control-lab')
BUNDLE = ROOT/'runtime/d1_b22'
WORK = Path(__file__).resolve().parents[1]


def identity(path):
    return {'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', type=Path, required=True)
    args = parser.parse_args()
    contract = args.contract.resolve(strict=True)
    required = ['gui25/run_gui25.py', 'root25/reader25.py',
                'root25/reader25_deferred.py']
    for filename in required:
        if not (WORK/filename).is_file():
            raise FileNotFoundError(filename)
    copies = {}
    for subdir in ('gui25', 'root25'):
        for source in sorted((WORK/subdir).glob('*.py')):
            dest = BUNDLE/'research/continuation25'/source.relative_to(WORK)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.read_bytes() != source.read_bytes():
                raise RuntimeError('never overwrite differing bundled source: '+str(dest))
            if not dest.exists():
                shutil.copyfile(source, dest)
            copies[str(source)] = {'bundle': str(dest), **identity(dest)}
    config = json.loads((BUNDLE/'runtime_manifest_v03.json').read_text())
    config['worker'] = str(BUNDLE/'research/continuation25/gui25/run_gui25.py')
    config['scope_revision'] = 'C25 separate corrective keyboard qualification; C23 failures preserved; C24 unchanged'
    config['archive_revision'] = 'bounded owner-thread deferred archive, unchanged atomic gzip9 publication after active end'
    config['qualification_scope'] = 'C23 same-trajectory numerical proof, C25 corrective keyboard, C24 true15mm; no RL superiority claim'
    extra_imports = [str(BUNDLE/'research/continuation25'/name) for name in ('gui25', 'root25')]
    config['runtime_environment']['PYTHONPATH'] += ':'+':'.join(extra_imports)
    for source, meta in copies.items():
        config['source_hashes'][meta['bundle']] = identity(Path(meta['bundle']))
    launcher = ROOT/'scripts/run_b22.py'
    config['source_hashes'][str(launcher)] = identity(launcher)
    manifest = BUNDLE/'runtime_manifest_v25.json'
    save(manifest, config)
    old_go = json.loads((WORK.parent/'continuation23/gui_source_go_04.json').read_text())
    go = copy.deepcopy(old_go)
    go['decision'] = 'PENDING_ASTRA_REVIEW'
    go['worker'] = config['worker']
    go['runtime_environment'] = config['runtime_environment']
    go['review'] = {'contract': str(contract), 'contract_identity': identity(contract),
                    'status': 'await actual Astra source review; not executable'}
    go['inputs'].update(config['source_hashes'])
    for path in [manifest, contract, launcher, *map(Path, copies)]:
        go['inputs'][str(path)] = identity(path)
    spec = copy.deepcopy(old_go['arms']['keyboard_gui'])
    spec['output_directory'] = str(WORK/'keyboard_gui_01')
    spec['execution_contract_id'] = 'C25_B22_GUI_reset_archive_v1'
    go['arms'] = {'keyboard_gui': spec}
    events = copy.deepcopy(old_go['x11_events_by_arm']['keyboard_gui'])
    for event in events:
        if event.get('key') == 'Escape':
            event['tick'] = 1100
    events.sort(key=lambda event: event['tick'])
    assert len(events) == 27 and spec['control_limit'] == 1200
    go['x11_events_by_arm'] = {'keyboard_gui': events}
    save(WORK/'gui_go_candidate_25.json', go)
    save(WORK/'bundle_copy_25.json', {'copies': copies, 'manifest': identity(manifest),
                                     'model_calls': 0, 'physics_calls': 0})
    print(json.dumps({'manifest': identity(manifest), 'candidate': identity(WORK/'gui_go_candidate_25.json'),
                      'files': len(config['source_hashes']), 'events': len(events)}))


if __name__ == '__main__':
    main()
