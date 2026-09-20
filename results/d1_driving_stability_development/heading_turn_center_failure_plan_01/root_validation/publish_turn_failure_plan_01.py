"""Archive immutable offline diagnosis and independent reproduction receipts."""
import hashlib
import json
from pathlib import Path
import shutil

R = Path('/home/lyh/wheel-legged-control-lab')
W = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920')
S = W / 'turn_center_failure_plan_01'
D = R / 'results/d1_driving_stability_development/heading_turn_center_failure_plan_01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = json.loads((S / 'checksums.json').read_text())
    for name, entry in checks['files'].items():
        assert sha(S / name) == entry['sha256'], name
    pairs = [('report.json', 'turn_center_failure_root_recheck.json'),
             ('body_rate_qualification.json', 'body_rate_qualification_root_recheck.json')]
    for original, recheck in pairs:
        assert (S / original).read_bytes() == (W / recheck).read_bytes()
    shutil.copytree(S, D, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    validation = D / 'root_validation'
    validation.mkdir()
    for _, recheck in pairs:
        shutil.copyfile(W / recheck, validation / recheck)
    shutil.copyfile(Path(__file__), validation / Path(__file__).name)
    with (validation / 'receipt.json').open('x') as stream:
        json.dump({'new_physics_steps': 0, 'controller_compute_calls': 0,
                   'new_contact_solves': 0, 'both_reports_byte_identical': True,
                   'checksums_sha256': sha(S / 'checksums.json'),
                   'reports': {name: sha(S / name) for name, _ in pairs}}, stream, indent=2)
        stream.write('\n')
    entries = {str(p.relative_to(D)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
               for p in sorted(D.rglob('*')) if p.is_file()}
    with (D / 'manifest.json').open('x') as stream:
        json.dump(entries, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'files': len(entries), 'bytes': sum(e['bytes'] for e in entries.values()),
                      'destination': str(D)}))


if __name__ == '__main__':
    main()
