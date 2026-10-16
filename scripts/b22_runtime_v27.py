"""Register/check the B22 v27 environment without importing models, GUI or physics.

Only two previously authorized source revisions and one exact Ubuntu libX11
update may differ from the pinned v26 manifest. Python bytecode is regenerable;
all other v26 inputs remain fixed. Registration is environment bookkeeping,
not a new control qualification or a GUI validation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = 'runtime/d1_b22/runtime_manifest_v26.json'
MANIFEST = 'runtime/d1_b22/runtime_manifest_v27.json'
BASE_IDENTITY = {'bytes': 854599,
                 'sha256': '6466fb0310295c7bfb564c3795f3652652d9cc6e282ec8da3876f1d9cddb6f21'}
SCHEMA = 'd1-b22-versioned-runtime-v27'
NEW_SOURCES = ('scripts/b22_runtime_v27.py', 'scripts/play_b22_v27.py')
LIBRARY = '/usr/lib/x86_64-linux-gnu/libX11.so.6.4.0'
PACKAGE = 'libx11-6:amd64'
PACKAGE_VERSION = '2:1.7.5-1ubuntu0.4'
PACKAGE_MD5 = '2efebaf129cfee942937eceb3be618c6'
UPGRADE_LINE = ('2026-10-14 06:17:21 upgrade libx11-6:amd64 '
                '2:1.7.5-1ubuntu0.3 2:1.7.5-1ubuntu0.4')
MODEL_SHA = '7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691'


def identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'input is not a regular non-symlink file: {path}')
    before = path.stat()
    digest, count = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
            count += len(block)
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if count != before.st_size or any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError(f'input changed while hashing: {path}')
    return {'bytes': count, 'sha256': digest.hexdigest()}


def rules(root=ROOT):
    return {
        str(root/'src/wheel_legged_control/experiments.py'): {
            'basis': 'owner-authorized source revision 1d8177c',
            'identity': {'bytes': 16240, 'sha256':
                         '7b5f8d73fa61aff9cd59c3ec384d11f5dab8c2399b29aafb8c241aff774b1cc4'}},
        str(root/'scripts/delay_metrics.py'): {
            'basis': 'source revision 79ba9df; missing-control-tick validation',
            'identity': {'bytes': 4157, 'sha256':
                         'aff8a818697f681407433e333f9fbad13228260e6a87e3dddd48709c6d9321db'}},
        LIBRARY: {
            'basis': 'explicit v27 Ubuntu package environment migration; GUI not revalidated',
            'identity': {'bytes': 1306280, 'sha256':
                         'e23ed094eaf0a74a822a6d9d14eee628ab9f7ec08aecbf20ffd803f019ff6e98'}}}


def is_cache(name):
    path = Path(name)
    return '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo')


def package_evidence():
    version = subprocess.run(['dpkg-query', '-W', '-f=${Version}', PACKAGE],
                             capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    listing = Path('/var/lib/dpkg/info/libx11-6:amd64.md5sums')
    rows = {name: digest for digest, name in
            (line.split() for line in listing.read_text().splitlines())}
    listed = rows.get(LIBRARY.lstrip('/'))
    actual = hashlib.md5(Path(LIBRARY).read_bytes()).hexdigest()
    if version != PACKAGE_VERSION or listed != PACKAGE_MD5 or actual != PACKAGE_MD5:
        raise ValueError('libX11 package version or installed package checksum differs from v27 migration')
    return {'package': PACKAGE, 'version': version, 'installed_md5': actual,
            'package_manifest': str(listing), 'package_manifest_identity': identity(listing),
            'upgrade_log': '/var/log/dpkg.log', 'upgrade_record': UPGRADE_LINE}


def read_base(root=ROOT):
    if identity(root/BASE) != BASE_IDENTITY:
        raise ValueError('trusted v26 manifest identity changed')
    return json.loads((root/BASE).read_text())


def expected_sources(base, root=ROOT):
    sources = {name: value for name, value in base['source_hashes'].items() if not is_cache(name)}
    for name, rule in rules(root).items():
        if name not in sources:
            raise ValueError(f'explicit migration input missing from v26: {name}')
        sources[name] = rule['identity']
    sources[str(root/BASE)] = BASE_IDENTITY
    return sources


def scan(sources):
    changed, verified_bytes = [], 0
    for name, expected in sources.items():
        try:
            actual, error = identity(name), None
            verified_bytes += actual['bytes']
        except (OSError, ValueError) as exc:
            actual, error = None, str(exc)
        if actual != expected:
            changed.append({'path': name, 'expected': expected, 'actual': actual, 'error': error})
    return changed, verified_bytes


def migration(base, root=ROOT):
    return {'upstream_manifest': BASE, 'upstream_identity': BASE_IDENTITY,
            'accepted_revisions': rules(root),
            'excluded_regenerable_python_caches': sorted(
                name for name in base['source_hashes'] if is_cache(name)),
            'qualification_changed': False, 'gui_revalidated': False,
            'new_source_paths': list(NEW_SOURCES)}


def validate_manifest(manifest, base, root=ROOT):
    if manifest.get('schema') != SCHEMA or manifest.get('migration') != migration(base, root):
        raise ValueError('v27 schema or explicit migration rules differ')
    for key, value in base.items():
        if key not in ('schema', 'source_hashes') and manifest.get(key) != value:
            raise ValueError(f'v27 changed frozen runtime configuration: {key}')
    required = expected_sources(base, root)
    sources = manifest['source_hashes']
    additions = {str(root/name) for name in NEW_SOURCES}
    if set(sources) != set(required) | additions:
        raise ValueError('v27 source closure differs from fixed v26 inputs plus new entries')
    if any(sources[name] != value for name, value in required.items()):
        raise ValueError('v27 redefined a source/model/library identity outside the migration rules')
    evidence = manifest['environment_evidence']
    if (evidence.get('package') != PACKAGE or evidence.get('version') != PACKAGE_VERSION
            or evidence.get('installed_md5') != PACKAGE_MD5 or evidence.get('upgrade_record') != UPGRADE_LINE):
        raise ValueError('v27 package migration evidence differs')


def inspect(root=ROOT, manifest=None):
    base = read_base(root)
    if manifest is None:
        manifest = json.loads((root/MANIFEST).read_text())
    validate_manifest(manifest, base, root)
    changed, verified_bytes = scan(manifest['source_hashes'])
    evidence, package_error = None, None
    try:
        evidence = package_evidence()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        package_error = str(exc)
    differences = []
    for name, rule in rules(root).items():
        differences.append({'path': name, 'v26': base['source_hashes'][name],
                            'v27': rule['identity'], 'basis': rule['basis']})
    report = {'schema': 'd1-b22-v27-inspection-v1', 'status': 'BLOCKED' if changed or package_error else 'READY',
              'scope': 'input/environment readiness only; no GUI or control qualification',
              'upstream_identity': BASE_IDENTITY, 'source_files': len(manifest['source_hashes']),
              'expected_bytes': sum(row['bytes'] for row in manifest['source_hashes'].values()),
              'verified_bytes': verified_bytes, 'accepted_differences': differences,
              'excluded_caches': manifest['migration']['excluded_regenerable_python_caches'],
              'unexpected_differences': changed, 'package_evidence': evidence,
              'package_error': package_error, 'model_loads': 0, 'gui_starts': 0, 'physics_steps': 0,
              'qualification_changed': False, 'gui_revalidated': False}
    return manifest, report


def build(root=ROOT):
    base = read_base(root)
    evidence = package_evidence()
    # The historical upgrade record is verified when registering, not required
    # to remain in the current (rotating) system log for every future launch.
    if UPGRADE_LINE not in Path(evidence['upgrade_log']).read_text().splitlines():
        raise ValueError('the documented libX11 upgrade record is absent during registration')
    candidate = copy.deepcopy(base)
    candidate.update(schema=SCHEMA, source_hashes=expected_sources(base, root),
                     migration=migration(base, root), environment_evidence=evidence)
    for name in NEW_SOURCES:
        candidate['source_hashes'][str(root/name)] = identity(root/name)
    _, report = inspect(root, candidate)
    if report['status'] != 'READY':
        raise ValueError('unknown input differences block v27 registration: '+json.dumps(report))
    with (root/MANIFEST).open('x') as stream:
        json.dump(candidate, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true', help='read-only, zero-model/GUI/physics inspection')
    action.add_argument('--build-manifest', action='store_true', help='register a new, exclusive v27 manifest')
    args = parser.parse_args(argv)
    try:
        report = build() if args.build_manifest else inspect()[1]
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(json.dumps({'status': 'BLOCKED', 'error': str(exc), 'model_loads': 0,
                          'gui_starts': 0, 'physics_steps': 0}), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['status'] == 'READY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
