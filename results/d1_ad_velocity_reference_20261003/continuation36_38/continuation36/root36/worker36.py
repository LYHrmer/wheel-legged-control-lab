"""C36 headless worker: reuse the frozen C35 orchestration with two declared overrides.

The entire behavioural diff from C35 is core36.Core36 and adapter36.PairController36; see
their module docstrings. This file injects them and supplies a C36 preflight so the
contract identity and budget recorded in the receipt are C36's own, not C35's.

The injection is explicit and auditable: four module attributes of the sealed worker35 are
rebound in this process before its executor runs. No byte of any C35 file is modified, and
every C35 path this depends on is hashed in source_go36.json and re-verified by preflight36
before anything is constructed.

    worker35.CONTRACT     -> the C36 contract identity, so the receipt is honest
    worker35.bootstrap35  -> bootstrap36, which swaps the controller class in the symbol
                             table execute35 already reads it from
    worker35.origins35    -> origins36, which records the C36 modules as well
    worker35.preflight35  -> preflight36, which checks the C36 contract and budget

Everything else - the owned runtime, the atomic native guard, the model ledger, the paired
IK, the coupled inertia, the force allocation, the wheel brake, the torque clipping, the
archive writer and the per-case 200/side/400 accounting - is the frozen C35 code, reused
unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import worker35
from adapter36 import PairController36
from worker30 import identity, write
from worker35 import execute35, origins35

CONTRACT36 = 'C36_preunload_transfer_two_support_feasibility_v1'
C36_MODULES = ('core36', 'adapter36')
RUNTIME_CONTROL_FENCE = 9000      # execute35 hardcodes this owned-runtime fence
RUNTIME_NATIVE_FENCE = 45000      # execute35 hardcodes this cumulative native fence


def origins36(sources):
    """C35 origins plus the two C36 modules, all required to be inside the freeze."""
    result = origins35(sources)
    for name in C36_MODULES:
        module = sys.modules.get(name)
        if module is None or not getattr(module, '__file__', None):
            raise RuntimeError('C36 missing actual module: '+name)
        path = str(Path(module.__file__).resolve(strict=True))
        if sources.get(path) != identity(path):
            raise RuntimeError('C36 actual module escaped source freeze: '+name)
        result[name] = dict(path=path, **identity(path))
    return result


def bootstrap36(session, output):
    """C35 bootstrap with the controller class replaced in the shared symbol table."""
    symbols = worker35.bootstrap35_frozen(session, output)
    if symbols['PairController35'].__name__ != 'PairController35':
        raise RuntimeError('C36 bootstrap did not receive the frozen controller')
    symbols['PairController35'] = PairController36
    symbols['PairController36'] = PairController36
    symbols['origins35'] = origins36(session['source_hashes'])
    write(output/'runtime36_module_origins_before.json', symbols['origins35'])
    return symbols


def preflight36(session, session_path, output):
    """C36 contract, budget, source and environment checks; mirrors preflight35."""
    spec = json.loads(Path(session['spec36_path']).read_text())
    for key, value in spec.items():
        if session.get(key) != value:
            raise RuntimeError('C36 session differs from spec: '+key)
    if (session['execution_contract_id'] != CONTRACT36 or session['argv'] != sys.argv
            or session['control_limit'] != RUNTIME_CONTROL_FENCE
            or session['normal_native_cap'] != RUNTIME_NATIVE_FENCE
            or session['cycles_limit'] != len(session['cases'])
            or session['planned_controls_max'] != 1800*len(session['cases'])
            or session['retry_permitted'] is not False
            or session['training_authorized'] is not False
            or Path(session['output_directory']).resolve() != output):
        raise RuntimeError('C36 session/argv/reservation differs')
    if session['runtime_environment']['PYTHONPATH'].split(os.pathsep)[0] != \
            '/home/lyh/.local/lib/python3.10/site-packages':
        raise RuntimeError('C36 requires installed site-packages first')
    for name, expected in session['source_hashes'].items():
        if identity(name) != expected:
            raise RuntimeError('C36 source mismatch: '+name)
    for key, value in session['runtime_environment'].items():
        if os.environ.get(key) != value:
            raise RuntimeError('C36 runtime environment differs: '+key)
    for key in ('spec36_path', 'contract36_path', 'interface36_path',
                'spec35_path', 'contract35_path', 'interface35_path'):
        path = str(Path(session[key]).resolve(strict=True))
        if session['source_hashes'].get(path) != identity(path):
            raise RuntimeError('C36 unfrozen specification: '+key)
    write(output/'preflight_ready.json', dict(pid=os.getpid(),
          session_sha256=identity(session_path)['sha256'], monotonic_s=time.monotonic()))


def install36():
    """Rebind the four sealed worker35 globals execute35 reads. Declared and idempotent."""
    if not hasattr(worker35, 'bootstrap35_frozen'):
        worker35.bootstrap35_frozen = worker35.bootstrap35
    worker35.CONTRACT = CONTRACT36
    worker35.bootstrap35 = bootstrap36
    worker35.origins35 = origins36
    worker35.preflight35 = preflight36


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    install36()
    path = args.session.resolve(strict=True)
    return execute35(json.loads(path.read_text()), path, args.output.resolve(strict=True))


if __name__ == '__main__':
    raise SystemExit(main())
