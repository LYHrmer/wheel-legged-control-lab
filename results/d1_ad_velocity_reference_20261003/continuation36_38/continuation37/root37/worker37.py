"""C37 headless worker: the C36 pre-unload controller plus the repaired fullM check.

Behavioural diff from the frozen C35 source, in three files and three methods:

    sol36/core36.py        Core36._all_loaded            pre-unload support regime
    sol36/adapter36.py     PairController36._reference_record35
                                                         declares that regime to the guard
    root37/runtime37.py    SideAccess37._checked_query   short-circuits the legacy fullM
                                                         branch that dereferences MjData.qM

Nothing else changes. The frozen native guard, the paired IK, the coupled inertia
arithmetic, the force allocation, the wheel brake, the torque clipping, the archive writer
and the per-case 200/side/400 accounting are the C35 code, reused byte for byte.

Four globals of the sealed worker35 are rebound in this process before its executor runs,
declared here and in source_go37.json:

    worker35.CONTRACT     -> the C37 contract identity, so the receipt is honest
    worker35.bootstrap35  -> bootstrap37, which swaps the controller and the side-access
                             class in the symbol table execute35 already reads them from
    worker35.origins35    -> origins37, which also records core36, adapter36 and runtime37
    worker35.preflight35  -> preflight37, which checks the C37 contract and budget

No byte of any C35 or C36 file is modified, and every path this depends on is hashed in
source_go37.json and re-verified by preflight37 before anything is constructed.
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
from runtime37 import SideAccess37
from worker30 import identity, write
from worker35 import execute35, origins35

CONTRACT37 = 'C37_preunload_transfer_with_repaired_fullm_check_v1'
EXTRA_MODULES = ('core36', 'adapter36', 'runtime37')
RUNTIME_CONTROL_FENCE = 9000      # execute35 hardcodes this owned-runtime fence
RUNTIME_NATIVE_FENCE = 45000      # execute35 hardcodes this cumulative native fence


def origins37(sources):
    """C35 origins plus the C36 and C37 modules, all required inside the freeze."""
    result = origins35(sources)
    for name in EXTRA_MODULES:
        module = sys.modules.get(name)
        if module is None or not getattr(module, '__file__', None):
            raise RuntimeError('C37 missing actual module: '+name)
        path = str(Path(module.__file__).resolve(strict=True))
        if sources.get(path) != identity(path):
            raise RuntimeError('C37 actual module escaped source freeze: '+name)
        result[name] = dict(path=path, **identity(path))
    return result


def bootstrap37(session, output):
    """C35 bootstrap with the controller and the side-access class replaced."""
    symbols = worker35.bootstrap35_frozen(session, output)
    if (symbols['PairController35'].__name__ != 'PairController35'
            or symbols['SideAccess35'].__name__ != 'SideAccess35'):
        raise RuntimeError('C37 bootstrap did not receive the frozen classes')
    symbols['PairController35'] = PairController36
    symbols['SideAccess35'] = SideAccess37
    symbols['PairController36'] = PairController36
    symbols['SideAccess37'] = SideAccess37
    symbols['origins35'] = origins37(session['source_hashes'])
    write(output/'runtime37_module_origins_before.json', symbols['origins35'])
    return symbols


def preflight37(session, session_path, output):
    """C37 contract, budget, source and environment checks; mirrors preflight35."""
    spec = json.loads(Path(session['spec37_path']).read_text())
    for key, value in spec.items():
        if session.get(key) != value:
            raise RuntimeError('C37 session differs from spec: '+key)
    if (session['execution_contract_id'] != CONTRACT37 or session['argv'] != sys.argv
            or session['control_limit'] != RUNTIME_CONTROL_FENCE
            or session['normal_native_cap'] != RUNTIME_NATIVE_FENCE
            or session['cycles_limit'] != len(session['cases'])
            or session['planned_controls_max'] != 1800*len(session['cases'])
            or session['retry_permitted'] is not False
            or session['training_authorized'] is not False
            or Path(session['output_directory']).resolve() != output):
        raise RuntimeError('C37 session/argv/reservation differs')
    if session['runtime_environment']['PYTHONPATH'].split(os.pathsep)[0] != \
            '/home/lyh/.local/lib/python3.10/site-packages':
        raise RuntimeError('C37 requires installed site-packages first')
    for name, expected in session['source_hashes'].items():
        if identity(name) != expected:
            raise RuntimeError('C37 source mismatch: '+name)
    for key, value in session['runtime_environment'].items():
        if os.environ.get(key) != value:
            raise RuntimeError('C37 runtime environment differs: '+key)
    for key in ('spec37_path', 'contract37_path', 'interface37_path',
                'spec36_path', 'contract36_path', 'interface36_path',
                'spec35_path', 'contract35_path', 'interface35_path'):
        path = str(Path(session[key]).resolve(strict=True))
        if session['source_hashes'].get(path) != identity(path):
            raise RuntimeError('C37 unfrozen specification: '+key)
    write(output/'preflight_ready.json', dict(pid=os.getpid(),
          session_sha256=identity(session_path)['sha256'], monotonic_s=time.monotonic()))


def install37():
    """Rebind the four sealed worker35 globals execute35 reads. Declared and idempotent."""
    if not hasattr(worker35, 'bootstrap35_frozen'):
        worker35.bootstrap35_frozen = worker35.bootstrap35
    worker35.CONTRACT = CONTRACT37
    worker35.bootstrap35 = bootstrap37
    worker35.origins35 = origins37
    worker35.preflight35 = preflight37


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    install37()
    path = args.session.resolve(strict=True)
    return execute35(json.loads(path.read_text()), path, args.output.resolve(strict=True))


if __name__ == '__main__':
    raise SystemExit(main())
