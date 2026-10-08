"""C38 headless worker: the C37 stack with the record-semantics fix.

Behavioural diff from the frozen C35 source, in four files and four methods:

    continuation36/sol36/core36.py     Core36._all_loaded             pre-unload regime
    continuation38/sol38/adapter38.py  PairController38._reference_record35
                                                                      declares that regime
                                                                      WITHOUT overloading
                                                                      swing_legs
    continuation37/root37/runtime37.py SideAccess37._checked_query     short-circuits the
                                                                      legacy fullM branch

adapter36.PairController36 is inherited for its Core36 installation; its record override is
replaced by adapter38's, which starts from the frozen record instead.

The record change cannot move the trajectory: the controller's own self.swing_legs, which
gates the coupled-inertia computation and the swing PD, is set by the frozen compute from
reference.phase, not from the declared record. C38 is therefore expected to reproduce C37's
299 controls and the same touchdown_dwell_timeout, which doubles as a determinism check, and
to let the frozen independent reader finish the audit it stopped on.

Four globals of the sealed worker35 are rebound in this process before its executor runs,
declared here and in source_go38.json: CONTRACT, bootstrap35, origins35, preflight35. No byte
of any C35, C36 or C37 file is modified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import worker35
from adapter38 import PairController38
from runtime37 import SideAccess37
from worker30 import identity, write
from worker35 import execute35, origins35

CONTRACT38 = 'C38_preunload_record_semantics_and_audit_v1'
EXTRA_MODULES = ('core36', 'adapter36', 'adapter38', 'runtime37')
RUNTIME_CONTROL_FENCE = 9000
RUNTIME_NATIVE_FENCE = 45000


def origins38(sources):
    result = origins35(sources)
    for name in EXTRA_MODULES:
        module = sys.modules.get(name)
        if module is None or not getattr(module, '__file__', None):
            raise RuntimeError('C38 missing actual module: '+name)
        path = str(Path(module.__file__).resolve(strict=True))
        if sources.get(path) != identity(path):
            raise RuntimeError('C38 actual module escaped source freeze: '+name)
        result[name] = dict(path=path, **identity(path))
    return result


def bootstrap38(session, output):
    symbols = worker35.bootstrap35_frozen(session, output)
    if (symbols['PairController35'].__name__ != 'PairController35'
            or symbols['SideAccess35'].__name__ != 'SideAccess35'):
        raise RuntimeError('C38 bootstrap did not receive the frozen classes')
    symbols['PairController35'] = PairController38
    symbols['SideAccess35'] = SideAccess37
    symbols['PairController38'] = PairController38
    symbols['SideAccess37'] = SideAccess37
    symbols['origins35'] = origins38(session['source_hashes'])
    write(output/'runtime38_module_origins_before.json', symbols['origins35'])
    return symbols


def preflight38(session, session_path, output):
    spec = json.loads(Path(session['spec38_path']).read_text())
    for key, value in spec.items():
        if session.get(key) != value:
            raise RuntimeError('C38 session differs from spec: '+key)
    if (session['execution_contract_id'] != CONTRACT38 or session['argv'] != sys.argv
            or session['control_limit'] != RUNTIME_CONTROL_FENCE
            or session['normal_native_cap'] != RUNTIME_NATIVE_FENCE
            or session['cycles_limit'] != len(session['cases'])
            or session['planned_controls_max'] != 1800*len(session['cases'])
            or session['retry_permitted'] is not False
            or session['training_authorized'] is not False
            or Path(session['output_directory']).resolve() != output):
        raise RuntimeError('C38 session/argv/reservation differs')
    if session['runtime_environment']['PYTHONPATH'].split(os.pathsep)[0] != \
            '/home/lyh/.local/lib/python3.10/site-packages':
        raise RuntimeError('C38 requires installed site-packages first')
    for name, expected in session['source_hashes'].items():
        if identity(name) != expected:
            raise RuntimeError('C38 source mismatch: '+name)
    for key, value in session['runtime_environment'].items():
        if os.environ.get(key) != value:
            raise RuntimeError('C38 runtime environment differs: '+key)
    for key in ('spec38_path', 'contract38_path', 'interface38_path',
                'spec37_path', 'contract37_path', 'interface37_path',
                'spec36_path', 'contract36_path', 'interface36_path',
                'spec35_path', 'contract35_path', 'interface35_path'):
        path = str(Path(session[key]).resolve(strict=True))
        if session['source_hashes'].get(path) != identity(path):
            raise RuntimeError('C38 unfrozen specification: '+key)
    write(output/'preflight_ready.json', dict(pid=os.getpid(),
          session_sha256=identity(session_path)['sha256'], monotonic_s=time.monotonic()))


def install38():
    if not hasattr(worker35, 'bootstrap35_frozen'):
        worker35.bootstrap35_frozen = worker35.bootstrap35
    worker35.CONTRACT = CONTRACT38
    worker35.bootstrap35 = bootstrap38
    worker35.origins35 = origins38
    worker35.preflight35 = preflight38


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    install38()
    path = args.session.resolve(strict=True)
    return execute35(json.loads(path.read_text()), path, args.output.resolve(strict=True))


if __name__ == '__main__':
    raise SystemExit(main())
