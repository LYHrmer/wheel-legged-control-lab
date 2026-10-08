"""Build the C37 independent-reader source GO.

Separate from the physics GO, following the C34 pattern of a distinct reader review. Inherits
the complete C37 physics input set, adds the C37 physics GO itself and the reader wrapper, and
points `reader` at read37.py instead of the frozen read35.py.

Reads files and writes one JSON document. No MuJoCo, no model, no physics, no training.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTRACT37 = 'C37_preunload_transfer_with_repaired_fullm_check_v1'
NEW_PATHS = (ROOT/'read37/read37.py', ROOT/'build_reader_go37.py')


def identity(path):
    data = Path(path).read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    target = ROOT/(sys.argv[1] if len(sys.argv) > 1 else 'reader_source_go37.json')
    physics = json.loads((ROOT/'source_go37.json').read_text())
    inputs = dict(physics['inputs'])
    inherited = len(inputs)
    changed = [path for path, expected in inputs.items()
               if not Path(path).is_file() or identity(path) != expected]
    if changed:
        raise RuntimeError('inherited C37 inputs changed: '+repr(changed[:5]))
    inputs[str((ROOT/'source_go37.json').resolve(strict=True))] = identity(ROOT/'source_go37.json')
    for path in NEW_PATHS:
        inputs[str(path.resolve(strict=True))] = identity(path)

    record = {
        'schema': 'd1-c37-independent-reader-source-go-v1',
        'execution_contract_id': CONTRACT37,
        'decision': 'GO',
        'signed_by': 'root (main Codex agent)',
        'astra_reviewed': False,
        'signing_basis': physics['signing_basis'],
        'signed_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': ('One saved-file readback of the C37 campaign. The frozen read35.read_run35 '
                  'performs every independent recomputation unchanged; read37.py replaces only '
                  'the C35-specific identity assertions inside sources35.'),
        'declared_read35_rebinds': ['CONTRACT', 'sources35'],
        'rebind_reason': ('the frozen assertions hardcode the C35 contract string, read the '
                          'C35 spec path and require exactly five cases; everything else in '
                          'that function is copied verbatim'),
        'dynamics_limitation': ('FK/COM, target FK/error/bounds, reference, allocation, PD, '
                                'feedforward, braking, clipping and the native chain are '
                                'independently recomputed. Jacobian, bias and fullM caches are '
                                'constrained by sealed source, API scope and counts; no second '
                                'rigid-body dynamics implementation is claimed.'),
        'physics_GO': False,
        'training_authorized': False,
        'reader_physics_calls_expected': 0,
        'reader_model_calls_expected': 0,
        'reader': str(ROOT/'read37/read37.py'),
        'reader_launcher': physics['reader_launcher'],
        'physics_source_go_identity': {'path': str(ROOT/'source_go37.json'),
                                       **identity(ROOT/'source_go37.json')},
        'prior_reader_attempt': {
            'review_used': str(ROOT/'source_go37.json'),
            'exit_code': 2,
            'error': 'C35 frozen source GO/session/spec differs',
            'produced_audit': False,
            'reader_physics_calls': 0, 'reader_model_calls': 0,
            'note': ('the frozen C35 reader correctly refused a C37 contract identity; the '
                     'attempt produced no audit result and is recorded, not reallocated')},
        'inventory': {'inherited_entries': inherited, 'added_paths': len(NEW_PATHS)+1,
                      'signed_total_entries': len(inputs)},
        'inputs': inputs,
    }
    write_new(target, record)
    print(json.dumps({'reader_go': str(target),
                      'inherited_inputs': inherited, 'added': len(NEW_PATHS)+1,
                      'total_inputs': len(inputs)}))


if __name__ == '__main__':
    main()
