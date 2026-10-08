"""Build the C38 independent-reader source GO. Same pattern as C37's build_reader_go37.py.

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
CONTRACT38 = 'C38_preunload_record_semantics_and_audit_v1'
NEW_PATHS = (ROOT/'read38/read38.py', ROOT/'read38/native_read38.py',
             ROOT/'read38/test_native_read38_pure.py',
             ROOT/'read38/ledger_read38.py', ROOT/'read38/test_ledger_read38_pure.py',
             ROOT/'build_reader_go38.py')


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
    target = ROOT/(sys.argv[1] if len(sys.argv) > 1 else 'reader_source_go38.json')
    physics = json.loads((ROOT/'source_go38.json').read_text())
    inputs = dict(physics['inputs'])
    inherited = len(inputs)
    changed = [path for path, expected in inputs.items()
               if not Path(path).is_file() or identity(path) != expected]
    if changed:
        raise RuntimeError('inherited C38 inputs changed: '+repr(changed[:5]))
    inputs[str((ROOT/'source_go38.json').resolve(strict=True))] = identity(ROOT/'source_go38.json')
    for path in NEW_PATHS:
        inputs[str(path.resolve(strict=True))] = identity(path)

    record = {
        'schema': 'd1-c38-independent-reader-source-go-v1',
        'execution_contract_id': CONTRACT38,
        'decision': 'GO', 'signed_by': 'root (main Codex agent)',
        'astra_reviewed': False, 'signing_basis': physics['signing_basis'],
        'signed_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': ('One saved-file readback of the C38 campaign. The frozen read35.read_run35 '
                  'performs every independent recomputation unchanged; read38.py replaces '
                  'only the C35-specific identity assertions inside sources35.'),
        'declared_read35_rebinds': ['CONTRACT', 'sources35', 'check_ledger35',
                                    'episode_read35.verify_native35'],
        'preunload_extension_rationale': ('frozen native_read35.verify_native35 and '
            'contact_math35.native_phase_support35 hardcode the pre-C36 assumption that '
            'transfer/transfer_restore is always the all-four regime; native_read38.py '
            'extends the mask/gate expectation to the pure-tested pre-unload window using '
            'fields the frozen reader already trusts (reference.phase, now also '
            'reference.phase_elapsed_s) and the frozen pair_support35 gate, with 8 pure '
            'tests verifying the extension agrees with the frozen selector everywhere the '
            'frozen selector was defined'),
        'ledger_extension_rationale': ('frozen ledger_read35.check_ledger35 hardcodes the '
            'C35 5-case budget shape as exact-equality literals (predict ceiling 3001, '
            'predict+probe-row ceiling 3032); ledger_read38.py generalises both from '
            'session["prepare_controls"]+session["retention_controls"] and '
            'len(session["cases"]), with a pure test confirming the generalisation reduces '
            'to the exact frozen literals at 5 cases'),
        'dynamics_limitation': ('FK/COM, target FK/error/bounds, reference, allocation, PD, '
                                'feedforward, braking, clipping and the native chain are '
                                'independently recomputed. Jacobian, bias and fullM caches are '
                                'constrained by sealed source, API scope and counts; no second '
                                'rigid-body dynamics implementation is claimed.'),
        'physics_GO': False, 'training_authorized': False,
        'reader_physics_calls_expected': 0, 'reader_model_calls_expected': 0,
        'reader': str(ROOT/'read38/read38.py'), 'reader_launcher': physics['reader_launcher'],
        'physics_source_go_identity': {'path': str(ROOT/'source_go38.json'),
                                       **identity(ROOT/'source_go38.json')},
        'determinism_check': ('C38 reproduced the saved C37 physical trajectory exactly: '
                              '1495 of 1495 native rows had identical qpos and qvel. Only '
                              'the record-semantics change is new; see interfaces38.md.'),
        'inventory': {'inherited_entries': inherited, 'added_paths': len(NEW_PATHS)+1,
                      'signed_total_entries': len(inputs)},
        'inputs': inputs,
    }
    write_new(target, record)
    print(json.dumps({'reader_go': str(target), 'inherited_inputs': inherited,
                      'total_inputs': len(inputs)}))


if __name__ == '__main__':
    main()
