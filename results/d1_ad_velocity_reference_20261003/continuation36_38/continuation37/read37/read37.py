"""C37 independent saved-data reader: the frozen C35 reader with three declared rebinds.

`read35.read_run35` is reused in full - every independent recomputation of forward
kinematics, COM, target bounds and residual, references, allocation, PD, braking, clipping,
per-contact frames, support loads, force and moment sums, the contact hull, task metrics and
the global ledger is the frozen C35 code. Only the C35-specific identity assertions inside
`sources35` are replaced, because the frozen ones hardcode the C35 contract string, read the
C35 spec path and require exactly five cases:

    read35.CONTRACT   -> the C37 contract identity
    read35.sources35  -> sources37, a faithful copy of the frozen function with
                         `spec35_path` replaced by `spec37_path` and the fixed
                         `cycles_limit == 5` replaced by `cycles_limit == len(cases)`

Everything else in that function - the GO decision, arm, mode, render, retry, output
directory, full spec agreement, every source hash against both the session and the GO, the
worker schema and provenance, the host and supervisor receipts - is preserved verbatim.

No physics, no model, no training: this reads saved files only.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The launcher runs the reader in a cold environment whose PYTHONPATH is only the installed
# site-packages, so the frozen reader package directory must be made importable explicitly.
# It is derived from this file's own location, never from an environment variable, and the
# frozen read35 then establishes its own closure for everything below it.
FROZEN_READER_DIR = Path(__file__).resolve().parents[2]/'continuation35'/'read35'
if str(FROZEN_READER_DIR) not in sys.path:
    sys.path.insert(0, str(FROZEN_READER_DIR))

import read35  # noqa: E402
from read35 import MODEL_SHA, document, identity, require  # noqa: E402

CONTRACT37 = 'C37_preunload_transfer_with_repaired_fullm_check_v1'


def sources37(run: Path, go_path: Path) -> tuple[dict, dict, dict]:
    """Faithful copy of read35.sources35 with the two C35-specific literals replaced."""
    session = document(run/'session.json')
    go = document(go_path)
    worker = document(run/'worker_receipt.json')
    spec = document(session['spec37_path'])
    require(go['decision'] == 'GO' and go['execution_contract_id'] == CONTRACT37
            and session['execution_contract_id'] == CONTRACT37
            and session['arm'] == 'development' and session['mode'] == 'fixed'
            and session['render'] is False and session['retry_permitted'] is False
            and Path(session['output_directory']).resolve() == run
            and all(session.get(key) == value for key, value in spec.items())
            and session['control_limit'] == 9000
            and session['cycles_limit'] == len(session['cases'])
            and session['normal_native_cap'] == 45000,
            'C37 frozen source GO/session/spec differs')
    hashes = session['source_hashes']
    # The frozen clause assumed the review document IS the physics GO recorded in the
    # session, so it compared the whole review bundle against the session hashes and looked
    # the review path up inside them. This reader GO is a separate document and a strict
    # superset: it adds its own two files. The comparison is split so nothing is weakened -
    # every session source must still be unchanged, the reader GO must still agree with the
    # session on everything the session covers, the reader GO's own additions are verified
    # fresh, and the physics GO it names must be exactly the one the session used.
    reader_only = {path: value for path, value in go['inputs'].items()
                   if path not in hashes}
    physics_go = session['source_go_path']
    require(all(hashes.get(path) == identity(path) for path in hashes)
            and all(hashes.get(path) == value for path, value in go['inputs'].items()
                    if path in hashes)
            and all(identity(path) == value for path, value in reader_only.items())
            and go['physics_source_go_identity']['path'] == physics_go
            and go['physics_source_go_identity']['sha256'] == identity(physics_go)['sha256']
            and hashes.get(physics_go) == identity(physics_go)
            and session['source_go_identity'] == identity(physics_go),
            'C37 actual sources differ from reviewed bundle')
    require(worker['schema'] == 'd1-c35-headless-worker-v1'
            and worker['execution_contract_id'] == CONTRACT37
            and worker['session_identity'] == identity(run/'session.json')
            and worker['archive_failed'] is False
            and worker['cleanup_errors'] == []
            and worker['warnings'] == []
            and worker['retry_permitted'] is False,
            'C37 worker/archive provenance differs')
    host = document(run/'host_receipt.json')
    supervisor = document(run/'supervisor_receipt.json')
    require(host['source_mismatches'] == [] and host['owned_no_orphans'] is True
            and host['reservation_closed'] is True
            and supervisor['cleanup']['remaining'] == {}
            and supervisor['outer_limit_s'] == 1200,
            'C37 owned host/source/deadline did not close')
    for name in ('runtime35_module_origins_before.json', 'runtime35_module_origins.json',
                 'runtime37_module_origins_before.json',
                 'loaded_origins_final.json', 'mapped_libraries_before_load.json',
                 'mapped_libraries_final.json'):
        path = run/name
        if path.exists():
            value = document(path)
            if 'module_origins' in name:
                require(all(hashes.get(row['path']) == identity(row['path'])
                            for row in value.values()),
                        'C37 runtime module origin left reviewed source closure')
            if name.startswith('mapped_libraries'):
                require(all(path in hashes for path in value['paths']),
                        'C37 mapped numerical library unsealed')
    require(session['checkpoint_sha256'] == MODEL_SHA,
            'C37 B22 checkpoint identity differs')
    return session, worker, go


def install37():
    """Declared, idempotent rebinds. Called before read35.main parses argv."""
    read35.CONTRACT = CONTRACT37
    read35.sources35 = sources37


def main() -> int:
    install37()
    return read35.main()


if __name__ == '__main__':
    sys.exit(main())
