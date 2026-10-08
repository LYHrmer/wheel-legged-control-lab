"""C38 independent saved-data reader: the frozen C35 reader with five declared rebinds.

Identical pattern to continuation37/read37/read37.py for the contract/spec rebinds. In
addition, episode_read35's native/mask verification is extended to the pre-unload regime
C36/C37/C38 introduced inside transfer/transfer_restore (native_read38.py), and the model
ledger's predict-call ceiling is generalised from the hardcoded C35 5-case literal to this
campaign's 2-case budget (ledger_read38.py). See each module's docstring for why these are
documentation gaps in the independent reader rather than weakenings.

No physics, no model, no training: this reads saved files only.
"""
from __future__ import annotations

import sys
from pathlib import Path

FROZEN_READER_DIR = Path(__file__).resolve().parents[2]/'continuation35'/'read35'
if str(FROZEN_READER_DIR) not in sys.path:
    sys.path.insert(0, str(FROZEN_READER_DIR))

import read35  # noqa: E402  (must import first: injects sys.path for everything below)
import episode_read35  # noqa: E402
from ledger_read38 import check_ledger38  # noqa: E402
from native_read38 import verify_native38  # noqa: E402
from read35 import MODEL_SHA, document, identity, require  # noqa: E402

CONTRACT38 = 'C38_preunload_record_semantics_and_audit_v1'


def sources38(run: Path, go_path: Path) -> tuple[dict, dict, dict]:
    """Faithful copy of read35.sources35 with the C35-specific literals replaced."""
    session = document(run/'session.json')
    go = document(go_path)
    worker = document(run/'worker_receipt.json')
    spec = document(session['spec38_path'])
    require(go['decision'] == 'GO' and go['execution_contract_id'] == CONTRACT38
            and session['execution_contract_id'] == CONTRACT38
            and session['arm'] == 'development' and session['mode'] == 'fixed'
            and session['render'] is False and session['retry_permitted'] is False
            and Path(session['output_directory']).resolve() == run
            and all(session.get(key) == value for key, value in spec.items())
            and session['control_limit'] == 9000
            and session['cycles_limit'] == len(session['cases'])
            and session['normal_native_cap'] == 45000,
            'C38 frozen source GO/session/spec differs')
    hashes = session['source_hashes']
    reader_only = {path: value for path, value in go['inputs'].items() if path not in hashes}
    physics_go = session['source_go_path']
    require(all(hashes.get(path) == identity(path) for path in hashes)
            and all(hashes.get(path) == value for path, value in go['inputs'].items()
                    if path in hashes)
            and all(identity(path) == value for path, value in reader_only.items())
            and go['physics_source_go_identity']['path'] == physics_go
            and go['physics_source_go_identity']['sha256'] == identity(physics_go)['sha256']
            and hashes.get(physics_go) == identity(physics_go)
            and session['source_go_identity'] == identity(physics_go),
            'C38 actual sources differ from reviewed bundle')
    require(worker['schema'] == 'd1-c35-headless-worker-v1'
            and worker['execution_contract_id'] == CONTRACT38
            and worker['session_identity'] == identity(run/'session.json')
            and worker['archive_failed'] is False
            and worker['cleanup_errors'] == []
            and worker['warnings'] == []
            and worker['retry_permitted'] is False,
            'C38 worker/archive provenance differs')
    host = document(run/'host_receipt.json')
    supervisor = document(run/'supervisor_receipt.json')
    require(host['source_mismatches'] == [] and host['owned_no_orphans'] is True
            and host['reservation_closed'] is True
            and supervisor['cleanup']['remaining'] == {}
            and supervisor['outer_limit_s'] == 1200,
            'C38 owned host/source/deadline did not close')
    for name in ('runtime35_module_origins_before.json', 'runtime35_module_origins.json',
                 'runtime38_module_origins_before.json',
                 'loaded_origins_final.json', 'mapped_libraries_before_load.json',
                 'mapped_libraries_final.json'):
        path = run/name
        if path.exists():
            value = document(path)
            if 'module_origins' in name:
                require(all(hashes.get(row['path']) == identity(row['path'])
                            for row in value.values()),
                        'C38 runtime module origin left reviewed source closure')
            if name.startswith('mapped_libraries'):
                require(all(path in hashes for path in value['paths']),
                        'C38 mapped numerical library unsealed')
    require(session['checkpoint_sha256'] == MODEL_SHA,
            'C38 B22 checkpoint identity differs')
    return session, worker, go


def install38():
    read35.CONTRACT = CONTRACT38
    read35.sources35 = sources38
    read35.check_ledger35 = check_ledger38
    episode_read35.verify_native35 = verify_native38


def main() -> int:
    install38()
    return read35.main()


if __name__ == '__main__':
    sys.exit(main())
