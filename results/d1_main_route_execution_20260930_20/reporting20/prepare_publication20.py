"""Prepare an additive C20 public snapshot; default invocation is read-only.

The full control/native archives remain local and receive a per-file SHA256
inventory. This tool neither runs the saved-data readers nor loads a model.
Use --copy only after the summary and final review have closed the experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

C = Path(__file__).resolve().parent.parent
R = Path('/home/lyh/wheel-legged-control-lab')
DEST = R / 'results' / 'd1_main_route_execution_20260930_20'
SCHEMA = 'd1-main-route-c20-selected-publication-v1'

# Selected before the final results are known. Missing samples are recorded,
# because a failed or unstarted arm must not be hidden by the packaging tool.
SAMPLE_GZIP = frozenset({
    *{f'train_{arm}_{stage}/training/{name}_0000.jsonl.gz'
      for arm in ('A', 'B') for stage in (1, 2)
      for name in ('full_control_records', 'native_block')},
    *{f'{phase}_{stage}/heldout/{case}_B/{name}'
      for phase, stage, case in (
          ('development', 1, 'dev_yaw_left'),
          ('final', 1, 'final_yaw_left'),
          ('final', 2, 'final_yaw_left'))
      for name in ('control_records_0002.jsonl.gz', 'native_block_0000.jsonl.gz')},
})


def identity(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError('not a regular local file: ' + str(path))
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def document(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('expected JSON object: ' + str(path))
    return value


def _inside(path: Path, parent: Path) -> Path:
    if path.is_symlink():
        raise ValueError('required C20 report is a symlink: ' + str(path))
    real = path.resolve(strict=True)
    if not real.is_file() or real.is_symlink() or not real.is_relative_to(parent):
        raise ValueError('required C20 report is not a regular file inside C20: ' + str(path))
    return real


def _complete_host(run: Path) -> dict:
    host_path = run / 'host_receipt.json'
    if not host_path.is_file():
        raise ValueError('reserved run has no closed host receipt: ' + str(run))
    host = document(host_path)
    if (host.get('postcheck_complete') is not True
            or host.get('no_live_owned_processes') is not True
            or host.get('fully_reserved_budget_closed') is not True):
        raise ValueError('reserved run is not closed: ' + str(run))
    worker_path = run / 'worker_receipt.json'
    worker = document(worker_path) if worker_path.is_file() else None
    return {
        'run': run.name,
        'host_identity': identity(host_path),
        'worker_identity': identity(worker_path) if worker is not None else None,
        'worker_receipt_present': worker is not None,
        'host_failure': host.get('failure'),
        'worker_failure': None if worker is None else worker.get('failure'),
        'worker_execution_complete': None if worker is None else worker.get('execution_complete'),
        'independent_readback_present': (C / f'{run.name}_readback.json').is_file(),
    }


def _included(relative: str) -> bool:
    path = Path(relative)
    name = path.name
    if name.endswith('.manifest.json'):
        return _included(str(path.with_name(name.removesuffix('.manifest.json'))))
    if name.endswith('.jsonl.gz') or name.endswith('.gz'):
        return relative in SAMPLE_GZIP
    if path.suffix == '.npz':
        if 'gradients' in path.parts:
            return True
        if name in ('initial_state.npz', 'states.npz',
                    'controls_0000.npz', 'gaussian_0000.npz'):
            return True
        if name == 'training_episode_000018_flat_corridor_nominal_path.npz':
            return True  # B1 failure's saved command-integrated path and manifest.
        return name == 'training_episode_000000_initial_state.npz'
    return True


def _inventory() -> tuple[dict, dict]:
    inventory = {}
    selected = {}
    for path in sorted(C.rglob('*')):
        if not path.is_file():
            continue
        relative_path = path.relative_to(C)
        if any(part in ('.ruff_cache', '__pycache__') for part in relative_path.parts):
            continue
        if path.is_symlink():
            raise ValueError('C20 publication source symlink: ' + str(path))
        relative = str(relative_path)
        included = _included(relative)
        row = {**identity(path), 'local_source': str(path),
               'included_in_public_package': included}
        inventory[relative] = row
        if included:
            selected[relative] = path
    return inventory, selected


def _write_exclusive(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def prepare(*, summary_path: Path, review_path: Path, copy: bool = False) -> dict:
    summary_path = _inside(summary_path, C)
    review_path = _inside(review_path, C)
    summary = document(summary_path)
    if not review_path.read_text(encoding='utf-8').strip():
        raise ValueError('final review is empty')
    if DEST.exists():
        raise FileExistsError('additive publication destination already exists: ' + str(DEST))

    plans = sorted(C.glob('plan_go_*_20.json'))
    if not plans:
        raise ValueError('C20 source GO is missing')
    source_go = {}
    for path in plans:
        plan = document(path)
        if plan.get('status') != 'GO' or plan.get('retry_permitted') is not False:
            raise ValueError('non-GO source plan: ' + str(path))
        for source, expected in plan['inputs'].items():
            if identity(Path(source)) != expected:
                raise ValueError('frozen GO input changed: ' + source)
        source_go[path.name] = identity(path)

    closed = []
    offline_reservations = []
    for reservation in sorted(C.glob('*_reservation.json')):
        reservation_value = document(reservation)
        if 'reservation_key' not in reservation_value:
            offline_reservations.append(reservation.name)
            continue
        key = reservation_value['reservation_key']
        if reservation.name != key+'_reservation.json':
            raise ValueError('worker reservation identity differs: '+str(reservation))
        run = C / key
        if not run.is_dir() or run.is_symlink():
            raise ValueError('reserved run directory missing: ' + str(run))
        closed.append(_complete_host(run))
    inventory, selected = _inventory()
    for required in (summary_path, review_path):
        if str(required.relative_to(C)) not in selected:
            raise ValueError('required final report omitted from selected files')

    absent_samples = sorted(SAMPLE_GZIP - inventory.keys())
    report = {
        'schema': SCHEMA,
        'summary_identity': identity(summary_path),
        'final_review_identity': identity(review_path),
        'reported_outcome': {key: summary[key] for key in
                             ('status', 'qualification_passed', 'training_valid',
                              'continue_training', 'final_task_passed') if key in summary},
        'source_go_plans': source_go,
        'closed_reserved_runs': closed,
        'offline_reservations_preserved': offline_reservations,
        'preregistered_sample_gzip': sorted(SAMPLE_GZIP),
        'absent_preregistered_samples': absent_samples,
        'local_files': len(inventory),
        'local_bytes': sum(row['bytes'] for row in inventory.values()),
        'selected_files': len(selected),
        'selected_bytes': sum(inventory[name]['bytes'] for name in selected),
        'full_control_native_archives_in_public_subset': False,
        'full_readback_requires_local_archives': True,
        'copied': False,
    }
    if not copy:
        return report

    DEST.mkdir(parents=False)
    payloads = {}
    for relative, source in selected.items():
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as original, target.open('xb') as copied:
            shutil.copyfileobj(original, copied, 1024 * 1024)
        expected = {key: inventory[relative][key] for key in ('sha256', 'bytes')}
        if identity(target) != expected:
            raise ValueError('public payload differs after copy: ' + relative)
        payloads[relative] = {**expected, 'local_source': str(source)}
    _write_exclusive(DEST / 'local_archive_inventory.json', {
        'schema': 'd1-c20-complete-local-file-inventory-v1',
        'files': inventory,
        'snapshot_excludes_publication_outputs': True,
        'public_subset_cannot_rerun_full_native_reader': True,
    })
    report['copied'] = True
    report['destination'] = str(DEST)
    report['payloads'] = payloads
    _write_exclusive(DEST / 'publication_receipt.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--final-review', type=Path, required=True)
    parser.add_argument('--copy', action='store_true',
                        help='copy the selected snapshot to a new results directory')
    args = parser.parse_args()
    report = prepare(summary_path=args.summary, review_path=args.final_review,
                     copy=args.copy)
    print(json.dumps({key: value for key, value in report.items() if key != 'payloads'},
                     sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
