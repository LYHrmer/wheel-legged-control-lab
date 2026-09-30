"""Copy an additive, explicitly partial public evidence package after closure."""
from pathlib import Path
import hashlib
import json

C = Path(__file__).resolve().parent.parent
R = Path('/home/lyh/wheel-legged-control-lab')
DEST = R / 'results/d1_main_route_execution_20260930_17'
RUNS = ('development_baseline_01', 'development_yaw_01', 'development_ramp_01',
        'qualification_primary_01', 'qualification_mirror_01')


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def dump(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    plan = json.loads((C / 'plan_go_C17.json').read_text())
    for path, expected in plan['inputs'].items():
        if identity(Path(path)) != expected:
            raise ValueError('frozen source mismatch: ' + path)
    for run in RUNS:
        host = json.loads((C / run / 'host_receipt.json').read_text())
        if (host['failure'] is not None or host['cleanup_errors'] or host['changed_sources']
                or not host['postcheck_complete'] or not host['no_live_owned_processes']):
            raise ValueError('host not closed: ' + run)
    for arm in ('primary', 'mirror'):
        reader = json.loads((C / f'qualification_{arm}_readback_17.json').read_text())
        if not reader['source_closure_verified']:
            raise ValueError('qualification reader not closed')
    for name in ('qualification_summary_17.json', 'final_review_17.md'):
        if not (C / name).is_file():
            raise ValueError('final summary/review absent: ' + name)
    sample_blocks = {
        'qualification_primary_01/heldout/flat_1p2_yaw_grouped_continue/control_records_0002.jsonl.gz',
        'qualification_primary_01/heldout/flat_1p2_yaw_grouped_continue/native_block_0000.jsonl.gz',
        'qualification_primary_01/heldout/ramp_0p45_complete_grouped_continue/control_records_0003.jsonl.gz',
        'qualification_primary_01/heldout/ramp_0p45_complete_grouped_continue/native_block_0000.jsonl.gz',
        'qualification_mirror_01/heldout/flat_1p2_yaw_grouped_continue/control_records_0002.jsonl.gz',
        'qualification_mirror_01/heldout/flat_1p2_yaw_grouped_continue/native_block_0000.jsonl.gz',
    }
    files = sorted(p for p in C.rglob('*') if p.is_file()
                   and not any(part in {'.ruff_cache', '__pycache__'} for part in p.relative_to(C).parts))
    inventory = {}
    for path in files:
        rel = str(path.relative_to(C))
        if path.is_symlink():
            raise ValueError('unexpected linked artifact: ' + rel)
        inventory[rel] = {**identity(path), 'local_source': str(path),
                          'included_in_public_package': not rel.endswith('.gz') or rel in sample_blocks}
    dump(C / 'local_archive_inventory_17.json', {
        'schema': 'd1-local-archive-inventory-17-v1', 'files': inventory,
        'total_bytes': sum(row['bytes'] for row in inventory.values()),
        'total_files': len(inventory), 'snapshot_excludes_itself_and_later_publication_receipts': True,
        'public_subset_cannot_rerun_complete_native_reader': True,
    })
    DEST.mkdir()
    selected = [p for p in files if inventory[str(p.relative_to(C))]['included_in_public_package']]
    selected.append(C / 'local_archive_inventory_17.json')
    copied = {}
    for path in selected:
        rel = Path('stage17') / path.relative_to(C)
        out = DEST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        expected = identity(path)
        with out.open('xb') as stream:
            stream.write(path.read_bytes())
        if identity(out) != expected:
            raise ValueError('copy mismatch: ' + str(rel))
        copied[str(rel)] = {**expected, 'local_source': str(path)}
    dump(DEST / 'copy_receipt.json', {
        'schema': 'd1-public-copy-17-v1', 'files': copied,
        'payload_bytes': sum(row['bytes'] for row in copied.values()),
        'payload_files': len(copied), 'source_plan_C_sha256': identity(C / 'plan_go_C17.json')['sha256'],
        'parent_commit': 'd5b8f8ccf1174a316365a3cbc83200478a4bac52',
        'checkpoint_sha256': '1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb',
        'contains_all_local_control_native_gzip': False,
    })
    print(json.dumps({'destination': str(DEST), 'files': len(copied),
                      'bytes': sum(row['bytes'] for row in copied.values()),
                      'local_files': len(inventory),
                      'local_bytes': sum(row['bytes'] for row in inventory.values())}))


if __name__ == '__main__':
    main()
