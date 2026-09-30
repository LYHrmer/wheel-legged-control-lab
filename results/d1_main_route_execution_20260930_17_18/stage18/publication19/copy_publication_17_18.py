"""Publish an additive selected evidence snapshot; full raw archives stay local."""
from pathlib import Path
import hashlib
import json

C = Path(__file__).resolve().parent.parent
W = C.parent
R = Path('/home/lyh/wheel-legged-control-lab')
DEST = R / 'results/d1_main_route_execution_20260930_17_18'
STAGES = {'stage17': W / 'continuation17', 'stage18': C}
RUNS = {
    'stage17': ('development_baseline_01', 'development_yaw_01', 'development_ramp_01',
                'qualification_primary_01', 'qualification_mirror_01'),
    'stage18': ('development_filtered_01', 'qualification_primary_01', 'qualification_mirror_01'),
}


def identity(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def dump(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    plan = json.loads((C / 'plan_go_C18.json').read_text())
    for name, expected in plan['inputs'].items():
        if identity(Path(name)) != expected:
            raise ValueError('frozen source changed: ' + name)
    summary = json.loads((C / 'qualification_summary_18.json').read_text())
    if summary['qualification_passed'] is not True:
        raise ValueError('final qualification has not passed')
    if not (C / 'final_review_17_18.md').is_file():
        raise ValueError('Astra final review missing')
    for stage, names in RUNS.items():
        for name in names:
            host = json.loads((STAGES[stage] / name / 'host_receipt.json').read_text())
            if (host['failure'] is not None or host['cleanup_errors'] or host['changed_sources']
                    or not host['postcheck_complete'] or not host['no_live_owned_processes']):
                raise ValueError('host not closed: ' + name)
    samples = set()
    for stage, run, case, block in (
        ('stage17', 'qualification_primary_01', 'rough_0p35', 1),
        ('stage18', 'qualification_primary_01', 'rough_0p35', 1),
        ('stage18', 'qualification_primary_01', 'flat_1p2_yaw', 2),
        ('stage18', 'qualification_primary_01', 'ramp_0p45_complete', 3),
        ('stage18', 'qualification_mirror_01', 'flat_1p2_yaw', 2),
    ):
        stem = f'{stage}/{run}/heldout/{case}_grouped_continue/'
        samples.add(stem + f'control_records_{block:04d}.jsonl.gz')
        samples.add(stem + 'native_block_0000.jsonl.gz')
    sources, inventory = {}, {}
    for stage, folder in STAGES.items():
        for source in sorted(folder.rglob('*')):
            if not source.is_file() or any(part in {'.ruff_cache', '__pycache__'}
                                          for part in source.relative_to(folder).parts):
                continue
            if source.is_symlink():
                raise ValueError('unexpected symlink: ' + str(source))
            relative = str(Path(stage) / source.relative_to(folder))
            selected = not relative.endswith('.gz') or relative in samples
            inventory[relative] = {**identity(source), 'local_source': str(source),
                                   'included_in_public_package': selected}
            if selected:
                sources[relative] = source
    local_inventory = C / 'local_archive_inventory_17_18.json'
    dump(local_inventory, {
        'schema': 'd1-local-archive-17-18-v1', 'files': inventory,
        'snapshot_excludes_itself_and_later_publication_receipts': True,
        'total_files': len(inventory), 'total_bytes': sum(row['bytes'] for row in inventory.values()),
        'public_subset_cannot_rerun_full_native_reader': True,
    })
    sources['stage18/local_archive_inventory_17_18.json'] = local_inventory
    DEST.mkdir()
    copied = {}
    for relative, source in sources.items():
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = identity(source)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        if identity(target) != expected:
            raise ValueError('copy mismatch: ' + relative)
        copied[relative] = {**expected, 'local_source': str(source)}
    receipt = {
        'schema': 'd1-selected-public-copy-17-18-v1', 'files': copied,
        'payload_files': len(copied), 'payload_bytes': sum(row['bytes'] for row in copied.values()),
        'local_archive_files': len(inventory), 'local_archive_bytes': sum(row['bytes'] for row in inventory.values()),
        'full_control_native_archive_uploaded': False,
        'parent_commit': 'd5b8f8ccf1174a316365a3cbc83200478a4bac52',
        'selected_checkpoint_sha256': '1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb',
        'old_tracked_files_modified': False,
    }
    dump(DEST / 'copy_receipt.json', receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != 'files'}))


if __name__ == '__main__':
    main()
