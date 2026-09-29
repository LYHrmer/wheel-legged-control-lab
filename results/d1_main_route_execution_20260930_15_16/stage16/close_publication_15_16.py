"""Close a selected evidence package with exclusive writes and full hash checks."""
from pathlib import Path
import hashlib
import json

C = Path(__file__).resolve().parent
R = Path('/home/lyh/wheel-legged-control-lab')
D = R / 'results/d1_main_route_execution_20260930_15_16'


def identity(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    base = json.loads((C / 'publication_base_copy_receipt_15_16.json').read_text())
    records = {}
    for row in base['files']:
        original = Path(row['original_source'])
        target = D / row['target']
        expected = {key: row[key] for key in ('bytes', 'sha256')}
        assert identity(original) == identity(target) == expected
        records[str(target.relative_to(R))] = {**expected, 'source': str(original)}
    names = (
        'eval_readback_16.json', 'eval_readback_16.stdout.log',
        'eval_readback_command_receipt_16.json', 'execution_summary_15_16.json',
        'result_summary_15_16.json', 'task_results_15_16.csv',
        'local_archive_inventory_15_16.json', 'frozen_source_post16.json',
        'continuation_state_16.json', 'final_review_15_16.md',
        'close_publication_15_16.py',
    )
    copies = [(C / name, D / 'stage16' / name) for name in names]
    copies.extend((p, D / 'stage16' / p.relative_to(C))
                  for p in sorted((C / 'pure_test_artifacts_16').rglob('*'))
                  if p.is_file() and not p.is_symlink())
    copies.extend((
        (C / 'public_readme_15_16.md', D / 'README.md'),
        (C / 'main_route_execution_20260930_15_16.md',
         R / 'docs/main_route_execution_20260930_15_16.md'),
    ))
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        assert identity(source) == identity(target)
        records[str(target.relative_to(R))] = {**identity(target), 'source': str(source)}
    summary = json.loads((C / 'result_summary_15_16.json').read_text())
    manifest = {
        'schema': 'd1-publication-manifest-15-16-v1', 'files': records,
        'payload_count': len(records),
        'payload_bytes': sum(row['bytes'] for row in records.values()),
        'old_tracked_files_modified': 0,
        'scope': 'Paired clipping continuation and separately reserved repaired evaluation.',
        'training_controls': 32768, 'evaluation_controls': 30600,
        'actual_normal_native': 316840, 'actual_compiler_native': 8,
        'reserved_controls': 93968, 'reserved_normal_native': 469840,
        'failed_stage15_evaluation_preserved': True,
        'grouped_global_promoted': summary['grouped_vs_global']['passed'],
        'qualified_for_default_GUI': False,
        'full_native_archive_uploaded': False,
        'full_local_archive_inventory': 'stage16/local_archive_inventory_15_16.json',
        'required_for_full_readback': 'Retained raw payloads plus the recorded frozen source/dependency layout.',
        'software_validation': {'stage15_necessary_tests': 22, 'stage16_import_regressions': 2,
                                'stage16_ast_and_fatal_lint': 'passed'},
    }
    with (D / 'publication_manifest.json').open('x') as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
    for name, row in records.items():
        assert identity(R / name) == {k: row[k] for k in ('bytes', 'sha256')}
    receipt = {'payload_count': len(records), 'payload_bytes': manifest['payload_bytes'],
               'all_copied_hashes_match': True,
               'manifest_identity': identity(D / 'publication_manifest.json')}
    with (C / 'publication_copy_receipt_15_16.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
