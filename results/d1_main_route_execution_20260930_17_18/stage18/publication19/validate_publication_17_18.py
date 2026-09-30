"""Check copied identities and public document links, then seal the payload list."""
from pathlib import Path
import ast
import hashlib
import json
import re

R = Path('/home/lyh/wheel-legged-control-lab')
P = R / 'results/d1_main_route_execution_20260930_17_18'
REPORT = R / 'docs/main_route_execution_20260930_17_18.md'


def identity(path):
    return {'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    destination = P / 'publication_manifest.json'
    if destination.exists():
        raise FileExistsError(destination)
    copies = json.loads((P / 'copy_receipt.json').read_text())
    for name, row in copies['files'].items():
        if identity(P / name) != {key: row[key] for key in ('bytes', 'sha256')}:
            raise ValueError('public copy changed: ' + name)
    entries = {}
    json_count = ast_count = 0
    secret = re.compile(rb'(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:ant-|proj-)[A-Za-z0-9_-]{30,}|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----)')
    for path in sorted(P.rglob('*')):
        if not path.is_file():
            continue
        rel = str(path.relative_to(P))
        data = path.read_bytes()
        if path.is_symlink() or len(data) >= 100 * 1024 * 1024:
            raise ValueError('linked or oversized public artifact: ' + rel)
        if path.suffix in ('.json', '.py', '.md', '.log', '.csv') and secret.search(data):
            raise ValueError('credential-like content requires review: ' + rel)
        if path.suffix == '.json':
            json.loads(data)
            json_count += 1
        if path.suffix == '.py':
            ast.parse(data, filename=str(path))
            ast_count += 1
        entries[rel] = {**identity(path), 'local_source': copies['files'].get(rel, {}).get('local_source')}
    links = []
    for document in (P / 'README.md', REPORT):
        for reference in re.findall(r'\[[^\]]*\]\(([^)]+)\)', document.read_text()):
            if reference.startswith(('https://', 'http://', '#')):
                continue
            path = (document.parent / reference.split('#')[0]).resolve()
            if path != destination and not path.is_file():
                raise ValueError('broken publication link: ' + reference)
            links.append({'document': str(document.relative_to(R)), 'target': reference})
    summary = json.loads((P / 'stage18/qualification_summary_18.json').read_text())
    manifest = {
        'schema': 'd1-publication-manifest-17-18-v1', 'files': entries,
        'payload_count': len(entries), 'payload_bytes': sum(row['bytes'] for row in entries.values()),
        'report': {'path': str(REPORT.relative_to(R)), **identity(REPORT)},
        'fixed_script_task_qualification': summary['qualification_passed'],
        'RL_contribution_passed': summary['primary_six_task_RL_contribution']['RL_contribution_passed'],
        'qualified_for_default_GUI': False, 'new_training_controls': 0,
        'actual_totals': summary['C17_plus_C18_actual_totals'],
        'full_native_archive_uploaded': False,
        'required_for_full_readback': 'Full local control/native gzip and frozen source/dependency paths',
        'software_validation': {'copied_identity_checks': len(copies['files']),
            'valid_json_files': json_count, 'AST_valid_python_files': ast_count,
            'relative_links_checked': links, 'credential_pattern_matches': 0,
            'new_runtime_pure_tests_C17': 11, 'new_runtime_pure_tests_C18': 10},
        'parent_commit': copies['parent_commit'], 'old_tracked_files_modified': False,
    }
    with destination.open('x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'manifest': identity(destination), 'payload_count': len(entries),
                      'payload_bytes': manifest['payload_bytes'], 'relative_links': len(links),
                      'JSON_files': json_count, 'AST_files': ast_count}))


if __name__ == '__main__':
    main()
