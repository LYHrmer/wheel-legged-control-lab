"""Validate the additive public C20 payload without executing its runtime."""
from pathlib import Path
import ast
import hashlib
import json
import re

R = Path('/home/lyh/wheel-legged-control-lab')
P = R / 'results/d1_main_route_execution_20260930_20'
REPORT = R / 'docs/main_route_execution_20260930_20.md'
STATUS = R / 'docs/current_status_20260930_20.md'


def identity(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    output = P / 'publication_manifest.json'
    if output.exists():
        raise FileExistsError(output)
    copies = json.loads((P / 'publication_receipt.json').read_text())
    for name, row in copies['payloads'].items():
        if identity(P / name) != {key: row[key] for key in ('bytes', 'sha256')}:
            raise ValueError('public copy changed: ' + name)
    files = {}
    json_count = ast_count = 0
    credentials = re.compile(rb'(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:ant-|proj-)[A-Za-z0-9_-]{30,}|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----)')
    for path in sorted(P.rglob('*')):
        if not path.is_file():
            continue
        rel = str(path.relative_to(P))
        data = path.read_bytes()
        if path.is_symlink() or len(data) >= 100 * 1024 * 1024:
            raise ValueError('linked or oversized artifact: ' + rel)
        if path.suffix in ('.json', '.py', '.md', '.log', '.csv') and credentials.search(data):
            raise ValueError('credential-like content: ' + rel)
        if path.suffix == '.json':
            json.loads(data)
            json_count += 1
        if path.suffix == '.py':
            ast.parse(data, filename=str(path))
            ast_count += 1
        files[rel] = identity(path)
    links = []
    for document in (P / 'README.md', REPORT, STATUS, R / 'README.md', R / 'docs/index.md'):
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', document.read_text()):
            if target.startswith(('https://', 'http://', '#', 'mailto:')):
                continue
            path = (document.parent / target.split('#')[0]).resolve()
            if path != output and not path.exists():
                raise ValueError('broken link in ' + str(document) + ': ' + target)
            links.append({'document': str(document.relative_to(R)), 'target': target})
    receipt = {
        'schema': 'd1-c20-publication-manifest-v1', 'files': files,
        'payload_count': len(files), 'payload_bytes': sum(x['bytes'] for x in files.values()),
        'reports': {str(p.relative_to(R)): identity(p) for p in (REPORT, STATUS)},
        'software_validation': {
            'copied_identity_checks': len(copies['payloads']),
            'valid_json_files': json_count, 'AST_valid_python_files': ast_count,
            'relative_links_checked': links, 'credential_pattern_matches': 0,
        },
        'full_native_control_archives_uploaded': False,
        'project_model_calls': 0, 'physics_calls': 0,
    }
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'manifest': identity(output), 'payload_count': len(files),
                      'payload_bytes': receipt['payload_bytes'], 'links': len(links),
                      'JSON_files': json_count, 'AST_files': ast_count}))


if __name__ == '__main__':
    main()
