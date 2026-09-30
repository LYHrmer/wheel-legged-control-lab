"""Validate the additive C22 public subset without running model or physics.

Default mode is read-only. --write-manifest creates one exclusive validation
manifest after every copied payload, JSON, Python AST, Markdown link, size,
symlink and credential check passes.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

R = Path('/home/lyh/wheel-legged-control-lab')
P = R/'results'/'d1_main_route_execution_20260930_22'
REPO_DOCS = (
    R/'docs'/'main_route_execution_20260930_22.md',
    R/'docs'/'current_status_20260930_22.md',
    R/'README.md',
    R/'docs'/'index.md',
)
MAX_FILE_BYTES = 100*1024*1024
TEXT_SUFFIXES = frozenset({'.json','.py','.md','.log','.csv','.txt','.ndjson','.yaml','.yml'})
CREDENTIAL = re.compile(rb'(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|'
                        rb'sk-(?:ant-|proj-)[A-Za-z0-9_-]{30,}|'
                        rb'-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----)')
MARKDOWN_LINK = re.compile(r'\[[^\]]*\]\(([^)]+)\)')


def identity(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError('nonregular public file: '+str(path))
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
            size += len(chunk)
    return {'bytes':size,'sha256':digest.hexdigest()}


def _local_links(path: Path, allowed_root: Path) -> list[dict]:
    result = []
    document = path.read_text(encoding='utf-8')
    for raw in MARKDOWN_LINK.findall(document):
        target = raw.strip().split(' ',1)[0].strip('<>')
        if target.startswith(('http://','https://','mailto:','data:','#')):
            continue
        filename = unquote(target.split('#',1)[0].split('?',1)[0])
        if not filename:
            continue
        linked = (path.parent/filename).resolve()
        if not linked.is_relative_to(allowed_root) or not linked.exists() or linked.is_symlink():
            raise ValueError('broken or external local Markdown link: '+str(path)+' -> '+raw)
        result.append({'document':str(path.relative_to(allowed_root)),'target':raw})
    return result


def validate(*, write_manifest: bool = False) -> dict:
    if not P.is_dir() or P.is_symlink():
        raise ValueError('C22 public destination is absent or linked')
    receipt_path = P/'publication_receipt.json'
    inventory_path = P/'local_archive_inventory.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    if (receipt.get('schema')!='d1-main-route-c22-selected-publication-v1'
            or receipt.get('copied') is not True
            or receipt.get('destination')!=str(P)
            or inventory.get('schema')!='d1-c22-complete-local-file-inventory-v1'
            or inventory.get('public_subset_cannot_rerun_full_native_reader') is not True):
        raise ValueError('C22 publication receipt/inventory identity differs')
    sample_set = set(receipt['preregistered_sample_gzip'])
    if set(receipt['absent_preregistered_samples']) != sample_set-set(inventory['files']):
        raise ValueError('absent preregistered C22 sample list differs from local inventory')
    payloads = receipt['payloads']
    for name,expected in payloads.items():
        path = P/name
        if not path.resolve().is_relative_to(P) or path.is_symlink():
            raise ValueError('public payload path escapes destination: '+name)
        if identity(path)!={key:expected[key] for key in ('bytes','sha256')}:
            raise ValueError('public copied payload changed: '+name)
        if inventory['files'][name]['included_in_public_package'] is not True:
            raise ValueError('copied payload lacks selected local inventory row: '+name)
        if {key:inventory['files'][name][key] for key in ('bytes','sha256')} != {
                key:expected[key] for key in ('bytes','sha256')}:
            raise ValueError('local inventory and public payload differ: '+name)
    for required,record in (('execution_summary_22.json',receipt['summary_identity']),
                            ('final_review_22.md',receipt['final_review_identity'])):
        if identity(P/required)!=record:
            raise ValueError('final C22 report changed in public copy: '+required)
    for required in ('predecessor_c21/execution_summary_21.json',
                     'predecessor_c21/failure_review_21.md',
                     'predecessor_c21/finite_geometry_21.json',
                     'predecessor_c21/geometry21.py',
                     'predecessor_c21/learning21.py',
                     'predecessor_c21/checkpoint21.py',
                     'predecessor_c21/state21.py',
                     'train_B_1/final_checkpoint/final_model.zip'):
        if required not in payloads:
            # A failure before checkpoint may legitimately have no final ZIP.
            if required.endswith('final_model.zip'):
                continue
            raise ValueError('mandatory C21/C22 source evidence omitted: '+required)

    manifest_path = P/'publication_manifest.json'
    if write_manifest and manifest_path.exists():
        raise FileExistsError('exclusive C22 validation manifest already exists')
    allowed_extra = {'publication_receipt.json','local_archive_inventory.json','README.md'}
    if not write_manifest and manifest_path.is_file():
        allowed_extra.add('publication_manifest.json')
    files = {}
    json_count = ast_count = 0
    links = []
    for path in sorted(P.rglob('*')):
        relative = str(path.relative_to(P))
        if path.is_symlink():
            raise ValueError('public package contains symlink: '+relative)
        if not path.is_file():
            continue
        if relative not in payloads and relative not in allowed_extra:
            raise ValueError('unrecorded public file: '+relative)
        size = path.stat().st_size
        if size>=MAX_FILE_BYTES:
            raise ValueError('public file is 100 MiB or larger: '+relative)
        if path.suffix in TEXT_SUFFIXES:
            data = path.read_bytes()
            if CREDENTIAL.search(data):
                raise ValueError('credential-like content in public file: '+relative)
            if path.suffix=='.json':
                json.loads(data)
                json_count += 1
            if path.suffix=='.py':
                ast.parse(data,filename=str(path))
                ast_count += 1
            if path.suffix=='.md':
                links.extend(_local_links(path,P))
        files[relative] = identity(path)
    repo_docs = {}
    for path in REPO_DOCS:
        if path.is_symlink() or not path.is_file() or path.stat().st_size>=MAX_FILE_BYTES:
            raise ValueError('required C22 repository document is absent/linked/oversized: '+str(path))
        raw = path.read_bytes()
        if CREDENTIAL.search(raw):
            raise ValueError('credential-like content in repository document: '+str(path))
        text = raw.decode('utf-8')
        if not text.strip():
            raise ValueError('required C22 repository document is empty: '+str(path))
        if path.name in ('main_route_execution_20260930_22.md',
                         'current_status_20260930_22.md') and 'C22' not in text:
            raise ValueError('C22 repository document lacks execution identity: '+str(path))
        links.extend(_local_links(path,R))
        repo_docs[str(path.relative_to(R))]=identity(path)
    receipt_info = {
        'schema':'d1-c22-publication-validation-v1',
        'files':files,
        'payload_count':len(payloads),
        'payload_bytes':sum(row['bytes'] for row in payloads.values()),
        'repository_documents':repo_docs,
        'software_validation':{
            'copied_identity_checks':len(payloads),
            'valid_json_files':json_count,
            'AST_valid_python_files':ast_count,
            'relative_markdown_links_checked':links,
            'credential_pattern_matches':0,
            'symlinks':0,
            'oversized_files':0,
        },
        'full_native_control_archives_uploaded':False,
        'full_reader_requires_local_archives':True,
        'project_model_calls':0,
        'physics_calls':0,
    }
    if write_manifest:
        with manifest_path.open('x',encoding='utf-8') as stream:
            json.dump(receipt_info,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write('\n')
    return receipt_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest',action='store_true')
    args = parser.parse_args()
    report = validate(write_manifest=args.write_manifest)
    print(json.dumps({'payload_count':report['payload_count'],
                      'payload_bytes':report['payload_bytes'],
                      'JSON_files':report['software_validation']['valid_json_files'],
                      'AST_files':report['software_validation']['AST_valid_python_files'],
                      'Markdown_links':len(report['software_validation']['relative_markdown_links_checked']),
                      'manifest_written':args.write_manifest},sort_keys=True))


if __name__=='__main__':
    main()
