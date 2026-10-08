"""Byte-check and publish the C36-C38 research subset after closure.

C36/C37/C38 are one continuous investigation: the pre-unload stance-pair correction,
physically confirmed, its latent fullM defect found and repaired, its record-semantics
defect found and repaired, and its independent audit finally closed. None of the three
closed a pair exchange; each stopped on a distinct, understood cause. This packer asserts
that shape and refuses to publish anything claiming otherwise.

Byte-exact copies for small text/JSON/Python/Markdown evidence; raw native/control/state
payloads (jsonl.gz, npz) stay local, matching the established C34/C35 exclusion policy.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

MAX_FILE = 10*1024*1024
PUBLIC_SUFFIXES = {'.py', '.json', '.md', '.txt', '.log', '.stderr', '.stdout'}
RAW_SUFFIXES = {'.npz', '.npy', '.gz', '.jsonl', '.pt', '.pth', '.ckpt',
                '.safetensors', '.onnx', '.bin', '.h5', '.hdf5', '.parquet', '.zip'}
SECTION_NAMES = ('continuation36', 'continuation37', 'continuation38')
EXTRA_ROOT_FILES = ('extract_rocking_diagnosis_37.py',)
RUN_SCRIPTS_SOURCE = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/'
                          'stability_20260924_full_drive01')
REQUIRED_PER_SECTION = {
    'continuation36': ('astra_plan/contract36.json', 'source_go36.json', 'request36_01.json',
                       'sol36/core36.py', 'sol36/adapter36.py', 'sol36/test_core36_pure.py',
                       'root36/worker36.py', 'development_01/worker_receipt.json',
                       'development_01/case35_00.json', 'development_01/campaign35.json',
                       'development_01/episode_0/cycle_receipt35.json',
                       'development_01/episode_0/entry35.json'),
    'continuation37': ('astra_plan/contract37.json', 'source_go37.json', 'request37_01.json',
                       'root37/runtime37.py', 'root37/worker37.py',
                       'root37/test_runtime37_pure.py',
                       'read37/read37.py', 'build_reader_go37.py',
                       'reader_source_go37.json',
                       'development_01/worker_receipt.json',
                       'development_01/case35_00.json', 'development_01/campaign35.json',
                       'development_01/episode_0/cycle_receipt35.json',
                       'independent_read37_01.json',
                       'rocking_diagnosis_37.json'),
    'continuation38': ('astra_plan/contract38.json', 'source_go38.json', 'request38_01.json',
                       'sol38/adapter38.py', 'sol38/test_adapter38_pure.py',
                       'root38/worker38.py',
                       'read38/read38.py', 'read38/native_read38.py',
                       'read38/ledger_read38.py',
                       'read38/test_native_read38_pure.py',
                       'read38/test_ledger_read38_pure.py',
                       'build_reader_go38.py', 'reader_source_go38.json',
                       'development_01/worker_receipt.json',
                       'development_01/case35_00.json', 'development_01/campaign35.json',
                       'development_01/episode_0/cycle_receipt35.json',
                       'independent_read38_05.json'),
}


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def verify_closure(work: Path) -> dict:
    for name, required in REQUIRED_PER_SECTION.items():
        section = work/name
        for item in required:
            path = section/item
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError('missing required evidence: '+str(path))
    c36 = json.loads((work/'continuation36/development_01/campaign35.json').read_text())
    c37 = json.loads((work/'continuation37/development_01/campaign35.json').read_text())
    c38 = json.loads((work/'continuation38/development_01/campaign35.json').read_text())
    c38_audit = json.loads((work/'continuation38/independent_read38_05.json').read_text())
    if (c36['closed_cases'] != 0 or c37['closed_cases'] != 0 or c38['closed_cases'] != 0
            or c36['pair_exchanges'] != 0 or c37['pair_exchanges'] != 0
            or c38['pair_exchanges'] != 0
            or c36['cases'][0]['failure']['message'].find('qM') < 0
            or 'touchdown_dwell_timeout' not in c37['cases'][0]['failure']['message']
            or 'touchdown_dwell_timeout' not in c38['cases'][0]['failure']['message']
            or c38_audit.get('record_auditable') is not True
            or c38_audit.get('ledger', {}).get('passed') is not True
            or c38_audit.get('ledger', {}).get('actual_normal_native') != 1495
            or c38_audit.get('qualification') is not False):
        raise RuntimeError('C36-C38 saved campaigns differ from the closed negative-plus-'
                           'audited shape this packer asserts')
    return {'c36_controls': c36['controls'], 'c37_controls': c37['controls'],
            'c38_controls': c38['controls'], 'c38_ledger_native_checked':
                c38_audit['ledger']['native_checked']}


def inventory(section: Path):
    regular, special = [], []
    for current, folders, names in os.walk(section, followlinks=False):
        folder = Path(current)
        kept = []
        for name in sorted(folders):
            path = folder/name
            if path.is_symlink():
                special.append({'path': path.relative_to(section).as_posix(),
                                'kind': 'directory', 'reason': 'symbolic_link_not_followed'})
            else:
                kept.append(name)
        folders[:] = kept
        for name in sorted(names):
            path = folder/name
            relative = path.relative_to(section).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISREG(mode):
                regular.append({'path': relative, **identity(path)})
            else:
                special.append({'path': relative, 'kind': 'file',
                                'reason': 'symbolic_link_not_followed' if stat.S_ISLNK(mode)
                                          else 'non_regular_file'})
    return sorted(regular, key=lambda row: row['path']), sorted(special, key=lambda row: row['path'])


def exclusion(record: dict) -> str | None:
    path = Path(record['path'])
    name = path.name.lower()
    critical = 'receipt' in name or 'failure' in name or name.startswith(
        ('independent_read', 'reader_source_go', 'source_go', 'contract', 'rocking_diagnosis'))
    if '__pycache__' in path.parts or name.endswith('.pyc'):
        return 'python_bytecode_cache'
    if path.suffix.lower() in RAW_SUFFIXES or name.endswith('.jsonl.gz'):
        if critical:
            raise ValueError('mandatory receipt is a raw payload: '+record['path'])
        return 'raw_native_control_state_payload'
    if path.suffix.lower() not in PUBLIC_SUFFIXES:
        if critical:
            raise ValueError('mandatory receipt has unsupported type: '+record['path'])
        return 'not_public_evidence_type'
    if record['bytes'] > MAX_FILE:
        if critical or path.suffix.lower() in {'.py', '.log', '.stderr', '.stdout'}:
            raise ValueError('mandatory evidence exceeds 10 MiB: '+record['path'])
        return 'per_file_10MiB_limit'
    return None


def copy_exact(section: Path, output: Path, section_name: str, row: dict):
    source = section/row['path']
    target = output/section_name/row['path']
    if not stat.S_ISREG(source.lstat().st_mode) or identity(source) != {
            'bytes': row['bytes'], 'sha256': row['sha256']}:
        raise RuntimeError('source changed before copy: '+row['path'])
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as incoming, target.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1 << 20)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if identity(target) != {'bytes': row['bytes'], 'sha256': row['sha256']}:
        raise IOError('copied file differs: '+row['path'])


def write_new(path: Path, value: bytes):
    with path.open('xb') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    if identity(path) != {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}:
        raise IOError('generated file differs: '+str(path))


def pack(work: Path, output: Path):
    work = work.resolve(strict=True)
    output = output.absolute()
    for name in SECTION_NAMES:
        if (output/name).exists():
            raise FileExistsError(output/name)
    closure = verify_closure(work)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 'd1-c36-c38-publication-subset-v1', 'source_root': str(work),
               'output_root': str(output), 'closure': closure, 'sections': {}}
    for name in SECTION_NAMES:
        section = work/name
        local, special = inventory(section)
        included, skipped = [], list(special)
        for row in local:
            why = exclusion(row)
            (included if why is None else skipped).append(
                row if why is None else {**row, 'reason': why})
        names = {row['path'] for row in included}
        if not set(REQUIRED_PER_SECTION[name]).issubset(names):
            raise RuntimeError('mandatory evidence excluded in '+name)
        (output/name).mkdir(mode=0o755, exist_ok=False)
        for row in included:
            copy_exact(section, output, name, row)
        manifest['sections'][name] = {
            'copied_files': [{'path': row['path'], 'bytes': row['bytes'],
                              'sha256': row['sha256']} for row in included],
            'copied_bytes': sum(row['bytes'] for row in included),
            'skipped': sorted(skipped, key=lambda row: row['path'])}
    tool_path = Path(__file__).resolve(strict=True)
    tool_identity = identity(tool_path)
    manifest['publication_tool'] = {'source': str(tool_path),
                                    'destination': 'publication_tools/pack_publication36_38.py',
                                    **tool_identity}
    extra_target_dir = output/'publication_tools'
    extra_target_dir.mkdir(parents=True, exist_ok=True)
    manifest['extra_root_files'] = []
    for name in EXTRA_ROOT_FILES:
        source = RUN_SCRIPTS_SOURCE/name
        target = extra_target_dir/name
        with source.open('rb') as incoming, target.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing, 1 << 20)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        manifest['extra_root_files'].append({'source': str(source),
                                             'destination': 'publication_tools/'+name,
                                             **identity(target)})
    note = (b"# C36-C38 bounded two-support correction subset\n\n"
            b"Three consecutive single-attempt campaigns, none closing a pair exchange, "
            b"each stopping for a distinct and understood reason. C36 physically confirmed "
            b"the pre-unload stance-pair correction (support-pair normal sum 343 to 507 N "
            b"against a 236 N requirement across 485 native substeps) then hit a latent "
            b"MuJoCo 3.12 defect in the inherited fullM static-query check. C37 repaired "
            b"that defect and reached touchdown_dwell_timeout: the two returning wheels are "
            b"almost never simultaneously in contact (4 of 300 substeps), rocking about the "
            b"support diagonal with a 67.8 mm height differential consistent with the "
            b"measured 4.15 deg attitude excursion. C38 repaired a record-semantics defect "
            b"introduced by the correction itself and closed the independent audit "
            b"(ledger.passed true, 1495 of 1495 native substeps independently recomputed). "
            b"No pair exchange has completed. No training was run or prepared. Raw native, "
            b"controls and state payloads remain local; the three source GOs are signed by "
            b"root, not gpt-6-astra ultra, and say so explicitly.\n")
    manifest['generated_readme'] = {'destination': 'README_publication36_38.md',
                                    'bytes': len(note),
                                    'sha256': hashlib.sha256(note).hexdigest()}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
                      +'\n').encode()
    tool_target = output/'publication_tools/pack_publication36_38.py'
    tool_target.parent.mkdir(parents=True, exist_ok=True)
    with tool_path.open('rb') as incoming, tool_target.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1 << 20)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    write_new(output/'README_publication36_38.md', note)
    write_new(output/'publication_manifest36_38.json', manifest_bytes)
    return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = pack(args.source_root, args.output)
    total = sum(section['copied_bytes'] for section in result['sections'].values())
    print(json.dumps({'output': str(args.output), 'total_bytes': total,
                      'per_section': {name: len(section['copied_files'])
                                     for name, section in result['sections'].items()}}))


if __name__ == '__main__':
    main()
