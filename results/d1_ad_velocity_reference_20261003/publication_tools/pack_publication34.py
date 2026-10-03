"""Byte-check and publish a C34 research subset after final closure; no reader/model runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat


MAX_FILE = 10 * 1024 * 1024
PUBLIC_SUFFIXES = {'.py', '.json', '.md', '.txt', '.log', '.stderr', '.stdout'}
RAW_SUFFIXES = {'.npz', '.npy', '.gz', '.jsonl', '.pt', '.pth', '.ckpt',
                '.safetensors', '.onnx', '.bin', '.h5', '.hdf5', '.parquet', '.zip'}
REQUIRED = (
    'astra_plan/contract34.json', 'astra_plan/spec34.json',
    'astra_plan/shared_session34_v2.json',
    'source_go34.json', 'request34_01.json',
    'development_01/worker_receipt.json',
    'development_01/campaign_selection34.json',
    'development_01/case_definition34.json',
    'development_01/episode_0/cycle_receipt.json',
    'development_01/episode_0/task31.json',
    'development_01/episode_0/macro_transitions31.json',
    'readback_execution34_01/receipt.json',
    'readback_execution34_01/stdout.log',
    'readback_execution34_02/receipt.json',
    'readback_execution34_02/stdout.log',
    'independent_read34_02.json',
    'continuation_state34.json', 'final_review34.json',
)


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def require_closure(section: Path) -> dict:
    for name in REQUIRED:
        path = section/name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError('C34 publication requires closed evidence: '+str(path))
    pure_dirs = sorted(path for path in section.iterdir()
                       if path.is_dir() and path.name.startswith('pure_'))
    if len(pure_dirs) < 5:
        raise RuntimeError('C34 pure-check history is incomplete')
    for folder in pure_dirs:
        if not (folder/'receipt.json').is_file():
            raise RuntimeError('pure check lacks receipt: '+str(folder))
        if not any(path.is_file() and path.suffix in {'.log', '.stderr', '.stdout'}
                   for path in folder.rglob('*')):
            raise RuntimeError('pure check lacks stdout/stderr/log: '+str(folder))
    report = json.loads((section/'independent_read34_02.json').read_text())
    readback = json.loads((section/'readback_execution34_02/receipt.json').read_text())
    failed_readback = json.loads((section/'readback_execution34_01/receipt.json').read_text())
    worker = json.loads((section/'development_01/worker_receipt.json').read_text())
    campaign = json.loads((section/'development_01/campaign_selection34.json').read_text())
    state = json.loads((section/'continuation_state34.json').read_text())
    review = json.loads((section/'final_review34.json').read_text())
    case = campaign.get('all_attempted_cases', [])
    first = case[0] if len(case) == 1 else {}
    metrics = first.get('metrics', {})
    gates = metrics.get('gates', {})
    selected = campaign.get('selected_setting34', {})
    if (worker.get('execution_complete') is not True
            or worker.get('result', {}).get('closed_cases') != 1
            or first.get('case_id') != 'd030_beta050_left'
            or first.get('terminal_kind') != 'success'
            or metrics.get('safety_complete') is not True
            or metrics.get('eligible') is not False
            or gates.get('goal_error_vs_zero') is not False
            or gates.get('retention_vs_zero') is not False
            or campaign.get('stop_reason') != 'controlled_task_or_quality_failure'
            or selected.get('archived') is not True
            or selected.get('distance_m') != .03 or selected.get('beta') != 0.
            or campaign.get('cancel_conditions', {}).get('reused_archived_cancel') is not True):
        raise RuntimeError('C34 saved campaign differs from the closed negative result')
    if (not isinstance(failed_readback.get('exit_code'), int)
            or failed_readback['exit_code'] == 0
            or failed_readback.get('new_physics_steps') != 0
            or readback.get('exit_code') != 0
            or readback.get('new_physics_steps') != 0
            or readback.get('new_model_calls') != 0
            or readback.get('failure') is not None
            or readback.get('source_mismatches') != []
            or readback.get('source_precheck_complete') is not True
            or readback.get('source_postcheck_complete') is not True
            or readback.get('owned_child_reaped') is not True
            or readback.get('elapsed_s', float('inf')) > 1100
            or (readback.get('elapsed_s', float('inf'))
                + failed_readback.get('elapsed_s', float('inf'))) > 1200
            or readback.get('output_identity') != identity(section/'independent_read34_02.json')
            or report.get('record_valid') is not True
            or report.get('RL_speed_benefit_proven') is not False
            or report.get('user_goal_complete') is not False
            or report.get('independent_selection') != selected):
        raise RuntimeError('C34 independent saved readback is not closed and valid')
    if (state.get('independent_readback') != 'independent_read34_02.json'
            or state.get('goal_complete') is not False
            or review.get('reviewer_agent') != '/root/astra32_resume'
            or not isinstance(review.get('decision'), str)
            or not review['decision']):
        raise RuntimeError('C34 continuation state or Astra review is not final')
    reviewed = state.get('reviews', {}).get('final_review34.json')
    if reviewed is not None and reviewed != identity(section/'final_review34.json'):
        raise RuntimeError('C34 closed state points to another Astra review')
    return {'pure_check_directories': [path.name for path in pure_dirs],
            'independent_report_identity': identity(section/'independent_read34_02.json'),
            'final_review_identity': identity(section/'final_review34.json'),
            'closed_state_identity': identity(section/'continuation_state34.json')}


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
    return sorted(regular, key=lambda row: row['path']), sorted(
        special, key=lambda row: row['path'])


def exclusion(record: dict) -> str | None:
    path = Path(record['path'])
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    critical = ('receipt' in name or 'failure' in name or
                name.startswith(('final_review', 'independent_read')))
    if '__pycache__' in parts or name.endswith('.pyc'):
        return 'python_bytecode_cache'
    if path.suffix.lower() in RAW_SUFFIXES or name.endswith('.jsonl.gz'):
        if critical:
            raise ValueError('mandatory receipt is a raw replay/weight payload: '+record['path'])
        return 'raw_native_control_state_or_weight_payload'
    if path.suffix.lower() not in PUBLIC_SUFFIXES:
        if critical:
            raise ValueError('mandatory receipt has unsupported type: '+record['path'])
        return 'not_public_evidence_type'
    if record['bytes'] > MAX_FILE:
        if critical or path.suffix.lower() in {'.py', '.log', '.stderr', '.stdout'}:
            raise ValueError('mandatory public evidence exceeds 10 MiB: '+record['path'])
        return 'per_file_10MiB_limit'
    return None


def copy_exact(section: Path, output: Path, row: dict):
    source = section/row['path']
    target = output/'continuation34'/row['path']
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
    if len(value) > MAX_FILE:
        raise ValueError('generated publication file exceeds 10 MiB')
    with path.open('xb') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    if identity(path) != {'bytes': len(value),
                          'sha256': hashlib.sha256(value).hexdigest()}:
        raise IOError('generated publication file differs: '+str(path))


def pack(source_root: Path, output: Path):
    source_root = source_root.resolve(strict=True)
    section = source_root/'continuation34'
    if not section.is_dir() or section.is_symlink():
        raise FileNotFoundError(section)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if output.name != 'd1_ad_velocity_reference_20261003':
        raise ValueError('C34 publication output name differs from the reviewed target')
    if output.resolve().is_relative_to(section):
        raise ValueError('output must be outside local continuation34')
    closure = require_closure(section)
    local, special = inventory(section)
    included, skipped = [], list(special)
    for row in local:
        why = exclusion(row)
        (included if why is None else skipped).append(
            row if why is None else {**row, 'reason': why})
    names = {row['path'] for row in included}
    if not set(REQUIRED).issubset(names):
        raise RuntimeError('mandatory C34 public evidence was excluded')
    for folder in closure['pure_check_directories']:
        if f'{folder}/receipt.json' not in names:
            raise RuntimeError('pure-check receipt was excluded: '+folder)
    tool_path = Path(__file__).resolve(strict=True)
    tool_identity = identity(tool_path)
    manifest = {'schema': 'd1-c34-publication-subset-v1',
        'source_root': str(source_root), 'output_root': str(output),
        'included_section': 'continuation34', 'max_file_bytes': MAX_FILE,
        'closure': closure,
        'publication_tool': {'source': str(tool_path),
                             'destination': 'publication_tools/pack_publication34.py',
                             **tool_identity},
        'local_archive_inventory': local,
        'skipped': sorted(skipped, key=lambda row: row['path']),
        'copied_files': [{'source': 'continuation34/'+row['path'],
                          'destination': 'continuation34/'+row['path'],
                          'bytes': row['bytes'], 'sha256': row['sha256']}
                         for row in included],
        'copied_bytes': sum(row['bytes'] for row in included),
        'new_policy_checkpoints_included': False,
        'portable_runtime_claim': False,
        'complete_reader_replay_from_public_subset': False,
        'full_raw_archive_remains_local': True,
        'scope_note': 'Byte-exact C34 source, contracts, GO, all pure-check receipts/logs, physical and failed/successful readback receipts, final report and review. Raw native, controls, states and model payloads remain local.'}
    note = ("# C34 bounded tracking research subset\n\n"
            "The only physical C34 campaign safely closed one 30 mm left beta=0.5 "
            "case. It was faster than the archived 9.37 s fixed reference but "
            "failed the external goal-error and retention quality gates, so "
            "no qualified improvement or 40 mm result is claimed. This subset "
            "contains no new weights and is not portable. Full native, controls, "
            "states and model replay payloads remain local; the complete saved "
            "reader cannot run from this subset alone. See publication_manifest34.json "
            "for local hashes and every exclusion reason.\n").encode()
    manifest['generated_readme'] = {
        'destination': 'README_publication34.md', 'bytes': len(note),
        'sha256': hashlib.sha256(note).hexdigest()}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True,
                                 allow_nan=False)+'\n').encode()
    if len(manifest_bytes) > MAX_FILE:
        raise ValueError('publication manifest exceeds 10 MiB')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o755, exist_ok=False)
    for row in included:
        copy_exact(section, output, row)
    tool_target = output/'publication_tools/pack_publication34.py'
    tool_target.parent.mkdir(parents=True, exist_ok=True)
    with tool_path.open('rb') as incoming, tool_target.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1 << 20)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if identity(tool_target) != tool_identity:
        raise IOError('publication tool copied bytes differ')
    write_new(output/'README_publication34.md', note)
    write_new(output/'publication_manifest34.json', manifest_bytes)
    for row in local:
        path = section/row['path']
        if (not stat.S_ISREG(path.lstat().st_mode) or
                identity(path) != {'bytes': row['bytes'], 'sha256': row['sha256']}):
            raise RuntimeError('C34 local source changed during packaging: '+row['path'])
    if identity(tool_path) != tool_identity:
        raise RuntimeError('publication tool changed during packaging')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = pack(args.source_root, args.output)
    print(json.dumps({'output': str(args.output),
                      'copied_files': len(result['copied_files']),
                      'skipped': len(result['skipped'])}))


if __name__ == '__main__':
    main()
