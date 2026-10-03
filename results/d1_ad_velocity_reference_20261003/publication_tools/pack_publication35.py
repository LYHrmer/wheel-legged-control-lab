"""Byte-check and publish the C35 research subset after closure; no reader/model runs.

C35 is a rejected feasibility pilot: the first case failed a hard native contact gate.
This packer therefore asserts the *negative* closure shape - one attempted case, zero
closed cases, a retained native contact failure record and a clean independent readback
of that partial record - and refuses to publish anything that looks like a pass.

The output directory already holds the C34 subset, so every write here is
exclusive-create and only new C35 paths are produced.
"""
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
SECTION_NAME = 'continuation35'
REQUIRED = (
    'astra_plan/contract35.json', 'astra_plan/spec35.json',
    'astra_plan/interfaces35.md', 'astra_plan/shared_session35.json',
    'astra_plan/clarifications35_01.json', 'astra_plan/clarifications35_02.json',
    'astra_plan/clarifications35_03.json', 'astra_plan/clarifications35_04.json',
    'source_go35.json', 'request35_01.json',
    'development_01/worker_receipt.json', 'development_01/host_receipt.json',
    'development_01/supervisor_receipt.json', 'development_01/campaign35.json',
    'development_01/case35_00.json', 'development_01/case_definition35.json',
    'development_01/progress35_00.json', 'development_01/worker_stdout.log',
    'development_01/episode_0/native_contact_failure_0000.json',
    'development_01/episode_0/cycle_receipt35.json',
    'development_01/episode_0/segment_receipt.json',
    'readback_execution35_01/receipt.json', 'readback_execution35_01/stdout.log',
    'independent_read35_01.json',
    'root_adjudication35_01.json', 'continuation_state35.json',
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
            raise FileNotFoundError('C35 publication requires closed evidence: '+str(path))
    pure_dirs = sorted(path for path in section.iterdir()
                       if path.is_dir() and path.name.startswith('pure_'))
    if len(pure_dirs) < 10:
        raise RuntimeError('C35 pure-check history is incomplete')
    for folder in pure_dirs:
        if not (folder/'receipt.json').is_file():
            raise RuntimeError('pure check lacks receipt: '+str(folder))
        if not any(path.is_file() and path.suffix in {'.log', '.stderr', '.stdout'}
                   for path in folder.rglob('*')):
            raise RuntimeError('pure check lacks stdout/stderr/log: '+str(folder))
    go = json.loads((section/'source_go35.json').read_text())
    worker = json.loads((section/'development_01/worker_receipt.json').read_text())
    host = json.loads((section/'development_01/host_receipt.json').read_text())
    campaign = json.loads((section/'development_01/campaign35.json').read_text())
    case = json.loads((section/'development_01/case35_00.json').read_text())
    report = json.loads((section/'independent_read35_01.json').read_text())
    readback = json.loads((section/'readback_execution35_01/receipt.json').read_text())
    verdict = json.loads((section/'root_adjudication35_01.json').read_text())
    state = json.loads((section/'continuation_state35.json').read_text())

    if (go.get('decision') != 'GO' or go.get('reviewer_model') != 'gpt-6-astra'
            or go.get('reviewer_reasoning_effort') != 'ultra'
            or go.get('physical_attempts_authorized') != 1
            or go.get('training_authorized') is not False
            or go.get('automatic_retry') is not False):
        raise RuntimeError('C35 source GO is not the signed single-attempt review')
    if (worker.get('execution_complete') is not False
            or worker.get('retry_permitted') is not False
            or worker.get('archive_failed') is not False
            or worker.get('cleanup_errors') != []
            or host.get('source_mismatches') != []
            or host.get('owned_no_orphans') is not True
            or campaign.get('attempted_cases') != 1
            or campaign.get('closed_cases') != 0
            or campaign.get('qualification') is not False
            or campaign.get('stop_reason') != 'fatal_incomplete_or_unsafe'
            or campaign.get('controls') != 204
            or campaign.get('normal_native') != 1021
            or campaign.get('pair_exchanges') != 0
            or case.get('case_id') != 'pair_stand_zero'
            or case.get('terminal_kind') != 'fatal_incomplete_or_unsafe'
            or case.get('task') is not None
            or 'all4 actual support/load bound' not in case['failure']['message']):
        raise RuntimeError('C35 saved campaign differs from the closed rejected pilot')
    proofs = report.get('cases', [{}])[0].get('partial_native_proofs', [{}])[0]
    if (readback.get('exit_code') != 0 or readback.get('failure') is not None
            or readback.get('source_mismatches') != []
            or readback.get('source_precheck_complete') is not True
            or readback.get('source_postcheck_complete') is not True
            or readback.get('owned_child_reaped') is not True
            or readback.get('new_physics_steps') != 0
            or readback.get('new_model_calls') != 0
            or readback.get('elapsed_s', float('inf')) > 1200
            or readback.get('output_identity') != identity(section/'independent_read35_01.json')
            or report.get('record_auditable') is not True
            or report.get('partial_record') is not True
            or report.get('pilot_fixed_feasible') is not False
            or report.get('qualification') is not False
            or report.get('RL_speed_benefit_proven') is not False
            or report.get('user_goal_complete') is not False
            or report.get('reader_physics_calls') != 0
            or report.get('reader_model_calls') != 0
            or report.get('ledger', {}).get('passed') is not True
            or report.get('completed_controls') != 204
            or report.get('actual_normal_native') != 1021
            or proofs.get('saved_native_proof_passed') is not False):
        raise RuntimeError('C35 independent saved readback is not a closed valid partial audit')
    if (verdict.get('pilot_fixed_feasible') is not False
            or verdict.get('continuous_lateral_qualified') is not False
            or verdict.get('measured_lateral_speed_mps') is not None
            or verdict.get('engineering_speed_milestone_achieved') is not False
            or verdict.get('RL_speed_benefit_proven') is not False
            or verdict.get('new_ad_gui_delivered') is not False
            or verdict.get('user_goal_complete') is not False
            or verdict.get('astra_final_adjudication_obtained') is not False
            or verdict.get('source_go_identity') != identity(section/'source_go35.json')
            or state.get('goal_complete') is not False
            or state.get('retry_permitted') is not False
            or state.get('root_adjudication') != 'root_adjudication35_01.json'
            or state.get('independent_readback') != 'independent_read35_01.json'
            or state.get('next_action_requires_new_contract_and_go') is not True):
        raise RuntimeError('C35 root adjudication or closed state is not final and negative')
    return {'pure_check_directories': [path.name for path in pure_dirs],
            'independent_report_identity': identity(section/'independent_read35_01.json'),
            'root_adjudication_identity': identity(section/'root_adjudication35_01.json'),
            'source_go_identity': identity(section/'source_go35.json'),
            'closed_state_identity': identity(section/'continuation_state35.json')}


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
                name.startswith(('root_adjudication', 'independent_read')))
    if '__pycache__' in parts or name.endswith('.pyc'):
        return 'python_bytecode_cache'
    if '.ruff_cache' in parts:
        return 'lint_cache'
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
    target = output/SECTION_NAME/row['path']
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
    section = source_root/SECTION_NAME
    if not section.is_dir() or section.is_symlink():
        raise FileNotFoundError(section)
    output = output.absolute()
    if output.name != 'd1_ad_velocity_reference_20261003':
        raise ValueError('C35 publication output name differs from the reviewed target')
    if not output.is_dir() or output.is_symlink():
        raise FileNotFoundError('C35 publishes into the existing C34 result directory')
    if (output/SECTION_NAME).exists():
        raise FileExistsError(output/SECTION_NAME)
    if output.resolve().is_relative_to(section):
        raise ValueError('output must be outside the local continuation35')
    closure = require_closure(section)
    local, special = inventory(section)
    included, skipped = [], list(special)
    for row in local:
        why = exclusion(row)
        (included if why is None else skipped).append(
            row if why is None else {**row, 'reason': why})
    names = {row['path'] for row in included}
    if not set(REQUIRED).issubset(names):
        raise RuntimeError('mandatory C35 public evidence was excluded')
    for folder in closure['pure_check_directories']:
        if f'{folder}/receipt.json' not in names:
            raise RuntimeError('pure-check receipt was excluded: '+folder)
    tool_path = Path(__file__).resolve(strict=True)
    tool_identity = identity(tool_path)
    manifest = {'schema': 'd1-c35-publication-subset-v1',
        'source_root': str(source_root), 'output_root': str(output),
        'included_section': SECTION_NAME, 'max_file_bytes': MAX_FILE,
        'closure': closure,
        'publication_tool': {'source': str(tool_path),
                             'destination': 'publication_tools/pack_publication35.py',
                             **tool_identity},
        'local_archive_inventory': local,
        'skipped': sorted(skipped, key=lambda row: row['path']),
        'copied_files': [{'source': SECTION_NAME+'/'+row['path'],
                          'destination': SECTION_NAME+'/'+row['path'],
                          'bytes': row['bytes'], 'sha256': row['sha256']}
                         for row in included],
        'copied_bytes': sum(row['bytes'] for row in included),
        'new_policy_checkpoints_included': False,
        'portable_runtime_claim': False,
        'complete_reader_replay_from_public_subset': False,
        'full_raw_archive_remains_local': True,
        'result_is_a_pass': False,
        'scope_note': ('Byte-exact C35 contracts, four clarifications, interfaces, signed source '
                       'GO, immutable request, all pure-check receipts/logs, the single physical '
                       'campaign receipts including the retained native contact failure record, '
                       'the independent saved readback and the root adjudication. Raw native, '
                       'controls, states and model payloads remain local.')}
    note = ("# C35 bounded two-support side-step pilot subset (rejected)\n\n"
            "The single authorized C35 campaign stopped inside its first case. After 200 "
            "clean B22 preparation controls and four side controls, the fifth side control "
            "hit the frozen all-four contact gate: the diagonal swing wheel's measured "
            "normal load reached exactly 0 N and the wheel separated by 0.87 mm at 0.05 s "
            "of a mandatory 0.06 s transfer phase. No pair exchange completed, no lateral "
            "speed was measured, and the four remaining cases were never attempted. "
            "This subset contains no new weights and is not portable; the full saved reader "
            "cannot run from it alone. See publication_manifest35.json for local hashes and "
            "every exclusion reason, and continuation35/root_adjudication35_01.json for the "
            "evidence-based localization.\n").encode()
    manifest['generated_readme'] = {
        'destination': 'README_publication35.md', 'bytes': len(note),
        'sha256': hashlib.sha256(note).hexdigest()}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True,
                                 allow_nan=False)+'\n').encode()
    if len(manifest_bytes) > MAX_FILE:
        raise ValueError('publication manifest exceeds 10 MiB')
    (output/SECTION_NAME).mkdir(mode=0o755, exist_ok=False)
    for row in included:
        copy_exact(section, output, row)
    tool_target = output/'publication_tools/pack_publication35.py'
    tool_target.parent.mkdir(parents=True, exist_ok=True)
    with tool_path.open('rb') as incoming, tool_target.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1 << 20)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if identity(tool_target) != tool_identity:
        raise IOError('publication tool copied bytes differ')
    write_new(output/'README_publication35.md', note)
    write_new(output/'publication_manifest35.json', manifest_bytes)
    for row in local:
        path = section/row['path']
        if (not stat.S_ISREG(path.lstat().st_mode) or
                identity(path) != {'bytes': row['bytes'], 'sha256': row['sha256']}):
            raise RuntimeError('C35 local source changed during packaging: '+row['path'])
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
                      'copied_bytes': result['copied_bytes'],
                      'skipped': len(result['skipped'])}))


if __name__ == '__main__':
    main()
