"""Stage one C31 research evidence subset without running a model or reader.

All paths below are existing saved files. This program copies bytes, validates
Python/JSON syntax, and inventories the full train/evaluation archives locally.
It never presents the published subset as a complete independent replay.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent.parent
C31 = ROOT / 'continuation31'
DEST_PARENT = ROOT / 'publication31'
MAX_FILE = 10 * 1024 * 1024
MAX_PACKAGE = 100 * 1024 * 1024
TEXT_SUFFIXES = {'.py', '.md', '.json', '.log'}
CHECKPOINT = C31 / 'train_01/final_checkpoint/lateral_final.pt'
MANDATORY_TRAIN = (
    'session.json', 'host_receipt.json', 'worker_receipt.json',
    'training_receipt31.json', 'final_checkpoint_receipt31.json',
    'final_checkpoint/manifest.json', 'final_checkpoint/probe_before.json',
    'final_checkpoint/probe_after.json',
)
MANDATORY_EPISODE = (
    'reset.json', 'segment_receipt.json', 'cycle_receipt.json',
    'macro_transitions31.json', 'task31.json',
    'states.npz', 'controls.jsonl.gz',
)
PUBLIC_NOTE = (
    '# C31 research evidence subset\n\n'
    'The `.pt` file is the C31 event-level lateral-skill checkpoint. It is '
    'separate from the frozen B22 rolling policy. This package is a source '
    'and selected saved-evidence snapshot, not a portable runnable bundle.\n\n'
    'The complete train/evaluation controls, native rows, state arrays and '
    'raw trajectories remain in the local archive. '
    '`publication_manifest31.json` records their paths, byte counts and SHA-256 '
    'hashes; the public subset cannot independently rerun the complete reader. '
    'Failed reader attempts and their receipts/logs are included as saved.\n'
)


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def checked_source(path: Path) -> Path:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError('missing or linked source: ' + str(path))
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(ROOT) or any(
            parent.is_symlink() for parent in path.parents if parent != ROOT):
        raise ValueError('source is linked or leaves the research tree: ' + str(path))
    return path


def relative(path: Path) -> str:
    checked_source(path)
    return str(path.relative_to(ROOT))


def validate_text(path: Path) -> None:
    if path.suffix == '.py':
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    elif path.suffix == '.json':
        json.loads(path.read_text(encoding='utf-8'))


def run_inventory(run: Path, label: str) -> list[dict]:
    if not run.is_dir() or run.is_symlink() or run.parent != C31:
        raise FileNotFoundError('missing or invalid C31 run directory: ' + str(run))
    entries = []
    for path in sorted(run.rglob('*')):
        if path.is_symlink():
            raise ValueError('linked item in saved run: ' + str(path))
        if path.is_file():
            entries.append({'run': label, 'path': relative(path), **identity(path)})
    return entries


def mandatory_files(train: Path, evaluation: Path, reports: tuple[Path, ...]) -> set[Path]:
    required = {checked_source(CHECKPOINT), *map(checked_source, reports)}
    required.update(checked_source(train / name) for name in MANDATORY_TRAIN)
    for run in (train, evaluation):
        required.update(checked_source(run / name) for name in
                        ('session.json', 'host_receipt.json', 'worker_receipt.json'))
        episodes = sorted(run.glob('episode_*'))
        if not episodes or any(not p.is_dir() or p.is_symlink() for p in episodes):
            raise FileNotFoundError('missing or linked episode in ' + str(run))
        expected_count = 64 if run == train else None
        if expected_count is not None and (len(episodes) != expected_count or
                {p.name for p in episodes} != {f'episode_{i}' for i in range(64)}):
            raise ValueError('training run lacks exactly 64 archived cycles')
        for episode in episodes:
            required.update(checked_source(episode / name) for name in MANDATORY_EPISODE)
            if not list(episode.glob('native_block_*.jsonl.gz')):
                raise FileNotFoundError('episode has no native block: ' + str(episode))
    return required


def choose_files(required: set[Path], evaluation: Path) -> tuple[dict[str, dict], list[dict]]:
    selected: dict[str, dict] = {}
    excluded: list[dict] = []
    for path in sorted(C31.rglob('*')):
        if path.is_symlink():
            fixture_root = C31 / 'synthetic_torch_01/pytest_tmp'
            target = path.resolve(strict=True)
            if (path.parent == fixture_root and path.name.endswith('current')
                    and target.parent == fixture_root and target.is_dir()
                    and not target.is_symlink()):
                link_text = os.readlink(path)
                excluded.append({'path': str(path.relative_to(ROOT)),
                    'reason': 'pytest_current_alias_only; real fixture directory selected normally',
                    'kind': 'symbolic_link_not_followed', 'link_target': link_text,
                    'bytes': len(link_text.encode()),
                    'sha256': hashlib.sha256(link_text.encode()).hexdigest()})
                continue
            raise ValueError('linked item in C31 tree: ' + str(path))
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        name = relative(path)
        info = identity(path)
        if info['bytes'] > MAX_FILE:
            if path in required and path.name != 'worker_receipt.json':
                raise ValueError('mandatory public evidence exceeds 10 MiB: ' + name)
            if any('readback' in part or 'reader' in part for part in path.parts):
                raise ValueError('reader failure/report exceeds 10 MiB: ' + name)
            excluded.append({'path': name, 'reason': 'per_file_10MiB_limit', **info})
            continue
        validate_text(path)
        selected[name] = info
    # The final event-policy checkpoint is deliberately included; no B22 model
    # or C30 snapshot is traversed into this C31-only staging tree.
    name = relative(CHECKPOINT)
    info = identity(CHECKPOINT)
    if info['bytes'] > MAX_FILE:
        raise ValueError('C31 lateral checkpoint exceeds 10 MiB')
    selected[name] = info
    for path in required:
        name = relative(path)
        if path.suffix in TEXT_SUFFIXES or path == CHECKPOINT:
            if name not in selected and path.name != 'worker_receipt.json':
                raise ValueError('required public file was omitted: ' + name)
    if not any(name.startswith(relative(evaluation / 'session.json')[:-len('session.json')])
               for name in selected):
        raise ValueError('evaluation run has no selected evidence')
    return selected, excluded


def copy_exact(name: str, destination: Path, expected: dict) -> dict:
    original = checked_source(ROOT / name)
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with original.open('rb') as incoming, target.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1 << 20)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    actual = identity(target)
    if actual != expected:
        raise IOError('copied bytes differ: ' + name)
    validate_text(target)
    return {'source': name, 'destination': name, **actual}


def stage(output: Path, training_report: Path, evaluation_run: Path,
          evaluation_report: Path, final_review: Path) -> None:
    if output.parent.resolve(strict=True) != DEST_PARENT.resolve(strict=True):
        raise ValueError('output must be a new direct child of W/publication31')
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    train = C31 / 'train_01'
    evaluation = evaluation_run.absolute()
    if evaluation.parent != C31 or evaluation == train:
        raise ValueError('evaluation run must be a distinct direct C31 child')
    reports = tuple(path.absolute() for path in
                    (training_report, evaluation_report, final_review))
    if len(set(reports)) != 3:
        raise ValueError('training/evaluation/final reports must be distinct saved files')
    required = mandatory_files(train, evaluation, reports)
    inventory = run_inventory(train, 'train_01') + run_inventory(evaluation, evaluation.name)
    selected, excluded = choose_files(required, evaluation)
    copied_bytes = sum(info['bytes'] for info in selected.values())
    receipt = {
        'schema': 'd1-c31-additive-publication-staging-v1',
        'source_root': str(ROOT), 'staging_root': str(output),
        'training_run': relative(train / 'session.json').removesuffix('/session.json'),
        'evaluation_run': relative(evaluation / 'session.json').removesuffix('/session.json'),
        'training_report': relative(reports[0]),
        'evaluation_report': relative(reports[1]),
        'final_review': relative(reports[2]),
        'lateral_event_checkpoint': relative(CHECKPOINT),
        'B22_checkpoint_included': False,
        'copied_files': [dict(source=name, destination=name, **selected[name])
                         for name in sorted(selected)],
        'copied_bytes': copied_bytes,
        'unpublished_items': excluded,
        'local_archive_inventory': inventory,
        'complete_reader_replay_from_public_subset': False,
        'portable_runtime_claim': False,
        'status': 'complete',
    }
    note_bytes = PUBLIC_NOTE.encode('utf-8')
    manifest_bytes = (json.dumps(receipt, sort_keys=True, indent=2,
                                 allow_nan=False) + '\n').encode('utf-8')
    if copied_bytes + len(note_bytes) + len(manifest_bytes) > MAX_PACKAGE:
        raise ValueError('C31 staging exceeds 100 MiB total cap')
    if len(manifest_bytes) > MAX_FILE:
        raise ValueError('publication inventory manifest exceeds 10 MiB')
    output.mkdir(mode=0o755)
    for name in sorted(selected):
        copy_exact(name, output, selected[name])
    for name, data in (('README_publication31.md', note_bytes),
                       ('publication_manifest31.json', manifest_bytes)):
        with (output / name).open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    if json.loads((output / 'publication_manifest31.json').read_text()) != receipt:
        raise IOError('publication manifest did not round-trip')
    if any(identity(output / name) != selected[name] for name in selected):
        raise IOError('post-copy publication file identity differs')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-report', type=Path, required=True)
    parser.add_argument('--evaluation-run', type=Path, required=True)
    parser.add_argument('--evaluation-report', type=Path, required=True)
    parser.add_argument('--final-review', type=Path, required=True)
    args = parser.parse_args()
    stage(args.output.absolute(), args.training_report, args.evaluation_run,
          args.evaluation_report, args.final_review)


if __name__ == '__main__':
    main()
