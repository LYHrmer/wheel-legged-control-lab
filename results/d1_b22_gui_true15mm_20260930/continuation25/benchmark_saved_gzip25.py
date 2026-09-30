"""One saved-only replay of the C23 reset archive at gzip levels 9 and 1.

This script imports only the standard library. It reserializes the two frozen
episode-0 JSONL payloads through the same transaction stages as ArchiveWriter,
then compares decoded bytes. It creates no model, policy, GUI, or physics call.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

FILES = ('native_block_0000.jsonl.gz', 'controls.jsonl.gz')
MAX_RAW_BYTES = 256 * 1024 * 1024
MAX_ROWS = 10_000
SCHEMA = 'd1-c25-saved-gzip-benchmark-v1'
ARCHIVE_SCHEMA = 'd1-archive-transaction-13-v1'
FORBIDDEN = ('mujoco', 'engine_binding', 'torch', 'stable_baselines3')


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
            size += len(chunk)
    return {'bytes':size, 'sha256':digest.hexdigest()}


def fsync_directory(folder: Path) -> None:
    fd = os.open(folder, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def exclusive_json(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def frozen_rows(path: Path) -> tuple[list[dict], bytes, dict]:
    manifest_path = path.with_name(path.name + '.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema') != ARCHIVE_SCHEMA or
            manifest.get('payloads') != [{'file':path.name, **identity(path)}]):
        raise RuntimeError('frozen archive manifest identity differs: ' + str(path))
    with gzip.open(path, 'rb') as stream:
        raw = stream.read(MAX_RAW_BYTES + 1)
    if len(raw) > MAX_RAW_BYTES or not raw.endswith(b'\n'):
        raise RuntimeError('saved JSONL exceeded fixed bounded input or lacks last newline')
    lines = raw.splitlines(keepends=True)
    if not 1 <= len(lines) <= MAX_ROWS:
        raise RuntimeError('saved JSONL row count outside one finite episode')
    rows = [json.loads(line) for line in lines]
    rebuilt = b''.join((json.dumps(row, sort_keys=True, allow_nan=False,
        separators=(',', ':')) + '\n').encode() for row in rows)
    if rebuilt != raw:
        raise RuntimeError('frozen decoded rows are not ArchiveWriter canonical JSONL')
    return rows, raw, {
        'payload':str(path), 'payload_identity':identity(path),
        'manifest':str(manifest_path), 'manifest_identity':identity(manifest_path),
        'raw_bytes':len(raw), 'raw_sha256':hashlib.sha256(raw).hexdigest(),
        'rows':len(rows),
    }


def replay(rows: list[dict], raw: bytes, output: Path, stem: str,
           level: int) -> dict:
    name = f'{stem}_level{level}.jsonl.gz'
    final = output / name
    staged = output / ('.' + name + '.partial')
    manifest = output / (name + '.manifest.json')
    staged_manifest = output / ('.' + name + '.manifest.partial')
    start = time.perf_counter_ns()
    with staged.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0,
                           compresslevel=level) as compressed:
            for row in rows:
                compressed.write((json.dumps(row, sort_keys=True,
                    allow_nan=False, separators=(',', ':')) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    payload_id = identity(staged)
    os.link(staged, final)
    fsync_directory(output)
    archive_record = {'schema':ARCHIVE_SCHEMA,
        'payloads':[{'file':name, **payload_id}]}
    with staged_manifest.open('xb') as stream:
        stream.write((json.dumps(archive_record, sort_keys=True,
            allow_nan=False, separators=(',', ':')) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.link(staged_manifest, manifest)
    fsync_directory(output)
    staged.unlink()
    staged_manifest.unlink()
    elapsed_s = (time.perf_counter_ns()-start) / 1e9
    with gzip.open(final, 'rb') as stream:
        recovered = stream.read(MAX_RAW_BYTES + 1)
    if recovered != raw or identity(final) != payload_id:
        raise RuntimeError('replayed gzip transaction did not preserve every decoded byte')
    if json.loads(manifest.read_text()) != archive_record:
        raise RuntimeError('replayed transaction manifest differs')
    return {'level':level, 'elapsed_s':elapsed_s,
        'payload':name, 'payload_identity':payload_id,
        'manifest':manifest.name, 'manifest_identity':identity(manifest),
        'decoded_exact':True, 'decoded_sha256':hashlib.sha256(recovered).hexdigest()}


def reset_gap(run: Path) -> tuple[float, float, dict]:
    import json as _json
    with gzip.open(run/'owner_decisions.jsonl.gz', 'rt') as stream:
        decisions = [_json.loads(line) for line in stream]
    before = [row for row in decisions if row['global_control_index'] == 723
              and row['episode_index'] == 0 and row['reason'] == 'simulation_reset_request']
    after = [row for row in decisions if row['global_control_index'] == 723
             and row['episode_index'] == 1]
    if len(before) != 1 or len(after) != 1:
        raise RuntimeError('frozen reset boundary is not the one reviewed C23 event')
    performance = _json.loads((run/'performance.json').read_text())
    active = (performance['active_end_ns']-performance['active_start_ns']) / 1e9
    gap = (after[0]['now_ns']-before[0]['now_ns']) / 1e9
    if not 0 < gap < active:
        raise RuntimeError('saved active/reset clocks are inconsistent')
    inputs = {}
    for name in ('owner_decisions.jsonl.gz','performance.json'):
        path = run/name
        inputs[str(path)] = identity(path)
        manifest = path.with_name(path.name+'.manifest.json')
        inputs[str(manifest)] = identity(manifest)
    return gap, active, inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='new, non-existing benchmark artifact directory')
    args = parser.parse_args()
    if any(key in sys.modules for key in FORBIDDEN):
        raise RuntimeError('saved-only process imported a forbidden model/physics module')
    run = args.run.resolve(strict=True)
    if run.name != 'keyboard_gui_01':
        raise RuntimeError('benchmark is fixed to the one C23 keyboard run')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(str(output))
    output.mkdir(exist_ok=False)
    def expired(*_):
        raise TimeoutError('one pure saved benchmark exceeded 120 seconds')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 120)
    started = time.monotonic()
    try:
        gap, active, identities = reset_gap(run)
        result = {}
        for filename in FILES:
            saved = run/'episode_0'/filename
            rows, raw, source = frozen_rows(saved)
            identities[source['payload']] = source['payload_identity']
            identities[source['manifest']] = source['manifest_identity']
            stem = filename.removesuffix('.jsonl.gz')
            # Both candidates use exactly the same rows and transaction stages.
            candidate9 = replay(rows, raw, output, stem, 9)
            candidate1 = replay(rows, raw, output, stem, 1)
            result[stem] = {'source':source, 'level9':candidate9,
                            'level1':candidate1,
                            'saved_elapsed_s':candidate9['elapsed_s']-candidate1['elapsed_s']}
        saving = sum(item['saved_elapsed_s'] for item in result.values())
        threshold = 1005 * .01 / .8
        required = active-threshold
        receipt = {'schema':SCHEMA, 'run':str(run),
            'benchmark_source':str(Path(__file__).resolve()),
            'benchmark_source_identity':identity(Path(__file__)),
            'inputs':identities, 'reset_gap_s':gap, 'active_wall_s':active,
            'rtf_threshold':.8, 'controls':1005,
            'active_wall_limit_s':threshold,
            'saving_required_s':required,
            'transactions':result,
            'measured_level9_minus_level1_s':saving,
            'saving_exceeds_required':saving>=required,
            'estimated_active_s_if_transfer_exact':active-saving,
            'estimated_rtf_if_transfer_exact':1005*.01/(active-saving),
            'estimate_is_not_a_new_physical_qualification':True,
            'model_loads':0, 'physics_calls':0, 'servo_calls':0,
            'elapsed_s':time.monotonic()-started}
        if any(key in sys.modules for key in FORBIDDEN):
            raise RuntimeError('saved-only benchmark imported model/physics')
        exclusive_json(output/'benchmark_receipt.json', receipt)
        print(json.dumps({'saving_s':saving,'required_s':required,
            'exceeds_required':saving>=required,'receipt':str(output/'benchmark_receipt.json')}),
            flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
