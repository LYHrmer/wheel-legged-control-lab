"""Replay one saved episode through the exact C25 deferred writer, with no model or physics.

Example: python replay_deferred25_saved.py --native EP/native_block_0000.jsonl.gz
    --controls EP/controls.jsonl.gz --states EP/states.npz --output NEW_DIRECTORY
"""

import argparse
import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

from archive13.atomic_archive_13 import ArchiveWriter, verify_manifest


def identity(path):
    data = Path(path).read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def writer_type(control_limit):
    source = Path(__file__).with_name('run_gui25.py')
    run = next(node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.FunctionDef) and node.name == 'run')
    writer = next(node for node in run.body
                  if isinstance(node, ast.ClassDef) and node.name == 'Writer')
    module = ast.fix_missing_locations(ast.Module(body=[writer], type_ignores=[]))
    namespace = {'ArchiveWriter': ArchiveWriter, 'Path': Path, 'threading': threading,
                 'time': time, 'copy': copy, 'json': json,
                 'common': SimpleNamespace(_jsonable=lambda value: value),
                 'session': {'control_limit': control_limit}, 'identity': identity}
    exec(compile(module, str(source), 'exec'), namespace)
    return namespace['Writer']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'controls', 'states', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    sources = {name: getattr(args, name).resolve(strict=True)
               for name in ('native', 'controls', 'states')}
    output = args.output.resolve()
    output.mkdir(parents=False, exist_ok=False)
    with gzip.open(sources['native'], 'rt') as stream:
        native = [json.loads(line) for line in stream]
    with gzip.open(sources['controls'], 'rt') as stream:
        controls = [json.loads(line) for line in stream]
    with np.load(sources['states'], allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if not controls or len(native) != 5 * len(controls):
        raise RuntimeError('saved native/control row count is incomplete')
    writer = writer_type(len(controls))()
    before = time.monotonic_ns()
    writer.begin_defer()
    writer.commit_native_block(output / sources['native'].name, native)
    writer.commit_gzip_rows(output / sources['controls'].name, controls)
    writer.commit_npz(output / sources['states'].name, arrays)
    queued = time.monotonic_ns()
    if writer.committed or any(output.iterdir()):
        raise RuntimeError('deferred enqueue published a file before drain')
    writer.drain()
    drained = time.monotonic_ns()
    if writer.failed or writer.pending or len(writer.flushed_jobs) != 3:
        raise RuntimeError('deferred replay did not close three archive jobs')
    for job in writer.queued_jobs:
        if not verify_manifest(output / job['manifest_name']):
            raise RuntimeError('archive manifest failed independent verification')
    for name in ('native', 'controls'):
        with gzip.open(sources[name], 'rb') as stream:
            original = stream.read()
        with gzip.open(output / sources[name].name, 'rb') as stream:
            replayed = stream.read()
        if replayed != original:
            raise RuntimeError(name + ' decoded payload differs from saved episode')
    with np.load(output / sources['states'].name, allow_pickle=False) as replayed:
        if set(replayed.files) != set(arrays):
            raise RuntimeError('states array keys differ')
        for name, original in arrays.items():
            value = replayed[name]
            if value.dtype != original.dtype or value.shape != original.shape or not np.array_equal(value, original):
                raise RuntimeError('state array differs: ' + name)
    report = {'schema': 'd1-c25-saved-deferred-replay-v1',
              'sources': {str(path): identity(path) for path in sources.values()},
              'writer_source': identity(Path(__file__).with_name('run_gui25.py')),
              'native_rows': len(native), 'control_rows': len(controls),
              'queued_jobs': len(writer.queued_jobs),
              'flushed_jobs': len(writer.flushed_jobs),
              'pending_count': len(writer.pending),
              'enqueue_elapsed_s': (queued - before) / 1e9,
              'drain_elapsed_s': (drained - queued) / 1e9,
              'decoded_exact': True, 'npz_arrays_exact': True,
              'model_calls': 0, 'physics_calls': 0}
    (output / 'receipt.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')


if __name__ == '__main__':
    main()
