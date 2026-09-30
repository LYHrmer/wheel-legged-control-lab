"""Pure archive checks for the exact nested C25 Writer source; no model or engine."""

import ast
import copy
import gzip
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from archive13.atomic_archive_13 import ArchiveWriter, verify_manifest


def _writer_type(control_limit=3):
    source = Path(__file__).with_name('run_gui25.py')
    tree = ast.parse(source.read_text())
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == 'run')
    writer = next(node for node in run.body if isinstance(node, ast.ClassDef)
                  and node.name == 'Writer')
    module = ast.fix_missing_locations(ast.Module(body=[writer], type_ignores=[]))
    namespace = {
        'ArchiveWriter': ArchiveWriter, 'Path': Path, 'threading': threading,
        'time': time, 'copy': copy, 'json': json,
        'common': SimpleNamespace(_jsonable=lambda value: value),
        'session': {'control_limit': control_limit},
        'identity': lambda path: {'bytes': path.stat().st_size},
    }
    exec(compile(module, str(source), 'exec'), namespace)
    return namespace['Writer']


def test_pending_is_not_durable_and_owned_rows_flush_exactly(tmp_path):
    writer = _writer_type()()
    writer.begin_defer()
    native = [{'native_index': 0}]
    controls = [{'control_index': 0}]
    receipt = {'prepared': {'executed': False}}
    writer.commit_native_block(tmp_path / 'native_block_0000.jsonl.gz', native)
    writer.commit_gzip_rows(tmp_path / 'controls.jsonl.gz', controls)
    writer.commit_json(tmp_path / 'reset.json', receipt)
    # The producer transfers the old list and starts a new one; mutable JSON
    # receipts are copied by Writer at enqueue time.
    native = []
    controls = []
    receipt['prepared']['executed'] = True
    native.append({'native_index': 99})
    controls.append({'control_index': 99})
    assert writer.peak_jobs == 3
    assert writer.committed == []
    assert not (tmp_path / 'reset.json').exists()
    assert not (tmp_path / 'reset.json.manifest.json').exists()
    writer.drain()
    assert not writer.pending and not writer.failed
    assert [job['id'] for job in writer.flushed_jobs] == [0, 1, 2]
    for job in writer.queued_jobs:
        assert verify_manifest(tmp_path / job['manifest_name'])
    with gzip.open(tmp_path / 'native_block_0000.jsonl.gz', 'rt') as stream:
        assert [json.loads(line) for line in stream] == [{'native_index': 0}]
    with gzip.open(tmp_path / 'controls.jsonl.gz', 'rt') as stream:
        assert [json.loads(line) for line in stream] == [{'control_index': 0}]
    assert json.loads((tmp_path / 'reset.json').read_text()) == {
        'prepared': {'executed': False}}


def test_bounds_and_drain_failure_remain_fail_closed(tmp_path):
    writer = _writer_type(control_limit=1)()
    writer.begin_defer()
    writer.commit_gzip_rows(tmp_path / 'controls.jsonl.gz', [{'control_index': 0}])
    writer._stage = lambda *_args: (_ for _ in ()).throw(OSError('injected stage failure'))
    with pytest.raises(OSError, match='injected stage failure'):
        writer.drain()
    assert writer.failed and len(writer.pending) == 1
    assert not (tmp_path / 'controls.jsonl.gz.manifest.json').exists()

    second = _writer_type(control_limit=1)()
    second.begin_defer()
    with pytest.raises(RuntimeError, match='control row bound'):
        second.commit_gzip_rows(tmp_path / 'controls.jsonl.gz', [{}, {}])
    assert second.failed and not second.pending
