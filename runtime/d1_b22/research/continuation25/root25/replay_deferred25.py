"""Full saved episode replay through the exact C25 writer; no physics."""
import gzip
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco', 'torch', 'stable_baselines3', 'gymnasium', 'engine_binding'}:
            raise RuntimeError('forbidden model/engine import: '+fullname)
        return None


def main():
    sys.meta_path.insert(0, NoEngine())
    sys.path.insert(0, str(ROOT/'gui25'))
    import numpy as np
    from test_deferred25_pure import _writer_type
    from archive13.atomic_archive_13 import verify_manifest
    from scripts import run_d1_latest_rl as common
    source = ROOT.parent/'continuation23/keyboard_gui_01/episode_0'
    output = ROOT/'deferred_replay_01'
    output.mkdir(exist_ok=False)
    kind = _writer_type(control_limit=1200)
    kind.commit_json.__globals__['common'] = common
    kind.commit_json.__globals__['identity'] = lambda p: {
        'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    writer = kind()
    native_path = source/'native_block_0000.jsonl.gz'
    control_path = source/'controls.jsonl.gz'
    native = [json.loads(line) for line in gzip.decompress(native_path.read_bytes()).splitlines()]
    controls = [json.loads(line) for line in gzip.decompress(control_path.read_bytes()).splitlines()]
    with np.load(source/'states.npz', allow_pickle=False) as stored:
        states = {key: [value.copy() for value in stored[key]] for key in stored.files}
    reset = json.loads((source/'reset.json').read_text())
    writer.begin_defer()
    start = time.monotonic()
    writer.commit_native_block(output/native_path.name, native)
    writer.commit_gzip_rows(output/control_path.name, controls)
    writer.commit_npz(output/'states.npz', states)
    writer.commit_json(output/'reset.json', reset)
    enqueue_s = time.monotonic()-start
    assert writer.committed == [] and len(writer.pending) == 4
    assert not any(output.iterdir())
    start = time.monotonic()
    writer.drain()
    flush_s = time.monotonic()-start
    exact = {}
    for path in (native_path, control_path):
        exact[path.name] = gzip.decompress(path.read_bytes()) == gzip.decompress((output/path.name).read_bytes())
    with np.load(source/'states.npz', allow_pickle=False) as old, np.load(output/'states.npz', allow_pickle=False) as new:
        exact['state_arrays'] = old.files == new.files and all(
            old[key].dtype == new[key].dtype and old[key].shape == new[key].shape
            and old[key].tobytes() == new[key].tobytes() for key in old.files)
    exact['reset_json'] = (source/'reset.json').read_bytes() == (output/'reset.json').read_bytes()
    manifests = all(verify_manifest(output/row['manifest_name']) for row in writer.queued_jobs)
    assert all(exact.values()) and manifests and not writer.pending and not writer.failed
    receipt = {'schema': 'd1-c25-full-saved-deferred-replay-v1', 'model_calls': 0, 'physics_calls': 0,
               'controls_replayed': len(controls), 'native_rows_replayed': len(native),
               'enqueue_s': enqueue_s, 'flush_s': flush_s, 'pending_after': len(writer.pending),
               'all_manifests_valid': manifests, 'byte_exact': exact,
               'writer_source_sha256': hashlib.sha256((ROOT/'gui25/run_gui25.py').read_bytes()).hexdigest(),
               'queued_jobs': writer.queued_jobs, 'flushed_jobs': writer.flushed_jobs,
               'is_physical_GUI_qualification': False}
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({key: value for key, value in receipt.items() if not key.endswith('_jobs')}, indent=2))


if __name__ == '__main__':
    main()
