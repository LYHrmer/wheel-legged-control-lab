"""One cold import inventory for the saved-only reader, without running it."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

C = Path(__file__).resolve().parent
W = C.parent
R = Path('/home/lyh/wheel-legged-control-lab')


def identity(path):
    raw = Path(path).read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    start = time.monotonic()
    def timeout(*_):
        raise TimeoutError('cold origin inventory exceeded 20 seconds')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 20)
    forbidden = ('LD_PRELOAD', 'LD_LIBRARY_PATH')
    if any(key in os.environ for key in forbidden):
        raise RuntimeError('origin inventory must have no engine preload')
    sys.path.insert(0, str(C/'read33'))
    import read33  # noqa: F401 - imports only; main/read_run33 never called
    from reader26 import FORBIDDEN, NoExecution
    assert not any(name.partition('.')[0] in FORBIDDEN for name in sys.modules)
    assert any(isinstance(finder, NoExecution) for finder in sys.meta_path)
    session = json.loads((C/'development_02/session.json').read_text())
    review = json.loads((C/'reader_source_go33.json').read_text())
    wanted = {**review['inputs'], **session['source_hashes']}
    sources, unsealed = {}, []
    roots = (W, R/'runtime/d1_b22/research')
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if name == '__main__' or path.suffix != '.py' or not any(
                path.is_relative_to(root) for root in roots):
            continue
        actual = identity(path)
        row = dict(module=name, path=str(path), **actual)
        sources[name] = row
        if wanted.get(str(path)) != actual:
            equivalents = [p for p, value in wanted.items() if value == actual]
            unsealed.append(dict(**row, byte_identical_sealed_paths=equivalents))
    result = dict(schema='d1-c33-cold-reader-origin-inventory-v1',
        elapsed_s=time.monotonic()-start, deadline_s=20,
        diagnostic_source=identity(__file__), loaded_sources=sources,
        unsealed=unsealed, reader_run_calls=0, numerical_episode_reads=0,
        new_model_calls=0, new_physics_steps=0, fits=0,
        forbidden_import_guard_active=True)
    with (C/'reader_origins33_01.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(elapsed_s=result['elapsed_s'], unsealed=unsealed)))
    signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
