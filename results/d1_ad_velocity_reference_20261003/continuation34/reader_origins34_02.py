"""Inventory one cold saved-reader import; never execute a reader or physics."""
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
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    return dict(bytes=size, sha256=digest.hexdigest())


def main():
    started = time.monotonic()
    destination = C / 'reader_origins34_02.json'
    if destination.exists():
        raise FileExistsError(destination)

    def timeout(*_):
        raise TimeoutError('C34 cold import inventory exceeded 20 seconds')

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 20)
    if any(key in os.environ for key in ('LD_PRELOAD', 'LD_LIBRARY_PATH')):
        raise RuntimeError('Cold reader inventory must have no engine preload')
    sys.path.insert(0, str(C / 'read34'))
    import read34  # noqa: F401 - main/read_run must never be called here
    from reader26 import FORBIDDEN, NoExecution
    assert not any(name.partition('.')[0] in FORBIDDEN for name in sys.modules)
    assert any(isinstance(finder, NoExecution) for finder in sys.meta_path)
    sources = {}
    roots = (W, R / 'runtime/d1_b22/research')
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if (name == '__main__' or path.suffix != '.py'
                or not any(path.is_relative_to(root) for root in roots)):
            continue
        sources[name] = dict(module=name, path=str(path), **identity(path))
    result = dict(
        schema='d1-c34-cold-reader-origin-inventory-v1',
        elapsed_s=time.monotonic() - started, deadline_s=20,
        diagnostic_source=identity(__file__), loaded_sources=sources,
        reader_run_calls=0, numerical_episode_reads=0, new_model_calls=0,
        new_physics_steps=0, fits=0, forbidden_import_guard_active=True,
    )
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(dict(path=str(destination), elapsed_s=result['elapsed_s'],
                         source_count=len(sources), **identity(destination))))


if __name__ == '__main__':
    main()
