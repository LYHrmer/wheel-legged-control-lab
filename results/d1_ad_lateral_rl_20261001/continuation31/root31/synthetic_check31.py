"""One signed synthetic-only invocation, with actual attempted/returned counters."""
from __future__ import annotations
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import signal
import sys
import time


def identity(path):
    data = Path(path).read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    contract_path, output = map(Path, sys.argv[1:])
    contract = json.loads(contract_path.read_text())
    if contract['decision'] != 'GO' or output.exists():
        raise RuntimeError('synthetic invocation not new/authorized')
    output.mkdir()
    for key, value in contract['environment'].items():
        if os.environ.get(key) != value:
            raise RuntimeError('synthetic environment differs: '+key)
    for path, expected in contract['source_identities'].items():
        if identity(path) != expected:
            raise RuntimeError('synthetic frozen source changed: '+path)
    blocked = []

    class NoRobot(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in ('mujoco', 'stable_baselines3', 'wheel_legged_control', 'engine_binding'):
                blocked.append(fullname)
                raise RuntimeError('synthetic test attempted forbidden robot import: '+fullname)
            return None

    sys.meta_path.insert(0, NoRobot())
    import torch
    import pytest
    import learning31 as learner
    import checkpoint31 as checkpoint
    limits = dict(construct=5, linear=10, optimizer=2, actor_calls=10, actor_rows=90,
                  value_calls=6, value_rows=65, backward=2, step=2, save=1, load=2,
                  probe=3, randn=1, randperm=3, update=2)
    counts = {}
    patches = []
    artifacts = []
    started = time.monotonic()

    def enter(key, amount=1):
        row = counts.setdefault(key, dict(attempted=0, returned=0))
        if row['attempted']+amount > limits[key]:
            raise RuntimeError('synthetic cap exceeded: '+key)
        row['attempted'] += amount

    def returned(key, amount=1):
        counts[key]['returned'] += amount

    def patch(owner, name, key, before=None, after=None):
        original = getattr(owner, name)
        def call(*args, **kwargs):
            enter(key)
            if before is not None:
                before(*args, **kwargs)
            value = original(*args, **kwargs)
            returned(key)
            if after is not None:
                after(value, *args, **kwargs)
            return value
        patches.append((owner, name, original))
        setattr(owner, name, call)

    def constructed(_, policy):
        for name in ('actor', 'value'):
            head = getattr(policy, name)
            original = head.forward
            def forward(x, *, name=name, original=original):
                n = int(x.shape[0])
                enter(name+'_calls')
                enter(name+'_rows', n)
                value = original(x)
                returned(name+'_calls')
                returned(name+'_rows', n)
                return value
            head.forward = forward

    def updated(receipt, policy, optimizer, batch, **kwargs):
        path = output/f'update_{len(artifacts):02d}.json'
        with path.open('x') as stream:
            json.dump(dict(synthetic=True, batch=batch, update=receipt), stream, allow_nan=False)
        artifacts.append(str(path))

    def probe_input(policy, rows):
        if len(rows) != 16 or any(len(row) != 54 for row in rows):
            raise RuntimeError('synthetic probe shape differs')

    def random_shape(n, **kwargs):
        if n != 3:
            raise RuntimeError('synthetic random action shape differs')

    def permutation_shape(n, **kwargs):
        if n != 8:
            raise RuntimeError('synthetic permutation shape differs')

    patch(torch.nn.Linear, '__init__', 'linear')
    patch(learner.LinearEventPolicy31, '__init__', 'construct', after=constructed)
    patch(torch.optim.Adam, '__init__', 'optimizer')
    patch(torch.optim.Adam, 'step', 'step')
    patch(torch.Tensor, 'backward', 'backward')
    patch(torch, 'save', 'save')
    patch(torch, 'load', 'load')
    patch(torch, 'randn', 'randn', before=random_shape)
    patch(torch, 'randperm', 'randperm', before=permutation_shape)
    patch(checkpoint, '_probe', 'probe', before=probe_input)
    patch(learner, 'update_batch31', 'update', after=updated)
    old_alarm = signal.signal(signal.SIGALRM,
        lambda *_: (_ for _ in ()).throw(TimeoutError('synthetic 60s deadline')))
    signal.setitimer(signal.ITIMER_REAL, contract['soft_wall_s'])
    code = None
    error = None
    args = ['-q', '-p', 'no:cacheprovider', '--basetemp', str(output/'pytest_tmp')]
    args.extend(contract['test_module']+'::'+node for node in contract['test_nodes'])
    try:
        code = pytest.main(args)
    except BaseException as exc:
        error = repr(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)
        changed = [path for path, expected in contract['source_identities'].items()
                   if identity(path) != expected]
        record = dict(schema='d1-c31-synthetic-execution-v1', synthetic_only=True,
            contract_identity=identity(contract_path), runner_identity=identity(__file__),
            source_identities=contract['source_identities'], source_mismatches=changed,
            counts=counts, limits=limits, blocked_robot_imports=blocked,
            robot_models=0, robot_controls=0, normal_native=0, compiler_native=0,
            artifacts=artifacts, pytest_argv=args, exit_code=code, error=error,
            elapsed_s=time.monotonic()-started,
            passed=code == 0 and error is None and not changed and not blocked)
        with (output/'receipt.json').open('x') as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        print(json.dumps(record), flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
