"""One bounded, owner-requested v27 GUI closeout. Default --plan is zero physics."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('b22_closeout_readback30', ROOT/'scripts/verify_b22_closeout30.py')
readback = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(readback)

START_BYTES, RESERVE_BYTES = 6*1024**3, 3*1024**3
TOTAL_WALL_S, LAUNCHER_WALL_S, READBACK_WALL_S = 900, 870, 20
AUTHORED = ('scripts/run_b22_closeout30.py', 'scripts/verify_b22_closeout30.py',
            'tests/test_b22_closeout30.py', 'docs/b22_closeout30_20261018.md')


def plan(output):
    output = Path(output).resolve()
    return {'schema': 'b22-v27-closeout-plan30-v1', 'owner_instruction': '完成收尾工作',
            'scope': 'one rendered flat_1p6/B/seed181202 session; no retries, new tasks, training, or new qualification',
            'output': str(output), 'argv': ['rtk', 'proxy', '/usr/bin/python3', '-B',
                str(ROOT/'scripts/play_b22_v27.py'), '--mode', 'script', '--profile', 'flat_1p6',
                '--actor', 'B', '--output', str(output)],
            'limits': {'controls': 1600, 'normal_native': 8000, 'compiler_native': 2,
                       'model_load': 1, 'torch_load': 3, 'predict': 1601,
                       'total_wall_s': TOTAL_WALL_S, 'launcher_wall_s': LAUNCHER_WALL_S,
                       'readback_wall_s': READBACK_WALL_S, 'poll_s': 1.,
                       'start_effective_bytes': START_BYTES, 'reserve_effective_bytes': RESERVE_BYTES},
            'qualification_granted': False, 'retry_permitted': False}


def source_identity():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True, timeout=5).strip()
    commit = git('rev-parse', 'HEAD')
    tracked = set(git('ls-files', '--', *AUTHORED).splitlines())
    readback.require(set(AUTHORED) <= tracked and not git('diff', '--name-only', 'HEAD', '--', *AUTHORED),
                     'commit the four exact closeout sources before execution')
    config = readback.manifest()
    for name in ('scripts/play_b22_v27.py', 'scripts/b22_runtime_v27.py', config['host'], config['worker']):
        path = Path(name) if Path(name).is_absolute() else ROOT/name
        readback.require(readback.identity(path) == config['source_hashes'][str(path)], 'v27 execution source changed')
    return {'git_commit': commit, 'authored_sha256': {name: readback.identity(ROOT/name) for name in AUTHORED},
            'v27_manifest_identity': readback.identity(ROOT/readback.MANIFEST)}


def read_memory(proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup')):
    fields = dict(line.split(':', 1) for line in (proc_root/'meminfo').read_text().splitlines())
    available = int(fields['MemAvailable'].split()[0])*1024
    scopes = (proc_root/'self/cgroup').read_text().splitlines()
    scope = next((line[3:] for line in scopes if line.startswith('0::')), None)
    readback.require(available >= 0 and scope is not None and '..' not in Path(scope).parts, 'unknown host/cgroup RAM accounting')
    current, remaining = cgroup_root/scope.lstrip('/'), []
    readback.require(current.is_dir(), 'current cgroup-v2 is not visible')
    while True:
        maximum = current/'memory.max'
        if maximum.exists() and maximum.read_text().strip() != 'max':
            limit, used = int(maximum.read_text()), int((current/'memory.current').read_text())
            readback.require(limit >= 0 and used >= 0, 'invalid cgroup RAM sample')
            remaining.append(max(0, limit-used))
        if current == cgroup_root:
            break
        current = current.parent
    cap = min(remaining) if remaining else None
    return {'host_available_bytes': available, 'cgroup_remaining_bytes': cap,
            'effective_available_bytes': min(available, cap) if cap is not None else available}


def owned_host():
    config = readback.manifest()
    path = Path(config['host'])
    return readback.pinned_module('closeout_owned_host23', path, config['source_hashes'][str(path)])


def run(output, *, memory_reader=read_memory, process_factory=subprocess.Popen, host_loader=owned_host,
        verifier=readback.verify, provenance=source_identity, clock=time.monotonic, sleeper=time.sleep):
    started = clock()
    output = Path(output).resolve()
    readback.require(output == readback.OUTPUT, 'this closeout authorizes only its one fixed output')
    sidecar = output.with_name(output.name+'_closeout')
    readback.require(not output.exists() and not sidecar.exists()
                     and not output.with_name(output.name+'_launch').exists()
                     and not output.with_name(output.name+'_reservation.json').exists(), 'one-shot output/reservation exists; no retry')
    source = provenance()
    sidecar.mkdir(parents=True, exist_ok=False)
    request = {**plan(output), 'source': source}
    readback.write_json(sidecar/'request.json', request)
    report = {'schema': 'b22-v27-closeout-supervision30-v1', 'status': 'FAIL', 'failure': None,
              'supervisor_pid': os.getpid(), 'launcher_pid': None,
              'launcher_spawned': False, 'returncode': None, 'cleanup': None, 'readback': None,
              'minimum_effective_bytes': None, 'elapsed_s': None, 'retry_permitted': False,
              'qualification_granted': False, 'source': source}
    process = host = None
    previous = {}
    def stop(signum, _frame):
        raise InterruptedError('closeout interrupted: '+signal.Signals(signum).name)
    def sample(phase):
        item = dict(memory_reader(), elapsed_s=clock()-started, phase=phase)
        value = item.get('effective_available_bytes')
        readback.require(type(value) is int and value >= 0, 'unknown effective RAM')
        report['minimum_effective_bytes'] = min(value, report['minimum_effective_bytes']) if report['minimum_effective_bytes'] is not None else value
        with (sidecar/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(item, allow_nan=False)+'\n')
        return value
    try:
        readback.require(sample('admission') >= START_BYTES, 'NOT_READY: effective RAM below 6 GiB')
        host = host_loader()
        host.subreaper()
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, stop)
        readback.require(clock()-started < LAUNCHER_WALL_S, 'wall limit before launcher')
        with (sidecar/'launcher_stdout.log').open('xb') as stream:
            process = process_factory(request['argv'], cwd=ROOT, stdin=subprocess.DEVNULL,
                                      stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                      env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            report['launcher_spawned'] = True
            report['launcher_pid'] = getattr(process, 'pid', None)
            while process.poll() is None:
                readback.require(clock()-started < LAUNCHER_WALL_S, 'outer launcher wall limit')
                readback.require(sample('running') >= RESERVE_BYTES, 'RAM reserve below 3 GiB')
                sleeper(min(1., max(0., LAUNCHER_WALL_S-(clock()-started))))
            report['returncode'] = process.returncode
        readback.require(sample('after_exit') >= RESERVE_BYTES, 'RAM reserve below 3 GiB after exit')
        readback.require(process.returncode == 0, 'v27 launcher failed')
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        if host is not None:
            try:
                report['cleanup'] = host.clean_descendants(os.getpid(), process)
                if report['cleanup'].get('remaining') != {}:
                    report['failure'] = report['failure'] or 'owned descendants remain'
            except (OSError, ValueError, RuntimeError) as exc:
                report['failure'] = report['failure'] or f'cleanup {type(exc).__name__}: {exc}'
        for sig in previous:
            signal.signal(sig, stop)
    if report['failure'] is None:
        remaining = min(READBACK_WALL_S, TOTAL_WALL_S-5-(clock()-started))
        def expired(*_):
            raise TimeoutError('independent readback wall limit')
        old = signal.signal(signal.SIGALRM, expired)
        try:
            readback.require(remaining > 0, 'no readback time remains')
            signal.setitimer(signal.ITIMER_REAL, remaining)
            report['readback'] = verifier(output)
            readback.require(report['readback']['status'] == 'PASS', 'independent readback failed')
            readback.require(provenance() == source, 'closeout sources changed during execution')
            readback.require(clock()-started < TOTAL_WALL_S-5, 'total closeout wall limit')
            report['status'] = 'PASS'
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            report['failure'] = f'{type(exc).__name__}: {exc}'
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    report['elapsed_s'] = clock()-started
    for sig, handler in previous.items():
        signal.signal(sig, handler)
    readback.write_json(sidecar/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=readback.OUTPUT)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({**plan(args.output), 'execution_started': False, 'model_loads': 0, 'physics_steps': 0}, indent=2))
        return 0
    report = run(args.output)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
