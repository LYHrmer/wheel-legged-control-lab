"""Exclusive, bounded stage18 cold-worker reservations; no physics in this host."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from worker18 import identity, write

C = Path(__file__).resolve().parent
W = C.parent
R = Path('/home/lyh/wheel-legged-control-lab')


def run(planpath, arm):
    started = time.monotonic()
    def interrupted(*_):
        raise InterruptedError('host termination requested')
    signal.signal(signal.SIGTERM, interrupted)
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError('host could not become subreaper')
    plan = json.loads(planpath.read_text())
    if plan['status'] != 'GO' or plan['retry_permitted'] is not False:
        raise RuntimeError('stage18 GO/one-shot identity differs')
    spec = plan['arms'][arm]
    for name, row in plan['inputs'].items():
        if identity(name) != row:
            raise RuntimeError('GO source changed: '+name)
    frozen = {**plan['inputs'], str(planpath): identity(planpath)}
    old = json.loads((W/'rl11/training_run_01/session.json').read_text())
    manifest = json.loads((W/'continuation15/grouped_01/final_checkpoint_manifest.json').read_text())
    output = C/spec['output_directory']
    reserve = C/(spec['output_directory']+'_reservation.json')
    if output.exists() or reserve.exists():
        raise FileExistsError('stage18 worker already reserved; no in-place retry')
    command = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(C/'worker18.py'),
               '--session', str(output/'session.json')]
    write(reserve, {'arm': arm, 'controls': spec['controls'], 'normal_native': 5*spec['controls'],
                   'compiler_native': 2, 'plan_identity': identity(planpath), 'command': command,
                   'hard_s': spec['hard_s'], 'retry_permitted': False,
                   'reservation_not_refundable': True})
    output.mkdir(exist_ok=False)
    environment = dict(old['runtime_environment'])
    environment['PYTHONPATH'] += ':'+str(C)+':'+str(W/'continuation15')+':'+str(W/'continuation13')
    child_env = os.environ.copy()
    for k, v in environment.items():
        if v is None:
            child_env.pop(k, None)
        else:
            child_env[k] = v
    session = {**old, 'arm': arm, 'schema': 'd1-control-repair-session-18-v1',
        'controller_variant': spec['controller_variant'], 'floor_seed': spec['floor_seed'],
        'case_specs': spec['case_specs'],
        'source_hashes': frozen, 'runtime_environment': environment,
        'output_directory': str(output), 'argv': command[4:], 'worker_argv': command,
        'session_path': str(output/'session.json'), 'run_id': 'control18_'+arm,
        'wall_clock_utc': datetime.now(timezone.utc).isoformat(),
        'control_limit': spec['controls'], 'normal_native_limit': 5*spec['controls'],
        'compiler_native_limit': 2, 'soft_s': spec['soft_s'], 'close_s': spec['close_s'],
        'hard_s': spec['hard_s'], 'phase': 'stage18_'+arm,
        'seed': spec['floor_seed'], 'segments': [600]+[s['horizon'] for s in spec['case_specs'] for _ in s['actors']],
        'training_control_limit': 0, 'heldout_control_limit': spec['controls'],
        'policy_predict_control_limit': 600+sum(s['horizon'] for s in spec['case_specs'] if 'grouped_continue' in s['actors']),
        'wallclock_limit_s': spec['hard_s'], 'contract_path': str(C/plan['contract_filename']),
        'contract_sha256': frozen[str(C/plan['contract_filename'])]['sha256'],
        'go_sha256': frozen[str(C/plan['go_filename'])]['sha256'],
        'preregistered_scoring_path': str(C/plan['contract_filename']),
        'host_started_monotonic': started, 'retry_permitted': False,
        'plan_path': str(planpath), 'plan_identity': identity(planpath),
        'contract_documents': {str(C/plan['contract_filename']): frozen[str(C/plan['contract_filename'])]},
        'dependency_hashes': {name: row for name, row in old['dependency_hashes'].items() if name in frozen},
        'eval_manifests': {'grouped': manifest}}
    # Retain runtime provenance, never an obsolete training reservation.
    for key in ('budget_spec', 'budget_spec_sha256', 'short_plan_schema',
                'short_plan_source_sha256', 'ppo_seed', 'command_seed', 'measurement_seed'):
        session.pop(key, None)
    write(output/'session.json', session)
    process = None
    failure = None
    cleanup, changed, members = [], [], []
    post_complete = False
    try:
        with (output/'stdout.log').open('xb') as stream:
            process = subprocess.Popen(command, cwd=R, env=child_env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            write(output/'child.json', {'pid': process.pid, 'host_pid': os.getpid()})
            soft = False
            while process.poll() is None:
                elapsed = time.monotonic()-started
                if elapsed >= spec['hard_s']:
                    raise TimeoutError('independent host hard deadline')
                if elapsed >= spec['soft_s'] and not soft:
                    owner = json.loads((output/'worker_pid.json').read_text())
                    if owner['session_identity'] != identity(output/'session.json') or os.getpgid(owner['pid']) != process.pid:
                        raise RuntimeError('soft-stop worker identity differs')
                    os.kill(owner['pid'], signal.SIGTERM)
                    soft = True
                if elapsed >= spec['close_s']:
                    raise TimeoutError('worker close deadline')
                time.sleep(.1)
        if process.returncode != 0:
            raise RuntimeError('worker exited unsuccessfully')
        maps = json.loads((output/'mapped_libraries.json').read_text())
        if any(p not in frozen for p in maps['paths']):
            raise RuntimeError('unfrozen worker compute library')
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                end = time.monotonic()+1
                while time.monotonic() < end:
                    try:
                        pid, _ = os.waitpid(-1, os.WNOHANG)
                        if pid == 0:
                            time.sleep(.02)
                    except ChildProcessError:
                        break
            except BaseException as error:
                cleanup.append(repr(error))
        def timeout(*_):
            raise TimeoutError('source close deadline')
        previous = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, max(.01, started+spec['hard_s']-time.monotonic()))
        try:
            changed = [name for name, row in frozen.items() if identity(name) != row]
            post_complete = True
        except BaseException as error:
            cleanup.append(repr(error))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        if process is not None:
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():
                    continue
                try:
                    fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
                    if fields[0] != 'Z' and int(fields[2]) == process.pid:
                        members.append(int(p.name))
                except (FileNotFoundError, ProcessLookupError):
                    pass
            if members:
                os.killpg(process.pid, signal.SIGKILL)
                cleanup.append('owned live descendants required kill: '+repr(members))
        if time.monotonic()-started > spec['hard_s']:
            cleanup.append('host hard wall exceeded')
        receipt = {'arm': arm, 'failure': failure, 'cleanup_errors': cleanup,
            'exit_code': None if process is None else process.returncode,
            'changed_sources': changed, 'postcheck_complete': post_complete,
            'no_live_owned_processes': not members, 'elapsed_s': time.monotonic()-started,
            'fully_reserved_budget_closed': True, 'retry_permitted': False}
        write(output/'host_receipt.json', receipt)
        print(json.dumps(receipt), flush=True)
    return 0 if failure is None and not cleanup and not changed and post_complete else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--arm')
    args = parser.parse_args()
    if args.plan and args.arm:
        raise SystemExit(run(args.plan.resolve(strict=True), args.arm))
    else:
        parser.error('choose --plan PLAN --arm ARM; source freezing uses freeze18.py')
