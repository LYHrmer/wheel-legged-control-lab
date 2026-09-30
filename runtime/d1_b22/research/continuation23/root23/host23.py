"""Root launcher for one frozen GUI23 or step24 reservation; never imports physics."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def identity(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return {'bytes': Path(path).stat().st_size, 'sha256': digest.hexdigest()}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def mismatches(sources):
    return [name for name, expected in sources.items()
            if not Path(name).is_file() or identity(name) != expected]


def members(group):
    found = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():
            continue
        try:
            data = (folder/'stat').read_text().rsplit(')', 1)[1].split()
            if data[0] != 'Z' and int(data[2]) == group:
                found.append(int(folder.name))
        except (OSError, ValueError):
            pass
    return found


def subreaper():
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0) != 0:
        raise OSError(ctypes.get_errno(),'cannot become child subreaper')


def descendants(root):
    """Identity includes Linux process birth tick; includes adopted setsid children."""
    rows={}
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():
            continue
        try:
            fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
            rows[int(folder.name)]=(int(fields[1]),int(fields[19]),fields[0])
        except (OSError,ValueError,IndexError):
            pass
    family={root}
    while True:
        added={pid for pid,(ppid,_,_) in rows.items() if ppid in family}-family
        if not added:
            break
        family.update(added)
    return {pid:rows[pid][1] for pid in family-{root} if rows[pid][2]!='Z'}


def signal_identity(pid,birth,number):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if int(fields[19])==birth:
            os.kill(pid,number)
    except (OSError,ValueError,IndexError):
        pass


def clean_descendants(root,process):
    start=time.monotonic()
    signalled={}
    while time.monotonic()-start<4.9:
        alive=descendants(root)
        if not alive:
            break
        number=signal.SIGTERM if time.monotonic()-start<2.0 else signal.SIGKILL
        for pid,birth in alive.items():
            signal_identity(pid,birth,number)
            signalled[pid]=birth
        time.sleep(.02)
    if process is not None:
        process.poll()
    while True:
        try:
            pid,_=os.waitpid(-1,os.WNOHANG)
            if pid==0:
                break
        except ChildProcessError:
            break
    return {'method':'linux_subreaper_ancestry_birth_tick',
            'signalled_identities':signalled,'remaining':descendants(root),
            'elapsed_s':time.monotonic()-start}


def execute(request_path):
    subreaper()
    def expired(*_):
        raise TimeoutError('source preflight exceeded 240 seconds')
    old_alarm = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL,240)
    request = json.loads(request_path.read_text())
    go_path = Path(request['source_go_path'])
    go = json.loads(go_path.read_text())
    if go.get('decision') != 'GO' or request['arm'] not in go['arms']:
        raise RuntimeError('missing reviewed source GO for arm')
    if go['arms'][request['arm']] != request['spec']:
        raise RuntimeError('reservation differs from source GO')
    if request.get('x11_events') != go.get('x11_events_by_arm',{}).get(request['arm']):
        raise RuntimeError('X11 event table differs from source GO')
    sources = dict(go['inputs'])
    bad = mismatches(sources)
    if bad:
        raise RuntimeError('source changed before reservation: '+repr(bad))
    signal.setitimer(signal.ITIMER_REAL,0)
    signal.signal(signal.SIGALRM,old_alarm)
    sources[str(go_path)] = identity(go_path)
    sources[str(request_path)] = identity(request_path)
    spec = dict(request['spec'])
    output = Path(spec['output_directory'])
    reservation = output.with_name(output.name+'_reservation.json')
    if output.exists() or reservation.exists():
        raise FileExistsError('arm was reserved already; no retry')
    write(reservation, {'schema':'d1-c23-c24-reservation-v1', 'arm':request['arm'],
          'controls':spec['control_limit'], 'normal_native':spec['control_limit']*5,
          'compiler_native':2, 'model_limits':spec['model_limits'],
          'source_go_identity':identity(go_path), 'retry_permitted':False})
    output.mkdir(exist_ok=False)
    env = os.environ.copy()
    runtime_env = dict(go['runtime_environment'])
    if spec.get('render'):
        runtime_env.update(DISPLAY=env.get('DISPLAY'), XAUTHORITY=env.get('XAUTHORITY'))
    for key, value in runtime_env.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    command = ['rtk','proxy','/usr/bin/python3','-B',go['worker'],
               '--session',str(output/'session.json'),'--output',str(output)]
    started = time.monotonic()
    session = {**go['shared_session'], **spec,
        'arm':request['arm'], 'source_hashes':sources,
        'runtime_environment':runtime_env, 'host_started_monotonic':started,
        'argv':command[4:], 'retry_permitted':False,
        'source_go_path':str(go_path), 'source_go_identity':identity(go_path)}
    write(output/'session.json',session)
    process = None
    failure = None
    event_driver = None
    execution_started = None
    def termination(*_):
        raise TimeoutError('host termination requested')
    previous_term=signal.signal(signal.SIGTERM,termination)
    try:
        with (output/'worker_stdout.log').open('xb') as stream:
            process = subprocess.Popen(command, cwd=go['repository'], env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True)
            write(output/'host_child.json',{'pid':process.pid,'argv':command})
            if request.get('x11_events'):
                from x11_events23 import EventDriver
                event_driver = EventDriver(output, request['x11_events'])
            soft_sent = False
            while process.poll() is None:
                elapsed = time.monotonic()-started
                ready = output/'preflight_ready.json'
                if execution_started is None and ready.exists():
                    try:
                        record = json.loads(ready.read_text())
                    except json.JSONDecodeError:
                        record = None
                    if record is not None:
                        if (record.get('pid') not in descendants(os.getpid())
                                or record.get('session_sha256') != identity(output/'session.json')['sha256']):
                            raise RuntimeError('readiness worker/session identity mismatch')
                        execution_started = float(record['monotonic_s'])
                        if not started <= execution_started <= time.monotonic():
                            raise RuntimeError('invalid readiness monotonic time')
                if event_driver is not None:
                    event_driver.advance()
                if execution_started is None and elapsed >= 240:
                    raise TimeoutError('worker source preflight exceeded 240 seconds')
                active_elapsed = 0 if execution_started is None else time.monotonic()-execution_started
                if active_elapsed >= spec['hard_s']:
                    raise TimeoutError('owned worker hard deadline')
                if active_elapsed >= spec['soft_s'] and not soft_sent:
                    os.killpg(process.pid, signal.SIGTERM)
                    soft_sent = True
                time.sleep(.01)
            if process.returncode:
                failure = {'type':'WorkerExit','code':process.returncode}
    except BaseException as error:
        failure = {'type':type(error).__name__,'message':str(error)}
    finally:
        cleanup=clean_descendants(os.getpid(),process)
        if event_driver is not None:
            try:
                event_driver.close()
            except BaseException as error:
                if failure is None:
                    failure={'type':type(error).__name__,'message':'X11 close: '+str(error)}
        def post_expired(*_):
            raise TimeoutError('source postcheck exceeded 180 seconds')
        previous = signal.signal(signal.SIGALRM,post_expired)
        signal.setitimer(signal.ITIMER_REAL,180)
        try:
            bad = mismatches(sources)
        except BaseException as error:
            bad = ['postcheck_failed: '+str(error)]
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,previous)
        receipt = {'schema':'d1-c23-c24-host-v1', 'arm':request['arm'],
            'failure':failure, 'exit_code':None if process is None else process.returncode,
            'source_mismatches':bad, 'owned_no_orphans':not cleanup['remaining'],
            'owned_cleanup':cleanup,
            'elapsed_s':time.monotonic()-started, 'reserved_controls':spec['control_limit'],
            'execution_started_monotonic':execution_started,
            'retry_permitted':False, 'reservation_closed':True}
        write(output/'host_receipt.json',receipt)
        print(json.dumps(receipt),flush=True)
        signal.signal(signal.SIGTERM,previous_term)
    return 0 if failure is None and not bad and receipt['owned_no_orphans'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--isolated-x11',type=Path)
    parser.add_argument('--owned-worker-host',action='store_true')
    parser.add_argument('--isolation-child',action='store_true')
    args = parser.parse_args()
    if args.isolated_x11 and args.isolation_child:
        import importlib.util
        spec = importlib.util.spec_from_file_location('isolated_x11',args.isolated_x11)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        request = json.loads(args.request.read_text())
        return module.run(['rtk','proxy','/usr/bin/python3','-B',str(Path(__file__).resolve()),
            '--request',str(args.request.resolve()),'--owned-worker-host'],
            Path(request['spec']['output_directory']))
    if args.owned_worker_host:
        return execute(args.request.resolve(strict=True))
    # A distinct parent remains responsive even if XSync, hashing or worker I/O blocks.
    subreaper()
    request=json.loads(args.request.read_text())
    limit=900 if request['spec']['hard_s']>=300 else 600
    started=time.monotonic()
    command=['rtk','proxy','/usr/bin/python3','-B',str(Path(__file__).resolve()),
             '--request',str(args.request.resolve())]
    if args.isolated_x11:
        command.extend(['--isolated-x11',str(args.isolated_x11.resolve()),'--isolation-child'])
    else:
        command.append('--owned-worker-host')
    process=None
    failure=None
    previous=signal.signal(signal.SIGTERM,
        lambda *_: (_ for _ in ()).throw(InterruptedError('supervisor termination')))
    try:
        process=subprocess.Popen(command,start_new_session=True)
        process.wait(timeout=limit-5)
    except subprocess.TimeoutExpired:
        failure='outer host hard deadline'
    except BaseException as error:
        failure=type(error).__name__+': '+str(error)
    finally:
        cleanup=clean_descendants(os.getpid(),process)
        signal.signal(signal.SIGTERM,previous)
    output=Path(request['spec']['output_directory'])
    record={'outer_limit_s':limit,'elapsed_s':time.monotonic()-started,
            'failure':failure,'exit_code':None if process is None else process.returncode,
            'cleanup':cleanup,'includes_isolated_x11_startup':bool(args.isolated_x11)}
    if output.is_dir():
        write(output/'supervisor_receipt.json',record)
    else:
        print(json.dumps(record),flush=True)
    return 0 if failure is None and process is not None and process.returncode==0 and not cleanup['remaining'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
