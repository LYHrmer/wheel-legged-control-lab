"""Once-only reservation, isolated worker, bounded host, and frozen postcheck."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from worker15 import identity, write

C = Path(__file__).resolve().parent
W = C.parent
R = Path('/home/lyh/wheel-legged-control-lab')


def sources():
    old = json.loads((W/'rl11/training_run_01/session.json').read_text())
    unused = str(R/'.local-deps/nvidia') + '/'
    expected = {name: row for name,row in old['source_hashes'].items() if not name.startswith(unused)}
    actual = {}
    for name,row in expected.items():
        actual[name] = identity(name)
        if actual[name] != row:
            raise RuntimeError('old frozen input changed: ' + name)
    # Actual user-site Torch resolves user-site NVIDIA; the unused second
    # 2.8 GB copy is excluded. Worker maps and module origins remain checked.
    c14 = json.loads((W/'continuation14/plan_go_14_02.json').read_text())
    for name,row in c14['inputs'].items():
        if 'site-packages/' in name and name not in actual:
            if identity(name) != row:
                raise RuntimeError('diagnostic dependency changed: ' + name)
            actual[name] = row
    parent = W/'rl11/training_run_01/final_checkpoint'
    extra = list(C.glob('*.py')) + list((W/'rl11').glob('*.py')) + list((W/'upright11').glob('*.py')) + [W/'verify_course_e_08_03.py', W/'verify_rl16_training_08.py',C/'astra_contract_15.md',C/'go_review_15.md',
        C/'pure_tests_receipt_15.json',W/'continuation14/final_review_14.md',
        W/'continuation14/independent_readback_14_01.json',
        W/'continuation13/archive13/atomic_archive_13.py',
        W/'rl11/training_run_01/session.json', W/'rl11/training_run_01/final_checkpoint_manifest.json',
        *parent.iterdir(), R/'docs/main_plan_20260929.md']
    for path in extra:
        actual[str(path)] = identity(path)
    frozen = json.loads((W.parent/'rl_improvement_20260914/frozen_source_before.json').read_text())['sha256']
    if len(frozen) != 77:
        raise RuntimeError('77-source inventory differs')
    for name, sha in frozen.items():
        path = R/name
        row = identity(path)
        if row['sha256'] != sha:
            raise RuntimeError('original 77 source mismatch: '+name)
        actual[str(path)] = row
    return dict(sorted(actual.items()))


def freeze():
    if 'GO' not in (C/'go_review_15.md').read_text():
        raise RuntimeError('review GO absent')
    plan = {'schema':'d1-groupclip-plan-15-v1','status':'GO','retry_permitted':False,
            'inputs':sources(), 'control_limit':63368,'normal_native_limit':316840,
            'compiler_native_limit':6,'contract':str(C/'astra_contract_15.md'),
            'arms':{'global': {'controls':16384,'soft_s':600,'close_s':720,'hard_s':750},
                    'grouped':{'controls':16384,'soft_s':600,'close_s':720,'hard_s':750},
                    'eval':{'controls':30600,'soft_s':1800,'close_s':2040,'hard_s':2100}}}
    write(C/'plan_go_15.json',plan)
    print(json.dumps({'plan':identity(C/'plan_go_15.json'),'sources':len(plan['inputs'])}))


def check_maps(session, output):
    # /proc/self/maps is captured inside worker. Verify loaded Torch/NVIDIA
    # libraries are in the same frozen input set (not just a Python import path).
    records = json.loads((output/'mapped_libraries.json').read_text())
    for path in records['paths']:
        if path not in session['source_hashes']:
            raise RuntimeError('loaded Torch/NVIDIA ELF outside frozen closure: '+path)
    return records


def run(arm):
    started=time.monotonic()
    def interrupted(*_):
        raise InterruptedError('host termination requested')
    signal.signal(signal.SIGTERM, interrupted)
    import ctypes
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError('host could not become subreaper')
    planpath=C/'plan_go_15.json'
    plan=json.loads(planpath.read_text())
    if plan['status']!='GO' or plan['control_limit']!=63368 or plan['retry_permitted'] is not False:
        raise RuntimeError('stage15 fixed plan differs')
    spec=plan['arms'][arm]
    for name,row in plan['inputs'].items():
        if identity(name)!=row:
            raise RuntimeError('GO source changed: '+name)
    old=json.loads((W/'rl11/training_run_01/session.json').read_text())
    frozen={**plan['inputs'],str(planpath):identity(planpath)}
    manifests={}
    if arm=='eval':
        for name in ('global','grouped'):
            runpath=C/(name+'_01')
            host=json.loads((runpath/'host_receipt.json').read_text())
            worker=json.loads((runpath/'worker_receipt.json').read_text())
            if host['exit_code']!=0 or host['failure'] is not None or host['changed_sources'] or worker['execution_complete'] is not True:
                raise RuntimeError('training arm is not qualified for evaluation: '+name)
            mpath=runpath/'final_checkpoint_manifest.json'
            manifest=json.loads(mpath.read_text())
            for filename, expected in manifest['files'].items():
                p=runpath/'final_checkpoint'/filename
                if identity(p)!=expected:
                    raise RuntimeError('new final payload differs')
                frozen[str(p)]=expected
            metadata=runpath/'final_checkpoint/final_metadata.json'
            if identity(metadata)['sha256']!=manifest['metadata_sha256']:
                raise RuntimeError('new final metadata differs')
            for p in (metadata,mpath,runpath/'worker_receipt.json',runpath/'host_receipt.json'):
                frozen[str(p)]=identity(p)
            manifests[name]=manifest
        pairpath=C/'training_readback_15.json'
        pair=json.loads(pairpath.read_text())
        if pair.get('engineering_passed') is not True:
            raise RuntimeError('independent training/first-rollout check did not pass')
        frozen[str(pairpath)]=identity(pairpath)
    output=C/(arm+'_01')
    reserve=C/(arm+'_reservation_01.json')
    if output.exists() or reserve.exists():
        raise FileExistsError('stage15 arm already reserved; no retry')
    command=['rtk','proxy','/usr/bin/python3','-B',str(C/'worker15.py'),'--session',str(output/'session.json')]
    write(reserve,{'arm':arm,'controls':spec['controls'],'normal_native':5*spec['controls'],
        'compiler_native':2,'plan_identity':identity(planpath),'command':command,
        'hard_s':spec['hard_s'],'retry_permitted':False,'reservation_not_refundable':True})
    output.mkdir(exist_ok=False)
    environment=dict(old['runtime_environment'])
    environment['PYTHONPATH']+=':'+str(C)+':'+str(W/'continuation13')
    child_env=os.environ.copy()
    for k,v in environment.items():
        if v is None:child_env.pop(k,None)
        else:child_env[k]=v
    session={**old,'arm':arm,'schema':'d1-groupclip-session-15-v1',
        'source_hashes':frozen,'runtime_environment':environment,
        'output_directory':str(output),'argv':command[4:],'worker_argv':command,
        'session_path':str(output/'session.json'),'run_id':'groupclip15_'+arm,
        'wall_clock_utc':datetime.now(timezone.utc).isoformat(),
        'control_limit':spec['controls'],'normal_native_limit':5*spec['controls'],
        'compiler_native_limit':2,'soft_s':spec['soft_s'],'close_s':spec['close_s'],'hard_s':spec['hard_s'],
        'phase':'stage15_'+arm,'seed':151001,'segments':([16384] if arm!='eval' else [600,600]+[1600]*15+[1800]*3),
        'training_control_limit':16384 if arm!='eval' else 0,'heldout_control_limit':30600 if arm=='eval' else 0,
        'wallclock_limit_s':spec['hard_s'],'contract_path':str(C/'astra_contract_15.md'),
        'contract_sha256':frozen[str(C/'astra_contract_15.md')]['sha256'],
        'go_sha256':frozen[str(C/'go_review_15.md')]['sha256'],
        'preregistered_scoring_path':str(C/'astra_contract_15.md'),
        'short_plan_schema':'d1-world-upright-short1000-stage15-preregistered-plan-v1',
        'host_started_monotonic':started,'retry_permitted':False,'plan_path':str(planpath),
        'plan_identity':identity(planpath),'ppo_seed':151001,'command_seed':151002,'measurement_seed':151003,
        'short_plan_source_sha256':frozen[str(C/'recipes15.py')]['sha256'],
        'contract_documents':{str(C/'astra_contract_15.md'):frozen[str(C/'astra_contract_15.md')]},
        'dependency_hashes':{name:row for name,row in old['dependency_hashes'].items() if name in frozen},
        'parentpaths':{'model':str(W/'rl11/training_run_01/final_checkpoint/final_model.zip'),
                       'metadata':str(W/'rl11/training_run_01/final_checkpoint/final_metadata.json')},
        'eval_manifests':manifests}
    from budget_spec_11 import BudgetSpec
    budget = BudgetSpec(16384,1024,256,4)
    session['budget_spec'] = budget.as_dict()
    session['budget_spec_sha256'] = budget.canonical_sha256()
    write(output/'session.json',session)
    process=None; failure=None; cleanup=[]; changed=[]; post_complete=False
    try:
        with (output/'stdout.log').open('xb') as stream:
            process=subprocess.Popen(command,cwd=R,env=child_env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            write(output/'child.json',{'pid':process.pid,'host_pid':os.getpid()})
            soft=False
            while process.poll() is None:
                elapsed=time.monotonic()-started
                if elapsed>=spec['hard_s']:
                    raise TimeoutError('independent host hard deadline')
                if elapsed>=spec['soft_s'] and not soft:
                    worker_pid = json.loads((output/'worker_pid.json').read_text())
                    if worker_pid['session_identity'] != identity(output/'session.json') or os.getpgid(worker_pid['pid']) != process.pid:
                        raise RuntimeError('soft-stop worker identity differs')
                    os.kill(worker_pid['pid'],signal.SIGTERM);soft=True
                if elapsed >= spec['close_s']:
                    raise TimeoutError('worker close deadline; preserve host time for reaping and posthash')
                time.sleep(.1)
        if process.returncode!=0:
            raise RuntimeError('worker exited unsuccessfully')
        check_maps(session,output)
    except BaseException as error:
        failure={'type':type(error).__name__,'message':str(error)}
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=5)
                reap_deadline = time.monotonic()+1
                while time.monotonic()<reap_deadline:
                    try:
                        pid, _status = os.waitpid(-1, os.WNOHANG)
                        if pid == 0: time.sleep(.02)
                    except ChildProcessError:
                        break
            except BaseException as error: cleanup.append(repr(error))
        # Pure source rehash, bounded independently and part of the same host cap.
        def timeout(*_): raise TimeoutError('host source close deadline')
        previous=signal.signal(signal.SIGALRM,timeout)
        signal.setitimer(signal.ITIMER_REAL,max(.01,started+spec['hard_s']-time.monotonic()))
        try:
            changed=[name for name,row in frozen.items() if identity(name)!=row]
            post_complete=True
        except BaseException as error:
            cleanup.append(repr(error))
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        # RTK waits for its owned worker; additionally reject any living member.
        members=[]
        if process is not None:
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():continue
                try:
                    fields=(p/'stat').read_text().rsplit(')',1)[1].split()
                    if fields[0]!='Z' and int(fields[2])==process.pid: members.append(int(p.name))
                except (FileNotFoundError,ProcessLookupError):pass
            if members:
                os.killpg(process.pid,signal.SIGKILL)
                cleanup.append('owned live descendants required kill: '+repr(members))
        if time.monotonic()-started > spec['hard_s']:
            cleanup.append('host hard wall bound exceeded; only kill/reap may use grace')
        receipt={'arm':arm,'failure':failure,'cleanup_errors':cleanup,
            'exit_code':None if process is None else process.returncode,
            'changed_sources':changed,'postcheck_complete':post_complete,'no_live_owned_processes':not members,
            'elapsed_s':time.monotonic()-started,'fully_reserved_budget_closed':True,'retry_permitted':False}
        write(output/'host_receipt.json',receipt)
        print(json.dumps(receipt),flush=True)
    return 0 if failure is None and not cleanup and not changed and post_complete else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--arm',choices=('global','grouped','eval'))
    args=parser.parse_args()
    if args.freeze:freeze()
    elif args.arm:raise SystemExit(run(args.arm))
    else:parser.error('choose --freeze or --arm')
