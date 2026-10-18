"""C38 serial cold binding / NumPy-only two-profile feedback; no training."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTERED_ROOT = Path('/home/lyh/wheel-legged-control-lab')
CONTRACT = 'C38_B22_numpy_half_gain_v1'
OUTPUT = ROOT/'results/b22_numpy38_20261018_01'
AUTHORED = ('scripts/run_b22_numpy38.py','scripts/read_b22_numpy38.py',
            'scripts/b22_numpy_backend38.py','scripts/run_b22_numpy_worker38.py',
            'tests/test_b22_numpy38.py','tests/test_b22_numpy_backend38.py',
            'tests/test_b22_numpy_worker38.py','docs/b22_numpy38.md')
OLD35_SHA = '82b8af91b32cdba16820b26df0542db7e034b3e5f99aef154dadde8ed5b5eb54'
OLD35_RECEIPTS = {
 'release.json':{'bytes':869062,'sha256':'c0d09c7e9a969a28987b128d9f393d5a3be09f656d3c28fbcb470db6e3f2e568'},
 'report.json':{'bytes':3476,'sha256':'1f2464480c13c57ee7a237f822bd35169adede9daeea77abd91017749cca71a3'}}
REFERENCES = {
 'flat_1p6_B_readback.json': {'bytes':33553,'sha256':'69a54f428d5bc9948259ed719a1f428836f40219585a4ca01f29819f932592f8'},
 'yaw_1p2_B_readback.json': {'bytes':31403,'sha256':'3a3858701ee9026170337a7bc4e8a04f08851935af37827ac306d198483eafa6'}}
LIMITS = {'controls':3200,'normal_native':16000,'compiler_native':4,
          'numpy_predict_calls':3202,'numpy_predict_rows':3264,'ppo_load':0,'torch_load':0,
          'worker_sessions':3,'total_wall_s':900,'stage_wall_s':240,'reader_wall_s':100,
          'bootstrap_wall_s':60,'start_effective_bytes':6 << 30,'reserve_effective_bytes':4 << 30,
          'admission_wait_s':300,'admission_poll_s':5.,
          'start_free_disk_bytes':2 << 30,'reserve_free_disk_bytes':1 << 30,
          'required_minimum_nice':10,'training_calls':0}

def load_file(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value

if hashlib.sha256((ROOT/'scripts/run_b22_gain_suite35.py').read_bytes()).hexdigest() != OLD35_SHA:
    raise ValueError('C38 pinned numeric/supervision helper changed')
old35 = load_file('numpy38_pinned35',ROOT/'scripts/run_b22_gain_suite35.py')
c,supervision = old35.c,old35.common
backend = load_file('b22_numpy38_cold_backend',ROOT/'scripts/b22_numpy_backend38.py')


def stages():
    return [{**copy.deepcopy(stage),'gain38':.5} for stage in c.stages()
            if stage['name'] in ('flat_1p6_B','yaw_1p2_B')]


def plan(output=None):
    c.require(output is None or Path(output).absolute() == OUTPUT,'C38 fixed new output required')
    return {'schema':'b22-numpy38-plan-v1','status':'PLAN_ONLY','output':str(OUTPUT),
        'owner_instruction':'把这些优化点，在这台电脑能力之内的都完成',
        'stage_order':['bootstrap',*[row['name'] for row in stages()]],'stages':stages(),'limits':dict(LIMITS),
        'execution_started':False,'retry_permitted':False,'training_calls':0,
        'qualification_granted':False,'automatic_model_replacement':False,
        'relative_sse_and_torque_upper_factor':1.001,
        'scope':'cold zero-NN zero-env zero-native binding bootstrap; NumPy half-gain two-profile feedback; historical C35 only comparison anchors'}


def atomic_json(path,value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)


def reference35():
    root = ROOT/'results/b22_gain_suite35_20261018_01'
    c.require(all(not (root/name).is_symlink() and c.identity(root/name) == expected
                  for name,expected in OLD35_RECEIPTS.items()),'C38 fixed old35 outer receipts changed')
    release = old35.validate_release(root/'release.json')
    parent = c.read_json(root/'report.json')
    c.require(parent['status'] == 'INCOMPLETE' and parent['actual_counts'] is None
              and parent['cleanup']['remaining'] == {} and parent['source'] == release['source'],
              'C38 old35 unknown total/closed source differs')
    files = {str(root/name):c.identity(root/name) for name in ('release.json','report.json')}
    reports = {}
    for stage in stages():
        filename = stage['name']+'_readback.json'
        path = root/filename
        c.require(not path.is_symlink() and c.identity(path) == REFERENCES[filename]
                  and parent['stage_reports'].get(filename) == REFERENCES[filename], 'C38 historical selected readback changed')
        saved = c.read_json(path)
        old35.stage_counts(stage,saved,c.identity(root/'release.json'),
                           release['inputs'][str(ROOT/'scripts/read_b22_gain_suite35.py')])
        c.require(saved['actual_counts']['controls'] == 1600,'C38 historical reference horizon incomplete')
        files[str(path)] = c.identity(path)
        for name,expected in saved['artifacts'].items():
            item = root/stage['name']/name
            c.require(not Path(name).is_absolute() and '..' not in Path(name).parts
                      and not item.is_symlink() and c.identity(item) == expected,'C38 historical raw changed')
            files[str(item)] = expected
        run = root/stage['name']
        session,receipt = (c.read_json(run/name) for name in ('session.json','worker_receipt.json'))
        old35.case_completion(stage,receipt)
        owner_path = ROOT/'scripts/run_b22_worker35.py'
        owner = c.load_pinned(owner_path,release['inputs'][str(owner_path)],'numpy38_original35_owner')
        c.require(receipt['session_identity'] == c.identity(run/'session.json')
            and session['execution_contract_id'] == old35.CONTRACT and session['gain35'] == .5
            and session['output_directory'] == str(run)
            and session['argv'] == [str(owner_path),'--session',str(run/'session.json'),'--output',str(run)]
            and all(session['source_hashes'].get(k) == v for k,v in release['inputs'].items())
            and c.read_json(run/'derived_sources35.json') == owner.derivation_proof(owner.derived_sources()),
            'C38 historical owner/source derivation differs')
        for item in run.rglob('*'):
            if item.is_file():
                c.require(not item.is_symlink(),'C38 historical symlink forbidden')
                if str(item) not in files:files[str(item)] = c.identity(item)
        reports[stage['name']] = saved
    return {'status':'TWO_PINNED_CLOSED_REFERENCES','old35_outer_actual_counts':None,
            'historical_total_resources_known':False,'inputs':files,'reports':reports,
            'old_inputs':release['inputs']}


def closure(source,references,exports,config):
    return {**config['source_hashes'],**source['authored'],**references['inputs'],**exports,
            **references['old_inputs'],str(ROOT/'scripts/run_b22_gain_suite35.py'):c.identity(ROOT/'scripts/run_b22_gain_suite35.py'),
            str(ROOT/c.MANIFEST):c.identity(ROOT/c.MANIFEST)}


def validate_release(path):
    path = Path(path).absolute()
    c.require(path == OUTPUT/'release.json' and not path.is_symlink(),'C38 fixed release required')
    release = c.read_json(path)
    references,exports = reference35(),backend.export_identities38()
    check_authored(release['source'])
    config = c.read_json(ROOT/c.MANIFEST)
    c.require(release.get('schema') == 'b22-numpy38-release-v1' and release.get('plan') == plan()
              and release.get('references35') == references['inputs']
              and release.get('export36') == exports
              and release.get('inputs') == closure(release['source'],references,exports,config), 'C38 source/provenance closure differs')
    for name,expected in release['inputs'].items():
        c.require(Path(name).is_absolute() and not Path(name).is_symlink() and c.identity(name) == expected,
                  'C38 frozen input changed: '+name)
    return release


def bootstrap_argv(session,output):
    return ['/usr/bin/python3','-B',str(ROOT/'scripts/run_b22_numpy_worker38.py'),
            '--bootstrap','--session',str(session),'--output',str(output)]


def bootstrap_request(output,release,release_path,nonce):
    run = output/'bootstrap'
    run.mkdir(exist_ok=False)
    config = c.read_json(ROOT/c.MANIFEST)
    session = {**config['shared_session'],'schema':'b22-numpy38-bootstrap-session-v1',
        'execution_contract_id':CONTRACT,'output_directory':str(run),
        'parent_pid':os.getpid(),'nonce':nonce,'release_path':str(release_path),
        'release_identity':c.identity(release_path),'runtime_environment':config['runtime_environment'],
        'source_hashes':{**release['inputs'],str(release_path):c.identity(release_path)},
        'controls':0,'normal_native':0,'compiler_native':0,'compiler_native_limit':0,'model_limits':dict.fromkeys(backend.API_LIMITS,0)}
    session['argv'] = bootstrap_argv(run/'session.json',run)
    c.write_json(run/'session.json',session)
    return session['argv']


def validate_bootstrap_launch38(session_path,output):
    session_path,output = Path(session_path).absolute(),Path(output).absolute()
    c.require(output == OUTPUT/'bootstrap' and session_path == output/'session.json'
              and not output.is_symlink() and not session_path.is_symlink(),'C38 fixed bootstrap paths required')
    session = c.read_json(session_path)
    release_path = Path(session['release_path'])
    release = validate_release(release_path)
    journal = c.read_json(OUTPUT/'report.json')
    c.require(journal.get('status') == 'RUNNING' and journal.get('active_stage') == 'bootstrap'
              and journal.get('supervisor_pid') == os.getppid() == session.get('parent_pid')
              and journal.get('release_identity') == session.get('release_identity') == c.identity(release_path)
              and journal.get('source') == release['source']
              and journal.get('nonce') == session.get('nonce') and isinstance(session['nonce'],str) and len(session['nonce']) == 48
              and session.get('argv') == journal.get('active_command') == bootstrap_argv(session_path,output)
              and session.get('schema') == 'b22-numpy38-bootstrap-session-v1'
              and session.get('execution_contract_id') == CONTRACT,'C38 active bootstrap parent required')
    config = c.read_json(ROOT/c.MANIFEST)
    c.require(all(session.get(k) == v for k,v in {**config['shared_session'],'compiler_native_limit':0}.items())
              and session.get('runtime_environment') == config['runtime_environment']
              and session.get('source_hashes') == {**release['inputs'],str(release_path):c.identity(release_path)}
              and all(type(session.get(k)) is int and session[k] == 0 for k in ('controls','normal_native','compiler_native'))
              and session.get('model_limits') == dict.fromkeys(backend.API_LIMITS,0), 'C38 bootstrap source/zero reservation differs')
    return session


def script_request(stage,output,release,config,release_path):
    launch,run = output/(stage['name']+'_launch'),output/stage['name']
    launch.mkdir(exist_ok=False)
    spec = {'schema':'d1-c23-gui-worker-v1','mode':'script','render':False,'profile':stage['profile'],
        'actor':'B','seed':stage['seed'],'mirror':False,'control_limit':1600,'seconds':16.,
        'output_directory':str(run),'qualification':False,'controller_variant':'combined','gain38':.5,
        'execution_contract_id':CONTRACT,'soft_s':180,'close_s':235,'hard_s':240,'frame_limit':1000,
        'require_event_driver':False,'model_limits':dict(backend.API_LIMITS)}
    go = {'decision':'GO','scope':'C38 NumPy-only fixed two-profile diagnostic; no training',
        'repository':str(ROOT),'worker':str(ROOT/'scripts/run_b22_numpy_worker38.py'),
        'inputs':{**release['inputs'],str(release_path):c.identity(release_path)},
        'runtime_environment':config['runtime_environment'],'shared_session':config['shared_session'],
        'arms':{stage['name']:spec},'x11_events_by_arm':{}}
    c.write_json(launch/'source_go.json',go)
    c.write_json(launch/'request.json',{'source_go_path':str(launch/'source_go.json'),'arm':stage['name'],'spec':spec})
    return ['rtk','proxy','/usr/bin/python3','-B',config['host'],'--request',str(launch/'request.json')]


def case_completion(stage,worker):
    c.require(worker.get('execution_contract_id') == CONTRACT,'C38 worker contract differs')
    path = ROOT/'scripts/b22_suite31_contract.py'
    node = copy.deepcopy(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name == 'case_completion'))
    changed = 0
    for value in ast.walk(node):
        if isinstance(value,ast.Constant) and value.value == 'd1-c23-gui-worker-receipt-v1':
            value.value = 'b22-numpy38-worker-receipt-v1'; changed += 1
    c.require(changed == 1,'C38 receipt schema seam changed')
    namespace = dict(c.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),namespace)  # noqa: S102 -- pinned pure receipt function; explicit new schema only
    return namespace['case_completion'](stage,worker)


def partial_counts(stage,run,release):
    try:
        session,worker = (c.read_json(run/name) for name in ('session.json','worker_receipt.json'))
        c.require(session.get('execution_contract_id') == worker.get('execution_contract_id') == CONTRACT
                  and worker.get('schema') == 'b22-numpy38-worker-receipt-v1'
                  and worker.get('session_identity') == c.identity(run/'session.json')
                  and session.get('output_directory') == str(run) and session.get('gain38') == .5
                  and all(session['source_hashes'].get(k) == v for k,v in release['inputs'].items()), 'foreign C38 partial')
        py,state = worker['result']['python'],worker['result']['C_final']
        p,n,k = py['control_completed'],py['native_returned'],state['construction_returns']
        c.require(all(type(v) is int for v in (p,n,k)) and 0 <= p <= 1600 and 5*p <= n <= min(8000,5*(p+1))
                  and state['control_returns'] == n and 0 <= k <= 2,'invalid C38 partial physical interval')
        return {'controls':p,'normal_native':n,'compiler_native':k}
    except (OSError,ValueError,KeyError,TypeError):
        return None


def counted(completed,active=None,partial=None):
    return old35.counted_prefix(completed,active,partial)


def stage_counts(stage,saved,release_id,reader_id):
    value = saved.get('actual_counts',{})
    n = value.get('controls')
    c.require(saved.get('schema') == 'b22-numpy38-stage-readback-v1' and saved.get('status') == 'COMPLETE'
        and saved.get('execution_contract_id') == CONTRACT and saved.get('stage') == stage['name']
        and saved.get('release_identity') == release_id and saved.get('reader_identity') == reader_id
        and type(n) is int and 0 < n <= 1600 and all(type(v) is int and v >= 0 for v in value.values())
        and value.get('normal_native') == 5*n and value.get('compiler_native') == 2
        and saved.get('gain_ledger_rows') == n and saved.get('numpy_predict_calls') == n+1
        and all(type(saved.get(key)) is int for key in ('numpy_predict_calls','numpy_predict_rows','ppo_load','torch_load'))
        and saved.get('numpy_predict_rows') == n+32 and saved.get('ppo_load') == saved.get('torch_load') == 0,
        'C38 stage proof/counts differ')
    return {key:value[key] for key in ('controls','normal_native','compiler_native')}


def source_identity():
    subprocess.run(['git', 'ls-files', '--error-unmatch', *AUTHORED], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, timeout=5)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *AUTHORED], cwd=ROOT, check=True, timeout=5)
    return {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, timeout=5).strip(),
            'authored': {str(ROOT/name): c.identity(ROOT/name) for name in AUTHORED}}

def check_authored(source):
    c.require(re.fullmatch('[0-9a-f]{40}', source.get('git_commit', ''))
              and set(source.get('authored', {})) == {str(ROOT/name) for name in AUTHORED}, 'C38 source commit/closure differs')
    for name in AUTHORED:
        path = ROOT/name
        c.require(not path.is_symlink() and c.identity(path) == source['authored'][str(path)], 'C38 authored source changed')
        blob = subprocess.check_output(['git', 'show', source['git_commit']+':'+name], cwd=ROOT, stderr=subprocess.PIPE, timeout=5)
        c.require(hashlib.sha256(blob).hexdigest() == source['authored'][str(path)]['sha256'], 'C38 uncommitted authored source')

def cleanup_owned(host, process):
    """Finish owned cleanup while retaining stop signals that31 temporarily ignores."""
    received = []
    previous = {number: signal.signal(number, lambda number, frame: received.append(signal.Signals(number).name))
                for number in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
    try:
        receipt = host.clean_descendants(os.getpid(), process)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    if received:
        raise InterruptedError('C38 termination during owned cleanup: '+','.join(received))
    return receipt

def supervise(argv, log, *, host, **kwargs):
    class OwnedHost:
        def clean_descendants(self, parent, process):
            return cleanup_owned(host, process)
    clock = kwargs.get('clock',time.monotonic)
    started = clock()
    code = supervision.supervise(argv, log, host=OwnedHost(), **kwargs)
    c.require(clock()-started < kwargs['limit'] and clock() < kwargs['deadline'],
              'C38 process exceeded stage/total bound during cleanup')
    return code


def execute(output=OUTPUT, *, source_reader=source_identity, references_reader=reference35,
            exports_reader=backend.export_identities38, config_reader=c.registered_config,
            supervisor=supervise, validator=validate_release, clock=time.monotonic):
    output = Path(output).absolute()
    c.require(ROOT == REGISTERED_ROOT and output == OUTPUT and not output.exists() and not output.is_symlink(),
              'C38 fixed new output required; no retry or overwrite')
    started = clock()
    source,references,exports = source_reader(),references_reader(),exports_reader()
    c.require(os.getpriority(os.PRIO_PROCESS,0) >= 10,'C38 requires external Nice10')
    initial_memory = supervision.memory()
    c.require(type(initial_memory) is int and initial_memory >= 0,'C38 effective startup RAM unknown')
    if initial_memory < 6 << 30 or shutil.disk_usage(ROOT).free < 2 << 30:
        return {'status':'NOT_READY','execution_started':False,'actual_counts':counted({})}
    output.mkdir(exist_ok=False)
    report = {'schema':'b22-numpy38-supervision-v1','status':'RUNNING','source':source,
        'supervisor_pid':os.getpid(),'active_stage':None,'active_command':None,'nonce':None,
        'stage_reports':{},'actual_counts':None,'known_actual_counts':counted({}),
        'admission_waits':[],
        'actual_nn_counts':None,'known_nn_counts':{'predict_calls':0,'predict_rows':0,'ppo_load':0,'torch_load':0},
        'old35_outer_actual_counts':None,'historical_total_resources_known':False,
        'qualification_granted':False,'automatic_model_replacement':False,'retry_permitted':False,
        'nice':os.getpriority(os.PRIO_PROCESS,0),'failure':None}
    atomic_json(output/'report.json',report)
    completed,nn_counts = {},{}
    host,active,release = None,None,None
    previous = {number:signal.getsignal(number) for number in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    deadline = started+LIMITS['total_wall_s']-10
    def stopped(signum,frame):
        raise InterruptedError('C38 interrupted: '+signal.Signals(signum).name)
    def sample(admission=False):
        available,free = supervision.memory(),shutil.disk_usage(output).free
        with (output/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps({'elapsed_s':clock()-started,'stage':report['active_stage'],
                'effective_available_bytes':available,'free_disk_bytes':free},allow_nan=False)+'\n')
        c.require(type(available) is int and available >= 0,'C38 effective RAM unknown')
        report['minimum_effective_bytes'] = min(report.get('minimum_effective_bytes',available),available)
        c.require(type(available) is int and available >= (6 if admission else 4) << 30,
                  'C38 RAM admission/runtime reserve exhausted')
        c.require(free >= 1 << 30 and clock() < deadline,'C38 disk/wall reserve exhausted')
        return available
    def journal(stage,argv,nonce=None,required_window=0):
        if stage is not None:
            c.require(clock()+required_window < deadline,'C38 insufficient window before child claim')
        report.update(active_stage=stage,active_command=argv,nonce=nonce)
        atomic_json(output/'report.json',report)
    def rebuild():
        report['known_actual_counts'] = counted(completed)
        report['known_nn_counts'] = {key:sum(row[key] for row in nn_counts.values()) for key in report['known_nn_counts']}
        c.require(all(report['known_actual_counts'][key] <= LIMITS[key] for key in ('controls','normal_native','compiler_native')),
                  'C38 physical budget exhausted')
    def admit(label,required_window):
        """Only after owned cleanup, before a new request/session/nonce is created."""
        receipt = cleanup_owned(host,None)
        c.require(receipt['remaining'] == {},'C38 admission requires no live owned child')
        journal(None,None)
        waiting,wait_started = None,clock()
        try:
            while True:
                available = sample(False)
                duration = clock()-wait_started
                c.require(clock()+required_window < deadline,'C38 insufficient bounded stage/readback window')
                if waiting is not None:
                    waiting.update(duration_s=duration)
                    waiting['samples'].append({'elapsed_s':clock()-started,'effective_available_bytes':available})
                if available >= LIMITS['start_effective_bytes']:
                    validator(release_path)
                    c.require(source_reader() == source,'C38 source changed during admission')
                    c.require(clock()+required_window < deadline,'C38 no window after source readback')
                    available = sample(False)
                    if available >= LIMITS['start_effective_bytes']:
                        c.require(clock()+required_window < deadline,'C38 no stage window after final RAM readback')
                        c.require(clock()-wait_started <= LIMITS['admission_wait_s'],'C38 admission late readiness')
                        if waiting is not None:
                            waiting.update(status='ADMITTED',duration_s=clock()-wait_started)
                            atomic_json(output/'report.json',report)
                        return
                if waiting is None:
                    waiting = {'stage':label,'status':'WAITING_FOR_ADMISSION','started_elapsed_s':clock()-started,
                        'duration_s':clock()-wait_started,
                        'samples':[{'elapsed_s':clock()-started,'effective_available_bytes':available}]}
                    report['admission_waits'].append(waiting)
                elapsed = clock()-wait_started
                c.require(elapsed < LIMITS['admission_wait_s'],'C38 admission wait exhausted')
                atomic_json(output/'report.json',report)
                time.sleep(min(LIMITS['admission_poll_s'],LIMITS['admission_wait_s']-elapsed))
        except BaseException as error:  # Preserve waiting stop/interrupt, then re-raise before any child claim.
            if waiting is not None:
                waiting.update(status='STOPPED',duration_s=clock()-wait_started,error=type(error).__name__+': '+str(error))
            raise
    for number in previous:
        signal.signal(number,stopped)
    signal.setitimer(signal.ITIMER_REAL,max(.01,deadline-clock()))
    try:
        config,readiness = config_reader()
        release = {'schema':'b22-numpy38-release-v1','plan':plan(),'source':source,
            'references35':references['inputs'],'export36':exports,
            'inputs':closure(source,references,exports,config),'v27_readiness':readiness}
        release_path = output/'release.json'
        c.write_json(release_path,release)
        validator(release_path)
        report['release_identity'] = c.identity(release_path)
        host = c.load_pinned(Path(config['host']),release['inputs'][config['host']],'numpy38_owned_host')
        host.subreaper()
        admit('bootstrap',LIMITS['bootstrap_wall_s'])
        nonce = secrets.token_hex(24)
        argv = bootstrap_request(output,release,release_path,nonce)
        journal('bootstrap',argv,nonce,LIMITS['bootstrap_wall_s'])
        environment = dict(os.environ)
        for key,value in config['runtime_environment'].items():
            if value is None:environment.pop(key,None)
            else:environment[key] = value
        def bootstrap_factory(*args,**kwargs):
            return subprocess.Popen(*args,**kwargs,env=environment)
        c.require(supervisor(argv,output/'bootstrap_stdout.log',host=host,sample=sample,
            limit=LIMITS['bootstrap_wall_s'],deadline=deadline,factory=bootstrap_factory) == 0,'C38 cold bootstrap failed')
        reader = load_file('numpy38_saved_reader',ROOT/'scripts/read_b22_numpy38.py')
        reader.verify_bootstrap(c.read_json(output/'bootstrap/bootstrap38_report.json'),c.read_json(output/'bootstrap/session.json'),
                                c.identity(output/'bootstrap/session.json'))
        completed['bootstrap'],nn_counts['bootstrap'] = counted({}),dict(report['known_nn_counts'])
        report['stage_reports']['bootstrap/bootstrap38_report.json'] = c.identity(output/'bootstrap/bootstrap38_report.json')
        rebuild()
        for active in stages():
            admit(active['name'],LIMITS['stage_wall_s']+LIMITS['reader_wall_s'])
            argv = script_request(active,output,release,config,release_path)
            journal(active['name'],argv,required_window=LIMITS['stage_wall_s']+LIMITS['reader_wall_s'])
            code = supervisor(argv,output/(active['name']+'_stdout.log'),host=host,sample=sample,
                limit=LIMITS['stage_wall_s'],deadline=deadline)
            case_completion(active,c.read_json(output/active['name']/'worker_receipt.json'))
            c.require(code == 0,'C38 NumPy worker/host failed')
            destination = output/(active['name']+'_readback.json')
            admit(active['name']+'_reader',LIMITS['reader_wall_s'])
            argv = ['rtk','proxy','/usr/bin/python3','-B',str(ROOT/'scripts/read_b22_numpy38.py'),
                '--release',str(release_path),'--stage',active['name'],'--report',str(destination)]
            journal(active['name']+'_reader',argv,required_window=LIMITS['reader_wall_s'])
            c.require(supervisor(argv,output/(active['name']+'_reader.log'),host=host,sample=sample,
                limit=LIMITS['reader_wall_s'],deadline=deadline) == 0,'C38 NumPy saved readback failed')
            saved = c.read_json(destination)
            completed[active['name']] = stage_counts(active,saved,c.identity(release_path),
                release['inputs'][str(ROOT/'scripts/read_b22_numpy38.py')])
            nn_counts[active['name']] = {'predict_calls':saved['numpy_predict_calls'],'predict_rows':saved['numpy_predict_rows'],
                'ppo_load':0,'torch_load':0}
            report['stage_reports'][destination.name] = c.identity(destination)
            rebuild()
            active = None
            journal(None,None)
        argv = ['rtk','proxy','/usr/bin/python3','-B',str(ROOT/'scripts/read_b22_numpy38.py'),
            '--release',str(release_path),'--compare','--report',str(output/'comparison.json')]
        admit('comparison',LIMITS['reader_wall_s'])
        journal('comparison',argv,required_window=LIMITS['reader_wall_s'])
        c.require(supervisor(argv,output/'comparison_stdout.log',host=host,sample=sample,
            limit=LIMITS['reader_wall_s'],deadline=deadline) == 0,'C38 comparison failed')
        compared = c.read_json(output/'comparison.json')
        c.require(compared.get('schema') == 'b22-numpy38-comparison-v1' and compared.get('status') == 'COMPLETE'
            and compared.get('release_identity') == c.identity(release_path)
            and compared.get('stage_reports') == {k:v for k,v in report['stage_reports'].items() if k.endswith('_readback.json')}
            and compared.get('new_actual_counts') == counted(completed)
            and compared.get('qualification_granted') is False and source_reader() == source,'C38 final proof differs')
        validator(release_path)
        sample(False)
        report.update(status='COMPLETE',actual_counts=counted(completed),actual_nn_counts=report['known_nn_counts'],
            comparison=c.identity(output/'comparison.json'),all_two_full_horizons=compared['all_two_full_horizons'],
            all_two_tasks_passed=compared['all_two_tasks_passed'],backend_compatibility_passed=compared['backend_compatibility_passed'],
            active_stage=None,active_command=None)
    except BaseException as error:  # noqa: BLE001 -- stop signals must close and write the owned attempt
        report.update(status='INCOMPLETE',failure=type(error).__name__+': '+str(error))
        if active is not None and release is not None:
            partial = partial_counts(active,output/active['name'],release)
            report['known_actual_counts'] = counted(completed,active,partial)
            report['active_stage_actual_counts'] = partial
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for number in previous:
            signal.signal(number,signal.SIG_IGN)
        if host is not None:
            try:
                report['cleanup'] = cleanup_owned(host,None)
                c.require(report['cleanup']['remaining'] == {},'C38 owned descendants remain')
            except BaseException as error:  # noqa: BLE001 -- cleanup interruption must invalidate COMPLETE
                report.update(status='INCOMPLETE',failure=(report['failure'] or '')+'; cleanup '+str(error))
        report['elapsed_s'] = clock()-started
        if report['elapsed_s'] >= LIMITS['total_wall_s']:
            report.update(status='INCOMPLETE',failure='C38 final wall limit exhausted')
        if report['status'] != 'COMPLETE':
            report.update(actual_counts=None,actual_nn_counts=None)
        try:
            atomic_json(output/'report.json',report)
        finally:
            for number,handler in previous.items():signal.signal(number,handler)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(plan(args.output),ensure_ascii=False,indent=2))
        return 0
    value = execute(args.output or OUTPUT)
    print(json.dumps(value,ensure_ascii=False,indent=2))
    return 0 if value['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
