"""C42 versioned worker adapters. Default/import is standard-library only.

Production execution requires the new parent's live launch reservation. Old
sources stay byte-identical; derived ASTs and injected globals are recorded.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import resource
import signal
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT/'scripts/worker_b22_roughtrain45.py'
CONTRACT = 'C45_B22_rough_terrain_curriculum_arm_v1'
REWARD_SCHEMA = 'd1-b22-training-tracking-deficit-alpha42-v1'
FINAL_SCHEMA = 'd1-b22-tracking42-final-checkpoint-v1'
SPEC_PATH = ROOT/'results/d1_main_route_execution_20260930_22/spec22.json'
SPEC_SHA = 'f061b64320e8fae35dd8ca0d186242579b4640bd5fecdd3e6da63d2f6983f8f4'
MANIFEST = ROOT/'runtime/d1_b22/runtime_manifest_v27.json'
RECEIPT_RESERVE = 1 << 20
EXTERNAL_FILE_LIMIT = 16 << 20
_quota42 = None
_ff44 = None
_reward_base_by_type42 = {}
SOURCES = {
 'worker': ('runtime/d1_b22/research/continuation22/worker22.py','9c02f789c3b15f0e57a6d0bc535595ee6a9d5c515ec07bd2b2283c735fa57fd1'),
 'recipes': ('runtime/d1_b22/research/continuation22/recipes22.py','335cfbcb1e36a35112cb23896894bbf1c0e7c80cea9725bc4bf2dac2253090d4'),
 'curriculum': ('runtime/d1_b22/research/continuation22/curriculum22.py','53932c20efc59ec0488328a9ea82d7a1904ec9f9f3e497bf9583d6a686db57e5'),
 'checkpoint': ('runtime/d1_b22/research/continuation21/checkpoint21.py','ef23653bd0419a8fd10d12d05325b8d2ad932ff0426175e60eab96b86aa0019f'),
 'heldout': ('runtime/d1_b22/research/continuation22/heldout22.py','3fba56a53ac216c1f7dd9370bbb3f90f09f16f79a0de5df51a6bb739c4e86863'),
 'train': ('runtime/d1_b22/research/continuation22/train22.py','5bf26ed060ebcf5638bb0a1877b3ec0e339be18164f5c2f3a222899f44bcf008'),
 'strict': ('runtime/d1_b22/research/rl11/rl16_learning_11.py','837b1b7c814fcb174cc8f1d8ce0ad1bfcbf623e2cc4aba63ad36c3304f5ff673'),
 'learning': ('runtime/d1_b22/research/continuation21/learning21.py','423141d1632916a605eb681d54ef15b51a90795995f9bcfdb17f41069d54b10b'),
 'metadata': ('runtime/d1_b22/research/rl11/run_short_rl16_training_11.py','c9c8127365ecf0fe071979bf42a2634f4169c8b7e8136f3837f971d004fff16e'),
 'geometry': ('runtime/d1_b22/research/continuation21/geometry21.py','166734144ff69582fe1e7eef97be8fd49a008f133d10b50aaddc9aab5a9575ef'),
 'servo': ('runtime/d1_b22/research/course_impl08/full_drive_servo_08.py','6f96f5a94e5824c269fead504c2f8a0fa9789c75cea42bebd08fccb88ea50ebb'),
 'controller': ('runtime/d1_b22/research/continuation18/controller18.py','b8f3b1d59880ca0bff7d7d4791ba823a101d396dd9f614e25414b83bd2133aba'),
 'support': ('runtime/d1_b22/research/continuation22/runtime_support22.py','ffed9b03e0baa329f87691fe3376cf93d9db916996df075c2da5193cde677411'),
}
PARENT_FILES = {
 'final_model.zip': '7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691',
 'final_metadata.json': 'af700d1f2612e441c8f0560cca38c77b6a854526d3e841e544ef9d5321672830',
 'final_probe_observations_f32.bin': '7c8112f2db89e67047d0317cf61c8a140ef4ab9be1aaa584bdd80b47b03efd1a',
 'final_probe_actions_f32.bin': 'e758c79730cb3b954c27e893532bc40bf48aa94f371c98008797c22b5cfadeef',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'C42 regular source required: '+str(path))
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train45 = load_file('b22_roughtrain45', ROOT/'scripts/b22_roughtrain45.py')


def package42():
    path = ROOT/'scripts/b22_runtime_packages39.py'
    require(identity(path)['sha256']=='7d5111bb4b6ecddc2ecf67ee6dc83197c68045eca00baae283e7229405b72868',
            'C42 original package helper changed')
    return load_file('b22_tracking42_package39', path)


def required_inputs42():
    """Only bounded file hashing/stdlib inventory; no package/model import."""
    values = {}
    for name, digest in SOURCES.values():
        p = ROOT/name
        values[str(p)] = identity(p)
        require(values[str(p)]['sha256']==digest, 'C42 frozen adapter source changed: '+name)
    require(identity(SPEC_PATH)['sha256']==SPEC_SHA, 'C42 original spec changed')
    values[str(SPEC_PATH)] = identity(SPEC_PATH)
    for name, digest in PARENT_FILES.items():
        p = ROOT/'runtime/d1_b22/checkpoint'/name
        values[str(p)] = identity(p)
        require(values[str(p)]['sha256']==digest, 'C42 B22 complete parent changed')
    p = ROOT/'scripts/b22_tracking_reward42.py'
    require(identity(p)['sha256']=='d05651d3e01bc5d4794721da3b403c95bfbc9be1364591ecbcddfbb59958e68e',
            'C42 pure reward source changed')
    values[str(p)] = identity(p)
    for extra in ('scripts/b22_roughtrain45.py','scripts/b22_yawtrain44.py','src/wheel_legged_control/d1/yaw_feedforward44.py'):
        values[str(ROOT/extra)] = identity(ROOT/extra)
    finite = read_json(SPEC_PATH)['finite_geometry']
    report = Path(finite['prior_preflight_path'])
    require(identity(report)==finite['prior_preflight_identity'], 'C42 old geometry report changed')
    values[str(report)] = identity(report)
    registered = read_json(report)['inputs']
    for terrain in ('flat','bumps','rough','ramp'):
        for suffix in ('geometry21.json','initial_state21.npz','control_reset21.json'):
            p = Path(finite['expected_templates_directory'])/('expected_'+terrain+'_'+suffix)
            require(identity(p)==registered[str(p)], 'C42 sealed geometry template changed')
            values[str(p)] = registered[str(p)]
    helper = package42()
    values[str(ROOT/'scripts/b22_runtime_packages39.py')] = identity(ROOT/'scripts/b22_runtime_packages39.py')
    for name, entry in helper.inventory39()['inputs'].items():
        require(name not in values or values[name]==entry, 'C42 package source conflict')
        values[name] = entry
    return values


def session_defaults42(mode, arm):
    require((mode=='train' and arm in train45.ARMS) or (mode=='bootstrap' and arm is None),
            'C44 unknown mode/reward arm')
    config = read_json(MANIFEST)
    metadata = read_json(ROOT/'runtime/d1_b22/checkpoint/final_metadata.json')
    controls = 32768 if mode=='train' else 30600 if mode=='eval' else 0
    limits = ({'load':2,'torch_load':6,'save':2,'learn':1,'train':32,'forward':32768,
               'predict':3,'predict_values':64,'evaluate_actions':512,'backward':512}
              if mode=='train' else
              {'load':2,'torch_load':6,'save':0,'learn':0,'train':0,'forward':0,
               'predict':20802,'predict_values':0,'evaluate_actions':0,'backward':0})
    if mode=='bootstrap':
        limits=dict.fromkeys(limits,0)
    return {
        'mode': mode,
        'arm': 'B' if mode == 'train' else mode,
        'reward_arm': arm,
        'alpha': train45.ARMS[arm]['alpha'] if mode == 'train' else None,
        'stage': 2,
        'course_arm': 'B',
        'seed': 201011,
        'ppo_seed': 201011,
        'command_seed': 201002,
        'measurement_seed': 201003,
        'source_episode_offset': 0,
        'short_plan_schema': 'd1-c18-command-coverage-short1000-c20-v1',
        'short_plan_source_sha256': SOURCES['recipes'][1],
        'budget_spec': metadata['budget_spec'] if mode!='bootstrap' else None,
        'budget_spec_sha256': metadata['budget_spec_sha256'] if mode!='bootstrap' else None,
        'parentpaths': {'model': str(ROOT / 'runtime/d1_b22/checkpoint/final_model.zip'), 'metadata': str(ROOT / 'runtime/d1_b22/checkpoint/final_metadata.json'), 'model_sha256': PARENT_FILES['final_model.zip']},
        'model_limits': limits,
        'controls': controls,
        'control_limit': controls,
        'normal_native_limit': controls * 5,
        'compiler_native_limit': 0 if mode=='bootstrap' else 2,
        'training_control_limit': 32768 if mode == 'train' else 0,
        'heldout_control_limit': 30600 if mode == 'eval' else 0,
        'segments': [controls] if controls else [],
        'controller_variant': 'combined',
        'reference_definition': metadata['reference_definition'],
        'dependency_hashes': copy.deepcopy(config['source_hashes']),
        'run_id': 'tracking42_' + ('train_' + arm if mode == 'train' else mode),
        'phase': 'tracking42_' + mode,
        'floor_seed': 201090,
        'include_floors': mode=='eval',
        'evaluation_split': 'tracking42_original_six',
        'contract_documents': {str(ROOT / 'docs/b22_roughtrain45.md'): identity(ROOT / 'docs/b22_roughtrain45.md')},
        'yaw_feedforward': train45.FEEDFORWARD if mode == 'train' else None,
        'curriculum': train45.ARMS[arm]['curriculum'] if mode == 'train' else None,
    }


def validate_numeric_session42(session):
    expected = session_defaults42(session['mode'], session['reward_arm'])
    require(all(session.get(k)==v for k,v in expected.items()), 'C42 numeric/session defaults differ')
    require(session.get('execution_contract_id')==CONTRACT and session.get('retry_permitted') is False,
            'C42 independent one-shot identity required')
    cap = 3 << 30 if session['mode']=='train' else 2 << 30 if session['mode']=='eval' else 64 << 20
    require(type(session.get('quota_bytes')) is int and session['quota_bytes']==cap,
            'C42 exact per-worker quota required')
    require(session.get('normal_native_limit')==5*session['controls'] and
            session.get('compiler_native_limit')==(0 if session['mode']=='bootstrap' else 2),
            'C42 exact 5T/construction reservation required')


class QuotaExceeded42(OSError):
    """A before-write quota rejection, rather than successful archive output."""


def tree_bytes42(folder):
    total = 0
    for p in Path(folder).rglob('*'):
        try:
            value = p.lstat()
        except FileNotFoundError:
            continue  # Concurrent committed tmp rename; this is a snapshot.
        require(not stat.S_ISLNK(value.st_mode), 'C42 quota tree contains symlink')
        if stat.S_ISREG(value.st_mode):
            total += value.st_size
    return total


class _QuotaStream42:
    def __init__(self, stream, ledger, *, receipt=False):
        self.stream, self.ledger, self.high = stream, ledger, stream.tell()
        self.receipt = receipt

    def write(self, value):
        growth = max(0, self.stream.tell()+len(value)-self.high)
        self.ledger.charge(growth,receipt=self.receipt)
        count = self.stream.write(value)
        self.high = max(self.high, self.stream.tell())
        return count

    def __getattr__(self, name):
        return getattr(self.stream, name)


class _QuotaText42:
    def __init__(self, stream, ledger):
        self.raw, self.charged = stream, _QuotaStream42(stream, ledger)
    def write(self, value):
        require(isinstance(value, str), 'C42 text archive needs text')
        self.charged.write(value.encode('utf-8'))
        return len(value)
    def __getattr__(self, name):
        return getattr(self.raw, name)


class StageQuota42:
    """Hard worker budget includes partial writes; global eight GiB is observed.

    Written-byte growth is never refunded after failure. Checkpoint reservations
    intentionally overcharge bounded files rather than risk an unbilled save.
    """
    def __init__(self, output, limit, *, execution_output=None, global_stop=8 << 30,
                 reserve=RECEIPT_RESERVE):
        self.output, self.limit = Path(output), limit
        self.execution_output = Path(execution_output or output)
        self.global_stop, self.reserve = global_stop, reserve
        require(type(limit) is int and limit>reserve, 'C42 invalid stage byte budget')
        self.initial = tree_bytes42(self.output)
        self.payload = self.reserved = 0
        self.exhausted = False
        self.check_global()

    def check_global(self):
        if tree_bytes42(self.execution_output)>self.global_stop:
            self.exhausted = True
            raise QuotaExceeded42('C42 monitored global output stop threshold exceeded')

    def charge(self, amount, *, receipt=False):
        require(type(amount) is int and amount>=0, 'C42 invalid write accounting')
        total = self.initial+self.payload+self.reserved+amount
        used = self.reserved+amount if receipt else self.payload+amount
        cap = self.reserve if receipt else self.limit-self.reserve-self.initial
        if total>self.limit or used>cap:
            self.exhausted = True
            raise QuotaExceeded42('C42 worker hard byte quota before write')
        if receipt:
            self.reserved += amount
        else:
            self.payload += amount

    def owned(self, path):
        path = Path(path)
        require(path.is_absolute() and path.is_relative_to(self.output)
                and not any(p.is_symlink() for p in (path,*path.parents)),
                'C42 quota write must stay in owned non-symlink output')
        return path

    def stage(self, path, write, *, receipt=False):
        self.check_global()
        path = self.owned(path)
        with path.open('xb') as stream:
            write(_QuotaStream42(stream,self,receipt=receipt))
            stream.flush()
            os.fsync(stream.fileno())

    def receipt_json(self, path, value):
        path = self.owned(path)
        payload = (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
        self.charge(len(payload),receipt=True)
        with path.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def open_text(self, path):
        self.check_global()
        return _QuotaText42(self.owned(path).open('xb'), self)

    @contextmanager
    def external_files(self, count):
        """Pinned checkpoint/import repair has at most count output files.

        Reserve count*16MiB first, then kernel-cap every individual file.
        No final or failure save bypasses this scope; quota includes errors.
        """
        self.check_global()
        self.charge(count*EXTERNAL_FILE_LIMIT)
        previous = resource.getrlimit(resource.RLIMIT_FSIZE)
        require(previous[1]==resource.RLIM_INFINITY or previous[1]>=EXTERNAL_FILE_LIMIT,
                'C42 inherited file limit is incompatible')
        def stopped(*_):
            self.exhausted = True
            raise QuotaExceeded42('C42 external checkpoint file exceeds 16 MiB')
        handler = signal.signal(signal.SIGXFSZ,stopped)
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE,(EXTERNAL_FILE_LIMIT,previous[1]))
            yield
        finally:
            resource.setrlimit(resource.RLIMIT_FSIZE,previous)
            signal.signal(signal.SIGXFSZ,handler)

    def report(self):
        return {'schema':'b22-tracking42-worker-byte-budget-v1','limit_bytes':self.limit,
            'initial_bytes':self.initial,'charged_payload_and_external_reservations':self.payload,
            'charged_receipt_bytes_before_final_receipt':self.reserved,
            'receipt_reserve_bytes':self.reserve,'exhausted':self.exhausted,
            'strict_worker_before_write':True,'global_threshold_bytes':self.global_stop,
            'global_threshold_is_monitored_not_atomic':True,
            'checkpoint_file_kernel_limit_bytes':EXTERNAL_FILE_LIMIT}


def quota_writer42(base, quota, jsonable):
    class Rescue42(base):
        def _stage(self, path, write):
            return quota.stage(path,write,receipt=True)
    class Writer42(base):
        def _stage(self, path, write):
            return quota.stage(path,write)
        def commit_json(self, path, value):
            return super().commit_json(path,jsonable(value))
    rescue_globals = dict(base.commit_failure_receipt.__globals__,ArchiveWriter=Rescue42)
    Writer42.commit_failure_receipt = FunctionType(base.commit_failure_receipt.__code__,
        rescue_globals,base.commit_failure_receipt.__name__,base.commit_failure_receipt.__defaults__)
    return Writer42


def reward_env42(base, session):
    """Exactly one transform after the pinned base world-upright reward."""
    helper = load_file('b22_tracking42_reward',ROOT/'scripts/b22_tracking_reward42.py')
    alpha, arm = session['alpha'], session['reward_arm']
    class TrainingRewardEnv42(base):
        reward_schema = REWARD_SCHEMA
        def _reward(self, transition, metrics, terminated):
            original = super()._reward(transition,metrics,terminated)
            self._original_terms42 = dict(original)
            return helper.apply_tracking_deficit42(original,alpha)
        def step(self, action):
            self._original_terms42 = None
            obs,reward,terminated,truncated,info = super().step(action)
            require(self._original_terms42 is not None, 'C42 missing original reward cache')
            info = dict(info)
            info['original_reward_terms42'] = dict(self._original_terms42)
            info['training_reward42'] = {'schema':REWARD_SCHEMA,'alpha':alpha,'reward_arm':arm}
            return obs,reward,terminated,truncated,info
    _reward_base_by_type42[TrainingRewardEnv42] = base
    return TrainingRewardEnv42


def controller_module42():
    name,digest = SOURCES['controller']
    module = importlib.import_module('controller18')
    require(Path(module.__file__).resolve()==ROOT/name and identity(ROOT/name)['sha256']==digest,
            'C42 original controller installer source/origin changed')
    return module


def controller_installer42(original, env_type):
    """Keep the fixed installer body and every exact check; map one owned type."""
    function = original.install_controller18
    require(isinstance(function,FunctionType) and function.__globals__ is original.__dict__,
            'C42 controller installer function has foreign globals')
    production = original.WorldUprightCourseEnv
    if env_type is production:
        installer, changed = function, []  # Evaluation retains the original path.
    else:
        require(_reward_base_by_type42.get(env_type) is production
                and env_type.__bases__==(production,) and env_type.reward_schema==REWARD_SCHEMA,
                'C42 controller installer only accepts its exact registered training env')
        namespace = dict(original.__dict__,WorldUprightCourseEnv=env_type)
        changed = sorted(key for key,value in original.__dict__.items() if namespace[key] is not value)
        require(changed==['WorldUprightCourseEnv'],'C42 controller installer globals widened')
        installer = FunctionType(function.__code__,namespace,function.__name__,
                                 function.__defaults__,function.__closure__)
        installer.__kwdefaults__ = copy.copy(function.__kwdefaults__)
    require(installer.__code__ is function.__code__,'C42 controller installer body changed')
    return installer, {'schema':'b22-tracking42-controller-installer-v1',
        'source':str(ROOT/SOURCES['controller'][0]),'source_sha256':SOURCES['controller'][1],
        'function_code_unchanged':installer.__code__ is function.__code__,
        'globals_changed':changed,'registered_training_env':env_type is not production,
        'old_module_globals_unchanged':original.WorldUprightCourseEnv is production,
        'other_exact_type_and_never_reset_checks_preserved':True}


def install_controller42(env, variant):
    installer,proof = controller_installer42(controller_module42(),type(env))
    controller = installer(env,variant)
    from wheel_legged_control.d1.yaw_feedforward44 import adopt_yaw_feedforward
    require(type(_ff44) is float,'C44 feedforward gain was not set from the session')
    adopt_yaw_feedforward(controller._nominal,_ff44)
    train45.keep_record_consistent(controller,_ff44)
    encoded = (json.dumps(dict(proof,variant=variant,installation_completed=True),
                          sort_keys=True,allow_nan=False)+'\n').encode()
    _quota42.stage(_quota42.output/'controller_installer42_proof.json',lambda stream:stream.write(encoded))
    return controller


class ServoCounter42:
    """Count actual nominal-path advances, excluding live control advances."""
    def __init__(self, servo_type, limit):
        self.servo_type, self.original, self.limit = servo_type, servo_type.advance, limit
        self.attempted = self.returned = self.other_returned = 0
        self.by_caller = {}

    def install(self):
        owner = self
        def advance(instance, *args, **kwargs):
            frame = sys._getframe(1)
            module, name = frame.f_globals.get('__name__'), frame.f_code.co_name
            nominal = ((module == 'b22_curriculum42' and name == '_nominal_path')
                       or (module == 'geometry21' and name == 'nominal_path21'))
            key = str(module)+':'+name
            if nominal:
                require(owner.attempted < owner.limit, 'C42 nominal servo budget exhausted')
                owner.attempted += 1
            result = owner.original(instance, *args, **kwargs)
            if nominal:
                owner.returned += 1
                owner.by_caller[key] = owner.by_caller.get(key, 0)+1
            else:
                owner.other_returned += 1
            return result
        self.wrapper = advance
        self.servo_type.advance = advance
        return self

    def close(self):
        require(self.servo_type.advance is self.wrapper, 'C42 servo counter was replaced')
        self.servo_type.advance = self.original

    def report(self):
        return {'attempted': self.attempted, 'returned': self.returned,
                'by_caller': dict(self.by_caller), 'other_live_servo_returns': self.other_returned,
                'limit': self.limit, 'pure_geometry_only': True}


def install_servo_counter42(session):
    servo = importlib.import_module('full_drive_servo_08')
    globals()['_servo_counter42'] = ServoCounter42(
        servo.FullDriveCommandServo, session['pure_servo_advance_max']).install()


def _dump42(tree):
    return ast.dump(tree,include_attributes=False)


def adapt_module42(kind):
    """Source-only AST derivation; no original module imports or execution."""
    name,digest = SOURCES[kind]
    raw = (ROOT/name).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==digest,'C42 frozen source changed: '+name)
    original = ast.parse(raw); tree = copy.deepcopy(original); edits = []
    def put(parent,field,index,value,label):
        old = getattr(parent,field)[index] if index is not None else getattr(parent,field)
        edits.append((parent,field,index,old,label))
        if index is None:setattr(parent,field,value)
        else:getattr(parent,field)[index]=value
    imports = {'recipes22':'b22_recipes42'}
    if kind=='train':
        imports.update(curriculum22='b22_curriculum42',checkpoint21='b22_checkpoint42',
                       run_short_rl16_training_11='b22_metadata42',runtime_support22='b22_support42')
    if kind=='checkpoint':
        imports['rl16_learning_11']='b22_rl42'
    if kind=='metadata':
        imports.update(world_upright_course_11='b22_world42',short_curriculum_11='b22_curriculum42')
    for node in ast.walk(tree):
        if kind=='support' and isinstance(node,ast.Import) and any(n.name=='curriculum22' for n in node.names):
            require(len(node.names)==1 and node.names[0].asname=='module',
                    'C42 atomic training I/O private curriculum seam changed')
            put(node,'names',None,[ast.alias(name='b22_curriculum42',asname='module')],
                'actual atomic context patches the same private curriculum')
        if isinstance(node,ast.ImportFrom) and node.module in imports:
            put(node,'module',None,imports[node.module],'versioned import '+node.module)
        if kind=='train' and isinstance(node,ast.ImportFrom) and node.module=='rl16_learning_11':
            names=[n for n in node.names if n.name!='save_failure_checkpoint']
            put(node,'names',None,names,'failure save injected through hard quota')
        if kind=='curriculum' and isinstance(node,ast.Assign) and any(
                isinstance(t,ast.Attribute) and t.attr=='_events' for t in node.targets):
            require(isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                    and node.value.func.attr=='open','C42 event output seam changed')
            put(node,'value',None,ast.parse("_quota42.open_text(self.output_dir/'curriculum_events.ndjson')",
                mode='eval').body,'actual event stream before-write quota')
    for node in tree.body:
        if isinstance(node,ast.If) and ast.unparse(node.test)=="__name__ == '__main__'":
            put(tree,'body',tree.body.index(node),ast.Pass(),'new42 owns CLI')
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            target=node.targets[0].id
            if kind=='recipes' and target=='SPEC':
                put(node,'value',None,ast.Name('_spec42',ast.Load()),'fixed original spec + explicit stage2 seed')
            if kind=='heldout' and target=='ACTORS':
                put(node,'value',None,ast.parse("('zero','control','deficit8')",mode='eval').body,'new actors, same recorder')
            if kind=='heldout' and target=='RECORD_SCHEMA':
                put(node,'value',None,ast.Constant('d1-b22-tracking42-heldout-record-v1'),'C42 raw record identity')
    result=copy.deepcopy(tree)
    for parent,field,index,old,_ in reversed(edits):
        if index is None:setattr(parent,field,old)
        else:getattr(parent,field)[index]=old
    require(_dump42(tree)==_dump42(original),'C42 undeclared AST difference')
    return ast.fix_missing_locations(result), {'source':str(ROOT/name),'source_sha256':digest,
        'derived_ast_sha256':hashlib.sha256(_dump42(result).encode()).hexdigest(),
        'inverse_ast_exact':True,'seams':[e[4] for e in edits]}


def _exec_module42(kind, namespace=None):
    tree,proof=adapt_module42(kind)
    module=ModuleType('b22_'+kind+'42');module.__file__=str(WORKER)
    module.__dict__.update(namespace or {})
    sys.modules[module.__name__]=module
    exec(compile(tree,str(WORKER),'exec'),module.__dict__)  # noqa: S102 -- full-SHA pinned inverse-checked AST
    return module,proof


def strict_module42(original):
    module=ModuleType('b22_rl42')
    namespace=module.__dict__
    namespace.update(vars(original))
    namespace.update(__name__='b22_rl42',__file__=str(WORKER),
        WORLD_UPRIGHT_REWARD_SCHEMA=REWARD_SCHEMA,FINAL_CHECKPOINT_SCHEMA=FINAL_SCHEMA)
    for key,value in vars(original).items():
        if isinstance(value,FunctionType) and value.__globals__ is original.__dict__:
            clone=FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            clone.__kwdefaults__=copy.copy(value.__kwdefaults__)
            namespace[key]=clone
    # All cloned functions share this module's exact globals mapping, including
    # recursive strict-loader helpers and metadata validators.
    sys.modules['b22_rl42']=module
    return module


def runtime_modules42(env_type, session, quota):
    spec=read_json(SPEC_PATH);spec['training']['ppo_seed_by_stage']=[201001,201011]
    recipes,_=_exec_module42('recipes',{'_spec42':spec})
    train45.apply_curriculum(recipes,session.get('curriculum') or 'original')
    original=importlib.import_module('rl16_learning_11')
    strict=strict_module42(original)
    world=ModuleType('b22_world42');world.__file__=str(WORKER)
    old_world=importlib.import_module('world_upright_course_11')
    world.__dict__.update(vars(old_world));world.__name__='b22_world42';world.__file__=str(WORKER)
    world.WORLD_UPRIGHT_REWARD_SCHEMA=REWARD_SCHEMA;world.WorldUprightCourseEnv=env_type
    sys.modules['b22_world42']=world
    curriculum,_=_exec_module42('curriculum',{'_quota42':quota})
    curriculum.WorldUprightCourseEnv=env_type
    curriculum.WORLD_UPRIGHT_REWARD_SCHEMA=REWARD_SCHEMA
    _exec_module42('support')
    _exec_module42('metadata')
    checkpoint,_=_exec_module42('checkpoint')
    final_save=checkpoint.save_final_and_verify
    def save_final(*args,**kwargs):
        with quota.external_files(4):
            return final_save(*args,**kwargs)
    checkpoint.save_final_and_verify=save_final
    def save_failure(*args,**kwargs):
        with quota.external_files(2):
            return strict.save_failure_checkpoint(*args,**kwargs)
    train,_=_exec_module42('train',{'save_failure_checkpoint':save_failure})
    original_metadata=train.train.__globals__
    # The legacy train function writes these actual training provenance fields
    # via the metadata producer, not by rewriting old records.
    metadata_module=sys.modules['b22_metadata42']
    metadata_fn=metadata_module._training_metadata
    def metadata(*args,**kwargs):
        value=metadata_fn(*args,**kwargs)
        value['tracking_reward42']={'schema':REWARD_SCHEMA,'alpha':session['alpha'],
                                    'reward_arm':session['reward_arm']}
        value['execution_contract_id']=CONTRACT
        value['yaw_feedforward44']={'feedforward':session.get('yaw_feedforward'),'curriculum':session.get('curriculum')}
        return value
    metadata_module._training_metadata=metadata
    require(original_metadata is train.__dict__,'C42 training globals are foreign')
    heldout,_=_exec_module42('heldout')
    return recipes,strict,train,heldout


def verify_training_io42():
    """Enter the real lazy bridge without training, arrays, or archive writes."""
    curriculum = sys.modules['b22_curriculum42']
    support = sys.modules['b22_support42']
    names = ('_NumericBlocks','_write_exclusive_json','_write_exclusive_npz')
    original = {name:getattr(curriculum,name) for name in names}
    writes = []
    def forbidden(*_args,**_kwargs):
        writes.append('attempted')
        raise RuntimeError('C42 bootstrap training-I/O writer call forbidden')
    writer = SimpleNamespace(commit_json=forbidden,commit_npz=forbidden)
    with support.atomic_training_io(writer):
        require(curriculum._NumericBlocks is not original['_NumericBlocks']
                and curriculum._NumericBlocks.__bases__==(original['_NumericBlocks'],)
                and curriculum._write_exclusive_json is forbidden
                and curriculum._write_exclusive_npz is not original['_write_exclusive_npz'],
                'C42 atomic training context did not bind its actual private curriculum')
    require(all(getattr(curriculum,name) is value for name,value in original.items()) and not writes,
            'C42 atomic training context did not restore or performed a write')
    return {'schema':'b22-tracking42-atomic-training-io-proof-v1',
        'source_sha256':SOURCES['support'][1],'private_curriculum':curriculum.__name__,
        'entered_and_exited':True,'original_references_restored':True,'archive_writer_calls':len(writes)}


def evaluate42(session,runtime,guard,env,output,writer,counter):
    recipes,strict,_train,heldout=runtime_modules42(type(env),session,_quota42)
    policies,hashes={},{}
    for actor in ('control','deficit8'):
        manifest=session['eval_manifests'][actor]
        counter.phase='probe'
        policies[actor],proof=strict.load_and_verify_final(manifest['folder'],manifest,return_model=True)
        hashes[actor]=manifest['files']['final_model.zip']['sha256']
        writer.commit_json(output/(actor+'_load.json'),proof)
    construction = read_json(output/'construction_receipt.json')
    writer.commit_json(output/'eval_identity.json', {
        'source_hashes': session['source_hashes'], 'execution_stage': 42,
        'execution_contract_id': CONTRACT, 'controller_variant': 'combined',
        'evaluation_split': session['evaluation_split'],
        'checkpoint_sha256_by_actor': hashes,
        'checkpoint_paths_by_actor': {actor:str(Path(session['eval_manifests'][actor]['folder'])/'final_model.zip')
                                     for actor in policies},
        'actual_geometry_binding': construction['geometry_binding'],
        'compiled_geometry': construction['compiled_geometry'], 'warnings': [],
        'evaluation_reward_unchanged': True, 'evaluation_gain': 1})
    floors,cases={},[]
    (output/'floor').mkdir(exist_ok=False)
    for actor in ('control','deficit8'):
        counter.phase='floor:'+actor
        receipt,_=heldout.record_case(runtime,guard,env,policies[actor],
            recipes.floor_schedule(session['floor_seed']),actor,output/'floor'/actor,
            writer=writer,pair_initial=None,checkpoint_sha256=hashes[actor])
        require(receipt['record_valid'],'C42 floor software/archive invalid')
        floors[actor]={'receipt':receipt,'numeric_gate_pending_independent_reader':True}
    (output/'heldout').mkdir(exist_ok=False)
    for case in recipes.ORDER:
        paired=None
        for actor in ('zero','control','deficit8'):
            counter.phase='heldout:'+case+':'+actor
            receipt,initial=heldout.record_case(runtime,guard,env,policies.get(actor),
                recipes.make_schedule(case),actor,output/'heldout'/(case+'_'+actor),
                writer=writer,pair_initial=paired,checkpoint_sha256=hashes.get(actor))
            require(receipt['record_valid'],'C42 eval software/archive invalid')
            if paired is None:paired=initial
            cases.append({k:receipt[k] for k in ('case_id','actor','seed','completed_controls',
                'record_valid','terminated','truncated','stop_reason','policy_predictions','checkpoint_sha256')})
    return {'floors':floors,'enabled_arms':['control','deficit8'],'cases':cases,
        'completed_controls':runtime.control_completed,'full_comparison_executed':len(cases)==18,
        'qualification_pending_independent_readback':True,'evaluation_reward_unchanged':True,'evaluation_gain':1}


def train42(session,runtime,guard,env,output,writer,counter):
    _,_,train,_=runtime_modules42(type(env),session,_quota42)
    return train.train(session,runtime,guard,env,output,writer,counter)


def verify_mujoco42():
    helper=package42()
    helper.verify_loaded_roots39(helper.inventory39())


def import_repair42(common,session,output):
    with _quota42.external_files(1):
        return common._load_import_repair(session,output)


def final_receipt42(value,session,quota):
    value.update(schema='b22-tracking42-worker-receipt-v1',execution_contract_id=CONTRACT,
        mode=session['mode'],reward_arm=session['reward_arm'],alpha=session['alpha'],
        quota42=quota.report())
    complete=(value['execution_complete'] and not value['stop_latched'] and not quota.exhausted)
    counts=None
    if isinstance(value.get('python'),dict) and isinstance(value.get('C_final'),dict):
        n=value['python'].get('control_completed')
        native=value['python'].get('native_returned')
        compiler=value['C_final'].get('construction_returns')
        if all(type(x) is int for x in (n,native,compiler)):
            counts={'controls':n,'normal_native':native,'compiler_native':compiler}
    servo = globals().get('_servo_counter42')
    servo_report = None if servo is None else servo.report()
    pure = (servo.returned if servo is not None and servo.attempted == servo.returned else None)
    if complete:
        try:
            require(counts is not None and counts['normal_native']==5*counts['controls']
                    and counts['compiler_native']==2,'C42 complete physical counts absent')
            require(session['mode']!='train' or counts['controls']==32768,'C42 incomplete training')
            require(counts['controls']<=session['controls'],'C42 control cap exceeded')
            require(type(pure) is int and pure<=session['pure_servo_advance_max'],
                    'C42 nominal servo ledger absent/incomplete')
        except ValueError as error:
            complete = False
            value['failure'] = {'type': type(error).__name__, 'message': str(error)}
    value.update(status='COMPLETE' if complete else 'INCOMPLETE',execution_complete=complete,
        actual_counts=counts if complete else None,known_partial_counts=counts,
        pure_servo42=servo_report,actual_pure_servo_advances=pure,
        known_pure_servo_returns=None if servo is None else servo.returned)
    return value


def worker_ast42():
    """Preserve the physical owner under full inverse-AST lock, not just labels."""
    tree,base_proof=adapt_module42('worker')
    original=copy.deepcopy(tree)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    edits=[]
    def put(parent,field,index,new,label):
        old=getattr(parent,field)[index] if index is not None else getattr(parent,field)
        edits.append((parent,field,index,old,label))
        if index is None:setattr(parent,field,new)
        else:getattr(parent,field)[index]=new
    def insert(parent,field,index,new,label):
        getattr(parent,field).insert(index,new)
        edits.append((parent,field,index,None,label))
    put(main,'body',None,ast.parse('sessionpath=_session_path42\nsession=_session42\noutput=_output42').body
        +main.body[5:],'new live-parent validated CLI/session')
    for node in ast.walk(main):
        if isinstance(node,ast.Try):
            for j,child in enumerate(list(node.body)):
                if isinstance(child,ast.If) and ast.unparse(child.test)=="session['mode'] == 'train'":
                    put(node,'body',j,ast.Pass(),'C42 numeric gate replaces old stage1-only reservation')
    # Walk each list backwards: insertion never invalidates a pending index.
    for parent in list(ast.walk(main)):
        for field,value in list(ast.iter_fields(parent)):
            children=list(reversed(list(enumerate(value)))) if isinstance(value,list) else [(None,value)]
            for index,child in children:
                if isinstance(child,ast.If) and 'session.get(' in ast.unparse(child.test) and 'execution_contract_id' in ast.unparse(child.test):
                    put(parent,field,index,ast.parse('validate_numeric_session42(session)').body[0],'C42 exact numeric/session gate')
                elif isinstance(child,ast.ClassDef) and child.name=='JsonWriter':
                    put(parent,field,index,ast.parse('JsonWriter=quota_writer42(ArchiveWriter,_quota42,common._jsonable)').body[0],'actual archive before-write quota')
                elif isinstance(child,ast.ImportFrom) and child.module=='train22':
                    put(parent,field,index,ast.parse('train=train42').body[0],'new training adapter')
                elif isinstance(child,ast.ImportFrom) and child.module=='eval22':
                    put(parent,field,index,ast.parse('evaluate=evaluate42').body[0],'both final/zero evaluation adapter')
                elif isinstance(child,ast.ImportFrom) and child.module=='controller18':
                    require([n.name for n in child.names]==['install_controller18'],
                            'C42 controller installer import seam changed')
                    put(parent,field,index,ast.parse('install_controller18=install_controller42').body[0],
                        'private exact registered reward-env installer globals; original body retained')
                elif isinstance(child,ast.Expr) and ast.unparse(child)=="common._load_import_repair(session, output)":
                    put(parent,field,index,ast.parse('import_repair42(common,session,output)').body[0],'bounded import-repair output')
                elif isinstance(child,ast.Import) and any(n.name=='mujoco' for n in child.names):
                    insert(parent,field,index+1,ast.parse('verify_mujoco42()').body[0],'actual full MuJoCo roots proof')
                elif isinstance(child,ast.ImportFrom) and child.module=='world_upright_course_11':
                    insert(parent,field,index+1,ast.parse("WorldUprightCourseEnv=reward_env42(WorldUprightCourseEnv,session) if session['mode']=='train' else WorldUprightCourseEnv").body[0],'train-only reward environment')
                    insert(parent,field,index+2,ast.parse('install_servo_counter42(session)').body[0],'actual nominal servo counter')
                elif isinstance(child,ast.Constant) and child.value=='C22_fixed_B_eval_repair_v1':
                    put(parent,field,index,ast.Constant(CONTRACT),'new execution identity')
                elif isinstance(child,ast.Constant) and child.value=='d1-coverage-worker-20-v1':
                    put(parent,field,index,ast.Constant('b22-tracking42-worker-receipt-v1'),'new receipt schema')
    # INT/XCPU use the same stop latch and bounded cleanup as TERM/ALRM.
    at=next(i for i,n in enumerate(main.body) if ast.unparse(n)=='signal.signal(signal.SIGALRM, signal_stop)')
    insert(main,'body',at+1,ast.parse('signal.signal(signal.SIGINT,signal_stop)').body[0],'INT stop latch')
    insert(main,'body',at+2,ast.parse('signal.signal(signal.SIGXCPU,signal_stop)').body[0],'CPU stop latch')
    exacts=[n for n in ast.walk(main) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
            and n.targets[0].id=='exact' and isinstance(n.value,ast.Dict)]
    require(len(exacts)==2,'C42 model ledger seam changed')
    for j,(key,value) in enumerate(zip(exacts[1].value.keys,exacts[1].value.values)):
        if key.value in ('load','torch_load'):
            expected,new={'load':(3,2),'torch_load':(9,6)}[key.value]
            require(value.value==expected,'C42 exact eval loading count seam changed')
            put(exacts[1].value,'values',j,ast.Constant(new),'two new final loads only')
        elif key.value=='predict':
            require(isinstance(value,ast.BinOp) and isinstance(value.left,ast.BinOp)
                    and value.left.left.value==3,'C42 eval probe count seam changed')
            put(value.left,'left',None,ast.Constant(2),'two actual final probes')
    finally_node=next(n for n in main.body if isinstance(n,ast.Try))
    at=next(i for i,n in enumerate(finally_node.finalbody)
            if isinstance(n,ast.Expr) and ast.unparse(n).startswith("write(output / 'worker_receipt.json'"))
    insert(finally_node,'finalbody',at,ast.parse('receipt=final_receipt42(receipt,session,_quota42)').body[0],'honest final counts/quota envelope')
    result=copy.deepcopy(tree)
    for parent,field,index,old,_ in reversed(edits):
        if old is None:getattr(parent,field).pop(index)
        elif index is None:setattr(parent,field,old)
        else:getattr(parent,field)[index]=old
    require(_dump42(tree)==_dump42(original),'C42 physical owner changed outside declared seams')
    proof={**base_proof,'schema':'b22-tracking42-worker-derivation-v1','inverse_owner_ast_exact':True,
        'adapted_owner_ast_sha256':hashlib.sha256(_dump42(result).encode()).hexdigest(),
        'declared_seams':[e[4] for e in edits],
        'namespace_injections':['validated session/path/output','quota receipts','train/evaluate42']}
    return ast.fix_missing_locations(result),proof


ZERO_NATIVE42 = ('construction_attempts','construction_returns','control_attempts','control_returns',
    'ccd_attempts','ccd_returns','violations','construction_limit','control_limit','target_model',
    'target_data','armed','phase','first_ccd_caller','first_construction_step_caller','first_control_step_caller')


def zero_native42(state):
    require(isinstance(state,dict) and all(type(state.get(k)) is int and state[k]==0 for k in ZERO_NATIVE42),
            'C42 bootstrap native counter/target/phase changed')


def bootstrap_components42(session, output):
    # Definitions/imports only. EngineGuard is never armed and has no model.
    # Keep the reviewed repair's actual "before EngineGuard" ordering. The
    # preloaded native counters are not reset by EngineGuard.__init__.
    common = importlib.import_module('scripts.run_d1_latest_rl')
    import_repair42(common,session,output)
    binding = importlib.import_module('engine_binding')
    require(str(Path(binding.__file__).resolve())==session['binding_module'] and
            identity(binding.__file__)==session['source_hashes'][session['binding_module']],
            'C42 bootstrap engine-binding origin changed')
    guard = binding.EngineGuard(session['library'])
    before = guard.snapshot()
    zero_native42(before)
    importlib.import_module('mujoco._functions')
    importlib.import_module('mujoco._rollout')
    verify_mujoco42()
    base = importlib.import_module('world_upright_course_11').WorldUprightCourseEnv
    geometry = importlib.import_module('wheel_legged_control.d1.wheel_leg_controller')._nominal_geometry
    counter = importlib.import_module('runtime_support22').ModelCounter
    return guard,before,base,geometry,counter


def constructor_targets42(base):
    # These imports belong to the reviewed dependency-import boundary, before
    # the instrumented adapter-definition block; they create no policy/env.
    return {'environment':base,
        'ppo':importlib.import_module('stable_baselines3').PPO,
        'base_algorithm':importlib.import_module('stable_baselines3.common.base_class').BaseAlgorithm,
        'nn_module':importlib.import_module('torch.nn').Module}


class ConstructionGuard42:
    """Reject and latch construction attempts inside adapter definitions."""
    def __init__(self, targets):
        self.targets=targets
        self.attempted=dict.fromkeys(targets,0)
        self.previous=[]

    def __enter__(self):
        for name,kind in self.targets.items():
            own='__init__' in kind.__dict__
            previous=kind.__dict__.get('__init__')
            def denied(*_args, _name=name, **_kwargs):
                self.attempted[_name]+=1
                raise RuntimeError('C42 bootstrap constructor forbidden: '+_name)
            kind.__init__=denied
            self.previous.append((kind,own,previous))
        return self

    def __exit__(self, *_):
        for kind,own,previous in reversed(self.previous):
            if own:kind.__init__=previous
            else:delattr(kind,'__init__')

    def report(self):
        return {'scope':'adapter definitions after reviewed dependency imports',
                'attempted':dict(self.attempted)}


def bootstrap42(session, output, quota):
    """Real cold import connectivity, zero NN/env/geometry/native calls."""
    failure, cleanup, proof = None, [], {}
    counter = servo = constructors = None
    previous = {}
    def stopped(*_):
        raise TimeoutError('C42 bootstrap stop/deadline')
    for number in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM,signal.SIGXCPU):
        previous[number] = signal.signal(number,stopped)
    signal.setitimer(signal.ITIMER_REAL,max(.01,session['host_started_monotonic']+session['soft_s']-__import__('time').monotonic()))
    try:
        guard,post_dependency_imports,base,geometry,counter_type = bootstrap_components42(session,output)
        cold = geometry.cache_info()._asdict()
        require(cold=={'hits':0,'misses':0,'maxsize':1,'currsize':0}, 'C42 bootstrap geometry cache not cold')
        # binding_proof resolves the original_step RTLD_NEXT pointer without
        # calling it. Establish the baseline after this reviewed lazy lookup.
        initial_binding = guard.binding_proof()
        require(initial_binding.get('passed') is True,'C42 initial ELF/GOT binding failed')
        zero_native42(initial_binding.get('counters'))
        before = guard.snapshot()
        zero_native42(before)
        install_servo_counter42(session)
        servo = globals()['_servo_counter42']
        counter = counter_type(False,session['model_limits'])
        constructors = ConstructionGuard42(constructor_targets42(base))
        # ModelCounter enters before any derived class/function graph is built.
        with counter, constructors:
            example = {**session,'mode':'train','reward_arm':'ffrough','alpha':train45.ALPHA,'curriculum':'rough_emphasis'}
            env_type = reward_env42(base,example)  # class definition; no ()
            _,controller_proof = controller_installer42(controller_module42(),env_type)
            recipes,strict,train,heldout = runtime_modules42(env_type,example,quota)
            training_io_proof = verify_training_io42()
            require(strict.load_and_verify_final.__globals__ is strict.__dict__ and
                    strict.FINAL_CHECKPOINT_SCHEMA==FINAL_SCHEMA and
                    strict.WORLD_UPRIGHT_REWARD_SCHEMA==REWARD_SCHEMA,
                    'C42 strict-loader globals/schema are not actually connected')
            require(sys.modules['b22_curriculum42'].WorldUprightCourseEnv is env_type and
                    sys.modules['b22_curriculum42'].WORLD_UPRIGHT_REWARD_SCHEMA==REWARD_SCHEMA and
                    recipes.T['ppo_seed_by_stage'][1]==201011 and
                    heldout.ACTORS==('zero','control','deficit8') and
                    train.train.__globals__ is train.__dict__, 'C42 actual adapter bindings differ')
        require(not any(constructors.attempted.values()),'C42 bootstrap attempted environment/policy construction')
        require(all(all(row[k]==0 for k in ('attempted','returned','rows_attempted','rows_returned'))
                    for row in counter.counts.values()),'C42 bootstrap invoked a counted model API')
        require(servo.attempted==servo.returned==servo.other_returned==0,
                'C42 bootstrap advanced servo geometry/control')
        binding = guard.binding_proof()
        after = guard.snapshot()
        for state in (post_dependency_imports,before,after,binding.get('counters')):zero_native42(state)
        require(before==after and geometry.cache_info()._asdict()==cold,
                'C42 bootstrap definitions changed native state or geometry cache')
        require(binding.get('passed') is True and binding.get('dso_path')==session['library'] and
                binding.get('dso_sha256')==session['source_hashes'][session['library']]['sha256'] and
                len(binding.get('jump_slots',()))==4 and all(
                    row.get('passed') is True and row.get('target')==row.get('expected_wrapper') and
                    row.get('elf_sha256')==session['source_hashes'][row['elf']]['sha256']
                    for row in binding['jump_slots']) and all(
                    session['source_hashes'][name]['sha256']==digest for name,digest in binding['elf_hashes'].items()),
                'C42 actual ELF/GOT bootstrap proof failed')
        proof = {'schema':'b22-tracking42-bootstrap-proof-v1','execution_contract_id':CONTRACT,
            'session_identity':identity(Path(output)/'session.json'),
            'state_after_dependency_imports':post_dependency_imports,
            'state_before_adapters':before,'state_after':after,'binding_proof':binding,
            'initial_binding_proof':initial_binding,
            'nominal_cache':cold,'pure_servo42':servo.report(),'model_calls':counter.summary(),
            'construction_guard42':constructors.report(),
            'controller_installer42':controller_proof,
            'atomic_training_io42':training_io_proof,
            'derived_modules':{kind:adapt_module42(kind)[1] for kind in
                               ('recipes','curriculum','metadata','checkpoint','train','heldout','support')},
            'actual_module_files':{name:module.__file__ for name,module in sys.modules.items()
                                   if name in ('b22_recipes42','b22_curriculum42','b22_metadata42',
                                               'b22_checkpoint42','b22_train42','b22_heldout42','b22_rl42','b22_support42')},
            'environment_constructions':constructors.attempted['environment'],
            'policy_construction_attempts':sum(constructors.attempted[k] for k in ('ppo','base_algorithm','nn_module')),
            'qualification_granted':False,
            'scope':'source-reviewed dependency imports; instrumented adapter definitions only; original_step is a resolved pointer'}
        encoded=(json.dumps(proof,sort_keys=True,allow_nan=False)+'\n').encode()
        quota.stage(Path(output)/'bootstrap42_proof.json',lambda stream:stream.write(encoded))
    except BaseException as error:  # noqa: BLE001 -- owned stop/interrupt must write an incomplete receipt
        failure={'type':type(error).__name__,'message':str(error)}
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for number in previous:signal.signal(number,signal.SIG_IGN)
        if servo is not None:
            try:servo.close()
            except BaseException as error:cleanup.append(str(error))  # noqa: BLE001 -- cleanup must invalidate success
            globals().pop('_servo_counter42',None)
        for number,handler in previous.items():signal.signal(number,handler)
    complete = failure is None and not cleanup and not quota.exhausted
    value={'schema':'b22-tracking42-worker-receipt-v1','execution_contract_id':CONTRACT,
        'session_identity':identity(Path(output)/'session.json'),'status':'COMPLETE' if complete else 'INCOMPLETE',
        'failure':failure,'cleanup_errors':cleanup,'bootstrap_verified42':complete,
        'actual_counts':dict.fromkeys(('controls','normal_native','compiler_native'),0) if complete else None,
        'known_partial_counts':None,'actual_pure_servo_advances':0 if complete else None,
        'known_pure_servo_returns':None if servo is None else servo.returned,
        'quota42':quota.report(),'model_calls':None if counter is None else counter.summary(),
        'construction_guard42':None if constructors is None else constructors.report(),
        'result':{'bootstrap_proof_identity':identity(Path(output)/'bootstrap42_proof.json')} if complete else {}}
    quota.receipt_json(Path(output)/'worker_receipt.json',value)
    return 0 if complete else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bootstrap',action='store_true')
    args=parser.parse_args(argv)
    parent=load_file('b22_tracking42_parent_gate',ROOT/'scripts/run_b22_roughtrain45.py')
    session=parent.validate_worker_launch42(args.session,args.output)
    validate_numeric_session42(session)
    globals()['_ff44']=session.get('yaw_feedforward')
    require(args.bootstrap is (session['mode']=='bootstrap'),'C42 bootstrap flag/stage mismatch')
    quota=StageQuota42(args.output,session['quota_bytes'],execution_output=session['execution_output'],
                       global_stop=session['output_quota_bytes'])
    globals()['_quota42']=quota
    if args.bootstrap:
        return bootstrap42(session,args.output,quota)
    tree,proof=worker_ast42()
    namespace=dict(globals(),__name__='b22_tracking42_physical_owner',__file__=str(WORKER),
        _session_path42=args.session,_session42=session,_output42=args.output)
    exec(compile(tree,str(WORKER),'exec'),namespace)  # noqa: S102 -- pinned original owner with explicit C42 seams
    namespace['write']=quota.receipt_json
    namespace['_quota42']=quota
    proof['modules'] = {kind:adapt_module42(kind)[1] for kind in
                        ('recipes','curriculum','checkpoint','heldout','train','metadata','support')}
    proof['strict_namespace42'] = {'source_sha256':SOURCES['strict'][1],
        'WORLD_UPRIGHT_REWARD_SCHEMA':REWARD_SCHEMA, 'FINAL_CHECKPOINT_SCHEMA':FINAL_SCHEMA,
        'shared_function_globals':True}
    quota.receipt_json(args.output/'adapter42_proof.json',proof)
    try:
        return namespace['main']()
    finally:
        servo = globals().pop('_servo_counter42',None)
        if servo is not None:
            servo.close()


if __name__=='__main__':
    raise SystemExit(main())
