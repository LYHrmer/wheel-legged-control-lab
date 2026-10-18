"""Bounded, serial C42 parent. The default describes a plan without executing it."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'C42_B22_tracking_deficit_comparison_v1'
OUTPUT = ROOT/'results/b22_tracking_reward42_20261018_02'
RELEASE = ROOT/'results/b22_tracking_reward42_plan_20261018_02.json'
ADMISSIONS = ROOT/'results/b22_tracking_reward42_admissions_20261018_02'
UNUSED_RELEASE = ROOT/'results/b22_tracking_reward42_plan_20261018_01.json'
UNUSED_OUTPUT = ROOT/'results/b22_tracking_reward42_20261018_01'
UNUSED_RELEASE_SHA = '7128e72d0fcfde9c0322d185e188f0648ef350c90911705fdb52e5fc8de597f0'
UNUSED_COMMIT = '4c55bbb2baeca84dc3db897ffeaeb07329faa38e'
RESOURCE_REVISION = 'owned_cgroup_v2'
DRIVER = ROOT/'scripts/run_b22_tracking_reward42.py'
WORKER = ROOT/'scripts/worker_b22_tracking_reward42.py'
READER = ROOT/'scripts/read_b22_tracking_reward42.py'
MANIFEST = ROOT/'runtime/d1_b22/runtime_manifest_v27.json'
MANIFEST_SHA = 'd3c38429362e82745179f2c526b2ee52d805526df390d52346c805e5a51fff09'
AUTHORED = (
    'scripts/b22_tracking_reward42.py', 'tests/test_b22_tracking_reward42.py',
    'docs/b22_tracking_reward42.md', 'scripts/run_b22_tracking_reward42.py',
    'tests/test_b22_tracking_reward42_parent.py', 'scripts/worker_b22_tracking_reward42.py',
    'tests/test_b22_tracking_reward42_worker.py', 'scripts/read_b22_tracking_reward42.py',
    'tests/test_b22_tracking_reward42_reader.py',
    'scripts/resource_guard42.py', 'tests/test_resource_guard42.py',
)
PINS = {
    'scripts/run_b22_suite31.py': 'f783b4d7e8380c94bd7a4755e73434179cb5c3f6a95bbcca336e235a43320c2e',
    'scripts/b22_suite31_contract.py': '21386cfb3b73f0d947f825f00dd367c7443f3ca29738dfec5768925f30070203',
}
LIMITS = {'controls': 96136, 'normal_native': 480680, 'compiler_native': 6,
              'worker_pure_servo_advances': 350600, 'reader_pure_servo_advances': 190600,
              'combined_pure_servo_advances': 541200,
              'total_wall_s': 5400, 'start_effective_bytes': 6 << 30,
              'stable_effective_bytes': 6 << 30, 'reserve_effective_bytes': 4 << 30,
              'stable_seconds': 10, 'admission_poll_s': 5, 'admission_max_sample_gap_s': 7.5,
              'admission_minimum_samples': 3, 'admission_max_s': 300,
              'start_free_disk_bytes': 10 << 30, 'reserve_free_disk_bytes': 2 << 30,
              'output_quota_bytes': 8 << 30, 'required_minimum_nice': 10}
COUNT_KEYS = ('controls', 'normal_native', 'compiler_native')
RESOURCE_POLICY = {'memory_high_bytes': 2048 << 20, 'memory_max_bytes': 3072 << 20,
                   'memory_swap_max_bytes': 0, 'oom_policy': 'kill',
                   'memory_oom_group': 1,
                   'unit': 'wheel-b22-tracking42-owned02-20261018-01.service',
                   'owned_scope': '/user.slice/user-1000.slice/user@1000.service/app.slice/wheel-b22-tracking42-owned02-20261018-01.service',
                   'reserve_scope': 'host and ancestors outside the owned job cgroup',
                   'covers': 'parent, source hashing, bootstrap, workers, saved readers, comparison'}
_resource_guard42 = None


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(),
            'C42 requires an absolute regular input: '+str(path))
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON '+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write_json(path, value, *, replace=False):
    path = Path(path)
    target = path.with_name(path.name+'.tmp.'+secrets.token_hex(12)) if replace else path
    try:
        with target.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(target, path)
    finally:
        if replace:
            target.unlink(missing_ok=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def supervision():
    for name, expected in PINS.items():
        require(identity(ROOT/name)['sha256'] == expected, 'C42 supervision source changed: '+name)
    return load_module('_b22_tracking42_supervision31', ROOT/'scripts/run_b22_suite31.py')


def worker_api():
    return load_module('_b22_tracking42_worker_api', WORKER)


def reader_api():
    return load_module('_b22_tracking42_reader_api', READER)


def resource_guard_api42():
    global _resource_guard42
    if _resource_guard42 is None:
        _resource_guard42 = load_module('_b22_tracking42_resource_guard', ROOT/'scripts/resource_guard42.py')
    return _resource_guard42


def resource_snapshot42():
    guard = resource_guard_api42()
    return guard.snapshot_memory(RESOURCE_POLICY['owned_scope'])


def check_resource_snapshot42(snapshot):
    resource_guard_api42().check_memory(snapshot)
    available = snapshot.get('effective_available_bytes')
    require(type(available) is int and available >= LIMITS['reserve_effective_bytes'],
            'C42 host/outside-ancestor RAM below 4 GiB runtime reserve or unknown')
    return available


def unused_budget_origin42():
    require(not UNUSED_OUTPUT.exists() and not UNUSED_OUTPUT.is_symlink(),
            'C42 resource revision cannot reuse a previously claimed original budget')
    entry = identity(UNUSED_RELEASE)
    require(entry['sha256'] == UNUSED_RELEASE_SHA, 'C42 unexecuted original release changed')
    value = read_json(UNUSED_RELEASE)
    require(value.get('execution_contract_id') == CONTRACT
            and value.get('source', {}).get('git_commit') == UNUSED_COMMIT
            and value.get('plan', {}).get('output') == str(UNUSED_OUTPUT)
            and all(value['plan']['limits'][key] == LIMITS[key] for key in COUNT_KEYS),
            'C42 original scientific budget identity differs')
    return {'release': str(UNUSED_RELEASE), 'identity': entry, 'source_commit': UNUSED_COMMIT,
            'original_output': str(UNUSED_OUTPUT), 'original_execution_started': False,
            'same_unused_scientific_budget': True}


def stages42():
    return [{'name': 'bootstrap', 'mode': 'bootstrap', 'reward_arm': None, 'alpha': None,
             'controls': 0, 'normal_native': 0, 'compiler_native': 0,
             'pure_servo_advance_max': 0, 'reader_pure_servo_advance_max': 0,
             'quota_bytes': 64 << 20, 'soft_s': 240, 'close_s': 275, 'hard_s': 300,
             'reader_wall_s': 0}] + [
        {'name': 'train_'+arm, 'mode': 'train', 'reward_arm': arm, 'alpha': alpha,
             'controls': 32768, 'normal_native': 163840, 'compiler_native': 2,
             'pure_servo_advance_max': 160000,
             'reader_pure_servo_advance_max': 80000,
             'quota_bytes': 3 << 30, 'soft_s': 1140, 'close_s': 1180, 'hard_s': 1200,
             'reader_wall_s': 600}
        for arm, alpha in (('control', 1), ('deficit8', 8))
    ] + [{'name': 'eval', 'mode': 'eval', 'reward_arm': None, 'alpha': None,
              'controls': 30600, 'normal_native': 153000, 'compiler_native': 2,
              'pure_servo_advance_max': 30600,
              'reader_pure_servo_advance_max': 30600,
              'quota_bytes': 2 << 30, 'soft_s': 1140, 'close_s': 1180, 'hard_s': 1200,
              'reader_wall_s': 600}]


def describe42():
    return {'schema': 'b22-tracking42-plan-v1', 'status': 'PLAN_ONLY',
            'execution_contract_id': CONTRACT, 'output': str(OUTPUT),
            'resource_revision': RESOURCE_REVISION, 'resource_policy': dict(RESOURCE_POLICY),
            'stages': stages42(), 'limits': dict(LIMITS),
            'actors': ['zero', 'control', 'deficit8'], 'evaluation_gain': 1,
            'partial_eligible': False, 'retry_permitted': False,
            'automatic_model_replacement': False, 'execution_started': False,
            'selection': 'both complete finals; original six gates; floors do not screen arms'}


def source_identity42():
    subprocess.run(['git', 'ls-files', '--error-unmatch', *AUTHORED], cwd=ROOT,
                   check=True, stdout=subprocess.DEVNULL, timeout=10)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *AUTHORED], cwd=ROOT,
                   check=True, timeout=10)
    return {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                  text=True, timeout=10).strip(),
            'authored': {str(ROOT/name): identity(ROOT/name) for name in AUTHORED}}


def inputs42(source):
    supervision()
    require(identity(MANIFEST)['sha256'] == MANIFEST_SHA, 'C42 original v27 manifest changed')
    config = read_json(MANIFEST)
    extra = worker_api().required_inputs42()
    reader_inputs = reader_api().required_inputs42()
    require(isinstance(extra, dict) and isinstance(reader_inputs, dict),
            'C42 worker/reader input closures must be identity maps')
    inputs = dict(config['source_hashes'])
    for mapping in (source['authored'], extra, reader_inputs,
                    {str(ROOT/name): identity(ROOT/name) for name in PINS},
                    {str(MANIFEST): identity(MANIFEST), str(UNUSED_RELEASE): identity(UNUSED_RELEASE)}):
        for name, entry in mapping.items():
            require(name not in inputs or inputs[name] == entry, 'C42 conflicting frozen identity: '+name)
            inputs[name] = entry
    return inputs


def validate_release42(path):
    path = Path(path)
    require(path == RELEASE, 'C42 fixed independent release path required')
    value = read_json(path)
    require(value.get('schema') == 'b22-tracking42-release-v1'
            and value.get('execution_contract_id') == CONTRACT
            and value.get('resource_revision') == RESOURCE_REVISION
            and value.get('unused_budget_origin') == unused_budget_origin42()
            and value.get('plan') == describe42(), 'C42 release scope differs')
    require(value['source'] == source_identity42(), 'C42 HEAD/authored bytes changed')
    require(value['inputs'] == inputs42(value['source']), 'C42 input closure differs')
    for name, expected in value['inputs'].items():
        require(identity(name) == expected, 'C42 frozen input changed: '+name)
    return value


def prepare42():
    require(not RELEASE.exists() and not OUTPUT.exists(), 'C42 release/output already exists')
    source = source_identity42()
    value = {'schema': 'b22-tracking42-release-v1', 'execution_contract_id': CONTRACT,
             'resource_revision': RESOURCE_REVISION, 'unused_budget_origin': unused_budget_origin42(),
             'plan': describe42(), 'source': source, 'inputs': inputs42(source)}
    for name, expected in value['inputs'].items():
        require(identity(name) == expected, 'C42 prepare input changed: '+name)
    write_json(RELEASE, value)
    return {'status': 'PREPARED', 'release': str(RELEASE), 'identity': identity(RELEASE),
            'execution_started': False}


def stable_admission42(sample, record, *, clock=time.monotonic, sleeper=time.sleep,
                       max_wait=300, stable_seconds=10, poll=5):
    """No reservation or physics occurs here. Runtime-low/unknown is a failure."""
    started, stable, previous, consecutive = clock(), None, None, 0
    while True:
        available = sample()
        now = clock()
        require(type(available) is int and available >= LIMITS['reserve_effective_bytes'],
                'C42 unknown/below 4 GiB admission memory')
        gap = None if previous is None else now-previous
        if gap is not None and (gap < 0 or gap > LIMITS['admission_max_sample_gap_s']):
            stable, consecutive = None, 0
        if available >= LIMITS['stable_effective_bytes']:
            stable = now if stable is None else stable
            consecutive += 1
        else:
            stable, consecutive = None, 0
        record({'elapsed_s': now-started, 'effective_available_bytes': available,
                'stable_elapsed_s': None if stable is None else now-stable,
                'sample_gap_s': gap, 'consecutive_qualified_samples': consecutive})
        previous = now
        after_record = clock()
        if (stable is not None and now-stable >= stable_seconds and after_record-started <= max_wait
                and after_record-now <= LIMITS['admission_max_sample_gap_s']
                and consecutive >= LIMITS['admission_minimum_samples']):
            return True
        if after_record-started >= max_wait:
            return False
        sleeper(min(poll, max_wait-(after_record-started)))


def directory_bytes42(path):
    """Stat payloads only; reject links so quota cannot miss an external writer."""
    total = 0
    if Path(path).exists():
        for item in Path(path).rglob('*'):
            try:
                snapshot = item.lstat()
            except FileNotFoundError:
                continue  # An owned archive/report temporary can be published during this snapshot.
            require(not stat.S_ISLNK(snapshot.st_mode), 'C42 output contains a symlink')
            if stat.S_ISREG(snapshot.st_mode):
                total += snapshot.st_size
    return total


def reader_environment42():
    environment = os.environ.copy()
    for name in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH'):
        environment.pop(name, None)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE'):
        environment[name] = '1'
    return environment


def stage_by_name42(name):
    return next(stage for stage in stages42() if stage['name'] == name)


def eval_inputs42():
    manifests, inputs = {}, {}
    for arm in ('control', 'deficit8'):
        run = OUTPUT/('train_'+arm)
        report_path = OUTPUT/('train_'+arm+'_readback.json')
        saved = read_json(report_path)
        require(saved.get('status') == 'COMPLETE' and saved.get('training_valid') is True
                and saved.get('release_identity') == identity(RELEASE),
                'C42 both complete training readbacks are required')
        manifest_path = run/'final_checkpoint_manifest.json'
        manifest = read_json(manifest_path)
        require(Path(manifest['folder']) == run/'final_checkpoint', 'C42 final-only model path differs')
        final_inputs = saved.get('final_checkpoint_inputs42')
        final_files = {str(manifest_path), *(str(file) for file in (run/'final_checkpoint').iterdir())}
        require(isinstance(final_inputs, dict) and set(final_inputs) == final_files,
                'C42 final model is not bound by its training reader')
        for name, expected in final_inputs.items():
            require(identity(name) == expected, 'C42 final checkpoint changed after training readback')
        for file in (report_path, manifest_path, run/'session.json', run/'worker_receipt.json'):
            inputs[str(file)] = identity(file)
        for file in (run/'final_checkpoint').iterdir():
            require(file.is_file() and not file.is_symlink(), 'C42 invalid final checkpoint member')
            inputs[str(file)] = identity(file)
        manifests[arm] = manifest
    return manifests, inputs


def verify_training_pair42():
    """Compare independent saved-data proofs, excluding the intended reward change."""
    rows = [read_json(OUTPUT/('train_'+arm+'_readback.json')) for arm in ('control', 'deficit8')]
    values = []
    for arm, row in zip(('control', 'deficit8'), rows):
        require(row.get('status') == 'COMPLETE' and row.get('training_valid') is True
                and row.get('stage') == 'train_'+arm and row.get('release_identity') == identity(RELEASE),
                'C42 first-rollout comparison needs both bound training readers')
        value = row.get('pairing42')
        require(isinstance(value, dict) and set(value) == {
            'schema', 'initial_state', 'initial_learning_state', 'first1024_numeric',
            'first1024_gaussian', 'seed', 'course_arm'}, 'C42 pairing proof fields missing')
        require(value['schema'] == 'b22-tracking42-pairing-v1'
                and value['seed'] == 201011 and value['course_arm'] == 'B'
                and all(isinstance(value[key], dict) and value[key] for key in (
                    'initial_state', 'initial_learning_state', 'first1024_numeric', 'first1024_gaussian')),
                'C42 pairing proof is empty or has a different initial course')
        values.append(value)
    require(values[0] == values[1], 'C42 initial policy/Adam/state or first-rollout nonreward data differ')
    return {'schema': 'b22-tracking42-pair-verification-v1', 'passed': True,
            'training_readers': {arm: identity(OUTPUT/('train_'+arm+'_readback.json'))
                                 for arm in ('control', 'deficit8')},
            'paired_proof': values[0]}


def make_session42(stage, release, *, parent_pid, nonce, started, wall_clock_utc=None):
    config = read_json(MANIFEST)
    defaults = worker_api().session_defaults42(stage['mode'], stage['reward_arm'])
    run = OUTPUT/stage['name']
    argv = [str(WORKER), '--session', str(run/'session.json'), '--output', str(run)]
    if stage['mode'] == 'bootstrap':
        argv.append('--bootstrap')
    source_hashes = {**release['inputs'], str(RELEASE): identity(RELEASE)}
    extra = {}
    if stage['mode'] == 'eval':
        manifests, inputs = eval_inputs42()
        source_hashes.update(inputs)
        extra['eval_manifests'] = manifests
    return {**config['shared_session'], **defaults, **stage, **extra,
            'schema': 'b22-tracking42-session-v1', 'execution_contract_id': CONTRACT,
            'resource_revision': RESOURCE_REVISION, 'resource_policy': dict(RESOURCE_POLICY),
            'output_directory': str(run), 'session_path': str(run/'session.json'),
            'source_hashes': source_hashes, 'runtime_environment': config['runtime_environment'],
            'parent_pid': parent_pid, 'nonce': nonce, 'argv': argv,
            'release_path': str(RELEASE), 'release_identity': identity(RELEASE),
            'worker_identity': identity(WORKER), 'host_started_monotonic': started,
            'wall_clock_utc': wall_clock_utc or datetime.now(timezone.utc).isoformat(),
            'wallclock_limit_s': stage['hard_s'],
            'retry_permitted': False, 'control_limit': stage['controls'],
            'normal_native_limit': stage['normal_native'], 'compiler_native_limit': stage['compiler_native'],
            'controller_variant': 'combined', 'output_quota_bytes': LIMITS['output_quota_bytes'],
            'execution_output': str(OUTPUT)}


def validate_worker_launch42(session_path, output):
    session_path, output = Path(session_path), Path(output)
    value = read_json(session_path)
    release = validate_release42(Path(value['release_path']))
    stage = stage_by_name42(value['name'])
    require(output == OUTPUT/stage['name'] and session_path == output/'session.json',
            'C42 worker output identity differs')
    require(type(value.get('parent_pid')) is int and value['parent_pid'] == os.getppid()
            and value.get('argv') == sys.argv and len(value.get('nonce', '')) == 48,
            'C42 worker parent/argv/nonce differs')
    expected = make_session42(stage, release, parent_pid=value['parent_pid'],
                              nonce=value['nonce'], started=value['host_started_monotonic'],
                              wall_clock_utc=value['wall_clock_utc'])
    require(value == expected, 'C42 canonical worker session differs')
    reservation = read_json(OUTPUT/(stage['name']+'_reservation.json'))
    require(reservation.get('session_identity') == identity(session_path)
            and reservation.get('nonce') == value['nonce'], 'C42 reservation binding differs')
    for key, expected in value['runtime_environment'].items():
        require(os.environ.get(key) == expected, 'C42 isolated runtime environment differs: '+key)
    require(os.getpriority(os.PRIO_PROCESS, 0) >= LIMITS['required_minimum_nice'],
            'C42 worker must run at Nice>=10')
    require(check_resource_snapshot42(resource_snapshot42()) >= LIMITS['start_effective_bytes'],
            'C42 worker admission below 6 GiB')
    require(time.monotonic() < value['host_started_monotonic']+value['soft_s'],
            'C42 worker source checks exhausted soft time')
    return value


def checked_counts42(value, stage, *, complete):
    require(isinstance(value, dict) and set(value) == set(COUNT_KEYS), 'C42 missing count fields')
    n, native, compiler = (value[key] for key in COUNT_KEYS)
    require(all(type(v) is int for v in (n, native, compiler))
            and 0 <= n <= stage['controls'] and 0 <= compiler <= stage['compiler_native']
            and 5*n <= native <= min(5*(n+1), stage['normal_native']),
            'C42 invalid physical count prefix')
    if complete:
        require(native == 5*n and compiler == stage['compiler_native'], 'C42 complete native ledger differs')
        if stage['mode'] == 'train':
            require(n == stage['controls'], 'C42 partial training is ineligible')
    return dict(value)


def known_counts42(stage):
    try:
        run = OUTPUT/stage['name']
        value = read_json(run/'worker_receipt.json')
        require(value['session_identity'] == identity(run/'session.json')
                and value['execution_contract_id'] == CONTRACT, 'C42 foreign partial receipt')
        actual, known = value.get('actual_counts'), value.get('known_partial_counts')
        if actual is not None and known is not None:
            require(checked_counts42(actual, stage, complete=False)
                    == checked_counts42(known, stage, complete=False),
                    'C42 conflicting exact and partial receipt counts')
        counts = known if actual is None else actual
        return checked_counts42(counts, stage, complete=False)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def validate_comparison42(value, accepted_stages):
    require(value.get('schema') == 'b22-tracking42-comparison-v1'
            and value.get('status') == 'COMPLETE'
            and value.get('execution_contract_id') == CONTRACT
            and value.get('release_identity') == identity(RELEASE)
            and value.get('reader_identity') == identity(READER),
            'C42 comparison provenance differs')
    reports = {entry['name']: entry['report'] for entry in accepted_stages
               if entry['name'] in ('train_control', 'train_deficit8', 'eval')}
    require(set(reports) == {'train_control', 'train_deficit8', 'eval'}
            and value.get('stage_reports') == reports,
            'C42 comparison differs from the three accepted numerical readbacks')
    for key in ('reader_model_calls', 'reader_physics_calls', 'actual_pure_servo_advances'):
        require(type(value.get(key)) is int and value[key] == 0,
                'C42 comparison must perform only saved-summary mathematics: '+key)


def execute42():
    started = time.monotonic()
    deadline = started+LIMITS['total_wall_s']
    require(not OUTPUT.exists(), 'C42 execution already claimed; no retry')
    unused_budget_origin42()
    require(os.getpriority(os.PRIO_PROCESS, 0) >= LIMITS['required_minimum_nice'],
            'C42 parent must run at Nice>=10')
    ADMISSIONS.mkdir(exist_ok=True)
    admission = ADMISSIONS/('admission_'+secrets.token_hex(12))
    admission.mkdir(exist_ok=False)
    report = {'schema': 'b22-tracking42-supervision-v1', 'status': 'WAITING',
              'execution_contract_id': CONTRACT, 'failure': None, 'actual_counts': None,
              'resource_revision': RESOURCE_REVISION, 'resource_policy': dict(RESOURCE_POLICY),
              'known_actual_counts': dict.fromkeys(COUNT_KEYS, 0), 'stages': [],
              'execution_started': False, 'retry_permitted': False, 'minimum_effective_bytes': None}
    common = host = None
    completed = {}
    pure_servo = {'worker': {}, 'reader': {}}
    active = None
    previous = {}
    def stopped(signum, _frame):
        raise InterruptedError('C42 signal '+signal.Signals(signum).name)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        previous[signum] = signal.signal(signum, stopped)
    signal.setitimer(signal.ITIMER_REAL, LIMITS['total_wall_s']-20)
    def save():
        report['elapsed_s'] = time.monotonic()-started
        write_json((OUTPUT if report['execution_started'] else admission)/'report.json', report, replace=True)
    def sample(_admission=False):
        snapshot = resource_snapshot42()
        report['last_memory_snapshot'] = snapshot
        free = shutil.disk_usage(ROOT).free
        with (admission/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps({'elapsed_s': time.monotonic()-started,
                                     'resource_revision': RESOURCE_REVISION,
                                     'memory': snapshot, 'free_disk_bytes': free})+'\n')
        available = check_resource_snapshot42(snapshot)
        report['minimum_effective_bytes'] = min(available, report['minimum_effective_bytes'] or available)
        require(free >= LIMITS['reserve_free_disk_bytes'], 'C42 disk runtime reserve exhausted')
        require(time.monotonic() < deadline, 'C42 total wall exhausted')
        require(directory_bytes42(OUTPUT) <= LIMITS['output_quota_bytes'], 'C42 total output quota exceeded')
        return available
    def admit(name):
        rows = admission/(name+'_samples.jsonl')
        def record(row):
            with rows.open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            report['active_stage'] = name+'_admission'
            save()
        ready = stable_admission42(sample, record)
        if not ready:
            raise TimeoutError('C42 '+name+' did not reach 6 GiB for 10 seconds within 300 seconds')
        require(shutil.disk_usage(ROOT).free >= LIMITS['start_free_disk_bytes'],
                'C42 disk admission below 10 GiB')
    def verify_completed_reports():
        for entry in report['stages']:
            file = entry.get('path', entry['name']+'_readback.json')
            require(identity(OUTPUT/file) == entry['report'],
                    'C42 completed readback changed: '+entry['name'])
    def run_process(argv, log, limit, env=None, stage=None):
        process, failure, code = None, None, None
        phase_started = time.monotonic()
        def factory(*args, **kwargs):
            nonlocal process
            process = subprocess.Popen(*args, **kwargs, env=env, cwd=ROOT)
            return process
        try:
            code = common.supervise(argv, log, host=host, sample=sample,
                                    limit=limit, deadline=deadline, factory=factory)
            return code
        except BaseException as error:
            failure = {'type': type(error).__name__, 'message': str(error)}
            raise
        finally:
            if stage is not None:
                cleanup = host.clean_descendants(os.getpid(), process)
                run = OUTPUT/stage['name']
                write_json(run/'supervisor_receipt.json', {
                    'schema': 'b22-tracking42-worker-supervision-v1', 'stage': stage['name'],
                    'release_identity': identity(RELEASE), 'session_identity': identity(run/'session.json'),
                    'worker_identity': identity(WORKER),
                    'exit_code': None if process is None else process.returncode,
                    'failure': failure, 'supervision_returned': code is not None,
                    'cleanup': cleanup, 'elapsed_s': time.monotonic()-phase_started})
                require(cleanup['remaining'] == {}, 'C42 worker closure contains owned processes')
    try:
        common = supervision()
        sample()  # Check actual job enforcement before hashing the multi-GiB source closure.
        release = validate_release42(RELEASE)
        config = read_json(MANIFEST)
        host_path = Path(config['host'])
        require(identity(host_path) == release['inputs'][str(host_path)], 'C42 cleanup host changed')
        host = load_module('_b22_tracking42_owned_host', host_path)
        host.subreaper()
        for stage in stages42():
            active = stage
            verify_completed_reports()
            if stage['mode'] == 'eval':
                write_json(OUTPUT/'training_pair.json', verify_training_pair42())
            admit(stage['name'])
            require(validate_release42(RELEASE) == release, 'C42 stage source identity changed')
            require(sample() >= LIMITS['stable_effective_bytes'], 'C42 RAM changed after final source check')
            comparison_reserve = 300 if stage['mode'] == 'eval' else 0
            require(time.monotonic()+stage['hard_s']+stage['reader_wall_s']+comparison_reserve+30 < deadline,
                    'C42 insufficient worker/readback/cleanup time remaining')
            if not report['execution_started']:
                OUTPUT.mkdir(exist_ok=False)
                report['execution_started'] = True
                report['release_identity'] = identity(RELEASE)
                report['source'] = release['source']
            run = OUTPUT/stage['name']
            run.mkdir(exist_ok=False)
            session = make_session42(stage, release, parent_pid=os.getpid(),
                                     nonce=secrets.token_hex(24), started=time.monotonic())
            write_json(run/'session.json', session)
            write_json(OUTPUT/(stage['name']+'_reservation.json'),
                       {'schema': 'b22-tracking42-reservation-v1', 'nonce': session['nonce'],
                        'session_identity': identity(run/'session.json'), 'budget': stage,
                        'retry_permitted': False, 'not_refundable': True})
            report.update(status='RUNNING', active_stage=stage['name'])
            save()
            env = os.environ.copy()
            for key, value in session['runtime_environment'].items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value
            code = run_process(['/usr/bin/python3', '-B', *session['argv']],
                               OUTPUT/(stage['name']+'_stdout.log'), stage['hard_s'], env, stage)
            worker = read_json(run/'worker_receipt.json')
            require(code == 0 and worker.get('status') == 'COMPLETE'
                    and worker.get('failure') is None and worker.get('cleanup_errors') == []
                    and worker.get('session_identity') == identity(run/'session.json'),
                    'C42 worker did not complete with clean ownership')
            completed[stage['name']] = checked_counts42(worker['actual_counts'], stage, complete=True)
            pure_count = worker.get('actual_pure_servo_advances')
            require(type(pure_count) is int and 0 <= pure_count <= stage['pure_servo_advance_max'],
                    'C42 worker pure-servo count missing or above independent budget')
            pure_servo['worker'][stage['name']] = pure_count
            report['known_actual_counts'] = {k: sum(v[k] for v in completed.values()) for k in COUNT_KEYS}
            if stage['mode'] == 'bootstrap':
                require(worker.get('bootstrap_verified42') is True
                        and completed[stage['name']] == dict.fromkeys(COUNT_KEYS, 0)
                        and pure_count == 0, 'C42 import bootstrap performed unregistered work')
                report['stages'].append({'name': stage['name'], 'path': 'bootstrap/worker_receipt.json',
                                        'report': identity(run/'worker_receipt.json'),
                                        'independent_saved_reader_executed': False})
                save()
                continue
            report['active_stage'] = stage['name']+'_readback'
            save()
            path = OUTPUT/(stage['name']+'_readback.json')
            argv = ['/usr/bin/python3', '-B', str(READER), '--release', str(RELEASE),
                    '--stage', stage['name'], '--report', str(path)]
            require(run_process(argv, OUTPUT/(stage['name']+'_reader.log'), stage['reader_wall_s'],
                                reader_environment42()) == 0,
                    'C42 saved readback failed')
            saved = read_json(path)
            require(saved.get('schema') == 'b22-tracking42-readback-v1' and saved.get('status') == 'COMPLETE'
                    and saved.get('stage') == stage['name'] and saved.get('release_identity') == identity(RELEASE)
                    and saved.get('session_identity') == identity(run/'session.json')
                    and saved.get('reader_identity') == identity(READER)
                    and saved.get('actual_counts') == completed[stage['name']], 'C42 readback binding/counts differ')
            require(stage['mode'] != 'train' or saved.get('training_valid') is True,
                    'C42 training validation failed')
            pure_count = saved.get('actual_pure_servo_advances')
            require(type(pure_count) is int and 0 <= pure_count <= stage['reader_pure_servo_advance_max'],
                    'C42 reader pure-servo count missing or above independent budget')
            pure_servo['reader'][stage['name']] = pure_count
            report['stages'].append({'name': stage['name'], 'report': identity(path)})
        report['active_stage'] = 'comparison'
        save()
        verify_completed_reports()
        require(time.monotonic()+300+30 < deadline, 'C42 insufficient comparison and cleanup window')
        path = OUTPUT/'comparison.json'
        argv = ['/usr/bin/python3', '-B', str(READER), '--release', str(RELEASE), '--compare', '--report', str(path)]
        require(run_process(argv, OUTPUT/'comparison_stdout.log', 300, reader_environment42()) == 0,
                'C42 comparison failed')
        compared = read_json(path)
        validate_comparison42(compared, report['stages'])
        require(validate_release42(RELEASE) == release, 'C42 final source identity changed')
        verify_completed_reports()
        sample()
        counts = report['known_actual_counts']
        require(all(counts[k] <= LIMITS[k] for k in COUNT_KEYS), 'C42 total physical budget exceeded')
        report.update(status='COMPLETE', active_stage=None, actual_counts=counts,
                      comparison_identity=identity(path), actual_pure_servo_advances={
                          side: sum(stages.values()) for side, stages in pure_servo.items()})
    except BaseException as error:  # noqa: BLE001 -- record every interrupted one-shot attempt before cleanup
        report.update(status='INCOMPLETE' if report['execution_started'] else 'NOT_READY',
                      failure={'type': type(error).__name__, 'message': str(error)})
        if active is not None and active['name'] not in completed and report['execution_started']:
            partial = known_counts42(active)
            if partial is not None:
                completed[active['name']] = partial
        report['known_actual_counts'] = {k: sum(v[k] for v in completed.values()) for k in COUNT_KEYS}
        report['known_pure_servo_advances'] = {
            side: sum(stages.values()) for side, stages in pure_servo.items()}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if host is not None:
                report['cleanup'] = host.clean_descendants(os.getpid(), None)
                require(not report['cleanup']['remaining'], 'C42 owned descendants remain')
        except BaseException as error:  # noqa: BLE001 -- cleanup failure must invalidate completion
            report.update(status='INCOMPLETE', actual_counts=None,
                          failure={'type': 'CleanupError', 'message': str(error)})
        if time.monotonic() >= deadline:
            report.update(status='INCOMPLETE', actual_counts=None,
                          failure={'type': 'TimeoutError', 'message': 'C42 total wall including cleanup exceeded'})
        save()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    value = prepare42() if args.prepare else execute42() if args.execute else describe42()
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if value['status'] in ('PLAN_ONLY', 'PREPARED', 'COMPLETE') else 2


if __name__ == '__main__':
    raise SystemExit(main())
