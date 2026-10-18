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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'C42_B22_tracking_deficit_comparison_v1'
OUTPUT = ROOT/'results/b22_tracking_reward42_20261018_05'
RELEASE = ROOT/'results/b22_tracking_reward42_plan_20261018_05.json'
ADMISSIONS = ROOT/'results/b22_tracking_reward42_admissions_20261018_05'
UNUSED_RELEASE = ROOT/'results/b22_tracking_reward42_plan_20261018_01.json'
UNUSED_OUTPUT = ROOT/'results/b22_tracking_reward42_20261018_01'
UNUSED_RELEASE_SHA = '7128e72d0fcfde9c0322d185e188f0648ef350c90911705fdb52e5fc8de597f0'
UNUSED_COMMIT = '4c55bbb2baeca84dc3db897ffeaeb07329faa38e'
RESOURCE_REVISION = 'owned_cgroup_v5'
PREDECESSOR_OUTPUT = ROOT/'results/b22_tracking_reward42_20261018_04'
PREDECESSOR_RELEASE = ROOT/'results/b22_tracking_reward42_plan_20261018_04.json'
PREDECESSOR_COMMIT = 'c2353be6e0c8a63055e58a13b7438a7188b1f73d'
PREDECESSOR_ERROR = 'C42 host/outside-ancestor RAM below 4 GiB runtime reserve or unknown'
EARLIER_FAILURE_REPORT = ROOT/'results/b22_tracking_reward42_20261018_02/report.json'
EARLIER_FAILURE_IDENTITY = {'bytes': 6404, 'sha256': '62d4c3ec599ed343d9b9b83836990959135e1f2f4585e58eb3401af7f6e31963'}
EARLIER_FAILURE_COMMIT = 'cd32a85f7815fae69f89b0b8d2c769e8f0d2262d'
OLDER_FAILURE_INPUTS = {
    'results/b22_tracking_reward42_plan_20261018_03.json':
        {'bytes': 914014, 'sha256': '48da3afd4eae303783173c142430699259714d7fb50de35548ccd6f09bb273c0'},
    'results/b22_tracking_reward42_20261018_03/report.json':
        {'bytes': 9244, 'sha256': '394e98d7483c68c74038f14dd3e5323e59abc320630b2070e066a0c652b93bac'},
    'results/b22_tracking_reward42_20261018_02/report.json':
        {'bytes': 6404, 'sha256': '62d4c3ec599ed343d9b9b83836990959135e1f2f4585e58eb3401af7f6e31963'},
    'results/b22_tracking_reward42_20261018_03/bootstrap/session.json':
        {'bytes': 1761593, 'sha256': '6e26e0a398c3503be07061701628f091afb916573a44dabe3e96580f3e8607ec'},
    'results/b22_tracking_reward42_20261018_03/bootstrap/worker_receipt.json':
        {'bytes': 1512, 'sha256': 'd4207344745eaf0e72e8069658acb0ef82b4cbb093d0d7f1c7de123e1fd5aa25'},
    'results/b22_tracking_reward42_20261018_03/bootstrap/supervisor_receipt.json':
        {'bytes': 738, 'sha256': '4da0ec20adc0d80814c0e79f6c0cee8ca6ab68dd6ee90bda4f7d7e1d66e1a51f'},
    'results/b22_tracking_reward42_20261018_03/train_control/session.json':
        {'bytes': 1761972, 'sha256': '306ca825aadb98192fe267a51699c965fdceb549c11b1881160808e0868b678b'},
    'results/b22_tracking_reward42_20261018_03/train_control/worker_receipt.json':
        {'bytes': 6733, 'sha256': '37c042e71ea1b2cbc9815f3accf72e9f92565ba6648dfb3fe29df303ac8931aa'},
    'results/b22_tracking_reward42_20261018_03/train_control/supervisor_receipt.json':
        {'bytes': 742, 'sha256': 'fa528d802a88aeb17eaab72dad94c5b830e45118dc6608bd9fb2f314147821a3'},
}
PREDECESSOR_INPUTS = {
    **OLDER_FAILURE_INPUTS,
    'results/b22_tracking_reward42_plan_20261018_04.json':
        {'bytes': 914956, 'sha256': '9dab65043a10b0b64b0db34ef2b929369cdb2e784974c02ac82667c16285e137'},
    'results/b22_tracking_reward42_20261018_04/report.json':
        {'bytes': 10003, 'sha256': '639f875ab1ff8e3b4d12ef33a1113b8c17b3f004ecbb64c1f258ac0625820282'},
    'results/b22_tracking_reward42_20261018_04/bootstrap/session.json':
        {'bytes': 1761802, 'sha256': '91eb8d30217c776a1e69d6b9f03536507132dae91d6c91425cb92a8be676283c'},
    'results/b22_tracking_reward42_20261018_04/bootstrap/worker_receipt.json':
        {'bytes': 1512, 'sha256': 'ad51ff228f830512d8a4de5c161e205896571eebe0817ae301b54a54b01ee767'},
    'results/b22_tracking_reward42_20261018_04/bootstrap/supervisor_receipt.json':
        {'bytes': 738, 'sha256': 'c05b101c1b1285de46b60e9ff82f57bc6d4930351d0642ca70fb2c6d79ae546b'},
    'results/b22_tracking_reward42_20261018_04/train_control/session.json':
        {'bytes': 1762181, 'sha256': '9ce542bf2a74e34cfb0e2c55711b4b5f3275ba8085649b9f45f3242fe921fa46'},
    'results/b22_tracking_reward42_20261018_04/train_control/supervisor_receipt.json':
        {'bytes': 859, 'sha256': '1dfe649ad63ca162461f457ce008a7159b80189b22188c1d97661c81c0a5ad80'},
    'results/b22_tracking_reward42_20261018_04/train_control/training/progress_rollout_0025.json':
        {'bytes': 4449, 'sha256': 'c60baf7c64e0d88bf978dda7091c92db58550a9780270762c674fe3c45ec5218'},
    'results/b22_tracking_reward42_20261018_04/train_control/training/progress_rollout_0025.json.manifest.json':
        {'bytes': 182, 'sha256': '8414536480c297a059e563b43f7f0d471b4073f781bdc469ac8255302668a08f'},
    'results/b22_tracking_reward42_20261018_04/train_control/failure_learning_receipt.json':
        {'bytes': 34076, 'sha256': '4d43e881a441eac2effac5363070422490f1fe1ca5bc1749b902b2c0c389deff'},
    'results/b22_tracking_reward42_20261018_04/train_control/failure_learning_receipt.json.manifest.json':
        {'bytes': 186, 'sha256': '5238c6ed7febf90ec3bbf67a8adf55ffbcd1e8379aec2f9d48364ca730570ff2'},
    'results/b22_tracking_reward42_20261018_04/train_control/training/training_segment_receipt.json':
        {'bytes': 2454, 'sha256': '3d05d1d466db2f0c4c68c7924b90f31492f24c5b5c819d028664ed51eac66da8'},
    'results/b22_tracking_reward42_20261018_04/train_control/training/training_segment_receipt.json.manifest.json':
        {'bytes': 185, 'sha256': 'aca72d99da53ad2c42bc9771bf7a9debb8624663d31eda610a83640f91fe8ee7'},
}
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
              'total_wall_s': 5400, 'start_effective_bytes': 4 << 30,
              'stable_effective_bytes': 4 << 30, 'reserve_effective_bytes': 2 << 30,
              'stable_seconds': 10, 'admission_poll_s': 5, 'admission_max_sample_gap_s': 7.5,
              'admission_minimum_samples': 3, 'admission_max_s': 300,
              'start_free_disk_bytes': 10 << 30, 'reserve_free_disk_bytes': 2 << 30,
              'output_quota_bytes': 8 << 30, 'required_minimum_nice': 10}
COUNT_KEYS = ('controls', 'normal_native', 'compiler_native')
RESOURCE_POLICY = {'memory_high_bytes': 2048 << 20, 'memory_max_bytes': 3072 << 20,
                   'memory_swap_max_bytes': 0, 'oom_policy': 'kill',
                   'memory_oom_group': 1,
                   'unit': 'wheel-b22-tracking42-owned05-20261018-01.service',
                   'owned_scope': '/user.slice/user-1000.slice/user@1000.service/app.slice/wheel-b22-tracking42-owned05-20261018-01.service',
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
            'C42 host/outside-ancestor RAM below 2 GiB runtime reserve or unknown')
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
            'original_scientific_budget_origin_only': True}


def predecessor_failure42():
    """Bind the trained04 interruption and its concrete02/03 history, never resume."""
    files = {str(ROOT/name): identity(ROOT/name) for name in PREDECESSOR_INPUTS}
    require(all(files[str(ROOT/name)] == entry for name, entry in PREDECESSOR_INPUTS.items()),
            'C42 predecessor receipt identity changed')
    def exact(value, expected):
        return (isinstance(value, dict) and set(value) == set(expected)
                and all(type(value[k]) is int and value[k] == v for k, v in expected.items()))
    release = read_json(PREDECESSOR_RELEASE)
    parent = read_json(PREDECESSOR_OUTPUT/'report.json')
    run = PREDECESSOR_OUTPUT/'train_control'
    session = read_json(run/'session.json')
    supervisor = read_json(run/'supervisor_receipt.json')
    zero = dict.fromkeys(COUNT_KEYS, 0)
    prior_two = {'controls': 0, 'normal_native': 0, 'compiler_native': 2}
    require(release.get('schema') == 'b22-tracking42-release-v1'
            and release.get('execution_contract_id') == CONTRACT
            and release.get('source', {}).get('git_commit') == PREDECESSOR_COMMIT
            and release.get('plan', {}).get('output') == str(PREDECESSOR_OUTPUT)
            and all(type(release['plan']['limits'].get(k)) is int
                    and release['plan']['limits'][k] == LIMITS[k] for k in COUNT_KEYS),
            'C42 predecessor release scope differs')
    old_worker = release['source']['authored'][str(WORKER)]
    failure = {'type': 'ValueError', 'message': PREDECESSOR_ERROR}
    require(parent.get('status') == 'INCOMPLETE' and parent.get('actual_counts') is None
            and parent.get('execution_started') is True and parent.get('active_stage') == 'train_control'
            and parent.get('execution_contract_id') == CONTRACT and parent.get('failure') == failure
            and parent.get('source') == release['source']
            and parent.get('release_identity') == files[str(PREDECESSOR_RELEASE)]
            and exact(parent.get('known_actual_counts'), zero)
            and parent.get('cleanup', {}).get('remaining') == {},
            'C42 predecessor RAM interruption or unknown total differs')
    require(session.get('execution_contract_id') == CONTRACT and session.get('name') == 'train_control'
            and session.get('mode') == 'train' and session.get('reward_arm') == 'control'
            and type(session.get('alpha')) is int and session['alpha'] == 1
            and session.get('execution_output') == str(PREDECESSOR_OUTPUT)
            and session.get('release_identity') == files[str(PREDECESSOR_RELEASE)]
            and session.get('worker_identity') == old_worker
            and supervisor.get('stage') == 'train_control'
            and supervisor.get('release_identity') == files[str(PREDECESSOR_RELEASE)]
            and supervisor.get('session_identity') == files[str(run/'session.json')]
            and supervisor.get('worker_identity') == old_worker
            and supervisor.get('failure') == failure and supervisor.get('exit_code') is None
            and supervisor.get('supervision_returned') is False
            and supervisor.get('cleanup', {}).get('remaining') == {}
            and not (run/'worker_receipt.json').exists(),
            'C42 predecessor session/cleanup or missing final receipt differs')
    prior = parent.get('predecessor_failure') or {}
    older_inputs = {str(ROOT/name): entry for name, entry in OLDER_FAILURE_INPUTS.items()}
    report03 = read_json(ROOT/'results/b22_tracking_reward42_20261018_03/report.json')
    report02 = read_json(EARLIER_FAILURE_REPORT)
    require(prior.get('inputs') == older_inputs
            and prior.get('historical_actual_counts') is None
            and exact(prior.get('historical_known_counts'), prior_two)
            and type(prior.get('historical_compiler_native_separate_from_new_budget')) is int
            and prior['historical_compiler_native_separate_from_new_budget'] == 4
            and prior.get('model_load_or_learning_occurred') is False
            and prior.get('earlier02_failure', {}).get('historical_actual_counts') is None
            and exact(prior.get('earlier02_failure', {}).get('historical_known_counts'), prior_two)
            and prior.get('earlier02_failure', {}).get('identity') == files[str(EARLIER_FAILURE_REPORT)]
            and all(row.get('status') == 'INCOMPLETE' and row.get('actual_counts') is None
                    and exact(row.get('known_actual_counts'), prior_two)
                    for row in (report02, report03)), 'C42 predecessor02/03 history was lost or changed')
    bootstrap = PREDECESSOR_OUTPUT/'bootstrap'
    boot = read_json(bootstrap/'worker_receipt.json')
    require(boot.get('status') == 'COMPLETE' and boot.get('bootstrap_verified42') is True
            and boot.get('session_identity') == files[str(bootstrap/'session.json')]
            and exact(boot.get('actual_counts'), zero)
            and parent.get('stages') == [{'name': 'bootstrap', 'path': 'bootstrap/worker_receipt.json',
                'report': files[str(bootstrap/'worker_receipt.json')],
                'independent_saved_reader_executed': False}], 'C42 predecessor bootstrap changed')
    names = ('training/progress_rollout_0025.json', 'failure_learning_receipt.json',
             'training/training_segment_receipt.json')
    for name in names:
        path = run/name
        transaction = read_json(path.with_name(path.name+'.manifest.json'))
        require(transaction.get('schema') == 'd1-archive-transaction-13-v1'
                and transaction.get('payloads') == [{'file': path.name, **files[str(path)]}],
                'C42 predecessor committed payload manifest differs: '+name)
    progress = read_json(run/names[0])
    require(max((run/'training').glob('progress_rollout_[0-9][0-9][0-9][0-9].json'), default=None) == run/names[0]
            and type(progress.get('completed_controls')) is int and progress['completed_controls'] == 26624
            and exact({k: progress.get('rollout', {}).get(k) for k in ('rollout_index', 'transitions', 'num_timesteps_at_end')},
                      {'rollout_index': 25, 'transitions': 1024, 'num_timesteps_at_end': 26624})
            and exact({k: progress.get('boundary', {}).get('C_state', {}).get(k) for k in
                       ('construction_attempts', 'construction_returns', 'control_attempts', 'control_returns')},
                      {'construction_attempts': 2, 'construction_returns': 2, 'control_attempts': 133120, 'control_returns': 133120})
            and exact({k: progress.get('boundary', {}).get('python', {}).get(k) for k in
                       ('control_attempted', 'control_completed', 'native_attempted', 'native_returned', 'clock_advanced_substeps')},
                      {'control_attempted': 26624, 'control_completed': 26624, 'native_attempted': 133120,
                       'native_returned': 133120, 'clock_advanced_substeps': 133120}),
            'C42 predecessor persisted rollout prefix differs')
    learning = read_json(run/names[1]);segment = read_json(run/names[2])
    require(learning.get('policy_state_changed') is True and learning.get('optimizer_state_changed') is True
            and learning.get('actor_mean_state_changed') is True
            and type(learning.get('actual', {}).get('optimizer_steps')) is int
            and learning['actual']['optimizer_steps'] > 0
            and type(learning['actual'].get('transitions')) is int and learning['actual']['transitions'] == 27000
            and type(segment.get('native_returned')) is int and segment['native_returned'] == 135000
            and learning.get('construction', {}).get('parent_final_zip_sha256') ==
                '7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691',
            'C42 predecessor actual learning or additional reported lower bound differs')
    return {'schema': 'b22-tracking42-trained-interruption-predecessor-v1',
            'output': str(PREDECESSOR_OUTPUT), 'release': str(PREDECESSOR_RELEASE),
            'source_commit': PREDECESSOR_COMMIT, 'inputs': files,
            'historical_attempt_occurred': True, 'historical_actual_counts': None,
            'parent_reported_known_counts': zero,
            'persisted_rollout_lower_bound': {'controls': 26624, 'normal_native': 133120, 'compiler_native': 2},
            'additional_reported_lower_bound': {'controls': 27000, 'normal_native': 135000, 'compiler_native': 2},
            'model_load_or_learning_occurred': True, 'worker_final_receipt_missing': True,
            'parent_zero_known_does_not_mean_zero_work': True,
            'earlier02_03_known_compiler_native': 4,
            'historical_compiler_native_separate_from_new_budget': 6,
            'partial_checkpoint_reused': False, 'old_unknown_total_is_not_rewritten': True,
            'new_attempt_is_independent_budget': True,
            'new_attempt_authority': 'continuing user authorisation for less conservative independent optimisation'}


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
            'predecessor_output': str(PREDECESSOR_OUTPUT),
            'historical_compiler_native_separate_from_new_budget': 6,
            'historical_actual_counts': None,
            'independent_new_budget': True, 'resume_predecessor_partial': False,
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
                    {str(MANIFEST): identity(MANIFEST), str(UNUSED_RELEASE): identity(UNUSED_RELEASE)},
                    predecessor_failure42()['inputs']):
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
            and value.get('original_scientific_budget_origin') == unused_budget_origin42()
            and value.get('predecessor_failure') == predecessor_failure42()
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
             'resource_revision': RESOURCE_REVISION,
             'original_scientific_budget_origin': unused_budget_origin42(),
             'predecessor_failure': predecessor_failure42(),
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
                'C42 unknown/below 2 GiB admission memory')
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
            'C42 worker admission below 4 GiB')
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


def phase_cleanup42(host, parent_pid, process, *, deadline, clock=time.monotonic, sleeper=time.sleep,
                    term_seen=None):
    """One TERM per owned birth identity, then the frozen bounded hard cleanup."""
    started = clock()
    until = min(started+15, deadline-5)
    seen = set() if term_seen is None else term_seen
    record = {'method': 'owned_birth_identity_term_grace_then_frozen_cleanup',
              'term_attempted_identities': [], 'errors': [], 'fallback': None}
    deferred = []
    prior_handlers = {}
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        prior_handlers[number] = signal.signal(number, lambda received, _frame: deferred.append(received))
    try:
        while True:
            alive = host.descendants(parent_pid)
            if not alive:
                break
            for pid, birth in alive.items():
                if (pid, birth) not in seen:
                    seen.add((pid, birth))
                    record['term_attempted_identities'].append({'pid': pid, 'birth_tick': birth})
                    host.signal_identity(pid, birth, signal.SIGTERM)
            if process is not None:
                process.poll()
            remaining = until-clock()
            if remaining <= 0:
                break
            sleeper(min(.25, remaining))
    except BaseException as error:  # noqa: BLE001 -- record interruption, still attempt owned hard cleanup
        record['errors'].append({'type': type(error).__name__, 'message': str(error)})
    finally:
        record['grace_elapsed_s'] = clock()-started
        try:
            record['fallback'] = host.clean_descendants(parent_pid, process)
        except BaseException as error:  # noqa: BLE001 -- supervisor receipt must retain cleanup failure
            record['errors'].append({'type': type(error).__name__, 'message': str(error)})
        record['elapsed_s'] = clock()-started
        for number, handler in prior_handlers.items():
            signal.signal(number, handler)
        if deferred:
            record['errors'].append({'type': 'InterruptedError',
                'message': 'cleanup deferred signals: '+','.join(signal.Signals(n).name for n in deferred)})
    return record


def execute42():
    started = time.monotonic()
    deadline = started+LIMITS['total_wall_s']
    require(not OUTPUT.exists(), 'C42 execution already claimed; no retry')
    unused_budget_origin42()
    predecessor = predecessor_failure42()
    require(os.getpriority(os.PRIO_PROCESS, 0) >= LIMITS['required_minimum_nice'],
            'C42 parent must run at Nice>=10')
    ADMISSIONS.mkdir(exist_ok=True)
    admission = ADMISSIONS/('admission_'+secrets.token_hex(12))
    admission.mkdir(exist_ok=False)
    report = {'schema': 'b22-tracking42-supervision-v1', 'status': 'WAITING',
              'execution_contract_id': CONTRACT, 'failure': None, 'actual_counts': None,
              'resource_revision': RESOURCE_REVISION, 'resource_policy': dict(RESOURCE_POLICY),
              'predecessor_failure': predecessor,
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
        measured = snapshot.get('effective_available_bytes')
        if type(measured) is int and measured >= 0:
            prior = report['minimum_effective_bytes']
            report['minimum_effective_bytes'] = measured if prior is None else min(measured, prior)
        available = check_resource_snapshot42(snapshot)
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
            raise TimeoutError('C42 '+name+' did not reach 4 GiB for 10 seconds within 300 seconds')
        require(shutil.disk_usage(ROOT).free >= LIMITS['start_free_disk_bytes'],
                'C42 disk admission below 10 GiB')
    def verify_completed_reports():
        for entry in report['stages']:
            file = entry.get('path', entry['name']+'_readback.json')
            require(identity(OUTPUT/file) == entry['report'],
                    'C42 completed readback changed: '+entry['name'])
    def run_process(argv, log, limit, env=None, stage=None):
        process, failure, code, primary = None, None, None, None
        phase_started = time.monotonic()
        cleanup_records = []
        term_seen = set()
        def factory(*args, **kwargs):
            nonlocal process
            process = subprocess.Popen(*args, **kwargs, env=env, cwd=ROOT)
            return process
        def phase_clean(parent_pid, owned_process):
            nonlocal primary, failure
            pending = sys.exc_info()[1]
            if primary is None and pending is not None:
                primary = pending
                failure = {'type': type(pending).__name__, 'message': str(pending)}
            record = phase_cleanup42(host, parent_pid, owned_process,
                                     deadline=min(deadline, phase_started+limit), term_seen=term_seen)
            cleanup_records.append(record)
            require(not record['errors'] and record['fallback'] is not None
                    and record['fallback'].get('remaining') == {}, 'C42 phase owned cleanup failed')
            return record['fallback']
        try:
            code = common.supervise(argv, log, host=SimpleNamespace(clean_descendants=phase_clean),
                                    sample=sample, limit=limit, deadline=deadline, factory=factory)
            return code
        except BaseException as error:
            primary = error if primary is None else primary
            failure = {'type': type(primary).__name__, 'message': str(primary)}
            if primary is not error:
                raise primary from error
            raise
        finally:
            cleanup, cleanup_error = None, None
            try:
                cleanup = phase_clean(os.getpid(), process)
            except BaseException as error:  # noqa: BLE001 -- always attempt receipt below
                cleanup_error = {'type': type(error).__name__, 'message': str(error)}
            finally:
                report.setdefault('phase_cleanup_records', []).append({
                    'log': Path(log).name, 'records': cleanup_records})
                if stage is not None:
                    run = OUTPUT/stage['name']
                    write_json(run/'supervisor_receipt.json', {
                        'schema': 'b22-tracking42-worker-supervision-v1', 'stage': stage['name'],
                        'release_identity': identity(RELEASE), 'session_identity': identity(run/'session.json'),
                        'worker_identity': identity(WORKER),
                        'exit_code': None if process is None else process.returncode,
                        'failure': failure, 'supervision_returned': code is not None,
                        'cleanup': cleanup, 'cleanup_error': cleanup_error,
                        'phase_cleanup_records': cleanup_records,
                        'elapsed_s': time.monotonic()-phase_started})
            if cleanup_error is not None:
                if primary is not None:
                    raise primary
                raise RuntimeError('C42 phase cleanup failed; see supervisor receipt')
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
