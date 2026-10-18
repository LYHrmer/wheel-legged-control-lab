"""Independent C37 missing-case budget; never resumes or rewrites C35."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTERED_ROOT = Path('/home/lyh/wheel-legged-control-lab')
CONTRACT = 'C37_B22_half_gain_missing_cases_v1'
OUTPUT = ROOT/'results/b22_gain_completion37_20261018_01'
OLD_ROOT = ROOT/'results/b22_gain_suite35_20261018_01'
AUTHORED = ('scripts/run_b22_gain_completion37.py', 'scripts/run_b22_worker37.py',
            'scripts/read_b22_gain_completion37.py', 'tests/test_b22_gain_completion37.py',
            'docs/b22_gain_completion37.md')
OLD_SOURCES = {'scripts/run_b22_gain_suite35.py': '82b8af91b32cdba16820b26df0542db7e034b3e5f99aef154dadde8ed5b5eb54',
               'scripts/run_b22_worker35.py': '07be70b151c383a4a88a4b2b2f7c0242cb02919c9bc9561ce577c2321d6900cb',
               'scripts/read_b22_gain_suite35.py': 'f9a5172dd375aae86e4f2ba71b87bbf41d72df1fbb892e46e96d267b131754ec'}
OLD_RECEIPTS = {'release.json': {'bytes': 869062, 'sha256': 'c0d09c7e9a969a28987b128d9f393d5a3be09f656d3c28fbcb470db6e3f2e568'}, 'report.json': {'bytes': 3476, 'sha256': '1f2464480c13c57ee7a237f822bd35169adede9daeea77abd91017749cca71a3'}, 'step_pair_readback.json': {'bytes': 113310, 'sha256': 'cf5c0188bb6e06dbd096eef6035da1848be6562d8e28f0b8d64f96a85144ced9'}, 'flat_0p6_zero_readback.json': {'bytes': 27164, 'sha256': '20e7509c0f46ec5bafc53bc99cef17706779bec17acf1b23f7e9da751cd4daa7'}, 'flat_0p6_B_readback.json': {'bytes': 27296, 'sha256': 'a62175ed5aef1cd8a2ca5dc247fab2fda623c8bcf223d146592ac14dfca03a59'}, 'flat_1p6_zero_readback.json': {'bytes': 33456, 'sha256': 'e9d7714b217773e3c2a53ab1a73de2e69562e4aaacf38f5bdc9329d6b9dd4412'}, 'flat_1p6_B_readback.json': {'bytes': 33553, 'sha256': '69a54f428d5bc9948259ed719a1f428836f40219585a4ca01f29819f932592f8'}, 'yaw_1p2_zero_readback.json': {'bytes': 31270, 'sha256': 'd3549f323c73eaded59e90b64c3a384183c33476e58370d9a1c79af9b3bae24a'}, 'yaw_1p2_B_readback.json': {'bytes': 31403, 'sha256': '3a3858701ee9026170337a7bc4e8a04f08851935af37827ac306d198483eafa6'}, 'bumps_0p4_zero_readback.json': {'bytes': 26395, 'sha256': '49c512d3ecc22fbfadbdf0f843fcbbae21502206f1ac549a8e8a607a036fc803'}, 'bumps_0p4_B_readback.json': {'bytes': 26535, 'sha256': 'd186b9284dab281573a9d0f8e41e33b556737f8d0651c4863fd60ec83ccd3d16'}, 'rough_0p35_zero_readback.json': {'bytes': 26046, 'sha256': '5633c0d1526e4062b585023013668b883ad86560376fe5940848b8d93e8a77fe'}, 'rough_0p35_B_readback.json': {'bytes': 26200, 'sha256': '9f6e720b5892687a4979329c8f750b7b356c112c2cc7026a5b04e3c728e08cd7'}}
REUSED_COUNTS = {'controls': 18400, 'normal_native': 92000, 'compiler_native': 22}
LIMITS = {'controls': 6800, 'normal_native': 34000, 'compiler_native': 8, 'worker_sessions': 4,
          'total_wall_s': 900, 'stage_wall_s': 300, 'reader_wall_s': 150,
          'start_effective_bytes': 6 << 30, 'reserve_effective_bytes': 4 << 30,
          'start_free_disk_bytes': 2 << 30, 'reserve_free_disk_bytes': 1 << 30,
          'required_minimum_nice': 10, 'training_calls': 0}


def load_old(name, relative):
    path = ROOT/relative
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != OLD_SOURCES[relative]:
        raise ValueError('C37 original source changed: '+relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old35 = load_old('completion37_original35', 'scripts/run_b22_gain_suite35.py')
c, supervision = old35.c, old35.common


def stages():
    return [copy.deepcopy(stage) for stage in old35.stages()[-4:]]


def plan(output=None):
    c.require(output is None or Path(output).absolute() == OUTPUT, 'C37 fixed output only')
    return {'schema': 'b22-gain-completion37-plan-v1', 'status': 'PLAN_ONLY', 'output': str(OUTPUT),
            'owner_instruction': '把这些优化点，在这台电脑能力之内的都完成',
            'stages': stages(), 'limits': LIMITS, 'reused_validated_case_counts': REUSED_COUNTS,
            'reused_old35_outer_actual_counts': None, 'historical_total_resources_known': False,
            'execution_started': False, 'training_calls': 0, 'retry_permitted': False,
            'automatic_model_replacement': False}


def source_identity():
    subprocess.run(['git', 'ls-files', '--error-unmatch', *AUTHORED], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, timeout=5)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *AUTHORED], cwd=ROOT, check=True, timeout=5)
    return {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, timeout=5).strip(),
            'authored': {str(ROOT/name): c.identity(ROOT/name) for name in AUTHORED}}


def check_authored(source):
    c.require(re.fullmatch('[0-9a-f]{40}', source.get('git_commit', ''))
              and set(source.get('authored', {})) == {str(ROOT/name) for name in AUTHORED}, 'C37 source commit/closure differs')
    for name in AUTHORED:
        path = ROOT/name
        c.require(not path.is_symlink() and c.identity(path) == source['authored'][str(path)], 'C37 authored source changed')
        blob = subprocess.check_output(['git', 'show', source['git_commit']+':'+name], cwd=ROOT, stderr=subprocess.PIPE, timeout=5)
        c.require(hashlib.sha256(blob).hexdigest() == source['authored'][str(path)]['sha256'], 'C37 uncommitted authored source')


def counts(stage, saved, release_id, reader_id):
    value = saved.get('actual_counts', {})
    n = value.get('controls')
    c.require(saved.get('schema') == 'b22-gain-completion37-stage-readback-v1'
              and saved.get('status') == 'COMPLETE' and saved.get('stage') == stage['name']
              and saved.get('execution_contract_id') == CONTRACT and type(saved.get('gain35')) is float
              and saved['gain35'] == .5 and saved.get('release_identity') == release_id
              and saved.get('reader_identity') == reader_id and type(n) is int and 0 < n <= stage['controls']
              and all(type(v) is int and v >= 0 for v in value.values())
              and value.get('normal_native') == 5*n and value.get('compiler_native') == 2
              and saved.get('gain_ledger_rows') == n, 'C37 stage identity/counts differ')
    return {key: value[key] for key in REUSED_COUNTS}


def load_reused35():
    """Trust pinned independent closed readbacks after rehashing all saved payloads."""
    files, reports = {}, {}
    for name, expected in OLD_RECEIPTS.items():
        path = OLD_ROOT/name
        c.require(not path.is_symlink() and c.identity(path) == expected, 'C37 old35 receipt changed: '+name)
        files[str(path)] = expected
    release = old35.validate_release(OLD_ROOT/'release.json')
    parent = c.read_json(OLD_ROOT/'report.json')
    c.require(parent.get('status') == 'INCOMPLETE' and parent.get('actual_counts') is None
              and parent.get('active_stage_actual_counts') is None and parent.get('known_actual_counts') == REUSED_COUNTS
              and parent.get('cleanup', {}).get('remaining') == {} and parent.get('source') == release['source']
              and parent.get('stage_reports') == {k: v for k, v in OLD_RECEIPTS.items() if k.endswith('_readback.json')},
              'C37 original incomplete/unknown C35 closure differs')
    worker = load_old('completion37_original_worker35', 'scripts/run_b22_worker35.py')
    total = {key: 0 for key in REUSED_COUNTS}
    for stage in old35.stages()[:-4]:
        path, run = OLD_ROOT/(stage['name']+'_readback.json'), OLD_ROOT/stage['name']
        saved = c.read_json(path)
        row = old35.stage_counts(stage, saved, OLD_RECEIPTS['release.json'],
                                  release['inputs'][str(ROOT/'scripts/read_b22_gain_suite35.py')])
        c.require(row['controls'] == stage['controls'], 'C37 reused case was not full-horizon')
        for name, expected in saved['artifacts'].items():
            relative = Path(name)
            c.require(not relative.is_absolute() and '..' not in relative.parts
                      and not (run/relative).is_symlink() and c.identity(run/relative) == expected,
                      'C37 reused raw payload changed: '+name)
        session, receipt = (c.read_json(run/name) for name in ('session.json', 'worker_receipt.json'))
        old35.case_completion(stage, receipt)
        c.require(receipt['session_identity'] == c.identity(run/'session.json')
                  and session['execution_contract_id'] == old35.CONTRACT and session['gain35'] == .5
                  and session['output_directory'] == str(run)
                  and session['argv'] == [str(ROOT/'scripts/run_b22_worker35.py'), '--session', str(run/'session.json'), '--output', str(run)]
                  and all(session['source_hashes'].get(k) == v for k, v in release['inputs'].items()),
                  'C37 reused worker/source owner differs')
        c.require(c.read_json(run/'derived_sources35.json') == worker.derivation_proof(worker.derived_sources()),
                  'C37 original worker derivation differs')
        for payload in run.rglob('*'):
            if payload.is_file():
                c.require(not payload.is_symlink(), 'C37 reused archive symlink')
                files[str(payload)] = c.identity(payload)
        reports[stage['name']] = saved
        total = {key: total[key]+row[key] for key in total}
    c.require(total == REUSED_COUNTS, 'C37 reused subtotal differs')
    return {'status': 'VALIDATED_CLOSED_CASES_ONLY', 'old_outer_actual_counts': None,
            'historical_total_resources_known': False, 'validated_case_counts': total,
            'inputs': files, 'reports': reports, 'old_release': release}


def closure(source, reused, config):
    return {**config['source_hashes'], **reused['old_release']['inputs'], **source['authored'],
            **reused['inputs'], **{str(ROOT/name): c.identity(ROOT/name) for name in OLD_SOURCES},
            str(ROOT/c.MANIFEST): c.identity(ROOT/c.MANIFEST)}


def reused_identity(reused):
    return {key: reused[key] for key in ('status', 'old_outer_actual_counts', 'historical_total_resources_known',
                                         'validated_case_counts', 'inputs')}


def validate_release(path):
    release = c.read_json(path)
    reused = load_reused35()
    c.require(release.get('schema') == 'b22-gain-completion37-release-v1' and release.get('plan') == plan()
              and Path(release['plan']['output']) == OUTPUT and release.get('reused35') == reused_identity(reused),
              'C37 fixed release/reused evidence differs')
    check_authored(release['source'])
    config = c.read_json(ROOT/c.MANIFEST)
    c.require(c.identity(ROOT/c.MANIFEST)['sha256'] == c.MANIFEST_SHA
              and set(release['source']['authored']) == {str(ROOT/name) for name in AUTHORED}
              and release['inputs'] == closure(release['source'], reused, config), 'C37 source closure differs')
    for name, expected in release['inputs'].items():
        c.require(not Path(name).is_symlink() and c.identity(name) == expected, 'C37 frozen input changed: '+name)
    return release


def request(stage, output, release, config, release_path):
    c.require(stage in stages(), 'C37 only missing four cases are authorized')
    launch, run = output/(stage['name']+'_launch'), output/stage['name']
    launch.mkdir(exist_ok=False)
    spec = {'schema': 'd1-c23-gui-worker-v1', 'mode': 'script', 'render': False, 'profile': stage['profile'],
            'actor': stage['actor'], 'seed': stage['seed'], 'mirror': stage['mirror'],
            'control_limit': stage['controls'], 'seconds': stage['controls']*.01, 'output_directory': str(run),
            'qualification': False, 'controller_variant': 'combined', 'gain35': .5,
            'execution_contract_id': CONTRACT, 'soft_s': 180, 'close_s': 235, 'hard_s': 240,
            'frame_limit': 1000, 'require_event_driver': False,
            'model_limits': c.model_limits(stage['actor'], stage['controls'], config['model_limit_keys'])}
    go = {'decision': 'GO', 'scope': 'independent C37 missing-case budget; no C35 resumption or training',
          'repository': str(ROOT), 'worker': str(ROOT/'scripts/run_b22_worker37.py'),
          'inputs': {**release['inputs'], str(release_path): c.identity(release_path)},
          'runtime_environment': config['runtime_environment'], 'shared_session': config['shared_session'],
          'arms': {stage['name']: spec}, 'x11_events_by_arm': {}}
    c.write_json(launch/'source_go.json', go)
    c.write_json(launch/'request.json', {'source_go_path': str(launch/'source_go.json'), 'arm': stage['name'], 'spec': spec})
    return ['rtk', 'proxy', '/usr/bin/python3', '-B', config['host'], '--request', str(launch/'request.json')]


def case_completion(stage, worker):
    c.require(worker.get('execution_contract_id') == CONTRACT, 'C37 worker identity differs')
    return c.case_completion(stage, worker)


def known_partial(stage, run, release):
    try:
        session, worker = (c.read_json(run/name) for name in ('session.json', 'worker_receipt.json'))
        c.require(session.get('execution_contract_id') == worker.get('execution_contract_id') == CONTRACT
                  and session.get('gain35') == .5 and session['output_directory'] == str(run)
                  and worker.get('session_identity') == c.identity(run/'session.json')
                  and all(session['source_hashes'].get(k) == v for k, v in release['inputs'].items()),
                  'foreign partial C37 receipt')
        py, state = worker['result']['python'], worker['result']['C_final']
        n, native, compiler = py['control_completed'], py['native_returned'], state['construction_returns']
        c.require(all(type(v) is int for v in (n, native, compiler)) and 0 <= n <= stage['controls']
                  and 5*n <= native <= min(5*stage['controls'], 5*(n+1)) and 0 <= compiler <= 2,
                  'invalid C37 partial physical prefix')
        return {'controls': n, 'normal_native': native, 'compiler_native': compiler}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def counted(completed, active=None, partial=None):
    return old35.counted_prefix(completed, active, partial)


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
        raise InterruptedError('C37 termination during owned cleanup: '+','.join(received))
    return receipt


def supervise(argv, log, *, host, **kwargs):
    class OwnedHost:
        def clean_descendants(self, parent, process):
            return cleanup_owned(host, process)
    return supervision.supervise(argv, log, host=OwnedHost(), **kwargs)


def execute(output=OUTPUT):
    output = Path(output).absolute()
    c.require(ROOT == REGISTERED_ROOT and output == OUTPUT and not output.exists() and not output.is_symlink(),
              'fixed new37 output already used/overridden; no retry')
    started, source, reused = time.monotonic(), source_identity(), load_reused35()
    c.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'C37 requires Nice10 or lower priority')
    if supervision.memory() < 6 << 30 or shutil.disk_usage(ROOT).free < 2 << 30:
        return {'schema': 'b22-gain-completion37-supervision-v1', 'status': 'NOT_READY', 'execution_started': False}
    output.mkdir(exist_ok=False)
    completed = {}
    report = {'schema': 'b22-gain-completion37-supervision-v1', 'status': 'INCOMPLETE', 'failure': None,
              'source': source, 'actual_counts': None, 'actual_scope': 'new37_only',
              'known_actual_counts': counted(completed), 'reused_validated_case_counts': dict(REUSED_COUNTS),
              'reused_old35_outer_actual_counts': None, 'historical_total_resources_known': False,
              'stage_reports': {}, 'retry_permitted': False, 'automatic_model_replacement': False,
              'nice': os.getpriority(os.PRIO_PROCESS, 0)}
    host, active, release, previous, deadline = None, None, None, {}, started+890
    def stopped(signum, frame):
        raise InterruptedError('C37 interrupted: '+signal.Signals(signum).name)
    def sample(admission=False):
        available, free = supervision.memory(), shutil.disk_usage(output).free
        with (output/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps({'elapsed_s': time.monotonic()-started, 'effective_available_bytes': available,
                                     'free_disk_bytes': free})+'\n')
        report['minimum_effective_bytes'] = min(report.get('minimum_effective_bytes', available), available)
        c.require(available >= (6 << 30 if admission else 4 << 30), 'C37 RAM admission/runtime reserve exhausted')
        c.require(free >= 1 << 30 and time.monotonic() < deadline, 'C37 disk/wall reserve exhausted')
    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.signal(number, stopped)
    old_alarm = signal.signal(signal.SIGALRM, stopped)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline-time.monotonic()))
    try:
        config, readiness = c.registered_config()
        release = {'schema': 'b22-gain-completion37-release-v1', 'plan': plan(), 'source': source,
                   'reused35': reused_identity(reused), 'inputs': closure(source, reused, config), 'v27_readiness': readiness}
        release_path = output/'release.json'
        c.write_json(release_path, release)
        validate_release(release_path)
        host = c.load_pinned(Path(config['host']), release['inputs'][config['host']], 'completion37_owned_host')
        host.subreaper()
        for active in stages():
            sample(True)
            c.require(time.monotonic()+LIMITS['reader_wall_s']+10 < deadline, 'C37 no bounded reader/cleanup window')
            argv = request(active, output, release, config, release_path)
            code = supervise(argv, output/(active['name']+'_stdout.log'), host=host, sample=sample,
                                         limit=LIMITS['stage_wall_s'], deadline=deadline)
            partial = known_partial(active, output/active['name'], release)
            report['known_actual_counts'] = counted(completed, active, partial)
            case_completion(active, c.read_json(output/active['name']/'worker_receipt.json'))
            c.require(code == 0, 'C37 worker/host failed')
            destination = output/(active['name']+'_readback.json')
            argv = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(ROOT/'scripts/read_b22_gain_completion37.py'),
                    '--release', str(release_path), '--stage', active['name'], '--report', str(destination)]
            c.require(supervise(argv, output/(active['name']+'_reader.log'), host=host, sample=sample,
                                           limit=LIMITS['reader_wall_s'], deadline=deadline) == 0, 'C37 saved reader failed')
            saved = c.read_json(destination)
            row = counts(active, saved, c.identity(release_path), release['inputs'][str(ROOT/'scripts/read_b22_gain_completion37.py')])
            completed[active['name']] = row
            report['known_actual_counts'] = counted(completed)
            c.require(all(report['known_actual_counts'][k] <= LIMITS[k] for k in REUSED_COUNTS), 'C37 new physical budget exhausted')
            report['stage_reports'][destination.name] = c.identity(destination)
            active = None
        argv = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(ROOT/'scripts/read_b22_gain_completion37.py'),
                '--release', str(release_path), '--compare', '--report', str(output/'comparison.json')]
        c.require(supervise(argv, output/'comparison_stdout.log', host=host, sample=sample,
                                       limit=LIMITS['reader_wall_s'], deadline=deadline) == 0, 'C37 comparison failed')
        compared = c.read_json(output/'comparison.json')
        c.require(compared.get('schema') == 'b22-gain-completion37-comparison-v1' and compared['status'] == 'COMPLETE'
                  and compared['release_identity'] == c.identity(release_path)
                  and compared['new_stage_reports'] == report['stage_reports']
                  and compared['new_actual_counts'] == counted(completed)
                  and compared['reused_validated_case_counts'] == REUSED_COUNTS
                  and compared['combined_validated_case_counts'] == {k: counted(completed)[k]+REUSED_COUNTS[k] for k in REUSED_COUNTS}
                  and compared['old35_outer_actual_counts'] is None and compared['historical_total_resources_known'] is False
                  and source_identity() == source,
                  'C37 final identity differs')
        sample(False)
        report.update(status='COMPLETE', actual_counts=counted(completed), comparison=c.identity(output/'comparison.json'),
                      combined_validated_case_counts=compared['combined_validated_case_counts'],
                      finite_suite_task_qualification_passed=compared['finite_suite_task_qualification_passed'],
                      RL_contribution_passed=compared['RL_contribution_passed'], no_task_regression=compared['no_task_regression'])
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report['failure'] = type(error).__name__+': '+str(error)
        if active is not None and release is not None:
            partial = known_partial(active, output/active['name'], release)
            report['known_actual_counts'] = counted(completed, active, partial)
            report['active_stage_actual_counts'] = partial
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        for number in previous:
            signal.signal(number, signal.SIG_IGN)
        if host is not None:
            try:
                report['cleanup'] = cleanup_owned(host, None)
                if report['cleanup']['remaining']:
                    report.update(status='INCOMPLETE', failure='C37 owned descendants remain', actual_counts=None)
            except (OSError, subprocess.SubprocessError) as error:
                report.update(status='INCOMPLETE', failure='C37 cleanup: '+str(error), actual_counts=None)
        report['elapsed_s'] = time.monotonic()-started
        if report['elapsed_s'] >= LIMITS['total_wall_s']:
            report.update(status='INCOMPLETE', failure='C37 final wall limit exhausted', actual_counts=None)
        if report['status'] != 'COMPLETE':
            report.update(finite_suite_task_qualification_passed=None, RL_contribution_passed=None, no_task_regression=None)
        try:
            c.write_json(output/'report.json', report)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(plan(args.output), indent=2, ensure_ascii=False))
        return 0
    value = execute(args.output or OUTPUT)
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0 if value['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
