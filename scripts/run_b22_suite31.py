"""Plan, or explicitly execute, the fixed serial B22 suite. No training imports."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

_spec = importlib.util.spec_from_file_location('suite31_contract', Path(__file__).with_name('b22_suite31_contract.py'))
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)


def memory(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
    fields = dict(line.split(':', 1) for line in (proc/'meminfo').read_text().splitlines())
    available = int(fields['MemAvailable'].split()[0])*1024
    scope = next((line[3:] for line in (proc/'self/cgroup').read_text().splitlines()
                  if line.startswith('0::')), None)
    c.require(scope is not None and '..' not in Path(scope).parts, 'unknown cgroup RAM scope')
    current, limits = cgroup/scope.lstrip('/'), []
    c.require(current.is_dir(), 'cgroup-v2 is not visible')
    while True:
        path = current/'memory.max'
        if path.exists() and path.read_text().strip() != 'max':
            maximum, used = int(path.read_text()), int((current/'memory.current').read_text())
            c.require(maximum >= 0 and used >= 0, 'invalid cgroup RAM accounting')
            limits.append(max(0, maximum-used))
        if current == cgroup:
            break
        current = current.parent
    c.require(available >= 0, 'invalid host available RAM')
    return min([available, *limits])


def script_request(stage, release, stage_output, config, release_path):
    launch = stage_output.with_name(stage_output.name+'_launch')
    launch.mkdir(exist_ok=False)
    spec = {'schema': 'd1-c23-gui-worker-v1', 'mode': 'script', 'render': False,
            'profile': stage['profile'], 'actor': stage['actor'], 'seed': stage['seed'],
            'mirror': stage['mirror'], 'control_limit': stage['controls'], 'seconds': stage['controls']*.01,
            'output_directory': str(stage_output), 'qualification': False,
            'model_limits': c.model_limits(stage['actor'], stage['controls'], config['model_limit_keys']),
            'soft_s': 180, 'close_s': 235, 'hard_s': 240, 'frame_limit': 1000,
            'require_event_driver': False, 'execution_contract_id': 'C31_B22_fixed_suite_v1'}
    inputs = {**release['inputs'], str(release_path): c.identity(release_path)}
    go = {'decision': 'GO', 'scope': 'suite31 fixed B/zero task matrix; no training',
          'repository': str(c.ROOT), 'worker': str(c.ROOT/'scripts/run_b22_worker31.py'),
          'inputs': inputs, 'runtime_environment': config['runtime_environment'],
          'shared_session': config['shared_session'], 'arms': {stage['name']: spec},
          'x11_events_by_arm': {}}
    c.write_json(launch/'source_go.json', go)
    c.write_json(launch/'request.json', {'source_go_path': str(launch/'source_go.json'),
                                      'arm': stage['name'], 'spec': spec})
    return ['rtk', 'proxy', '/usr/bin/python3', '-B', config['host'],
            '--request', str(launch/'request.json')]


def step_request(release, stage_output, config, release_path):
    launch = stage_output.with_name(stage_output.name+'_launch')
    launch.mkdir(exist_ok=False)
    limits = {name: 0 for name in config['model_limit_keys']}
    limits.update(load=1, torch_load=3, predict=1201)
    spec = {'mode': 'eval', 'arm': 'step_pair', 'controls': 2400, 'control_limit': 2400,
            'controller_variant': 'combined', 'output_directory': str(stage_output),
            'soft_s': 240, 'close_s': 270, 'hard_s': 300, 'model_limits': limits,
            'render': False, 'execution_contract_id': 'C24_true_15mm_B22_v1'}
    inputs = {**release['inputs'], str(release_path): c.identity(release_path)}
    go = {'decision': 'GO', 'scope': 'suite31 new repetition of unchanged C24 protocol',
          'repository': str(c.ROOT), 'worker': str(c.ROOT/'runtime/d1_b22/research/continuation24/worker24.py'),
          'inputs': inputs, 'runtime_environment': config['runtime_environment'],
          'shared_session': config['shared_session'], 'arms': {'step_pair': spec},
          'x11_events_by_arm': {}}
    c.write_json(launch/'source_go.json', go)
    request = {'source_go_path': str(launch/'source_go.json'), 'arm': 'step_pair', 'spec': spec}
    c.write_json(launch/'request.json', request)
    return ['rtk', 'proxy', '/usr/bin/python3', '-B', config['host'],
            '--request', str(launch/'request.json')]


def supervise(argv, log, *, host, sample, limit, deadline, factory=subprocess.Popen,
              clock=time.monotonic, sleeper=time.sleep):
    process = None
    started = clock()
    cleanup = None
    try:
        with log.open('xb') as stream:
            process = factory(argv, stdin=subprocess.DEVNULL, stdout=stream,
                              stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                sample(False)
                c.require(clock()-started < limit and clock() < deadline,
                          'bounded process wall limit exhausted')
                sleeper(1.)
            sample(False)
            c.require(clock()-started < limit and clock() < deadline,
                      'bounded process wall limit exhausted at completion')
            return process.returncode
    finally:
        prior = {}
        for signum in (signal.SIGTERM, signal.SIGINT):
            prior[signum] = signal.signal(signum, signal.SIG_IGN)
        try:
            cleanup = host.clean_descendants(os.getpid(), process)
        finally:
            for signum, handler in prior.items():
                signal.signal(signum, handler)
        c.require(cleanup['remaining'] == {}, 'owned process cleanup incomplete')


def collect_matrix(stages, collect):
    """Collect each planned arm after task failures, stop on any engineering exception."""
    reports = []
    for stage in stages:
        report = collect(stage)
        c.require(report.get('status') == 'COMPLETE' and report.get('stage') == stage['name'],
                  'stage readback incomplete or wrong stage')
        counts = report.get('actual_counts', {})
        count = counts.get('controls')
        c.require(type(count) is int and 0 < count <= stage['controls']
                  and counts.get('normal_native') == 5*count
                  and counts.get('compiler_native') == 2, 'stage actual counts invalid')
        reports.append(report)
    return reports


def execute(output, *, root=c.ROOT):
    output = output.resolve()
    c.require(root == c.ROOT and output.parent == root/'results'
              and output.name.startswith('b22_suite31_') and not output.exists(),
              'use a new direct results/b22_suite31_* output; no retry or overwrite')
    started = time.monotonic()
    source = c.source_identity(root)
    c.require(memory() >= c.LIMITS['start_effective_bytes'], 'startup effective RAM below 6 GiB')
    c.require(shutil.disk_usage(root).free >= c.LIMITS['start_free_disk_bytes'], 'startup disk reserve insufficient')
    output.mkdir(exist_ok=False)
    report = {'schema': 'b22-suite31-supervision-v1', 'status': 'INCOMPLETE',
              'source': source, 'stages': [], 'failure': None, 'retry_permitted': False,
              'actual_counts': None, 'finite_suite_task_qualification_passed': None}
    old_handlers = {}
    def stopped(signum, frame):
        raise InterruptedError('suite interrupted: '+signal.Signals(signum).name)
    for signum in (signal.SIGTERM, signal.SIGINT):
        old_handlers[signum] = signal.signal(signum, stopped)
    old_alarm = signal.signal(signal.SIGALRM, stopped)
    signal.setitimer(signal.ITIMER_REAL, max(1., c.LIMITS['total_wall_s']-5))
    host = None
    deadline = started+c.LIMITS['total_wall_s']
    def sample(admission=False):
        available = memory()
        free = shutil.disk_usage(output).free
        row = {'elapsed_s': time.monotonic()-started, 'effective_available_bytes': available,
               'free_disk_bytes': free}
        with (output/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        minimum = report.get('minimum_effective_bytes')
        report['minimum_effective_bytes'] = available if minimum is None else min(minimum, available)
        minimum = c.LIMITS['start_effective_bytes' if admission else 'reserve_effective_bytes']
        c.require(available >= minimum, 'effective RAM below '+str(minimum >> 30)+' GiB '+
                  ('admission' if admission else 'runtime reserve'))
        c.require(free >= c.LIMITS['reserve_free_disk_bytes'], 'disk reserve exhausted')
        c.require(time.monotonic() < deadline, 'suite total wall exhausted')
    try:
        sample(True)
        config, readiness = c.registered_config(root)
        inputs = {**config['source_hashes'], **source['authored'], str(root/c.MANIFEST): c.identity(root/c.MANIFEST)}
        release = {'schema': 'b22-suite31-release-v1', 'repository': str(root),
                   'plan': c.plan(output), 'source': source, 'inputs': inputs,
                   'v27_manifest': c.identity(root/c.MANIFEST), 'v27_readiness': readiness}
        release_path = output/'release.json'
        c.write_json(release_path, release)
        host_path = Path(config['host'])
        host = c.load_pinned(host_path, inputs[str(host_path)], 'suite31_owned_host')
        host.subreaper()
        def collect(stage):
            sample(True)
            c.require(time.monotonic()+c.LIMITS['reader_wall_s']+10 < deadline,
                      'no remaining bounded reader/cleanup window')
            stage_output = output/stage['name']
            argv = (step_request(release, stage_output, config, release_path)
                    if stage['kind'] == 'step' else script_request(stage, release, stage_output, config, release_path))
            code = supervise(argv, output/(stage['name']+'_stdout.log'), host=host,
                             sample=sample, limit=c.LIMITS['stage_wall_s'], deadline=deadline)
            worker = c.read_json(stage_output/'worker_receipt.json')
            c.case_completion(stage, worker)
            # The new worker finalizes legitimate task failures normally. A nonzero
            # host exit remains an engineering failure, irrespective of task outcome.
            c.require(code == 0, 'launcher process failed')
            destination = output/(stage['name']+'_readback.json')
            read_argv = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(root/'scripts/read_b22_suite31.py'),
                         '--release', str(release_path), '--stage', stage['name'],
                         '--report', str(destination)]
            reader_code = supervise(read_argv, output/(stage['name']+'_reader.log'), host=host,
                                    sample=sample, limit=c.LIMITS['reader_wall_s'], deadline=deadline)
            c.require(reader_code == 0, 'independent stage reader failed')
            item = c.read_json(destination)
            c.require(item.get('release_identity') == c.identity(release_path), 'reader release differs')
            report['stages'].append({'name': stage['name'], 'report': c.identity(destination)})
            return item
        collected = collect_matrix(c.stages(), collect)
        destination = output/'comparison.json'
        argv = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(root/'scripts/read_b22_suite31.py'),
                '--release', str(release_path), '--compare', '--report', str(destination)]
        c.require(supervise(argv, output/'comparison_stdout.log', host=host, sample=sample,
                            limit=c.LIMITS['reader_wall_s'], deadline=deadline) == 0,
                  'independent suite comparison failed')
        compared = c.read_json(destination)
        c.require(compared.get('status') == 'COMPLETE'
                  and compared.get('release_identity') == c.identity(release_path), 'comparison incomplete')
        c.require(c.source_identity(root) == source, 'new execution source changed')
        total = {key: sum(item['actual_counts'][key] for item in collected)
                 for key in ('controls', 'normal_native', 'compiler_native')}
        c.require(all(total[key] <= c.LIMITS[key] for key in total), 'suite exceeded fixed physical budget')
        report.update(status='COMPLETE', actual_counts=total, comparison=c.identity(destination),
                      finite_suite_task_qualification_passed=compared['finite_suite_task_qualification_passed'],
                      RL_contribution_passed=compared['RL_contribution_passed'])
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, InterruptedError) as error:
        report['failure'] = type(error).__name__+': '+str(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        for signum in old_handlers:
            signal.signal(signum, signal.SIG_IGN)
        if host is not None:
            try:
                report['cleanup'] = host.clean_descendants(os.getpid(), None)
                if report['cleanup']['remaining']:
                    report.update(status='INCOMPLETE', failure='owned descendants remain after cleanup')
            except (OSError, subprocess.SubprocessError) as error:
                report.update(status='INCOMPLETE', failure='cleanup: '+str(error))
        report['elapsed_s'] = time.monotonic()-started
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        c.write_json(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(c.plan(args.output), indent=2, ensure_ascii=False))
        return 0
    if args.output is None:
        parser.error('--execute requires an explicit new --output')
    result = execute(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
