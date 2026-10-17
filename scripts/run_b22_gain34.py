"""Plan, or explicitly run, four new two-profile B22 residual-gain sessions."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTERED_ROOT = Path('/home/lyh/wheel-legged-control-lab')
AUTHORED = ('scripts/run_b22_gain34.py', 'scripts/run_b22_worker34.py',
            'scripts/read_b22_gain34.py', 'tests/test_b22_gain34.py', 'docs/b22_gain34.md')
PINS = {
    'scripts/run_b22_suite31.py': 'f783b4d7e8380c94bd7a4755e73434179cb5c3f6a95bbcca336e235a43320c2e',
    'scripts/b22_suite31_contract.py': '21386cfb3b73f0d947f825f00dd367c7443f3ca29738dfec5768925f30070203',
    'scripts/read_b22_suite31.py': '08a79bb53c449b90f3306d727326d250c2af83093d7ed1eab8ebe00e78d4ea17',
    'scripts/run_b22_worker31.py': '49feaecf2c6e68cf39c99238d5bde04b88dafb10de72479d0f3ce5c999e5d7c0',
}
BASELINE = 'results/b22_suite31_20261018_01'
BASELINE_RELEASE = {'bytes': 857698, 'sha256': '17c6926c731015e06174a592427546f0a69d7614830e5242a63b248b1caa6310'}
BASELINE_COMPARISON = {'bytes': 493845, 'sha256': 'f5dd040d05fe2a7b19811763ffc0b45fe6ab5c201a25a4da249f3431f6822456'}
BASELINE_OUTER = {'bytes': 4826, 'sha256': '2720b7f6913cf95b356adff44619b684e7401dcddfd5239207d935c13f382c6a'}
CONTRACT = 'C34_B22_residual_gain_probe_v1'
LIMITS = {'controls': 6400, 'normal_native': 32000, 'compiler_native': 8,
          'worker_sessions': 4, 'per_session_controls': 1600, 'total_wall_s': 900,
          'per_session_load': 1, 'per_session_torch_load': 3, 'per_session_predict': 1601, 'training_calls': 0,
          'stage_wall_s': 300, 'reader_wall_s': 180,
          'start_effective_bytes': 6 << 30, 'reserve_effective_bytes': 4 << 30}


def pinned(name, path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[str(path.relative_to(ROOT))]:
        raise ValueError('pinned suite31 input changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


for _name, _digest in PINS.items():
    if hashlib.sha256((ROOT/_name).read_bytes()).hexdigest() != _digest:
        raise ValueError('pinned suite31 input changed: '+_name)
common = pinned('gain34_supervision31', ROOT/'scripts/run_b22_suite31.py')
c = common.c


def stages():
    result = []
    for profile in ('flat_1p6', 'yaw_1p2'):
        base = next(item for item in c.stages() if item.get('profile') == profile and item.get('actor') == 'B')
        for label, gain in (('gain1', 1.0), ('gain0p5', .5)):
            result.append({**copy.deepcopy(base), 'name': profile+'_'+label, 'gain34': gain})
    return result


def plan(output=None):
    return {'schema': 'b22-gain34-plan-v1', 'status': 'PLAN_ONLY',
            'owner_instruction': '把这些优化点，在这台电脑能力之内的都完成',
            'output': None if output is None else str(Path(output).resolve()),
            'stages': stages(), 'limits': LIMITS, 'retry_permitted': False,
            'training_calls': 0, 'execution_started': False,
            'trigger': 'full31 COMPLETE; original-six prerequisites pass and contribution false; selected two zero/B full task pass, B SSE and cost both higher',
            'scope': 'two-profile local gain probe; four new B sessions; no six-task contribution/generalization claim'}


def trigger_decision(outer, comparison, reports):
    c.require(outer.get('status') == comparison.get('status') == 'COMPLETE', 'full31 is not engineering COMPLETE')
    original = comparison['original_six_contribution']
    c.require(original['prerequisites']['passed'] is True and original['RL_contribution_passed'] is False
              and comparison['RL_contribution_passed'] is False, 'original-six trigger prerequisites differ')
    c.require(set(reports) == {item['name'] for item in c.stages()}
              and all(item['status'] == 'COMPLETE' for item in reports.values()), 'full31 stage set incomplete')
    selected = {}
    for profile in ('flat_1p6', 'yaw_1p2'):
        zero, policy = (reports[profile+'_'+actor] for actor in ('zero', 'B'))
        c.require(zero['pairing_identity'] == policy['pairing_identity'], 'selected31 pair identity differs')
        a, b = zero['cases'][0], policy['cases'][0]
        c.require(all(row['task_passed'] is True and row['record']['full_horizon'] is True for row in (a, b))
                  and a in comparison['script_cases'] and b in comparison['script_cases'],
                  'selected31 tasks are incomplete, failed or unbound')
        values = [row['rl_terms'][key] for row in (a, b) for key in ('sse_total', 'torque_cost_mean')]
        c.require(all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in values),
                  'selected31 relative comparison has nonfinite/zero terms')
        sse = b['rl_terms']['sse_total']/a['rl_terms']['sse_total']
        cost = b['rl_terms']['torque_cost_mean']/a['rl_terms']['torque_cost_mean']
        c.require(sse > 1. and cost > 1., 'selected31 policy does not worsen both SSE and cost')
        selected[profile] = {'SSE_B_over_zero': sse, 'cost_B_over_zero': cost}
    return selected


def read_trigger():
    root = ROOT/BASELINE
    release_path, outer_path, comparison_path = (root/name for name in ('release.json', 'report.json', 'comparison.json'))
    c.require(c.identity(release_path) == BASELINE_RELEASE and c.identity(outer_path) == BASELINE_OUTER
              and c.identity(comparison_path) == BASELINE_COMPARISON, 'fixed suite31 final evidence differs')
    release, outer, comparison = map(c.read_json, (release_path, outer_path, comparison_path))
    c.require(outer.get('comparison') == c.identity(comparison_path)
              and outer['source'] == release['source']
              and comparison['release_identity'] == BASELINE_RELEASE, 'full31 comparison/source closure differs')
    evidence = {str(path): c.identity(path) for path in (release_path, outer_path, comparison_path)}
    reports = {}
    c.require(set(comparison['stage_reports']) == {item['name']+'_readback.json' for item in c.stages()},
              'full31 comparison stage hashes incomplete')
    for name, expected in comparison['stage_reports'].items():
        path = root/name
        c.require(c.identity(path) == expected, 'full31 stage readback changed: '+name)
        item = c.read_json(path)
        c.require(item['release_identity'] == BASELINE_RELEASE, 'full31 stage release differs')
        reports[item['stage']] = item
        evidence[str(path)] = expected
    return {'status': 'READY', 'selected_profiles': trigger_decision(outer, comparison, reports),
            'inputs': evidence, 'baseline_source_commit': release['source']['git_commit']}


def source_identity():
    subprocess.run(['git', 'ls-files', '--error-unmatch', *AUTHORED], cwd=ROOT,
                   check=True, stdout=subprocess.DEVNULL, timeout=5)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *AUTHORED], cwd=ROOT, check=True, timeout=5)
    return {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, timeout=5).strip(),
            'authored': {str(ROOT/name): c.identity(ROOT/name) for name in AUTHORED}}


def validate_release(path):
    release = c.read_json(path)
    c.require(release['schema'] == 'b22-gain34-release-v1'
              and release['plan'] == plan(release['plan']['output'])
              and Path(release['plan']['output']).parent == ROOT/'results'
              and Path(release['plan']['output']).name.startswith('b22_gain34_')
              and release['trigger'] == read_trigger(), 'gain34 release/trigger scope differs')
    config = c.read_json(ROOT/c.MANIFEST)
    c.require(c.identity(ROOT/c.MANIFEST)['sha256'] == c.MANIFEST_SHA
              and set(release['source']['authored']) == {str(ROOT/name) for name in AUTHORED},
              'gain34 manifest/authored scope differs')
    expected = {**config['source_hashes'], **release['source']['authored'], **release['trigger']['inputs'],
                **{str(ROOT/name): c.identity(ROOT/name) for name in PINS},
                str(ROOT/c.MANIFEST): c.identity(ROOT/c.MANIFEST)}
    c.require(release['inputs'] == expected, 'gain34 source closure differs')
    for name, expected in release['inputs'].items():
        c.require(not Path(name).is_symlink() and c.identity(name) == expected, 'gain34 input changed: '+name)
    return release


def request(stage, output, release, config, release_path):
    run, launch = output/stage['name'], output/(stage['name']+'_launch')
    launch.mkdir(exist_ok=False)
    spec = {'schema': 'd1-c23-gui-worker-v1', 'mode': 'script', 'render': False,
            'profile': stage['profile'], 'actor': 'B', 'seed': stage['seed'], 'mirror': False,
            'control_limit': 1600, 'seconds': 16., 'gain34': stage['gain34'],
            'output_directory': str(run), 'qualification': False,
            'model_limits': c.model_limits('B', 1600, config['model_limit_keys']),
            'soft_s': 180, 'close_s': 235, 'hard_s': 240, 'frame_limit': 1000,
            'require_event_driver': False, 'execution_contract_id': CONTRACT}
    go = {'decision': 'GO', 'scope': 'new34 local residual gain probe; no training', 'repository': str(ROOT),
          'worker': str(ROOT/'scripts/run_b22_worker34.py'),
          'inputs': {**release['inputs'], str(release_path): c.identity(release_path)},
          'runtime_environment': config['runtime_environment'], 'shared_session': config['shared_session'],
          'arms': {stage['name']: spec}, 'x11_events_by_arm': {}}
    c.write_json(launch/'source_go.json', go)
    c.write_json(launch/'request.json', {'source_go_path': str(launch/'source_go.json'), 'arm': stage['name'], 'spec': spec})
    return ['rtk', 'proxy', '/usr/bin/python3', '-B', config['host'], '--request', str(launch/'request.json')]


def stage_counts(stage, saved, release_identity, reader_identity):
    counts = saved.get('actual_counts', {})
    count = counts.get('controls')
    c.require(saved.get('schema') == 'b22-gain34-stage-readback-v1'
              and saved.get('status') == 'COMPLETE' and saved.get('stage') == stage['name']
              and saved.get('execution_contract_id') == CONTRACT
              and type(saved.get('gain34')) is float and saved['gain34'] == stage['gain34']
              and saved.get('release_identity') == release_identity
              and saved.get('reader_identity') == reader_identity
              and type(count) is int and 0 < count <= 1600
              and counts.get('normal_native') == 5*count and counts.get('compiler_native') == 2
              and saved.get('gain_ledger_rows') == count, 'gain34 stage identity/actual counters incomplete')
    return {key: counts[key] for key in ('controls', 'normal_native', 'compiler_native')}


def execute(output):
    output = Path(output).resolve()
    c.require(ROOT == REGISTERED_ROOT and output.parent == ROOT/'results'
              and output.name.startswith('b22_gain34_') and not output.exists(), 'new registered-machine gain34 output required')
    started, source, trigger = time.monotonic(), source_identity(), read_trigger()
    c.require(common.memory() >= 6 << 30 and shutil.disk_usage(ROOT).free >= 2 << 30, 'NOT_READY: startup resources')
    output.mkdir(exist_ok=False)
    report = {'schema': 'b22-gain34-supervision-v1', 'status': 'INCOMPLETE', 'failure': None,
              'source': source, 'actual_counts': None, 'stage_reports': {}, 'retry_permitted': False}
    host, previous, deadline = None, {}, started+890
    def stopped(signum, frame):
        raise InterruptedError('gain34 interrupted: '+signal.Signals(signum).name)
    def sample(admission=False):
        available = common.memory()
        with (output/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps({'elapsed_s': time.monotonic()-started, 'effective_available_bytes': available})+'\n')
        report['minimum_effective_bytes'] = min(report.get('minimum_effective_bytes', available), available)
        c.require(available >= (6 << 30 if admission else 4 << 30), 'gain34 RAM admission/runtime reserve exhausted')
        c.require(shutil.disk_usage(output).free >= 1 << 30 and time.monotonic() < deadline, 'gain34 disk/wall reserve exhausted')
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, stopped)
    old_alarm = signal.signal(signal.SIGALRM, stopped)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline-time.monotonic()))
    try:
        config, readiness = c.registered_config()
        frozen = {**config['source_hashes'], **source['authored'], **trigger['inputs'],
                  **{str(ROOT/name): c.identity(ROOT/name) for name in PINS},
                  str(ROOT/c.MANIFEST): c.identity(ROOT/c.MANIFEST)}
        release = {'schema': 'b22-gain34-release-v1', 'plan': plan(output), 'source': source,
                   'trigger': trigger, 'inputs': frozen, 'v27_readiness': readiness}
        release_path = output/'release.json'
        c.write_json(release_path, release)
        validate_release(release_path)
        host = c.load_pinned(Path(config['host']), frozen[config['host']], 'gain34_owned_host')
        host.subreaper()
        total = {'controls': 0, 'normal_native': 0, 'compiler_native': 0}
        for stage in stages():
            sample(True)
            argv = request(stage, output, release, config, release_path)
            c.require(common.supervise(argv, output/(stage['name']+'_stdout.log'), host=host, sample=sample,
                                       limit=LIMITS['stage_wall_s'], deadline=deadline) == 0, 'gain34 worker/host failed')
            destination = output/(stage['name']+'_readback.json')
            argv = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(ROOT/'scripts/read_b22_gain34.py'),
                    '--release', str(release_path), '--stage', stage['name'], '--report', str(destination)]
            c.require(common.supervise(argv, output/(stage['name']+'_reader.log'), host=host, sample=sample,
                                       limit=LIMITS['reader_wall_s'], deadline=deadline) == 0, 'gain34 saved reader failed')
            saved = c.read_json(destination)
            counts = stage_counts(stage, saved, c.identity(release_path),
                                  frozen[str(ROOT/'scripts/read_b22_gain34.py')])
            for key in total:
                total[key] += counts[key]
            report['stage_reports'][destination.name] = c.identity(destination)
        argv = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(ROOT/'scripts/read_b22_gain34.py'),
                '--release', str(release_path), '--compare', '--report', str(output/'comparison.json')]
        c.require(common.supervise(argv, output/'comparison_stdout.log', host=host, sample=sample,
                                   limit=LIMITS['reader_wall_s'], deadline=deadline) == 0, 'gain34 comparison failed')
        compared = c.read_json(output/'comparison.json')
        c.require(compared.get('schema') == 'b22-gain34-comparison-v1' and compared['status'] == 'COMPLETE'
                  and compared['release_identity'] == c.identity(release_path)
                  and compared.get('stage_reports') == report['stage_reports']
                  and all(total[k] <= LIMITS[k] for k in total) and source_identity() == source,
                  'gain34 final identity/counts incomplete')
        sample(False)
        report.update(status='COMPLETE', actual_counts=total, comparison=c.identity(output/'comparison.json'))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report['failure'] = type(error).__name__+': '+str(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        if host is not None:
            try:
                report['cleanup'] = host.clean_descendants(os.getpid(), None)
                if report['cleanup']['remaining']:
                    report.update(status='INCOMPLETE', failure='owned descendants remain')
            except (OSError, subprocess.SubprocessError) as error:
                report.update(status='INCOMPLETE', failure='cleanup: '+str(error))
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        report['elapsed_s'] = time.monotonic()-started
        c.write_json(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--execute', action='store_true')
    action.add_argument('--check-trigger', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.check_trigger:
        try:
            result = read_trigger()
        except (OSError, ValueError, KeyError, TypeError) as error:
            result = {'status': 'NOT_READY', 'reason': str(error)}
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'READY' else 1
    if not args.execute:
        print(json.dumps(plan(args.output), ensure_ascii=False, indent=2))
        return 0
    if args.output is None:
        parser.error('--execute requires an explicit new --output')
    report = execute(args.output)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
