"""Plan or explicitly run the complete new zero/half-gain B22 matrix."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTERED_ROOT = Path('/home/lyh/wheel-legged-control-lab')
CONTRACT = 'C35_B22_half_gain_fixed_suite_v1'
AUTHORED = ('scripts/run_b22_gain_suite35.py', 'scripts/run_b22_worker35.py',
            'scripts/read_b22_gain_suite35.py', 'tests/test_b22_gain_suite35.py', 'docs/b22_gain_suite35.md')
PINS = {
    'scripts/run_b22_suite31.py': 'f783b4d7e8380c94bd7a4755e73434179cb5c3f6a95bbcca336e235a43320c2e',
    'scripts/b22_suite31_contract.py': '21386cfb3b73f0d947f825f00dd367c7443f3ca29738dfec5768925f30070203',
    'scripts/read_b22_suite31.py': '08a79bb53c449b90f3306d727326d250c2af83093d7ed1eab8ebe00e78d4ea17',
    'scripts/run_b22_worker34.py': '2f71895b709ec1abbd947c96673ec969de3a483bb4d145843315754513c3eb5d',
    'runtime/d1_b22/research/continuation24/worker24.py': '638f518e2cba500f60bee1ea1cc8a6e8e59d0fd4452ca345a6bed090bc85bb2b',
    'runtime/d1_b22/research/continuation24/eval24.py': '927e373aa41bd6687ea6d06b36c0b7b8bee6722c3e698a44c171a2f3f465d073',
    'runtime/d1_b22/research/continuation24/heldout24.py': '0b40b73c834dba65f4cc45a8ea18267a48c32b46d69799a31ec7846f3509f92f',
    'runtime/d1_b22/research/continuation24/read24.py': 'a7e3cd1c5fe718c634dd8690d824cdcf5814de191f712698c81a38e186379464',
}
TRIGGERS = {
    'b22_suite31_20261018_01': {
        'release.json': {'bytes': 857698, 'sha256': '17c6926c731015e06174a592427546f0a69d7614830e5242a63b248b1caa6310'},
        'report.json': {'bytes': 4826, 'sha256': '2720b7f6913cf95b356adff44619b684e7401dcddfd5239207d935c13f382c6a'},
        'comparison.json': {'bytes': 493845, 'sha256': 'f5dd040d05fe2a7b19811763ffc0b45fe6ab5c201a25a4da249f3431f6822456'}},
    'b22_gain34_20261018_01': {
        'release.json': {'bytes': 861364, 'sha256': '01e486e8dcc561757949c9a6635974723905d5cccd3a637e5375ac2f1952df4a'},
        'report.json': {'bytes': 2255, 'sha256': '892a6ee5cc529ff06d6d17072c5f9f8014350374696f9c5ba641610cfa6b5d1e'},
        'comparison.json': {'bytes': 116128, 'sha256': '831c631a5a81721cf963e4fccaecab31d73648ae7111dbbfff9ae083e3bbff11'}},
}
LIMITS = {'controls': 25200, 'normal_native': 126000, 'compiler_native': 30,
          'logical_arm_records': 16, 'worker_sessions': 15, 'total_wall_s': 1800,
          'stage_wall_s': 300, 'reader_wall_s': 180, 'start_effective_bytes': 6 << 30,
          'reserve_effective_bytes': 4 << 30, 'start_free_disk_bytes': 2 << 30,
          'reserve_free_disk_bytes': 1 << 30, 'required_minimum_nice': 10, 'training_calls': 0}


def pinned(name, path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[str(path.relative_to(ROOT))]:
        raise ValueError('C35 pinned source changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


for _name, _digest in PINS.items():
    if hashlib.sha256((ROOT/_name).read_bytes()).hexdigest() != _digest:
        raise ValueError('C35 pinned source changed: '+_name)
common = pinned('gain_suite35_supervision31', ROOT/'scripts/run_b22_suite31.py')
c = common.c


def stages():
    return [{**copy.deepcopy(stage), 'gain35': .5} for stage in c.stages()]


def plan(output=None):
    return {'schema': 'b22-gain-suite35-plan-v1', 'status': 'PLAN_ONLY',
            'owner_instruction': '把这些优化点，在这台电脑能力之内的都完成',
            'output': None if output is None else str(Path(output).resolve()),
            'stages': stages(), 'limits': LIMITS, 'execution_started': False,
            'retry_permitted': False, 'training_calls': 0, 'automatic_model_replacement': False,
            'scope': 'fresh zero/half-gain pairs on original seven scripts and true 15mm step; unchanged original-six contribution gate'}


def trigger_decision(baseline, probe):
    c.require(baseline['status'] == probe['status'] == 'COMPLETE'
              and baseline['original_six_contribution']['prerequisites']['passed'] is True
              and baseline['RL_contribution_passed'] is False, 'C31 complete contribution trigger differs')
    c.require(probe['all_four_full_horizons'] is True and probe['all_four_tasks_passed'] is True
              and probe['all_two_profile_SSE_and_cost_lower'] is True and probe['full_window_ratios_available'] is True,
              'C34 full two-profile gain improvement trigger differs')
    pairs = probe['pairs']
    c.require([row['profile'] for row in pairs] == ['flat_1p6', 'flat_1p2_yaw']
              and all(0 < row['SSE_gain0p5_over_gain1'] < 1. and 0 < row['cost_gain0p5_over_gain1'] < 1. for row in pairs),
              'C34 selected improvements differ')
    return {'original_six_contribution': False, 'local_two_profile_improvement': True,
            'scope': 'screening trigger only; neither old run supplies a fresh C35 paired arm'}


def read_trigger():
    evidence, comparisons = {}, {}
    for name, summaries in TRIGGERS.items():
        root = ROOT/'results'/name
        for filename, expected in summaries.items():
            path = root/filename
            c.require(c.identity(path) == expected, 'fixed C31/C34 trigger changed: '+str(path))
            evidence[str(path)] = expected
        release, outer, compared = (c.read_json(root/file) for file in ('release.json', 'report.json', 'comparison.json'))
        rid = summaries['release.json']
        c.require(outer['status'] == compared['status'] == 'COMPLETE'
                  and outer['comparison'] == summaries['comparison.json'] and outer['source'] == release['source']
                  and compared['release_identity'] == rid, 'C31/C34 trigger closure differs')
        expected_names = ({stage['name']+'_readback.json' for stage in c.stages()} if 'suite31' in name else
                          {profile+'_'+gain+'_readback.json' for profile in ('flat_1p6', 'yaw_1p2') for gain in ('gain1', 'gain0p5')})
        c.require(set(compared['stage_reports']) == expected_names, 'C31/C34 trigger stage set differs')
        for filename, expected in compared['stage_reports'].items():
            path = root/filename
            item = c.read_json(path)
            c.require(c.identity(path) == expected and item['status'] == 'COMPLETE'
                      and item['release_identity'] == rid, 'C31/C34 readback changed: '+str(path))
            evidence[str(path)] = expected
        comparisons[name] = compared
    return {'status': 'READY', 'decision': trigger_decision(*comparisons.values()), 'inputs': evidence}


def source_identity():
    subprocess.run(['git', 'ls-files', '--error-unmatch', *AUTHORED], cwd=ROOT, check=True, stdout=subprocess.DEVNULL, timeout=5)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *AUTHORED], cwd=ROOT, check=True, timeout=5)
    return {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, timeout=5).strip(),
            'authored': {str(ROOT/name): c.identity(ROOT/name) for name in AUTHORED}}


def closure(source, trigger, config):
    return {**config['source_hashes'], **source['authored'], **trigger['inputs'],
            **{str(ROOT/name): c.identity(ROOT/name) for name in PINS}, str(ROOT/c.MANIFEST): c.identity(ROOT/c.MANIFEST)}


def validate_release(path):
    release = c.read_json(path)
    output = Path(release['plan']['output'])
    c.require(release['schema'] == 'b22-gain-suite35-release-v1' and release['plan'] == plan(output)
              and output.parent == ROOT/'results' and output.name.startswith('b22_gain_suite35_')
              and release['trigger'] == read_trigger(), 'C35 fixed release/trigger differs')
    config = c.read_json(ROOT/c.MANIFEST)
    c.require(c.identity(ROOT/c.MANIFEST)['sha256'] == c.MANIFEST_SHA
              and set(release['source']['authored']) == {str(ROOT/name) for name in AUTHORED}
              and release['inputs'] == closure(release['source'], release['trigger'], config), 'C35 source closure differs')
    for name, expected in release['inputs'].items():
        c.require(not Path(name).is_symlink() and c.identity(name) == expected, 'C35 frozen input changed: '+name)
    return release


def request(stage, output, release, config, release_path):
    run, launch = output/stage['name'], output/(stage['name']+'_launch')
    launch.mkdir(exist_ok=False)
    spec = {'mode': 'eval' if stage['kind'] == 'step' else 'script', 'render': False,
            'control_limit': stage['controls'], 'controller_variant': 'combined',
            'output_directory': str(run), 'gain35': .5, 'execution_contract_id': CONTRACT,
            'soft_s': 240 if stage['kind'] == 'step' else 180, 'close_s': 270 if stage['kind'] == 'step' else 235,
            'hard_s': 300 if stage['kind'] == 'step' else 240,
            'model_limits': c.model_limits('B' if stage['kind'] == 'step' else stage['actor'],
                                          1200 if stage['kind'] == 'step' else stage['controls'], config['model_limit_keys'])}
    if stage['kind'] == 'step':
        spec.update(arm='step_pair', controls=2400)
    else:
        spec.update(schema='d1-c23-gui-worker-v1', profile=stage['profile'], actor=stage['actor'], seed=stage['seed'],
                    mirror=stage['mirror'], seconds=stage['controls']*.01, qualification=False, frame_limit=1000,
                    require_event_driver=False)
    go = {'decision': 'GO', 'scope': 'new C35 full zero/half-gain matrix; no training', 'repository': str(ROOT),
          'worker': str(ROOT/'scripts/run_b22_worker35.py'), 'inputs': {**release['inputs'], str(release_path): c.identity(release_path)},
          'runtime_environment': config['runtime_environment'], 'shared_session': config['shared_session'],
          'arms': {stage['name']: spec}, 'x11_events_by_arm': {}}
    c.write_json(launch/'source_go.json', go)
    c.write_json(launch/'request.json', {'source_go_path': str(launch/'source_go.json'), 'arm': stage['name'], 'spec': spec})
    return ['rtk', 'proxy', '/usr/bin/python3', '-B', config['host'], '--request', str(launch/'request.json')]


def case_completion(stage, worker):
    c.require(worker.get('execution_contract_id') == CONTRACT, 'C35 worker contract differs')
    if stage['kind'] != 'step':
        return c.case_completion(stage, worker)
    # Preserve the entire old count/ending check; explicitly adapt only its
    # versioned receipt schema, never the saved document.
    node = next(n for n in ast.parse((ROOT/'scripts/b22_suite31_contract.py').read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'case_completion')
    found = 0
    for item in ast.walk(node):
        if isinstance(item, ast.Constant) and item.value == 'd1-true15mm-worker-24-v1':
            item.value = 'd1-true15mm-worker-35-v1'
            found += 1
    c.require(found == 1, 'C35 step receipt seam changed')
    namespace = dict(c.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), __file__, 'exec'), namespace)  # noqa: S102 -- pinned pure receipt schema adaptation
    return namespace['case_completion'](stage, worker)


def stage_counts(stage, saved, release_id, reader_id):
    count = saved.get('actual_counts', {}).get('controls')
    c.require(saved.get('schema') == 'b22-gain-suite35-stage-readback-v1' and saved.get('status') == 'COMPLETE'
              and saved.get('stage') == stage['name'] and saved.get('execution_contract_id') == CONTRACT
              and saved.get('gain35') == .5 and saved.get('release_identity') == release_id
              and saved.get('reader_identity') == reader_id and type(count) is int and 0 < count <= stage['controls']
              and saved['actual_counts'].get('normal_native') == 5*count and saved['actual_counts'].get('compiler_native') == 2
              and saved.get('gain_ledger_rows') == count, 'C35 stage identity/actual counters differ')
    return {key: saved['actual_counts'][key] for key in ('controls', 'normal_native', 'compiler_native')}


def known_partial(stage, run, release):
    """Reliable owned receipt prefix only; absent/invalid receipts stay unknown."""
    try:
        session, worker = (c.read_json(run/name) for name in ('session.json', 'worker_receipt.json'))
        c.require(session.get('execution_contract_id') == worker.get('execution_contract_id') == CONTRACT
                  and session.get('gain35') == .5 and session.get('output_directory') == str(run)
                  and worker.get('session_identity') == c.identity(run/'session.json')
                  and all(session['source_hashes'].get(k) == v for k, v in release['inputs'].items()), 'foreign partial C35 receipt')
        result = worker.get('result', {})
        py = worker.get('python') if stage['kind'] == 'step' else result.get('python')
        state = worker.get('C_final') if stage['kind'] == 'step' else result.get('C_final')
        n, native, compiler = py['control_completed'], py['native_returned'], state['construction_returns']
        c.require(all(type(v) is int for v in (n, native, compiler)) and 0 <= n <= stage['controls']
                  and 5*n <= native <= min(5*stage['controls'], 5*(n+1)) and 0 <= compiler <= 2,
                  'invalid partial C35 physical prefix')
        return {'controls': n, 'normal_native': native, 'compiler_native': compiler}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def counted_prefix(completed, active=None, partial=None):
    """Rebuild unique stage facts, including interruption during finalization."""
    rows = dict(completed)
    if active is not None and active['name'] not in rows and partial is not None:
        rows[active['name']] = partial
    return {key: sum(row[key] for row in rows.values()) for key in ('controls', 'normal_native', 'compiler_native')}


def execute(output):
    output = Path(output).resolve()
    c.require(ROOT == REGISTERED_ROOT and output.parent == ROOT/'results' and output.name.startswith('b22_gain_suite35_')
              and not output.exists(), 'new registered-machine C35 output required; no retry')
    started, source, trigger = time.monotonic(), source_identity(), read_trigger()
    c.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'C35 requires external Nice=10 or lower priority')
    c.require(common.memory() >= 6 << 30 and shutil.disk_usage(ROOT).free >= 2 << 30, 'NOT_READY: C35 startup resources')
    output.mkdir(exist_ok=False)
    completed = {}
    total = counted_prefix(completed)
    report = {'schema': 'b22-gain-suite35-supervision-v1', 'status': 'INCOMPLETE', 'failure': None, 'source': source,
              'actual_counts': None, 'known_actual_counts': dict(total), 'stage_reports': {}, 'retry_permitted': False,
              'automatic_model_replacement': False, 'nice': os.getpriority(os.PRIO_PROCESS, 0)}
    host, active, release, previous, deadline = None, None, None, {}, started+1790
    def stopped(signum, frame):
        raise InterruptedError('C35 interrupted: '+signal.Signals(signum).name)
    def sample(admission=False):
        available, free = common.memory(), shutil.disk_usage(output).free
        with (output/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps({'elapsed_s': time.monotonic()-started, 'effective_available_bytes': available,
                                     'free_disk_bytes': free})+'\n')
        report['minimum_effective_bytes'] = min(report.get('minimum_effective_bytes', available), available)
        c.require(available >= (6 << 30 if admission else 4 << 30), 'C35 RAM admission/runtime reserve exhausted')
        c.require(free >= 1 << 30 and time.monotonic() < deadline, 'C35 disk/wall reserve exhausted')
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, stopped)
    old_alarm = signal.signal(signal.SIGALRM, stopped)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline-time.monotonic()))
    try:
        config, readiness = c.registered_config()
        release = {'schema': 'b22-gain-suite35-release-v1', 'plan': plan(output), 'source': source,
                   'trigger': trigger, 'inputs': closure(source, trigger, config), 'v27_readiness': readiness}
        release_path = output/'release.json'
        c.write_json(release_path, release)
        validate_release(release_path)
        host = c.load_pinned(Path(config['host']), release['inputs'][config['host']], 'gain_suite35_owned_host')
        host.subreaper()
        for active in stages():
            sample(True)
            c.require(time.monotonic()+LIMITS['reader_wall_s']+10 < deadline, 'C35 no bounded reader/cleanup window')
            argv = request(active, output, release, config, release_path)
            code = common.supervise(argv, output/(active['name']+'_stdout.log'), host=host, sample=sample,
                                    limit=LIMITS['stage_wall_s'], deadline=deadline)
            partial = known_partial(active, output/active['name'], release)
            report['known_actual_counts'] = counted_prefix(completed, active, partial)
            case_completion(active, c.read_json(output/active['name']/'worker_receipt.json'))
            c.require(code == 0, 'C35 worker/host failed')
            destination = output/(active['name']+'_readback.json')
            argv = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(ROOT/'scripts/read_b22_gain_suite35.py'),
                    '--release', str(release_path), '--stage', active['name'], '--report', str(destination)]
            c.require(common.supervise(argv, output/(active['name']+'_reader.log'), host=host, sample=sample,
                                       limit=LIMITS['reader_wall_s'], deadline=deadline) == 0, 'C35 saved reader failed')
            saved = c.read_json(destination)
            counts = stage_counts(active, saved, c.identity(release_path), release['inputs'][str(ROOT/'scripts/read_b22_gain_suite35.py')])
            completed[active['name']] = counts
            total = counted_prefix(completed)
            c.require(all(total[key] <= LIMITS[key] for key in total), 'C35 physical budget exhausted')
            report['stage_reports'][destination.name] = c.identity(destination)
            report['known_actual_counts'] = dict(total)
            active = None
        argv = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(ROOT/'scripts/read_b22_gain_suite35.py'),
                '--release', str(release_path), '--compare', '--report', str(output/'comparison.json')]
        c.require(common.supervise(argv, output/'comparison_stdout.log', host=host, sample=sample,
                                   limit=LIMITS['reader_wall_s'], deadline=deadline) == 0, 'C35 comparison failed')
        compared = c.read_json(output/'comparison.json')
        c.require(compared.get('schema') == 'b22-gain-suite35-comparison-v1' and compared['status'] == 'COMPLETE'
                  and compared['release_identity'] == c.identity(release_path) and compared['stage_reports'] == report['stage_reports']
                  and source_identity() == source, 'C35 final identity differs')
        sample(False)
        report.update(status='COMPLETE', actual_counts=total, comparison=c.identity(output/'comparison.json'),
                      finite_suite_task_qualification_passed=compared['finite_suite_task_qualification_passed'],
                      RL_contribution_passed=compared['RL_contribution_passed'], no_task_regression=compared['no_task_regression'])
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report['failure'] = type(error).__name__+': '+str(error)
        if active is not None and release is not None:
            partial = known_partial(active, output/active['name'], release)
            report['known_actual_counts'] = counted_prefix(completed, active, partial)
            report['active_stage_actual_counts'] = partial
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        if host is not None:
            try:
                report['cleanup'] = host.clean_descendants(os.getpid(), None)
                if report['cleanup']['remaining']:
                    report.update(status='INCOMPLETE', failure='C35 owned descendants remain', actual_counts=None)
            except (OSError, subprocess.SubprocessError) as error:
                report.update(status='INCOMPLETE', failure='C35 cleanup: '+str(error), actual_counts=None)
        report['elapsed_s'] = time.monotonic()-started
        if report['elapsed_s'] >= LIMITS['total_wall_s']:
            report.update(status='INCOMPLETE', failure='C35 final wall limit exhausted', actual_counts=None)
        if report['status'] != 'COMPLETE':
            report.update(finite_suite_task_qualification_passed=None, RL_contribution_passed=None, no_task_regression=None)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        c.write_json(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(plan(args.output), indent=2, ensure_ascii=False))
        return 0
    if args.output is None:
        parser.error('--execute requires an explicit new --output')
    result = execute(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
