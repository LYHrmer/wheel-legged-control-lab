"""One current-environment repetition of the unchanged C26 XTest protocol."""
from __future__ import annotations

import argparse
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
AUTHORED = ('scripts/run_b22_interaction33.py', 'tests/test_b22_interaction33.py',
            'docs/b22_interaction33.md')
GO = 'results/d1_b22_gui_true15mm_20260930/continuation26/gui_source_go_26.json'
GO_ID = {'bytes': 895449, 'sha256': 'f819a94455a5561eed702545e53d901cb301ed712f904434acf4a5beb77e1315'}
ISOLATION = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/gui_isolation/isolated_x11.py')
EXTRAS = {
    str(ISOLATION): {'bytes': 6096, 'sha256': '33dceb3402f2a73c664e9f4cfefff67423cf117c78ba588ae43b31eb3bc8e9c4'},
    str(ISOLATION.parent/'rootfs/usr/bin/Xvfb'): {
        'bytes': 2073056, 'sha256': 'cf71d78eab8321f7c568271053af449f966a75422140117ea1f244bf72a7e81e'},
}
HELPERS = {
    'scripts/run_b22_suite31.py': '698530945f85181223e6d46a3738f5ddb64cf60e9458bb3a31dab6f7c38e874f',
    'scripts/b22_suite31_contract.py': 'a1a32d3d6e65913401c966347545c7608921db00a7986e81370cc00c0a6a9e19',
}
WORKER = 'runtime/d1_b22/research/continuation26/gui26/run_gui26.py'
READER = 'runtime/d1_b22/research/continuation26/root26/reader26_deferred.py'
BASE_READER = 'runtime/d1_b22/research/continuation26/root26/reader26.py'
TOTAL_S, RUN_S, READER_S = 900, 600, 180
CONTRACT = 'C26_B22_GUI_owner_close_v1'


def _shared():
    # Pin both executable helper files before either is imported. No model imports.
    for name, expected in HELPERS.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('committed suite31 helper changed: '+name)
    spec = importlib.util.spec_from_file_location('interaction33_supervision31', ROOT/'scripts/run_b22_suite31.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _shared()
c = common.c


def template():
    c.require(c.identity(ROOT/GO) == GO_ID, 'historical C26 event source changed')
    original = c.read_json(ROOT/GO)
    c.require(original['decision'] == 'GO' and original['worker'] == str(REGISTERED_ROOT/WORKER)
              and all(original['inputs'].get(name) == value for name, value in EXTRAS.items()),
              'historical C26 worker/isolation evidence differs')
    return copy.deepcopy(original['arms']['keyboard_gui']), copy.deepcopy(original['x11_events_by_arm']['keyboard_gui'])


def plan(output=None):
    spec, events = template()
    spec.pop('output_directory')
    return {'schema': 'b22-interaction33-plan-v1', 'status': 'PLAN_ONLY',
            'owner_instruction': '把这些优化点，在这台电脑能力之内的都完成',
            'scope': 'one new C26 keyboard/OS-input repetition on an owned private X11 server; no training',
            'output': None if output is None else str(Path(output).resolve()),
            'execution_spec_without_output': spec, 'x11_events': events,
            'limits': {'controls': 1200, 'normal_native': 6000, 'compiler_native': 2,
                       'load': 1, 'torch_load': 3, 'predict': 1201, 'total_wall_s': TOTAL_S,
                       'host_wall_s': RUN_S, 'saved_reader_wall_s': READER_S,
                       'start_effective_bytes': 6 << 30, 'reserve_effective_bytes': 3 << 30},
            'reset_refunds_budget': False, 'retry_permitted': False,
            'physical_human_keyboard_tested': False, 'historical_GO_is_new_authorization': False,
            'execution_started': False, 'model_loads': 0, 'physics_steps': 0}


def source_identity():
    subprocess.run(['git', 'ls-files', '--error-unmatch', *AUTHORED], cwd=ROOT,
                   stdout=subprocess.DEVNULL, check=True, timeout=5)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *AUTHORED], cwd=ROOT, check=True, timeout=5)
    return {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                  text=True, timeout=5).strip(),
            'authored': {str(ROOT/name): c.identity(ROOT/name) for name in AUTHORED}}


def inputs(config, source):
    return {**config['source_hashes'], **source['authored'], **EXTRAS,
            **{str(ROOT/name): c.identity(ROOT/name) for name in HELPERS},
            str(ROOT/GO): GO_ID, str(ROOT/c.MANIFEST): c.identity(ROOT/c.MANIFEST)}


def validate_sources(frozen):
    for name, expected in frozen.items():
        c.require(Path(name).is_absolute() and not Path(name).is_symlink()
                  and c.identity(name) == expected, 'interaction33 input changed: '+name)


def make_request(output, config, release, release_path):
    run, launch = output/'keyboard_gui', output/'launch'
    launch.mkdir(exist_ok=False)
    spec, events = template()
    spec['output_directory'] = str(run)
    go = {'decision': 'GO', 'scope': 'new interaction33 owner authorization; unchanged C26 numerical/input protocol',
          'repository': str(ROOT), 'worker': str(ROOT/WORKER),
          'inputs': {**release['inputs'], str(release_path): c.identity(release_path)},
          'runtime_environment': config['runtime_environment'], 'shared_session': config['shared_session'],
          'arms': {'keyboard_gui': spec}, 'x11_events_by_arm': {'keyboard_gui': events}}
    c.write_json(launch/'source_go.json', go)
    c.write_json(launch/'request.json', {'source_go_path': str(launch/'source_go.json'),
                                      'arm': 'keyboard_gui', 'spec': spec, 'x11_events': events})
    return ['rtk', 'proxy', '/usr/bin/python3', '-B', config['host'],
            '--request', str(launch/'request.json'), '--isolated-x11', str(ISOLATION)]


def readback_argv(output):
    return ['rtk', 'proxy', '/usr/bin/python3', '-B', str(ROOT/READER),
            '--run', str(output/'keyboard_gui'), '--output', str(output/'readback.json')]


def validate_isolation(receipt, session, output, config):
    run = output/'keyboard_gui'
    log = output/'keyboard_gui.x11'
    display = receipt.get('display', '')
    c.require(display.startswith(':') and display[1:].isdigit() and 100 <= int(display[1:]) < 200
              and receipt.get('child_output') == str(run) and receipt.get('logs') == str(log)
              and receipt.get('command') == ['rtk', 'proxy', '/usr/bin/python3', '-B', config['host'],
                  '--request', str(output/'launch/request.json'), '--owned-worker-host']
              and receipt.get('status') == 'passed' and receipt.get('child_returncode') == 0
              and receipt.get('auth_cookie_logged') is False and receipt.get('auth_removed') is True
              and receipt.get('owned_server_stopped') is True
              and session['runtime_environment']['DISPLAY'] == display
              and session['runtime_environment']['XAUTHORITY'] == str(log/'Xauthority'),
              'owned X11 server/session/cleanup binding differs')


def validate_readback(report, *, run, source_inputs, returncode, session_id, worker_id):
    c.require(report.get('schema') == 'd1-c26-deferred-independent-readback-v1'
              and report.get('run') == str(run) and report.get('arm') == 'keyboard_gui'
              and report.get('execution_contract_id') == CONTRACT and report.get('record_valid') is True
              and report.get('source_and_actual_origin_verified') is True
              and report.get('session_identity') == session_id and report.get('worker_receipt_identity') == worker_id
              and report.get('reader_source_identity') == source_inputs[str(ROOT/READER)]
              and report.get('base_reader_source_identity') == source_inputs[str(ROOT/BASE_READER)]
              and report.get('checkpoint_sha256') == c.MODEL_SHA
              and report.get('observation_action_shape') == [99, 16]
              and report.get('reader_model_calls') == report.get('reader_physics_calls') == 0,
              'independent C26 reader identity/integrity differs')
    count = report.get('completed_controls')
    c.require(type(count) is int and 0 < count <= 1200
              and report.get('normal_native_returns') == 5*count and report.get('compiler_native_returns') == 2
              and report.get('actual_episode_count') == 2
              and report.get('input', {}).get('actual_reset_count') == 1
              and report['input'].get('controls_with_exact_preparation') == count,
              'actual input consumption/reset/global physical count differs')
    calls = report.get('model_calls', {})
    counts = calls.get('counts', {})
    expected = {'load': 1, 'torch_load': 3, 'predict': count+1, 'save': 0, 'learn': 0, 'train': 0,
                'forward': 0, 'evaluate_actions': 0, 'predict_values': 0, 'backward': 0}
    c.require(all(counts.get(key, {}).get('attempted', 0) == counts.get(key, {}).get('returned', 0) == value
                  for key, value in expected.items())
              and calls.get('actor_rows') == count+32 and calls.get('critic_rows') == 0,
              'actual B22 model/probe budget differs')
    qualified = report.get('arm_qualification_passed')
    c.require(type(qualified) is bool and report.get('execution_prerequisite_passed') is qualified
              and returncode == (0 if qualified else 2), 'reader exit/qualification mismatch')
    # The frozen reader establishes these gates from consumed records; callbacks
    # alone or a runtime request cannot substitute for its numerical joins.
    gates = ('physical_safety_passed', 'actual_nonzero_B22_residual')
    c.require(all(type(report.get(key)) is bool for key in gates), 'missing C26 numerical gates')
    checks = [report[key] for key in gates]
    for name in ('real_events', 'render', 'last100_stop', 'deferred_archive', 'owner_join'):
        c.require(type(report.get(name, {}).get('passed')) is bool, 'missing C26 gate: '+name)
        checks.append(report[name]['passed'])
    c.require(qualified == all(checks), 'C26 qualification differs from mandatory gates')
    return {'controls': count, 'normal_native': 5*count, 'compiler_native': 2}, qualified


def execute(output):
    output = Path(output).resolve()
    c.require(ROOT == REGISTERED_ROOT and output.parent == ROOT/'results'
              and output.name.startswith('b22_interaction33_')
              and not output.exists(), 'use a new direct results/b22_interaction33_* output; no retry')
    started = time.monotonic()
    source = source_identity()
    c.require(common.memory() >= 6 << 30, 'NOT_READY: effective RAM below 6 GiB')
    c.require(shutil.disk_usage(ROOT).free >= 2 << 30, 'NOT_READY: disk below 2 GiB')
    output.mkdir(exist_ok=False)
    report = {'schema': 'b22-interaction33-supervision-v1', 'status': 'INCOMPLETE',
              'qualification_passed': None, 'source': source, 'failure': None, 'actual_counts': None,
              'minimum_effective_bytes': None, 'retry_permitted': False,
              'physical_human_keyboard_tested': False, 'manual_visual_review_performed': False}
    host, previous = None, {}
    deadline = started+TOTAL_S-10
    def stopped(signum, frame):
        raise InterruptedError('interaction33 interrupted: '+signal.Signals(signum).name)
    def sample(admission=False):
        available, free = common.memory(), shutil.disk_usage(output).free
        row = {'elapsed_s': time.monotonic()-started, 'effective_available_bytes': available, 'free_disk_bytes': free}
        with (output/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        prior = report['minimum_effective_bytes']
        report['minimum_effective_bytes'] = available if prior is None else min(prior, available)
        c.require(available >= (6 << 30 if admission else 3 << 30), 'effective RAM reserve exhausted')
        c.require(free >= 1 << 30 and time.monotonic() < deadline, 'disk/total wall reserve exhausted')
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, stopped)
    old_alarm = signal.signal(signal.SIGALRM, stopped)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline-time.monotonic()))
    try:
        sample(True)
        config, ready = c.registered_config()
        frozen = inputs(config, source)
        validate_sources(frozen)
        release = {'schema': 'b22-interaction33-release-v1', 'plan': plan(output), 'source': source,
                   'inputs': frozen, 'v27_readiness': ready, 'historical_event_source_identity': GO_ID,
                   'isolation_dependencies': EXTRAS}
        release_path = output/'release.json'
        c.write_json(release_path, release)
        host_path = Path(config['host'])
        host = c.load_pinned(host_path, frozen[str(host_path)], 'interaction33_owned_host')
        host.subreaper()
        sample(True)
        argv = make_request(output, config, release, release_path)
        code = common.supervise(argv, output/'host_stdout.log', host=host, sample=sample,
                                limit=RUN_S, deadline=deadline)
        c.require(code == 0, 'C26 host/worker failed; preserve original evidence without retry')
        reader_code = common.supervise(readback_argv(output), output/'reader_stdout.log', host=host,
                                       sample=sample, limit=READER_S, deadline=deadline)
        saved = c.read_json(output/'readback.json')
        run = output/'keyboard_gui'
        actual, qualified = validate_readback(saved, run=run, source_inputs=frozen, returncode=reader_code,
                                             session_id=c.identity(run/'session.json'),
                                             worker_id=c.identity(run/'worker_receipt.json'))
        isolation = c.read_json(output/'keyboard_gui.x11/launcher_receipt.json')
        validate_isolation(isolation, c.read_json(run/'session.json'), output, config)
        validate_sources(frozen)
        c.require(source_identity() == source, 'new interaction33 source changed')
        sample(False)
        report.update(status='COMPLETE', qualification_passed=qualified, actual_counts=actual,
                      release_identity=c.identity(release_path), saved_readback_identity=c.identity(output/'readback.json'),
                      isolated_X11_receipt_identity=c.identity(output/'keyboard_gui.x11/launcher_receipt.json'),
                      screenshots=[str(run/name) for name in ('frame_initial.png', 'frame_drive.png', 'frame_final.png')])
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
        report['elapsed_s'] = time.monotonic()-started
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
        parser.error('--execute requires a new explicit --output')
    result = execute(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['status'] == 'COMPLETE' and result['qualification_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
