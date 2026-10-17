"""Plan or explicitly collect one new baseline/gated B22 inference pair."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'runtime/d1_b22/runtime_manifest_v27.json'
MANIFEST_SHA = 'd3c38429362e82745179f2c526b2ee52d805526df390d52346c805e5a51fff09'
BASE_WORKER = 'scripts/run_b22_worker31.py'
BASE_WORKER_SHA = '49feaecf2c6e68cf39c99238d5bde04b88dafb10de72479d0f3ce5c999e5d7c0'
MODEL_SHA = '7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691'
AUTHORED = ('scripts/b22_inference32.py', 'scripts/run_b22_inference32.py',
            'scripts/read_b22_inference32.py', 'scripts/run_b22_worker32.py',
            'tests/test_b22_inference32.py', 'tests/test_b22_inference32_pair.py',
            'docs/b22_inference32.md', BASE_WORKER)
LIMITS = {'controls': 3200, 'normal_native': 16000, 'compiler_native': 4,
          'worker_sessions': 2, 'per_session_controls': 1600, 'per_session_predict': 1601,
          'stage_wall_s': 300, 'reader_wall_s': 180, 'total_wall_s': 900,
          'start_effective_bytes': 6 << 30, 'reserve_effective_bytes': 4 << 30,
          'start_free_disk_bytes': 1 << 30, 'reserve_free_disk_bytes': 512 << 20}


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    digest, size = hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            digest.update(data); size += len(data)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_module(name, path, expected):
    require(identity(path) == expected, 'pinned helper changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def derive_worker32(source):
    """Exact declared edits; every other whole-module AST node stays C31."""
    require(hashlib.sha256(source.encode()).hexdigest() == BASE_WORKER_SHA, 'C31 base worker changed')
    patches = [
        ('C31 fixed headless B22 baseline; preserve the C26 numerical owner.',
         'C32 independent baseline/gated inference pair; preserve C31 control mathematics.'),
        ("'C31_B22_fixed_suite_v1'", "'C32_B22_inference_pair_v1'"),
        ("or session.get('profile') not in PROFILE_CASE", "or session.get('profile') != 'flat_1p6'\n            or session.get('actor') != 'B' or session.get('seed') != 181202\n            or session.get('inference_mode32') not in ('baseline','gated')"),
        ("or not 1 <= session['control_limit'] <= 6000", "or session['control_limit'] != 1600"),
        ("                        writer.commit_json(output/'strict_load.json',load_report)",
         ("                        writer.commit_json(output/'strict_load.json',load_report)\n"
         "                        import importlib.util\n"
         "                        inference_path32=Path(session['inference_helper32'])\n"
         "                        if identity(inference_path32)!=session['source_hashes'][str(inference_path32)]:\n"
         "                            raise RuntimeError('inference32 helper source differs')\n"
         "                        helper_spec32=importlib.util.spec_from_file_location('bound_inference32',inference_path32)\n"
         "                        helper_module32=importlib.util.module_from_spec(helper_spec32)\n"
         "                        helper_spec32.loader.exec_module(helper_module32)\n"
         "                        inference32=helper_module32.GatedInference32('B',policy,\n"
         "                            numeric_actor_action=_numeric_actor_action,manifest=manifest,strict_load=load_report)")),
        ("                            action,predicted=_numeric_actor_action(\n                                'zero' if policy is None else 'final_policy',policy,observation)\n                            if bool(predicted)!=(policy is not None):",
         ("                            gate32=env.controller.action_gate(env.decision.world_command)\n"
         "                            if type(gate32) is not bool:\n"
         "                                raise RuntimeError('inference32 current gate is not bool')\n"
         "                            if session['inference_mode32']=='gated':\n"
         "                                action,predicted,inference_row32=inference32.select(observation,action_enabled=gate32)\n"
         "                            else:\n"
         "                                action,predicted=_numeric_actor_action('final_policy',policy,observation)\n"
         "                                inference_row32={'schema':helper_module32.SCHEMA,'actor':'B',\n"
         "                                    'action_gate_enabled':gate32,'predict_called':predicted,\n"
         "                                    'policy_action_observed':True,'reason':'BASELINE_PREDICTED',\n"
         "                                    'raw_policy_action16':action.tolist(),\n"
         "                                    'controller_input_action16':action.tolist(),\n"
         "                                    'policy_output_kind':'SB3_clipped_deterministic_action',\n"
         "                                    'checkpoint_sha256':MODEL_SHA}\n"
         "                            if bool(predicted)!=(session['inference_mode32']=='baseline' or gate32):")),
        ("'input_observation99':input_obs,'raw_action16':action,", "'input_observation99':input_obs,\n                                'raw_action16':inference_row32['raw_policy_action16'],\n                                'inference32':inference_row32,"),
        ("'policy_predictions':predictions,", "'policy_predictions':predictions,\n                            'inference_mode32':session['inference_mode32'],"),
    ]
    result = source
    for before, after in patches:
        require(result.count(before) == 1, 'C32 source patch seam missing/duplicated: '+before[:70])
        result = result.replace(before, after)
    ast.parse(result)
    return result


def check_worker32(root=ROOT):
    expected = derive_worker32((root/BASE_WORKER).read_text())
    require(ast.dump(ast.parse(expected), include_attributes=False)
            == ast.dump(ast.parse((root/'scripts/run_b22_worker32.py').read_text()), include_attributes=False),
            'worker32 changed outside its exact inference/preflight/ledger edits')


def plan(output=None):
    return {'schema': 'b22-inference32-plan-v1',
            'owner_instruction': '把这些优化点，在这台电脑能力之内的都完成',
            'output': None if output is None else str(Path(output).resolve()),
            'modes': ['baseline', 'gated'], 'profile': 'flat_1p6', 'actor': 'B', 'seed': 181202,
            'limits': dict(LIMITS), 'training_allowed': False, 'retry_permitted': False,
            'behavior_equivalence': 'NOT_VALIDATED', 'expected_control_prediction_saving': 880,
            'qualification_granted': False, 'reuse_suite31_or_closeout30': False}


def source_identity(root=ROOT):
    check_worker32(root)
    names = list(AUTHORED)
    subprocess.run(['git', 'ls-files', '--error-unmatch', *names], cwd=root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *names], cwd=root, check=True)
    return {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'authored': {str(root/name): identity(root/name) for name in names}}


def validate_release(path, root=ROOT):
    release = read_json(path)
    output = Path(release['plan']['output'])
    require(release.get('schema') == 'b22-inference32-release-v1' and release.get('plan') == plan(output)
            and release.get('repository') == str(root) and output.parent == root/'results'
            and output.name.startswith('b22_inference32_'), 'new32 release scope differs')
    require(identity(root/MANIFEST)['sha256'] == MANIFEST_SHA, 'old v27 manifest changed')
    config = read_json(root/MANIFEST)
    authored = release['source']['authored']
    require(set(authored) == {str(root/name) for name in AUTHORED}, 'new32 source set differs')
    expected = {**config['source_hashes'], **authored, str(root/MANIFEST): identity(root/MANIFEST)}
    require(release['inputs'] == expected, 'new32 source closure differs')
    for name, value in expected.items():
        require(Path(name).is_absolute() and not Path(name).is_symlink() and identity(name) == value,
                'new32 frozen input changed: '+name)
    for name, value in authored.items():
        relative = str(Path(name).relative_to(root))
        data = subprocess.check_output(['git', 'show', release['source']['git_commit']+':'+relative], cwd=root)
        require(hashlib.sha256(data).hexdigest() == value['sha256'], 'new32 source is not committed: '+relative)
    check_worker32(root)
    return release


def registered_config(root=ROOT):
    require(identity(root/MANIFEST)['sha256'] == MANIFEST_SHA, 'old v27 manifest changed')
    config = read_json(root/MANIFEST)
    path = root/'scripts/b22_runtime_v27.py'
    checked, readiness = load_module('inference32_v27_inspect', path, config['source_hashes'][str(path)]).inspect(root)
    require(checked == config and readiness['status'] == 'READY', 'v27 environment is not READY')
    return config, readiness


def read_memory():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(fields['MemAvailable'].split()[0])*1024
    scope = next((line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')), None)
    require(available >= 0 and scope is not None and '..' not in Path(scope).parts, 'unknown host/cgroup RAM')
    top = Path('/sys/fs/cgroup')
    current, remaining = top/scope.lstrip('/'), []
    require(current.is_dir(), 'cgroup-v2 is not visible')
    while True:
        maximum = current/'memory.max'
        if maximum.exists() and maximum.read_text().strip() != 'max':
            limit, used = int(maximum.read_text()), int((current/'memory.current').read_text())
            require(limit >= 0 and used >= 0, 'invalid cgroup RAM sample')
            remaining.append(max(0, limit-used))
        if current == top: break
        current = current.parent
    return min([available, *remaining])


def request(mode, release, output, config, release_path):
    require(mode in ('baseline','gated') and output == Path(release['plan']['output'])/mode, 'fixed pair mode/output required')
    launch = output.with_name(output.name+'_launch')
    launch.mkdir(exist_ok=False)
    limits = dict.fromkeys(config['model_limit_keys'], 0)
    limits.update(load=1, torch_load=3, predict=1601)
    spec = {'schema': 'd1-c23-gui-worker-v1', 'mode': 'script', 'render': False,
            'profile': 'flat_1p6', 'actor': 'B', 'seed': 181202, 'mirror': False,
            'control_limit': 1600, 'seconds': 16., 'output_directory': str(output), 'qualification': False,
            'model_limits': limits, 'soft_s': 180, 'close_s': 235, 'hard_s': 240, 'frame_limit': 1000,
            'require_event_driver': False, 'execution_contract_id': 'C32_B22_inference_pair_v1',
            'inference_mode32': mode, 'inference_helper32': str(ROOT/'scripts/b22_inference32.py')}
    go = {'decision': 'GO', 'scope': 'new32 controlled baseline/gated pair; no training',
          'repository': str(ROOT), 'worker': str(ROOT/'scripts/run_b22_worker32.py'),
          'inputs': {**release['inputs'], str(release_path): identity(release_path)},
          'runtime_environment': config['runtime_environment'], 'shared_session': config['shared_session'],
          'arms': {mode: spec}, 'x11_events_by_arm': {}}
    write_json(launch/'source_go.json', go)
    write_json(launch/'request.json', {'source_go_path': str(launch/'source_go.json'), 'arm': mode, 'spec': spec})
    return ['rtk', 'proxy', '/usr/bin/python3', '-B', config['host'], '--request', str(launch/'request.json')]


@contextmanager
def signal_scope():
    state = {'defer': False, 'received': None}
    def handler(sig, _):
        state['received'] = signal.Signals(sig).name
        for item in (signal.SIGINT, signal.SIGTERM):
            signal.signal(item, signal.SIG_IGN)
        if not state['defer']:
            raise InterruptedError(state['received'])
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, handler)
        yield state
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


def supervise(argv, log, host, sample, limit, deadline, *, factory=subprocess.Popen,
              clock=time.monotonic, sleeper=time.sleep):
    process, started = None, clock()
    with signal_scope() as signals:
        try:
            with Path(log).open('xb') as stream:
                signals['defer'] = True
                try:
                    process = factory(argv, stdin=subprocess.DEVNULL, stdout=stream,
                                      stderr=subprocess.STDOUT, start_new_session=True)
                finally:
                    signals['defer'] = False
                require(not signals['received'], 'interrupted while obtaining process handle')
                while process.poll() is None:
                    sample(False)
                    require(clock()-started < limit and clock() < deadline, 'process wall limit')
                    sleeper(1.)
                sample(False)
                require(clock()-started < limit and clock() < deadline, 'process wall limit after exit')
                return process.returncode
        finally:
            previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                for sig in previous:
                    signal.signal(sig, signal.SIG_IGN)
                require(host.clean_descendants(os.getpid(), process)['remaining'] == {}, 'owned process cleanup incomplete')
            finally:
                for sig, old in previous.items():
                    signal.signal(sig, old)


def known_counts(worker, session, mode, run, release_path, release):
    require(session.get('inference_mode32') == mode and session.get('arm') == mode
            and session.get('execution_contract_id') == 'C32_B22_inference_pair_v1'
            and session.get('output_directory') == str(run) and session.get('checkpoint_sha256') == MODEL_SHA
            and session.get('source_hashes', {}).get(str(release_path)) == identity(release_path)
            and all(session.get('source_hashes', {}).get(k) == v for k,v in release['inputs'].items())
            and worker.get('session_identity') == identity(run/'session.json')
            and worker.get('execution_contract_id') == session['execution_contract_id']
            and worker.get('arm') == mode, 'foreign worker/session cannot supply known counters')
    result = worker.get('result', {})
    count, py, native = result.get('completed_controls'), result.get('python', {}), result.get('C_final', {})
    require(type(count) is int and 0 < count <= 1600 and py.get('control_completed') == count
            and all(type(py.get(k)) is int for k in ('control_attempted','control_completed','native_attempted','native_returned','native_failed'))
            and py.get('control_attempted') == count and py.get('native_failed') == 0
            and py.get('native_attempted') == py.get('native_returned') == count*5
            and result.get('inference_mode32') == mode
            and all(type(native.get(k)) is int for k in ('construction_attempts','construction_returns','control_attempts','control_returns'))
            and native.get('construction_attempts') == native.get('construction_returns') == 2
            and native.get('control_attempts') == native.get('control_returns') == count*5,
            'unknown worker physical counters')
    return {'controls':count,'normal_native':5*count,'compiler_native':2}


def execute(output):
    with signal_scope():
        return _execute(output)


def _execute(output):
    output = Path(output).absolute()
    require(output.parent == ROOT/'results' and output.name.startswith('b22_inference32_')
            and not output.exists() and not output.is_symlink(), 'new direct results/b22_inference32_* output required')
    source = source_identity()
    require(read_memory() >= LIMITS['start_effective_bytes'], 'NOT_READY: effective RAM below 6 GiB')
    require(shutil.disk_usage(ROOT).free >= LIMITS['start_free_disk_bytes'], 'NOT_READY: disk reserve')
    output.mkdir(exist_ok=False)
    started, deadline = time.monotonic(), time.monotonic()+LIMITS['total_wall_s']
    report = {'schema': 'b22-inference32-supervision-v1', 'status': 'INCOMPLETE', 'source': source,
              'stages': [], 'failure': None, 'actual_counts': None, 'known_actual_counts': dict.fromkeys(('controls','normal_native','compiler_native'), 0),
              'unknown_count_stages': [], 'retry_permitted': False, 'behavior_equivalence': 'NOT_VALIDATED'}
    host = None
    def sample(admission=False):
        value, free = read_memory(), shutil.disk_usage(output).free
        with (output/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps({'elapsed_s': time.monotonic()-started, 'effective_available_bytes': value,
                                     'free_disk_bytes': free})+'\n')
        report['minimum_effective_bytes'] = min(report.get('minimum_effective_bytes', value), value)
        require(value >= LIMITS['start_effective_bytes' if admission else 'reserve_effective_bytes'],
                'effective RAM below 6 GiB' if admission else 'effective RAM below 4 GiB')
        require(free >= LIMITS['reserve_free_disk_bytes'] and time.monotonic() < deadline, 'disk/total wall reserve exhausted')
    try:
        config, readiness = registered_config()
        release = {'schema': 'b22-inference32-release-v1', 'repository': str(ROOT), 'plan': plan(output),
                   'source': source, 'inputs': {**config['source_hashes'], **source['authored'], str(ROOT/MANIFEST): identity(ROOT/MANIFEST)},
                   'v27_readiness': readiness}
        release_path = output/'release.json'
        write_json(release_path, release)
        validate_release(release_path)
        path = Path(config['host'])
        host = load_module('inference32_owned_host', path, release['inputs'][str(path)])
        host.subreaper()
        for mode in ('baseline', 'gated'):
            sample(True); validate_release(release_path)
            argv = request(mode, release, output/mode, config, release_path)
            report['unknown_count_stages'].append(mode)
            code = supervise(argv, output/(mode+'_stdout.log'), host, sample, LIMITS['stage_wall_s'], deadline)
            worker = read_json(output/mode/'worker_receipt.json')
            actual = known_counts(worker,read_json(output/mode/'session.json'),mode,output/mode,release_path,release)
            result, count = worker['result'], actual['controls']
            report['unknown_count_stages'].remove(mode)
            for key, value in actual.items():
                report['known_actual_counts'][key] += value
            report['stages'].append({'name': mode, 'worker_receipt': identity(output/mode/'worker_receipt.json')})
            require(code == 0 and worker.get('failure') is None and worker.get('warnings') == []
                    and count == 1600 and result.get('terminated') is False and result.get('truncated') is True,
                    'new32 session did not complete its full fixed horizon')
        destination = output/'comparison.json'
        argv = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(ROOT/'scripts/read_b22_inference32.py'),
                '--release', str(release_path), '--report', str(destination)]
        require(supervise(argv, output/'reader_stdout.log', host, sample, LIMITS['reader_wall_s'], deadline) == 0,
                'new32 independent reader failed')
        compared = read_json(destination)
        require(compared.get('status') == 'PASS' and compared.get('release_identity') == identity(release_path),
                'new32 comparison is not bound to this release')
        validate_release(release_path)
        sample(False)
        report.update(status='PASS', behavior_equivalence='PASS', comparison=identity(destination))
    except BaseException as exc:  # noqa: BLE001 - preserve failure receipt for signals before owned cleanup
        report['failure'] = type(exc).__name__+': '+str(exc)
    finally:
        if host is not None:
            with signal_scope():
                prior = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
                try:
                    report['cleanup'] = host.clean_descendants(os.getpid(), None)
                    if report['cleanup']['remaining']:
                        report.update(status='INCOMPLETE', behavior_equivalence='NOT_VALIDATED', failure='owned descendants remain')
                except BaseException as exc:  # noqa: BLE001 - cleanup failure always revokes completion
                    report.update(status='INCOMPLETE', behavior_equivalence='NOT_VALIDATED', failure='cleanup: '+str(exc))
                finally:
                    for sig, old in prior.items(): signal.signal(sig, old)
        report['actual_counts'] = None if report['unknown_count_stages'] else report['known_actual_counts']
        report['elapsed_s'] = time.monotonic()-started
        write_json(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(plan(args.output), ensure_ascii=False, indent=2)); return 0
    require(args.output is not None, '--execute requires an explicit new --output')
    value = execute(args.output)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return 0 if value['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
