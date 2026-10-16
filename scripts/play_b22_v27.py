"""B22 v27: inspect the registered environment or launch one bounded user session.

The v26 controller, worker and host stay unchanged. New launches require the
explicit v27 migration identities and complete, mutually bound execution receipts.
--check never imports or starts a model, GUI, policy, or simulator.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Pin the sibling helper by path; ROS also installs an unrelated `scripts` package.
_spec = importlib.util.spec_from_file_location('b22_runtime_v27', ROOT/'scripts/b22_runtime_v27.py')
runtime = importlib.util.module_from_spec(_spec)
_previous_bytecode = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    _spec.loader.exec_module(runtime)
finally:
    sys.dont_write_bytecode = _previous_bytecode

PROFILES = ('flat_0p6', 'flat_1p6', 'yaw_1p2', 'bumps_0p4', 'rough_0p35', 'ramp_0p45_complete')


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def completion(output, spec, returncode):
    """A zero host exit alone does not prove that the worker ever ran."""
    errors, counts = [], {'completed_controls': None, 'native_returned': None}
    try:
        if returncode != 0:
            errors.append('host_returncode_nonzero')
        session_id = runtime.identity(output/'session.json')
        ready = json.loads((output/'preflight_ready.json').read_text())
        host = json.loads((output/'host_receipt.json').read_text())
        outer = json.loads((output/'supervisor_receipt.json').read_text())
        worker = json.loads((output/'worker_receipt.json').read_text())
        started = ready.get('monotonic_s')
        if (isinstance(started, bool) or not isinstance(started, (int, float))
                or not math.isfinite(started) or started <= 0
                or ready.get('session_sha256') != session_id['sha256']
                or ready.get('full_source_sha256_checked') is not True
                or host.get('execution_started_monotonic') != started):
            errors.append('worker_readiness_not_bound_to_session')
        if (host.get('failure') is not None or type(host.get('exit_code')) is not int
                or host['exit_code'] != 0 or host.get('source_mismatches') != []
                or host.get('owned_no_orphans') is not True
                or host.get('reservation_closed') is not True
                or host.get('reserved_controls') != spec['control_limit']):
            errors.append('host_receipt_failed_or_incomplete')
        if (outer.get('failure') is not None or type(outer.get('exit_code')) is not int
                or outer['exit_code'] != 0 or outer.get('cleanup', {}).get('remaining') != {}):
            errors.append('supervisor_receipt_failed_or_incomplete')
        if (worker.get('session_identity') != session_id or worker.get('failure') is not None
                or worker.get('archive_failed') is not False
                or worker.get('execution_contract_id') != spec['execution_contract_id']):
            errors.append('worker_receipt_failed_or_not_bound_to_session')
        result = worker.get('result') or {}
        completed, python = result.get('completed_controls'), result.get('python') or {}
        counts = {'completed_controls': completed, 'native_returned': python.get('native_returned')}
        if (type(completed) is not int or not 1 <= completed <= spec['control_limit']
                or (spec['mode'] == 'script' and completed != spec['control_limit'])):
            errors.append('worker_did_not_execute_required_controls')
        elif (type(python.get('native_returned')) is not int
              or python['native_returned'] != 5*completed
              or type(python.get('native_attempted')) is not int
              or python['native_attempted'] != 5*completed
              or type(python.get('control_completed')) is not int
              or python['control_completed'] != completed
              or type(python.get('native_failed')) is not int or python['native_failed'] != 0
              or type(python.get('forbidden_entries')) is not int or python['forbidden_entries'] != 0):
            errors.append('worker_physics_accounting_incomplete_or_inconsistent')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        errors.append(f'receipt_read_failed: {type(exc).__name__}: {exc}')
    return {'status': 'PASS' if not errors else 'FAIL', 'errors': errors, 'observed_counts': counts,
            'host_returncode': returncode, 'qualification_granted': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--mode', choices=('script', 'keyboard'), default='script')
    parser.add_argument('--profile', choices=PROFILES)
    parser.add_argument('--actor', choices=('B', 'zero'), default='B')
    parser.add_argument('--mirror', action='store_true')
    parser.add_argument('--seconds', type=int, default=60)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    profile = args.profile or ('flat_1p6' if args.mode == 'script' else 'yaw_1p2')
    if args.mode == 'keyboard' and (profile != 'yaw_1p2' or args.headless or args.mirror):
        parser.error('keyboard uses rendered yaw_1p2; no headless or mirror mode')
    if args.mirror and profile != 'yaw_1p2':
        parser.error('--mirror only applies to yaw_1p2')
    if not 12 <= args.seconds <= 60:
        parser.error('--seconds must be in 12..60')
    try:
        config, check = runtime.inspect()
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(json.dumps({'status': 'BLOCKED', 'error': str(exc), 'model_loads': 0,
                          'gui_starts': 0, 'physics_steps': 0}), file=sys.stderr)
        return 1
    if args.check or check['status'] != 'READY':
        print(json.dumps(check, indent=2, allow_nan=False))
        return 0 if check['status'] == 'READY' else 1
    horizon = ((1800 if profile == 'ramp_0p45_complete' else 1600)
               if args.mode == 'script' else 100*args.seconds)
    output = (args.output or ROOT/'results'/(
        'b22_v27_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()))).resolve()
    launch = output.with_name(output.name+'_launch')
    if output.exists() or launch.exists():
        parser.error('output or its launch directory already exists; choose a new session')
    launch.mkdir(parents=True, exist_ok=False)
    limits = {key: 0 for key in config['model_limit_keys']}
    if args.actor == 'B':
        limits.update(load=1, torch_load=3, predict=horizon+1)
    spec = {'schema': 'd1-c23-gui-worker-v1', 'mode': args.mode, 'render': not args.headless,
            'profile': profile, 'actor': args.actor, 'control_limit': horizon, 'seconds': horizon*.01,
            'seed': config['mirror_seed'] if args.mirror else config['profile_seeds'][profile],
            'mirror': args.mirror, 'output_directory': str(output), 'model_limits': limits,
            'soft_s': 180, 'close_s': 235, 'hard_s': 240, 'qualification': False,
            'frame_limit': 1000, 'require_event_driver': False,
            'execution_contract_id': 'C26_B22_user_session_v1'}
    sources = dict(config['source_hashes'])
    sources[str(ROOT/runtime.MANIFEST)] = runtime.identity(ROOT/runtime.MANIFEST)
    arm = 'user_'+args.mode
    go = {'decision': 'GO', 'scope': 'v27 environment migration; bounded user session, no new qualification',
          'repository': str(ROOT), 'worker': config['worker'], 'inputs': sources,
          'runtime_environment': config['runtime_environment'], 'shared_session': config['shared_session'],
          'arms': {arm: spec}, 'x11_events_by_arm': {}}
    save(launch/'source_go.json', go)
    save(launch/'request.json', {'source_go_path': str(launch/'source_go.json'), 'arm': arm, 'spec': spec})
    save(launch/'v27_readiness.json', check)
    print('B22 + C18 | v27 environment | '+profile+' | output='+str(output), flush=True)
    try:
        code = subprocess.run(['rtk', 'proxy', '/usr/bin/python3', '-B', config['host'],
                               '--request', str(launch/'request.json')], check=False).returncode
        report = completion(output, spec, code)
    except (OSError, KeyboardInterrupt) as exc:
        report = {'status': 'FAIL', 'errors': [f'{type(exc).__name__}: {exc}'],
                  'observed_counts': None, 'host_returncode': None, 'qualification_granted': False}
    save(launch/'v27_completion.json', report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
