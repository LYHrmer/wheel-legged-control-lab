"""Execute worker31.bootstrap31 once, armed with zero integration permission."""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time
import traceback

from worker31 import bootstrap31, runtime_origins31, identity, write


def check(candidate, output, evidence):
    go = json.loads(candidate.read_text())
    if go['decision'] != 'PENDING' or go['bootstrap_only_candidate'] is not True:
        raise RuntimeError('C31 requires its unexecuted cold candidate')
    if any(os.environ.get(k) != v for k, v in go['runtime_environment'].items()):
        raise RuntimeError('C31 cold environment differs')
    if go['runtime_environment']['PYTHONPATH'].split(':')[0] != '/home/lyh/.local/lib/python3.10/site-packages':
        raise RuntimeError('C31 installed MuJoCo must remain first')
    session = {**go['shared_session'], **go['arms']['train'],
               'source_hashes': go['inputs']}
    import engine_binding
    guard = engine_binding.EngineGuard(session['library'])
    guard.arm(0, 0)

    def native():
        state = engine_binding._CState()
        if guard._lib.epa01_get_state(ctypes.byref(state), ctypes.sizeof(state)):
            raise RuntimeError('C31 native counter read failed')
        return {k: int(getattr(state, k)) for k, _ in state._fields_}

    attempts = []
    forbidden = {'from_xml_path', 'from_xml_string', 'from_binary_path', 'mj_step',
                 'mj_step1', 'mj_step2', 'mj_forward', 'mj_resetData',
                 'load', 'predict', 'learn', 'train', 'save'}

    def profile(frame, event, function):
        if event == 'call':
            name, module = frame.f_code.co_name, frame.f_globals.get('__name__', '')
        elif event == 'c_call':
            name, module = getattr(function, '__name__', ''), getattr(function, '__module__', '') or ''
        else:
            return
        if name in forbidden and module.startswith(('mujoco', 'torch', 'stable_baselines3')):
            attempts.append(dict(module=module, entry=name))
            raise RuntimeError('C31 forbidden bootstrap function entry: '+module+'.'+name)

    evidence['native_before'] = native()
    started = time.monotonic()
    sys.setprofile(profile)
    try:
        actual = bootstrap31(session, output)
        if actual['origins31'] != runtime_origins31(session['source_hashes']):
            raise RuntimeError('C31 same early/final origin function differs')
    finally:
        sys.setprofile(None)
        evidence['native_after'] = native()
        evidence['observable_forbidden_function_entries'] = attempts
    keys = ('construction_attempts', 'construction_returns', 'control_attempts',
            'control_returns', 'ccd_attempts', 'ccd_returns', 'violations')
    if any(evidence[phase][key] for phase in ('native_before', 'native_after') for key in keys):
        raise RuntimeError('C31 bootstrap used native calls')
    return dict(schema='d1-c31-actual-bootstrap-v1', passed=True,
        candidate_identity=identity(candidate), worker_identity=identity(go['worker']),
        actual_runtime_origins=actual['origins31'], nominal_cache=actual['cache']._asdict(),
        elapsed_s=time.monotonic()-started,
        evidence_kind=dict(native='actual preloaded CState armed 0/0',
            constructors='source-reviewed import-only path and cold nominal cache',
            profile='observable function entries; not a pybind constructor fence'), **evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--go', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = {}
    try:
        result = check(args.go.resolve(strict=True), args.output.resolve(strict=True), evidence)
    except BaseException as error:
        write(args.output/'receipt.json', dict(passed=False, **evidence,
            error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())))
        raise
    write(args.output/'receipt.json', result)


if __name__ == '__main__':
    main()
