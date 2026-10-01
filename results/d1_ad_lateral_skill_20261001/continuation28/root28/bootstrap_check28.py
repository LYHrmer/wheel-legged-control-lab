"""One C28 cold import check; the root launcher owns the process and deadline."""

import argparse
import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


SITE = '/home/lyh/.local/lib/python3.10/site-packages'
FORBIDDEN = {'MjModel', 'MjData', 'from_xml_path', 'from_xml_string',
             'from_binary_path',
             'mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_resetData',
             'load', 'predict', 'learn', 'train', 'save'}
EXPECTED_IMPORTS = (
    'mujoco', 'numpy', 'archive13.atomic_archive_13',
    'async_course_renderer_12', 'full_drive_command_08', 'gui13_bridge',
    'side_runtime27', 'hybrid_adapter27', 'kinematics24',
    'run_rl16_training_08', 'worker22', 'engine_binding',
    'run_world_upright_reference_11', 'state21', 'runtime_support22',
    'wheel_legged_control.d1.wheel_leg_controller', 'world_upright_course_11',
    'rl16_learning_11',
)
EVIDENCE = {}


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def prefix_imports(worker):
    tree = ast.parse(Path(worker).read_text(), filename=str(worker))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == 'run')
    first_local = next(i for i, node in enumerate(run.body)
                       if isinstance(node, ast.FunctionDef) and node.name == 'capture_state27')
    nodes = [node for node in run.body[:first_local]
             if isinstance(node, (ast.Import, ast.ImportFrom))]
    if not nodes or ast.unparse(nodes[0]) != 'from scripts import run_d1_latest_rl as common':
        raise RuntimeError('frozen worker import-repair prefix changed')
    modules = tuple(node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                    for node in nodes[1:])
    if modules != EXPECTED_IMPORTS:
        raise RuntimeError('frozen worker import sequence changed')
    return nodes[1:]


def check(go_path, output):
    go = json.loads(go_path.read_text())
    if go.get('decision') != 'PENDING' or not go.get('inputs') or not go.get('worker'):
        raise RuntimeError('bootstrap needs the reviewed pending C28 candidate')
    environment = go['runtime_environment']
    if (environment['PYTHONPATH'].split(':')[0] != SITE
            or os.environ.get('PYTHONPATH') != environment['PYTHONPATH']
            or os.environ.get('LD_PRELOAD') != environment['LD_PRELOAD']
            or os.environ.get('LD_BIND_NOW') != '1'
            or 'LD_LIBRARY_PATH' in os.environ
            or any(os.environ.get(key) != '1' for key in
                   ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'))):
        raise RuntimeError('cold bootstrap environment differs from repaired candidate')
    worker = Path(go['worker']).resolve(strict=True)
    if identity(worker) != go['inputs'].get(str(worker)):
        raise RuntimeError('frozen worker identity differs')
    nodes = prefix_imports(worker)
    session = {**go['shared_session'], 'source_hashes': go['inputs']}
    import engine_binding
    if Path(engine_binding.__file__).resolve() != Path(session['binding_module']):
        raise RuntimeError('engine binding origin differs')
    guard = engine_binding.EngineGuard(session['library'])
    guard.arm(0, 0)

    def native():
        state = engine_binding._CState()
        if guard._lib.epa01_get_state(ctypes.byref(state), ctypes.sizeof(state)) != 0:
            raise RuntimeError('preloaded native counters unavailable')
        return {name: int(getattr(state, name)) for name, _ in state._fields_}

    before = native()
    EVIDENCE['native_before'] = before
    calls = []

    def tripwire(frame, event, arg):
        if event == 'call':
            name = frame.f_code.co_name
            if name not in FORBIDDEN:
                return
            module = frame.f_globals.get('__name__', '')
        elif event == 'c_call':
            name = getattr(arg, '__name__', '')
            if name not in FORBIDDEN:
                return
            module = getattr(arg, '__module__', '') or ''
        else:
            return
        if (module.startswith(('mujoco', 'stable_baselines3', 'torch'))
                or name in ('from_xml_path', 'from_xml_string', 'from_binary_path')):
            calls.append({'module': module, 'entry': name})
            raise RuntimeError('bootstrap forbidden model/physics API: '+module+'.'+name)

    started = time.monotonic()
    sys.setprofile(tripwire)
    try:
        from scripts import run_d1_latest_rl as common
        common._load_import_repair(session, output)
        if os.environ.get('LD_PRELOAD') != session['library'] or 'LD_LIBRARY_PATH' in os.environ:
            raise RuntimeError('frozen import repair did not restore engine isolation')
        namespace = {}
        for node in nodes:
            module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
            exec(compile(module, str(worker), 'exec'), namespace)
    finally:
        sys.setprofile(None)
        try:
            EVIDENCE['native_after'] = native()
        except BaseException as error:
            EVIDENCE['native_after_error'] = repr(error)
        EVIDENCE['tripwire_forbidden_calls'] = list(calls)
    after = native()
    counters = ('construction_attempts', 'construction_returns',
                'control_attempts', 'control_returns', 'ccd_attempts',
                'ccd_returns', 'violations')
    if (any(before[key] or after[key] for key in counters)
            or before['construction_limit'] != 0 or before['control_limit'] != 0
            or after['construction_limit'] != 0 or after['control_limit'] != 0
            or calls):
        raise RuntimeError('cold bootstrap used model/physics API')
    cache = namespace['_nominal_geometry'].cache_info()._asdict()
    if cache != {'hits': 0, 'misses': 0, 'maxsize': 1, 'currsize': 0}:
        raise RuntimeError('nominal geometry cache was not cold after worker imports')
    origins = {}
    origin_identities = {}
    for name in ('scripts.run_d1_latest_rl', *EXPECTED_IMPORTS):
        module = sys.modules.get(name)
        file = None if module is None else getattr(module, '__file__', None)
        if file is None:
            raise RuntimeError('worker dependency was not actually imported: '+name)
        path = str(Path(file).resolve(strict=True))
        actual_identity = identity(path)
        if go['inputs'].get(path) != actual_identity:
            raise RuntimeError('actual imported module escaped source closure: '+name)
        origins[name] = path
        origin_identities[name] = actual_identity
    return {'schema': 'd1-c28-cold-bootstrap-v1', 'passed': True,
            'candidate_path': str(go_path), 'candidate_identity': identity(go_path),
            'worker_path': str(worker), 'worker_identity': identity(worker),
            'environment_pythonpath': environment['PYTHONPATH'],
            'actual_origins': origins, 'actual_origin_identities': origin_identities,
            'worker_prefix_imports': list(EXPECTED_IMPORTS),
            'native_before': before, 'native_after': after,
            'tripwire_forbidden_calls': calls,
            'evidence_kind': {
                'worker_dependency_imports': 'executed_AST_prefix_with_actual_origin_hashes',
                'function_entry_tripwire': 'Python_call_and_c_call_only;_pybind_class_construction_not_proven_by_profile',
                'native_counts': 'actual_preloaded_CState_armed_0_0_before_and_after',
                'model_data_nonconstruction': 'source_reviewed_import_only_path_plus_cold_nominal_cache',
            },
            'nominal_cache': cache,
            'controls': 0, 'physics_steps': 0,
            'elapsed_s': time.monotonic()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--go', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(strict=True)
    go_path = args.go.resolve(strict=True)
    if not output.is_dir():
        raise RuntimeError('root must reserve the bootstrap output directory')
    try:
        receipt = check(go_path, output)
        save(output/'receipt.json', receipt)
    except BaseException as error:
        save(output/'receipt.json', {
            'schema': 'd1-c28-cold-bootstrap-v1', 'passed': False,
            'candidate_path': str(go_path), 'candidate_identity': identity(go_path),
            **EVIDENCE,
            'error': {'type': type(error).__name__, 'message': str(error),
                      'traceback': traceback.format_exc()}})
        raise


if __name__ == '__main__':
    main()
