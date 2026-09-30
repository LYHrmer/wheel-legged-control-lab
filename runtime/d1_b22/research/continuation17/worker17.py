"""One reserved stage17 fixed-controller evaluation-only worker, with explicit physics and model call owners."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback


def identity(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return {'bytes': Path(path).stat().st_size, 'sha256': digest.hexdigest()}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def mapped_libraries(frozen):
    resolved = {str(Path(name).resolve()) for name in frozen}
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        name = fields[5]
        if '/torch/' not in name and '/nvidia/' not in name:
            continue
        if name.endswith(' (deleted)'):
            raise RuntimeError('deleted compute library mapping')
        actual = str(Path(name).resolve(strict=True))
        if actual not in resolved:
            raise RuntimeError('unfrozen compute library mapping: ' + actual)
        paths.add(actual)
    return {'paths': sorted(paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', required=True, type=Path)
    sessionpath = parser.parse_args().session.resolve(strict=True)
    session = json.loads(sessionpath.read_text())
    output = Path(session['output_directory'])
    write(output/'worker_pid.json', {'pid':os.getpid(),'ppid':os.getppid(),'session_identity':identity(sessionpath)})
    started = time.monotonic()
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGALRM, lambda *_: stop.__setitem__(0, True))
    signal.setitimer(signal.ITIMER_REAL, max(.01, session['host_started_monotonic'] + session['soft_s'] - time.monotonic()))
    failure = None
    cleanup = []
    runtime = guard = env = counter = writer = None
    warnings = []
    result = {}
    old_warning = None
    state = ledger = native = None
    caller = False
    try:
        if session['argv'] != sys.argv or session['retry_permitted'] is not False:
            raise RuntimeError('reserved argv/one-shot identity differs')
        for name, expected in session['source_hashes'].items():
            if identity(name) != expected:
                raise RuntimeError('frozen source differs: ' + name)
        for key, expected in session['runtime_environment'].items():
            if os.environ.get(key) != expected:
                raise RuntimeError('isolated environment differs: ' + key)
        from scripts import run_d1_latest_rl as common
        common._load_import_repair(session, output)
        if 'LD_LIBRARY_PATH' in os.environ or os.environ.get('LD_PRELOAD') != session['library']:
            raise RuntimeError('import repair changed isolated engine environment')
        import engine_binding
        import mujoco
        from archive13.atomic_archive_13 import ArchiveWriter, AtomicCourseNativeGuard
        from full_drive_command_08 import FullDriveCommand, QualifiedCommandCaps
        from scripts.d1_rolling_engine_runtime import RollingEngineRuntime
        from world_upright_course_11 import WorldUprightCourseEnv
        from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry
        from run_rl16_training_08 import _control_step_caller_verified, _boundary, _audited_learning_origins
        from run_world_upright_reference_11 import _binding
        from runtime_support15 import ModelCounter
        if str(Path(engine_binding.__file__).resolve()) != session['binding_module']:
            raise RuntimeError('engine binding origin differs')
        if _nominal_geometry.cache_info().misses != 0 or _nominal_geometry.cache_info().currsize != 0:
            raise RuntimeError('worker geometry cache was not cold')
        class JsonWriter(ArchiveWriter):
            def commit_json(self, path, value):
                return super().commit_json(path, common._jsonable(value))
        writer = JsonWriter()
        def should_stop():
            if stop[0]:
                writer.request_stop()
            return writer.stop_requested
        class BoundedRuntime(RollingEngineRuntime):
            def control_step(self, bound_env, action):
                if should_stop():
                    raise TimeoutError('soft stop before next control')
                return super().control_step(bound_env, action)
        def warn(message):
            warnings.append(str(message))
            raise RuntimeError('MuJoCo warning: ' + str(message))
        old_warning = mujoco.get_mju_user_warning()
        mujoco.set_mju_user_warning(warn)
        if should_stop():
            raise TimeoutError('soft stop during preflight')
        if session['controller_variant'] not in ('baseline', 'yaw', 'ramp', 'combined'):
            raise RuntimeError('unknown stage17 frozen controller variant')
        training = False
        counter = ModelCounter(False)
        counter.limits.update(load=1, torch_load=3, predict=session['policy_predict_control_limit']+1)
        with counter:
            writer.commit_json(output/'mapped_libraries_before_load.json', mapped_libraries(session['source_hashes']))
            with BoundedRuntime(Path(session['library']), control_limit=session['control_limit'], construction_limit=2) as runtime:
                with AtomicCourseNativeGuard(runtime, writer) as guard:
                    caps = QualifiedCommandCaps(1.6, 0., 0., .3, False)
                    env = runtime.construct(lambda: WorldUprightCourseEnv(
                        caps=caps, command_source=lambda *_: FullDriveCommand(),
                        spawn_position_m=(-8., -4.7, .455), max_steps=1000,
                        mode='train' if training else 'eval'))
                    from controller17 import install_controller17
                    install_controller17(env, session['controller_variant'])
                    construction = runtime.state()
                    if construction['construction_attempts'] != 2 or construction['construction_returns'] != 2 or _nominal_geometry.cache_info().misses != 1:
                        raise RuntimeError('course+nominal cold construction count differs')
                    writer.commit_json(output / 'construction_receipt.json', {
                        'C_state': construction, 'nominal_cache': _nominal_geometry.cache_info()._asdict(),
                        'compiled_geometry': env.plant.collision_terrain_metadata,
                        'geometry_binding': _binding(env.plant), 'proof': runtime.proof,
                        'boundary': _boundary(runtime)})
                    env.set_native_interval_reader(guard.interval_summary, begin_interval=guard.begin_interval)
                    from eval17 import evaluate
                    result = evaluate(session, runtime, guard, env, output, writer, counter)
                    counts = counter.counts
                    if any(row['attempted'] != row['returned'] or row['rows_attempted'] != row['rows_returned'] for row in counts.values()):
                        raise RuntimeError('model attempted/returned ledger differs')
                    actual_predictions = sum(row['policy_predictions'] for row in result['cases'])
                    # Each completed floor can legally end early; sum actual API rows.
                    floor_predictions = sum(row['phases'].get('floor:'+arm, {}).get('returned', 0)
                        for key,row in counts.items() if key == 'predict' for arm in ('grouped',))
                    exact = {'load': 1, 'torch_load': 3, 'save': 0, 'learn': 0,
                             'train': 0, 'forward': 0, 'evaluate_actions': 0,
                             'predict_values': 0, 'predict': 1+actual_predictions+floor_predictions,
                             'backward': 0}
                    if any(counts.get(key, {}).get('returned', 0) != expected for key,expected in exact.items()):
                        raise RuntimeError('exact model API count differs from stage')
                    state = runtime.state()
                    actual = runtime.control_completed
                    caller = _control_step_caller_verified(state, runtime.proof, session['source_hashes'], engine_binding)
                    if (warnings or runtime.control_attempted != actual or runtime.returned != 5*actual
                            or runtime.attempted != runtime.returned or runtime.failed != 0
                            or runtime.advanced_substeps != runtime.returned
                            or guard.returned != runtime.returned or guard.checked != runtime.returned
                            or guard.attempted != guard.returned or guard.failure is not None
                            or state['control_returns'] != 5*actual
                            or state['control_attempts'] != state['control_returns']
                            or state['native_construction_caller_verified'] is not True
                            or not caller or state['violations'] != 0
                            or runtime.ledger_receipt()['forbidden_entries'] != 0
                            or state['ccd_attempts'] != state['ccd_returns']
                            or (state['ccd_attempts'] > 0 and state['native_ccd_caller_verified'] is not True)):
                        raise RuntimeError('C/Python/native physical ledger mismatch')
                    writer.commit_json(output / 'loaded_origins_final.json', _audited_learning_origins(session))
                    writer.commit_json(output/'mapped_libraries.json', mapped_libraries(session['source_hashes']))
        state, ledger, native = runtime.state(), runtime.ledger_receipt(), guard.report()
        if state['phase'] != 0 or state['target_model'] != 0 or state['target_data'] != 0:
            raise RuntimeError('runtime did not close owned physical pointers')
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(),
                   'training_cleanup_errors': getattr(error, 'stage15_cleanup_errors', [])}
        print(failure['traceback'], flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if old_warning is not None or 'mujoco' in sys.modules:
            try:
                sys.modules['mujoco'].set_mju_user_warning(old_warning)
            except BaseException as error:
                cleanup.append(repr(error))
        if runtime is not None:
            try:
                runtime.unbind()
                state, ledger = runtime.state(), runtime.ledger_receipt()
                native = None if guard is None else guard.report()
            except BaseException as error:
                cleanup.append(repr(error))
        try:
            write(output/'mapped_libraries_finally.json', mapped_libraries(session['source_hashes']))
        except BaseException as error:
            cleanup.append(repr(error))
        receipt = {'schema': 'd1-control-repair-worker-17-v1', 'arm': session['arm'],
            'session_identity': identity(sessionpath), 'failure': failure, 'cleanup_errors': cleanup,
            'warnings': warnings, 'C_final': state, 'python': ledger, 'native_guard': native,
            'model_calls': None if counter is None else counter.summary(), 'result': result,
            'control_step_caller_verified': caller, 'elapsed_wall_s': time.monotonic()-started,
            'stop_latched': stop[0], 'archive_failed': None if writer is None else writer.failed,
            'retry_permitted': False, 'execution_complete': failure is None and not cleanup}
        write(output / 'worker_receipt.json', common._jsonable(receipt) if 'common' in locals() else receipt)
        print(json.dumps({'arm': session['arm'], 'complete': receipt['execution_complete'],
            'controls': None if runtime is None else runtime.control_completed,
            'seconds': receipt['elapsed_wall_s']}), flush=True)
    return 0 if receipt['execution_complete'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
