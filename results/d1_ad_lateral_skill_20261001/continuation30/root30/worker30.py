"""C30 bounded headless skill owner; no physics or learning imports at module load."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

CONTRACT = 'C30_lateral_skill_feasibility_v1'


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def runtime_origins30(sources):
    names = ('runtime30', 'record30', 'macro30', 'side_skill30', 'hybrid_adapter27',
             'side_runtime27', 'd1_fast_side_step', 'd1_side_step', 'world_upright_course_11',
             'archive13.atomic_archive_13', 'engine_binding', 'runtime_support22',
             'rl16_learning_11', 'controller18', 'full_drive_command_08', 'kinematics24',
             'state21', 'run_rl16_training_08', 'scripts.run_d1_latest_rl')
    result = {}
    for name in names:
        module = sys.modules.get(name)
        if module is None or not getattr(module, '__file__', None):
            raise RuntimeError('C30 used module lacks actual origin: '+name)
        path = str(Path(module.__file__).resolve(strict=True))
        actual = identity(path)
        if sources.get(path) != actual:
            raise RuntimeError('C30 actual module left frozen sources: '+name)
        result[name] = dict(path=path, **actual)
    path = str(Path(__file__).resolve(strict=True))
    if sources.get(path) != identity(path):
        raise RuntimeError('C30 worker is outside frozen sources')
    result['worker30'] = dict(path=path, **identity(path))
    return result


def bootstrap30(session, output):
    """The actual import/origin path used by both cold check and physical worker."""
    from scripts import run_d1_latest_rl as common
    common._load_import_repair(session, output)
    if os.environ.get('LD_PRELOAD') != session['library'] or 'LD_LIBRARY_PATH' in os.environ:
        raise RuntimeError('C30 import repair changed the engine isolation')
    import mujoco
    import engine_binding
    import d1_fast_side_step  # noqa: F401 - actual lazy dependency checked before construction
    import d1_side_step  # noqa: F401 - actual lazy dependency checked before construction
    from archive13.atomic_archive_13 import ArchiveWriter, AtomicCourseNativeGuard
    from full_drive_command_08 import FullDriveCommand, QualifiedCommandCaps
    from side_runtime27 import side_owner_runtime_type
    from runtime30 import SideAccess30, fence_copy_data30, NativeWindow30
    from side_skill30 import make_side30
    from hybrid_adapter27 import install_hybrid27
    from world_upright_course_11 import WorldUprightCourseEnv
    from kinematics24 import capture_binding24
    from run_world_upright_reference_11 import _binding
    from run_rl16_training_08 import _boundary, _control_step_caller_verified, _audited_learning_origins
    from runtime_support22 import ModelCounter
    from rl16_learning_11 import load_and_verify_final
    from worker22 import mapped_libraries
    from record30 import record_case30
    from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry
    cache = _nominal_geometry.cache_info()
    if cache.misses or cache.currsize:
        raise RuntimeError('C30 worker geometry cache was not cold')
    if str(Path(engine_binding.__file__).resolve()) != session['binding_module']:
        raise RuntimeError('C30 binding module identity differs')
    origins = runtime_origins30(session['source_hashes'])
    write(output/'runtime_module_origins_before.json', origins)
    return locals()


def preflight30(session, session_path, output):
    if (session['execution_contract_id'] != CONTRACT or session['render'] is not False
            or session['control_limit'] != 2200 or session['side_arm'] != 'teacher'
            or session['argv'] != sys.argv or session['retry_permitted'] is not False
            or Path(session['output_directory']).resolve() != output
            or session['spawn_position_m'] != [-8., -4.7, .455]
            or (session['soft_s'], session['close_s'], session['hard_s']) != (240, 270, 300)):
        raise RuntimeError('C30 fixed headless contract differs')
    for path, expected in session['source_hashes'].items():
        if identity(path) != expected:
            raise RuntimeError('C30 frozen input differs: '+path)
    for name, value in session['runtime_environment'].items():
        if os.environ.get(name) != value:
            raise RuntimeError('C30 environment differs: '+name)
    write(output/'preflight_ready.json', dict(pid=os.getpid(),
        session_sha256=identity(session_path)['sha256'], monotonic_s=time.monotonic()))


def execute30(session, session_path, output):
    stop = threading.Event()
    failure = None
    runtime = guard = env = counter = writer = access = None
    original_copy = None
    copy_edges = None
    previous_warning = None
    warnings, cleanup = [], []
    result = {}
    started = time.monotonic()

    def stop_request(*_):
        stop.set()
        if writer is not None:
            writer.request_stop()

    previous_term = signal.signal(signal.SIGTERM, stop_request)
    previous_alarm = signal.signal(signal.SIGALRM, stop_request)
    symbols = None
    try:
        preflight30(session, session_path, output)
        signal.setitimer(signal.ITIMER_REAL, session['soft_s'])
        symbols = bootstrap30(session, output)
        common, mj = symbols['common'], symbols['mujoco']

        class JsonWriter(symbols['ArchiveWriter']):
            def commit_json(self, path, value):
                return super().commit_json(path, common._jsonable(value))

        writer = JsonWriter()

        def warning(message):
            warnings.append(str(message))
            stop.set()
            raise RuntimeError('MuJoCo warning: '+str(message))

        previous_warning = mj.get_mju_user_warning()
        mj.set_mju_user_warning(warning)
        runtime_type = symbols['side_owner_runtime_type'](stop)
        counter = symbols['ModelCounter'](False, session['model_limits'])
        with counter:
            writer.commit_json(output/'mapped_libraries_before_load.json',
                               symbols['mapped_libraries'](session['source_hashes']))
            with runtime_type(Path(session['library']), control_limit=2200, construction_limit=2) as runtime:
                with symbols['AtomicCourseNativeGuard'](runtime, writer) as guard:
                    env = runtime.construct(lambda: symbols['WorldUprightCourseEnv'](
                        caps=symbols['QualifiedCommandCaps'](1.6, 0., 0., .3, False),
                        command_source=lambda *_: symbols['FullDriveCommand'](),
                        spawn_position_m=session['spawn_position_m'], max_steps=2200, mode='eval'))
                    binding = symbols['capture_binding24'](env.plant)
                    clock = {'control_index': 0}
                    hybrid = symbols['install_hybrid27'](env, side_arm='teacher',
                        side_factory=lambda plant: symbols['make_side30'](plant, session['arm'],
                            binding, lambda: clock['control_index']))
                    access = symbols['SideAccess30'](mj, runtime, env.plant, hybrid.side.scratch)
                    access.side = hybrid.side
                    hybrid.attach_side_access(access)
                    runtime.register_side_access(access)
                    access.install()
                    original_copy, copy_edges = symbols['fence_copy_data30'](
                        mj, plant=env.plant, side_access=access, stop_event=stop)
                    window = symbols['NativeWindow30'](binding)
                    guard.native_sink = window.sink
                    state = runtime.state()
                    if state['construction_returns'] != 2 or symbols['_nominal_geometry'].cache_info().misses != 1:
                        raise RuntimeError('C30 cold construction count differs')
                    writer.commit_json(output/'construction_receipt.json', dict(C_state=state,
                        nominal_cache=symbols['_nominal_geometry'].cache_info()._asdict(),
                        compiled_geometry=env.plant.collision_terrain_metadata,
                        geometry_binding=symbols['_binding'](env.plant), side_kinematic_binding=binding,
                        skill30_joint_limits=dict(jnt_range=env.plant.model.jnt_range.tolist(),
                                                  jnt_limited=env.plant.model.jnt_limited.tolist()),
                        side_profile=asdict(hybrid.side.cfg), proof=runtime.proof,
                        boundary=symbols['_boundary'](runtime), data_addresses=dict(
                            live=int(env.plant.data._address), measurement=int(env.plant.measurement_data._address),
                            side_scratch=int(hybrid.side.scratch._address))))
                    env.set_native_interval_reader(guard.interval_summary, begin_interval=guard.begin_interval)
                    counter.phase = 'probe'
                    before_load = symbols['_boundary'](runtime)
                    policy, load_report = symbols['load_and_verify_final'](
                        session['checkpoint_folder'], session['checkpoint_manifest'], return_model=True)
                    if symbols['_boundary'](runtime) != before_load or load_report['probe_actions_byte_exact'] is not True:
                        raise RuntimeError('C30 strict B22 probe changed physics or policy')
                    writer.commit_json(output/'strict_load.json', load_report)
                    counter.phase = 'control'
                    result = symbols['record_case30'](session, runtime, guard, env, hybrid, policy,
                        writer, window, clock, stop, output)
                    caller = symbols['_control_step_caller_verified'](runtime.state(), runtime.proof,
                                session['source_hashes'], symbols['engine_binding'])
                    if (not caller or runtime.control_completed != result['completed_controls']
                            or runtime.control_attempted != runtime.control_completed
                            or runtime.returned != 5*runtime.control_completed
                            or runtime.attempted != runtime.returned or runtime.thread_violations
                            or guard.checked != runtime.returned or not access.report()['within_actual_bounds']
                            or access.rejected or copy_edges['rejected']):
                        raise RuntimeError('C30 physical/static/copy ledger does not close')
                    actual_model = counter.summary()['counts']
                    expected_model = dict(load=1, torch_load=3,
                        predict=result['policy_predictions']+1, forward=0,
                        evaluate_actions=0, predict_values=0, backward=0, learn=0, train=0, save=0)
                    if any(actual_model.get(k, {}).get('returned', 0) != v
                           for k, v in expected_model.items()):
                        raise RuntimeError('C30 exact model call ledger differs')
                    result['control_step_caller_verified'] = caller
                    origins = runtime_origins30(session['source_hashes'])
                    if origins != symbols['origins']:
                        raise RuntimeError('C30 actual early/final runtime origins differ')
                    writer.commit_json(output/'runtime_module_origins.json', origins)
                    writer.commit_json(output/'loaded_origins_final.json', symbols['_audited_learning_origins'](session))
                    writer.commit_json(output/'mapped_libraries_final.json', symbols['mapped_libraries'](session['source_hashes']))
    except BaseException as error:
        result = getattr(error, 'c30_partial_result', result)
        failure = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        print(failure['traceback'], flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for name, action in (
                ('restore_copy', lambda: setattr(symbols['mujoco'], 'mj_copyData', original_copy)
                 if symbols is not None and original_copy is not None else None),
                ('restore_static_queries', lambda: access.restore() if access is not None else None),
                ('restore_warning', lambda: symbols['mujoco'].set_mju_user_warning(previous_warning)
                 if symbols is not None else None)):
            try:
                action()
            except BaseException as error:
                cleanup.append(name+': '+repr(error))
        receipt = dict(schema='d1-c30-headless-worker-v1', execution_contract_id=CONTRACT,
            arm=session['arm'], session_identity=identity(session_path), failure=failure,
            cleanup_errors=cleanup, warnings=warnings, result=result,
            C_final=None if runtime is None else runtime.state(),
            python=None if runtime is None else runtime.ledger_receipt(),
            native_guard=None if guard is None else guard.report(),
            model_calls=None if counter is None else counter.summary(),
            side_access=None if access is None else access.report(), copy_edges=copy_edges,
            archive_failed=None if writer is None else writer.failed,
            elapsed_wall_s=time.monotonic()-started, retry_permitted=False,
            execution_complete=failure is None and not cleanup,
            qualification=False, training_controls=0, optimizer_steps=0)
        write(output/'worker_receipt.json', receipt if symbols is None else symbols['common']._jsonable(receipt))
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGALRM, previous_alarm)
    return 0 if receipt['execution_complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = args.session.resolve(strict=True)
    return execute30(json.loads(path.read_text()), path, args.output.resolve(strict=True))


if __name__ == '__main__':
    raise SystemExit(main())
