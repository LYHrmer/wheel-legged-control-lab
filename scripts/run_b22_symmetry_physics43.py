"""One bounded zero/raw42/mirror43 physical comparison and saved-only readback."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
DRIVER = Path(__file__).resolve()
OUTPUT = ROOT/'results/b22_symmetry43_physics_20261018_01'
PLAN = ROOT/'results/b22_symmetry43_physics_plan_20261018_01.json'
UNIT = 'wheel-symmetry43-physics01-20261018-01.service'
SCOPE = f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{UNIT}'
MANIFEST = ROOT/'runtime/d1_b22/runtime_manifest_v27.json'
NUMERIC = ROOT/'results/b22_symmetry43_numeric_20261018_03'
MODEL_SHA = '44cbf9a4363ff42af78d8e54049e0dbade2bea74701e9a27073bcb9e656f3239'
WEIGHTS_SHA = '7320ec626cee2f6bf98ed4322840d4e4cacbc90ebd42726085b1385e5fc03c91'
AUTHORED = ('scripts/run_b22_symmetry_physics43.py', 'scripts/b22_symmetry_physics43.py',
            'src/wheel_legged_control/d1/policy_symmetry43.py', 'tests/test_symmetry_physics43.py',
            'docs/b22_symmetry43_physics.md', 'scripts/worker_b22_tracking_reward42.py',
            'scripts/read_b22_tracking_reward42.py', 'scripts/b22_runtime_packages39.py',
            'scripts/b22_numpy_backend38.py', 'scripts/resource_guard42.py',
            'scripts/export_b22_numpy36.py')


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular file required: '+str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def helpers():
    return load('symmetry43_physics_helpers', ROOT/'scripts/b22_symmetry_physics43.py')


def configure_paths():
    config = read(MANIFEST)
    sys.path[:0] = config['runtime_environment']['PYTHONPATH'].split(os.pathsep)
    # This repository's scripts is a namespace; ROS also has a regular scripts
    # package. Bind the intended source root explicitly before frozen imports.
    package = ModuleType('scripts')
    package.__path__ = [str(ROOT/'scripts')]
    sys.modules['scripts'] = package


def scope():
    return {'schema': 'b22-symmetry43-physical-plan-v1', 'unit': UNIT, 'output': str(OUTPUT),
            'actors': ['zero', 'raw42', 'mirror43'], 'floor_per_policy': 600,
            'original_six_per_actor': 9800, 'controls_max': 30600, 'normal_native_max': 153000,
            'compiler_native_max': 2, 'numpy_actor_calls_max': 31200,
            'training': 0, 'ppo_load': 0, 'torch_load': 0, 'residual_gain': 1,
            'start_effective_bytes': 4 << 30, 'reserve_effective_bytes': 2 << 30,
            'admission_s': 10, 'admission_max_s': 300, 'total_execution_s': 2100,
            'worker_wall_s': 1200, 'reader_wall_s': 900, 'output_quota_bytes': 8 << 30,
            'retry_or_resume': False, 'automatic_model_replacement': False}


def source(proof=None):
    if proof is None:
        proof = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 'files': {name: identity(ROOT/name) for name in AUTHORED}}
    require(set(proof['files']) == set(AUTHORED), '43 source closure differs')
    for name, entry in proof['files'].items():
        require(identity(ROOT/name) == entry, '43 source changed')
        blob = subprocess.check_output(['git', 'show', proof['commit']+':'+name], cwd=ROOT, timeout=10)
        require(hashlib.sha256(blob).hexdigest() == entry['sha256'], 'commit43 source before prepare')
    return proof


def inputs():
    h = helpers()
    worker, reader = h.api42(), h.reader42()
    # These frameworks are forbidden by an import gate in both execution paths;
    # unused training binaries do not belong to this NumPy execution closure.
    excluded = ('/torch/', '/nvidia/', '/stable_baselines3/', '/cv2/')
    files = {name: entry for name, entry in read(MANIFEST)['source_hashes'].items()
             if not any(part in name for part in excluded)}
    for mapping in (worker.required_inputs42(), reader.required_inputs42(),
                    {str(ROOT/name): identity(ROOT/name) for name in AUTHORED}):
        for name, entry in mapping.items():
            require(name not in files or files[name] == entry, '43 conflicting source identity')
            files[name] = entry
    report = read(NUMERIC/'report.json')
    require(report['status'] == 'COMPLETE' and report['error'] is None and
            report['source']['commit'] == 'c2f1f65b3def761d6e60aede828a8f455941e8c1' and
            report['base_checkpoint_sha256'] == MODEL_SHA and report['max_absolute_error'] <= 1e-6 and
            report['symmetric_max_absolute_error'] <= 1e-6 and identity(NUMERIC/'actor.npz')['sha256'] == WEIGHTS_SHA,
            '43 completed real42 export required')
    require(all(identity(NUMERIC/name)['sha256'] == digest for name, digest in report['artifacts'].items()),
            '43 numerical export artifacts differ from completed report')
    for p in (MANIFEST, NUMERIC/'report.json', NUMERIC/'actor.npz', NUMERIC/'numerical.npz'):
        files[str(p)] = identity(p)
    for name, entry in files.items():
        require(identity(name) == entry, '43 input changed: '+name)
    return files


def verified_plan():
    plan = read(PLAN)
    require(plan['scope'] == scope(), '43 physical scope changed')
    source(plan['source'])
    for name, entry in plan['inputs'].items():
        require(identity(name) == entry, '43 bound input changed: '+name)
    return plan


def worker():
    plan = verified_plan()
    session = read(OUTPUT/'session.json')
    h = helpers()
    require(session['parent_pid'] == os.getppid() and session['plan_identity'] == identity(PLAN)
            and session['execution_contract_id'] == h.CONTRACT, '43 live parent/session required')
    configure_paths()
    backend = load('symmetry43_forbidden_frameworks', ROOT/'scripts/b22_numpy_backend38.py')
    backend.deny_frameworks38()
    # isort: off
    # Extension order retained from the successful39/41 NumPy bootstrap.
    import mujoco
    import mujoco._functions
    import mujoco._rollout
    import engine_binding
    import numpy as np
    from archive13.atomic_archive_13 import ArchiveWriter, AtomicCourseNativeGuard
    from controller18 import install_controller18
    from full_drive_command_08 import FullDriveCommand, QualifiedCommandCaps
    from run_rl16_training_08 import _boundary, _control_step_caller_verified
    from run_world_upright_reference_11 import _binding
    from world_upright_course_11 import WorldUprightCourseEnv

    from scripts import run_d1_latest_rl as common
    from scripts.d1_rolling_engine_runtime import RollingEngineRuntime
    from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry
    # isort: on
    package = h.api42().package42()
    package.verify_loaded_roots39(package.inventory39())
    require(_nominal_geometry.cache_info().misses == 0, '43 cold geometry required')
    library = read(MANIFEST)['runtime_environment']['LD_PRELOAD']
    require(os.environ.get('LD_PRELOAD') == library and os.environ.get('LD_BIND_NOW') == '1'
            and 'LD_LIBRARY_PATH' not in os.environ, '43 native library environment differs')
    old_warning = mujoco.get_mju_user_warning()
    def warning(message):
        raise RuntimeError('MuJoCo warning: '+str(message))
    mujoco.set_mju_user_warning(warning)
    class Writer(ArchiveWriter):
        def commit_json(self, path, value):
            return super().commit_json(path, common._jsonable(value))
    writer = Writer()
    def stop(*_):
        writer.request_stop()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, scope()['worker_wall_s']-20)
    class Runtime(RollingEngineRuntime):
        def control_step(self, env, action):
            require(not writer.stop_requested, '43 stop requested')
            return super().control_step(env, action)
    result = {'status': 'INCOMPLETE', 'error': None, 'cases': [], 'cleanup_errors': [],
              'plan_identity': identity(PLAN), 'counts': None, 'known_counts': None,
              'training': 0, 'ppo_load': 0, 'torch_load': 0, 'numpy_bundle_loads': 0}
    runtime = guard = env = servo = None
    actors = {}
    try:
        with np.load(NUMERIC/'actor.npz', allow_pickle=False) as saved:
            weights = {k: saved[k] for k in saved.files}
        result['numpy_bundle_loads'] = 1
        math36 = load('symmetry43_numpy_shape', ROOT/'scripts/export_b22_numpy36.py')
        math36.validate_weights(weights, np)
        actors = {name: h.Actor(weights, symmetric=name == 'mirror43', limit=10400 if name == 'raw42' else 20800)
                  for name in ('raw42', 'mirror43')}
        recipe, recorder, proofs = h.recorder()
        writer.commit_json(OUTPUT/'derivation_worker.json', proofs)
        api = h.api42()
        api.install_servo_counter42({'pure_servo_advance_max': 30600})
        servo = api._servo_counter42
        with (Runtime(Path(library), control_limit=30600, construction_limit=2) as runtime,
              AtomicCourseNativeGuard(runtime, writer) as guard):
            env = runtime.construct(lambda: WorldUprightCourseEnv(
                caps=QualifiedCommandCaps(1.6, 0., 0., .3, False),
                command_source=lambda *_: FullDriveCommand(),
                spawn_position_m=(-8., -4.7, .455), max_steps=1000, mode='eval'))
            install_controller18(env, 'combined')
            state = runtime.state()
            require(state['construction_attempts'] == state['construction_returns'] == 2
                    and _nominal_geometry.cache_info().misses == 1, '43 cold construction ledger differs')
            writer.commit_json(OUTPUT/'construction_receipt.json', {
                'C_state': state, 'nominal_cache': _nominal_geometry.cache_info()._asdict(),
                'compiled_geometry': env.plant.collision_terrain_metadata,
                'geometry_binding': _binding(env.plant), 'proof': runtime.proof,
                'boundary': _boundary(runtime)})
            env.set_native_interval_reader(guard.interval_summary, begin_interval=guard.begin_interval)
            (OUTPUT/'floor').mkdir()
            for actor in ('raw42', 'mirror43'):
                receipt, _ = recorder.record_case(runtime, guard, env, actors[actor],
                    recipe.floor_schedule(201090), actor, OUTPUT/'floor'/actor,
                    writer=writer, pair_initial=None, checkpoint_sha256=MODEL_SHA)
                require(receipt['record_valid'], '43 floor record invalid')
                result['cases'].append({'folder': 'floor/'+actor, 'controls': receipt['completed_controls']})
            (OUTPUT/'heldout').mkdir()
            for case in recipe.ORDER:
                paired = None
                for actor in h.ACTORS:
                    receipt, initial = recorder.record_case(runtime, guard, env, actors.get(actor),
                        recipe.make_schedule(case), actor, OUTPUT/'heldout'/(case+'_'+actor),
                        writer=writer, pair_initial=paired, checkpoint_sha256=MODEL_SHA if actor != 'zero' else None)
                    require(receipt['record_valid'], '43 case record invalid')
                    if paired is None:
                        paired = initial
                    result['cases'].append({'folder': 'heldout/'+case+'_'+actor,
                                            'controls': receipt['completed_controls']})
                    write(OUTPUT/'progress.json', {'cases': result['cases'], 'controls': runtime.control_completed})
            state = runtime.state()
            caller = _control_step_caller_verified(state, runtime.proof, plan['inputs'], engine_binding)
            require(runtime.control_attempted == runtime.control_completed and
                    runtime.attempted == runtime.returned == runtime.advanced_substeps == 5*runtime.control_completed
                    and runtime.failed == 0 and guard.attempted == guard.returned == guard.checked == runtime.returned
                    and guard.failure is None and state['control_attempts'] == state['control_returns'] == runtime.returned
                    and state['native_construction_caller_verified'] is True and caller
                    and state['violations'] == 0 and runtime.ledger_receipt()['forbidden_entries'] == 0
                    and state['ccd_attempts'] == state['ccd_returns'] and
                    (state['ccd_attempts'] == 0 or state['native_ccd_caller_verified'] is True), '43 native count/caller mismatch')
            result.update(status='COMPLETE', native_caller_verified=True,
                          counts={'controls': runtime.control_completed, 'normal_native': runtime.returned,
                                  'compiler_native': state['construction_returns']})
    except BaseException as exc:  # noqa: BLE001 -- preserve original failure and bounded cleanup
        result['error'] = f'{type(exc).__name__}: {exc}'
        (OUTPUT/'worker_error.txt').write_text(traceback.format_exc())
    finally:
        for target in (env, servo):
            if target is not None:
                try:
                    target.close()
                except BaseException as exc:  # noqa: BLE001 -- preserve original failure and bounded cleanup
                    result['cleanup_errors'].append(repr(exc))
        mujoco.set_mju_user_warning(old_warning)
        signal.setitimer(signal.ITIMER_REAL, 0)
        result['actor_counts'] = {k: v.counts() for k, v in actors.items()}
        if runtime is not None:
            result['C_final'] = runtime.state()
            result['python'] = runtime.ledger_receipt()
            result['known_counts'] = {'controls': runtime.control_completed, 'normal_native': runtime.returned,
                                      'compiler_native': runtime.state()['construction_returns']}
        result['pure_servo'] = None if servo is None else servo.report()
        result['forbidden_frameworks_absent'] = backend.frameworks_absent38()
        if writer.failed or writer.stop_requested or result['cleanup_errors']:
            result['status'], result['counts'] = 'INCOMPLETE', None
        write(OUTPUT/'worker_report.json', result)
    return int(result['status'] != 'COMPLETE')


def saved_reader():
    verified_plan()
    configure_paths()
    h = helpers()
    maths = h.saved_math()
    reader = maths.reader
    worker_result = read(OUTPUT/'worker_report.json')
    require(worker_result['status'] == 'COMPLETE' and worker_result['error'] is None and
            not worker_result['cleanup_errors'] and worker_result['plan_identity'] == identity(PLAN), '43 worker incomplete')
    construction = reader.document(OUTPUT/'construction_receipt.json')
    binding = reader.check_binding(construction['geometry_binding'])
    geometry = construction['compiled_geometry']
    require(len(geometry['world_collision_geoms']) == 92 and geometry['control_dt_s'] == .01
            and geometry['native_dt_s'] == .002, '43 compiled geometry dimensions differ')
    scores, summaries, floors, parsed = [], [], {}, []
    pure_helper = h.reader42()
    with pure_helper.geometry_counter42(sys.modules['geometry21'], 30600) as pure_count:
        floor_spec = ('floor_0p4_600', 'flat', .4, 201090, 600, 325, 425)
        for actor in ('raw42', 'mirror43'):
            summary, canonical = reader.heldout_case(OUTPUT, floor_spec, actor, binding, geometry,
                expected_variant='combined', folder_override=OUTPUT/'floor'/actor)
            require(canonical['checkpoint_sha256'] == MODEL_SHA, '43 floor checkpoint differs')
            floors[actor] = maths.score.score_floor_canonical(canonical, expected_seed=201090)
            summaries.append(summary)
        for case in maths.recipes.ORDER:
            task = maths.recipes.TASK_BY_ID[case]
            spec = (case, task.terrain, task.speed_mps, task.seed, task.horizon, task.hold_start, task.release_tick)
            parsed.append((spec, False, h.ACTORS))
            for actor in h.ACTORS:
                summary, canonical = reader.heldout_case(OUTPUT, spec, actor, binding, geometry, expected_variant='combined')
                require(canonical['checkpoint_sha256'] == (MODEL_SHA if actor != 'zero' else None), '43 case checkpoint differs')
                score = maths.score.score_case(canonical, expected_seed=task.seed, mirror=False)
                scores.append(dict(score, experiment_actor=actor, checkpoint_sha256=canonical['checkpoint_sha256']))
                summaries.append(summary)
                write(OUTPUT/'readback_progress.json', {'completed_cases': len(scores), 'last': [case, actor]})
    pairs = reader._new_pairs(OUTPUT, parsed, ['raw42', 'mirror43'])
    total = sum(row['completed_controls'] for row in summaries)
    require(total == worker_result['counts']['controls'] and worker_result['counts']['normal_native'] == 5*total
            and worker_result['counts']['compiler_native'] == 2, '43 saved physical counts differ')
    for actor, multiplier in (('raw42', 1), ('mirror43', 2)):
        predictions = sum(row['predict_calls'] for row in summaries if row['actor'] == actor)
        require(worker_result['actor_counts'][actor] == {'predictions': predictions, 'attempts': multiplier*predictions,
                                                       'returns': multiplier*predictions}, '43 NumPy inference counts differ')
    result = {'status': 'COMPLETE', 'execution_contract_id': h.CONTRACT, 'plan_identity': identity(PLAN),
              'worker_report_identity': identity(OUTPUT/'worker_report.json'), 'cases': scores, 'floors': floors,
              'pairs': pairs, 'summaries': summaries, 'derivation': maths.proofs,
              'contribution': {actor: maths.score.score_pairs(scores, policy_actor=actor)
                               for actor in ('raw42', 'mirror43')},
              'counts': worker_result['counts'], 'reader_inference': 0, 'reader_physics': 0,
              'pure_servo': pure_count, 'automatic_model_replacement': False}
    write(OUTPUT/'readback.json', result)
    return 0


def execute():
    plan = verified_plan()
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, '43 Nice10 required')
    OUTPUT.mkdir(exist_ok=False)
    guard = load('symmetry43_resources', ROOT/'scripts/resource_guard42.py')
    started = time.monotonic()
    admitted = None
    result = {'status': 'INCOMPLETE', 'error': None, 'plan_identity': identity(PLAN),
              'counts': None, 'source': plan['source'], 'cleanup_complete': True}
    child = None
    def poll():
        now = time.monotonic()
        require(now-(started if admitted is None else admitted) < (300 if admitted is None else 2100), '43 wall budget')
        memory = guard.snapshot_memory(SCOPE)
        guard.check_memory(memory)
        with (OUTPUT/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps({**memory, 'elapsed_s': now-started})+'\n')
        require(memory['effective_available_bytes'] >= 2 << 30, '43 runtime reserve')
        require(sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file()) <= 8 << 30, '43 archive quota')
        return memory['effective_available_bytes']
    def interrupt(signum, _frame):
        raise RuntimeError('43 received '+signal.Signals(signum).name)
    old = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        stable = None
        while True:
            available, now = poll(), time.monotonic()
            stable = (now if stable is None else stable) if available >= 4 << 30 else None
            if stable is not None and now-stable >= 10:
                admitted = now
                break
            time.sleep(1)
        write(OUTPUT/'session.json', {'execution_contract_id': helpers().CONTRACT,
            'parent_pid': os.getpid(), 'plan_identity': identity(PLAN), 'source_hashes': plan['inputs']})
        for mode, wall in (('worker', 1200), ('read', 900)):
            env = os.environ.copy()
            for key, value in read(MANIFEST)['runtime_environment'].items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value
            if mode == 'read':
                env.pop('LD_PRELOAD', None)
            phase = time.monotonic()
            with (OUTPUT/(mode+'.log')).open('x') as log:
                child = subprocess.Popen(['/usr/bin/python3', str(DRIVER), '--'+mode], env=env, cwd=ROOT,
                                         stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    poll()
                    require(time.monotonic()-phase < wall, '43 child wall budget')
                    time.sleep(1)
                require(child.returncode == 0, '43 '+mode+' failed; saved logs retained')
                child = None
            if mode == 'worker':
                result['known_counts'] = read(OUTPUT/'worker_report.json')['known_counts']
            result[mode+'_elapsed_s'] = time.monotonic()-phase
        verified_plan()
        saved = read(OUTPUT/'readback.json')
        require(saved['status'] == 'COMPLETE', '43 readback incomplete')
        result.update(status='COMPLETE', counts=saved['counts'], readback_identity=identity(OUTPUT/'readback.json'))
        poll()
    except BaseException as exc:  # noqa: BLE001 -- preserve original failure and bounded cleanup
        result.update(status='INCOMPLETE', counts=None, error=f'{type(exc).__name__}: {exc}')
        if (OUTPUT/'worker_report.json').exists():
            result['known_counts'] = read(OUTPUT/'worker_report.json')['known_counts']
        (OUTPUT/'parent_error.txt').write_text(traceback.format_exc())
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        result['cleanup_complete'] = child is None or child.poll() is not None
        result['elapsed_s'] = time.monotonic()-started
        write(OUTPUT/'report.json', result)
        for signum, handler in old.items():
            signal.signal(signum, handler)
    print(json.dumps({k: result[k] for k in ('status', 'error', 'counts', 'elapsed_s')}))
    return int(result['status'] != 'COMPLETE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    for option in ('prepare', 'execute', 'worker', 'read'):
        group.add_argument('--'+option, action='store_true')
    args = parser.parse_args()
    if args.worker:
        return worker()
    if args.read:
        return saved_reader()
    if args.execute:
        return execute()
    if args.prepare:
        require(not PLAN.exists() and not OUTPUT.exists(), '43 unique new execution only')
        write(PLAN, {'scope': scope(), 'source': source(), 'inputs': inputs()})
    print(json.dumps({'prepared': args.prepare, **scope()}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
