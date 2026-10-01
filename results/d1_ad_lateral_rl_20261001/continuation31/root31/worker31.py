"""Single C31 pilot owner; one plant, cumulative budgets, archived real episodes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

from worker30 import identity, write, bootstrap30, runtime_origins30

CONTRACT = 'C31_event_lateral_RL_pilot_v1'


def runtime_origins31(sources):
    result = runtime_origins30(sources)
    for name in ('runtime31', 'ledger31', 'record31', 'macro31', 'task31',
                 'observation31', 'side_skill31', 'learning31', 'checkpoint31'):
        module = sys.modules.get(name)
        if module is None or not getattr(module, '__file__', None):
            raise RuntimeError('C31 missing actual module origin: '+name)
        path = str(Path(module.__file__).resolve(strict=True))
        actual = identity(path)
        if sources.get(path) != actual:
            raise RuntimeError('C31 module is outside frozen source: '+name)
        result[name] = dict(path=path, **actual)
    path = str(Path(__file__).resolve(strict=True))
    if sources.get(path) != identity(path):
        raise RuntimeError('C31 worker is outside frozen source')
    result['worker31'] = dict(path=path, **identity(path))
    return result


def bootstrap31(session, output):
    symbols = bootstrap30(session, output)
    from runtime31 import SideAccess31, NativeWindow31, EpisodeBudget31, install_reset_yaw31
    from ledger31 import ModelLedger31, LateralLedger31, parameters31
    from side_skill31 import make_side31
    from record31 import record_case31
    from learning31 import LinearEventPolicy31, LatchSampler31, prepare_batch31, update_batch31
    from checkpoint31 import save_and_reload_final31, load_final31
    imports = locals().copy()
    imports.pop('symbols')
    symbols.update(imports)
    symbols['origins31'] = runtime_origins31(session['source_hashes'])
    write(output/'runtime31_module_origins_before.json', symbols['origins31'])
    return symbols


def preflight31(session, session_path, output):
    train = session['arm'] == 'train'
    expected = (140800, 64, 256, 4200, 4500, 4560, 4800) if train else (
        22000, 10, 40, 900, 960, 1020, 1500)
    observed = tuple(session[k] for k in ('control_limit', 'cycles_limit', 'macros_limit',
        'soft_s', 'close_s', 'hard_s', 'outer_s'))
    if (session['execution_contract_id'] != CONTRACT or observed != expected
            or session['arm'] not in ('train', 'evaluation')
            or session['render'] is not False or session['retry_permitted'] is not False
            or session['argv'] != sys.argv
            or Path(session['output_directory']).resolve() != output
            or session['spawn_position_m'] != [-8., -4.7, .455]
            or session['seed'] != 271001 or session['policy_seed'] != 310031):
        raise RuntimeError('C31 preregistered finite session differs')
    for path, expected_identity in session['source_hashes'].items():
        if identity(path) != expected_identity:
            raise RuntimeError('C31 source changed: '+path)
    for key, expected_value in session['runtime_environment'].items():
        if os.environ.get(key) != expected_value:
            raise RuntimeError('C31 environment differs: '+key)
    write(output/'preflight_ready.json', dict(pid=os.getpid(),
        session_sha256=identity(session_path)['sha256'], monotonic_s=time.monotonic()))


def execute_cycles31(s, session, output, runtime, guard, env, hybrid, B22,
                     writer, window, access, clock, yaw_state, stop, counter):
    train = session['arm'] == 'train'
    budget = s['EpisodeBudget31'](cycles=session['cycles_limit'],
        controls=session['control_limit'], macros=session['macros_limit'])
    clock['budget31'] = budget
    pending, results, batches = [], [], []
    steps = attempts = total_failures = consecutive_failures = 0
    predictions = 0
    actor_changed = False
    with s['LateralLedger31'](training=train, model_counter=counter) as lateral:
        clock['lateral31'] = lateral
        lateral.set_phase('construct' if train else 'lateral_load')
        if train:
            policy = s['LinearEventPolicy31']()
            optimizer = policy.optimizer31()
        else:
            policy = s['load_final31'](session['lateral_checkpoint_folder'],
                                       session['lateral_checkpoint_manifest'])
            optimizer = None
        writer.commit_json(output/'initial_parameters31.json', s['parameters31'](policy))
        sampler = s['LatchSampler31'](policy, mode='train' if train else 'learned',
                                     seed=session['policy_seed'])
        clock['sampler'] = sampler
        for index in range(session['cycles_limit']):
            if stop.is_set() or writer.stop_requested:
                raise RuntimeError('C31 stopped before full preregistered cycle list')
            reservation = budget.begin(controls=runtime.control_completed, native=runtime.returned)
            case = dict(session, cycle_index=index, control_limit=reservation['control_limit'])
            if train:
                case.update(mode='train', direction=1 if index % 2 == 0 else -1,
                            initial_yaw_rad=0., cancel=False, case_id=f'train_{index:02d}')
            else:
                case.update(session['evaluation_cases'][index])
            lateral.set_phase('control')
            result, cycle = s['record_case31'](case, runtime, guard, env, hybrid, B22,
                writer, window, clock, stop, output, cycle_index=index,
                global_control_offset=reservation['global_control_begin'],
                access=access, yaw_state=yaw_state)
            closed = bool(cycle and cycle.get('terminal_kind') in (
                'success', 'controlled_failure', 'safe_cancel'))
            actual = budget.finish(controls=runtime.control_completed, native=runtime.returned,
                                   macros=result['actual_event_transitions'], complete=closed)
            writer.commit_json(output/f'budget_cycle_{index:02d}.json', actual)
            results.append(result)
            clock['last_cycle_result31'] = result
            predictions += result['policy_predictions']
            if not closed or result.get('fatal_failure'):
                raise RuntimeError('C31 incomplete or unsafe cycle; reset rescue forbidden')
            writer.commit_json(output/f'progress_{index:02d}.json', dict(
                closed_cycles=len(results), actual_controls=runtime.control_completed,
                actual_normal_native=runtime.returned, actual_macros=budget.macros,
                actual_optimizer_steps=steps, actual_attempts=attempts,
                current_case=case['case_id'], terminal_kind=cycle['terminal_kind']))
            if not train:
                if cycle['terminal_kind'] == 'controlled_failure':
                    raise RuntimeError('C31 evaluation task failed; no replacement case')
                continue
            failed = cycle['terminal_kind'] == 'controlled_failure'
            total_failures += int(failed)
            consecutive_failures = consecutive_failures+1 if failed else 0
            pending.append(cycle)
            if consecutive_failures >= 2 or total_failures >= 8:
                raise RuntimeError('C31 controlled failure stop reached')
            if len(pending) == 8:
                batch = s['prepare_batch31'](pending)
                lateral.set_phase('lateral_update')
                before = s['parameters31'](policy)
                update = s['update_batch31'](policy, optimizer, batch,
                    batch_index=len(batches), step_count_before=steps,
                    permutation_seed=session['policy_seed']+len(batches))
                steps += update['actual_optimizer_steps']
                attempts += update['actual_evaluated_minibatches']
                after = s['parameters31'](policy)
                actor_gradient = any(a.get('gradients', {}).get('actor', {}).get(
                    'raw_norm', 0.) > 0 for a in update['attempts'])
                actor_changed |= actor_gradient and any(
                    before[k] != after[k] for k in ('actor.weight', 'actor.bias'))
                path = output/f'batch_{len(batches):02d}.json'
                writer.commit_json(path, dict(prepared_batch=batch, update_receipt=update,
                    cycle_receipts={str(c['cycle_index']): identity(
                        output/f"episode_{c['cycle_index']}"/'cycle_receipt.json') for c in pending}))
                batches.append(dict(path=str(path), **identity(path)))
                if (attempts > 64 or steps > 64 or update['hard_stop'] is not None
                        or not update['valid_train_call']):
                    raise RuntimeError('C31 learning gate failed: '+str(update['hard_stop']))
                pending = []
        final_parameters = s['parameters31'](policy)
        writer.commit_json(output/'final_parameters31.json', final_parameters)
        receipt = dict(status='complete', actual_batches=len(batches),
            actual_optimizer_steps=steps, actual_minibatch_attempts=attempts,
            actor_nonzero_gradient_and_parameter_change=bool(actor_changed),
            cycles=len(results), controls=runtime.control_completed, normal_native=runtime.returned,
            compiler_native=runtime.state()['construction_returns'], true_macros=budget.macros,
            controlled_failures=total_failures, actual_sampler_actor_rows=sampler.actor_rows,
            actual_sampler_value_rows=sampler.value_rows, actual_rng_samples=sampler.rng_samples,
            batches=batches, training=train, pending_cycles=len(pending))
        if train:
            if len(batches) != 8 or pending or not actor_changed:
                raise RuntimeError('C31 full pilot or real actor update is missing')
            lateral.set_phase('lateral_final')
            # Frozen probe rows are selected before training and never selected by outcome.
            independent, manifest = s['save_and_reload_final31'](policy,
                output/'final_checkpoint', metadata=dict(execution_contract_id=CONTRACT,
                    source_hashes=session['source_hashes'], policy_seed=session['policy_seed']),
                training_receipt=receipt, probe_observations54=session['final_probe_observations54'])
            if s['parameters31'](independent) != final_parameters:
                raise RuntimeError('C31 independently reloaded complete parameter set differs')
            writer.commit_json(output/'final_checkpoint_receipt31.json', manifest)
        ledger = lateral.report()
        receipt['lateral_model_calls'] = ledger
        receipt['budget'] = dict(limits=budget.limits, closed=budget.closed, macros=budget.macros)
        receipt['policy_predictions'] = predictions
        receipt['qualification'] = False
        receipt['independent_readback_pending'] = True
        writer.commit_json(output/'training_receipt31.json' if train else output/'evaluation_receipt31.json', receipt)
    return receipt


def execute31(session, session_path, output):
    stop = threading.Event()
    failure = None
    runtime = guard = env = counter = writer = access = None
    original_copy = previous_warning = symbols = None
    copy_edges = None
    warnings, cleanup = [], []
    result = {}
    clock = {}
    started = time.monotonic()

    def stop_request(*_):
        stop.set()
        if writer is not None:
            writer.request_stop()

    previous_term = signal.signal(signal.SIGTERM, stop_request)
    previous_alarm = signal.signal(signal.SIGALRM, stop_request)
    try:
        preflight31(session, session_path, output)
        signal.setitimer(signal.ITIMER_REAL, session['soft_s'])
        symbols = bootstrap31(session, output)
        s = symbols
        common, mj = s['common'], s['mujoco']

        class JsonWriter(s['ArchiveWriter']):
            def commit_json(self, path, value):
                return super().commit_json(path, common._jsonable(value))

        writer = JsonWriter()

        def warning(message):
            warnings.append(str(message))
            stop.set()
            raise RuntimeError('MuJoCo warning: '+str(message))

        previous_warning = mj.get_mju_user_warning()
        mj.set_mju_user_warning(warning)
        runtime_type = s['side_owner_runtime_type'](stop)
        counter = s['ModelLedger31'](False, session['model_limits'])
        with counter:
            writer.commit_json(output/'mapped_libraries_before_load.json', s['mapped_libraries'](session['source_hashes']))
            with runtime_type(Path(session['library']), control_limit=session['control_limit'], construction_limit=2) as runtime:
                with s['AtomicCourseNativeGuard'](runtime, writer) as guard:
                    env = runtime.construct(lambda: s['WorldUprightCourseEnv'](
                        caps=s['QualifiedCommandCaps'](1.6, 0., 0., .3, False),
                        command_source=lambda *_: s['FullDriveCommand'](),
                        spawn_position_m=session['spawn_position_m'], max_steps=2200, mode='eval'))
                    binding = s['capture_binding24'](env.plant)
                    clock = dict(control_index=0, cycle_index=0, global_offset=0, sampler=None)

                    def sample(*args):
                        if clock['sampler'] is None:
                            raise RuntimeError('C31 side callback before lateral model readiness')
                        return clock['sampler'](*args)

                    hybrid = s['install_hybrid27'](env, side_arm='teacher',
                        side_factory=lambda plant: s['make_side31'](plant,
                            mode='train' if session['arm'] == 'train' else 'learned',
                            geometry_binding=binding,
                            control_index_provider=lambda: clock['control_index'],
                            cycle_index_provider=lambda: clock['cycle_index'], sample_latch31=sample))
                    access = s['SideAccess31'](mj, runtime, env.plant, hybrid.side.scratch,
                                             cycles_limit=session['cycles_limit'])
                    access.side = hybrid.side
                    hybrid.attach_side_access(access)
                    runtime.register_side_access(access)
                    access.install()
                    original_copy, copy_edges = s['fence_copy_data30'](mj,
                        plant=env.plant, side_access=access, stop_event=stop)
                    window = s['NativeWindow31'](binding)
                    guard.native_sink = window.sink
                    yaw_state = s['install_reset_yaw31'](env)
                    if runtime.state()['construction_returns'] != 2 or s['_nominal_geometry'].cache_info().misses != 1:
                        raise RuntimeError('C31 one cold construction count differs')
                    writer.commit_json(output/'construction_receipt.json', dict(C_state=runtime.state(),
                        nominal_cache=s['_nominal_geometry'].cache_info()._asdict(),
                        compiled_geometry=env.plant.collision_terrain_metadata,
                        geometry_binding=s['_binding'](env.plant), side_kinematic_binding=binding,
                        skill30_joint_limits=dict(jnt_range=env.plant.model.jnt_range.tolist(),
                                                  jnt_limited=env.plant.model.jnt_limited.tolist()),
                        side_profile=asdict(hybrid.side.cfg), proof=runtime.proof,
                        boundary=s['_boundary'](runtime), data_addresses=dict(
                            live=int(env.plant.data._address), measurement=int(env.plant.measurement_data._address),
                            side_scratch=int(hybrid.side.scratch._address))))
                    env.set_native_interval_reader(guard.interval_summary, begin_interval=guard.begin_interval)
                    counter.phase = 'probe'
                    before_load = s['_boundary'](runtime)
                    B22, load_report = s['load_and_verify_final'](
                        session['checkpoint_folder'], session['checkpoint_manifest'], return_model=True)
                    if s['_boundary'](runtime) != before_load or load_report['probe_actions_byte_exact'] is not True:
                        raise RuntimeError('C31 strict B22 probe changed physics or policy')
                    writer.commit_json(output/'strict_load.json', load_report)
                    result = execute_cycles31(s, session, output, runtime, guard, env, hybrid, B22,
                        writer, window, access, clock, yaw_state, stop, counter)
                    caller = s['_control_step_caller_verified'](runtime.state(), runtime.proof,
                        session['source_hashes'], s['engine_binding'])
                    if (not caller or runtime.control_attempted != runtime.control_completed
                            or runtime.returned != 5*runtime.control_completed or runtime.attempted != runtime.returned
                            or runtime.thread_violations or guard.checked != runtime.returned
                            or not access.report()['within_actual_bounds'] or access.rejected or copy_edges['rejected']):
                        raise RuntimeError('C31 cumulative physical/static/copy ledger does not close')
                    counts = counter.summary()['counts']
                    expected = dict(load=1, torch_load=4, predict=result['policy_predictions']+1,
                        forward=0, evaluate_actions=0, predict_values=0,
                        backward=result['actual_optimizer_steps'], learn=0, train=0, save=0)
                    if any(counts.get(k, {}).get('returned', 0) != v for k, v in expected.items()):
                        raise RuntimeError('C31 global model ledger does not match archived actual calls')
                    if runtime_origins31(session['source_hashes']) != s['origins31']:
                        raise RuntimeError('C31 early/final actual origins differ')
                    writer.commit_json(output/'runtime31_module_origins.json', s['origins31'])
                    writer.commit_json(output/'reset_yaw_receipt31.json', yaw_state)
                    writer.commit_json(output/'loaded_origins_final.json', s['_audited_learning_origins'](session))
                    writer.commit_json(output/'mapped_libraries_final.json', s['mapped_libraries'](session['source_hashes']))
    except BaseException as error:
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
        receipt = dict(schema='d1-c31-headless-worker-v1', execution_contract_id=CONTRACT,
            arm=session['arm'], session_identity=identity(session_path), failure=failure,
            cleanup_errors=cleanup, warnings=warnings, result=result,
            C_final=None if runtime is None else runtime.state(),
            python=None if runtime is None else runtime.ledger_receipt(),
            native_guard=None if guard is None else guard.report(),
            model_calls=None if counter is None else counter.summary(),
            lateral_model_calls=None if 'lateral31' not in clock else clock['lateral31'].report(),
            partial_budget=None if 'budget31' not in clock else dict(
                limits=clock['budget31'].limits, closed=clock['budget31'].closed,
                open=clock['budget31'].open, macros=clock['budget31'].macros),
            last_cycle_result=clock.get('last_cycle_result31'),
            side_access=None if access is None else access.report(), copy_edges=copy_edges,
            archive_failed=None if writer is None else writer.failed,
            elapsed_wall_s=time.monotonic()-started, retry_permitted=False,
            execution_complete=failure is None and not cleanup,
            qualification=False)
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
    return execute31(json.loads(path.read_text()), path, args.output.resolve(strict=True))


if __name__ == '__main__':
    raise SystemExit(main())
