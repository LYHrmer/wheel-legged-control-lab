"""C35 actual case archive; no old single-leg task/macro classification."""
from __future__ import annotations

import time
import traceback
from dataclasses import asdict
import numpy as np

from scripts import run_d1_latest_rl as common
from run_rl16_training_08 import _boundary, _numeric_actor_action, _require_unchanged_reset
from full_drive_command_08 import FullDriveCommand
from record31 import FIELDS31, _capture31
from state21 import control_state20
from core35 import Command35
from task35 import sensed35, entry35, score35, stopped_post35


def command35(definition, *, control, side_control, cancel_control):
    cancel = cancel_control is not None and control >= cancel_control
    release = side_control >= definition['active_controls']
    stopping = cancel or release
    return Command35(vy_mps=0. if stopping else definition['vy_mps'],
        gait_enabled=not stopping, issued_control_index=control,
        expires_control_index=control+20,
        stop_kind='cancel' if cancel else 'release' if release else 'none')


def record_case35(session, definition, runtime, guard, env, owner, policy,
                  writer, window, access, stop_event, output, index, baseline_integral):
    folder = output/f'episode_{index}'
    folder.mkdir(exist_ok=False)
    offset = runtime.control_completed
    native_start = runtime.returned
    states = {k: [] for k in FIELDS31}
    rows, events = [], []
    completed = predictions = 0
    side_end = cancel_index = stop_index = None
    first_error = None
    entry = task = final_native = access_receipt = pending = None
    guard_started = bound = access_started = False
    terminated = truncated = False
    started = time.monotonic()

    def save(name, value):
        return writer.commit_json(folder/name, value)

    def snapshot(observation):
        value = _capture31(env, observation)
        for key in states:
            states[key].append(value[key])
        return value

    try:
        if offset >= 9000 or runtime.returned != offset*5 or index >= 5:
            raise RuntimeError('C35 case reservation exceeds cumulative authority')
        runtime.start_segment('c35_'+definition['id'], min(1800, 9000-offset))
        runtime.bind(env)
        bound = True
        guard.start_segment(env.plant, folder, name='c35_'+definition['id'],
                            control_limit=1800, mode='heldout')
        guard_started = True
        access.begin_case35(index)
        access_started = True
        window.begin_case31(index, offset)
        before = _boundary(runtime)
        observation, reset_info = env.reset(seed=session['seed'], options={
            'command_source': lambda *_: FullDriveCommand(),
            'spawn_position_m': session['spawn_position_m']})
        after = _boundary(runtime)
        _require_unchanged_reset(before, after)
        initial = snapshot(observation)
        save('reset.json', dict(case_index=index, case_id=definition['id'], before=before,
            after=after, reset_info=reset_info, control_state=control_state20(env),
            model_address=int(env.plant.model._address), data_address=int(env.plant.data._address),
            global_control_offset=offset))
        writer.commit_npz(folder/'initial_state.npz', initial)
        while completed < 1800:
            if stop_event.is_set() or writer.stop_requested:
                raise RuntimeError('C35 owner soft stop')
            obs_input = np.asarray(observation).copy()
            pre = _capture31(env, obs_input)
            pre_control_state = control_state20(env)
            side_requested = completed >= 200 and side_end is None
            cmd = sensed = None
            if side_requested:
                if completed-200 >= 1200:
                    raise RuntimeError('C35 side control budget exhausted')
                sensed = sensed35(window.binding31, pre['qpos'], pre['qvel'],
                                  window.native_rows31[-5:], completed)
                if completed == 200:
                    entry = entry35(sensed, pre['qpos'], pre['qvel'], window.binding31)
                    save('entry35.json', entry)
                    if not entry['passed']:
                        raise RuntimeError('C35 unchanged entry stability gate failed')
                cmd = command35(definition, control=completed, side_control=completed-200,
                                cancel_control=cancel_index)
                if cmd.stop_kind != 'none' and stop_index is None:
                    stop_index = completed
            owner.set_command35(cmd, sensed, begin=completed == 200)
            action, predicted = _numeric_actor_action('zero' if side_requested else 'final_policy',
                                               policy, obs_input)
            if predicted is not (not side_requested):
                raise RuntimeError('C35 actual B22 prediction permission differs')
            predictions += int(not side_requested)
            pending = dict(control_index=completed, global_control_index=offset+completed,
                           policy_action=np.asarray(action).copy(), side_requested=side_requested,
                           pre_state=pre)
            window.begin(completed)
            next_obs, reward, terminated, truncated, info = runtime.control_step(env, action)
            interval = window.complete()
            actual_actor = info['controller_record'].get('torque_source')
            if side_requested != (actual_actor == 'continuous_pair35'):
                raise RuntimeError('C35 actual torque owner differs from declared intent')
            actual_side = info['controller_record'].get('controller_record35')
            if actual_side is not None:
                new_events = actual_side.get('pair_events', [])
                events.extend(new_events)
                if len(events) > 16:
                    raise RuntimeError('C35 pair exchange cap exceeded')
                reference = actual_side['reference']
                velocities = np.asarray(reference['feet_velocity_world_mps'])
                horizontal = bool(np.any(np.linalg.norm(velocities[:, :2], axis=1) > 1e-12))
                if definition['kind'] == 'cancel' and cancel_index is None and horizontal:
                    cancel_index = completed+1
            observation = np.asarray(next_obs).copy()
            post = snapshot(observation)
            traces = tuple(env.plant.last_control_interval_actuator_traces)
            if len(traces) != 5:
                raise RuntimeError('C35 actual control must contain five actuator traces')
            row = dict(schema='d1-c35-control-row-v1', control_index=completed,
                episode_index=index, episode_tick=completed,
                cycle_index=index, global_control_index=offset+completed,
                native_index_range=[5*(offset+completed), 5*(offset+completed+1)],
                pre_state_index=completed, post_state_index=completed+1,
                pre_control_state=pre_control_state,
                next_control_state=control_state20(env),
                input_observation=obs_input, next_observation=observation.copy(),
                actor=actual_actor if side_requested else 'final_policy', policy_action=np.asarray(action),
                raw_command=asdict(FullDriveCommand()),
                native_actuator_traces=[asdict(t) for t in traces],
                pre_state=pre, post_state=post,
                input_observation99=obs_input, raw_action16=np.asarray(action),
                action16=np.asarray(action), policy_input_action=np.asarray(action),
                control_stages18=info['controller_record'],
                policy_predict_called=not side_requested,
                checkpoint_sha256=None if side_requested else
                    '7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691',
                actual_start_time_s=float(pre['time']), actual_end_time_s=float(post['time']),
                info=info, reward=float(reward), terminated=bool(terminated), truncated=bool(truncated),
                side_interval35=interval, handoff_record=owner.last_handoff_record,
                start_record=owner.last_start_record, side_requested=side_requested)
            rows.append(row)
            completed += 1
            pending = None
            if actual_side is not None and owner.side.done and side_end is None:
                side_end = completed
                recent = window.native_rows31[-250:][4::5]
                if len(recent) != 50 or not all(stopped_post35(r, window.binding31) for r in recent):
                    raise RuntimeError('C35 handoff missing 50 actual final-post-state stop gates')
                if owner.side.failure is not None or not owner.side.success:
                    raise RuntimeError('C35 side finished with controller failure')
            if owner.side.failure is not None and side_requested:
                raise RuntimeError('C35 controller failure: '+str(owner.side.failure))
            if stop_index is not None and side_end is None and completed-stop_index >= 250:
                raise RuntimeError('C35 stop/handoff deadline exceeded')
            if terminated or truncated:
                raise RuntimeError('C35 original environment terminated/truncated')
            if side_end is not None and completed-side_end == 400:
                break
        arrays = {k: np.asarray(v) for k, v in states.items()}
        task = score35(rows, arrays, window.native_rows31, binding=window.binding31,
            geometry={int(r['geom_id']): r for r in
                      env.plant.collision_terrain_metadata['world_collision_geoms']},
            torque_limits=env.plant.actuator_torque_limit_nm, definition=definition,
            side_end=side_end, stop_index=stop_index, pair_events=events,
            baseline_integral=baseline_integral, entry=entry)
    except BaseException as error:
        first_error = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        if pending is not None:
            pending['actual_controller_record'] = owner.last_record
            pending['actual_last_side_record'] = owner.side.last_record
    finally:
        for flag, name, close in ((guard_started, 'native', guard.finish_segment),
                                  (bound, 'unbind', runtime.unbind),
                                  (access_started, 'access', access.finish_case35)):
            if flag:
                try:
                    result = close()
                    if name == 'native':
                        final_native = result
                    elif name == 'access':
                        access_receipt = result
                except BaseException as error:
                    if first_error is None:
                        first_error = dict(type=type(error).__name__, message=str(error), where=name)
        arrays = {k: np.asarray(v) for k, v in states.items()}
        native_end = runtime.returned
        terminal = ('fatal_incomplete_or_unsafe' if first_error is not None else
                    'success' if task and task['eligible'] else 'controlled_quality_failure')
        result = dict(case_index=index, case_id=definition['id'], completed_controls=completed,
            actual_runtime_control_returns=runtime.control_completed-offset,
            actual_normal_native=native_end-native_start, policy_predictions=predictions,
            side_done_completed_controls=side_end, pair_exchanges=len(events),
            failure=first_error, terminal_kind=terminal, task=task,
            elapsed_s=time.monotonic()-started, pending_control=pending,
            terminated=bool(terminated), truncated=bool(truncated),
            qualification=False, independent_readback_pending=True)
        if not writer.failed:
            writer.commit_npz(folder/'states.npz', arrays)
            writer.commit_gzip_rows(folder/'controls.jsonl.gz', [common._jsonable(r) for r in rows])
            save('segment_receipt.json', dict(completed_controls=completed, native_segment=final_native,
                global_control_begin=offset, global_control_end=runtime.control_completed,
                global_native_begin=native_start, global_native_end=native_end, boundary=_boundary(runtime)))
            save('cycle_receipt35.json', dict(schema='d1-c35-cycle-receipt-v1',
                case_id=definition['id'], case_index=index, direction=definition['direction'],
                kind=definition['kind'], terminal_kind=terminal,
                control_range=[offset, runtime.control_completed], native_range=[native_start, native_end],
                command=dict(definition=definition, side_begin_control=200, side_end_control=side_end,
                             stop_control_index=stop_index, cancel_control_index=cancel_index),
                pair_events=events, result=result, side_access=access_receipt,
                archive_completed=first_error is None and final_native is not None
                    and final_native['record_valid'], independent_readback_pending=True))
    return result
