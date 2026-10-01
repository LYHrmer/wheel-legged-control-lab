"""One headless C30 case, using the existing prepared 100 Hz / native 5T chain."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from scripts import run_d1_latest_rl as common
from run_rl16_training_08 import _boundary, _capture_state, _numeric_actor_action, _require_unchanged_reset
from full_drive_command_08 import FullDriveCommand
from state21 import control_state20
from macro30 import build_macros30, components30

FIELDS = ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart', 'observation', 'time')
MODEL_SHA = '7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691'


class PhysicalGateFailure30(RuntimeError):
    pass


def capture30(env, observation):
    state = _capture_state(env, observation)
    state['act'] = np.asarray(env.plant.data.act).copy()
    return state


def record_case30(session, runtime, guard, env, hybrid, policy, writer, window,
                  clock, stop_event, output):
    folder = Path(output)/'episode_0'
    folder.mkdir()
    rows, events = [], []
    states = {name: [] for name in FIELDS}
    completed = predictions = 0
    side_end = cancel_index = None
    previous_handoff = None
    first_error = None
    stop_reason = 'control_cap'
    terminated = truncated = False
    final_info = None
    observation = None
    bound = guard_started = False
    direction = int(session['direction'])
    started_ns = time.monotonic_ns()

    def snapshot(obs):
        value = capture30(env, obs)
        for name in FIELDS:
            states[name].append(value[name])
        return value

    def save(name, value):
        return writer.commit_json(folder/name, common._jsonable(value))

    try:
        runtime.start_segment('skill30_global', session['control_limit'])
        runtime.bind(env)
        bound = True
        guard.start_segment(env.plant, folder, 'skill30_0', session['control_limit'], 'heldout')
        guard_started = True
        before = _boundary(runtime)
        clock['control_index'] = 0
        observation, info = env.reset(seed=session['seed'], options={
            'command_source': lambda *_: FullDriveCommand(),
            'spawn_position_m': session['spawn_position_m']})
        after = _boundary(runtime)
        _require_unchanged_reset(before, after)
        initial = snapshot(observation)
        save('reset.json', dict(before=before, after=after, seed=session['seed'],
            episode_metadata=info['episode_metadata'], control_reset_state=control_state20(env),
            model_address=int(env.plant.model._address), data_address=int(env.plant.data._address)))
        writer.commit_npz(folder/'initial_state.npz', initial)

        while completed < session['control_limit']:
            if stop_event.is_set() or writer.stop_requested:
                stop_reason = 'soft_stop'
                break
            clock['control_index'] = completed
            pre = capture30(env, observation)
            input_obs = np.asarray(observation, dtype=np.float32).copy()
            intent = direction if completed == 200 else 0
            last_skill = hybrid.side.last_skill30_record
            request_cancel = False
            if (session['arm'].startswith('cancel_') and cancel_index is None
                    and hybrid.side_active and last_skill):
                velocity = np.asarray(last_skill['foot_reference_velocity_mps'], dtype=float)
                request_cancel = bool(np.linalg.norm(velocity[:2]) > 1e-10 and abs(velocity[2]) > 1e-10)
                if request_cancel:
                    cancel_index = completed
            side_requested = hybrid.side_active or intent != 0
            hybrid.set_side_intent(intent, cancel=request_cancel)
            action, predicted = _numeric_actor_action('zero' if side_requested else 'final_policy',
                                                      policy, observation)
            predictions += int(predicted)
            if predicted is not (not side_requested):
                raise RuntimeError('C30 B22 side prediction permission differs')
            window.begin(completed)
            observation, reward, terminated, truncated, info = runtime.control_step(env, action)
            completed += 1
            final_info = info
            interval = window.complete()
            evidence = interval['native_rows']
            hybrid.side.set_interval_evidence(
                control_index=completed-1,
                contact_free_5_by_wheel=tuple(not any(r['active_wheel_terrain_contact'][j]
                                                     for r in evidence) for j in range(4)),
                native_wheel_min_z_5x4=[r['whole_wheel_min_z_m'] for r in evidence])
            new_events = hybrid.side.drain_macro_events()
            events.extend(copy.deepcopy(new_events))
            record = info['controller_record']
            side_tick = record.get('torque_source') == 'traditional_side'
            handoff = None
            if hybrid.last_handoff_record is not previous_handoff:
                handoff = copy.deepcopy(hybrid.last_handoff_record)
                previous_handoff = hybrid.last_handoff_record
            post = snapshot(observation)
            traces = env.plant.last_control_interval_actuator_traces
            if len(traces) != 5 or info['native_interval_summary']['native_returns'] != 5:
                raise RuntimeError('C30 control does not have five native actuator traces')
            skill = copy.deepcopy(hybrid.side.last_skill30_record) if side_tick else None
            rows.append(dict(control_index=completed-1, episode_index=0, episode_tick=completed-1,
                actor='traditional_side' if side_tick else ('zero_rejected_side' if side_requested else 'B'),
                checkpoint_sha256=MODEL_SHA if predicted else None,
                policy_predict_called=predicted, controller_handoff_receipt=handoff,
                side_start_receipt=copy.deepcopy(hybrid.last_start_record),
                input_observation99=input_obs, raw_action16=action, action16=action,
                policy_input_action=action, raw_command=asdict(FullDriveCommand()),
                reward=float(reward), terminated=terminated, truncated=truncated,
                info=info, pre_state=pre, post_state=post, control_stages18=record,
                native_actuator_traces=[asdict(t) for t in traces], skill30=skill,
                macro_events30=copy.deepcopy(new_events), cancel_requested30=request_cancel))
            horizontal_interval = bool(skill and skill['phase'] in ('lift', 'swing', 'lower')
                and (skill['horizontal_active'] or (last_skill or {}).get('horizontal_active', False)))
            if horizontal_interval:
                leg = skill['leg_index']
                if (leg is None or any(r['whole_wheel_min_z_m'][leg] <= .012
                                       or r['active_wheel_terrain_contact'][leg] for r in evidence)):
                    raise PhysicalGateFailure30('actual horizontal interval lost whole-wheel clearance/contact gate')
            if side_requested and not side_tick:
                stop_reason = 'side_start_rejected'
                break
            if side_tick and hybrid.side.done and side_end is None:
                side_end = completed
            if cancel_index is not None and side_end is None and completed-cancel_index >= 300:
                stop_reason = 'cancel_handoff_timeout'
                break
            if side_end is not None and completed >= side_end+400:
                stop_reason = 'retention_complete'
                break
            if terminated or truncated:
                stop_reason = info['terminal_reason']
                break
    except BaseException as error:
        first_error = error
        stop_reason = 'physical_failure' if isinstance(error, PhysicalGateFailure30) else 'software_failure'
    finally:
        ended_ns = time.monotonic_ns()
        final_native = None
        if guard_started:
            try:
                final_native = guard.finish_segment()
                guard_started = False
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if bound:
            runtime.unbind()
        arrays = {k: np.asarray(v) for k, v in states.items()}
        end_reason = ('physical_failure' if isinstance(first_error, PhysicalGateFailure30) else
                      'software_failure' if first_error is not None else
                      'safe_cancel' if stop_reason == 'retention_complete' and hybrid.side.failure == 'cancelled' else
                      'success_after_retention' if stop_reason == 'retention_complete' and hybrid.side.success else
                      'physical_failure' if hybrid.side.failure is not None or terminated else 'budget_truncated')
        macros = []
        whole_components = None
        if events and len(rows) == completed and len(states['qpos']) == completed+1:
            macros = build_macros30(arrays, rows, events,
                torque_limits=env.plant.actuator_torque_limit_nm,
                origin_xy=hybrid.side.initial_pose[:2], yaw0=hybrid.side.yaw,
                direction=direction, end_reason=end_reason)
            whole_components = components30(arrays['qpos'],
                [[t['applied_nm'] for t in row['native_actuator_traces']] for row in rows],
                env.plant.actuator_torque_limit_nm,
                origin_xy=hybrid.side.initial_pose[:2], yaw0=hybrid.side.yaw, direction=direction)
        if not writer.failed:
            writer.commit_npz(folder/'states.npz', arrays)
            writer.commit_gzip_rows(folder/'controls.jsonl.gz', [common._jsonable(r) for r in rows])
            save('segment_receipt.json', dict(episode_index=0, completed_controls=completed,
                native_segment=final_native, boundary=_boundary(runtime),
                terminated=terminated, truncated=truncated, stop_reason=stop_reason))
            save('macro_transitions30.json', dict(schema='d1-c30-macros-v1', events=events,
                transitions=macros, full_case_components=whole_components,
                elapsed_controls=completed, event_transition_count=len(macros),
                direction=direction, origin_xy=hybrid.side.initial_pose[:2], yaw0=hybrid.side.yaw,
                end_reason=end_reason, training=False))
        result = dict(completed_controls=completed, policy_predictions=predictions,
            side_status=hybrid.side.status, stop_reason=stop_reason,
            side_done_completed_controls=side_end, cancel_control_index=cancel_index,
            actual_event_transitions=len(macros), pure_fk_native_calls=window.pure_fk_calls,
            task_failure=None if not isinstance(first_error, PhysicalGateFailure30) else str(first_error),
            end_reason=end_reason,
            active_start_ns=started_ns, active_end_ns=ended_ns,
            terminated=terminated, truncated=truncated,
            final_info_metrics=None if final_info is None else final_info.get('metrics'),
            qualification=False, independent_readback_pending=True)
        if first_error is not None and not isinstance(first_error, PhysicalGateFailure30):
            first_error.c30_partial_result = result
            raise first_error
    return result
