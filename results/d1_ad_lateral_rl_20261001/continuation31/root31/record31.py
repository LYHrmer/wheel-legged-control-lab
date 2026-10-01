"""One C31 headless cycle on the sole prepared-control/native owner."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from scripts import run_d1_latest_rl as common
from run_rl16_training_08 import (_boundary, _capture_state,
                                  _numeric_actor_action, _require_unchanged_reset)
from full_drive_command_08 import FullDriveCommand
from state21 import control_state20
from macro31 import build_macros31
from task31 import GATES31, score_task31


FIELDS31 = ('qpos','qvel','act','ctrl','qacc_warmstart','observation','time')
MODEL_SHA31 = '7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691'


class PhysicalGateFailure31(RuntimeError):
    pass


def _capture31(env,observation):
    state=_capture_state(env,observation)
    state['act']=np.asarray(env.plant.data.act).copy()
    return state


def record_case31(session,runtime,guard,env,hybrid,policy,writer,window,clock,
                  stop_event,output,*,cycle_index,global_control_offset,
                  access,yaw_state):
    """Return (physical result, one *online* closed learning cycle or None).

    Native and control numbers are cumulative in their authority; each saved
    episode also carries the explicit local mapping. A physical/safety/archive
    failure can never be turned into a controlled reward and rescued by reset.
    """
    folder=Path(output)/f'episode_{cycle_index}'
    folder.mkdir(mode=0o755,exist_ok=False)
    rows,events=[],[]
    states={name:[] for name in FIELDS31}
    completed=predictions=0
    side_end=cancel_index=None
    previous_handoff=None
    first_error=None
    stop_reason='control_cap'
    terminated=truncated=False
    final_info=None
    observation=None
    bound=guard_started=access_started=False
    direction=int(session['direction'])
    cancel_expected=bool(session['cancel'])
    if (type(cycle_index) is not int or type(global_control_offset) is not int
            or cycle_index<0 or global_control_offset<0
            or session['cycle_index']!=cycle_index
            or session['mode'] not in ('train','learned','fixed','zero')
            or direction not in (-1,1)
            or not 1<=session['control_limit']<=2200
            or runtime.control_completed!=global_control_offset
            or runtime.returned!=5*global_control_offset):
        raise ValueError('C31 cycle reservation/mode/global offset differs')
    started_ns=time.monotonic_ns()
    fk_before=window.pure_fk_calls

    def snapshot(obs):
        value=_capture31(env,obs)
        for name in FIELDS31:
            states[name].append(value[name])
        return value

    def save(name,value):
        return writer.commit_json(folder/name,common._jsonable(value))

    try:
        runtime.start_segment(f'c31_cycle_{cycle_index}',session['control_limit'])
        runtime.bind(env)
        bound=True
        guard.start_segment(env.plant,folder,f'c31_{cycle_index}',
                            session['control_limit'],'heldout')
        guard_started=True
        access.begin_case31(cycle_index)
        access_started=True
        window.begin_case31(cycle_index,global_control_offset)
        clock.update(control_index=0,cycle_index=cycle_index,
                     global_offset=global_control_offset)
        yaw_state['initial_yaw_rad']=float(session['initial_yaw_rad'])
        before=_boundary(runtime)
        observation,info=env.reset(seed=session['seed'],options={
            'command_source':lambda *_:FullDriveCommand(),
            'spawn_position_m':session['spawn_position_m']})
        after=_boundary(runtime)
        _require_unchanged_reset(before,after)
        hybrid.side.begin_episode31(cycle_index=cycle_index,
            mode=session['mode'],direction=direction)
        initial=snapshot(observation)
        save('reset.json',dict(before=before,after=after,seed=session['seed'],
            episode_metadata=info['episode_metadata'],
            control_reset_state=control_state20(env),
            requested_initial_yaw_rad=session['initial_yaw_rad'],
            model_address=int(env.plant.model._address),
            data_address=int(env.plant.data._address)))
        writer.commit_npz(folder/'initial_state.npz',initial)

        while completed<session['control_limit']:
            if stop_event.is_set() or writer.stop_requested:
                stop_reason='soft_stop'
                break
            clock['control_index']=completed
            pre=_capture31(env,observation)
            input_obs=np.asarray(observation,dtype=np.float32).copy()
            intent=direction if completed==200 else 0
            last_skill=hybrid.side.last_skill30_record
            request_cancel=False
            if (cancel_expected and cancel_index is None and
                    hybrid.side_active and last_skill):
                velocity=np.asarray(last_skill['foot_reference_velocity_mps'],dtype=float)
                request_cancel=bool(np.linalg.norm(velocity[:2])>1e-10
                                    and abs(velocity[2])>1e-10)
                if request_cancel:
                    cancel_index=completed
            side_requested=hybrid.side_active or intent!=0
            hybrid.set_side_intent(intent,cancel=request_cancel)
            action,predicted=_numeric_actor_action(
                'zero' if side_requested else 'final_policy',policy,observation)
            predictions+=int(predicted)
            if predicted is not (not side_requested):
                raise RuntimeError('C31 B22 side prediction permission differs')
            window.begin(completed)
            observation,reward,terminated,truncated,info=runtime.control_step(env,action)
            completed+=1
            final_info=info
            interval=window.complete()
            evidence=interval['native_rows']
            hybrid.side.set_interval_evidence(
                control_index=completed-1,
                contact_free_5_by_wheel=tuple(not any(
                    r['active_wheel_terrain_contact'][j] for r in evidence)
                    for j in range(4)),
                native_wheel_min_z_5x4=[r['whole_wheel_min_z_m'] for r in evidence])
            new_events=hybrid.side.drain_macro_events()
            events.extend(copy.deepcopy(new_events))
            record=info['controller_record']
            side_tick=record.get('torque_source')=='traditional_side'
            handoff=None
            if hybrid.last_handoff_record is not previous_handoff:
                handoff=copy.deepcopy(hybrid.last_handoff_record)
                previous_handoff=hybrid.last_handoff_record
            post=snapshot(observation)
            traces=env.plant.last_control_interval_actuator_traces
            if len(traces)!=5 or info['native_interval_summary']['native_returns']!=5:
                raise RuntimeError('C31 control lacks five actual native actuator traces')
            skill=copy.deepcopy(hybrid.side.last_skill30_record) if side_tick else None
            rows.append(dict(control_index=completed-1,
                global_control_index=global_control_offset+completed-1,
                episode_index=cycle_index,episode_tick=completed-1,
                actor='traditional_side' if side_tick else
                    ('zero_rejected_side' if side_requested else 'B'),
                checkpoint_sha256=MODEL_SHA31 if predicted else None,
                policy_predict_called=predicted,
                controller_handoff_receipt=handoff,
                side_start_receipt=copy.deepcopy(hybrid.last_start_record),
                input_observation99=input_obs,raw_action16=action,
                action16=action,policy_input_action=action,
                raw_command=asdict(FullDriveCommand()),
                reward=float(reward),terminated=terminated,truncated=truncated,
                info=info,pre_state=pre,post_state=post,control_stages18=record,
                native_actuator_traces=[asdict(t) for t in traces],
                skill30=skill,macro_events30=copy.deepcopy(new_events),
                macro_events31=copy.deepcopy(new_events),
                cancel_requested30=request_cancel))
            horizontal=bool(skill and skill['phase'] in ('lift','swing','lower')
                and (skill['horizontal_active'] or
                     (last_skill or {}).get('horizontal_active',False)))
            if horizontal:
                leg=skill['leg_index']
                if (leg is None or any(r['whole_wheel_min_z_m'][leg]<=.012
                        or r['active_wheel_terrain_contact'][leg] for r in evidence)):
                    raise PhysicalGateFailure31(
                        'actual horizontal interval lost whole-wheel clearance/contact gate')
            if side_requested and not side_tick:
                stop_reason='side_start_rejected'
                break
            if side_tick and hybrid.side.done and side_end is None:
                side_end=completed
            if cancel_index is not None and side_end is None and completed-cancel_index>=300:
                stop_reason='cancel_handoff_timeout'
                break
            if side_end is not None and completed>=side_end+400:
                stop_reason='retention_complete'
                break
            if terminated or truncated:
                stop_reason=info['terminal_reason']
                break
    except BaseException as error:
        first_error=error
        stop_reason=('physical_failure' if isinstance(error,PhysicalGateFailure31)
                     else 'software_failure')
    finally:
        ended_ns=time.monotonic_ns()
        final_native=None
        if guard_started:
            try:
                final_native=guard.finish_segment()
            except BaseException as error:
                if first_error is None:
                    first_error=error
        if bound:
            try:
                runtime.unbind()
            except BaseException as error:
                if first_error is None:
                    first_error=error
        access_receipt=None
        if access_started:
            try:
                access_receipt=access.finish_case31()
            except BaseException as error:
                if first_error is None:
                    first_error=error
        arrays={key:np.asarray(value) for key,value in states.items()}
        end_reason=('physical_failure' if isinstance(first_error,PhysicalGateFailure31)
            else 'software_failure' if first_error is not None else
            'safe_cancel' if stop_reason=='retention_complete'
                and hybrid.side.failure=='cancelled' else
            'success_after_retention' if stop_reason=='retention_complete'
                and hybrid.side.success else
            'physical_failure' if hybrid.side.failure is not None or terminated
                else 'budget_truncated')
        macro_document=None
        if (events and len(rows)==completed and
                len(states['qpos'])==completed+1):
            try:
                macro_document=build_macros31(arrays,rows,events,
                    torque_limits=env.plant.actuator_torque_limit_nm,
                    origin_xy=hybrid.side.initial_pose[:2],
                    yaw0=hybrid.side.yaw,direction=direction,
                    end_reason=end_reason,
                    global_control_offset=global_control_offset)
            except BaseException as error:
                if first_error is None:
                    first_error=error
        online_task=None
        if (first_error is None and stop_reason=='retention_complete'
                and macro_document is not None and
                len(window.native_rows31)==5*completed):
            try:
                online_task=score_task31(rows,arrays,window.native_rows31,
                    binding=window.binding31,
                    geometry={int(row['geom_id']):row for row in
                        env.plant.collision_terrain_metadata['world_collision_geoms']},
                    gates=GATES31,direction=direction,
                    global_control_offset=global_control_offset,
                    cancel_expected=cancel_expected,cancel_index=cancel_index)
            except BaseException as error:
                if first_error is None:
                    first_error=error
        if not writer.failed:
            try:
                writer.commit_npz(folder/'states.npz',arrays)
                writer.commit_gzip_rows(folder/'controls.jsonl.gz',
                    [common._jsonable(row) for row in rows])
                save('segment_receipt.json',dict(cycle_index=cycle_index,
                    global_control_begin=global_control_offset,
                    global_control_end=global_control_offset+completed,
                    global_native_begin=5*global_control_offset,
                    global_native_end=5*(global_control_offset+completed),
                    completed_controls=completed,native_segment=final_native,
                    boundary=_boundary(runtime),terminated=terminated,
                    truncated=truncated,stop_reason=stop_reason))
                if macro_document is not None:
                    save('macro_transitions31.json',macro_document)
                if online_task is not None:
                    save('task31.json',online_task)
            except BaseException as error:
                if first_error is None:
                    first_error=error
        native_valid=bool(final_native is not None and
            final_native.get('record_valid') is True and
            len(window.native_rows31)==5*completed)
        archived=bool(first_error is None and not writer.failed and
            macro_document is not None and online_task is not None and native_valid)
        safe=bool(archived and online_task['safety_complete'])
        terminal_kind=(
            'safe_cancel' if safe and online_task['online_safe_cancel'] else
            'success' if safe and online_task['online_success'] else
            'controlled_failure' if safe and online_task['online_controlled_failure_eligible'] else
            'fatal_incomplete_or_unsafe')
        result=dict(cycle_index=cycle_index,completed_controls=completed,
            global_control_begin=global_control_offset,
            global_control_end=global_control_offset+completed,
            global_native_begin=5*global_control_offset,
            global_native_end=5*(global_control_offset+completed),
            policy_predictions=predictions,
            side_status=hybrid.side.status,stop_reason=stop_reason,
            side_done_completed_controls=side_end,
            cancel_control_index=cancel_index,
            actual_event_transitions=0 if macro_document is None else
                len(macro_document['transitions']),
            pure_fk_native_calls=window.pure_fk_calls-fk_before,
            task_failure=None if not isinstance(first_error,PhysicalGateFailure31)
                else str(first_error),
            end_reason=end_reason,active_start_ns=started_ns,
            active_end_ns=ended_ns,terminated=terminated,truncated=truncated,
            final_info_metrics=None if final_info is None else final_info.get('metrics'),
            online_task=online_task,terminal_kind=terminal_kind,
            fatal_failure=terminal_kind=='fatal_incomplete_or_unsafe',
            qualification=False,independent_readback_pending=True)
        learning_cycle=None
        if terminal_kind!='fatal_incomplete_or_unsafe':
            learning_cycle=dict(cycle_index=cycle_index,
                terminal_kind=terminal_kind,actual_seconds=completed*.01,
                archive_valid=True,native_valid=True,safe_handoff=True,
                retention_complete=True,macros=macro_document['transitions'],
                archive_paths={'cycle_receipt':str(folder/'cycle_receipt.json'),
                    'macro_transitions':str(folder/'macro_transitions31.json'),
                    'segment_receipt':str(folder/'segment_receipt.json'),
                    'states':str(folder/'states.npz'),
                    'controls':str(folder/'controls.jsonl.gz')},
                independent_readback_pending=True)
        if not writer.failed:
            try:
                save('cycle_receipt.json',dict(schema='d1-c31-cycle-receipt-v1',
                    cycle_index=cycle_index,mode=session['mode'],direction=direction,
                    case_id=session['case_id'],requested_initial_yaw_rad=session['initial_yaw_rad'],
                    result=result,side_access=access_receipt,
                    online_task=online_task,terminal_kind=terminal_kind,
                    learning_cycle=learning_cycle,
                    archive_completed=archived,
                    independent_readback_pending=True))
            except BaseException as error:
                if first_error is None:
                    first_error=error
                learning_cycle=None
                result['fatal_failure']=True
        if first_error is not None:
            first_error.c31_partial_result=result
            raise first_error
    return result,learning_cycle
