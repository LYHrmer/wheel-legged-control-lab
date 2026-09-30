"""One bounded B22/C18 owner, with a detached GLFW display and saved evidence.

This module imports no policy, viewer, or physics package before the reviewed
source/environment preflight. The control owner is the only physics thread.
"""
from __future__ import annotations

import argparse
import copy
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

from input23 import Event23, InputController23, LatestInput23, Snapshot23

PAIR = ('qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'observation')
MODEL_SHA = '7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691'
PROFILE_CASE = {'flat_0p6':'flat_0p6', 'flat_1p6':'flat_1p6',
                'yaw_1p2':'flat_1p2_yaw', 'bumps_0p4':'bumps_0p4',
                'rough_0p35':'rough_0p35',
                'ramp_0p45_complete':'ramp_0p45_complete'}
BUTTONS = {'stop':{'x':720,'y':40}, 'reset':{'x':720,'y':80}}


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return {'bytes':path.stat().st_size, 'sha256':digest.hexdigest()}


def exclusive_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def replace_json(path, value):
    target = Path(path)
    temporary = target.with_name('.'+target.name+'.'+str(os.getpid())+'.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def preflight(session, session_path, output):
    if (session.get('schema') != 'd1-c23-gui-worker-v1'
            or session.get('execution_contract_id') not in
               ('C25_B22_GUI_reset_archive_v1', 'C25_B22_user_session_v1')
            or session.get('mode') not in ('script','keyboard')
            or session.get('actor') not in ('B','zero')
            or session.get('profile') not in PROFILE_CASE
            or type(session.get('control_limit')) is not int
            or not 1 <= session['control_limit'] <= 6000
            or session.get('retry_permitted') is not False
            or session.get('output_directory') != str(output)
            or session_path != output/'session.json'
            or session.get('argv') != sys.argv):
        raise RuntimeError('GUI23 session/argv/one-shot identity differs')
    if (session['mode']=='keyboard' and
            (not session.get('render') or session['profile']!='yaw_1p2')):
        raise RuntimeError('keyboard is restricted to rendered yaw_1p2')
    if session.get('qualification'):
        expected = {'script_headless':('script',False,600),
                    'script_gui':('script',True,600),
                    'keyboard_gui':('keyboard',True,1200)}
        if (session.get('arm') not in expected or
                (session['mode'],session['render'],session['control_limit'])
                != expected[session['arm']]):
            raise RuntimeError('GUI23 qualification arm differs')
        if (session.get('actor')!='B' or session.get('seed')!=
                (231002 if session['mode']=='keyboard' else 231001) or
                session['profile']!=
                ('yaw_1p2' if session['mode']=='keyboard' else 'flat_0p6')):
            raise RuntimeError('GUI23 qualification actor/profile/seed differs')
    if abs(float(session['seconds'])-.01*session['control_limit'])>1e-9:
        raise RuntimeError('physical seconds differ from controls')
    sources = session.get('source_hashes')
    if not isinstance(sources,dict) or not sources:
        raise RuntimeError('source closure missing')
    for name, expected in sources.items():
        path = Path(name)
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or identity(path)!=expected):
            raise RuntimeError('frozen source changed: '+name)
    for key,value in session['runtime_environment'].items():
        if os.environ.get(key)!=value:
            raise RuntimeError('isolated environment differs: '+key)
    if ('LD_LIBRARY_PATH' in os.environ or
            os.environ.get('LD_PRELOAD')!=session['library'] or
            os.environ.get('LD_BIND_NOW')!='1'):
        raise RuntimeError('fixed engine preload differs')
    if session['actor']=='B':
        manifest = session['checkpoint_manifest']
        model = Path(session['checkpoint_folder'])/'final_model.zip'
        if (identity(model)['sha256']!=MODEL_SHA or
                manifest['files']['final_model.zip']['sha256']!=MODEL_SHA or
                session.get('checkpoint_sha256')!=MODEL_SHA or
                manifest.get('folder')!=session['checkpoint_folder'] or
                session['checkpoint_relocation']['current_folder']!=session['checkpoint_folder']):
            raise RuntimeError('B22 model SHA differs')
    go_path = Path(session['source_go_path'])
    go=json.loads(go_path.read_text())
    if (session['source_go_identity']!=identity(go_path) or
            go.get('decision')!='GO' or
            session['arm'] not in go['arms'] or
            any(session.get(key)!=value for key,value in go['arms'][session['arm']].items()) or
            any(session.get(key)!=value for key,value in go['shared_session'].items()) or
            any(sources.get(name)!=value for name,value in go['inputs'].items())):
        raise RuntimeError('reviewed source GO differs')
    exclusive_json(output/'preflight_ready.json', {
        'pid':os.getpid(), 'session_sha256':identity(session_path)['sha256'],
        'monotonic_s':time.monotonic(), 'wall_ns':time.monotonic_ns(),
        'source_count':len(sources), 'full_source_sha256_checked':True,
        'model_loaded':False, 'engine_imported':False})


def serialize_snapshot(sample):
    if sample is None:
        return None
    return {'sequence':sample.sequence,'timestamp_ns':sample.timestamp_ns,
            'held':sorted(sample.held),'focused':sample.focused,
            'closed':sample.closed,'events':[asdict(e) for e in sample.events],
            'events_truncated':sample.events_truncated}


def runtime_module_origins(sources):
    names=('input23','controller18','residual18','world_upright_course_11',
           'full_drive_env_08','gui13_bridge','async_course_renderer_12',
           'latest_frame_mailbox_12','engine_binding','worker22',
           'recipes18','run_rl16_training_08','rl16_learning_11','state21')
    result={'run_gui23':str(Path(__file__).resolve())}
    for name in names:
        module=sys.modules.get(name)
        path=None if module is None else getattr(module,'__file__',None)
        if path is None:
            raise RuntimeError('GUI23 used module has no imported origin: '+name)
        result[name]=str(Path(path).resolve(strict=True))
    if any(path not in sources for path in result.values()):
        raise RuntimeError('GUI23 runtime module origin left frozen source closure')
    return result


def schedule_for(session):
    from recipes18 import make_schedule
    from full_drive_command_08 import FullDriveCommand
    if session['mode']=='keyboard':
        return None, 'flat', (-8.,-4.7,.455), None
    if (session.get('qualification') and session['profile']=='flat_0p6'
            and session['control_limit']==600):
        raw = tuple(FullDriveCommand(.6 if 175<=t<425 else 0.)
                    for t in range(600))
        digest=hashlib.sha256(json.dumps([asdict(c) for c in raw],
            sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        return raw,'flat',(-8.,-4.7,.455),digest
    case = PROFILE_CASE[session['profile']]
    if session.get('mirror') and case!='flat_1p2_yaw':
        raise RuntimeError('mirror is permitted only for fixed yaw case')
    table = make_schedule(case,session['seed'],mirror=bool(session.get('mirror')))
    if len(table.raw_commands)!=session['control_limit']:
        raise RuntimeError('script horizon differs from frozen command table')
    return table.raw_commands,table.terrain,table.spawn_position_m,table.command_sha256


def run(session, session_path, output):
    from scripts import run_d1_latest_rl as common
    common._load_import_repair(session,output)
    if (os.environ.get('LD_PRELOAD')!=session['library'] or
            'LD_LIBRARY_PATH' in os.environ):
        raise RuntimeError('import repair changed engine isolation')
    import mujoco
    import numpy as np
    from archive13.atomic_archive_13 import ArchiveWriter, AtomicCourseNativeGuard
    from async_course_renderer_12 import AsyncCourseRenderer
    from full_drive_command_08 import FullDriveCommand, QualifiedCommandCaps
    from gui13_bridge import fence_copy_data, mailbox_type, owner_runtime_type
    from run_rl16_training_08 import (_boundary,_capture_state,
        _numeric_actor_action,_require_unchanged_reset,
        _control_step_caller_verified,_audited_learning_origins)
    from worker22 import mapped_libraries
    import engine_binding
    from run_world_upright_reference_11 import _binding
    from state21 import control_state20
    from runtime_support22 import ModelCounter
    from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry
    from world_upright_course_11 import WorldUprightCourseEnv
    from rl16_learning_11 import load_and_verify_final

    if _nominal_geometry.cache_info().misses or _nominal_geometry.cache_info().currsize:
        raise RuntimeError('nominal cache is not cold')
    if str(Path(engine_binding.__file__).resolve())!=session['binding_module']:
        raise RuntimeError('engine binding origin differs from frozen source')
    class Writer(ArchiveWriter):
        """Transfer bounded immutable-owned payload callbacks; drain on owner after active time."""
        def __init__(self):
            super().__init__()
            self.defer_active=False
            self.defer_begin_ns=None
            self.flush_start_ns=None
            self.flush_end_ns=None
            self.owner_ident=None
            self.pending=[]
            self.queued_jobs=[]
            self.flushed_jobs=[]
            self.peak_jobs=0
            self.native_rows=0
            self.control_rows=0
            self.episode_count=0
            self.max_jobs=64

        def begin_defer(self):
            if self.defer_active or self.pending or self.owner_ident is not None:
                raise RuntimeError('deferred archive can be armed only once')
            self.owner_ident=threading.get_ident()
            self.defer_begin_ns=time.monotonic_ns()
            self.defer_active=True

        def _commit(self,payloads):
            if not self.defer_active:
                return super()._commit(payloads)
            if threading.get_ident()!=self.owner_ident or self.failed:
                raise RuntimeError('deferred archive changed owner or already failed')
            try:
                if len(self.queued_jobs)>=self.max_jobs or not payloads:
                    raise RuntimeError('bounded deferred archive job limit exhausted')
                paths=[Path(path) for path,_ in payloads]
                manifest=paths[0].with_name(paths[0].name+'.manifest.json')
                if (any(path.parent!=paths[0].parent or path.exists() for path in paths)
                        or manifest.exists() or not paths[0].parent.is_dir()
                        or len(set(paths))!=len(paths)):
                    raise RuntimeError('deferred archive targets are not new same-folder paths')
                job_id=len(self.queued_jobs)
                kind=paths[0].name.split('_')[0]
                record={'id':job_id,'kind':kind,'payload_paths':[str(p) for p in paths],
                    'manifest_name':manifest.name,'enqueued_ns':time.monotonic_ns(),
                    'archive_status':'pending'}
                self.pending.append((job_id,payloads,manifest))
                self.queued_jobs.append(record)
                self.peak_jobs=max(self.peak_jobs,len(self.pending))
                # AtomicCourseNativeGuard only consumes the manifest name here.
                # No file or durable receipt exists until drain() completes.
                return {'manifest':manifest.name,'payloads':[],
                        'archive_status':'pending','deferred_job_id':job_id}
            except BaseException as error:
                self._first_error=error
                raise

        def commit_json(self,path,value):
            if self.defer_active:
                # Freeze the small receipt now; convert and write it only after
                # active timing ends. In particular prepared input is mutable.
                owned=copy.deepcopy(value)
                def write(stream):
                    encoded=(json.dumps(common._jsonable(owned),sort_keys=True,
                        allow_nan=False,separators=(',',':'))+'\n').encode('utf-8')
                    stream.write(encoded)
                return self._commit([(Path(path),write)])
            return super().commit_json(path,common._jsonable(value))

        def commit_gzip_rows(self,path,rows):
            if self.defer_active and Path(path).name=='controls.jsonl.gz':
                self.control_rows+=len(rows)
                if self.control_rows>session['control_limit']:
                    error=RuntimeError('deferred control row bound exceeded')
                    self._first_error=error
                    raise error
            class JsonableRows:
                def __iter__(self_nonlocal):
                    for row in rows:
                        yield common._jsonable(row)
            return super().commit_gzip_rows(path,JsonableRows())

        def commit_native_block(self,gzip_path,rows,npz_path=None,arrays=None):
            if self.defer_active:
                self.native_rows+=len(rows)
                if self.native_rows>5*session['control_limit']:
                    error=RuntimeError('deferred native row bound exceeded')
                    self._first_error=error
                    raise error
            return super().commit_native_block(gzip_path,rows,npz_path,arrays)

        def drain(self):
            if self.owner_ident!=threading.get_ident():
                raise RuntimeError('only physical owner may drain archive')
            self.defer_active=False
            self.flush_start_ns=time.monotonic_ns()
            try:
                while self.pending:
                    job_id,payloads,manifest=self.pending[0]
                    receipt=super()._commit(payloads)
                    if (receipt['manifest']!=manifest.name or
                            not manifest.is_file()):
                        raise RuntimeError('deferred archive did not publish expected manifest')
                    self.flushed_jobs.append({'id':job_id,'manifest_name':manifest.name,
                        'actual_manifest_identity':identity(manifest),
                        'completed_ns':time.monotonic_ns()})
                    self.pending.pop(0)
            except BaseException as error:
                if self._first_error is None:
                    self._first_error=error
                raise
            finally:
                self.flush_end_ns=time.monotonic_ns()

        def deferred_report(self,active_start_ns,active_end_ns,before,after):
            return {'schema':'d1-c25-deferred-archive-v1',
                'active_start_ns':active_start_ns,'active_end_ns':active_end_ns,
                'defer_begin_ns':self.defer_begin_ns,
                'flush_start_ns':self.flush_start_ns,
                'flush_end_ns':self.flush_end_ns,
                'owner_thread_ident':self.owner_ident,
                'max_jobs':self.max_jobs,'peak_jobs':self.peak_jobs,
                'control_rows':self.control_rows,'native_rows':self.native_rows,
                'episode_count':self.episode_count,
                'queued_jobs':self.queued_jobs,'flushed_jobs':self.flushed_jobs,
                'pending_count':len(self.pending),
                'failure':None if not self.failed else repr(self._first_error),
                'boundary_before_flush':before,'boundary_after_flush':after}
    writer=Writer()
    stop=threading.Event()
    ready=threading.Event()
    start=threading.Event()
    done=threading.Event()
    initial_pause=threading.Event()
    initial_ack=threading.Event()
    final_pause=threading.Event()
    final_ack=threading.Event()
    input_box=LatestInput23()
    shared={'main_ident':threading.get_ident(),'completed_controls':0}
    warning_messages=[]
    input_polls=[]
    render_frames=[]
    poll_times=[]
    owner_decisions=[]
    published_frames=[]
    raw,terrain,spawn,raw_hash=schedule_for(session)
    writer.commit_json(output/'command_identity.json',{
        'mode':session['mode'],'profile':session['profile'],
        'actor':session['actor'],'seed':session['seed'],
        'terrain':terrain,'spawn_position_m':spawn,
        'control_limit':session['control_limit'],
        'raw_command_sha256':raw_hash,
        'raw_commands':None if raw is None else [asdict(c) for c in raw],
        'keyboard_keymap':'W=forward,Q/E=+/-0.3,Space/X=stop,R=reset,Esc=exit',
        'input_ttl_s':.25})
    previous_term=signal.signal(signal.SIGTERM,lambda *_:stop.set())

    def owner():
        runtime=guard=env=counter=None
        original_copy=None
        copy_edges=None
        old_warning=None
        failure=None
        segment_receipts=[]
        completed=predictions=0
        episode_index=0
        episode_tick=0
        rows=[]
        states={name:[] for name in (*PAIR,'time')}
        observation=None
        final_info=None
        active_start_ns=active_end_ns=None
        stop_reason='control_cap'
        last_terminated=last_truncated=False
        source_started=time.monotonic()
        current_command=FullDriveCommand()
        controller=InputController23(session['profile']) if session['mode']=='keyboard' else None
        pause_proofs=[]
        prepared={'sample':None,'decision':None,'row':None,'command':current_command}
        episode_global_start=0
        first_bootstrap=True
        script_stop=False
        script_exit=False
        script_reset_used=False
        script_seen_focus=False

        def command_source(_tick,_time):
            nonlocal current_command,first_bootstrap,script_stop,script_exit
            nonlocal script_reset_used,script_seen_focus
            sample=None if first_bootstrap else input_box.latest()
            now_ns=time.monotonic_ns()
            decision=None
            origin='bootstrap_zero' if first_bootstrap else (
                'keyboard_glfw_snapshot' if session['mode']=='keyboard' else
                'frozen_script_with_gui_safety')
            if session['mode']=='keyboard':
                decision=None if sample is None else controller.update(sample,now_ns=now_ns)
                current_command=FullDriveCommand() if decision is None else FullDriveCommand(
                    decision.forward_mps,0.,decision.yaw_rps,.455,False)
                reason='bootstrap_zero' if first_bootstrap else (
                    'no_snapshot' if decision is None else decision.reason)
            else:
                if sample is not None:
                    was_focused=script_seen_focus
                    script_seen_focus |= sample.focused
                    if (sample.closed or 'escape' in sample.held or any(
                            e.kind=='close' or e.kind=='press' and e.key=='escape'
                            for e in sample.events)):
                        script_exit=True
                    if ((was_focused and not sample.focused) or
                            sample.events_truncated or
                            now_ns-sample.timestamp_ns>250_000_000 or
                            any(e.kind=='focus_lost' or e.kind=='press' and
                                e.key in ('space','x') for e in sample.events) or
                            'space' in sample.held or 'x' in sample.held):
                        script_stop=True
                    if (not script_reset_used and any(e.kind=='press' and e.key=='r'
                            for e in sample.events)):
                        script_reset_used=True
                        decision={'reset_requested':True,'exit_requested':False}
                if script_exit:
                    decision={'reset_requested':False,'exit_requested':True}
                current_command=(FullDriveCommand() if first_bootstrap or script_stop
                                 or script_exit else raw[_tick])
                reason=('bootstrap_zero' if first_bootstrap else
                        'script_exit' if script_exit else
                        'script_stop' if script_stop else 'frozen_script')
            row={'global_control_index':episode_global_start+_tick,
                 'episode_index':episode_index,'episode_tick':_tick,
                 'prepare_time_s':_time,'now_ns':now_ns,
                 'snapshot':serialize_snapshot(sample),
                 'decision':asdict(decision) if decision is not None and
                     not isinstance(decision,dict) else decision,
                 'reason':reason,'origin':origin,'raw_command':asdict(current_command),
                 'executed':False,'consumed_by_control_index':None}
            owner_decisions.append(row)
            prepared.update(sample=sample,decision=decision,row=row,command=current_command)
            first_bootstrap=False
            return current_command

        def snapshot_state():
            item=_capture_state(env,observation)
            for name in states:
                states[name].append(item[name])
            return item

        def publish(info=None):
            state=_capture_state(env,observation)
            digest=hashlib.sha256()
            for field in (*PAIR,'time'):
                value=np.asarray(state[field])
                digest.update(field.encode()+b'\0')
                digest.update(str(value.dtype).encode()+b'\0')
                digest.update(str(value.shape).encode()+b'\0')
                digest.update(value.tobytes(order='C'))
            meta={'episode_id':episode_index,'control_index':completed,
                  'episode_tick':episode_tick,'sim_time_s':float(env.plant.data.time),
                  'source_wall_ns':time.monotonic_ns(),
                  'state_sha256':digest.hexdigest(),
                  'state_hash_protocol':'fields+NUL,dtype+NUL,shape+NUL,C_bytes',
                  'requested_com_speed_mps':(current_command.forward_velocity_mps
                      if info is None else info['raw_operator_command']['forward_velocity_mps']),
                  'applied_com_speed_mps':None if info is None else
                      info['consumed_command']['forward_velocity_mps'],
                  'actual_com_speed_mps':None if info is None else
                      info['metrics'].get('body_com_vx_mps'),
                  'policy_mode':'B22 '+MODEL_SHA[:12]+' 99D/16D C18 '+session['actor'],
                  'terrain_state':terrain,
                  'stop_state':'stopped' if stop.is_set() else 'running',
                  'input_reason':None if prepared['row'] is None else prepared['row']['reason'],
                  'remaining_controls':session['control_limit']-completed}
            publication=shared['mailbox'].publish(lambda slot: mujoco.mj_copyData(
                slot,env.plant.model,env.plant.measurement_data),meta)
            published_frames.append({'metadata':meta,'publication':publication})

        def stable_pause(stage, request,ack):
            before=_boundary(runtime)
            names=('qpos','qvel','ctrl','qacc_warmstart')
            live=tuple(np.asarray(getattr(env.plant.data,name)).copy() for name in names)
            measured=tuple(np.asarray(getattr(env.plant.measurement_data,name)).copy()
                           for name in names)
            request.set()
            if not ack.wait(timeout=10):
                raise TimeoutError(stage+' stable-frame acknowledgement missing')
            after=_boundary(runtime)
            unchanged=(before==after and all(np.array_equal(a,getattr(env.plant.data,n))
                for a,n in zip(live,names)) and all(
                np.array_equal(a,getattr(env.plant.measurement_data,n))
                for a,n in zip(measured,names)))
            pause_proofs.append({'stage':stage,'unchanged':unchanged,
                'before_boundary':before,'after_boundary':after})
            if not unchanged:
                raise RuntimeError(stage+' render changed physical state/counters')

        def finish_episode():
            nonlocal rows,states
            if guard is None or guard._segment is None:
                return
            folder=output/f'episode_{episode_index}'
            segment=guard.finish_segment()
            segment_receipts.append(segment)
            if not writer.failed:
                writer.commit_gzip_rows(folder/'controls.jsonl.gz',rows)
                writer.commit_npz(folder/'states.npz',states)
                writer.commit_json(folder/'segment_receipt.json',{
                    'episode_index':episode_index,'completed_controls':len(rows),
                    'native_segment':segment,'boundary':_boundary(runtime),
                    'last_info_metrics':None if final_info is None else final_info.get('metrics')})
                writer.episode_count+=1
                if writer.episode_count>2:
                    raise RuntimeError('deferred archive episode bound exceeded')
            rows=[]
            states={name:[] for name in (*PAIR,'time')}

        def reset_episode():
            nonlocal observation,episode_tick,current_command,final_info,episode_global_start
            folder=output/f'episode_{episode_index}'
            folder.mkdir(exist_ok=False)
            remaining=session['control_limit']-completed
            env.max_steps=remaining
            current_command=FullDriveCommand()
            episode_global_start=completed
            guard.start_segment(env.plant,folder,f'gui23_{episode_index}',remaining,'heldout')
            before=_boundary(runtime)
            observation,reset_info=env.reset(seed=session['seed'],options={
                'command_source':command_source,'spawn_position_m':spawn})
            after=_boundary(runtime)
            _require_unchanged_reset(before,after)
            episode_tick=0
            final_info=None
            initial=_capture_state(env,observation)
            writer.commit_json(folder/'reset.json',{
                'before':before,'after':after,'seed':session['seed'],
                'episode_metadata':reset_info['episode_metadata'],
                'control_reset_state':control_state20(env),
                'prepared_source':prepared['row'],
                'model_address':int(env.plant.model._address),
                'data_address':int(env.plant.data._address)})
            writer.commit_npz(folder/'initial_state.npz',
                {name:initial[name] for name in (*PAIR,'time')})
            snapshot_state()
            publish()

        try:
            def warn(message):
                warning_messages.append(str(message))
                stop.set()
                raise RuntimeError('MuJoCo warning: '+str(message))
            old_warning=mujoco.get_mju_user_warning()
            mujoco.set_mju_user_warning(warn)
            runtime_type=owner_runtime_type(stop)
            with ModelCounter(False,session['model_limits']) as counter:
                writer.commit_json(output/'mapped_libraries_before_load.json',
                    mapped_libraries(session['source_hashes']))
                with runtime_type(Path(session['library']),
                                  control_limit=session['control_limit'],
                                  construction_limit=2) as runtime:
                    with AtomicCourseNativeGuard(runtime,writer) as guard:
                        caps=QualifiedCommandCaps(1.6,0.,0.,.3,False)
                        env=runtime.construct(lambda:WorldUprightCourseEnv(
                            caps=caps,command_source=command_source,
                            spawn_position_m=spawn,max_steps=session['control_limit'],
                            mode='eval'))
                        from controller18 import install_controller18
                        install_controller18(env,'combined')
                        if (_nominal_geometry.cache_info().misses!=1 or
                                runtime.state()['construction_returns']!=2):
                            raise RuntimeError('cold construction/nominal cache differs')
                        writer.commit_json(output/'construction_receipt.json',{
                            'C_state':runtime.state(),
                            'nominal_cache':_nominal_geometry.cache_info()._asdict(),
                            'compiled_geometry':env.plant.collision_terrain_metadata,
                            'geometry_binding':_binding(env.plant),
                            'proof':runtime.proof,'boundary':_boundary(runtime)})
                        env.set_native_interval_reader(guard.interval_summary,
                            begin_interval=guard.begin_interval)
                        policy=None
                        load_report=None
                        if session['actor']=='B':
                            counter.phase='probe'
                            manifest=session['checkpoint_manifest']
                            before_load=_boundary(runtime)
                            policy,load_report=load_and_verify_final(
                                session['checkpoint_folder'],manifest,return_model=True)
                            if _boundary(runtime)!=before_load or not load_report.get(
                                    'probe_actions_byte_exact'):
                                raise RuntimeError('strict B22 probe changed physical state')
                            writer.commit_json(output/'strict_load.json',load_report)
                        model=env.plant.model
                        slots=tuple(mujoco.MjData(model) for _ in range(3))
                        display=mujoco.MjData(model)
                        mailbox=mailbox_type()(slots)
                        original_copy,copy_edges=fence_copy_data(mujoco,
                            plant=env.plant,slots=slots,display=display,mailbox=mailbox,
                            owner_ident=threading.get_ident(),
                            main_ident=shared['main_ident'],stop_event=stop)
                        shared.update(model=model,display=display,mailbox=mailbox,
                                      original_copy=original_copy,
                                      data_addresses={
                                          'live':int(env.plant.data._address),
                                          'measurement':int(env.plant.measurement_data._address),
                                          'slots':[int(slot._address) for slot in slots],
                                          'display':int(display._address)},
                                      owner_thread_ident=threading.get_ident())
                        ready.set()
                        if not start.wait(timeout=30):
                            raise TimeoutError('main renderer did not start')
                        writer.begin_defer()
                        runtime.start_segment('gui23_global',session['control_limit'])
                        runtime.bind(env)
                        reset_episode()
                        stable_pause('initial',initial_pause,initial_ack)
                        active_start_ns=time.monotonic_ns()
                        while completed<session['control_limit']:
                            if stop.is_set() or writer.stop_requested:
                                stop_reason='owner_soft_stop'
                                break
                            if time.monotonic()-source_started>=session['soft_s']:
                                writer.request_stop()
                                stop_reason='owner_soft_deadline'
                                break
                            # Only the source callback inside reset/previous step
                            # prepares the command and 99D observation for this tick.
                            if session['mode']=='keyboard':
                                deadline=active_start_ns+completed*10_000_000
                                remaining_ns=deadline-time.monotonic_ns()
                                if remaining_ns>0:
                                    time.sleep(min(remaining_ns/1e9,.01))
                                if stop.is_set():
                                    break
                            sample=prepared['sample']
                            decision=prepared['decision']
                            owner_row=prepared['row']
                            consumed_command=prepared['command']
                            reset_requested=(decision.get('reset_requested',False)
                                if isinstance(decision,dict) else
                                False if decision is None else decision.reset_requested)
                            exit_requested=(decision.get('exit_requested',False)
                                if isinstance(decision,dict) else
                                False if decision is None else decision.exit_requested)
                            if reset_requested:
                                owner_row['reset_before_control_index']=completed
                                finish_episode()
                                episode_index+=1
                                reset_episode()
                                continue
                            if exit_requested:
                                stop_reason='operator_exit'
                                break
                            if owner_row['global_control_index']!=completed or owner_row['episode_tick']!=episode_tick:
                                raise RuntimeError('prepared input/observation tick differs')
                            pre=_capture_state(env,observation)
                            input_obs=np.asarray(observation,dtype=np.float32).copy()
                            counter.phase='control'
                            action,predicted=_numeric_actor_action(
                                'zero' if policy is None else 'final_policy',policy,observation)
                            if bool(predicted)!=(policy is not None):
                                raise RuntimeError('B22 actor prediction identity differs')
                            predictions+=int(predicted)
                            observation,reward,terminated,truncated,info=runtime.control_step(env,action)
                            completed+=1
                            episode_tick+=1
                            shared['completed_controls']=completed
                            if completed%5==0 or completed==session['control_limit']:
                                replace_json(output/'live_status.json',{
                                    'completed_controls':completed,'episode_index':episode_index,
                                    'episode_tick':episode_tick,'wall_ns':time.monotonic_ns()})
                            owner_row['executed']=True
                            owner_row['consumed_by_control_index']=completed-1
                            final_info=info
                            last_terminated,last_truncated=bool(terminated),bool(truncated)
                            post=snapshot_state()
                            traces=env.plant.last_control_interval_actuator_traces
                            if (len(traces)!=5 or
                                    info['native_interval_summary']['native_returns']!=5):
                                raise RuntimeError('GUI23 control/native chain incomplete')
                            rows.append({'control_index':completed-1,
                                'episode_index':episode_index,'episode_tick':episode_tick-1,
                                'actor':session['actor'],'checkpoint_sha256':
                                    MODEL_SHA if policy is not None else None,
                                'input_observation99':input_obs,'raw_action16':action,
                                'action16':action,'policy_input_action':action,
                                'policy_predict_called':predicted,
                                'raw_command':asdict(consumed_command),
                                'input_snapshot_sequence':None if sample is None else sample.sequence,
                                'reward':float(reward),'terminated':terminated,
                                'truncated':truncated,'info':info,
                                'pre_state':pre,'post_state':post,
                                'control_stages18':info['controller_record'],
                                'native_actuator_traces':[asdict(t) for t in traces]})
                            if session['render'] and (completed%5==0 or terminated or truncated):
                                publish(info)
                            if terminated or truncated:
                                stop_reason=info['terminal_reason']
                                if session['mode']=='script' and completed<session['control_limit']:
                                    raise RuntimeError('fixed script terminated before reserved horizon')
                                break
                        active_end_ns=time.monotonic_ns()
                        publish(final_info)
                        stable_pause('final',final_pause,final_ack)
                        finish_episode()
                        runtime.unbind()
                        boundary_before_flush=_boundary(runtime)
                        writer.drain()
                        boundary_after_flush=_boundary(runtime)
                        if (boundary_after_flush!=boundary_before_flush or writer.pending
                                or writer.failed or writer.control_rows!=completed
                                or writer.native_rows!=5*completed):
                            raise RuntimeError('deferred drain changed physical ledger or missed rows')
                        writer.commit_json(output/'deferred_archive_receipt.json',
                            writer.deferred_report(active_start_ns,active_end_ns,
                                boundary_before_flush,boundary_after_flush))
                        if (runtime.control_completed!=completed or
                                runtime.returned!=5*completed or
                                runtime.attempted!=runtime.returned or
                                guard.checked!=5*completed or
                                any(not s['record_valid'] for s in segment_receipts) or
                                runtime.thread_violations or copy_edges['rejected']):
                            raise RuntimeError('owner/native/copy-fence ledger differs')
                        caller=_control_step_caller_verified(runtime.state(),
                            runtime.proof,session['source_hashes'],engine_binding)
                        if not caller:
                            raise RuntimeError('frozen native control-step caller differs')
                        counts=counter.summary()['counts']
                        expected_predict=predictions+(1 if policy is not None else 0)
                        if (counts.get('predict',{}).get('returned',0)!=expected_predict
                                or counts.get('load',{}).get('returned',0)!=
                                   (1 if policy is not None else 0)
                                or counts.get('torch_load',{}).get('returned',0)!=
                                   (3 if policy is not None else 0)
                                or any(counts.get(key,{}).get('attempted',0)!=0
                                    for key in ('forward','evaluate_actions',
                                                'predict_values','backward','learn',
                                                'train','save'))):
                            raise RuntimeError('B22 model API budget differs')
                        writer.commit_json(output/'loaded_origins_final.json',
                            _audited_learning_origins(session))
                        writer.commit_json(output/'runtime_module_origins.json',
                            runtime_module_origins(session['source_hashes']))
                        writer.commit_json(output/'mapped_libraries_final.json',
                            mapped_libraries(session['source_hashes']))
                        shared['result']={'completed_controls':completed,
                            'policy_predictions':predictions,
                            'segment_receipts':segment_receipts,
                            'active_start_ns':active_start_ns,
                            'active_end_ns':active_end_ns,
                            'C_final':runtime.state(),
                            'python':runtime.ledger_receipt(),
                            'native_guard':guard.report(),
                            'model_calls':counter.summary(),
                            'thread_violations':runtime.thread_violations,
                            'copy_edges':copy_edges,
                            'owner_thread_ident':shared['owner_thread_ident'],
                            'main_thread_ident':shared['main_ident'],
                            'data_addresses':shared['data_addresses'],
                            'pause_render_proofs':pause_proofs,
                            'control_step_caller_verified':caller,
                            'warnings':warning_messages,
                            'mailbox':mailbox.snapshot_statistics(),
                            'load_report':load_report,
                            'script_raw_sha256':raw_hash,
                            'input_reset_used':False if controller is None else controller.reset_used}
                        shared['result'].update(stop_reason=stop_reason,
                            terminated=last_terminated,truncated=last_truncated,
                            final_info_metrics=None if final_info is None else final_info.get('metrics'))
            if 'result' in shared:
                shared['result']['C_final']=runtime.state()
                shared['result']['python']=runtime.ledger_receipt()
        except BaseException as error:
            failure={'type':type(error).__name__,'message':str(error),
                     'traceback':traceback.format_exc()}
            shared['owner_failure']=failure
            stop.set()
        finally:
            try:
                if guard is not None and guard._segment is not None:
                    finish_episode()
            except BaseException as error:
                shared['owner_cleanup_failure']=repr(error)
            try:
                if runtime is not None and runtime.active_plant is not None:
                    runtime.unbind()
            except BaseException as error:
                shared['owner_cleanup_failure']=repr(error)
            try:
                if writer.defer_active:
                    writer.drain()
            except BaseException as error:
                shared['owner_cleanup_failure']=repr(error)
            if original_copy is not None:
                mujoco.mj_copyData=original_copy
            mujoco.set_mju_user_warning(old_warning)
            ready.set()
            done.set()

    thread=threading.Thread(target=owner,name='gui23-physical-owner',daemon=False)
    thread.start()
    if not ready.wait(timeout=45):
        stop.set()
        thread.join(timeout=5)
        raise TimeoutError('owner did not provide detached renderer data')
    viewer=None
    main_failure=None
    seq=0
    pause_started_ns=None
    pause_end_ns=None
    button_events=[]
    unpublished_events=[]
    unpublished_overflow=False
    render_calls=0
    key_map={}
    frame_limit=int(session.get('frame_limit',1000))
    try:
        if 'model' not in shared:
            raise RuntimeError('owner failed before model/viewer setup')
        if session['render']:
            import glfw
            key_map={glfw.KEY_W:'w',glfw.KEY_Q:'q',glfw.KEY_E:'e',
                     glfw.KEY_R:'r',glfw.KEY_SPACE:'space',glfw.KEY_X:'x',
                     glfw.KEY_ESCAPE:'escape',glfw.KEY_A:'a',glfw.KEY_D:'d'}
            class Viewer23(AsyncCourseRenderer):
                def _draw_hud(self,meta,viewport):
                    mj=self.mj
                    def speed(value):
                        return '--' if value is None else f'{float(value):.3f}'
                    lines=(self.title+'\n'+
                        'raw '+speed(meta.get('requested_com_speed_mps'))+
                        '  applied '+speed(meta.get('applied_com_speed_mps'))+
                        '  actual '+speed(meta.get('actual_com_speed_mps'))+' m/s\n'+
                        'input '+str(meta.get('input_reason'))+
                        '  remaining '+str(meta.get('remaining_controls'))+'\n'+
                        'W forward, Q/E turn; A/D and jump disabled')
                    mj.mjr_overlay(mj.mjtFont.mjFONT_NORMAL,
                        mj.mjtGridPos.mjGRID_TOPLEFT,viewport,lines,'',self.context)
                    height=viewport.height
                    stop_rect=mj.MjrRect(650,height-60,140,40)
                    reset_rect=mj.MjrRect(650,height-100,140,40)
                    mj.mjr_rectangle(stop_rect,0.55,0.08,0.08,1.)
                    mj.mjr_overlay(mj.mjtFont.mjFONT_NORMAL,
                        mj.mjtGridPos.mjGRID_TOPLEFT,stop_rect,'STOP','',self.context)
                    mj.mjr_rectangle(reset_rect,0.10,0.20,0.55,1.)
                    mj.mjr_overlay(mj.mjtFont.mjFONT_NORMAL,
                        mj.mjtGridPos.mjGRID_TOPLEFT,reset_rect,'RESET','',self.context)
            viewer=Viewer23(shared['model'],shared['display'],shared['mailbox'],
                key_codes=tuple(key_map),width=800,height=500,max_fps=15,
                title='B22 '+MODEL_SHA[:12]+' | C18 | 99D/16D | '+session['profile'])
            def on_mouse(_window,button,action,_mods):
                if button!=glfw.MOUSE_BUTTON_LEFT or action!=glfw.PRESS:
                    return
                x,y=glfw.get_cursor_pos(viewer.window)
                key=('space' if 650<=x<=790 and 20<=y<60 else
                     'r' if 650<=x<=790 and 60<=y<100 else None)
                if key is not None:
                    button_events.append({'type':'button','key':key,
                                          'wall_ns':time.monotonic_ns(),'x':x,'y':y})
            glfw.set_mouse_button_callback(viewer.window,on_mouse)
            x11_id=int(glfw.get_x11_window(viewer.window)) if hasattr(glfw,'get_x11_window') else 0
            if not x11_id:
                raise RuntimeError('actual X11 window id unavailable')
            exclusive_json(output/'window_ready.json',{
                'x11_window':x11_id,'pid':os.getpid(),
                'session_sha256':identity(session_path)['sha256'],
                'wall_ns':time.monotonic_ns(),'buttons':BUTTONS})
            if session.get('require_event_driver'):
                limit=time.monotonic()+20
                while not (output/'event_driver_ready.json').exists():
                    if time.monotonic()>limit:
                        raise TimeoutError('real XTest event driver did not attach')
                    time.sleep(.01)
        start.set()
        while not initial_pause.is_set() and not done.is_set():
            time.sleep(.005)
        if initial_pause.is_set() and viewer is not None:
            for index in range(2):
                frame=viewer.render_latest(force=True,
                    capture_path=output/'frame_initial.png' if index==0 else None)
                frame['render_return_ns']=time.monotonic_ns()
                render_frames.append(frame)
                render_calls+=1
        initial_ack.set()
        last_render_call_ns=time.monotonic_ns() if render_calls else 0
        while not done.is_set() and not final_pause.is_set():
            if viewer is not None:
                poll=viewer.poll()
                poll_times.append(poll['wall_ns'])
                button_now=tuple(button_events)
                button_events.clear()
                events=[]
                for item in poll['events']:
                    if item['type']=='focus':
                        events.append(Event23('focus_gained' if item['focused'] else 'focus_lost'))
                    elif item['type']=='key' and item['action'] in (glfw.PRESS,glfw.RELEASE):
                        events.append(Event23('press' if item['action']==glfw.PRESS else 'release',
                                              key_map.get(item['key'],'unsupported')))
                events.extend(Event23('press',item['key']) for item in button_now)
                held=frozenset(key_map[key] for key in poll['held_keys'])
                unpublished_events.extend(events)
                unpublished_overflow |= poll['events_truncated']
                if len(unpublished_events)>4096:
                    unpublished_events.clear()
                    unpublished_overflow=True
                sample=Snapshot23(seq,poll['wall_ns'],held,poll['focused'],
                    poll['window_close'],tuple(unpublished_events),unpublished_overflow)
                seq+=1
                paused=False
                if session.get('test_input_pause'):
                    at=session['test_input_pause']['start_control']
                    duration=float(session['test_input_pause']['duration_s'])
                    if shared['completed_controls']>=at and pause_started_ns is None:
                        pause_started_ns=time.monotonic_ns()
                    if (pause_started_ns is not None and pause_end_ns is None and
                            time.monotonic_ns()-pause_started_ns<int(duration*1e9)):
                        paused=True
                    elif pause_started_ns is not None and pause_end_ns is None:
                        pause_end_ns=time.monotonic_ns()
                publish_begin_ns=None
                publish_wall_ns=None
                if not paused:
                    publish_begin_ns=time.monotonic_ns()
                    input_box.publish(sample)
                    publish_wall_ns=time.monotonic_ns()
                    unpublished_events.clear()
                    unpublished_overflow=False
                input_polls.append({'poll':poll,'raw_button_events':button_now,
                    'snapshot':serialize_snapshot(sample),'published':not paused,
                    'publish_begin_ns':publish_begin_ns,
                    'publish_wall_ns':publish_wall_ns,
                    'global_control_index':shared['completed_controls']})
                render_now_ns=time.monotonic_ns()
                if (render_calls<frame_limit-2 and
                        render_now_ns-last_render_call_ns>=66_666_667):
                    midpoint=shared['completed_controls']>=session['control_limit']//2
                    capture=(output/'frame_drive.png' if midpoint and not any(
                        f.get('capture') and f['capture']['path'].endswith('frame_drive.png')
                        for f in render_frames) else None)
                    frame=viewer.render_latest(capture_path=capture)
                    frame['render_return_ns']=time.monotonic_ns()
                    render_calls+=1
                    last_render_call_ns=render_now_ns
                    if frame['status']!='skipped':
                        render_frames.append(frame)
            else:
                poll_times.append(time.monotonic_ns())
            time.sleep(.015)
        if final_pause.is_set() and viewer is not None:
            for index in range(2):
                frame=viewer.render_latest(force=True,
                    capture_path=output/'frame_final.png' if index==0 else None)
                frame['render_return_ns']=time.monotonic_ns()
                render_frames.append(frame)
                render_calls+=1
        final_ack.set()
    except BaseException as error:
        main_failure={'type':type(error).__name__,'message':str(error),
                      'traceback':traceback.format_exc()}
        stop.set()
        start.set()
        initial_ack.set()
        final_ack.set()
    finally:
        if viewer is not None:
            viewer.close()
        thread.join(timeout=8)
        signal.signal(signal.SIGTERM,previous_term)
    if thread.is_alive():
        raise TimeoutError('owner still active; cannot serialize concurrent evidence')
    result=shared.get('result',{})
    failure=main_failure or shared.get('owner_failure')
    if shared.get('owner_cleanup_failure'):
        failure=failure or {'type':'OwnerCleanupError','message':shared['owner_cleanup_failure']}
    try:
        if not writer.failed:
            writer.commit_gzip_rows(output/'input_snapshots.jsonl.gz',
                [common._jsonable(row) for row in input_polls])
            writer.commit_gzip_rows(output/'owner_decisions.jsonl.gz',
                [common._jsonable(row) for row in owner_decisions])
            writer.commit_json(output/'render_frames.json',render_frames)
            writer.commit_json(output/'performance.json',{
                'active_start_ns':result.get('active_start_ns'),
                'active_end_ns':result.get('active_end_ns'),
                'poll_timestamps_ns':poll_times,
                'render_api_calls':render_calls,
                'render_frames':render_frames,
                'published_frame_snapshots':published_frames,
                'input_pause_start_ns':pause_started_ns,
                'input_pause_end_ns':pause_end_ns})
    except BaseException as error:
        failure=failure or {'type':type(error).__name__,'message':str(error)}
    receipt={'schema':'d1-c23-gui-worker-receipt-v1',
        'execution_contract_id':session['execution_contract_id'],
        'arm':session['arm'],'session_identity':identity(session_path),
        'failure':failure,'result':result,'warnings':warning_messages,
        'archive_failed':writer.failed,'writer_committed':list(writer.committed),
        'writer_partial':list(writer.partial),'input_polls':len(input_polls),
        'owner_decisions':len(owner_decisions),
        'render_frames':len(render_frames),'retry_permitted':False,
        'independent_readback_pending':True}
    if writer.failed:
        writer.commit_failure_receipt(output/'worker_failure_receipt.json',
            common._jsonable(receipt))
    else:
        writer.commit_json(output/'worker_receipt.json',receipt)
    return 0 if failure is None and not writer.failed else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    session_path=args.session.resolve(strict=True)
    output=args.output.resolve(strict=True)
    session=json.loads(session_path.read_text())
    try:
        preflight(session,session_path,output)
        return run(session,session_path,output)
    except BaseException as error:
        failure={'type':type(error).__name__,'message':str(error),
                 'traceback':traceback.format_exc(),'phase':'preflight_or_import'}
        if not (output/'worker_receipt.json').exists():
            exclusive_json(output/'worker_receipt.json',{
                'schema':'d1-c23-gui-worker-receipt-v1','failure':failure,
                'retry_permitted':False})
        print(failure['traceback'],flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
