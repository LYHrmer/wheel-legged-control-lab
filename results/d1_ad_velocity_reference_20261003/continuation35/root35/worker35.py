"""C35 sole physical owner; finite continuous-pair feasibility campaign."""
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
from worker30 import bootstrap30, identity, runtime_origins30, write

CONTRACT = 'C35_continuous_diagonal_pair_fixed_feasibility_v1'


def origins35(sources):
    result = runtime_origins30(sources)
    for name in ('runtime31','ledger31','record31','task31','runtime35','owner35',
                 'record35','task35','core35','adapter35','pair_ik_targets35'):
        module = sys.modules.get(name)
        if module is None or not getattr(module,'__file__',None):
            raise RuntimeError('C35 missing actual module: '+name)
        path = str(Path(module.__file__).resolve(strict=True))
        if sources.get(path) != identity(path):
            raise RuntimeError('C35 actual module escaped source freeze: '+name)
        result[name] = dict(path=path, **identity(path))
    return result


def bootstrap35(session, output):
    symbols = bootstrap30(session, output)
    from runtime31 import install_reset_yaw31
    from ledger31 import ModelLedger31
    from runtime35 import SideAccess35, NativeWindow35
    from owner35 import install_owner35
    from adapter35 import PairController35
    from record35 import record_case35
    additions=locals().copy()
    additions.pop('symbols')
    symbols.update(additions)
    symbols['origins35'] = origins35(session['source_hashes'])
    write(output/'runtime35_module_origins_before.json',symbols['origins35'])
    return symbols


def preflight35(session, session_path, output):
    spec=json.loads(Path(session['spec35_path']).read_text())
    for key,value in spec.items():
        if session.get(key) != value:
            raise RuntimeError('C35 session differs from spec: '+key)
    if (session['execution_contract_id'] != CONTRACT or session['argv'] != sys.argv
            or session['control_limit'] != 9000 or session['normal_native_cap'] != 45000
            or session['cycles_limit'] != 5 or session['retry_permitted'] is not False
            or Path(session['output_directory']).resolve() != output):
        raise RuntimeError('C35 session/argv/reservation differs')
    if session['runtime_environment']['PYTHONPATH'].split(os.pathsep)[0] != '/home/lyh/.local/lib/python3.10/site-packages':
        raise RuntimeError('C35 requires installed site-packages first')
    for name,expected in session['source_hashes'].items():
        if identity(name) != expected:
            raise RuntimeError('C35 source mismatch: '+name)
    for key,value in session['runtime_environment'].items():
        if os.environ.get(key) != value:
            raise RuntimeError('C35 runtime environment differs: '+key)
    for key in ('spec35_path','contract35_path','interface35_path'):
        path=str(Path(session[key]).resolve(strict=True))
        if session['source_hashes'].get(path) != identity(path):
            raise RuntimeError('C35 unfrozen specification: '+key)
    write(output/'preflight_ready.json',dict(pid=os.getpid(),
        session_sha256=identity(session_path)['sha256'],monotonic_s=time.monotonic()))


def run_cases35(s, session, output, runtime, guard, env, owner, policy, writer, window, access, stop):
    rows=[]
    writer.commit_json(output/'case_definition35.json',{'schema':'d1-c35-case-definition-v1','cases':session['cases']})
    baseline={}
    for side,path in session['baseline_c33_integral_paths35'].items():
        path=Path(path).resolve(strict=True)
        if session['source_hashes'].get(str(path)) != identity(path):
            raise RuntimeError('C35 unfrozen C33 baseline')
        document=json.loads(path.read_text())
        baseline[side]=document['full_case_components']['torque_normalized_square_integral_s']
    writer.commit_json(output/'baseline35.json',baseline)
    for index,definition in enumerate(session['cases']):
        if stop.is_set() or writer.stop_requested:
            raise RuntimeError('C35 stopped before next case')
        result=s['record_case35'](session,definition,runtime,guard,env,owner,policy,
            writer,window,access,stop,output,index,
            baseline['right' if definition['direction'] < 0 else 'left'])
        rows.append(result)
        writer.commit_json(output/f'case35_{index:02d}.json',result)
        writer.commit_json(output/f'progress35_{index:02d}.json',dict(
            attempted_cases=len(rows),closed_cases=sum(x['failure'] is None for x in rows),
            controls=runtime.control_completed,normal_native=runtime.returned,last_case=result))
        if result['terminal_kind'] != 'success':
            break
    result=dict(attempted_cases=len(rows),closed_cases=sum(x['failure'] is None for x in rows),
        controls=runtime.control_completed,normal_native=runtime.returned,
        pair_exchanges=sum(x['pair_exchanges'] for x in rows),
        policy_predictions=sum(x['policy_predictions'] for x in rows),
        all_cases_online_passed=len(rows)==5 and all(x['terminal_kind']=='success' for x in rows),
        stop_reason=rows[-1]['terminal_kind'] if rows else 'no_case',
        cases=rows,qualification=False,RL_speed_benefit_proven=False,user_goal_complete=False)
    writer.commit_json(output/'campaign35.json',result)
    if any(x['failure'] is not None for x in rows):
        error=RuntimeError('C35 fatal case stopped campaign; partial work retained')
        error.partial35=result
        raise error
    return result


def execute35(session, session_path, output):
    stop = threading.Event()
    failure = None
    runtime = guard = env = counter = writer = access = symbols = None
    original_copy = previous_warning = copy_edges = None
    warnings, cleanup, result = [], [], {}
    started = time.monotonic()

    def stop_request(*_):
        stop.set()
        if writer is not None:
            writer.request_stop()

    previous_term = signal.signal(signal.SIGTERM, stop_request)
    previous_alarm = signal.signal(signal.SIGALRM, stop_request)
    try:
        preflight35(session, session_path, output)
        signal.setitimer(signal.ITIMER_REAL, session["soft_s"])
        s = symbols = bootstrap35(session, output)
        common, mj = s["common"], s["mujoco"]

        class JsonWriter(s["ArchiveWriter"]):
            def commit_json(self, path, value):
                return super().commit_json(path, common._jsonable(value))

        writer = JsonWriter()

        def warning(message):
            warnings.append(str(message))
            stop.set()
            raise RuntimeError("MuJoCo warning: "+str(message))

        previous_warning = mj.get_mju_user_warning()
        mj.set_mju_user_warning(warning)
        runtime_type = s["side_owner_runtime_type"](stop)
        counter = s["ModelLedger31"](False, session["model_limits"])
        with counter:
            writer.commit_json(output/"mapped_libraries_before_load.json",
                               s["mapped_libraries"](session["source_hashes"]))
            with runtime_type(Path(session["library"]), control_limit=9000,
                              construction_limit=2) as runtime:
                with s["AtomicCourseNativeGuard"](runtime, writer) as guard:
                    env = runtime.construct(lambda: s["WorldUprightCourseEnv"](
                        caps=s["QualifiedCommandCaps"](1.6, 0., 0., .3, False),
                        command_source=lambda *_: s["FullDriveCommand"](),
                        spawn_position_m=session["spawn_position_m"],
                        max_steps=1800, mode="eval"))
                    binding = s["capture_binding24"](env.plant)
                    hybrid = s['install_owner35'](env, s['PairController35'])
                    access = s['SideAccess35'](mj, runtime, env.plant, hybrid.side.scratch)
                    access.side = hybrid.side
                    hybrid.attach_side_access(access)
                    runtime.register_side_access(access)
                    access.install()
                    original_copy, copy_edges = s["fence_copy_data30"](mj,
                        plant=env.plant, side_access=access, stop_event=stop)
                    window = s['NativeWindow35'](binding,
                        lambda: hybrid.consumed_reference35,
                        env.plant.model.jnt_range, env.plant.model.jnt_limited)
                    guard.native_sink = window.sink
                    yaw_state = s["install_reset_yaw31"](env)
                    if (runtime.state()["construction_returns"] != 2 or
                            s["_nominal_geometry"].cache_info().misses != 1):
                        raise RuntimeError("C35 one cold construction count differs")
                    writer.commit_json(output/"construction_receipt.json", dict(
                        C_state=runtime.state(), nominal_cache=s["_nominal_geometry"]
                            .cache_info()._asdict(),
                        compiled_geometry=env.plant.collision_terrain_metadata,
                        geometry_binding=s["_binding"](env.plant),
                        side_kinematic_binding=binding,
                        skill30_joint_limits=dict(jnt_range=env.plant.model.jnt_range.tolist(),
                          jnt_limited=env.plant.model.jnt_limited.tolist()),
                        side_profile=asdict(hybrid.side.cfg), proof=runtime.proof,
                        boundary=s["_boundary"](runtime), data_addresses=dict(
                          live=int(env.plant.data._address),
                          measurement=int(env.plant.measurement_data._address),
                          side_scratch=int(hybrid.side.scratch._address))))
                    env.set_native_interval_reader(guard.interval_summary,
                                                   begin_interval=guard.begin_interval)
                    counter.phase = "probe"
                    before_load = s["_boundary"](runtime)
                    B22, load_report = s["load_and_verify_final"](
                        session["checkpoint_folder"], session["checkpoint_manifest"],
                        return_model=True)
                    if (s["_boundary"](runtime) != before_load or
                            load_report["probe_actions_byte_exact"] is not True or
                            load_report["probe_rows"] != 32):
                        raise RuntimeError("C35 strict B22 probe changed physics/policy")
                    writer.commit_json(output/"strict_load.json", load_report)
                    counter.phase = "control"
                    result = run_cases35(s, session, output, runtime, guard, env,
                        hybrid, B22, writer, window, access, stop)
                    caller = s["_control_step_caller_verified"](runtime.state(),
                        runtime.proof, session["source_hashes"], s["engine_binding"])
                    access_report = access.report()
                    static_caps = session["static_API_global_caps"]
                    if (not caller or runtime.control_attempted != runtime.control_completed
                            or runtime.returned != 5*runtime.control_completed
                            or runtime.returned > 45000
                            or runtime.attempted != runtime.returned or runtime.thread_violations
                            or guard.checked != runtime.returned
                            or not access_report["within_actual_bounds"]
                            or any(access_report["counts"][key] > cap
                                   for key, cap in static_caps.items())
                            or access.rejected or copy_edges["rejected"]):
                        raise RuntimeError("C35 cumulative physical/static/copy ledger differs")
                    counts = counter.summary()["counts"]
                    expected = dict(load=1, torch_load=3,
                        predict=result["policy_predictions"]+1,
                        forward=0, evaluate_actions=0, predict_values=0,
                        backward=0, learn=0, train=0, save=0)
                    if any(counts.get(key, {}).get("returned", 0) != value
                           for key, value in expected.items()):
                        raise RuntimeError("C35 B22 model ledger differs")
                    if origins35(session["source_hashes"]) != s["origins35"]:
                        raise RuntimeError("C35 module origins changed during campaign")
                    writer.commit_json(output/"runtime35_module_origins.json", s["origins35"])
                    writer.commit_json(output/"runtime31_module_origins.json", s["origins35"])
                    writer.commit_json(output/"reset_yaw_receipt31.json", yaw_state)
                    writer.commit_json(output/"loaded_origins_final.json",
                                       s["_audited_learning_origins"](session))
                    writer.commit_json(output/"mapped_libraries_final.json",
                                       s["mapped_libraries"](session["source_hashes"]))
    except BaseException as error:
        result = getattr(error, 'partial35', result)
        failure = dict(type=type(error).__name__, message=str(error),
                       traceback=traceback.format_exc())
        print(failure["traceback"], flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for name, action in (
            ("restore_copy", lambda: setattr(symbols["mujoco"], "mj_copyData", original_copy)
             if symbols is not None and original_copy is not None else None),
            ("restore_static_queries", lambda: access.restore() if access is not None else None),
            ("restore_warning", lambda: symbols["mujoco"].set_mju_user_warning(previous_warning)
             if symbols is not None else None)):
            try:
                action()
            except BaseException as error:
                cleanup.append(name+": "+repr(error))
        receipt = dict(schema="d1-c35-headless-worker-v1",
            execution_contract_id=CONTRACT, arm=session["arm"],
            session_identity=identity(session_path), failure=failure,
            cleanup_errors=cleanup, warnings=warnings, result=result,
            C_final=None if runtime is None else runtime.state(),
            python=None if runtime is None else runtime.ledger_receipt(),
            native_guard=None if guard is None else guard.report(),
            model_calls=None if counter is None else counter.summary(),
            
            side_access=None if access is None else access.report(),
            copy_edges=copy_edges, archive_failed=None if writer is None else writer.failed,
            elapsed_wall_s=time.monotonic()-started, retry_permitted=False,
            execution_complete=failure is None and not cleanup, qualification=False)
        write(output/"worker_receipt.json", receipt if symbols is None else
              symbols["common"]._jsonable(receipt))
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGALRM, previous_alarm)
    return 0 if receipt["execution_complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.session.resolve(strict=True)
    return execute35(json.loads(path.read_text()), path, args.output.resolve(strict=True))


if __name__ == "__main__":
    raise SystemExit(main())
