"""Fixed 489 saved states / 1449 independent algebra calls, no transitions."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import traceback
from unittest.mock import patch

import mujoco
import numpy as np

from scripts.d1_flat_plane_env import D1FlatPlanePlant
from scripts.d1_jump_heave_action import heave_action_metadata, map_heave_action
from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.model import (
    JOINT_POSITION_HIGH, JOINT_POSITION_LOW, JOINT_TORQUE_LIMIT, JOINT_VELOCITY_LIMIT,
)
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource
from wheel_legged_control.d1.wheel_leg_controller import D1WheelLegController

R = Path('/home/lyh/wheel-legged-control-lab')
W = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920')
CASES = [('jump_late_a',175),('jump_late_b',225),('jump_lower_friction',175),
         ('jump_higher_friction',225),('no_jump_hold',None)]
LEG = np.array([k for k in range(16) if k % 4 != 3])
WHEEL = np.array([3,7,11,15])
TOL = 1e-10
COUNTS = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compare(a, b, errors, name):
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert x.shape == y.shape, name
    error = float(np.max(np.abs(x-y)))
    errors[name] = max(errors.get(name,0.),error)
    np.testing.assert_allclose(x,y,rtol=0,atol=TOL,err_msg=name)


def same_result(actual, expected, errors, prefix='zero'):
    for key,value in actual.items():
        if isinstance(value,dict):
            same_result(value,expected[key],errors,prefix+'.'+key)
        elif value is None:
            assert expected[key] is None
        else:
            compare(value,expected[key],errors,prefix+'.'+key)


def run(output):
    inputs=[Path(__file__),R/'scripts/d1_jump_heave_action.py',
            W/'rl_jump_failure_plan_01/next_contract.md']
    for name,_ in CASES:
        inputs += [W/'rl_jump_evaluation_01'/f'{name}__zero'/f
                   for f in ('states.npz','trace.jsonl.gz')]
    frozen=json.loads((R/'results/d1_budget_study/protocol.json').read_text())['source_sha256']
    inputs += [R/p for p in frozen]
    initial_hashes={str(p):sha(p) for p in inputs}
    assert len(frozen)==77 and all(sha(R/p)==digest for p,digest in frozen.items())
    plant=D1FlatPlanePlant()
    template=D1MujocoTruthStateSource(plant).reset()
    model,data=plant.model,plant.measurement_data
    controller=D1WheelLegController()
    assert controller.leg_action_scale_m==.04 and controller.wheel_action_scale_rad_s==4.
    jac=np.zeros((3,model.nv))
    errors={}; counts={'saved_states':0,'controller_calls':0,'zero_reconstructions':0,
                       'preview_calls':0,'boundaries_or_hold':0}
    conditions=[]
    with (output/'algebra.jsonl').open('x') as log:
        for name,start in CASES:
            folder=W/'rl_jump_evaluation_01'/f'{name}__zero'
            with gzip.open(folder/'trace.jsonl.gz','rt') as f:
                rows=[json.loads(line) for line in f]
            states=np.load(folder/'states.npz')
            ticks=[200] if start is None else list(range(start-1,start+121))
            case_counts={'name':name,'saved_states':len(ticks),'extension_clip_calls':0,
                         'leg_target_rate_clip_calls':0,'torque_protection_calls':0,
                         'nonzero_active_calls':0,'same_leg_torque_as_zero_calls':0,
                         'peak_requested_leg_torque_nm':0.,'peak_safe_leg_torque_nm':0.,
                         'peak_requested_wheel_torque_nm':0.}
            for k in ticks:
                row=rows[k]; archived=row['controller']
                assert row['raw_command']['forward_velocity_mps']==0.
                assert row['raw_command']['yaw_rate_rps']==0.
                assert row['info']['heading_task']['decision_tick_before']==k
                assert row['info']['heading_task']['appended_observation_decision_tick']==k+1
                assert np.array_equal(row['action'],np.zeros(8))
                # Same immutable reconstructed state is supplied to each separate call.
                data.qpos[:],data.qvel[:]=states['qpos'][k],states['qvel'][k]
                mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data);mujoco.mj_comVel(model,data)
                foot_jac=np.empty((4,3,4))
                for leg,body in enumerate(plant.wheel_body_ids_by_leg):
                    mujoco.mj_jacBody(model,data,jac,None,int(body))
                    foot_jac[leg]=jac[:,plant.dof_addresses[4*leg:4*leg+4]]
                lb,ab=plant.base_velocity(local=True);lw,aw=plant.base_velocity(local=False)
                state=replace(template,sequence=k,control_time_s=k*.01,measurement_time_s=k*.01,
                    base_position=plant.base_position,base_rotation=data.xmat[plant.base_body_id].reshape(3,3),
                    base_linear_velocity_body=lb,base_angular_velocity_body=ab,
                    base_linear_velocity_world=lw,base_angular_velocity_world=aw,
                    joint_position=plant.joint_position,joint_velocity=plant.joint_velocity,
                    foot_position=data.xpos[np.asarray(plant.wheel_body_ids_by_leg)],foot_jacobian=foot_jac)
                command=D1Command(**row['world_command'])
                active=start is not None and start<=k<start+120
                values=(-1.,0.,1.) if active else (.75,)
                counts['saved_states']+=1
                counts['boundaries_or_hold']+=int(not active)
                for value in values:
                    controller.reset()
                    before=np.array(archived['memory_before']['wheel_integral_nm'])
                    controller._wheel_integral_nm[:]=before
                    p1=controller.nominal_targets(command,state)
                    p2=controller.nominal_targets(command,state)
                    counts['preview_calls']+=2
                    assert controller.last_result is None
                    np.testing.assert_array_equal(controller.control_memory.wheel_integral_nm,before)
                    same_result(asdict(p1),asdict(p2),errors,'preview')
                    physical=map_heave_action([value],executed_tick=k,request_tick=start)
                    expected=np.array([value]*4+[0.]*4) if active else np.zeros(8)
                    np.testing.assert_array_equal(physical,expected)
                    torque=controller.compute(command,state,physical)
                    counts['controller_calls']+=1
                    result=controller.last_result
                    actual=asdict(result)
                    if not active or value==0.:
                        same_result(actual,archived,errors)
                        counts['zero_reconstructions']+=1
                    for field in ('wheel_speed_target_rad_s','wheel_nm','support_nm','support_force_n'):
                        compare(actual[field],archived[field],errors,'same_state.'+field)
                    compare(result.memory_before.wheel_integral_nm,before,errors,'pi_before')
                    # Independently account for one original wheel PI/antiwindup update.
                    wheel_error=result.wheel_speed_target_rad_s-state.joint_velocity[WHEEL]
                    candidate=np.clip(before+3.*wheel_error*.01,-4.,4.)
                    tentative=2.2*wheel_error+candidate
                    pi_after=np.where((np.abs(tentative)<=12.)|(tentative*wheel_error<0),candidate,before)
                    compare(result.memory_after.wheel_integral_nm,pi_after,errors,'pi_after')
                    compare(result.wheel_nm[WHEEL],2.2*wheel_error+pi_after,errors,'pi_request')
                    requested=controller._extension(command,0.)+.04*physical[:4]
                    compare(result.requested_extension_m,requested,errors,'requested_extension')
                    clipped=np.clip(requested,-.08,.08)
                    compare(result.leg_extension_target_m,clipped,errors,'extension_domain')
                    geometric=controller._joint_targets(clipped)
                    limited=np.clip(geometric,state.joint_position-JOINT_VELOCITY_LIMIT*.01,
                                    state.joint_position+JOINT_VELOCITY_LIMIT*.01)
                    limited=np.clip(limited,JOINT_POSITION_LOW,JOINT_POSITION_HIGH)
                    limited[WHEEL]=state.joint_position[WHEEL]
                    compare(result.joint_target_rad,limited,errors,'joint_target_limit')
                    safe=np.clip(result.requested_torque_nm,-JOINT_TORQUE_LIMIT,JOINT_TORQUE_LIMIT)
                    outward=((state.joint_position>=JOINT_POSITION_HIGH)&(safe>0))|((state.joint_position<=JOINT_POSITION_LOW)&(safe<0))
                    outward|=(np.abs(state.joint_velocity)>=JOINT_VELOCITY_LIMIT)&(safe*state.joint_velocity>0)
                    safe[outward]=0
                    compare(torque,safe,errors,'rated_safe_torque')
                    assert np.isfinite(requested).all() and np.isfinite(torque).all()
                    assert np.all(np.abs(torque)<=JOINT_TORQUE_LIMIT)
                    assert data.time==plant.data.time==0.
                    extension_clip=bool(np.any(requested!=clipped))
                    leg_rate_clip=bool(np.any(result.joint_target_rate_limited[LEG]))
                    protection=bool(np.any(result.torque_limited))
                    case_counts['extension_clip_calls']+=int(extension_clip)
                    case_counts['leg_target_rate_clip_calls']+=int(leg_rate_clip)
                    case_counts['torque_protection_calls']+=int(protection)
                    if active and value:
                        case_counts['nonzero_active_calls']+=1
                        case_counts['same_leg_torque_as_zero_calls']+=int(np.allclose(torque[LEG],np.array(archived['torque_nm'])[LEG],rtol=0,atol=TOL))
                    for label,vector in [('peak_requested_leg_torque_nm',result.requested_torque_nm[LEG]),
                                         ('peak_safe_leg_torque_nm',torque[LEG]),
                                         ('peak_requested_wheel_torque_nm',result.requested_torque_nm[WHEEL])]:
                        case_counts[label]=max(case_counts[label],float(np.max(np.abs(vector))))
                    log.write(json.dumps({'case':name,'tick':k,'policy_action':value,'active':active,
                        'physical_action':physical.tolist(),'requested_extension_m':requested.tolist(),
                        'extension_clipped':extension_clip,'leg_rate_clipped':leg_rate_clip,
                        'torque_protected':protection,'requested_torque_nm':result.requested_torque_nm.tolist(),
                        'safe_torque_nm':torque.tolist(),'PI_before':before.tolist(),'PI_after':pi_after.tolist()})+'\n')
            conditions.append(case_counts)
    assert counts['saved_states']==489 and counts['controller_calls']==1449
    assert counts['zero_reconstructions']==489 and counts['boundaries_or_hold']==9
    assert all(sha(Path(p))==digest for p,digest in initial_hashes.items())
    result={'schema':'d1-shared-heave-saved-state-qualification-v1','local_algebra_passed':True,
        'complete_environment_loader_qualified':False,'new_physics_steps':0,'new_training_steps':0,
        'tolerance_absolute':TOL,'counts':counts,'api_counts':COUNTS,'max_absolute_errors':errors,
        'action_metadata':heave_action_metadata(),'conditions':conditions,'input_sha256':initial_hashes,
        'scope':'Pure mapping and independent same-state original controller algebra only; not a rollout or reachability proof.',
        'limitations':['Fresh environment and matrix/gate-aware checkpoint interface remain unimplemented.',
            'Boundary states and PI histories are archived zero-policy states, not future candidate states.',
            'Original geometry/IK construction performs counted mj_forward calls; no integration is allowed.',
            'Reset-template contact fields are unused by this original controller; no reconstructed contact-force claims.',
            'Torque saturation or dead regions are findings, not reasons to widen limits.']}
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('local_algebra_passed','counts','api_counts','conditions')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        with ExitStack() as stack:
            for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_kinematics','mj_comPos','mj_comVel','mj_jacBody'):
                original=getattr(mujoco,name);COUNTS[name]=0
                def tracked(*a,_name=name,_original=original,**kw):
                    COUNTS[_name]+=1
                    if _name in ('mj_step','mj_step1','mj_step2'):
                        raise AssertionError('integration forbidden: '+_name)
                    return _original(*a,**kw)
                stack.enter_context(patch.object(mujoco,name,tracked))
            run(args.output)
    except BaseException:
        (args.output/'failure.json').write_text(json.dumps({'traceback':traceback.format_exc(),'api_counts':COUNTS},indent=2)+'\n')
        raise


if __name__=='__main__':
    main()
