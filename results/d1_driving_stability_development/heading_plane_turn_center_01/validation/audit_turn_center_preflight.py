"""Check Opus target/PI algebra on 100 independent saved states, no integration."""
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from scripts.d1_flat_plane_env import D1FlatPlanePlant
from scripts.d1_turn_center_compensation import TurnCenterCompensationController
from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource

R=Path('/home/lyh/wheel-legged-control-lab')
W=Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920')


def main():
    report=json.loads((W/'plane_turn_diagnosis_01/report.json').read_text())
    assert hashlib.sha256((W/'plane_turn_root_recheck.json').read_bytes()).hexdigest()==hashlib.sha256((W/'plane_turn_diagnosis_01/report.json').read_bytes()).hexdigest()
    plant=D1FlatPlanePlant()
    template=D1MujocoTruthStateSource(plant).reset()
    model,data=plant.model,plant.measurement_data
    controller=TurnCenterCompensationController()
    jac=np.zeros((3,model.nv))
    counts=0
    max_target_error=max_torque_error=0.
    for case in report['cases']:
        folder=W/'flat_plane_02'/case['case']
        with gzip.open(folder/'plane_diagnostics.jsonl.gz','rt') as f:
            rows=[json.loads(line) for line in f]
        with np.load(folder/'states.npz') as z:
            qpos,qvel=z['qpos'].copy(),z['qvel'].copy()
        for shadow in case['one_step_shadow']:
            k=shadow['execution_tick']
            data.qpos[:]=qpos[k]
            data.qvel[:]=qvel[k]
            mujoco.mj_kinematics(model,data)
            mujoco.mj_comPos(model,data)
            mujoco.mj_comVel(model,data)
            foot_jac=np.empty((4,3,4))
            for i,body in enumerate(plant.wheel_body_ids_by_leg):
                mujoco.mj_jacBody(model,data,jac,None,int(body))
                foot_jac[i]=jac[:,plant.dof_addresses[4*i:4*i+4]]
            linear_body,angular_body=plant.base_velocity(local=True)
            linear_world,angular_world=plant.base_velocity(local=False)
            state=replace(template,sequence=k,control_time_s=k*.01,measurement_time_s=k*.01,
                base_position=plant.base_position,base_rotation=data.xmat[plant.base_body_id].reshape(3,3),
                base_linear_velocity_body=linear_body,base_angular_velocity_body=angular_body,
                base_linear_velocity_world=linear_world,base_angular_velocity_world=angular_world,
                joint_position=plant.joint_position,joint_velocity=plant.joint_velocity,
                foot_position=data.xpos[np.asarray(plant.wheel_body_ids_by_leg)],foot_jacobian=foot_jac)
            # Contact fields in the template are unused by this controller;
            # this audit makes no current contact-force or contact-state claim.
            d=rows[k]
            controller.reset()
            controller._wheel_integral_nm[:]=d['wheel_integral_before_nm']
            controller.bind_raw_command(forward_velocity_mps=d['raw_user_command']['forward_velocity_mps'],
                yaw_rate_rps=d['raw_user_command']['yaw_rate_rps'],control_time_s=k*.01)
            command=D1Command(forward_velocity_mps=d['servo_command']['forward_velocity_mps'],
                yaw_rate_rps=d['servo_command']['yaw_rate_rps'],base_height_m=d['servo_command']['clearance_m'])
            controller.compute(command,state,np.zeros(8))
            target=controller.last_result.nominal_wheel_speed_rad_s
            torque=controller.last_result.wheel_nm[3::4]
            np.testing.assert_allclose(target,shadow['candidate_target_rad_s'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(torque,shadow['candidate_one_step_wheel_request_nm'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(controller.control_memory.wheel_integral_nm,shadow['candidate_one_step_integral_after_nm'],rtol=0,atol=1e-12)
            max_target_error=max(max_target_error,float(np.max(np.abs(target-shadow['candidate_target_rad_s']))))
            max_torque_error=max(max_torque_error,float(np.max(np.abs(torque-shadow['candidate_one_step_wheel_request_nm']))))
            counts+=1
            assert data.time==plant.data.time==0.
    assert counts==100
    frozen=json.loads((R/'results/d1_budget_study/protocol.json').read_text())['source_sha256']
    assert len(frozen)==77 and all(hashlib.sha256((R/p).read_bytes()).hexdigest()==v for p,v in frozen.items())
    files=[Path(__file__),R/'scripts/d1_turn_center_compensation.py',R/'scripts/probe_d1_heading_turn_center.py',R/'tests/test_d1_turn_center_compensation.py',W/'plane_turn_diagnosis_01/report.json',W/'plane_turn_diagnosis_01/next_contract.md']
    result={'passed':True,'new_physics_steps':0,'independent_single_state_computations':counts,
        'trajectory_or_performance_prediction':False,'max_target_error_rad_s':max_target_error,
        'max_wheel_request_error_nm':max_torque_error,'frozen77_unchanged':True,
        'root_diagnosis_byte_identical':True,
        'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    with (W/'turn_center_shadow_preflight.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    def forbidden(*args,**kwargs):
        raise AssertionError('no integration during candidate preflight')
    with patch.object(mujoco,'mj_step',forbidden),patch.object(mujoco,'mj_step1',forbidden),patch.object(mujoco,'mj_step2',forbidden):
        main()
