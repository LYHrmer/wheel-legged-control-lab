"""Root comparison of 104 separate saved-state calls; never a trajectory."""
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from scripts.d1_flat_plane_env import D1FlatPlanePlant
from scripts.d1_turn_yaw_authority import TurnYawAuthorityController
from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource

R = Path('/home/lyh/wheel-legged-control-lab')
W = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report_path = W/'turn_actuation_plan_01/report.json'
    report = json.loads(report_path.read_text())
    assert report['algebra_checks_passed'] and report['saved_pose_count'] == 208
    for path, digest in report['input_sha256'].items():
        assert sha(Path(path)) == digest
    plant = D1FlatPlanePlant()
    template = D1MujocoTruthStateSource(plant).reset()
    model, data = plant.model, plant.measurement_data
    controller = TurnYawAuthorityController()
    jac = np.zeros((3, model.nv))
    count, errors = 0, {}
    for case in report['cases']:
        if case['source'] != 'baseline':
            continue
        folder = W/'flat_plane_02'/f"stationary_turn_{case['side']}_hold"
        with gzip.open(folder/'plane_diagnostics.jsonl.gz', 'rt') as f:
            rows = [json.loads(line) for line in f]
        with np.load(folder/'states.npz') as z:
            qpos, qvel = z['qpos'].copy(), z['qvel'].copy()
        for shadow in case['samples']:
            k = shadow['execution_tick']
            data.qpos[:], data.qvel[:] = qpos[k], qvel[k]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_comVel(model, data)
            foot_jac = np.empty((4, 3, 4))
            for leg, body in enumerate(plant.wheel_body_ids_by_leg):
                mujoco.mj_jacBody(model, data, jac, None, int(body))
                foot_jac[leg] = jac[:, plant.dof_addresses[4*leg:4*leg+4]]
            lb, ab = plant.base_velocity(local=True)
            lw, aw = plant.base_velocity(local=False)
            state = replace(template, sequence=k, control_time_s=k*.01, measurement_time_s=k*.01,
                base_position=plant.base_position, base_rotation=data.xmat[plant.base_body_id].reshape(3, 3),
                base_linear_velocity_body=lb, base_angular_velocity_body=ab,
                base_linear_velocity_world=lw, base_angular_velocity_world=aw,
                joint_position=plant.joint_position, joint_velocity=plant.joint_velocity,
                foot_position=data.xpos[np.asarray(plant.wheel_body_ids_by_leg)], foot_jacobian=foot_jac)
            d = rows[k]
            controller.reset()
            controller._wheel_integral_nm[:] = shadow['I_before_nm']
            controller.bind_raw_command(forward_velocity_mps=d['raw_user_command']['forward_velocity_mps'],
                yaw_rate_rps=d['raw_user_command']['yaw_rate_rps'], control_time_s=k*.01)
            command = D1Command(forward_velocity_mps=d['servo_command']['forward_velocity_mps'],
                yaw_rate_rps=d['servo_command']['yaw_rate_rps'], base_height_m=d['servo_command']['clearance_m'])
            torque = controller.compute(command, state, np.zeros(8))
            record = controller.last_authority
            assert record.active == shadow['active']
            assert record.inner_cap_applied == (not shadow['active'])
            pairs = {
                'effective_yaw': (record.effective_yaw_request_rps, shadow['restored_inner_rps']),
                'target': (record.wheel_target_rad_s, shadow['restored_target_rad_s']),
                'PI_memory': (record.wheel_integral_after_nm, shadow['one_sample_I_after_nm']),
                'unlimited_wheel': (record.wheel_request_nm, shadow['one_sample_wheel_unlimited_request_nm']),
                'actual_wheel': (torque[3::4], shadow['one_sample_wheel_actual_safe_nm']),
                'base_target': (record.base_wheel_target_rad_s, shadow['old_target_rad_s']),
                'unchanged_leg_request': (np.asarray(controller.last_result.requested_torque_nm).reshape(4,4)[:,:3],
                                          np.asarray(d['unlimited_torque_nm']).reshape(4,4)[:,:3]),
                'unchanged_leg_torque': (torque.reshape(4,4)[:,:3],
                                         np.asarray(d['requested_torque_nm']).reshape(4,4)[:,:3]),
            }
            for label, (actual, expected) in pairs.items():
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
                errors[label] = max(errors.get(label, 0.), float(np.max(np.abs(np.asarray(actual)-expected))))
            np.testing.assert_array_equal(record.wheel_target_clipped, shadow['target_clipped'])
            count += 1
            assert data.time == plant.data.time == 0.
    assert count == 104
    files = [Path(__file__), report_path, R/'scripts/d1_turn_yaw_authority.py',
             R/'scripts/probe_d1_heading_turn_yaw_authority.py', R/'tests/test_d1_turn_yaw_authority.py']
    result = {'passed': True, 'new_physics_steps': 0, 'independent_single_state_compute_calls': count,
        'max_absolute_errors': errors, 'trajectory_or_performance_prediction': False,
        'note': 'Original saved PI memory is restored before every call. No new closed-loop states or solved-contact qualification.',
        'input_sha256': {str(p): sha(p) for p in files}}
    with (W/'turn_authority_shadow_preflight.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    def forbidden(*args, **kwargs):
        raise AssertionError('candidate preflight must not integrate')
    with patch.object(mujoco, 'mj_step', forbidden), patch.object(mujoco, 'mj_step1', forbidden), patch.object(mujoco, 'mj_step2', forbidden):
        main()
