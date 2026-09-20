"""Check the integrated Opus core at 104 independent original saved states."""
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from scripts.d1_flat_plane_env import D1FlatPlanePlant
from scripts.d1_turn_leg_damping import TurnLegDampingController
from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.model import JOINT_POSITION_HIGH, JOINT_POSITION_LOW, JOINT_TORQUE_LIMIT, JOINT_VELOCITY_LIMIT
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource

R = Path('/home/lyh/wheel-legged-control-lab')
W = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report_path = W/'turn_next_mechanics_plan_01/report.json'
    report = json.loads(report_path.read_text())
    assert report['algebra_checks_passed'] and report['saved_pose_samples'] == 208
    for path, digest in report['input_sha256'].items():
        assert sha(Path(path)) == digest
    plant = D1FlatPlanePlant()
    template = D1MujocoTruthStateSource(plant).reset()
    model, data = plant.model, plant.measurement_data
    controller = TurnLegDampingController()
    jac = np.zeros((3, model.nv))
    count, max_delta, max_request, max_wheel = 0, 0., 0., 0.
    for episode in report['episodes']:
        if episode['summary']['source'] != 'baseline':
            continue
        folder = W/'flat_plane_02'/f"stationary_turn_{episode['summary']['side']}_hold"
        with gzip.open(folder/'plane_diagnostics.jsonl.gz', 'rt') as f:
            rows = [json.loads(line) for line in f]
        with np.load(folder/'states.npz') as z:
            qpos, qvel = z['qpos'].copy(), z['qvel'].copy()
        for shadow in episode['samples']:
            k = shadow['state_tick']
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
            controller._wheel_integral_nm[:] = d['wheel_integral_before_nm']
            controller.bind_raw_command(forward_velocity_mps=d['raw_user_command']['forward_velocity_mps'],
                yaw_rate_rps=d['raw_user_command']['yaw_rate_rps'], control_time_s=k*.01)
            command = D1Command(forward_velocity_mps=d['servo_command']['forward_velocity_mps'],
                yaw_rate_rps=d['servo_command']['yaw_rate_rps'], base_height_m=d['servo_command']['clearance_m'])
            torque = controller.compute(command, state, np.zeros(8))
            record = controller.last_damping
            assert record.active == shadow['active']
            delta = np.asarray(shadow['delta_joint_torque_nm'])
            np.testing.assert_allclose(record.delta_torque_nm, delta, rtol=0, atol=1e-12)
            np.testing.assert_allclose(record.base_requested_torque_nm, d['unlimited_torque_nm'], rtol=0, atol=1e-12)
            request = np.asarray(d['unlimited_torque_nm'])+delta
            safe = np.clip(request, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
            q, qd = state.joint_position, state.joint_velocity
            safe[((q >= JOINT_POSITION_HIGH) & (safe > 0.)) | ((q <= JOINT_POSITION_LOW) & (safe < 0.))
                 | ((np.abs(qd) >= JOINT_VELOCITY_LIMIT) & (safe*qd > 0.))] = 0.
            np.testing.assert_allclose(torque, safe, rtol=0, atol=1e-12)
            np.testing.assert_allclose(controller.last_result.wheel_speed_target_rad_s, d['wheel_target_rad_s'], rtol=0, atol=1e-12)
            np.testing.assert_allclose(controller.control_memory.wheel_integral_nm, d['wheel_integral_after_nm'], rtol=0, atol=1e-12)
            max_delta = max(max_delta, float(np.max(np.abs(record.delta_torque_nm-delta))))
            max_request = max(max_request, float(np.max(np.abs(torque-safe))))
            max_wheel = max(max_wheel, float(np.max(np.abs(torque[3::4]-np.asarray(d['requested_torque_nm'])[3::4]))))
            count += 1
            assert data.time == plant.data.time == 0.
    assert count == 104
    files = [Path(__file__), report_path, R/'scripts/d1_turn_leg_damping.py',
             R/'scripts/probe_d1_heading_turn_damping.py', R/'tests/test_d1_turn_leg_damping.py']
    result = {'passed': True, 'new_physics_steps': 0, 'independent_single_state_compute_calls': count,
        'max_delta_error_nm': max_delta, 'max_protected_request_error_nm': max_request,
        'max_original_wheel_request_difference_nm': max_wheel, 'trajectory_or_performance_prediction': False,
        'note': 'Each compute restarts from original saved PI memory. Initial scratch-model setup is nonintegrating; no claim of new closed-loop states or solved-contact qualification.',
        'input_sha256': {str(p): sha(p) for p in files}}
    with (W/'turn_damper_shadow_preflight.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    def forbidden(*args, **kwargs):
        raise AssertionError('candidate algebra preflight must not integrate')
    with patch.object(mujoco, 'mj_step', forbidden), patch.object(mujoco, 'mj_step1', forbidden), patch.object(mujoco, 'mj_step2', forbidden):
        main()
