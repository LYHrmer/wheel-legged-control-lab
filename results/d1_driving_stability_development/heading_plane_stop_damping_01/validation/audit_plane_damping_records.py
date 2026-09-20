"""Root independent saved-state/receipt audit; integration is forbidden."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from audit_flat_plane_records import expected_raw, near, same
from scripts.d1_flat_plane_env import D1FlatPlanePlant
from wheel_legged_control.d1.model import (
    JOINT_POSITION_HIGH, JOINT_POSITION_LOW, JOINT_TORQUE_LIMIT,
    JOINT_VELOCITY_LIMIT,
)


def rows(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected(torque, position, velocity):
    value = np.clip(torque, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
    value[((position >= JOINT_POSITION_HIGH) & (value > 0.))
          | ((position <= JOINT_POSITION_LOW) & (value < 0.))
          | ((np.abs(velocity) >= JOINT_VELOCITY_LIMIT) & (value*velocity > 0.))] = 0.
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists()
    protocol = json.loads((args.input/'protocol.json').read_text())
    aggregate = json.loads((args.input/'summary.json').read_text())
    for path, value in protocol['input_sha256'].items():
        assert digest(Path(path)) == value, path
    assert aggregate['comparison_valid'] and aggregate['candidate_passed_both_stop_cases']
    plant = D1FlatPlanePlant()
    model, data = plant.model, plant.data
    jac = np.zeros((3, model.nv))
    gain = 126.4374005337902
    reports, total = [], 0
    for case in protocol['cases']:
        folder = args.input/case['name']
        baseline = args.baseline/case['name']
        for location in (folder, baseline):
            manifest = json.loads((location/'complete_manifest.json').read_text())
            assert all(digest(location/p) == v['sha256'] for p, v in manifest.items())
        trace = rows(folder/'trace.jsonl.gz')
        diag = rows(folder/'damping_trace.jsonl.gz')
        powers = rows(folder/'damping_contact_power.jsonl.gz')
        native = rows(folder/'native_physics_entries.jsonl.gz')
        base_diag = rows(baseline/'plane_diagnostics.jsonl.gz')
        base_native = rows(baseline/'native_physics_entries.jsonl.gz')
        summary = json.loads((folder/'candidate_summary.json').read_text())
        n = len(trace)
        assert n == len(diag) == len(powers) == case['max_transitions']
        assert len(native) == 5*n
        with np.load(folder/'states.npz') as z:
            states = {k:z[k].copy() for k in z.files}
        prefix = case['command']['stop_tick'] or n
        with np.load(baseline/'states.npz') as z:
            for key in z.files:
                count = prefix if key.endswith('actions') else prefix+1
                assert same(z[key][:count], states[key][:count]), key
        # Compare parsed native entries as canonical JSON bytes, independently
        # of the runner's fixed-size SHA arrays. Includes force/cache metadata.
        for a,b in zip(base_native[:5*prefix], native[:5*prefix]):
            assert json.dumps(a,sort_keys=True,separators=(',',':')) == json.dumps(b,sort_keys=True,separators=(',',':'))
        for a,b in zip(base_diag[:prefix], diag[:prefix]):
            for key in ('requested_torque_nm','applied_torque_nm','wheel_integral_before_nm','wheel_integral_after_nm'):
                assert same(a[key], b[key]), key
            for key in ('raw_user_command','servo_command'):
                assert json.dumps(a[key],sort_keys=True) == json.dumps(b[key],sort_keys=True)
            assert same(a['unlimited_torque_nm'], b['damping']['total_requested_torque_nm'])
        with np.load(folder/'execution_states.npz') as z:
            assert len(z['qpos']) == n+2
            assert same(z['qpos'][:-1],states['qpos']) and same(z['qvel'][:-1],states['qvel'])
            assert same(z['qpos'][-1],z['qpos'][-2]) and same(z['qvel'][-1],z['qvel'][-2])
            near(z['time_s'][:-1],np.arange(n+1)*.01)
        assert all(len(states[k]) == n+1 for k in ('qpos','qvel','observations','truth_positions_world_m'))
        assert not np.any(states['requested_actions']) and not np.any(states['applied_actions'])
        max_delta_error = 0.
        max_protected_power = 0.
        errors = []
        for k,(t,d,p) in enumerate(zip(trace,diag,powers)):
            r = d['damping']
            assert t['tick'] == d['tick'] == p['tick'] == k
            assert t['endpoint_tick'] == p['endpoint_tick'] == k+1
            raw = expected_raw(case,k)
            assert raw == t['heading_task']['user_command_before'] == d['raw_user_command']
            assert r['forward_command_mps'] == raw['forward_velocity_mps'] == d['servo_command']['forward_velocity_mps']
            active = case['command']['stop_tick'] is not None and k >= case['command']['stop_tick']
            assert r['active'] == active and r['damping_nspm'] == gain
            data.qpos[:] = states['qpos'][k]
            data.qvel[:] = states['qvel'][k]
            mujoco.mj_kinematics(model,data)
            mujoco.mj_comPos(model,data)
            forward = data.xmat[plant.base_body_id].reshape(3,3)[:,0]
            qd = data.qvel[plant.dof_addresses]
            q = data.qpos[plant.qpos_addresses]
            delta, velocities = np.zeros(16), np.zeros(4)
            if active:
                for leg,body in enumerate(plant.wheel_body_ids_by_leg):
                    mujoco.mj_jacBody(model,data,jac,None,int(body))
                    columns = plant.dof_addresses[4*leg:4*leg+3]
                    jx = forward @ jac[:,columns]
                    velocities[leg] = jx @ qd[4*leg:4*leg+3]
                    delta[4*leg:4*leg+3] = -gain*jx*velocities[leg]
            near(r['delta_torque_nm'],delta)
            near(r['leg_relative_forward_mps'],velocities)
            near(r['joint_power_w'],delta@qd)
            near(r['joint_power_w'],-gain*(velocities@velocities))
            assert r['joint_power_w'] <= 1e-12
            base = np.asarray(r['base_requested_torque_nm'])
            near(r['total_requested_torque_nm'],base+delta)
            safe, base_safe = protected(base+delta,q,qd), protected(base,q,qd)
            near(r['safe_torque_nm'],safe)
            near(d['requested_torque_nm'],safe)
            near(p['base_safe_torque_nm_same_state'],base_safe)
            near(p['joint_velocity_before_rad_s'],qd)
            near(p['protected_increment_joint_power_w'],(safe-base_safe)@qd)
            assert p['endpoint_contacts']['sampling'] == 'synchronized_endpoint_not_control_average'
            max_delta_error = max(max_delta_error,float(np.max(np.abs(delta-r['delta_torque_nm']))))
            max_protected_power = max(max_protected_power,p['protected_increment_joint_power_w'])
            errors.append(t['body_forward_mps']-raw['forward_velocity_mps'])
            near(t['user_forward_error_mps'],errors[-1])
            for offset,e in enumerate(native[k*5:(k+1)*5]):
                near(e['start_time_s'],(k*5+offset)*.002)
                near(e['actual_dt_s'],.002)
                assert e['returned']
                near(e['ctrl_nm'],d['applied_torque_nm'][offset],0)
                pulse = case['external_wrench']
                expected = np.zeros(6)
                if pulse and pulse['start_tick'] <= k < pulse['end_tick_exclusive']:
                    expected = np.asarray(pulse['force_xyz_n']+pulse['torque_xyz_nm'])
                near(e['wrench_world'],expected,0)
                c = e['contacts']
                assert c['sampling'] == 'native_step_solved_cache_not_synchronized_endpoint'
                assert c['max_horizontal_normal'] <= 1e-12 and c['max_vertical_normal_error'] <= 1e-12
                force,moment = np.zeros(3),np.zeros(3)
                for contact in c['contacts']:
                    f = np.asarray(contact['force_world_n'])
                    normal = np.asarray(contact['normal_terrain_to_robot_world'])
                    near(contact['normal_force_world_n'],(f@normal)*normal)
                    near(f,np.asarray(contact['normal_force_world_n'])+contact['tangent_force_world_n'])
                    force += f
                    moment += np.cross(np.asarray(contact['pos_world_m'])-c['reference_world_m'],f)+contact['torque_world_nm']
                near(c['total_wrench_world_6'],np.r_[force,moment])
        rms = float(np.sqrt(np.mean(np.square(errors))))
        near(rms,summary['velocity_rmse_mps'])
        impulse = sum(e['wrench_world'][5]*e['actual_dt_s'] for e in native)
        near(impulse,summary['actual_signed_yaw_impulse_nms'])
        report = {'case':case['name'],'transitions':n,'physics_substeps':5*n,
            'original_gates_passed':summary['gates']['passed'],'velocity_rmse_mps':rms,
            'native_and_control_prefix_bitwise_passed':True,'compared_transitions':prefix,
            'compared_states':prefix+1,'independent_delta_max_abs_error_nm':max_delta_error,
            'maximum_sampled_protected_increment_power_w':max_protected_power,'actual_signed_yaw_impulse_nms':impulse}
        if case['command']['stop_tick'] is not None:
            peak = max(abs(t['body_forward_mps']) for t in trace if 500 <= t['endpoint_tick'] < 800)
            path = float(np.linalg.norm(np.diff(states['truth_positions_world_m'][500:701,:2],axis=0),axis=1).sum())
            near(peak,summary['late_stop_max_abs_body_vx_mps'])
            near(path,summary['late_stop_cumulative_planar_path_m'])
            assert peak <= .03 and path <= .05 and rms <= .05
            report.update(late_stop_peak_mps=peak,late_stop_path_m=path)
        assert all(summary['gates']['checks'].values())
        reports.append(report)
        total += n
    assert total == aggregate['actual_new_control_transitions'] == 4000
    assert 5*total == aggregate['actual_new_physics_substeps'] == 20000
    assert plant.data.time == 0.
    root = Path('/home/lyh/wheel-legged-control-lab')
    frozen = json.loads((root/'results/d1_budget_study/protocol.json').read_text())['source_sha256']
    assert len(frozen)==77 and all(digest(root/p)==v for p,v in frozen.items())
    result = {'passed':True,'plant_semantics_valid':True,'execution_evidence_valid':True,
        'pairing_passed':True,'all_four_original_case_gates_passed':True,'new_integration_steps':0,
        'saved_state_kinematic_evaluations':total,'frozen77_unchanged':True,'episodes':reports,
        'candidate_control_transitions':total,'candidate_physics_substeps':total*5,
        'reused_baseline_control_transitions':4000,'reused_baseline_physics_substeps':20000,
        'input_sha256':{str(p):digest(p) for p in (Path(__file__),Path(__file__).with_name('audit_flat_plane_records.py'),args.input/'protocol.json',args.input/'manifest.json')},
        'limitation':'finite development plane set only; sampled algebraic power is not integrated passivity or full driving qualification'}
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    def forbidden(*args,**kwargs):
        raise AssertionError('offline audit must not integrate')
    with patch.object(mujoco,'mj_step',forbidden), patch.object(mujoco,'mj_step1',forbidden), patch.object(mujoco,'mj_step2',forbidden):
        main()
