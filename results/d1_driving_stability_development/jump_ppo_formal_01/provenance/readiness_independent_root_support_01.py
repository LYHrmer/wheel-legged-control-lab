"""Bounded saved-record audit. Never integrates or reconstructs contact forces.

Run from the repository with PYTHONPATH=.local-deps:src:. and one BLAS thread.
Only the requested new JSON output is written, exclusively (no overwrites).
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import mujoco

FORBIDDEN_CALLS = []
FORBIDDEN_NAMES = ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse",
                   "mj_collision", "mj_contactForce", "mj_resetData", "mj_resetDataKeyframe")


def forbidden(name):
    def reject(*args, **kwargs):
        FORBIDDEN_CALLS.append(name)
        raise RuntimeError(f"saved-record audit forbids {name}")
    return reject


for _name in FORBIDDEN_NAMES:
    if hasattr(mujoco, _name):
        setattr(mujoco, _name, forbidden(_name))
for _name in dir(mujoco):
    if _name.startswith(("mj_fwd", "mj_inv")):
        setattr(mujoco, _name, forbidden(_name))

from wheel_legged_control.d1.model import (
    D1_JOINT_NAMES, JOINT_POSITION_HIGH, JOINT_POSITION_LOW,
    JOINT_TORQUE_LIMIT, JOINT_VELOCITY_LIMIT, build_d1_model,
)

ROOT = Path('/home/lyh/wheel-legged-control-lab')
WORK = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920')
SOURCE = WORK / 'jump_readiness_01'
CASES = ('stationary_height_hold', 'stationary_height_profile')
API_COUNTS = {'model_compile': 0, 'MjData': 0, 'mj_kinematics': 0,
              'mj_comPos': 0, 'mj_jacBodyCom': 0, 'mj_name2id': 0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, 'rt') as handle:
        return [json.loads(line) for line in handle]


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def maximum_error(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=float)-np.asarray(b, dtype=float))))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def expected_height(case, tick):
    if case.endswith('hold') or tick < 200 or tick >= 320:
        return .455
    return .405 if tick < 225 else .500 if tick < 240 else .455 if tick < 275 else .435


def name_id(model, kind, name):
    API_COUNTS['mj_name2id'] += 1
    return int(mujoco.mj_name2id(model, kind, name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = read_json(SOURCE/'manifest.json')
    hashes = {str(SOURCE/k): sha(SOURCE/k) for k in manifest['files']}
    assert all(hashes[str(SOURCE/k)] == v['sha256'] for k, v in manifest['files'].items())
    hashes[str(SOURCE/'manifest.json')] = sha(SOURCE/'manifest.json')
    preflight = read_json(SOURCE/'preflight.json')
    dependency_paths = [ROOT/'src/wheel_legged_control/d1/model.py',
                        ROOT/'src/wheel_legged_control/d1/terrain.py',
                        ROOT/'src/wheel_legged_control/d1/assets/urdf/robot.urdf']
    for path in dependency_paths:
        hashes[str(path)] = sha(path)
        expected = preflight['input_sha256'].get(str(path))
        if expected is not None:
            assert hashes[str(path)] == expected, str(path)

    model = build_d1_model(timestep=.002, ground_friction=.9, arena='flat')
    API_COUNTS['model_compile'] += 1
    data = mujoco.MjData(model)
    API_COUNTS['MjData'] += 1
    bodies = np.arange(1, model.nbody)
    mass = model.body_mass[bodies].copy()
    base = name_id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    floor = name_id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    joint_ids = [name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in D1_JOINT_NAMES]
    qa, va = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    binding = read_json(SOURCE/CASES[0]/'receipt.json')['binding']
    wheels = np.asarray(binding['wheel_geom_ids'], dtype=int)
    assert model.nhfield == 0 and model.geom_type[floor] == 0
    assert model.ngeom == binding['model_ngeom'] and floor == binding['plane_geom_id']
    assert np.all(model.geom_type[wheels] == 5)
    assert same(model.geom_bodyid[wheels].astype(np.int64), np.asarray(binding['wheel_body_ids'], dtype=np.int64))
    assert np.array_equal(model.geom_size[wheels, :2], np.array([binding['radii_m'], binding['half_lengths_m']]).T)
    assert maximum_error(model.geom_pos[wheels], binding['wheel_local_positions']) == 0
    assert maximum_error(model.geom_quat[wheels], binding['wheel_local_quaternions']) == 0
    assert np.all(model.geom_margin[wheels] == .001) and model.geom_margin[floor] == 0
    assert np.all(model.body_parentid[bodies] < bodies) and base == 1
    assert np.isclose(mass.sum(), model.body_subtreemass[base], rtol=0, atol=1e-12)

    def reconstruct(qpos, qvel):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_kinematics(model, data)
        API_COUNTS['mj_kinematics'] += 1
        mujoco.mj_comPos(model, data)
        API_COUNTS['mj_comPos'] += 1
        rotations = data.geom_xmat.reshape(model.ngeom, 3, 3)
        normal = rotations[floor, :, 2]
        axial = rotations[wheels, :, 2] @ normal
        support = model.geom_size[wheels, 0]*np.sqrt(np.maximum(0., 1.-axial**2))
        support += model.geom_size[wheels, 1]*np.abs(axial)
        gaps = data.geom_xpos[wheels] @ normal - data.geom_xpos[floor] @ normal - support
        position = np.sum(mass[:, None]*data.xipos[bodies], axis=0)/mass.sum()
        velocity = np.zeros(3)
        base_velocity = None
        for body, weight in zip(bodies, mass, strict=True):
            jac = np.zeros((3, model.nv))
            mujoco.mj_jacBodyCom(model, data, jac, None, int(body))
            API_COUNTS['mj_jacBodyCom'] += 1
            body_velocity = jac @ data.qvel
            velocity += weight*body_velocity
            if body == base:
                base_velocity = data.xmat[base].reshape(3, 3).T @ body_velocity
        return gaps.copy(), position, velocity/mass.sum(), float(base_velocity[0])

    results, saved = {}, {}
    for case in CASES:
        folder = SOURCE/case
        trace, native = rows(folder/'trace.jsonl.gz'), rows(folder/'native.jsonl.gz')
        record, receipt = read_json(folder/'records.json'), read_json(folder/'receipt.json')
        with np.load(folder/'states.npz', allow_pickle=False) as z:
            states = {key: z[key].copy() for key in z.files}
        with np.load(folder/'execution_states.npz', allow_pickle=False) as z:
            execution = {key: z[key].copy() for key in z.files}
        assert len(trace) == 600 and len(native) == 3000 and len(record['endpoints']) == 601
        assert all(len(value) == 601 for value in states.values())
        assert len(execution['tag']) == 602 and execution['tag'][0] == 'reset'
        assert all(tag == 'completed_step' for tag in execution['tag'][1:601])
        assert execution['tag'][-1] == 'close'
        assert same(execution['qpos'][:601], states['qpos']) and same(execution['qvel'][:601], states['qvel'])
        assert same(execution['qpos'][-1], states['qpos'][-1]) and same(execution['qvel'][-1], states['qvel'][-1])
        assert execution['time_s'][0] == 0. and execution['time_s'][-1] == execution['time_s'][-2]
        assert receipt['height_callback_count'] == 601 and receipt['native_receipt']['foreign_pass_through_calls'] == 0
        time_error = 0.
        max_torque = np.zeros(16)
        for i, row in enumerate(native):
            assert row['index'] == i and row['returned'] is True and row['error'] is None
            assert row['contact_sample_tag'] == 'native_step_solved_cache_not_synchronized_endpoint'
            time_error = max(time_error, abs(row['start_time_s']-.002*i), abs(row['end_time_s']-.002*(i+1)))
            assert abs(row['actual_dt_s']-(row['end_time_s']-row['start_time_s'])) < 1e-14
            assert not np.any(row['xfrc_applied']) and not np.any(row['qfrc_applied'])
            ctrl = np.asarray(row['ctrl_nm'])
            assert np.all(np.abs(ctrl) <= JOINT_TORQUE_LIMIT+1e-12)
            assert same(ctrl, trace[i//5]['controller']['torque_nm'])
            max_torque = np.maximum(max_torque, np.abs(ctrl))
            assert all(np.isfinite(row[key]).all() for key in ('qpos_before', 'qvel_before', 'qpos_returned', 'qvel_returned', 'ctrl_nm'))
            if i:
                assert same(row['qpos_before'], native[i-1]['qpos_returned'])
                assert same(row['qvel_before'], native[i-1]['qvel_returned'])
            if i % 5 == 0:
                assert same(row['qpos_before'], states['qpos'][i//5])
                assert same(row['qvel_before'], states['qvel'][i//5])
            if i % 5 == 4:
                assert same(row['qpos_returned'], states['qpos'][i//5+1])
                assert same(row['qvel_returned'], states['qvel'][i//5+1])
            detail = row['contacts']['detail']
            assert not detail['invalid_load'] and not detail['invalid_normal_values']
            assert maximum_error(detail['active_contact_count'], record['intervals'][i]['active_wheel_contacts']) == 0
            assert maximum_error(detail['wheel_normal_load_n'], record['intervals'][i]['wheel_normal_load_n']) == 0
        assert time_error < 1e-10
        protection_error = 0.
        for tick, row in enumerate(trace):
            assert row['tick'] == tick and not np.any(row['action'])
            for command in (row['raw_command'], row['info']['heading_task']['user_command_before']):
                assert command == {'forward_velocity_mps': 0., 'yaw_rate_rps': 0., 'clearance_m': expected_height(case, tick)}
            assert row['servo_command']['clearance_m'] == expected_height(case, tick)
            assert row['world_command']['base_height_m'] == expected_height(case, tick)
            assert not row['readiness']['stop_latch_active'] and not row['readiness']['authority_gate_active']
            assert row['terminated'] is False and row['truncated'] is (tick == 599)
            c = row['controller']
            request = np.asarray(c['requested_torque_nm'])
            safe = np.clip(request, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
            q, v = states['qpos'][tick, qa], states['qvel'][tick, va]
            blocked = ((q >= JOINT_POSITION_HIGH) & (safe > 0.)) | ((q <= JOINT_POSITION_LOW) & (safe < 0.)) | ((np.abs(v) >= JOINT_VELOCITY_LIMIT) & (safe*v > 0.))
            safe[blocked] = 0.
            protection_error = max(protection_error, maximum_error(safe, c['torque_nm']))
        assert protection_error <= 1e-12

        control_errors = dict(wheel_gap_m=0., com_position_m=0., com_velocity_mps=0., body_vx_mps=0.)
        rebuilt_controls = []
        for tick, endpoint in enumerate(record['endpoints']):
            assert endpoint['tick'] == tick and abs(endpoint['time_s']-states['time'][tick]) < 1e-14
            gaps, com, velocity, body_vx = reconstruct(states['qpos'][tick], states['qvel'][tick])
            for key, rebuilt in [('wheel_gap_m', gaps), ('com_position_m', com), ('com_velocity_mps', velocity), ('body_vx_mps', body_vx)]:
                control_errors[key] = max(control_errors[key], maximum_error(rebuilt, endpoint[key]))
            rebuilt_controls.append((gaps, com, velocity, body_vx))
        assert max(control_errors.values()) <= 1e-12
        selected = []
        selected_errors = dict(wheel_gap_m=0., returned_com_velocity_mps=0., start_com_vz_mps=0.)
        for i in range(1200, 1212):
            row, stored = native[i], record['intervals'][i]
            gaps, com, velocity, _ = reconstruct(row['qpos_returned'], row['qvel_returned'])
            _, _, start_velocity, _ = reconstruct(row['qpos_before'], row['qvel_before'])
            selected_errors['wheel_gap_m'] = max(selected_errors['wheel_gap_m'], maximum_error(gaps, stored['endpoint_wheel_gap_m']))
            selected_errors['returned_com_velocity_mps'] = max(selected_errors['returned_com_velocity_mps'], maximum_error(velocity, stored['returned_com_velocity_mps']))
            selected_errors['start_com_vz_mps'] = max(selected_errors['start_com_vz_mps'], abs(start_velocity[2]-stored['start_com_vz_mps']))
            selected.append({'native_index': i, 'time_s': row['end_time_s'], 'wheel_gap_m': gaps.tolist(), 'minimum_gap_m': float(gaps.min()), 'com_position_m': com.tolist(), 'returned_com_velocity_mps': velocity.tolist(), 'start_com_vz_mps': float(start_velocity[2]), 'active_contacts': stored['active_wheel_contacts'], 'normal_load_n': stored['wheel_normal_load_n']})
        assert max(selected_errors.values()) <= 1e-12
        intervals = record['intervals']
        mask = [i >= 1000 and not any(n['contacts']['detail']['active_contact_count']) and not any(n['contacts']['detail']['wheel_normal_load_n']) and float(intervals[i]['endpoint_min_gap_m']) > .001 for i, n in enumerate(native)]
        runs = []
        i = 0
        while i < len(mask):
            if not mask[i]:
                i += 1
                continue
            end = i+1
            while end < len(mask) and mask[end]:
                end += 1
            runs.append({'first': i, 'end_exclusive': end, 'native_intervals': end-i, 'duration_s': sum(native[k]['end_time_s']-native[k]['start_time_s'] for k in range(i,end)), 'start_com_vz_mps': intervals[i]['start_com_vz_mps']})
            i = end
        native_best = max(range(1000, 3000), key=lambda i: intervals[i]['endpoint_min_gap_m'])
        late = rebuilt_controls[400:600]
        late_report = {'height_rmse_m': float(np.sqrt(np.mean((states['qpos'][400:600,2]-.455)**2))), 'max_abs_body_vx_mps': max(abs(x[3]) for x in late), 'max_abs_com_vz_mps': max(abs(x[2][2]) for x in late), 'planar_path_m': float(np.linalg.norm(np.diff(states['qpos'][400:601,:2],axis=0),axis=1).sum()), 'per_wheel_positive_load_fraction': (np.asarray([n['contacts']['detail']['wheel_normal_load_n'] for n in native[-1000:]])>0).mean(axis=0).tolist()}
        late_report['passed'] = bool(late_report['height_rmse_m'] <= .015 and late_report['max_abs_body_vx_mps'] <= .03 and late_report['max_abs_com_vz_mps'] <= .03 and late_report['planar_path_m'] <= .05 and min(late_report['per_wheel_positive_load_fraction']) >= .95)
        results[case] = {'counts': {'control': 600, 'state': 601, 'native': 3000, 'execution_archive': 602}, 'execution_extra_row': 'close: exact final pose/velocity/time duplicate, not integration', 'max_native_clock_error_s': time_error, 'zero_external_wrench_every_native': True, 'native_control_hold_exact': True, 'native_state_chain_and_T_plus_1_exact': True, 'schedule_all600_exact': True, 'max_abs_torque_by_joint_nm': max_torque.tolist(), 'same_state_original_protection_max_error_nm': protection_error, 'control_geometry_COM_errors': control_errors, 'selected_native_errors': selected_errors, 'selected_native_1200_to_1211': selected, 'saved_native_gap_maximum': {'index': native_best, 'raw_gap_m': intervals[native_best]['endpoint_min_gap_m'], 'net_above_margin_m': intervals[native_best]['endpoint_min_gap_m']-.001, 'independently_rebuilt': 1200 <= native_best < 1212}, 'recorded_unloaded_gap_runs_after_request': runs, 'late_settled_recomputed': late_report}
        saved[case] = dict(states=states, trace=trace, native=native)

    hold, profile = [saved[c] for c in CASES]
    prefix = {key: same(hold['states'][key][:(200 if key=='observation' else 201)], profile['states'][key][:(200 if key=='observation' else 201)]) for key in hold['states']}
    def stripped_trace(row):
        value = copy.deepcopy(row)
        value['readiness'].pop('condition')
        value['readiness'].pop('display_phase')
        return canonical(value)
    prefix['execution_0_to_199'] = all(stripped_trace(a)==stripped_trace(b) for a,b in zip(hold['trace'][:200],profile['trace'][:200],strict=True))
    prefix['native_0_to_999'] = all(canonical(a)==canonical(b) for a,b in zip(hold['native'][:1000],profile['native'][:1000],strict=True))
    assert all(prefix.values())
    assert not FORBIDDEN_CALLS
    assert all(sha(path) == value for path,value in hashes.items())
    report = {'schema': 'd1-readiness-independent-bounded-audit-v1', 'passed': True, 'scope': 'Full raw trace/native counting, schedule, input-wrench, held/protected-torque, state-chain, archive hash and common-prefix checks for both episodes. Independent geometry/whole-COM reconstruction: all 601 control endpoints per episode plus start/returned states for native indices 1200..1211 per episode. Other native geometry is read from prior records, not independently reconstructed.', 'input_sha256': hashes, 'helper_sha256': sha(__file__), 'cases': results, 'prefix': prefix, 'phase_semantics': 'Contacts/forces are saved native_step_solved_cache_not_synchronized_endpoint evaluations; geometric gaps and COM are from the separately reconstructed returned qpos/qvel. No contact force was regenerated or paired with endpoint velocity as work.', 'chronology': 'Observer source encloses reset and all 600 steps; both native archives start at reset time 0, have exactly 3000 continuous calls and match every T+1 state. Close snapshots duplicate the final state. No excess integrations are evidenced inside this observed window. Saved archives cannot independently exclude uninstrumented work outside that window.', 'api_counts': API_COUNTS, 'forbidden_calls': FORBIDDEN_CALLS, 'new_physics_steps': 0, 'new_controller_calls': 0, 'limitations': ['This is not a fresh rollout or a full independent reconstruction of all 6000 native endpoints.', 'Force/load validity relies on saved solved-cache values, their phase tags and input/source provenance; no force solve or collision recomputation performed.', 'Recorded intervals/endpoints do not prove unsampled continuous-time flight.'], 'conclusion': 'Within this bounded independent audit, fixed budget, commands, native phase, zero external wrench, original torque protection, common prefix and late settling are consistent. Profile saved peak/run remain far below the fixed 21 mm / 20 ms readiness thresholds.'}
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(args.output), 'sha256': sha(args.output), 'api_counts': API_COUNTS, 'new_physics_steps': 0, 'cases': {c: {'errors': results[c]['control_geometry_COM_errors'], 'native_errors': results[c]['selected_native_errors'], 'peak': results[c]['saved_native_gap_maximum'], 'runs': results[c]['recorded_unloaded_gap_runs_after_request'], 'late': results[c]['late_settled_recomputed']} for c in CASES}}, sort_keys=True))


if __name__ == '__main__':
    main()
