"""Audit recorded manual GUI input without advancing or changing its simulation."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import tarfile

import mujoco
import numpy as np

WORK = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912')
RAW = WORK/'manual_side_drive_01'
OUT = WORK/'manual_side_drive_01_audit'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def runs(mask):
    mask = np.asarray(mask, dtype=bool)
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(int(a), int(b)) for a, b in zip(boundaries[::2], boundaries[1::2])]


def main():
    assert not OUT.exists()
    protocol = json.loads((RAW/'protocol.json').read_text())
    summary = json.loads((RAW/'summary.json').read_text())
    manifest = json.loads((RAW/'manifest.json').read_text())
    assert all(sha(RAW/name) == digest for name, digest in manifest.items())
    with tarfile.open(RAW/'source.tar.gz') as archive:
        assert all(hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
                   for name, digest in protocol['source_sha256'].items())
    rows = list(csv.DictReader((RAW/'telemetry.csv').open()))
    events = [json.loads(line) for line in (RAW/'keyboard_events.jsonl').read_text().splitlines()]
    with np.load(RAW/'states.npz', allow_pickle=False) as archive:
        states = {name: archive[name] for name in archive.files}
    count = len(rows)
    qpos, qvel = states['qpos'], states['qvel']
    assert count == summary['steps'] and len(qpos) == count + 1
    assert summary['segments'] == 1 and set(states['segment_id']) == {0}
    assert summary['source_unchanged'] and summary['stop_reason'] == 'viewer_closed'
    assert states['applied_torque_nm'].shape == (count, 16)
    assert all(np.isfinite(a).all() for a in (qpos, qvel, states['applied_torque_nm']))
    for i, row in enumerate(rows):
        assert int(row['tick']) == int(row['state_after_index']) == i + 1
        assert int(row['state_before_index']) == i
    np.testing.assert_allclose(np.diff(states['segment_time_s']), .01, atol=1e-10, rtol=0)
    assert sha(RAW/'model.mjb') == protocol['compiled_model_sha256']
    model = mujoco.MjModel.from_binary_path(str(RAW/'model.mjb'))
    data = mujoco.MjData(model)
    legs = ('FL', 'FR', 'RL', 'RR')
    foot_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name+'_foot') for name in legs]
    foot_joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name+'_foot_joint') for name in legs]
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or '' for i in range(model.ngeom)]
    terrain = {i for i, name in enumerate(names) if name.startswith(('floor', 'terrain_'))}
    # This model is an analysis-only reload. Groups are changed solely so vertical
    # ray queries ignore the robot; groups do not change collision constraints.
    model.geom_group[:] = 5
    model.geom_group[list(terrain)] = 0
    groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    hit = np.zeros(1, dtype=np.int32)
    velocities, angular, rpy, lowest, ground, contact_count = [], [], [], [], [], []
    ground_contacts, nonwheel_contacts, wheel_contacts = [], [], []
    for state_index in range(count+1):
        data.qpos[:] = qpos[state_index]
        data.qvel[:] = qvel[state_index]
        mujoco.mj_forward(model, data)
        rotation = data.xmat[base].reshape(3, 3)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base, velocity, 0)
        origin = velocity[3:] - np.cross(velocity[:3], rotation@model.body_ipos[base])
        velocities.append(rotation.T@origin)
        angular.append(float(np.linalg.norm(velocity[:3])))
        rpy.append([np.arctan2(rotation[2, 1], rotation[2, 2]),
                    np.arcsin(-np.clip(rotation[2, 0], -1, 1)),
                    np.arctan2(rotation[1, 0], rotation[0, 0])])
        az = np.clip(data.xaxis[foot_joints, 2], -1, 1)
        feet = data.xpos[foot_bodies]
        lowest.append(feet[:, 2] - .087*np.sqrt(1-az*az) - .020*np.abs(az))
        heights = []
        for point in feet:
            start = point + [0., 0., 2.]
            distance = mujoco.mj_ray(model, data, start, np.array([0., 0., -1.]), groups, 1, -1, hit)
            assert distance >= 0
            heights.append(float(start[2]-distance))
        ground.append(heights)
        touching, terrain_names, body_names = set(), set(), set()
        for c in data.contact:
            a, b = int(c.geom1), int(c.geom2)
            ground_geom = a if a in terrain else b if b in terrain else None
            if ground_geom is None or c.efc_address < 0:
                continue
            other = b if a in terrain else a
            body = int(model.geom_bodyid[other])
            terrain_names.add(names[ground_geom])
            if body in foot_bodies:
                touching.add(foot_bodies.index(body))
            else:
                body_names.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body))
        contact_count.append(len(touching))
        wheel_contacts.append(sorted(touching))
        ground_contacts.append(sorted(terrain_names))
        nonwheel_contacts.append(sorted(body_names))
    velocities, angular, rpy = np.array(velocities), np.array(angular), np.array(rpy)
    lowest, ground = np.array(lowest), np.array(ground)
    clearance = lowest-ground
    contact_count = np.array(contact_count)
    fallen = (qpos[:, 2] < .22) | (np.abs(rpy[:, :2]).max(axis=1) > .85)
    requested = np.array([float(row['requested_forward_mps']) for row in rows])
    ready = np.array([row['jump_phase'] == 'ready' for row in rows])
    slow = (requested > .25) & (np.abs(velocities[1:, 0]) < .03) & ready
    stalls = []
    for a, b in runs(slow):
        if b-a < 50:
            continue
        stalls.append({'first_tick': a+1, 'last_tick': b, 'start_s': a*.01,
                       'duration_s': (b-a)*.01, 'position_start_m': qpos[a, :3].tolist(),
                       'delta_xy_m': (qpos[b, :2]-qpos[a, :2]).tolist(),
                       'mean_forward_mps': float(velocities[a+1:b+1, 0].mean()),
                       'terrain_contact_state_counts': dict(Counter(name for item in ground_contacts[a+1:b+1] for name in item))})
    jump_runs = runs([row['jump_phase'] != 'ready' for row in rows])
    jumps = []
    for a, b in jump_runs:
        selected = slice(a+1, b+1)
        airborne = contact_count[selected] == 0
        strict_clearance = clearance[selected].min(axis=1)
        peak = a+1+int(np.argmax(strict_clearance))
        jumps.append({'number': len(jumps)+1, 'start_s': a*.01, 'end_s': b*.01,
                      'phases': sorted({row['jump_phase'] for row in rows[a:b]}),
                      'completed_counter_after': int(rows[b]['completed_jumps']) if b<count else None,
                      'initial_base_z_m': float(qpos[a, 2]), 'peak_base_z_m': float(qpos[selected, 2].max()),
                      'base_z_gain_m': float(qpos[selected, 2].max()-qpos[a, 2]),
                      'all_wheels_no_contact_s': float(np.count_nonzero(airborne)*.01),
                      'peak_min_wheel_clearance_m': float(strict_clearance.max()),
                      'wheel_clearance_at_peak_m': clearance[peak].tolist(),
                      'peak_clearance_state_index': peak,
                      'terrain_contact_state_counts': dict(Counter(name for item in ground_contacts[selected] for name in item))})
    side = []
    for i, row in enumerate(rows):
        if row['side_success'] != 'True' or row['applied_controller'] != 'd1_fast_side_step':
            continue
        if i and rows[i-1]['side_success'] == 'True':
            continue
        entry = next(j for j in range(i, -1, -1) if j == 0 or rows[j-1]['applied_controller'] != 'd1_fast_side_step')
        heading = rpy[entry, 2]
        delta = qpos[i+1, :2]-qpos[entry, :2]
        side.append({'entry_tick': entry+1, 'done_tick': i+1, 'duration_s': (i-entry+1)*.01,
                     'lateral_m': float(delta@[-np.sin(heading), np.cos(heading)]),
                     'forward_m': float(delta@[np.cos(heading), np.sin(heading)]),
                     'yaw_delta_rad': float(np.arctan2(np.sin(rpy[i+1, 2]-heading), np.cos(rpy[i+1, 2]-heading))),
                     'done_wheel_contacts': int(contact_count[i+1]),
                     'done_speed_mps': float(np.linalg.norm(velocities[i+1]))})
    presses = [event for event in events if event.get('action') == 1]
    key_intervals = []
    for press in presses:
        if press['key'] not in (87, 83, 68, 81, 69):
            continue
        release = next((e for e in events if e.get('key') == press['key'] and e.get('action') == 0
                        and e['wall_time_s'] > press['wall_time_s']), None)
        end = release['simulation_time_s'] if release else summary['integrated_duration_s']
        a, b = round(press['simulation_time_s']*100), round(end*100)
        key_intervals.append({'key': chr(press['key']), 'start_s': press['simulation_time_s'], 'end_s': end,
                              'wall_hold_s': release['wall_time_s']-press['wall_time_s'] if release else None,
                              'position_start_m': qpos[a, :3].tolist(), 'position_end_m': qpos[b, :3].tolist(),
                              'mean_actual_forward_mps': float(velocities[a+1:b+1, 0].mean()),
                              'peak_requested_forward_mps': float(requested[a:b].max()),
                              'side_controller_ticks': sum(row['applied_controller'] == 'd1_fast_side_step' for row in rows[a:b])})
    second_d = slice(6825, 6955)
    gate_data = {'reconstruction_only': 'Integrated before-step states, not the legacy_mixed final-substep cache; gate reasons below are diagnostic evidence, not a replay of exact runtime internals.',
                 'first_before_state': 6825, 'last_before_state': 6954,
                 'four_contact_states': int(np.count_nonzero(contact_count[second_d] == 4)),
                 'linear_speed_over_004_states': int(np.count_nonzero(np.linalg.norm(velocities[second_d], axis=1) >= .04)),
                 'angular_speed_over_008_states': int(np.count_nonzero(angular[second_d] >= .08)),
                 'absolute_world_wheel_bottom_over_001_states': int(np.count_nonzero(np.max(np.abs(lowest[second_d]), axis=1) > .01)),
                 'attitude_over_012_states': int(np.count_nonzero(np.max(np.abs(rpy[second_d, :2]), axis=1) > .12)),
                 'speed_minmax_mps': [float(np.linalg.norm(velocities[second_d], axis=1).min()),float(np.linalg.norm(velocities[second_d], axis=1).max())],
                 'terrain_contact_state_counts': dict(Counter(name for item in ground_contacts[second_d] for name in item))}
    result = {'status': 'audit_complete', 'human_acceptance': 'failed_user_feedback',
              'user_feedback': '整体运行慢；W 不能直接过台阶；跳跃太矮；要求 Shift 加速。原始 summary 未修改。',
              'summary': summary, 'render_quality': protocol['render_quality'],
              'integrity': {'manifest_files': len(manifest), 'archived_source_files': len(protocol['source_sha256']),
                            'finite_states_and_torques': True, 'single_segment_contiguous_001s': True,
                            'input_sha256': manifest},
              'source': 'Root launched the normal-quality desktop runner without the automated key harness. Keyboard events are GLFW observations; event records alone cannot identify the physical person or distinguish all possible external event sources.',
              'key_press_counts': dict(Counter(chr(e['key']) if 32 <= e['key'] <= 90 else str(e['key']) for e in presses)),
              'focus_loss_events': [e for e in events if e['type'] == 'focus' and not e['focused']],
              'key_intervals': key_intervals, 'successful_side_cycles': side,
              'side_cancel': {'start_tick': 3073, 'done_tick': 3118, 'handoff_tick': 3119,
                              'control_intervals': 46, 'contacts_at_done': int(contact_count[3118]),
                              'speed_at_done_mps': float(np.linalg.norm(velocities[3118])),
                              'cause': 'D release observed at 30.72 s; no focus loss or X/R event.',
                              'stale_cancelled_rows_after_handoff': sum(row['side_failure'] == 'cancelled' for row in rows[3118:])},
              'second_d_request': gate_data, 'jumps': jumps, 'stalls': stalls,
              'physical_summary': {'min_base_z_m': float(qpos[:,2].min()), 'max_base_z_m': float(qpos[:,2].max()),
                                   'max_abs_roll_pitch_rad': np.max(np.abs(rpy[:,:2]), axis=0).tolist(),
                                   'fallen_state_count': int(np.count_nonzero(fallen)),
                                   'nonwheel_contact_state_count': sum(bool(item) for item in nonwheel_contacts),
                                   'nonwheel_contact_body_counts': dict(Counter(name for item in nonwheel_contacts for name in item)),
                                   'terrain_contact_state_counts': dict(Counter(name for item in ground_contacts for name in item)),
                                   'actual_forward_minmax_mps': [float(velocities[:,0].min()),float(velocities[:,0].max())],
                                   'full_forward_ready_mean_mps': float(velocities[1:,0][(requested>.25)&ready].mean()),
                                   'full_forward_ready_stalled_fraction': float(np.count_nonzero(slow)/np.count_nonzero((requested>.25)&ready))},
              'methods': ['All geometry uses a separate reloaded MjModel/MjData with mj_forward; no mj_step or original file changes.',
                          'Wheel lowest point accounts for cylinder camber using radius 0.087 m and half-width 0.020 m.',
                          'Ground elevation is a vertical downward terrain-only ray at each wheel center; clearance is lowest point minus that elevation. Near a step edge this is local vertical clearance, not swept-volume obstacle clearance.',
                          'All-wheels-no-contact uses active wheel/terrain geometric contacts at recorded control states, not a flight label.',
                          'Stall means requested forward >0.25 m/s, jump ready, and actual body-frame forward speed magnitude <0.03 m/s for at least 0.5 s.',
                          'Fall criterion matches existing D1StateEstimate: world z<0.22 m or |roll|/|pitch|>0.85 rad.'],
              'limitations': 'One manual session with explicit negative user feedback. A/R, other spawn zones, physical self-righting and higher-speed/height candidates are not validated here.'}
    heading = rpy[1847, 2]
    delta = qpos[3118, :2]-qpos[1847, :2]
    result['side_cancel'].update(
        retained_lateral_from_first_cycle_entry_m=float(delta@[-np.sin(heading), np.cos(heading)]),
        retained_forward_from_first_cycle_entry_m=float(delta@[np.cos(heading), np.sin(heading)]))
    gates = ((angular[second_d] < .08)
             & (np.linalg.norm(velocities[second_d], axis=1) < .04)
             & (np.max(np.abs(rpy[second_d, :2]), axis=1) <= .12)
             & (np.max(np.abs(lowest[second_d]), axis=1) <= .01)
             & (contact_count[second_d] == 4))
    result['second_d_request']['all_reconstructed_entry_gates_satisfied_states'] = int(gates.sum())
    heading = rpy[6369, 2]
    delta = qpos[6825, :2]-qpos[6369, :2]
    result['stair_approach_w'] = dict(
        start_s=63.69, end_s=68.25, duration_s=4.56,
        body_start_xyz_m=qpos[6369, :3].tolist(), body_end_xyz_m=qpos[6825, :3].tolist(),
        net_forward_from_initial_heading_m=float(delta@[np.cos(heading), np.sin(heading)]),
        mean_actual_body_forward_mps=float(velocities[6370:6826, 0].mean()),
        peak_requested_mps=float(requested[6369:6825].max()), includes_two_jump_cycles=True,
        obstacle_description='Recorded contacts with terrain_stair_4/5; these boxes are 7.5/6.0 cm high with southern edge y=1.58 m. Robot approaches from y about 1.30-1.34, rather than from the 1.5 cm first step along +x.')
    jump_presses = [event for event in presses if event['key'] in (32, 74)]
    busy = [event for event in jump_presses
            if rows[max(0, round(event['simulation_time_s']*100)-1)]['jump_phase'] != 'ready']
    result['jump_requests'] = dict(
        space_presses=sum(event['key'] == 32 for event in jump_presses),
        j_presses=sum(event['key'] == 74 for event in jump_presses),
        started_cycles=len(jumps), completed_cycles=int(rows[-1]['completed_jumps']),
        busy_phase_presses_discarded=len(busy))
    OUT.mkdir()
    (OUT/'report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    with (OUT/'recomputed_states.npz').open('xb') as stream:
        np.savez_compressed(stream, body_velocity=velocities, angular_speed=angular, rpy=rpy,
                            wheel_lowest_world_z=lowest, terrain_height=ground,
                            wheel_clearance=clearance, wheel_contact_count=contact_count)
    print(json.dumps({'output': str(OUT), 'side': side, 'stalls': stalls, 'jumps': jumps,
                      'physical_summary': result['physical_summary'], 'second_d': gate_data},indent=2))


if __name__ == '__main__':
    main()
