"""Independently audit a completed five-segment real GLFW side-drive trial."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile

import mujoco
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yaw(qpos):
    w, x, y, z = qpos[3:7]
    return float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-render-quality', choices=('normal', 'low'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.attempt/'rollout'
    summary = json.loads((raw/'summary.json').read_text())
    protocol = json.loads((raw/'protocol.json').read_text())
    manifest = json.loads((raw/'manifest.json').read_text())
    assert all(sha(raw/name) == digest for name, digest in manifest.items())
    with tarfile.open(raw/'source.tar.gz') as archive:
        assert all(hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
                   for name, digest in protocol['source_sha256'].items())
    rows = list(csv.DictReader((raw/'telemetry.csv').open()))
    events = [json.loads(line) for line in (raw/'keyboard_events.jsonl').read_text().splitlines()]
    segments = json.loads((raw/'segments.json').read_text())
    assert len(segments) == 5 and summary['segments'] == 5
    assert summary['source_unchanged'] and summary['stop_reason'] == 'keyboard_escape'
    assert summary['steps'] == len(rows)
    assert protocol['side_step_profile'] == 'fast'
    assert protocol['compiled_model_sha256'] == sha(raw/'model.mjb')
    if args.expected_render_quality is not None:
        assert protocol['render_quality'] == args.expected_render_quality
    with np.load(raw/'states.npz', allow_pickle=False) as archive:
        states = {name: archive[name] for name in archive.files}
    assert len(states['qpos']) == len(rows)+len(segments)
    assert states['applied_torque_nm'].shape == (len(rows), 16)
    assert all(np.isfinite(states[name]).all() for name in ('qpos', 'qvel', 'applied_torque_nm'))
    for index, row in enumerate(rows):
        before, after = int(row['state_before_index']), int(row['state_after_index'])
        assert int(row['tick']) == index+1
        assert states['segment_id'][before] == states['segment_id'][after] == int(row['segment_id'])
        assert abs(states['segment_time_s'][after]-states['segment_time_s'][before]-.01) < 1e-8
        assert row['applied_controller'] in ('legacy', 'd1_fast_side_step')
    for segment in segments:
        index = segment['first_state_index']
        np.testing.assert_array_equal(states['qpos'][index, :3], [0., 0., .455])
        np.testing.assert_array_equal(states['qvel'][index], np.zeros(states['qvel'].shape[1]))
        assert states['segment_time_s'][index] == 0.

    # Recompute selected contacts on separate MjData at the recorded integrated
    # states. This never advances or modifies the original controlled plant.
    model = mujoco.MjModel.from_binary_path(str(raw/'model.mjb'))
    data = mujoco.MjData(model)
    legs = ('FL', 'FR', 'RL', 'RR')
    bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, leg+'_foot') for leg in legs]
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, leg+'_foot_joint') for leg in legs]
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    terrain = {i for i in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or '').startswith(('floor', 'terrain_'))}

    def geometry(state_index):
        data.qpos[:] = states['qpos'][state_index]
        data.qvel[:] = states['qvel'][state_index]
        mujoco.mj_forward(model, data)
        touching = set()
        for c in data.contact:
            a, b = int(c.geom1), int(c.geom2)
            other = b if a in terrain else a if b in terrain else None
            if other is not None and c.efc_address >= 0:
                body = int(model.geom_bodyid[other])
                if body in bodies:
                    touching.add(bodies.index(body))
        az = np.clip(data.xaxis[joints, 2], -1., 1.)
        lowest = data.xpos[bodies, 2] - .087*np.sqrt(1-az*az) - .020*np.abs(az)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base, velocity, 0)
        origin_velocity = velocity[3:] - np.cross(velocity[:3], data.xmat[base].reshape(3, 3)@model.body_ipos[base])
        return touching, lowest, float(np.linalg.norm(origin_velocity))

    measurements = []
    for segment, direction in ((0, 1), (1, -1)):
        selected = [row for row in rows if int(row['segment_id']) == segment]
        entry = next(row for row in selected if row['applied_controller'] == 'd1_fast_side_step')
        done = next(row for row in selected if row['side_success'] == 'True')
        before, after = int(entry['state_before_index']), int(done['state_after_index'])
        heading = yaw(states['qpos'][before])
        left = np.array([-np.sin(heading), np.cos(heading)])
        forward = np.array([np.cos(heading), np.sin(heading)])
        delta = states['qpos'][after, :3]-states['qpos'][before, :3]
        lateral, drift = float(delta[:2]@left), float(delta[:2]@forward)
        assert direction*lateral > .02 and abs(drift) < .03
        yaw_error = yaw(states['qpos'][after])-heading
        yaw_error = float(np.arctan2(np.sin(yaw_error), np.cos(yaw_error)))
        assert abs(yaw_error) < .12
        contacts, _, speed = geometry(after)
        assert len(contacts) == 4 and speed < .04
        lifted = {}
        for leg_index, leg in enumerate(legs):
            swing = [row for row in selected if row['side_phase'] == 'swing'
                     and row['side_active_leg'] == leg and int(row['tick']) <= int(done['tick'])]
            assert swing
            midpoint = swing[len(swing)//2]
            touching, lowest, _ = geometry(int(midpoint['state_after_index']))
            assert leg_index not in touching and lowest[leg_index] > .012
            lifted[leg] = float(lowest[leg_index])
        measurements.append(dict(segment=segment, direction=direction,
                                 duration_s=float(done['segment_after_time_s'])-float(entry['segment_before_time_s']),
                                 lateral_m=lateral, forward_drift_m=drift, yaw_error_rad=yaw_error,
                                 done_contacts=4, done_origin_speed_mps=speed, swing_clearance_m=lifted))

    cancellations = []
    for segment in (2, 3):
        selected = [row for row in rows if int(row['segment_id']) == segment]
        cancel = next(row for row in selected if row['side_failure'] == 'cancelled')
        done = next(row for row in selected if row['side_done'] == 'True')
        active = [row for row in selected if int(cancel['tick']) <= int(row['tick']) <= int(done['tick'])]
        assert all(row['applied_controller'] == 'd1_fast_side_step' for row in active)
        next_row = rows[int(done['tick'])]
        assert next_row['applied_controller'] == 'legacy'
        contacts, _, speed = geometry(int(done['state_after_index']))
        assert len(contacts) == 4 and speed < .04
        if segment == 3:
            assert cancel['input_focused'] == 'False'
        assert all(row['jump_phase'] == 'ready' for row in selected)
        cancellations.append(dict(segment=segment, cancel_tick=int(cancel['tick']),
                                  landed_tick=int(done['tick']), contacts=4,
                                  handoff_origin_speed_mps=speed,
                                  focus_at_cancel=cancel['input_focused']))
    assert any(event['type'] == 'jump_blocked' for event in events)
    assert any(event['type'] == 'focus' and not event['focused'] for event in events)
    final_rows = [row for row in rows if row['segment_id'] == '4']
    phases = {row['jump_phase'] for row in final_rows}
    assert {'crouch', 'thrust', 'flight', 'landing'} <= phases
    assert max(int(row['completed_jumps']) for row in final_rows) >= 1
    max_heading = max(float(row['actual_heading_rad']) for row in final_rows)
    assert max_heading > .2
    assert min(states['qpos'][:, 2]) > .30
    result = dict(status='passed', automatic_test_not_human_acceptance=True,
                  attempt=args.attempt.name, segments=len(segments), rows=len(rows),
                  render_quality=protocol.get('render_quality'),
                  manifest_verified=len(manifest), archived_source_files_verified=len(protocol['source_sha256']),
                  input_sha256=manifest, measurements=measurements, cancellations=cancellations,
                  jump_phases=sorted(phases), completed_jumps=max(int(row['completed_jumps']) for row in final_rows),
                  final_segment_max_heading_rad=max_heading,
                  state_source='Recorded qpos/qvel; selected contact checks recomputed on separate MjData after mj_forward.',
                  reset_integrity='All five initial states match explicit safe spawn; no cross-segment transitions.',
                  limitations='Independent audit of an isolated real GLFW software-rendered test; no human or hardware acceptance claim.')
    args.output.mkdir(exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
