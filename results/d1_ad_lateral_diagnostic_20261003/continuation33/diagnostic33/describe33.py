"""Bounded saved-control description; no native files, model, fit or physics."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

SCHEMA = 'd1-c33-saved-descriptive-contract-v1'
OUT_SCHEMA = 'd1-c33-saved-descriptive-v1'
DT = .01
BODY = ('shift', 'recenter')
SWING = ('unload', 'lift', 'swing', 'lower', 'abort_land')


def need(value, message):
    if not value:
        raise ValueError(message)


def finite(values, shape=None):
    a = np.asarray(values, dtype=float)
    need((shape is None or a.shape == shape) and np.isfinite(a).all(),
         'saved numeric shape/nonfinite value differs')
    return a


def summary(values):
    if not values:
        return dict(n=0, mean=None, rms=None, max_abs=None, p95_abs=None)
    a = finite(values)
    return dict(n=len(a), mean=float(a.mean()), rms=float(np.sqrt(np.mean(a*a))),
                max_abs=float(np.max(np.abs(a))),
                p95_abs=float(np.percentile(np.abs(a), 95)))


def reject_constant(value):
    raise ValueError('nonfinite JSON constant: '+value)


def load_json(path):
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def rotation(quaternion):
    w, x, y, z = finite(quaternion, (4,))
    need(abs(float(w*w+x*x+y*y+z*z)-1.) <= 1e-5,
         'saved base quaternion differs from a unit rotation')
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def hash_file(path, deadline):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'input missing/symlink: '+str(path))
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        while True:
            need(time.monotonic() < deadline, 'C33 descriptive 60-second wall limit')
            block = stream.read(1 << 20)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return dict(bytes=size, sha256=digest.hexdigest())


def gate_mask(phase, before, predicate, profile):
    """Exclusive failed-gate set only after the phase's nominal time is ready."""
    if phase not in ('shift', 'recenter', 'load'):
        return None
    planned = (float(before['shift_time_s']) if phase in BODY
               else float(profile['load_time_s']))
    if float(before['phase_time_s'])+DT < planned:
        return None
    need(predicate['source_phase'] == phase and predicate['time_ready'] is True,
         'saved ready predicate/phase clock differs')
    failed = []
    if not predicate['all_wheels_contact']:
        failed.append('contact')
    if predicate['speed_ready'] is not True:
        failed.append('speed')
    if phase == 'shift' and predicate['margin_ready'] is not True:
        failed.append('margin')
    if (phase == 'recenter' and not failed):
        if predicate['position_ready'] is not True:
            failed.append('position')
        if predicate['yaw_ready'] is not True:
            failed.append('yaw')
    return '+'.join(failed) if failed else 'ready'


def clip_stats(calc, before, after, profile):
    target = finite(calc['target_position_rad'], (16,))
    previous = finite(before['previous_target'], (16,))
    previous_v = finite(before['previous_velocity_target'], (16,))
    velocity = finite(calc['velocity_target_rad_s'], (16,))
    acceleration = finite(calc['accel_target_rad_s2'], (16,))
    vcap, acap = float(profile['target_velocity_clip']), float(profile['target_accel_clip'])
    need(vcap > 0 and acap > 0, 'invalid old target clip')
    raw_v = (target-previous)/DT
    raw_a = (velocity-previous_v)/DT
    need(np.allclose(velocity, np.clip(raw_v, -vcap, vcap), atol=1e-8, rtol=0)
         and np.allclose(acceleration, np.clip(raw_a, -acap, acap), atol=1e-8, rtol=0),
         'saved 16-joint target clip is not the old Fast arithmetic')
    vremoved = np.abs(raw_v-velocity)
    aremoved = np.abs(raw_a-acceleration)
    leg = after['leg']
    swinging = leg is not None and after['phase'] in SWING
    swing_indices = set(range(4*int(leg), 4*int(leg)+3)) if swinging else set()
    categories = {'swing_leg_3': [], 'stance_leg': [], 'wheel': []}
    for j in range(16):
        key = 'wheel' if j%4 == 3 else ('swing_leg_3' if j in swing_indices else 'stance_leg')
        categories[key].append(j)
    rows = {}
    for key, indices in categories.items():
        rows[key] = dict(channels=len(indices),
            velocity_clipped=sum(bool(vremoved[j] > 1e-9) for j in indices),
            accel_clipped=sum(bool(aremoved[j] > 1e-9) for j in indices),
            velocity_removed_rad_s=float(vremoved[indices].sum()) if indices else 0.,
            accel_removed_rad_s2=float(aremoved[indices].sum()) if indices else 0.)
    # Wheels use a separate position brake; their velocity-target PD terms are
    # cancelled in Fast.compute. Estimate pre-actuator changes for leg joints.
    kd = np.full(16, float(profile['stance_kd']))
    if swing_indices:
        kd[list(swing_indices)] = float(profile['swing_kd'])
    pd_delta = float(np.sum(np.abs((velocity-raw_v)[:16].reshape(4, 4)[:, :3]
                                   *kd.reshape(4, 4)[:, :3])))
    return rows, pd_delta


def describe_episode(item, deadline):
    construction = load_json(item['construction_path'])
    profile = construction['side_profile']
    binding = construction['side_kinematic_binding']
    mass = float(finite(binding['kinematics24']['body_mass']).sum())
    ipos = finite(binding['base_body_ipos'], (3,))
    need(mass > 0 and profile['torque_source'] == 'd1_fast_side_step',
         'saved side construction/profile differs')
    runs, body_data, gate_counts = [], [], {}
    body_groups = {'moving_reference': [], 'stationary_reference': []}
    clips = {name: dict(velocity_clipped=0, accel_clipped=0,
                        velocity_removed_rad_s=0., accel_removed_rad_s2=0.)
             for name in ('swing_leg_3', 'stance_leg', 'wheel')}
    clip_pd, foot_v, foot_a = [], [], []
    force_terms = {name: [] for name in ('accel_N', 'position_N', 'damping_N',
                                        'original_xy_N', 'wrench_residual_N')}
    current = None
    count = side_count = 0
    with gzip.open(item['controls_path'], 'rt') as stream:
        for line in stream:
            need(time.monotonic() < deadline, 'C33 descriptive 60-second wall limit')
            row = json.loads(line, parse_constant=reject_constant)
            need(row['control_index'] == count and count < 2200,
                 'saved control index gap or per-episode cap exceeded')
            count += 1
            if row['actor'] != 'traditional_side':
                need(row['skill30'] is None, 'non-side row has a side reference')
                if current is not None:
                    runs.append(current)
                    current = None
                continue
            side_count += 1
            record, skill = row['info']['controller_record'], row['skill30']
            before, after = record['diagnostic_before'], record['diagnostic_after']
            phase = before['phase']
            need(record['schema'] == 'd1-c27-hybrid-traditional-side-v1'
                 and skill['schema'] == 'd1-c30-lateral-reference-skill-v1'
                 and skill['control_index'] == count-1,
                 'saved side record/header identity differs')
            if item['kind'] in ('development', 'cancel'):
                need(skill['body_curve33']['shape'] ==
                     'symmetric_trapezoidal_acceleration_r0.05',
                     'C33 body-curve header differs')
            elif item['kind'] == 'baseline':
                need('body_curve33' not in skill, 'old baseline contains C33 timing')
            if current is None or current['phase'] != phase:
                if current is not None:
                    runs.append(current)
                planned = before['nominal_duration_s']
                current = dict(phase=phase, control_start=count-1,
                    control_end=count, controls=0, planned_T_s=planned,
                    gate_failure_masks={}, gate_ready_after_T_controls=0,
                    transition_to=None, body_samples=[])
            current['controls'] += 1
            current['control_end'] = count
            if after['phase'] != phase:
                current['transition_to'] = after['phase']
                current['transition_control_index'] = count-1
            mask = gate_mask(phase, before, record['ready_predicates'], profile)
            if mask is not None:
                current['gate_failure_masks'][mask] = current['gate_failure_masks'].get(mask, 0)+1
                phase_masks = gate_counts.setdefault(phase, {})
                phase_masks[mask] = phase_masks.get(mask, 0)+1
                current['gate_ready_after_T_controls'] += int(mask == 'ready')
            pre = row['pre_state']
            q, v = finite(pre['qpos'], (23,)), finite(pre['qvel'], (22,))
            calc = record['side_calculation']
            categories, pd_delta = clip_stats(calc, before, after, profile)
            for name, values in categories.items():
                for key in clips[name]:
                    clips[name][key] += values[key]
            clip_pd.append(pd_delta)
            foot_v.append(float(np.linalg.norm(finite(skill['foot_reference_velocity_mps'], (3,)))))
            foot_a.append(float(np.linalg.norm(finite(skill['foot_reference_accel_mps2'], (3,)))))
            if phase in BODY:
                ref_p = finite(skill['body_reference_position_m'], (3,))
                ref_v = finite(skill['body_reference_velocity_mps'], (3,))
                acc = finite(skill['body_reference_accel_mps2'], (3,))
                rmat = rotation(q[3:7])
                com_v = v[:3]+np.cross(rmat@v[3:6], rmat@ipos)
                f_acc = mass*acc[:2]
                f_pos = mass*float(profile['body_kp'])*(ref_p[:2]-q[:2])
                f_damp = -mass*float(profile['body_kd'])*com_v[:2]
                original = finite(calc['original_wrench'], (6,))[:2]
                residual = float(np.linalg.norm(original-(f_acc+f_pos+f_damp)))
                need(residual <= 1e-5, 'saved original XY wrench differs from legacy moving-body PD')
                for name, force in (('accel_N', f_acc), ('position_N', f_pos),
                                    ('damping_N', f_damp), ('original_xy_N', original)):
                    force_terms[name].append(float(np.linalg.norm(force)))
                force_terms['wrench_residual_N'].append(residual)
                span = finite(before['body_to'], (3,))[:2]-finite(before['body_from'], (3,))[:2]
                norm = float(np.linalg.norm(span))
                along = span/norm if norm else np.array([0., 0.])
                sample = dict(control_index=count-1,
                    pose_error_m=float(np.linalg.norm(ref_p[:2]-q[:2])),
                    pose_error_along_m=float((ref_p[:2]-q[:2])@along),
                    origin_speed_mps=float(np.linalg.norm(v[:2])),
                    reference_speed_mps=float(np.linalg.norm(ref_v[:2])),
                    velocity_error_along_mps=float((ref_v[:2]-v[:2])@along),
                    com_velocity_error_along_mps=float((ref_v[:2]-com_v[:2])@along))
                current['body_samples'].append(sample)
                body_data.append(sample)
                body_groups['moving_reference' if np.linalg.norm(ref_v[:2]) > 1e-12
                            else 'stationary_reference'].append(sample)
    if current is not None:
        runs.append(current)
    need(0 < count <= 2200 and side_count > 0, 'empty/oversized saved episode')
    for run in runs:
        values = run.pop('body_samples')
        run['actual_duration_s'] = DT*run['controls']
        planned = run['planned_T_s']
        run['actual_minus_planned_s'] = (run['actual_duration_s']-planned
                                        if planned is not None else None)
        run['post_nominal_wait_s'] = (max(0., run['actual_duration_s']-planned)
                                      if planned is not None else None)
        run['duration_to_planned_ratio'] = (run['actual_duration_s']/planned
                                             if planned is not None and planned > 0 else None)
        run['body_tracking'] = {key: summary([r[key] for r in values]) for key in
            ('pose_error_m', 'pose_error_along_m', 'origin_speed_mps',
             'reference_speed_mps', 'velocity_error_along_mps',
             'com_velocity_error_along_mps')}
    return dict(id=item['id'], kind=item['kind'], direction=item['direction'],
        controls=count, side_controls=side_count, phase_runs=runs,
        gate_failure_masks_after_planned_T=gate_counts,
        body_tracking={key: summary([r[key] for r in body_data]) for key in
            ('pose_error_m', 'pose_error_along_m', 'origin_speed_mps',
             'reference_speed_mps', 'velocity_error_along_mps',
             'com_velocity_error_along_mps')},
        body_tracking_by_reference_motion={group: {key: summary([r[key] for r in samples])
            for key in ('pose_error_m', 'pose_error_along_m', 'origin_speed_mps',
                        'reference_speed_mps', 'velocity_error_along_mps',
                        'com_velocity_error_along_mps')}
            for group, samples in body_groups.items()},
        saved_total_mass_kg=mass,
        moving_body_wrench_terms_N={key: summary(value) for key, value in force_terms.items()},
        moving_body_wrench_terms_per_kg={key: summary([x/mass for x in value])
                                          for key, value in force_terms.items()},
        foot_reference_speed_mps=summary(foot_v),
        foot_reference_accel_mps2=summary(foot_a),
        joint_target_clip_by_role=clips,
        pre_actuator_pd_clip_delta_abs_nm=summary(clip_pd))


def phase_totals(episode):
    totals = {}
    for run in episode['phase_runs']:
        phase = run['phase']
        entry = totals.setdefault(phase, dict(actual_s=0., planned_s=0.,
                                              post_nominal_wait_s=0., runs=0))
        entry['actual_s'] += run['actual_duration_s']
        entry['planned_s'] += run['planned_T_s'] or 0.
        entry['post_nominal_wait_s'] += run['post_nominal_wait_s'] or 0.
        entry['runs'] += 1
    return totals


def comparisons(episodes):
    """Fixed arithmetic contrasts; no significance, causal, or success inference."""
    by_id = {episode['id']: episode for episode in episodes}
    result = []
    for direction, suffix in ((1, 'left'), (-1, 'right')):
        fixed = by_id['C30_fixed_nonzero_'+suffix]
        zero = by_id['C30_zero_'+suffix]
        fixed_phases = phase_totals(fixed)
        zero_phases = phase_totals(zero)
        for tag in ('085', '0925', '100'):
            candidate = by_id['alpha_'+tag+'_'+suffix]
            candidate_phases = phase_totals(candidate)
            row = dict(direction=direction, candidate_id=candidate['id'],
                       fixed_id=fixed['id'], zero_id=zero['id'],
                       candidate_phase_totals=candidate_phases,
                       fixed_phase_totals=fixed_phases,
                       zero_phase_totals=zero_phases,
                       candidate_minus_fixed_s={}, candidate_minus_zero_s={})
            for phase in sorted(set(candidate_phases) | set(fixed_phases) | set(zero_phases)):
                for baseline, key in ((fixed_phases, 'candidate_minus_fixed_s'),
                                      (zero_phases, 'candidate_minus_zero_s')):
                    if phase in candidate_phases and phase in baseline:
                        row[key][phase] = {metric: candidate_phases[phase][metric]-baseline[phase][metric]
                                           for metric in ('actual_s', 'planned_s', 'post_nominal_wait_s')}
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started+60.
    script_path = Path(__file__).resolve(strict=True)
    contract_path = args.contract.resolve(strict=True)
    need(args.output != args.contract and args.output.resolve() != script_path,
         'output aliases contract or source')
    source_before = hash_file(script_path, deadline)
    contract_before = hash_file(contract_path, deadline)
    contract = load_json(contract_path)
    need(contract['schema'] == SCHEMA and contract['wall_limit_s'] == 60,
         'C33 descriptive contract header differs')
    episodes = contract['episodes']
    need(type(episodes) is list and len(episodes) == contract['max_episodes'] == 12
         and sum(item['kind'] == 'development' for item in episodes) == 6
         and sum(item['kind'] == 'cancel' for item in episodes) == 2
         and sum(item['kind'] == 'baseline' for item in episodes) == 4
         and all(item['kind'] in ('development', 'cancel', 'baseline')
                 and item['direction'] in (-1, 1)
                 and ((item['alpha'] in (.85, .925, 1.) and item['cycle_index'] in range(8))
                      if item['kind'] != 'baseline'
                      else item['alpha'] is None)
                 for item in episodes)
         and len({item['id'] for item in episodes}) == len(episodes)
         and contract['max_controls_per_episode'] == 2200
         and contract['max_total_controls'] == 26400
         and contract['control_dt_s'] == DT,
         'C33 descriptive episode cap/schema differs')
    expected_ids = {f'alpha_{tag}_{side}' for tag in ('085', '0925', '100')
                    for side in ('left', 'right')}
    expected_ids |= {f'C30_{base}_{side}' for base in ('fixed_nonzero', 'zero')
                     for side in ('left', 'right')}
    expected_ids |= {'cancel_100_left', 'cancel_100_right'}
    need({item['id'] for item in episodes} == expected_ids,
         'C33 descriptive fixed episode identities differ')
    paths = {str(Path(item[key]).resolve(strict=True))
             for item in episodes for key in ('controls_path', 'construction_path')}
    case_paths = {path for path in contract['inputs']
                  if Path(path).name == 'case_definition33.json'}
    need(len(case_paths) == 1, 'C33 case definition input is missing or ambiguous')
    paths.update(case_paths)
    need(all(Path(item['controls_path']).name == 'controls.jsonl.gz'
             and Path(item['construction_path']).name == 'construction_receipt.json'
             for item in episodes)
         and paths == set(contract['inputs']),
         'C33 descriptive input set contains missing/extra paths')
    checked = {path: hash_file(path, deadline) for path in sorted(paths)}
    need(checked == contract['inputs'], 'C33 descriptive input hash differs before reading')
    case_definition = load_json(next(iter(case_paths)))
    need(case_definition['schema'] == 'd1-c33-case-definition-v1'
         and len(case_definition['cases']) == 8,
         'C33 case definition header/count differs')
    for item in episodes:
        if item['kind'] != 'baseline':
            case = case_definition['cases'][item['cycle_index']]
            need(case['case_id'] == item['id'] and case['cycle_index'] == item['cycle_index']
                 and case['alpha'] == item['alpha'] and case['direction'] == item['direction']
                 and case['kind'] == item['kind'],
                 'C33 case definition/episode metadata differs')
    result = []
    total_controls = 0
    for item in episodes:
        episode = describe_episode(item, deadline)
        total_controls += episode['controls']
        need(total_controls <= contract['max_total_controls'],
             'C33 total saved control cap exceeded')
        result.append(episode)
    need({path: hash_file(path, deadline) for path in sorted(paths)} == checked,
         'C33 descriptive input changed during reading')
    need(hash_file(script_path, deadline) == source_before
         and hash_file(contract_path, deadline) == contract_before,
         'C33 descriptive source or contract changed during reading')
    need(time.monotonic() < deadline, 'C33 descriptive 60-second wall limit')
    output = dict(schema=OUT_SCHEMA, scope='saved-controls-only_descriptive_no_qualification',
        contract_path=str(contract_path),
        contract_identity=contract_before, script_path=str(script_path),
        script_identity=source_before, inputs=checked,
        episodes=result, episode_count=len(result),
        total_controls=total_controls, fixed_direction_comparisons=comparisons(result),
        cancellation_episodes=[episode['id'] for episode in result if episode['kind']=='cancel'],
        elapsed_s_before_write=time.monotonic()-started,
        native_files_opened=0, model_calls=0, physics_steps=0,
        source_notes=dict(body_P_D_damping='inherited Fast controller, zero-velocity damping on actual base inertial COM; no claim of a new C33 defect',
            body_reference_velocity_is_not_joint_target_clipped=True,
            joint_clip_torque_deltas_are_pre_actuator_estimates=True,
            gate_masks_are_exclusive_combinations_not_additive_counts=True))
    payload = json.dumps(output, indent=2, sort_keys=True, allow_nan=False)+'\n'
    need(time.monotonic() < deadline and not args.output.exists(),
         'C33 descriptive deadline or exclusive output differs')
    with args.output.open('x') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    need(time.monotonic() < deadline, 'C33 descriptive 60-second wall limit')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Preserve one small failure receipt at the requested fresh destination.
        # This never retries, reads extra episode files, or replaces a result.
        if '--output' in sys.argv:
            index = sys.argv.index('--output') + 1
            if index < len(sys.argv):
                failed_path = Path(sys.argv[index])
                if not failed_path.exists():
                    failure = dict(schema='d1-c33-saved-descriptive-failure-v1',
                                   error_type=type(exc).__name__, error=str(exc),
                                   native_files_opened=0, model_calls=0, physics_steps=0)
                    with failed_path.open('x') as stream:
                        json.dump(failure, stream, indent=2, allow_nan=False)
                        stream.write('\n')
                        stream.flush()
                        os.fsync(stream.fileno())
        raise
