"""Root reconstruction of original-feedback turn physics evidence; never integrate."""
import copy
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

import mujoco
import numpy as np

W = Path('/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920')
R = Path('/home/lyh/wheel-legged-control-lab')
sys.path.insert(0, str(W/'plane_turn_center_audit_01'))
import audit_candidate as common
from scripts import probe_d1_heading_g1 as g1
from wheel_legged_control.d1.model import (D1_JOINT_NAMES, LEG_PREFIXES, JOINT_TORQUE_LIMIT,
    JOINT_POSITION_LOW, JOINT_POSITION_HIGH, JOINT_VELOCITY_LIMIT, build_d1_model)


def safe(request, q, qd):
    result = np.clip(request, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
    result[((q >= JOINT_POSITION_HIGH) & (result > 0.)) | ((q <= JOINT_POSITION_LOW) & (result < 0.))
           | ((np.abs(qd) >= JOINT_VELOCITY_LIMIT) & (result*qd > 0.))] = 0.
    return result


def audit_case(a, folder, baseline, case, gates, model):
    a.context = case['name']
    a.manifest(folder, 'complete_manifest.json', require_complete=True)
    a.manifest(baseline, 'complete_manifest.json', require_complete=True)
    z, bz = common.arrays(folder/'states.npz'), common.arrays(baseline/'states.npz')
    diag, bd = common.rows(folder/'plane_diagnostics.jsonl.gz'), common.rows(baseline/'plane_diagnostics.jsonl.gz')
    damping = common.rows(folder/'turn_authority.jsonl.gz')
    native, bn = common.rows(folder/'native_physics_entries.jsonl.gz'), common.rows(baseline/'native_physics_entries.jsonl.gz')
    trace = common.rows(folder/'trace.jsonl.gz')
    summary = common.read_json(folder/'candidate_summary.json')
    a.need(len(diag) == len(damping) == len(trace) == 800 and len(native) == 4000, 'fixed evidence lengths')
    prefix = 200 if case['command']['yaw_pulse'] else 800
    for key in z:
        count = prefix if key.endswith('actions') or (prefix == 200 and key == 'observations') else prefix+1
        a.bits(z[key][:count], bz[key][:count], 'saved prefix '+key)
    for x, y in zip(native[:prefix*5], bn[:prefix*5]):
        a.exact(x, y, 'complete native prefix')
    for x, y in zip(diag[:prefix], bd[:prefix]):
        a.exact(x, y, 'complete diagnostic prefix')
    a.need(not np.any(z['requested_actions']) and not np.any(z['applied_actions']), 'zero actions')
    execution = common.arrays(folder/'execution_states.npz')
    for key in ('qpos', 'qvel'):
        a.need(len(execution[key]) == 802, 'archive includes final close snapshot')
        a.bits(execution[key][:-1], z[key], 'physical archive/state match')
        a.bits(execution[key][-1], execution[key][-2], 'no terminal extra integration')
    data = mujoco.MjData(model)
    base = model.body('base_link').id
    wheels = [model.body(p+'_foot').id for p in LEG_PREFIXES]
    joints = np.array([model.joint(n).id for n in D1_JOINT_NAMES])
    dofs, qadr = model.jnt_dofadr[joints].reshape(4, 4), model.jnt_qposadr[joints]
    leg_dofs, wheel_dofs = dofs[:, :3].ravel(), dofs[:, 3]
    jac, velocity = np.zeros((3, model.nv)), np.zeros(6)
    geometry = []
    for k in range(801):
        data.qpos[:], data.qvel[:] = z['qpos'][k], z['qvel'][k]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        a.need(data.time == 0., 'scratch time zero')
        a.bits(data.qpos, z['qpos'][k], 'kinematic qpos immutable')
        a.bits(data.qvel, z['qvel'][k], 'kinematic qvel immutable')
        rotation = data.xmat[base].reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1., 1.)))
        lateral = (data.xpos[wheels]-data.xpos[base]) @ np.array([-np.sin(yaw), np.cos(yaw), 0.])
        jx = np.zeros((4, 3))
        for leg, body in enumerate(wheels):
            mujoco.mj_jacBody(model, data, jac, None, body)
            jx[leg] = rotation[:, 0] @ jac[:, dofs[leg, :3]]
        u = np.einsum('ki,ki->k', jx, data.qvel[dofs[:, :3]])
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base, velocity, 0)
        vx, rate = float(rotation[:, 0] @ velocity[3:]), float(rotation[:, 2] @ velocity[:3])
        a.close(z['truth_positions_world_m'][k], data.xpos[base], 'state visible body position', 1e-12)
        geometry.append({'yaw': yaw, 'rate': rate, 'vx': vx, 'lateral': lateral, 'u': u, 'jx': jx})
        if k:
            row = trace[k-1]
            a.close(row['truth_position_world_m'], data.xpos[base], 'trace/state position', 1e-12)
            a.close(row['body_forward_mps'], vx, 'state body vx', 2e-12)
            a.close(row['heading_task']['truth_heading_after'], yaw, 'state yaw', 2e-12)
            a.close(row['heading_task']['truth_yaw_rate_after_rps'], rate, 'state body yaw rate', 2e-12)
            a.close(row['actual_roll_pitch_rad'], [roll, pitch], 'state roll pitch', 2e-12)
            a.close(row['metrics']['height_error_m'], data.xpos[base, 2]-case['command']['height_m'], 'state height', 2e-12)
            common.validate_endpoint(a, diag[k-1]['endpoint_contacts'], k, model, data, base, wheels, leg_dofs, wheel_dofs, jac)
    reference = float(common.read_json(folder/'episode_metadata.json')['heading_reference_initial']['heading_rad'])
    a.close(reference, geometry[0]['yaw'], 'initial heading')
    independent = copy.deepcopy(trace)
    old_integral = np.zeros(4)
    active_ticks = []
    for k, (row, d, dr) in enumerate(zip(trace, diag, damping)):
        a.context = f"{case['name']} control {k}"
        pre, end, record = geometry[k], geometry[k+1], dr['record']
        raw = asdict(g1.command_at_tick(case['command'], k))
        a.exact(raw, d['raw_user_command'], 'raw command diagnostic')
        a.exact(raw, row['heading_task']['user_command_before'], 'raw command scorer')
        a.need(row['tick'] == d['tick'] == dr['tick'] == k and dr['endpoint_tick'] == k+1, 'execution/endpoint ticks')
        a.need(dr['raw_callback_count_after_prepare'] == k+2, 'one raw callback each new decision')
        a.close(record['control_time_s'], k*.01, 'execution binding time', 2e-11)
        a.exact(record['raw_forward_mps'], raw['forward_velocity_mps'], 'raw bound forward')
        a.exact(record['raw_yaw_rate_rps'], raw['yaw_rate_rps'], 'raw bound yaw')
        active = raw['forward_velocity_mps'] == 0. and raw['yaw_rate_rps'] != 0.
        a.need(record['active'] is active and record['enabled'] is True, 'executed gate')
        if active:
            active_ticks.append(k)
        h = row['heading_task']
        servo = float(np.clip(raw['yaw_rate_rps']+2.*common.wrap(reference-pre['yaw'])-.4*(pre['rate']-raw['yaw_rate_rps']), -1., 1.))
        a.close(d['servo_command']['yaw_rate_rps'], servo, 'original outer heading law', 2e-10)
        a.close(record['servo_yaw_rate_rps'], servo, 'recorded servo', 2e-10)
        a.exact(d['servo_command']['forward_velocity_mps'], raw['forward_velocity_mps'], 'unshaped forward')
        r0 = servo+4.*(servo-pre['rate'])
        capped = float(np.clip(r0, -.6, .6))
        effective = r0 if active else capped
        target = np.clip((raw['forward_velocity_mps']-effective*pre['lateral'])/.087, -30., 30.)
        a.close(d['effective_yaw_request_rps'], effective, 'active original feedback / inactive original cap', 2e-10)
        a.close(d['wheel_target_rad_s'], target, 'original wheel targets', 2e-10)
        q, qd = z['qpos'][k, qadr], z['qvel'][k, dofs.ravel()]
        before = np.array(d['wheel_integral_before_nm'])
        a.bits(before, old_integral, 'PI recurrence')
        error = target-qd[3::4]
        trial = np.clip(before+3.*.01*error, -4., 4.)
        request_wheel = 2.2*error+trial
        accept = (np.abs(request_wheel) <= JOINT_TORQUE_LIMIT[3::4]) | (request_wheel*error < 0.)
        after = np.where(accept, trial, before)
        a.close(d['wheel_integral_after_nm'], after, 'original PI antiwindup', 2e-10)
        a.close(d['wheel_error_before_rad_s'], error, 'original PI error', 2e-10)
        old_integral = np.array(d['wheel_integral_after_nm'])
        unlimited = np.array(d['unlimited_torque_nm'])
        a.close(unlimited[3::4], 2.2*error+after, 'original wheel PI torque', 5e-10)
        a.bits(np.array(record['wheel_request_nm']), unlimited[3::4], 'record unlimited wheel request')
        a.close(record['wheel_target_rad_s'], target, 'record restored target', 2e-10)
        base_target = np.clip((raw['forward_velocity_mps']-capped*pre['lateral'])/.087, -30., 30.)
        unclipped = (raw['forward_velocity_mps']-effective*pre['lateral'])/.087
        a.close(record['base_wheel_target_rad_s'], base_target, 'record original target', 2e-10)
        a.close(record['unclipped_wheel_target_rad_s'], unclipped, 'record unclipped target', 2e-10)
        a.bits(np.array(record['wheel_target_clipped']), target != unclipped, 'target clip mask')
        a.close(record['original_unclipped_yaw_rps'], r0, 'original yaw formula', 2e-10)
        a.close(record['original_capped_yaw_rps'], capped, 'original cap diagnostic', 2e-10)
        a.close(record['effective_yaw_request_rps'], effective, 'record actual yaw request', 2e-10)
        a.close(record['body_yaw_rate_rps'], pre['rate'], 'record body rate', 2e-10)
        a.bits(np.array(record['wheel_integral_before_nm']), before, 'record PI before')
        a.bits(np.array(record['wheel_integral_after_nm']), old_integral, 'record PI after')
        a.need(record['inner_cap_applied'] is (not active), 'actual inner cap application')
        would = abs(capped) >= .6-1e-12
        a.need(record['original_inner_cap_would_be_occupied'] == would, 'original cap occupancy')
        a.need(record['inner_cap_occupied'] == ((not active) and would), 'actual cap occupancy')
        a.need(d['inner_yaw_limit_occupied'] == record['inner_cap_occupied'], 'corrected inherited cap diagnostic')
        if active:
            a.need(d['inner_cap_applied'] is False and d['legacy_abs_effective_yaw_ge_0p6'] == (abs(effective) >= .6-1e-12), 'explicit active diagnostic interpretation')
        actual = safe(unlimited, q, qd)
        a.bits(np.array(d['requested_torque_nm']), actual, 'all axis original protections')
        a.bits(np.array(record['wheel_torque_nm']), actual[3::4], 'record safe wheel torque')
        a.bits(np.array(d['applied_torque_nm']), np.tile(actual, (5, 1)), 'native held torques')
        dt = sum(e['actual_dt_s'] for e in row['physics_wrench_samples'])
        a.close(h['actual_dt_s'], dt, 'actual raw integration interval', 1e-12)
        reference = common.wrap(reference+dt*raw['yaw_rate_rps'])
        error_heading = common.wrap(end['yaw']-reference)
        a.close(row['heading_error_rad'], error_heading, 'original raw heading error', 2e-11)
        a.close(row['cross_track_after_m'], z['truth_positions_world_m'][k+1, 1]-z['truth_positions_world_m'][0, 1], 'original cross track', 1e-12)
        ir = independent[k]
        ir['body_forward_mps'] = end['vx']
        ir['user_forward_error_mps'] = end['vx']-raw['forward_velocity_mps']
        ir['heading_error_rad'] = error_heading
        ir['heading_task']['user_yaw_rate_error_after'] = end['rate']-raw['yaw_rate_rps']
        a.need(not row['terminated'] and row['truncated'] == (k == 799), 'full original duration')
    a.exact(active_ticks, list(range(200, 250)) if case['command']['yaw_pulse'] else [], 'exact raw active set')
    metrics, gate_result = g1.gates_and_metrics(case, trace, z['truth_positions_world_m'], gates)
    for key, value in metrics.items():
        a.exact(summary[key], value, 'original metric '+key)
    a.exact(summary['gates'], gate_result, 'original gate reduction')
    im, ig = g1.gates_and_metrics(case, independent, z['truth_positions_world_m'], gates)
    for key in ('heading_peak_rad', 'heading_rmse_rad', 'velocity_rmse_mps', 'user_yaw_rate_rmse_rps'):
        a.close(im[key], metrics[key], 'independent raw score '+key, 2e-10)
    a.exact(ig, gate_result, 'independent raw gates')
    native_report = common.validate_native(a, native, diag, trace, case, model, wheels, g1)
    receipt = common.read_json(folder/'plane_execution_receipt.json')
    a.need(receipt['actual_physics_substeps_from_clock'] == 4000 and receipt['actual_completed_control_intervals'] == 800
           and receipt['partial_interval_physics_substeps'] == 0 and receipt['error'] is None, 'native duration receipt')
    if prefix == 800:
        original = common.read_json(baseline/'summary.json')
        for key, value in original.items():
            if key != 'model':
                a.exact(summary[key], value, 'all original noop summary '+key)
    return {'case': case['name'], 'original_gates': gate_result, 'heading_peak_rad': im['heading_peak_rad'],
        'prefix_control_intervals': prefix, 'prefix_states': prefix+1, 'prefix_observations': prefix if prefix == 200 else prefix+1,
        'native': native_report, 'active_intervals': len(active_ticks)}


def main():
    output = W/'turn_authority_independent_audit.json'
    if output.exists():
        raise FileExistsError(output)
    a = common.Audit()
    result = {'passed': False, 'new_physics_steps': 0, 'new_contact_solves': 0, 'controller_compute_calls': 0}
    try:
        with ExitStack() as guards:
            for name in ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_inverse', 'mj_collision'):
                guards.enter_context(patch.object(mujoco, name, a.forbid(name)))
            candidate, baseline = W/'plane_turn_authority_01', W/'flat_plane_02'
            a.remember(__file__)
            a.remember(common.__file__, 'b4ffd91d27b6218c675a5da024f6660a1642f3668ca7427686ca57ca2e525bb6')
            a.manifest(candidate, 'manifest.json', require_complete=True)
            protocol, aggregate = common.read_json(candidate/'protocol.json'), common.read_json(candidate/'summary.json')
            original_path = R/'results/d1_driving_stability_development/heading_g1_01/evaluation_protocol.json'
            a.remember(original_path, g1.PROTOCOL_SHA256)
            original = common.read_json(original_path)
            a.exact(protocol['cases'], [original['cases'][i] for i in (2, 3, 0, 1)], 'fixed original cases')
            a.exact(protocol['original_gates'], original['proposed_gates'], 'original gates')
            for path, digest in protocol['input_sha256'].items():
                a.remember(path, digest)
            frozen = common.read_json(R/'results/d1_budget_study/protocol.json')['source_sha256']
            a.need(len(frozen) == 77, 'frozen77 count')
            for path, digest in frozen.items():
                a.remember(R/path, digest)
            model = build_d1_model()
            a.need(model.nhfield == 0 and model.nv == 22 and model.geom_type[0] == mujoco.mjtGeom.mjGEOM_PLANE, 'native plane model')
            for case in protocol['cases']:
                a.cases.append(audit_case(a, candidate/case['name'], baseline/case['name'], case, protocol['original_gates'], model))
                pair = common.read_json(candidate/f"pair_{case['name']}.json")
                a.need(pair['passed'] and all(pair['checks'].values()), 'pair claims pass')
                a.exact(pair, next(p for p in aggregate['pairs'] if p['comparison'] == case['name']), 'aggregate pair identity')
            a.need(aggregate['actual_new_control_transitions'] == 3200 and aggregate['actual_new_physics_substeps'] == 16000, 'actual fixed budget')
            a.need(aggregate['candidate_passed_both_turn_cases'] == all(c['original_gates']['passed'] for c in a.cases[:2]), 'turn result separation')
            for path, digest in a.input_sha256.items():
                a.need(common.sha(path) == digest, 'inputs immutable')
            result.update(passed=True, candidate_control_intervals=3200, candidate_native_substeps=16000,
                          frozen77_unchanged=True, candidate_passed_both_turn_cases=aggregate['candidate_passed_both_turn_cases'])
    except BaseException as error:
        result['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    result.update(check_count=a.checks, maximum_errors=a.maximum_errors, cases=a.cases,
                  input_sha256=a.input_sha256, prohibited_api_calls=a.forbidden_calls,
                  limitation='Leg PD/support base request is archived, not independently reimplemented. Original/restored yaw formula, wheel PI, protections, native/endpoint evidence and original raw scores are reconstructed. Valid evidence does not imply task gates pass.')
    with output.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k: result[k] for k in ('passed', 'check_count', 'cases', 'error') if k in result}))
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
