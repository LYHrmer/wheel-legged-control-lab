"""Synthetic chains built from immutable saved values, never new physics."""
import ast
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from body_speed_validator_06 import (
    validate_body_precontrol_archive,
    validate_body_trace,
)
from drive_damping_math_04 import protect_requested

W = Path(__file__).resolve().parent
WHEELS = [3, 7, 11, 15]


def transform(row, before, ipos):
    result = row['controller_result']
    drive = result['drive_damping']
    rotation = np.array(drive['base_rotation'])
    raw = np.array(before['raw_precontrol_qvel'])
    body_velocity = rotation.T @ raw[:3] + np.cross(raw[3:6], ipos)
    before['provider_base_linear_velocity_body'] = body_velocity.tolist()
    omega = np.array(drive['joint_velocity'])[WHEELS]
    forward = row['raw_command']['forward_velocity_mps']
    scalar = 2.2 * (np.mean(omega) - body_velocity[0] / .087) if forward else 0.0
    delta = np.zeros(16)
    delta[WHEELS] = scalar
    parent = np.array(drive['total_requested_torque_nm'])
    parent_safe = np.array(drive['total_protected_torque_nm'])
    parent_wheel = np.array(result['wheel_nm'])
    request = parent + delta
    safe, limited = protect_requested(request, drive['joint_position'], drive['joint_velocity'])
    body = {
        'schema': 'd1-drive-body-common-p-record-v1', 'active': forward != 0.0,
        'raw_forward_mps': forward, 'servo_forward_mps': forward,
        'control_time_s': drive['control_time_s'],
        'provider_state_sequence': drive['provider_state_sequence'], 'provider_state_age_s': 0.0,
        'base_linear_velocity_body': body_velocity.tolist(),
        'base_rotation': drive['base_rotation'], 'joint_position': drive['joint_position'],
        'joint_velocity': drive['joint_velocity'], 'wheel_omega_rad_s': omega.tolist(),
        'mean_omega_rad_s': float(np.mean(omega)),
        'body_equivalent_omega_rad_s': float(body_velocity[0] / .087),
        'common_error_rad_s': float(np.mean(omega) - body_velocity[0] / .087),
        'scalar_delta_nm': float(scalar), 'delta_torque_nm': delta.tolist(),
        'wheel_kp': 2.2, 'wheel_radius_m': .087,
        'parent_requested_torque_nm': parent.tolist(),
        'parent_protected_torque_nm': parent_safe.tolist(),
        'parent_wheel_nm': parent_wheel.tolist(),
        'total_requested_torque_nm': request.tolist(), 'total_protected_torque_nm': safe.tolist(),
        'total_wheel_nm': (parent_wheel + delta).tolist(),
        'actual_protected_increment_nm': (safe - parent_safe).tolist(),
        'total_torque_limited': limited.tolist(),
        'original_pi_integral_before_nm': row['authority_record']['wheel_integral_before_nm'],
        'original_pi_integral_after_nm': row['authority_record']['wheel_integral_after_nm'],
        'nominal_wheel_speed_rad_s': result['nominal_wheel_speed_rad_s'],
        'wheel_speed_target_rad_s': row['authority_record']['wheel_target_rad_s'],
    }
    result['body_common_p'] = body
    result['wheel_nm'] = body['total_wheel_nm']
    result['requested_torque_nm'] = request.tolist()
    result['torque_nm'] = safe.tolist()
    result['torque_limited'] = limited.tolist()
    row['torque_nm'] = safe.tolist()
    for name in ('base_requested_torque_nm', 'total_requested_torque_nm'):
        row['stop_record'][name] = request.tolist()
    row['stop_record']['safe_torque_nm'] = safe.tolist()
    return row, before


def fixture():
    with gzip.open(W / 'body_validator_fixture_02.json.gz', 'rt') as stream:
        original = json.load(stream)
    ipos = original['base_body_ipos_local_m']
    pairs = [transform(row, before, ipos) for row, before in zip(original['trace'], original['precontrol'])]
    return [x[0] for x in pairs], [x[1] for x in pairs], ipos


def test_chain_accepts_true_intermediate_authority_without_rewriting_it():
    rows, before, _ = fixture()
    row = rows[-1]
    assert not np.array_equal(row['authority_record']['wheel_torque_nm'],
                              np.array(row['torque_nm'])[WHEELS])
    identity = copy.deepcopy(row['authority_record'])
    check = validate_body_trace(rows, expected_raw_schedule=[r['raw_command']['forward_velocity_mps'] for r in rows], pre_control_states=before)
    assert check['record_valid'] and check['body_common_p_active_controls'] == 226
    assert row['authority_record'] == identity


def test_chain_rejects_each_broken_intermediate_and_pi_link():
    source, before, _ = fixture()
    paths = [
        ('authority_record', 'wheel_torque_nm', 0),
        ('authority_record', 'wheel_request_nm', 0),
        ('controller_result', 'body_common_p', 'parent_requested_torque_nm', 3),
        ('controller_result', 'body_common_p', 'original_pi_integral_after_nm', 0),
        ('controller_result', 'body_common_p', 'delta_torque_nm', 0),
        ('controller_result', 'body_common_p', 'total_protected_torque_nm', 3),
        ('stop_record', 'base_requested_torque_nm', 3),
        ('torque_nm', 3),
    ]
    for path in paths:
        row = copy.deepcopy(source)
        target = row[-1]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] += .125
        with pytest.raises(ValueError):
            validate_body_trace(row, pre_control_states=before)


def test_full_body_com_raw_binding_accepts_offset_and_rejects_lateral_forgery():
    _, prefix, ipos = fixture()
    before = prefix[-1]
    before['tick'] = 0
    before['provider_state_sequence'] = 0
    before['provider_control_time_s'] = 0.0
    qpos = np.array(before['raw_precontrol_qpos'])
    qvel = np.array(before['raw_precontrol_qvel'])
    qvel[:6] = [.3, -.2, .4, .6, -.7, .8]
    before['raw_precontrol_qvel'] = qvel.tolist()
    rotation = np.array(before['provider_base_rotation'])
    expected = rotation.T @ qvel[:3] + np.cross(qvel[3:6], ipos)
    before['provider_base_linear_velocity_body'] = expected.tolist()
    args = {'saved_qpos': np.array([qpos, qpos]), 'saved_qvel': np.array([qvel, qvel]),
            'joint_qpos_addresses': np.arange(7, 23), 'joint_dof_addresses': np.arange(6, 22),
            'base_body_ipos_local_m': ipos,
            'endpoints': [{'tick': 0, 'body_vx_mps': float(expected[0])}, {'tick': 1}]}
    checked = validate_body_precontrol_archive([before], **args)
    assert checked['full_body_com_velocity_linked']
    assert not np.allclose(expected, rotation.T @ qvel[:3])
    for channel in range(3):
        bad = copy.deepcopy(before)
        bad['provider_base_linear_velocity_body'][channel] += .1
        with pytest.raises(ValueError):
            validate_body_precontrol_archive([bad], **args)
    bad_args = copy.deepcopy(args)
    bad_args['endpoints'][0]['body_vx_mps'] += .1
    with pytest.raises(ValueError):
        validate_body_precontrol_archive([before], **bad_args)


def test_source_chain_and_transfer_reuse_the_frozen_never_reset_guard():
    controller = ast.parse((W / 'body_speed_controller_06.py').read_text())
    classes = {node.name: node for node in controller.body if isinstance(node, ast.ClassDef)}
    assert [ast.unparse(x) for x in classes['BodyCommonPRollingController'].bases] == [
        'RollingResidualCompositionController', 'BodyCommonPStageController']
    assert [ast.unparse(x) for x in classes['BodyCommonPStageController'].bases] == [
        'DriveDampingStageController']
    rolling = ast.unparse(classes['BodyCommonPRollingController'])
    assert 'assert_frozen_arithmetic_identity()' in rolling
    transfer = ast.parse((W / 'body_speed_transfer_env_06.py').read_text())
    imports = [node for node in transfer.body if isinstance(node, ast.ImportFrom)]
    assert any(node.module == 'drive_fresh_lifecycle_05'
               and any(x.name == 'assert_never_reset_inactive' for x in node.names) for node in imports)
    install = next(node for node in transfer.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'install_body_controller')
    calls = [node for node in ast.walk(install) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'assert_never_reset_inactive']
    assert len(calls) == 2
    assert not any(isinstance(node, ast.Attribute) and node.attr == '_steps'
                   for node in ast.walk(install))
