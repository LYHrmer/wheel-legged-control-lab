"""Pure adversarial record-link checks; no controller/engine import."""
import ast
import copy
from pathlib import Path

import numpy as np
import pytest
from drive_damping_validator_04 import validate_drive_trace, validate_precontrol_archive

B = 126.4374005337902


def fixture():
    rows, before = [], []
    position = np.tile([0.0, .8, -1.5, 0.0], 4)
    velocity = np.tile([.02, 0.0, 0.0, 0.0], 4)
    jacobian = np.zeros((4, 3, 4))
    jacobian[:, 0, 0] = 1.0
    damping = np.tile([-B * .02, 0.0, 0.0, 0.0], 4)
    base = np.tile([1.0, 2.0, -3.0, .5], 4)
    qpos = np.r_[[-3.8, 0.0, .455, 1.0, 0.0, 0.0, 0.0], position]
    qvel = np.r_[np.zeros(6), velocity]
    for tick, forward in enumerate([0.0, .2, 0.0]):
        drive = damping if forward else np.zeros(16)
        stop_delta = damping if tick == 2 else np.zeros(16)
        drive_total = base + drive
        final = drive_total + stop_delta
        record = {
            'schema': 'd1-drive-jx-leg-damping-record-v1', 'active': bool(forward),
            'raw_forward_mps': forward, 'servo_forward_mps': forward,
            'damping_nspm': B, 'provider_state_sequence': tick,
            'provider_state_age_s': 0.0, 'control_time_s': tick * .01,
            'base_rotation': np.eye(3), 'foot_jacobian': jacobian.copy(),
            'joint_position': position.copy(), 'joint_velocity': velocity.copy(),
            'projected_jx': np.tile([1.0, 0.0, 0.0], (4, 1)),
            'relative_forward_mps': np.full(4, .02), 'delta_torque_nm': drive.copy(),
            'raw_joint_power_w': -B * 4 * .02**2 if forward else 0.0,
            'base_requested_torque_nm': base.copy(), 'base_protected_torque_nm': base.copy(),
            'base_leg_pd_nm': base.copy(), 'total_requested_torque_nm': drive_total.copy(),
            'total_protected_torque_nm': drive_total.copy(),
            'actual_protected_increment_nm': drive.copy(),
        }
        rows.append({
            'tick': tick, 'raw_command': {'forward_velocity_mps': forward},
            'controller_result': {'drive_damping': record, 'leg_pd_nm': final.copy(),
                                  'torque_nm': final.copy(), 'requested_torque_nm': final.copy(),
                                  'clipped_action': np.zeros(8)},
            'stop_record': {'active': tick == 2, 'base_requested_torque_nm': drive_total.copy(),
                            'base_leg_pd_nm': drive_total.copy(),
                            'delta_torque_nm': stop_delta.copy(),
                            'total_requested_torque_nm': final.copy(), 'safe_torque_nm': final.copy()},
            'stop_active': tick == 2, 'torque_nm': final.copy(),
            'action': np.zeros(8), 'applied_action': np.zeros(8),
            'authority_record': {'wheel_torque_nm': final[[3, 7, 11, 15]].copy()},
        })
        before.append({
            'tick': tick, 'provider_state_sequence': tick, 'provider_state_age_s': 0.0,
            'provider_control_time_s': tick * .01, 'provider_base_rotation': np.eye(3),
            'provider_foot_jacobian': jacobian.copy(), 'provider_joint_position': position.copy(),
            'provider_joint_velocity': velocity.copy(), 'raw_precontrol_qpos': qpos.copy(),
            'raw_precontrol_qvel': qvel.copy(), 'raw_joint_position': position.copy(),
            'raw_joint_velocity': velocity.copy(),
        })
    return rows, before, np.tile(qpos, (4, 1)), np.tile(qvel, (4, 1))


def test_valid_initial_drive_stop_chain():
    rows, before, _, _ = fixture()
    result = validate_drive_trace(rows, expected_raw_schedule=[0.0, .2, 0.0],
                                  pre_control_states=before)
    assert result['record_valid']
    assert result['drive_active_controls'] == result['stop_active_controls'] == 1


def test_reject_gate_timing_wheel_and_stage_tampering():
    rows, before, _, _ = fixture()
    for field, value in [('active', False), ('provider_state_sequence', 0),
                         ('provider_state_age_s', .01), ('control_time_s', float('nan'))]:
        bad = copy.deepcopy(rows)
        bad[1]['controller_result']['drive_damping'][field] = value
        with pytest.raises((ValueError, TypeError)):
            validate_drive_trace(bad, pre_control_states=before)
    for section, field, index in [('stop_record', 'base_requested_torque_nm', 0),
                                  ('controller_result', 'torque_nm', 0),
                                  ('authority_record', 'wheel_torque_nm', 0)]:
        bad = copy.deepcopy(rows)
        bad[1][section][field][index] += .01
        with pytest.raises(ValueError):
            validate_drive_trace(bad, pre_control_states=before)
    bad = copy.deepcopy(rows)
    bad[1]['controller_result']['drive_damping']['delta_torque_nm'][3] = 1e-12
    with pytest.raises(ValueError):
        validate_drive_trace(bad, pre_control_states=before)


def test_provider_to_raw_to_saved_npz_links():
    rows, before, qpos, qvel = fixture()
    kwargs = {'saved_qpos': qpos, 'saved_qvel': qvel,
              'joint_qpos_addresses': np.arange(7, 23),
              'joint_dof_addresses': np.arange(6, 22)}
    assert validate_precontrol_archive(before, **kwargs)['record_valid']
    bad = copy.deepcopy(before)
    bad[1]['provider_joint_velocity'][0] += .001
    with pytest.raises(ValueError):
        validate_drive_trace(rows, pre_control_states=bad)
    with pytest.raises(ValueError):
        validate_precontrol_archive(bad, **kwargs)
    bad = copy.deepcopy(before)
    bad[1]['raw_precontrol_qvel'][6] += .001
    with pytest.raises(ValueError):
        validate_precontrol_archive(bad, **kwargs)
    bad = copy.deepcopy(before)
    bad[1]['provider_base_rotation'][0, 0] = .999
    with pytest.raises(ValueError):
        validate_precontrol_archive(bad, **kwargs)


def test_cooperative_parent_chain_is_single_source_call():
    source = Path(__file__).with_name('drive_damping_controller_04.py').read_text()
    tree = ast.parse(source)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    rolling = classes['DriveRollingController']
    stage = classes['DriveDampingStageController']
    assert [ast.unparse(base) for base in rolling.bases] == [
        'RollingResidualCompositionController', 'DriveDampingStageController']
    assert [ast.unparse(base) for base in stage.bases] == ['TurnYawAuthorityController']
    compute = next(node for node in stage.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'compute')
    parent_calls = [node for node in ast.walk(compute) if isinstance(node, ast.Call)
                    and ast.unparse(node.func) == 'super().compute']
    assert len(parent_calls) == 1
    assert not any(isinstance(node, (ast.Assign, ast.AnnAssign))
                   and 'last_damping' in ast.unparse(node) for node in ast.walk(compute))
