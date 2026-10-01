"""Two minimal final-evaluation decision/actual-state pairing regressions."""
from __future__ import annotations

import copy

from eval_read31 import np
from pairing_read31 import pair_entry31, gain_gates31, ARRAYS


def test_speed_against_both_references_and_physical_qualification_are_required():
    zero = dict(cycle_s=10., goal_error_m=.004, retention_ratio=.98, full_torque_square_mean=.01)
    fixed = dict(cycle_s=9., goal_error_m=.004, retention_ratio=.98, full_torque_square_mean=.01)
    learned = dict(cycle_s=8.5, goal_error_m=.006, retention_ratio=.96, full_torque_square_mean=.011)
    assert gain_gates31(learned, zero, fixed, True, True, True)['passed']
    # Nine percent faster than zero still fails the fixed ten-percent gate.
    assert not gain_gates31({**learned, 'cycle_s': 9.1}, zero, fixed, True, True, True)['passed']
    assert not gain_gates31(learned, zero, fixed, True, True, False)['passed']
    assert not gain_gates31(learned, zero, fixed, False, True, True)['passed']


def test_new_episode_label_is_allowed_but_changed_warmstart_entry_is_not():
    names = ('controller18', 'hybrid_adapter27', 'side_runtime27', 'd1_fast_side_step',
        'd1_side_step', 'state21', 'world_upright_course_11', 'runtime30', 'side_skill30',
        'kinematics24', 'full_drive_command_08', 'engine_binding', 'run_rl16_training_08')
    entry = dict(input_observation99=[0.]*99, raw_command={'vx': 0.}, side_start_receipt={'accepted': True},
        info=dict(controller_record=dict(preview_consumed={'sequence': 200}, wheel_memory_before={'PI': [0.]*4}),
            raw_operator_command={'vx': 0.}, consumed_command={'vx': 0.}, servo_receipt={'tick': 200}))
    one = dict(initial={k: np.zeros(1) for k in ARRAYS},
        prefix_arrays={k: np.zeros((201, 1)) for k in ARRAYS}, reset_memory={'provider_sequence': 0},
        reset_metadata={'episode_index': 0, 'mode': 'eval'}, prefix_sha256='same_actual_prefix',
        compiled={'geometry': 'same'}, profile={'teacher': 'same'}, checkpoint_sha256='same_B22',
        origins={name: {'bytes': 1, 'sha256': name} for name in names}, loaded_sources={}, frozen_sources={},
        entry=entry, first_event=dict(teacher_body_from_m=[0.,0.,.455], teacher_body_to_m=[.01,.01,.455],
            teacher_shift_time_s=.7, plan_com_height_m=.45, initial_body_yaw_rad=0.))
    other = copy.deepcopy(one)
    other['reset_metadata']['episode_index'] = 7
    assert pair_entry31(other, one, prior_C30=True)['passed']
    other['prefix_arrays']['qacc_warmstart'][200, 0] = 1e-12
    proof = pair_entry31(other, one, prior_C30=True)
    assert not proof['passed']
    assert not proof['checks']['preparation_through_entry_qacc_warmstart_bytes']
