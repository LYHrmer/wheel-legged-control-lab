"""C33/C30 saved entry pairing with only the body timing difference admitted."""
from __future__ import annotations

import math
import numpy as np

from reader26 import require, close
from pairing_read31 import entry_snapshot31
from audit_pairs30 import canonical, byte_equal, ARRAYS


def pair_entry33(current, baseline, *, alpha33):
    require(float(alpha33) in (.85, .925, 1.), 'C33 comparison alpha is not frozen')
    checks = {}
    for key in ARRAYS:
        checks['initial_'+key+'_bytes'] = byte_equal(current['initial'][key], baseline['initial'][key])
        checks['preparation_through_entry_'+key+'_bytes'] = byte_equal(
            current['prefix_arrays'][key], baseline['prefix_arrays'][key])
    for key in ('reset_memory', 'prefix_sha256', 'compiled', 'profile', 'checkpoint_sha256'):
        checks[key] = canonical(current[key]) == canonical(baseline[key])
    a = {k: v for k, v in current['reset_metadata'].items() if k != 'episode_index'}
    b = {k: v for k, v in baseline['reset_metadata'].items() if k != 'episode_index'}
    checks['reset_metadata_except_episode_label'] = a == b
    common = set(current['origins']) & set(baseline['origins'])
    required = {'controller18', 'hybrid_adapter27', 'side_runtime27', 'd1_fast_side_step',
                'd1_side_step', 'state21', 'world_upright_course_11', 'runtime30',
                'side_skill30', 'kinematics24', 'full_drive_command_08',
                'engine_binding', 'run_rl16_training_08'}
    checks['unchanged_required_reset_and_control_sources'] = required <= common and all(
        current['origins'][name] == baseline['origins'][name] for name in common)
    shared_loaded = set(current['loaded_sources']) & set(baseline['loaded_sources'])
    checks['old_numerical_loaded_sources_retained'] = (
        set(baseline['loaded_sources']) <= set(current['frozen_sources']) and all(
            current['frozen_sources'][p] == baseline['loaded_sources'][p]
            for p in baseline['loaded_sources']))
    checks['shared_numerical_loaded_identities'] = all(
        current['loaded_sources'][p] == baseline['loaded_sources'][p] for p in shared_loaded)
    shared_frozen = set(current['frozen_sources']) & set(baseline['frozen_sources'])
    checks['all_shared_frozen_identities_equal'] = all(
        current['frozen_sources'][p] == baseline['frozen_sources'][p] for p in shared_frozen)
    entry, prior_entry = current['entry'], baseline['entry']
    for key in ('input_observation99', 'raw_command', 'side_start_receipt'):
        checks['entry_'+key] = canonical(entry[key]) == canonical(prior_entry[key])
    for key in ('preview_consumed', 'wheel_memory_before'):
        checks['entry_'+key] = canonical(entry['info']['controller_record'][key]) == canonical(
            prior_entry['info']['controller_record'][key])
    for key in ('raw_operator_command', 'consumed_command', 'servo_receipt'):
        checks['entry_'+key] = canonical(entry['info'][key]) == canonical(prior_entry['info'][key])
    actual, prior = current['first_event'], baseline['first_event']
    for key in ('teacher_body_from_m', 'teacher_body_to_m', 'plan_com_height_m',
                'initial_body_yaw_rad'):
        checks['teacher_before_action_'+key] = canonical(actual[key]) == canonical(prior[key])
    origin = np.asarray(actual['teacher_body_from_m'], dtype=float)
    target = np.asarray(actual['teacher_body_to_m'], dtype=float)
    height = float(actual['plan_com_height_m'])
    distance = float(np.linalg.norm(target[:2]-origin[:2]))
    old_t = max(.30, math.sqrt(5.7735*distance*height/(9.81*.015)))
    new_t = max(.30, math.sqrt((4./.9)*distance*height/(9.81*.015*alpha33)))
    checks['prior_teacher_quintic_duration'] = close(prior['teacher_shift_time_s'], old_t)
    checks['current_teacher_S_duration'] = close(actual['teacher_shift_time_s'], new_t)
    checks['different_timing_is_only_teacher_entry_exception'] = all(
        checks['teacher_before_action_'+key] for key in
        ('teacher_body_from_m', 'teacher_body_to_m', 'plan_com_height_m',
         'initial_body_yaw_rad'))
    return dict(passed=all(checks.values()), checks=checks,
                proof_kind='strict_saved_C30_entry_pair_with_independently_sized_S_time',
                timing_exception='teacher_shift_time_s only; each formula verified',
                actual_new_teacher_time_s=float(actual['teacher_shift_time_s']),
                actual_prior_teacher_time_s=float(prior['teacher_shift_time_s']),
                address_fields_compared=False, ignored_metadata_fields=['episode_index'])
