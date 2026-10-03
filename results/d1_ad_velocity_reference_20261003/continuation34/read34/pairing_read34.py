"""Strict common reset/B22 entry plus independently checked C34 D exceptions."""
from __future__ import annotations

import math
import numpy as np

from reader26 import require, close
from pairing_read33 import pair_entry33
from pairing_read31 import pair_entry31
from side_math27 import scalar_state


def pair_entry34(current, baseline, *, direction, distance_m,
                 archived_c33=False):
    require(direction in (-1, 1) and distance_m in (.03, .04),
            'C34 pairing direction/distance differs')
    old = (pair_entry31(current, baseline) if archived_c33 else
           pair_entry33(current, baseline, alpha33=1.))
    require(old['passed'], 'C34 reset/B22/source/entry/first teacher differs')
    row = current['entry']
    before = row['info']['controller_record']['diagnostic_before']
    after = row['info']['controller_record']['diagnostic_after']
    start = row['info']['controller_record']['side_calculation']['c34_start_distance']
    q = np.asarray(current['prefix_arrays']['qpos'][-1], dtype=float)
    v = np.asarray(current['prefix_arrays']['qvel'][-1], dtype=float)
    yaw = scalar_state(q, v, current['compiled'])['rpy'][2]
    delta = direction*distance_m*np.array([-math.sin(yaw), math.cos(yaw), 0.])
    swing = float(np.clip(1.875*distance_m/.16, .22, .90))
    checks = dict(strict_common_entry=old['passed'],
        actual_start_distance=start == dict(incoming_hybrid_distance_m=.03,
            configured_distance_m=distance_m, forwarded_fast_distance_m=distance_m,
            accepted=True),
        actual_initial_pose=close(before['initial_pose'], q[:3]),
        actual_delta=close(before['delta'], delta) and close(after['delta'], delta),
        actual_swing_duration=close(after['swing_time_s'], swing),
        actual_initial_yaw=close(before['start_yaw_rad'], yaw))
    return dict(passed=all(checks.values()), checks=checks,
                inherited_pair=old, actual_commanded_distance_m=distance_m,
                expected_delta_m=delta.tolist(), expected_swing_duration_s=swing,
                permitted_changed_fields=['commanded_distance', 'actual_delta',
                    'task_goal', 'swing_duration'],
                unchanged_common_reset_B22_and_teacher_entry=True)
