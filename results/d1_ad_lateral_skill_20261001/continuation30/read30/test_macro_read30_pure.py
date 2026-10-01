"""Pure independent reward/range fixtures; no controller, model, or engine."""
import math
import numpy as np
import pytest

from reward_read30 import components30, macro_partition30


def test_actual_dt_body_frame_goal_cap_backtrack_and_5T_cost():
    signed = np.array([0., .02, .04, .03])
    q = np.zeros((4, 23))
    q[:, 0] = -signed  # initial yaw pi/2: initial-body left points world -x
    q[:, 3], q[:, 6] = math.sqrt(.5), math.sqrt(.5)
    limits = np.tile([80., 80., 80., 12.], 4)
    native = [{'native_index': i, 'after': {'ctrl': (.5*limits).tolist()}} for i in range(15)]
    result = components30({'qpos': q, 'time': np.arange(4)*.01}, native,
                          0, 3, 0, 1, np.arange(16), limits)
    assert result['progress_potential_difference'] == pytest.approx(1.)
    assert result['backtrack_normalized'] == pytest.approx(1/3)
    assert result['lateral_goal_error_square_integral_m2_s'] == pytest.approx(2e-6)
    assert result['torque_normalized_square_integral_s'] == pytest.approx(.0075)
    assert result['elapsed_controls'] == 3 and result['actual_native_steps'] == 15
    assert result['weighted_scalar_reward'] is None
    native[8]['native_index'] = 7
    with pytest.raises(AssertionError, match='skipped/repeated'):
        components30({'qpos': q, 'time': np.arange(4)*.01}, native,
                     0, 3, 0, 1, np.arange(16), limits)


def test_real_macro_latches_partition_and_duplicate_reject():
    events = [{'control_index': 200, 'macro_index': 0}, {'control_index': 550, 'macro_index': 1},
              {'control_index': 850, 'macro_index': 2}, {'control_index': 1150, 'macro_index': 3}]
    assert macro_partition30(events, 2000, 200) == [(200, 550), (550, 850), (850, 1150), (1150, 2000)]
    events[2]['control_index'] = 550
    with pytest.raises(AssertionError, match='distinct'):
        macro_partition30(events, 2000, 200)
