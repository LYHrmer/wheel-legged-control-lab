"""Small adversarial tests of new pairing/comparison gates; no saved run imports."""
import copy
import numpy as np

from audit_pairs30 import byte_equal, compare_direction


def cases():
    zero = dict(evidence_valid=True, qualification_passed=True,
                metrics=dict(cycle_s=10., goal_error_m=.006, retention_ratio=.99,
                             full_torque_square_mean=.02, nonzero_xy=False, nonzero_action=False, overlap_legs=0))
    fixed = copy.deepcopy(zero)
    fixed['metrics'].update(cycle_s=9.4, goal_error_m=.007, retention_ratio=.98,
                            full_torque_square_mean=.021, nonzero_xy=True, nonzero_action=True, overlap_legs=2)
    return fixed, zero


def test_bitwise_pairing_rejects_signed_zero_and_dtype_change():
    original = np.array([0., 1.], dtype=np.float64)
    assert byte_equal(original, original.copy())
    assert not byte_equal(original, np.array([-0., 1.], dtype=np.float64))
    assert not byte_equal(original, original.astype(np.float32))


def test_two_direction_gate_cannot_be_rescued_by_pooled_speed_or_missing_pair():
    fast, zero = cases()
    slow = copy.deepcopy(fast)
    fast['metrics']['cycle_s'] = 8.
    slow['metrics']['cycle_s'] = 9.6
    assert compare_direction(fast, zero, True)['passed']
    fast['metrics']['nonzero_xy'] = False  # Effective lambda alone is permitted.
    assert compare_direction(fast, zero, True)['passed']
    assert not compare_direction(slow, zero, True)['gates']['cycle_at_least_5pct_faster']
    assert not compare_direction(fast, zero, False)['passed']


def test_shorter_time_does_not_excuse_mean_torque_or_qualification_failure():
    fixed, zero = cases()
    assert compare_direction(fixed, zero, True)['passed']
    fixed['metrics'].update(cycle_s=8., full_torque_square_mean=.025)
    # Its integral could be equal or smaller; the registered gate is the mean.
    assert not compare_direction(fixed, zero, True)['gates']['full_torque_square_mean_le_1p20zero']
    fixed['metrics']['full_torque_square_mean'] = .02
    fixed['qualification_passed'] = False
    assert not compare_direction(fixed, zero, True)['passed']
