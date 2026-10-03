"""Pure saved-reader/actual worker interface fixtures; no model or physics."""
from pathlib import Path
import sys
import unittest

W = Path(__file__).resolve().parents[2]
for path in (W/'continuation30/root30', W/'continuation34/root34'):
    sys.path.insert(0, str(path))

from worker34 import pair34, score34
from read34 import _compare_online, _pair, _quality
from episode_read34 import next_side_index34


BASELINE = {
    'fixed': dict(full_tau2_mean=1., final_xy_error_m=.009, retention_ratio=.94),
    'zero': dict(full_tau2_mean=1., final_xy_error_m=.010, retention_ratio=.93)}


def fixture(*, cancel=False, cost=.8):
    task = dict(side_duration_s=8., final_xy_error_m=.009,
        retention_ratio=.94, signed_lateral_m=.039,
        safety_complete=True, entry_gates={'exact_200_preparation': True},
        online_success=not cancel, online_safe_cancel=cancel,
        cancel_to_handoff_controls=60 if cancel else None)
    macro = dict(full_case_components=dict(
        torque_normalized_square_integral_s=cost*10., elapsed_s=10.))
    report = dict(task=dict(task), qualification_passed=True,
                  online_mandatory_gates={'all': True}, macros=dict(
                      full_scene_components=dict(
                          torque_normalized_square_integral_s=cost*10.,
                          torque_normalized_square_mean=cost, elapsed_s=10.)))
    return task, macro, report


def worker_case_metrics(task, macro, *, cancel):
    """Use worker.score34, then its run_one four-field recording arithmetic."""
    saved = score34(task, macro, BASELINE, cancel=cancel)
    goal_finished = not cancel and task['online_success'] is True
    retained = (task['signed_lateral_m']*task['retention_ratio']
                if task['retention_ratio'] is not None else None)
    saved['signed_lateral_m'] = task['signed_lateral_m']
    saved['retained_signed_lateral_m'] = retained
    saved['nominal_commanded_speed_mps'] = (.04/saved['side_duration_s']
        if goal_finished else None)
    saved['actual_retained_progress_per_cycle_mps'] = (
        retained/saved['side_duration_s'] if goal_finished and retained is not None
        else None)
    return saved


class WorkerReaderInterface(unittest.TestCase):
    def test_c27_actual_side_index_is_one_based_and_contiguous(self):
        index = 0
        for actual in (1, 2, 3):
            index = next_side_index34(index, actual)
        self.assertEqual(index, 3)
        self.assertEqual(next_side_index34(index, 4), 4)
        for bad in (0, 2, 5):
            with self.assertRaises(ValueError):
                next_side_index34(3, bad)

    def test_success_and_external_quality_failure_keep_goal_speed(self):
        for cost, eligible in ((.8, True), (1.21, False)):
            with self.subTest(cost=cost):
                task, macro, report = fixture(cost=cost)
                actual = worker_case_metrics(task, macro, cancel=False)
                expected = _quality(report, BASELINE, distance_m=.04)
                _compare_online(actual, expected, distance_m=.04, cancel=False)
                self.assertIs(actual['eligible'], eligible)
                self.assertAlmostEqual(actual['nominal_commanded_speed_mps'], .04/8.)
                self.assertAlmostEqual(actual['actual_retained_progress_per_cycle_mps'],
                                       .039*.94/8.)

    def test_safe_cancel_has_progress_but_no_goal_speed(self):
        task, macro, report = fixture(cancel=True)
        actual = worker_case_metrics(task, macro, cancel=True)
        expected = _quality(report, BASELINE, distance_m=.04, cancel=True)
        _compare_online(actual, expected, distance_m=.04, cancel=True)
        self.assertIsNone(actual['nominal_commanded_speed_mps'])
        self.assertIsNone(actual['actual_retained_progress_per_cycle_mps'])
        self.assertEqual(actual['retained_signed_lateral_m'], .039*.94)

    def test_complete_pair_schema_matches_actual_worker(self):
        left = dict(case_id='d040_beta000_left', direction=1, distance_m=.04,
                    beta=0., metrics=dict(eligible=True, side_duration_s=8.,
                                          full_tau2_mean=.8))
        right = dict(case_id='d040_beta000_right', direction=-1, distance_m=.04,
                     beta=0., metrics=dict(eligible=True, side_duration_s=8.2,
                                           full_tau2_mean=.9))
        self.assertEqual(_pair(left, right, distance_m=.04, beta=0.),
                         pair34(left, right, distance_m=.04, beta=0.))


if __name__ == '__main__':
    unittest.main()
