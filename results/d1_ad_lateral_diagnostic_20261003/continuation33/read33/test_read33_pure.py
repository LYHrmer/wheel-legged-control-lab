"""No-model checks for the independent C33 formula and selection rule."""
from pathlib import Path
import sys

import pytest

W = Path(__file__).resolve().parents[2]
for path in (W/'continuation26/root26', W/'continuation24',
             W/'continuation27/read27', W/'continuation30/read30',
             W/'continuation30/pair_read30', W/'continuation31/read31',
             Path(__file__).resolve().parent):
    sys.path.insert(0, str(path))

from reference_read33 import K, integrated_curve33, planned_duration33
from comparison_read33 import select_alpha33


def test_piecewise_jerk_kinematics_and_duration_budget():
    assert integrated_curve33(0, 1) == (0, 0, 0)
    assert integrated_curve33(1, 1) == (1, 0, 0)
    assert integrated_curve33(.5, 1) == pytest.approx((.5, 2., 0.), abs=2e-14)
    assert integrated_curve33(.05, 1)[2] == pytest.approx(K)
    for u in (.025, .2, .475, .7, .975):
        q, v, a = integrated_curve33(u, 1)
        h = 1e-5
        qm, vm, _ = integrated_curve33(u-h, 1)
        qp, vp, _ = integrated_curve33(u+h, 1)
        assert v == pytest.approx((qp-qm)/(2*h), abs=2e-8)
        assert a == pytest.approx((vp-vm)/(2*h), abs=2e-8)
    duration = planned_duration33([.08, 0., 0.], .45, .925)
    assert duration >= .30
    assert K*.08*.45/(9.81*duration**2) <= .015*.925 + 1e-14


def test_selection_requires_both_directions_then_worst_cycle_mean_cost_alpha():
    def row(cycle, cost, eligible=True):
        return dict(eligible=eligible, engineering_speed_passed=True,
                    actual=dict(cycle_s=cycle, full_torque_square_mean=cost))
    data = {.85: {1: row(8., 1.), -1: row(9., 1.)},
            .925: {1: row(9., .9), -1: row(8., .9)},
            1.: {1: row(7., .1), -1: row(7., .1, False)}}
    assert select_alpha33(data)['selected_alpha33'] == .925
    data[.925][1] = row(9., 1.)
    data[.925][-1] = row(8., 1.)
    assert select_alpha33(data)['selected_alpha33'] == .85
