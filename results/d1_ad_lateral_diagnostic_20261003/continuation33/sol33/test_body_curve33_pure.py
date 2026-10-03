"""Pure C33 shape and adapter fixtures; never construct or step a plant."""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from body_curve33 import ALPHAS33, PEAK_ACCEL33, RAMP_FRACTION33, curve33, duration33
from side_timing33 import make_side33


def test_curve_endpoints_symmetry_and_finite_jerk():
    assert curve33(-1, 1) == (0, 0, 0)
    assert curve33(0, 1) == (0, 0, 0)
    assert curve33(1, 1) == (1, 0, 0)
    assert curve33(2, 1) == (1, 0, 0)
    for u in (0.01, 0.05, 0.25, 0.45, 0.50, 0.55, 0.75, 0.95, 0.99):
        q, v, a = curve33(u, 1)
        qr, vr, ar = curve33(1-u, 1)
        assert q + qr == pytest.approx(1, abs=2e-13)
        assert v == pytest.approx(vr, abs=2e-13)
        assert a == pytest.approx(-ar, abs=2e-13)
    assert curve33(RAMP_FRACTION33, 1)[2] == pytest.approx(PEAK_ACCEL33)
    for knot in (0.05, 0.45, 0.55, 0.95):
        left = curve33(knot-1e-8, 1)
        right = curve33(knot+1e-8, 1)
        assert max(abs(a-b) for a, b in zip(left, right)) < 2e-6


def test_curve_derivatives_and_budget():
    h = 1e-5
    for t in (0.025, 0.2, 0.475, 0.7, 0.975):
        q, v, a = curve33(t, 1.7)
        qm, vm, _ = curve33(t-h, 1.7)
        qp, vp, _ = curve33(t+h, 1.7)
        assert v == pytest.approx((qp-qm)/(2*h), abs=2e-8)
        assert a == pytest.approx((vp-vm)/(2*h), abs=2e-8)
    span = (0.08, -0.03)
    height = 0.45
    for alpha in ALPHAS33:
        duration = duration33(span, height, 0.015, alpha)
        assert duration >= 0.30
        peak_zmp = PEAK_ACCEL33*math.hypot(*span)*height/(9.81*duration**2)
        assert peak_zmp <= 0.015*alpha + 1e-14
    assert duration33((0, 0), height, 0.015, 1.0) == 0.30


def test_adapter_changes_only_body_reference(monkeypatch):
    class FakeSide:
        def __init__(self):
            self.cfg = types.SimpleNamespace(adaptive_timing=True,
                zmp_budget_m=0.015, shift_time_bounds=(0.30, 1.40))
            self.dt = 0.1
            self.com_height = 0.45
            self.phase = 'shift'
            self.phase_time = 0.0
            self.shift_time = 0.4
            self.body_from = np.array([0., 0., .45])
            self.body_to = np.array([.08, 0., .45])
            self.pose = self.body_from.copy()
            self.body_accel = np.zeros(3)
            self.foot_offset = np.array([.01, -.02, .035])
            self.last_skill30_record = None
            self.gates_called = 0
            self.consumed = None

        def _advance(self):
            self.gates_called += 1
            self.phase_time += self.dt
            if self.phase == 'load':
                self.body_from = self.pose.copy()
                self.body_to = np.array([.10, 0., .45])
                self.body_accel[:] = 0.
                self.phase = 'recenter'
                self.phase_time = 0.
                return
            # A fake old teacher reference; the adapter must replace it before
            # compute consumes pose/acceleration, preserving all foot state.
            self.pose = self.body_from + (self.body_to-self.body_from)*0.123
            self.body_accel = np.array([99., 0., 0.])
            if self.phase in ('shift', 'recenter') and self.phase_time >= self.shift_time:
                self.pose = self.body_to.copy()
                self.body_accel[:] = 0.
                self.phase = 'unload' if self.phase == 'shift' else 'done'
                self.phase_time = 0.

        def compute(self):
            self._advance()
            self.consumed = (self.pose.copy(), self.body_accel.copy(),
                             self.foot_offset.copy())
            self.last_skill30_record = {
                'body_reference_velocity_mps': [99., 99., 99.],
                'foot_reference_position_m': self.foot_offset.tolist()}
            return np.array([7.])

    fake_module = types.ModuleType('side_skill31')
    fake_module.make_side31 = lambda *args, **kwargs: FakeSide()
    monkeypatch.setitem(sys.modules, 'side_skill31', fake_module)
    side = make_side33(None, alpha=1., geometry_binding=None,
                       control_index_provider=lambda: 0,
                       cycle_index_provider=lambda: 0)
    assert side._move_time(np.array([.08, 0., 0.])) == pytest.approx(
        duration33((.08, 0.), .45, .015, 1.))
    with pytest.raises(RuntimeError):
        side.set_alpha33(.925)
    for tick in range(1, 5):
        assert side.compute().tolist() == [7.]
        pose, accel, foot = side.consumed
        q, v, a = curve33(tick*side.dt, side.shift_time)
        assert pose[0] == pytest.approx(.08*q)
        assert accel[0] == pytest.approx(.08*a)
        assert np.array_equal(foot, [.01, -.02, .035])
        expected_velocity = .08*v if side.phase == 'shift' else 0.
        assert side.last_skill30_record['body_reference_velocity_mps'][0] == pytest.approx(
            expected_velocity)
        assert side.last_skill30_record['foot_reference_position_m'] == [.01, -.02, .035]
    assert side.phase == 'unload' and side.gates_called == 4
    side.phase = 'load'
    side.phase_time = 0.
    side.compute()
    assert side.phase == 'recenter'
    assert side.consumed[0][0] == pytest.approx(.08)
    assert side.consumed[1][0] == 0.
    assert side.last_skill30_record['body_reference_velocity_mps'] == [0., 0., 0.]
    for _ in range(4):
        side.compute()
    assert side.phase == 'done'
    assert side.consumed[0][0] == pytest.approx(.10)
    assert side.consumed[1][0] == 0.
    assert side.gates_called == 9
    side.phase = 'idle'
    side.set_alpha33(.925)
    assert side._body_alpha33 == .925
    with pytest.raises(ValueError):
        side.set_alpha33(.9)
