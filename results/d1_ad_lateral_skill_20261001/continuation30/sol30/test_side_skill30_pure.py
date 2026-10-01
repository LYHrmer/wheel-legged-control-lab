"""No-engine checks of the C30 reference seam; root runs these tests."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from side_skill30 import _curve, make_side30


class FakeTeacher:
    def __init__(self):
        self.reset_calls = 0
        self.prepare_calls = 0
        self.ik_calls = 0
        self.index = 0
        self.yaw = 0.0
        self.quat = np.array([1., 0., 0., 0.])
        self.com_height = .45
        self.phase = 'idle'
        self.failure = None
        self.leg = None
        self.pose = np.array([0., 0., .455])
        self.feet = np.zeros((4, 3))
        self.foot_offset = np.zeros(3)
        self.scratch = types.SimpleNamespace(qpos=np.zeros(23))

    def reset(self):
        self.reset_calls += 1

    def _prepare_leg(self):
        self.prepare_calls += 1
        self.phase = 'shift'
        self.failure = None
        self.leg = 0
        self.body_from = np.array([0., 0., .455])
        self.body_to = np.array([0., .03, .455])
        self.shift_time = .5

    def _ik(self, *args, **kwargs):
        self.ik_calls += 1
        assert kwargs == {'outer': 2, 'inner': 6}
        return (np.zeros(16), 0., np.zeros((4, 3)),
                np.array([.05, 0., .455]))

    def _move_time(self, span):
        assert np.linalg.norm(span[:2]) == pytest.approx(.022)
        return .4

    @property
    def status(self):
        return {'active': self.phase not in ('idle', 'done', 'failed')}

    def cancel(self):
        self.phase = 'abort_land'


def _side(monkeypatch, arm):
    teacher = FakeTeacher()
    calls = []

    def factory(plant, side_arm):
        calls.append((plant, side_arm))
        return teacher

    monkeypatch.setitem(sys.modules, 'hybrid_adapter27',
                        types.SimpleNamespace(make_fast_side27=factory))
    binding = {'kinematics24': {},
               'wheel_index_by_body_id': {10: 0, 11: 1, 12: 2, 13: 3}}
    plant = object()
    side = make_side30(plant, arm, binding, lambda: 0)
    assert side is teacher  # The factory creates exactly one controller/scratch.
    assert calls == [(plant, 'teacher')]
    return side


def test_curve_has_exact_finite_endpoints():
    assert _curve(-1., 2.) == (0., 0., 0.)
    assert _curve(2., 2.) == (1., 0., 0.)
    assert _curve(3., 2.) == (1., 0., 0.)
    middle = _curve(1., 2.)
    assert middle[0] == pytest.approx(.5)
    assert middle[1] > 0.


def test_zero_arm_preserves_teacher_plan_and_skips_extra_ik(monkeypatch):
    side = _side(monkeypatch, 'zero_left')
    side._prepare_leg()
    event, = side.drain_macro_events()
    assert side.prepare_calls == 1 and side.ik_calls == 0
    assert side.shift_time == .5
    assert np.array_equal(side.body_to, np.array([0., .03, .455]))
    assert event['reason'] == 'teacher_exact_zero'
    assert event['consumed_action'] == (0., 0., 0.)
    assert event['plan_com_height_m'] == .45
    assert side.drain_macro_events() == ()


def test_fixed_plan_one_extra_ik_and_new_span_time(monkeypatch):
    side = _side(monkeypatch, 'fixed_nonzero_left')
    monkeypatch.setitem(sys.modules, 'd1_fast_side_step',
                        types.SimpleNamespace(_edges=lambda points: [
                            (np.zeros(2), np.array([1., 0.]))]))
    monkeypatch.setitem(sys.modules, 'wheel_legged_control.d1.model',
                        types.SimpleNamespace(JOINT_POSITION_LOW=np.full(16, -1.),
                                              JOINT_POSITION_HIGH=np.full(16, 1.)))
    side._prepare_leg()
    event, = side.drain_macro_events()
    assert side.prepare_calls == 1 and side.ik_calls == 1
    assert event['reason'] == 'body_xy_projected_and_consumed'
    assert event['consumed_action'] == pytest.approx((0., -.008, .5))
    assert event['extra_plan_candidate_qpos23'] == [0.] * 23
    assert side.body_to[1] == pytest.approx(.022)
    assert side.shift_time == .4


def test_previous_native_evidence_is_monotone_and_complete(monkeypatch):
    side = _side(monkeypatch, 'fixed_nonzero_left')
    side.set_interval_evidence(control_index=2,
                               contact_free_5_by_wheel=(True, True, False, True),
                               native_wheel_min_z_5x4=np.ones((5, 4)))
    with pytest.raises(ValueError):
        side.set_interval_evidence(control_index=2,
                                   contact_free_5_by_wheel=(True,) * 4,
                                   native_wheel_min_z_5x4=np.ones((5, 4)))
    with pytest.raises(ValueError):
        side.set_interval_evidence(control_index=3,
                                   contact_free_5_by_wheel=(True,) * 4,
                                   native_wheel_min_z_5x4=np.ones((4, 4)))
    assert side._skill30_interval_evidence['control_index'] == 2


def test_teacher_plan_abort_never_uses_extra_ik_or_latches_macro(monkeypatch):
    side = _side(monkeypatch, 'fixed_nonzero_left')

    def abort_plan(self):
        self.prepare_calls += 1
        self.phase = 'abort_hold'
        self.failure = 'plan_nonfinite'

    monkeypatch.setattr(FakeTeacher, '_prepare_leg', abort_plan)
    side._prepare_leg()
    assert side.prepare_calls == 1 and side.ik_calls == 0
    assert side.drain_macro_events() == ()


def test_airborne_cancel_latches_without_reference_jump(monkeypatch):
    side = _side(monkeypatch, 'fixed_nonzero_left')
    side.phase = 'lift'
    side.leg = 2
    side._skill30_lambda = .5
    side.foot_offset = np.array([0., .002, .021])
    before = side.foot_offset.copy()
    side.cancel()
    assert side.phase == 'lift'
    assert np.array_equal(side.foot_offset, before)
    assert side._skill30_cancel_requested
    assert side._skill30_cancel_at == 0
