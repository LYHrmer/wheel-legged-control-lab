"""One pure state-machine fixture for C30's new airborne reference clock."""
from __future__ import annotations

import sys
import types

import numpy as np

from side_skill30 import make_side30


class _Teacher:
    def __init__(self, plant):
        self.plant = plant
        self.dt = .01
        self.vadr = np.arange(16).reshape(4, 4)
        self.cfg = types.SimpleNamespace(
            lift_time_s=.26, lift_height_m=.035, lift_clearance_m=.01,
            liftoff_extra_s=.20, margin_gate_m=.02,
            margin_fault_m=-.004, margin_fault_time_s=.10)
        self.reset()

    def reset(self):
        self.phase = 'idle'
        self.failure = None
        self.leg = None

    @property
    def status(self):
        return {'active': self.phase not in ('idle', 'done', 'failed')}

    def _contacts(self):
        return np.array([True, True, False, True])

    def _geometry(self, data):
        points = np.zeros((4, 3))
        points[2, 2] = .03
        return points, np.zeros(4)

    def _com(self, data):
        return np.array([.05, 0., .455])

    def _enter(self, phase):
        self.phase = phase
        self.phase_time = 0.

    def _abort(self, reason):
        self.failure = reason
        self.phase = 'abort_land'

    def cancel(self):
        raise AssertionError('new airborne cancel must not call old abrupt abort')


def test_airborne_clock_gates_cancel_and_clearance(monkeypatch):
    data = types.SimpleNamespace(qvel=np.zeros(16))
    velocity_queries = []

    def base_origin_velocity():
        velocity_queries.append(1)
        return np.zeros(3)

    plant = types.SimpleNamespace(measurement_data=data, base_rpy=np.zeros(3),
        base_position=np.array([.05, 0., .455]),
        base_origin_velocity=base_origin_velocity)
    monkeypatch.setitem(sys.modules, 'hybrid_adapter27',
        types.SimpleNamespace(make_fast_side27=lambda target, arm: _Teacher(target)))
    monkeypatch.setitem(sys.modules, 'd1_fast_side_step',
        types.SimpleNamespace(G=9.81,
            _edges=lambda points: [(np.zeros(2), np.array([1., 0.]))]))
    clock = {'i': 100}
    binding = {'kinematics24': {},
        'wheel_index_by_body_id': {10: 0, 11: 1, 12: 2, 13: 3}}
    side = make_side30(plant, 'fixed_nonzero_left', binding, lambda: clock['i'])
    side.phase = 'lift'
    side.leg = 2
    side.phase_time = .13
    side.time = 1.
    side._skill30_lambda = .5
    side.foot_offset = np.zeros(3)
    side.delta = np.array([0., .03, 0.])
    side.swing_time = .30
    side.body_accel = np.zeros(3)
    side.lift_peaks = np.zeros(4)
    side.missing_support_time = 0.
    side.margin_fault_time = 0.
    minz = {'value': .02}
    side._wheel_shape_min30 = lambda data: np.array([.02, .02, minz['value'], .02])

    # Current shape alone cannot authorize motion: the prior complete 5T is absent.
    side._advance()
    assert not side._skill30_horizontal_started
    assert side.foot_offset[1] == 0.

    native = np.full((5, 4), .02)
    side.set_interval_evidence(control_index=100,
        contact_free_5_by_wheel=(True,) * 4, native_wheel_min_z_5x4=native)
    clock['i'] = 101
    side._advance()
    assert side._skill30_horizontal_started
    assert side._skill30_horizontal_start_control == 101
    assert side._skill30_horizontal_elapsed == 0.
    assert side.foot_offset[1] == 0.
    first_z = float(side.foot_offset[2])

    side.set_interval_evidence(control_index=101,
        contact_free_5_by_wheel=(True,) * 4, native_wheel_min_z_5x4=native)
    clock['i'] = 102
    side._advance()
    assert side._skill30_horizontal_elapsed == side.dt
    assert side.foot_offset[1] > 0.
    assert side.foot_offset[2] > first_z
    before_cancel = side.foot_offset.copy()

    side.cancel()
    assert side.phase == 'lift'
    assert np.array_equal(side.foot_offset, before_cancel)
    side.set_interval_evidence(control_index=102,
        contact_free_5_by_wheel=(True,) * 4, native_wheel_min_z_5x4=native)
    clock['i'] = 103
    side._advance()
    assert side.foot_offset[1] > before_cancel[1]
    assert side.foot_offset[2] > before_cancel[2]

    side.set_interval_evidence(control_index=103,
        contact_free_5_by_wheel=(True,) * 4, native_wheel_min_z_5x4=native)
    minz['value'] = .012
    clock['i'] = 104
    side._advance()
    assert side.phase == 'abort_land'
    assert side.failure == 'horizontal_actual_whole_wheel_clearance_lost'
    assert len(velocity_queries) == 5  # one frozen _advance query per tick
