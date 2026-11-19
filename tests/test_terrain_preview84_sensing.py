from types import SimpleNamespace

import numpy as np
import pytest

from scripts.terrain_preview84_static import check
from wheel_legged_control.d1 import terrain_preview as preview


def test_first_returns_distinguish_grades_and_reject_invalid_packets():
    result = check()
    assert result['passed'] and result['integration_steps'] == 0
    assert len(result['geometries']) == 9
    assert {v['reason'] for v in result['faults'].values()} == {'unobserved', 'expired', 'time_order'}


def test_held_packet_keeps_capture_time_and_a_gap_cannot_be_interpolated(monkeypatch):
    distances = np.tile(0.51 / np.sin(preview.DEPRESSION_RAD), (2, 1))
    distances[0, 1] = np.nan
    calls = []

    def sample(*args):
        calls.append(1)
        return distances

    monkeypatch.setattr(preview, 'read_ranges', sample)
    plant = SimpleNamespace(model=None, measurement_data=None, base_body_id=0)
    channel = preview.PreviewChannel()

    def observe(t):
        return channel.observe(plant, t, measured_body_pitch_rad=0, measured_body_roll_rad=0,
                               travelled_m=0, forward_track_x_m=0.24, speed_mps=0.35)

    assert observe(0).reason == 'waiting'
    assert observe(0.03).reason == 'waiting'
    a, b = observe(0.04), observe(0.08)
    assert len(calls) == 1 and a.frame is b.frame
    assert a.reason == b.reason == 'unobserved'
    assert b.captured_s == 0 and b.available_s == pytest.approx(0.04)
