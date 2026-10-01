"""Two tiny arithmetic fixtures; no saved archive, reader or simulator runs."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_saved31 import goal_rate_fields, phase_before_counts, task_windows


def test_compute_before_phase_counts_transition_tick_once():
    rows = [
        {'info': {'controller_record': {'diagnostic_before': {'phase': 'shift'},
                                        'diagnostic_after': {'phase': 'unload'}}}},
        {'info': {'controller_record': {'diagnostic_before': {'phase': 'unload'},
                                        'diagnostic_after': {'phase': 'lift'}}}},
        {'info': {'controller_record': {'diagnostic_before': {'phase': 'lift'},
                                        'diagnostic_after': {'phase': 'lift'}}}},
    ]
    counts = phase_before_counts(rows, [0, 1, 2])
    assert counts == {'shift': 1, 'unload': 1, 'lift': 1}
    assert sum(counts.values()) == 3


def test_cancel_handoff_retention_has_no_completed_goal_speed():
    task = {'side_control_window': [200, 395],
            'retention_control_window': [395, 795],
            'is_cancellation_case': True,
            'side_duration_s': 1.95,
            'signed_lateral_m': -.004,
            'actual_net_per_cycle_mps': -.004 / 1.95}
    assert task_windows('cancel_left', task, list(range(200, 395)), 795) == (200, 395)
    assert goal_rate_fields(task) == {'nominal_goal_m': None,
                                      'nominal_goal_per_cycle_mps': None,
                                      'actual_net_per_cycle_mps': None}
    with pytest.raises(ValueError):
        task_windows('cancel_left', task, list(range(200, 395)), 794)
