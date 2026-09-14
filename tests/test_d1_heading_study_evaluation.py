"""The new zero-policy evaluator preserves the tested external-servo physics."""
import gzip
import json

import numpy as np
import pytest

from scripts.evaluate_d1_heading_study import run_episode
from scripts.probe_d1_heading_hold import run_episode as run_old_probe


def test_zero_evaluation_matches_original_hold_trajectory_and_records_new_reward(tmp_path):
    case = {
        'seed': 55101, 'episode_seconds': .2,
        'terrain': {'layout': 'flat', 'slope_deg': 0., 'cross_slope_deg': 0.,
                    'ripple_amplitude_m': 0., 'step_height_m': 0.},
        'command': {'target_forward_mps': .25, 'target_yaw_rps': 0.,
                    'settle_seconds': 0., 'ramp_seconds': .1, 'height_m': .455},
    }
    original = run_old_probe(tmp_path/'old', case, mode='hold')
    current = run_episode(tmp_path/'new', case)
    assert current['actual_transitions'] == original['actual_transitions'] == 20
    with np.load(tmp_path/'old/states.npz') as old, np.load(tmp_path/'new/states.npz') as new:
        for key in ('qpos', 'qvel', 'applied_actions'):
            np.testing.assert_array_equal(old[key], new[key])
        np.testing.assert_array_equal(old['observations'], new['observations'][:, :82])
        assert new['observations'].shape == (21, 85)
    with gzip.open(tmp_path/'new/trace.jsonl.gz', 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 20
    assert current['servo_command_return'] == original['servo_command_return']
    assert current['task_return'] == pytest.approx(
        current['servo_command_return']+current['heading_goal_return'], abs=1e-14,
    )
    assert all(row['reward'] == sum(row['reward_terms'].values()) for row in rows)
    assert rows[-1]['truncated'] and not rows[-1]['terminated']
