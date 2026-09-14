"""Real-plant parity and before/after command timing for the heading probe."""
import json

import numpy as np

from scripts.probe_d1_heading_hold import run_episode
from scripts.run_d1_wheel_common_mean_study import case_command
from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv
from wheel_legged_control.d1.locomotion_terrain import D1LocomotionTerrainConfig


def case():
    return {
        'seed': 55101, 'episode_seconds': .2,
        'terrain': {'layout': 'flat', 'slope_deg': 0., 'cross_slope_deg': 0.,
                    'ripple_amplitude_m': 0., 'step_height_m': 0.},
        'command': {'target_forward_mps': .25, 'target_yaw_rps': 0.,
                    'settle_seconds': 0., 'ramp_seconds': .1, 'height_m': .455},
    }


def test_original_probe_matches_unmodified_control_loop_every_state(tmp_path):
    spec = case()
    summary = run_episode(tmp_path/'probe', spec, mode='original')
    assert summary['actual_transitions'] == 20
    env = D1LocomotionEnv(
        baseline='wheel_leg', action_mode='independent8', episode_seconds=.2,
        terrain=D1LocomotionTerrainConfig(**spec['terrain']),
        command_source=case_command(spec['command']),
    )
    try:
        obs, _ = env.reset(seed=spec['seed'])
        with np.load(tmp_path/'probe/states.npz') as saved:
            for tick in range(21):
                np.testing.assert_array_equal(saved['qpos'][tick], env.plant.data.qpos)
                np.testing.assert_array_equal(saved['qvel'][tick], env.plant.data.qvel)
                np.testing.assert_array_equal(saved['observations'][tick], obs)
                if tick < 20:
                    obs, *_ = env.step(np.zeros(8))
    finally:
        env.close()


def test_hold_trace_distinguishes_actor_command_user_goal_and_terminal_tick(tmp_path):
    spec = case()
    summary = run_episode(tmp_path/'hold', spec, mode='hold')
    rows = [json.loads(line) for line in (tmp_path/'hold/trace.jsonl').read_text().splitlines()]
    assert len(rows) == summary['actual_transitions'] == 20
    assert rows[-1]['truncated'] and not rows[-1]['terminated']
    with np.load(tmp_path/'hold/states.npz') as saved:
        assert saved['observations'].dtype == np.float32
        for tick, row in enumerate(rows):
            servo = row['servo_command_before']['yaw_rate_rps']
            np.testing.assert_allclose(saved['observations'][tick, 39], servo/.5, atol=1e-8)
            assert row['reference_before']['simulation_time_s'] == tick*.01
            assert row['reference_heading_after_rad'] == 0.
            assert row['user_yaw_rate_error_rps'] == row['metrics']['yaw_rate_rps']
            assert row['metrics']['yaw_rate_error_rps'] == row['metrics']['yaw_rate_rps']-servo
