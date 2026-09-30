"""Arithmetic-only checks of the causal course difference and sealed commands."""
import json
import pytest
from dataclasses import asdict
from recipes20 import select_episode, make_schedule, CASE_SPECS, TRAIN_BUDGET


def test_only_preregistered_slots_differ():
    for i in range(40):
        a, b = select_episode(i, 'A'), select_episode(i, 'B')
        if i < 8 or (i-8) % 8 not in (2, 3):
            assert a.raw_commands == b.raw_commands
        else:
            assert a.raw_commands != b.raw_commands
        assert a.terrain == b.terrain
        assert a.raw_commands[:175] == b.raw_commands[:175]
    assert TRAIN_BUDGET.total_controls == 32768


def test_source_offset_preserves_mapping_without_warmup():
    for arm in ('A', 'B'):
        continued = select_episode(0, arm, source_offset=33)
        reference = select_episode(33, arm)
        assert continued.raw_commands == reference.raw_commands
        assert json.loads(continued.choice_json)['source_episode_index'] == 33
        assert not json.loads(continued.choice_json)['warmup']


def test_new_eval_commands_are_distinct_and_exact():
    hashes = []
    for name, s in CASE_SPECS.items():
        schedule = make_schedule(name)
        assert schedule.control_cap == s['horizon']
        if 'source_case_id' in s:
            continue
        hashes.append(schedule.command_sha256)
        for tick, row in enumerate(schedule.raw_commands):
            assert row.forward_velocity_mps == (s['speed_mps'] if s['drive_start'] <= tick < s['release_tick'] else 0.)
            assert row.yaw_rate_rps == next((y for a,b,y in s['yaw_segments'] if a <= tick < b), 0.)
            assert row.clearance_m == .455
            assert not row.jump_requested and row.lateral_velocity_mps == 0.
    assert len(set(hashes)) == 8
    with pytest.raises(ValueError):
        make_schedule('dev_yaw_left', seed=1)


def test_b_endpoint_and_bumps_cycle_table():
    for cycle, speed in enumerate((.4,.325,.25)):
        bump = select_episode(8 + 8*cycle + 3, 'B')
        assert bump.raw_commands[175].forward_velocity_mps == speed
    left = select_episode(10, 'B').raw_commands
    right = select_episode(26, 'B').raw_commands
    assert left[400].yaw_rate_rps == .3 and left[500].yaw_rate_rps == -.3
    assert right[450].yaw_rate_rps == -.3 and right[550].yaw_rate_rps == .3
