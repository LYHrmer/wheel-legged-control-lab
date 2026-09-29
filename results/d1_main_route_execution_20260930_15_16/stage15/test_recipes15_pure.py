"""Synthetic recipe identity checks; no plant, model, policy or physics."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

W = Path(__file__).resolve().parent.parent
for directory in (W / "course_impl08", W / "rl11", W / "continuation15", W):
    sys.path.insert(0, str(directory))

import rl16_heldout_score_11 as old_score
import score15 as score
import short_heldout_recipe_11 as old_heldout
import short_plan_recipe_11 as old_plan
from recipes15 import (
    FLOOR_SEED, MEASUREMENT_SEED, ORDER, PLAN_SCHEMA, PPO_SEED,
    SELECTION_SEED, TASKS, TRAIN_BUDGET, FloorSchedule, build_plan,
    floor_schedule, heldout_schedule, recipe_table, select_episode,
)
from short_heldout_11 import WorldUprightHeldoutSchedule


def test_budget_and_selector_use_local_indices_with_only_seed_changed():
    assert (PPO_SEED, SELECTION_SEED, MEASUREMENT_SEED) == (151001, 151002, 151003)
    assert TRAIN_BUDGET.as_dict()["optimizer_steps"] == 256
    assert TRAIN_BUDGET.as_dict()["rollouts"] == 16
    plan = build_plan(source_sha256="a" * 64)
    assert plan.schema == PLAN_SCHEMA
    assert plan.source_sha256 == "a" * 64

    # Clone the old function's globals for this synthetic comparison. The
    # imported 11-S module and its frozen constants are never modified.
    old_globals = dict(vars(old_plan), SELECTION_SEED=SELECTION_SEED)
    reference = types.FunctionType(
        old_plan.select_episode.__code__, old_globals, "old_rule_with_new_seed",
    )
    for index in range(17):
        actual = plan.episode(index)
        expected = reference(index)
        assert actual.terrain == expected.terrain
        assert actual.raw_commands == expected.raw_commands
        actual_choice = json.loads(actual.choice_json)
        expected_choice = json.loads(expected.choice_json)
        for key in ("phase", "slot", "draw_order", "speed_mps",
                    "yaw_amplitude_rps", "yaw_first_sign", "yaw_windows"):
            assert actual_choice[key] == expected_choice[key]
        assert actual.selection_seed == SELECTION_SEED
        if index < 8:
            assert actual_choice["phase"] == "first_eight_fixed_flat"
    assert old_plan.SELECTION_SEED == 88622
    with pytest.raises(ValueError):
        select_episode(True)


def test_six_schedules_have_old_raw_patterns_and_new_seeds():
    assert ORDER == old_heldout.ORDER
    assert len(TASKS) == 6
    for index, case_id in enumerate(ORDER):
        new = heldout_schedule(case_id)
        old = old_heldout.heldout_schedule(case_id)
        task = TASKS[index]
        assert type(new) is WorldUprightHeldoutSchedule
        assert (new.terrain, new.spawn_position_m, new.raw_commands) == (
            old.terrain, old.spawn_position_m, old.raw_commands,
        )
        assert new.seed == task.seed == 151101 + index
        assert new.control_cap == task.horizon
        assert score.case_windows(case_id) == old_score.case_windows(case_id)
        assert score.HELDOUT_CASES[case_id][3] == task.seed
    with pytest.raises(ValueError):
        heldout_schedule("floor_0p4_600")


def test_floor_is_exact_600_control_heldout_interface():
    floor = floor_schedule()
    assert isinstance(floor, WorldUprightHeldoutSchedule)
    assert isinstance(floor, FloorSchedule)
    assert (floor.case_id, floor.seed, floor.control_cap) == (
        "floor_0p4_600", FLOOR_SEED, 600,
    )
    assert all(row.forward_velocity_mps == 0.0 for row in floor.raw_commands[:175])
    assert all(row.forward_velocity_mps == .4 for row in floor.raw_commands[175:425])
    assert all(row.forward_velocity_mps == 0.0 for row in floor.raw_commands[425:])
    assert floor.command_source(599, 5.99) == floor.raw_commands[-1]
    with pytest.raises(ValueError):
        floor.command_source(600, 6.0)
    with pytest.raises(ValueError):
        replace(floor, raw_commands=floor.raw_commands[:-1])
    with pytest.raises(ValueError):
        replace(floor, seed=88701)


def test_recipe_table_is_json_ready_and_score_rejects_old_seed():
    table = recipe_table(source_sha256="b" * 64)
    assert json.loads(json.dumps(table, allow_nan=False)) == table
    assert table["total_new_controls"] == 63368
    assert table["normal_native_limit"] == 316840
    assert table["cold_workers"] == 3
    assert table["heldout_order"] == list(ORDER)
    assert table["floor"]["drive"] == [175, 425]

    case_id = ORDER[0]
    row = {
        "case_id": case_id, "actor": "zero", "terrain": "flat",
        "task_schema": score.TASK_SCHEMA,
        "reference_schema": score.REFERENCE_SCHEMA,
        "reward_schema": score.REWARD_SCHEMA,
        "seed": 88701,
    }
    with pytest.raises(ValueError, match="identity"):
        score.score_case(row)
    row["seed"] = 151101
    with pytest.raises(KeyError, match="completed_controls"):
        score.score_case(row)
