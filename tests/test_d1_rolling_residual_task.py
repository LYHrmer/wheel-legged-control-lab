"""Pure rolling-task boundaries; no environment, engine, or model imports."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from scripts.d1_rolling_residual_task import (
    RollingEpisodeSpec,
    expand_policy_action,
    raw_command_at_tick,
    rolling_task_definition,
)


def test_action_rejects_boolean_complex_string_and_object_values():
    for action in ([True], [1j], ["0.1"], np.array([0.1], dtype=object)):
        with pytest.raises(TypeError):
            expand_policy_action(action, 0.2)


def test_action_requires_one_dimensional_length_one():
    for action in (0.0, np.array(0.0), [], [0.0, 0.0], [[0.0]]):
        with pytest.raises(ValueError):
            expand_policy_action(action, 0.2)


def test_nonfinite_action_is_rejected_even_when_gate_is_closed():
    for value in (float("nan"), float("inf"), -float("inf")):
        for forward in (0.0, 0.2):
            with pytest.raises(ValueError):
                expand_policy_action([value], forward)


def test_raw_forward_command_requires_finite_real_scalar():
    for forward in (True, np.bool_(False), "0.2", 0.2j, [0.2]):
        with pytest.raises(TypeError):
            expand_policy_action([0.0], forward)
    for forward in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            expand_policy_action([0.0], forward)


def test_drive_clips_policy_without_losing_raw_action_or_adding_wheel_residual():
    for raw, clipped, leg in ((-3.0, -1.0, -0.25), (-1.0, -1.0, -0.25),
                              (0.5, 0.5, 0.125), (1.0, 1.0, 0.25),
                              (3.0, 1.0, 0.25)):
        action = np.array([raw], dtype=np.float64)
        before = action.tobytes()
        result = expand_policy_action(action, 0.25)
        assert result.policy_scalar == raw
        assert result.clipped_scalar == clipped
        assert result.forward_gate_active is True
        assert result.physical_action == (leg, leg, leg, leg, 0.0, 0.0, 0.0, 0.0)
        assert action.tobytes() == before
        assert max(abs(v * 0.04) for v in result.physical_action[:4]) <= 0.01


def test_gate_uses_exact_raw_zero_and_retains_nonzero_policy_evidence():
    for forward in (0.0, -0.0):
        result = expand_policy_action([-2.0], forward)
        assert result.policy_scalar == -2.0 and result.clipped_scalar == -1.0
        assert result.forward_gate_active is False
        assert np.asarray(result.physical_action).tobytes() == np.zeros(8).tobytes()
    for forward in (np.nextafter(0.0, 1.0), -0.2):
        result = expand_policy_action([1.0], forward)
        assert result.forward_gate_active is True
        assert result.physical_action[:4] == (0.25,) * 4


def test_schedule_has_exactly_800_executed_drive_ticks_and_prepared_terminal():
    for speed, onset in ((0.18, 200), (0.18, 250), (0.20, 200),
                         (0.20, 250), (0.225, 200), (0.225, 250),
                         (0.20, 175), (0.25, 175)):
        spec = RollingEpisodeSpec(speed, onset, onset + 800)
        rows = [raw_command_at_tick(spec, tick) for tick in range(1201)]
        assert rows == [raw_command_at_tick(spec, tick) for tick in range(1201)]
        assert sum(row["forward_velocity_mps"] != 0.0 for row in rows[:1200]) == 800
        assert rows[onset - 1]["forward_velocity_mps"] == 0.0
        assert rows[onset]["forward_velocity_mps"] == speed
        assert rows[onset + 799]["forward_velocity_mps"] == speed
        assert rows[onset + 800]["forward_velocity_mps"] == 0.0
        assert rows[1200] == {"tick": 1200, "forward_velocity_mps": 0.0,
                              "yaw_rate_rps": 0.0, "world_height_m": 0.455}
        assert all(row["yaw_rate_rps"] == 0.0 and row["world_height_m"] == 0.455
                   for row in rows)


def test_spec_and_command_reject_invalid_speed_tick_duration_and_obstacle_identity():
    for speed in (True, "0.2", 0.2j, 0.0, -0.1, 0.250001, float("nan")):
        with pytest.raises((TypeError, ValueError)):
            RollingEpisodeSpec(speed, 200, 1000)
    for start, stop in ((True, 801), (200.0, 1000), (200, 999),
                        (200, 1001), (-1, 799), (401, 1201)):
        with pytest.raises((TypeError, ValueError)):
            RollingEpisodeSpec(0.2, start, stop)
    for obstacle in (1, 0, np.bool_(True), "box"):
        with pytest.raises(TypeError):
            RollingEpisodeSpec(0.2, 200, 1000, obstacle)
    spec = RollingEpisodeSpec(0.2, 200, 1000)
    for tick in (True, 1.0, -1, 1201):
        with pytest.raises((TypeError, ValueError)):
            raw_command_at_tick(spec, tick)
    with pytest.raises(TypeError):
        raw_command_at_tick(spec.as_dict(), 200)


def test_task_definition_records_constants_and_gate_and_returns_independent_json():
    definition = rolling_task_definition()
    encoded = json.loads(json.dumps(definition, allow_nan=False))
    assert encoded["policy_action_size"] == 1 and encoded["physical_action_size"] == 8
    assert encoded["policy_to_physical_indices"] == [0, 0, 0, 0, None, None, None, None]
    assert encoded["constant_physical_channels"] == {"wheel_indices": [4, 5, 6, 7],
                                                    "value": 0.0}
    assert encoded["leg_normalized_scale"] == 0.25
    assert encoded["max_leg_extension_residual_m"] == 0.01
    assert encoded["forward_gate"] == "raw_forward_velocity_mps != 0.0"
    assert encoded["observation_truth_appended"] is False
    assert encoded["reward_changes_from_heading_parent"] is False
    definition["constant_physical_channels"]["wheel_indices"].clear()
    assert rolling_task_definition() == encoded


def test_episode_and_action_records_are_immutable_and_preserve_physical_trace():
    spec = RollingEpisodeSpec(0.25, 175, 975, False)
    with pytest.raises(FrozenInstanceError):
        spec.forward_velocity_mps = 0.2
    result = expand_policy_action([-0.5], 0.25)
    with pytest.raises(FrozenInstanceError):
        result.policy_scalar = 0.0
    row = json.loads(json.dumps(result.as_dict(), allow_nan=False))
    assert row == {"policy_scalar": -0.5, "clipped_scalar": -0.5,
                   "forward_gate_active": True,
                   "physical_action": [-0.125, -0.125, -0.125, -0.125, 0.0, 0.0, 0.0, 0.0]}
    row["physical_action"][0] = 0.0
    assert result.physical_action[0] == -0.125


def test_action_record_accepts_actual_residual_and_rejects_forged_physical_trace():
    from scripts.d1_rolling_residual_scoring import validate_rolling_action_record

    row = {"policy_action": [0.5], "clipped_policy_scalar": 0.5,
           "forward_gate_active": True,
           "action": [0.125] * 4 + [0.0] * 4,
           "applied_action": [0.125] * 4 + [0.0] * 4}
    validate_rolling_action_record(row, 0.25)
    for field, value in (("action", [0.0] * 8),
                         ("applied_action", [0.125] * 4 + [0.01, 0.0, 0.0, 0.0]),
                         ("clipped_policy_scalar", 0.0),
                         ("forward_gate_active", False)):
        forged = deepcopy(row)
        forged[field] = value
        with pytest.raises(ValueError):
            validate_rolling_action_record(forged, 0.25)
    with pytest.raises(ValueError):
        validate_rolling_action_record(row, 0.0)


def test_command_record_requires_real_speed_schedule_and_world_height_conversion():
    from scripts.d1_rolling_residual_scoring import validate_rolling_command_record

    spec = RollingEpisodeSpec(0.25, 175, 975)
    row = {"tick": 175, "forward_velocity_mps": 0.25, "yaw_rate_rps": 0.0,
           "raw_world_height_m": 0.455, "prepared_world_height_m": 0.455,
           "ground_height_m": 0.015, "motion_clearance_m": 0.455 - 0.015}
    validate_rolling_command_record(row, 175, spec)
    for field, value in (("forward_velocity_mps", 0.2), ("tick", 174),
                         ("yaw_rate_rps", 0.01), ("raw_world_height_m", 0.44),
                         ("prepared_world_height_m", 0.47),
                         ("motion_clearance_m", 0.455)):
        forged = dict(row)
        forged[field] = value
        with pytest.raises(ValueError):
            validate_rolling_command_record(forged, 175, spec)
    held = {**row, "tick": 975, "forward_velocity_mps": 0.0}
    validate_rolling_command_record(held, 975, spec)
