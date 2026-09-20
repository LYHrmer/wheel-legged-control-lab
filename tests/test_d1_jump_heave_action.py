"""Pure boundary and action-contract checks; intentionally no simulation imports."""

import json

import numpy as np
import pytest

from scripts.d1_jump_heave_action import heave_action_metadata, map_heave_action


@pytest.mark.parametrize("start", [175, 225])
@pytest.mark.parametrize("offset,active", [(-1, False), (0, True), (119, True), (120, False)])
@pytest.mark.parametrize("value,clipped", [(-2.0, -1.0), (0.5, 0.5), (2.0, 1.0)])
def test_fixed_window_and_physical_channel_order(start, offset, active, value, clipped):
    result = map_heave_action([value], executed_tick=start + offset, request_tick=start)
    expected = [clipped] * 4 + [0.0] * 4 if active else [0.0] * 8
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float64 and result.flags.owndata
    assert not np.signbit(result[4:]).any()
    if not active:
        assert not np.signbit(result).any()


def test_no_request_all600_ticks_and_owned_results():
    action = np.array([-1.0])
    previous = np.ones(8)
    for tick in range(600):
        result = map_heave_action(action, executed_tick=tick, request_tick=None)
        assert result.tobytes() == np.zeros(8).tobytes()
        assert not np.shares_memory(result, action)
        result[:] = 7
    np.testing.assert_array_equal(action, [-1.0])
    np.testing.assert_array_equal(previous, np.ones(8))


@pytest.mark.parametrize("start", [None, 175])
@pytest.mark.parametrize("action", [0.0, [], [0.0, 0.0], [[0.0]], [True], [np.bool_(False)],
                                    [float("nan")], [float("inf")], [-float("inf")],
                                    ["0.5"], [1j], [None]])
def test_malformed_actions_rejected_even_when_inactive(start, action):
    with pytest.raises((TypeError, ValueError)):
        map_heave_action(action, executed_tick=0, request_tick=start)


@pytest.mark.parametrize("value", [-1, 0.0, True, np.bool_(False), float("nan"), "2"])
@pytest.mark.parametrize("field", ["executed_tick", "request_tick"])
def test_invalid_ticks(value, field):
    kwargs = {"executed_tick": 0, "request_tick": None, field: value}
    with pytest.raises((TypeError, ValueError)):
        map_heave_action([0.5], **kwargs)


def test_numpy_integers_repeated_calls_and_gate_end_history():
    before = map_heave_action([1], executed_tick=np.int64(294), request_tick=np.int32(175))
    again = map_heave_action([1], executed_tick=294, request_tick=175)
    after = map_heave_action([1], executed_tick=295, request_tick=175)
    assert before.tobytes() == again.tobytes()
    np.testing.assert_array_equal(before, [1, 1, 1, 1, 0, 0, 0, 0])
    assert float(np.sum((after - before) ** 2)) == 4.0
    # A pure map leaves its caller's old physical action intact at gate exit.
    np.testing.assert_array_equal(before, again)


def test_metadata_identity_units_matrix_and_fresh_copies():
    metadata = heave_action_metadata()
    assert metadata["action_schema"] == "d1-jump-shared-heave-window-v1"
    assert metadata["policy_action_size"] == 1 and metadata["physical_action_size"] == 8
    assert metadata["leg_action_scale_m"] == 0.04
    assert metadata["wheel_action_scale_rad_s"] == 4.0
    np.testing.assert_array_equal(metadata["active_embedding_matrix"],
                                  np.array([1, 1, 1, 1, 0, 0, 0, 0]).reshape(8, 1))
    assert metadata["gate"]["window_ticks"] == 120
    assert metadata["gate"]["clears_controller_or_action_memory"] is False
    snapshot = json.dumps(metadata, sort_keys=True, allow_nan=False)
    metadata["active_embedding_matrix"][0][0] = 0
    metadata["gate"]["window_ticks"] = 1
    assert json.dumps(heave_action_metadata(), sort_keys=True) == snapshot
