"""Scientific-boundary tests for the isolated offline prediction pilot."""

import numpy as np
import pytest

from scripts.world_model51 import (
    features,
    fit_baseline,
    gate,
    read_transition,
    rollout,
    validate_episode,
)


def valid_row():
    qpos = [0.0] * 23
    qpos[3] = 1.0
    before = {"qpos": qpos, "qvel": [0.0] * 22, "time": 0.0}
    after = {"qpos": qpos.copy(), "qvel": [1.0] * 22, "time": 0.01}
    trace = {
        name: [2.0] * 16 for name in ("requested_nm", "limited_nm", "delayed_nm", "applied_nm")
    }
    return {
        "pre_state": before,
        "post_state": after,
        "native_actuator_traces": [dict(trace) for _ in range(5)],
        "info": {
            "controller_record": {"calculation": {"safe_torque_nm": [2.0] * 16}},
            "metrics": {"body_com_vx_mps": 999.0},
        },
    }


def test_post_metrics_are_not_used_as_pre_state():
    before, request, after = read_transition(valid_row())
    np.testing.assert_array_equal(before[23:], 0.0)
    np.testing.assert_array_equal(after[23:], 1.0)
    np.testing.assert_array_equal(request, 2.0)


def test_delayed_actuator_rejected_instead_of_using_future_torque():
    row = valid_row()
    row["native_actuator_traces"][3]["applied_nm"] = [1.0] * 16
    with pytest.raises(ValueError, match="non-passthrough"):
        read_transition(row)


def test_partial_and_reset_crossing_rejected():
    states = np.arange(3.0)[:, None]
    targets = states + 1
    validate_episode(states, targets, np.arange(3), [False, False, True], length=3)
    with pytest.raises(ValueError, match="incomplete"):
        validate_episode(states[:-1], targets[:-1], np.arange(2), [False, True], length=3)
    corrupt = states.copy()
    corrupt[2] = 99
    with pytest.raises(ValueError, match="discontinuity"):
        validate_episode(corrupt, targets, np.arange(3), [False, False, True], length=3)


def test_eight_step_rollout_propagates_its_own_predictions():
    # qvel[0] doubles at every step. Refeeding the initial truth would yield 2,
    # whereas free propagation must reach 256 after eight steps.
    model = {"mean": np.zeros(65), "scale": np.ones(65), "coef": np.zeros((66, 45))}
    model["coef"][23, 23] = 1.0
    initial = np.zeros((1, 45))
    initial[0, 3] = initial[0, 23] = 1.0
    prediction = rollout(model, initial, np.zeros((1, 8, 16)), np.zeros(1, dtype=int))
    assert prediction[0, 23] == 256.0
    assert initial[0, 23] == 1.0


def test_normalization_and_fit_ignore_holdout_values():
    rng = np.random.default_rng(51)
    s = rng.normal(size=(100, 45))
    data = {
        "s": s,
        "u": rng.normal(size=(100, 16)),
        "target": s + 0.01,
        "terrain": np.arange(100) % 4,
    }
    mask = np.arange(100) < 50
    original = fit_baseline(data, mask, 1e-3)
    data["s"][~mask] = 1e12
    data["target"][~mask] = -1e12
    changed = fit_baseline(data, mask, 1e-3)
    for key in original:
        np.testing.assert_array_equal(original[key], changed[key])
    assert features(data["s"], data["u"], data["terrain"]).shape == (100, 65)


def test_terrain_regression_and_near_zero_denominator_fail_closed():
    terrains = ("flat", "bumps", "rough", "ramp")

    def scores(value):
        return {
            h: {"velocity_nrmse": value, "by_terrain": dict.fromkeys(terrains, value)}
            for h in ("1", "8")
        }

    limits = {"h1": 1.0, "h8": 0.9, "terrain_h8": 1.05, "denominator_floor": 1e-8}
    baseline, candidate = scores(1.0), scores(0.8)
    assert gate(baseline, candidate, limits)["passed"]
    candidate["8"]["by_terrain"]["rough"] = 1.051
    assert not gate(baseline, candidate, limits)["passed"]
    assert not gate(scores(0.0), scores(0.0), limits)["passed"]
    for invalid in (float("nan"), float("inf"), -1.0):
        assert not gate(scores(1.0), scores(invalid), limits)["passed"]
