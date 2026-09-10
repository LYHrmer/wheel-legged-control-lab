"""Policy dimensions are not physical actuator dimensions in saved records."""

from copy import deepcopy

import numpy as np
import pytest

from scripts import audit_d1_locomotion as audit


def shared_record():
    episode = {
        "action_mode": "shared2",
        "policy_action_size": 2,
        "physical_action_size": 8,
        "physical_action_schema": "d1-wheel-leg-extension-speed-v1",
        "policy_to_physical_indices": [0, 0, 0, 0, 1, 1, 1, 1],
        "action_schema": "d1-shared-wheel-leg-extension-speed-v1",
    }
    policy = np.array([[0.2, -0.4], [-1, 1]], dtype=np.float32)
    arrays = {
        "qpos": np.zeros((3, 23)),
        "actions": policy,
        "policy_actions": policy.copy(),
        "physical_actions": policy[:, [0, 0, 0, 0, 1, 1, 1, 1]].copy(),
    }
    arguments = {"baseline": "wheel_leg", "action_mode": "shared2", "policy": "trusted.zip"}
    return arrays, episode, arguments


def test_shared_policy_record_uses_physical_eight_for_action_metrics():
    arrays, episode, arguments = shared_record()
    _, physical = audit.audit_action_arrays(arrays, episode, arguments, 2)
    assert physical.shape == (2, 8)
    np.testing.assert_array_equal(
        np.mean(physical**2, axis=1),
        np.mean(arrays["actions"] ** 2, axis=1),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("action_mode", "independent8"),
        ("policy_action_size", 8),
        ("physical_action_size", 2),
        ("physical_action_schema", "d1-v3-body-force-residual-v1"),
        ("policy_to_physical_indices", [0, 1, 0, 1, 0, 1, 0, 1]),
        ("policy_to_physical_indices", [0, 0, 0, 0, True, True, True, True]),
        ("action_schema", "d1-v3-body-force-residual-v1"),
        ("policy_action_size", 2.0),
    ],
)
def test_mapping_metadata_cannot_be_relabelled(field, value):
    arrays, episode, arguments = shared_record()
    episode[field] = value
    with pytest.raises(ValueError):
        audit.audit_action_arrays(arrays, episode, arguments, 2)


@pytest.mark.parametrize("key", ["physical_actions", "policy_actions"])
def test_missing_or_wrong_explicit_arrays_fail(key):
    arrays, episode, arguments = shared_record()
    absent = deepcopy(arrays)
    del absent[key]
    with pytest.raises(ValueError):
        audit.audit_action_arrays(absent, episode, arguments, 2)
    arrays[key][0, 0] += 0.01
    with pytest.raises(ValueError):
        audit.audit_action_arrays(arrays, episode, arguments, 2)


def test_same_policy_size_does_not_make_shared2_a_force_controller():
    arrays, episode, arguments = shared_record()
    arguments["baseline"] = "lqr"
    with pytest.raises(ValueError, match="force baseline"):
        audit.audit_action_arrays(arrays, episode, arguments, 2)


@pytest.mark.parametrize("baseline,dimension", [("wheel_leg", 8), ("lqr", 2), ("mpc", 2)])
def test_historical_records_without_mapping_still_audit(baseline, dimension):
    arrays = {"qpos": np.zeros((3, 23)), "actions": np.zeros((2, dimension))}
    _, physical = audit.audit_action_arrays(arrays, {}, {"baseline": baseline}, 2)
    assert physical.shape == (2, dimension)


def test_keyboard_record_keeps_two_policy_and_eight_physical_channels(tmp_path, monkeypatch):
    from scripts import run_d1_locomotion as runner
    from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv

    action = np.array([0.2, -0.3], dtype=np.float32)

    class Policy:
        def predict(self, observation, deterministic):
            assert observation.shape == (82,) and deterministic
            return action.copy(), None

    # Loading/serialization is separately tested; here exercise the real physics
    # record path with a known policy output, without loading any pickle.
    monkeypatch.setattr(runner, "load_locomotion_policy", lambda *args: Policy())
    model, metadata = tmp_path / "trusted.zip", tmp_path / "trusted.json"
    model.write_bytes(b"test fixture, never deserialized")
    metadata.write_text("{}")
    output = tmp_path / "run"
    summary = runner.run(
        D1LocomotionEnv(episode_seconds=0.03, action_mode="shared2"),
        output,
        policy_path=model,
        metadata_path=metadata,
    )
    assert summary["completed"] and summary["steps"] == 3
    with np.load(output / "states.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["policy_action"], np.tile(action, (3, 1)))
        expected = np.tile(action[[0, 0, 0, 0, 1, 1, 1, 1]], (3, 1))
        np.testing.assert_array_equal(arrays["raw_action"], expected)
        np.testing.assert_array_equal(arrays["applied_action"], expected)
        assert arrays["observation"].shape == (4, 82)
