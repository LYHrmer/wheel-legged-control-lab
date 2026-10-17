"""NumPy/fake actor checks; never import the heavy worker or execute physics."""
import ast
import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load_file("test_gated_inference32", ROOT/"scripts/b22_inference32.py")
residual = load_file("pure_residual_for32", ROOT/"runtime/d1_b22/research/continuation18/residual18.py")


def numeric_actor():
    source = ROOT/"runtime/d1_b22/research/run_rl16_training_08.py"
    node = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "_numeric_actor_action")
    namespace = {"Any": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102 - only the pinned pure numeric helper
    return namespace["_numeric_actor_action"]


def proof():
    receipt = {"actor_layers": copy.deepcopy(helper.ACTOR_LAYERS),
               "critic_layers": copy.deepcopy(helper.ACTOR_LAYERS),
               "observation_space": {"dtype": "float32", "low": -5., "high": 5., "shape": [99]},
               "action_space": {"dtype": "float32", "low": -1., "high": 1., "shape": [16]},
               "loaded_policy_state_sha256": helper.POLICY_STATE_SHA,
               "matches_training_final_policy_state": True, "probe_rows": 32,
               "probe_actions_byte_exact": True, "verified_without_engine_or_reset": True,
               "metadata_sha256": helper.METADATA_SHA}
    manifest = {"files": {"final_model.zip": {"sha256": helper.MODEL_SHA}},
                "metadata_sha256": helper.METADATA_SHA, "reload_verification": copy.deepcopy(receipt)}
    return manifest, receipt


class StatelessActor:
    def __init__(self):
        self.calls = 0

    def predict(self, observation, *, deterministic):
        assert deterministic is True
        self.calls += 1  # API accounting only; the returned action has no memory.
        obs = np.asarray(observation)
        return np.tanh(obs[:16]+.2*obs[63:79]).astype(np.float32), None


def selector(actor=None):
    actor = StatelessActor() if actor is None else actor
    manifest, receipt = proof()
    return helper.GatedInference32("B", actor, numeric_actor_action=numeric_actor(),
                                   manifest=manifest, strict_load=receipt), actor


def inputs(gate=False):
    return {"joint_position_rad": np.tile([0., .8, -1.5, 0.], 4),
            "joint_velocity_rad_s": np.tile([.03, -.05, .04, 1.], 4),
            "nominal_joint_target_rad": np.tile([0., .8, -1.5, 0.], 4),
            "nominal_wheel_speed_rad_s": np.array([2., 1.8, 2.1, 1.9]),
            "support_torque_nm": np.zeros(16), "base_rotation_world_from_body": np.eye(3),
            "foot_jacobian_world": np.zeros((4, 3, 4)), "body_com_forward_mps": .1,
            "servo_forward_mps": .2, "controller_variant": "combined", "action_enabled": gate,
            "leg_longitudinal_damping_active": True, "body_common_p_active": True}


def test_import_does_not_import_model_engine_or_worker():
    path = ROOT/"scripts/b22_inference32.py"
    code = (f'import importlib.util,sys;s=importlib.util.spec_from_file_location("light32",{str(path)!r});'
            'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
            'assert not {"torch","mujoco","stable_baselines3","run_gui26"}&sys.modules.keys()')
    done = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=5, check=False)
    assert done.returncode == 0, done.stderr


def test_gate_off_skips_prediction_but_does_not_fabricate_policy_output():
    gate, actor = selector()
    observation = np.linspace(-1., 1., 99, dtype=np.float32)
    original = observation.copy()
    action, predicted, row = gate.select(observation, action_enabled=False)
    assert actor.calls == 0 and predicted is False
    assert action.shape == (16,) and action.dtype == np.float32 and not np.any(action)
    assert row["raw_policy_action16"] is None and row["policy_action_observed"] is False
    assert row["policy_output_kind"] == "NOT_OBSERVED" and row["reason"] == "GATE_OFF"
    assert np.array_equal(observation, original)


def test_gate_on_delegates_to_unchanged_numeric_actor_exactly():
    gate, actor = selector()
    observation = np.linspace(-.5, .7, 99, dtype=np.float32)
    expected, _ = numeric_actor()("final_policy", StatelessActor(), observation)
    action, predicted, row = gate.select(observation, action_enabled=True)
    assert np.array_equal(action, expected) and actor.calls == 1 and predicted is True
    assert row["raw_policy_action16"] == action.tolist() and row["policy_action_observed"] is True
    assert row["reason"] == "PREDICTED"


def test_720_enabled_880_disabled_controls_save_exactly_880_control_predictions():
    gate, actor = selector()
    for tick in range(1600):
        enabled = 175 <= tick <= 894
        _, _, row = gate.select(np.zeros(99, dtype=np.float32), action_enabled=enabled)
        assert row["predict_called"] is enabled
    counts = gate.summary()
    assert actor.calls == counts["control_predictions"] == 720
    assert counts["control_decisions"] == 1600 and counts["gate_off_skips"] == 880
    assert counts["required_loader_probe_predict_calls"] == 1 and counts["expected_total_model_predict_calls"] == 721
    assert counts["full_closed_loop_equivalence"] == "NOT_VALIDATED"


def test_zero_actor_is_not_miscounted_as_gate_off_optimization():
    gate = helper.GatedInference32("zero", None, numeric_actor_action=numeric_actor())
    for enabled in (True, False):
        action, predicted, row = gate.select(np.zeros(99), action_enabled=enabled)
        assert not np.any(action) and predicted is False and row["reason"] == "ZERO_ACTOR"
    assert gate.summary()["gate_off_skips"] == 0 and gate.summary()["zero_actor_decisions"] == 2
    assert gate.summary()["expected_total_model_predict_calls"] == 0


@pytest.mark.parametrize("fault", ["missing", "model", "metadata", "metadata_missing", "layers", "state", "probe", "shape"])
def test_skipping_is_restricted_to_bound_strict_feedforward_proof(fault):
    manifest, receipt = proof()
    if fault == "missing": receipt = None
    elif fault == "model": manifest["files"]["final_model.zip"]["sha256"] = "f"*64
    elif fault == "metadata": receipt["metadata_sha256"] = "f"*64
    elif fault == "metadata_missing":
        receipt.pop("metadata_sha256"); manifest.pop("metadata_sha256")
        manifest["reload_verification"].pop("metadata_sha256")
    elif fault == "layers": receipt["actor_layers"] = [{"type": "LSTM"}]
    elif fault == "state": receipt["loaded_policy_state_sha256"] = "f"*64
    elif fault == "probe": receipt["probe_actions_byte_exact"] = False
    else: receipt["observation_space"]["shape"] = [100]
    with pytest.raises(ValueError):
        helper.GatedInference32("B", StatelessActor(), numeric_actor_action=numeric_actor(),
                               manifest=manifest, strict_load=receipt)


@pytest.mark.parametrize("gate_value", [0, 1, np.bool_(False), None, "false"])
def test_gate_is_the_original_explicit_boolean_not_a_threshold_guess(gate_value):
    gate, actor = selector()
    with pytest.raises(ValueError, match="gate must be bool"):
        gate.select(np.zeros(99), action_enabled=gate_value)
    assert actor.calls == 0 and gate.summary()["control_decisions"] == 0


@pytest.mark.parametrize("bad_obs", [np.zeros(98), np.full(99, np.nan), np.full(99, 5.01),
                                      np.array(["x"]*99), np.zeros(99, dtype=bool),
                                      np.full(99, np.iinfo(np.int64).min, dtype=np.int64)])
def test_invalid_observation_still_fails_while_gate_is_closed(bad_obs):
    gate, actor = selector()
    with pytest.raises(ValueError, match="observation99"):
        gate.select(bad_obs, action_enabled=False)
    assert actor.calls == 0 and gate.summary()["control_decisions"] == 0


def test_recurrent_output_state_is_rejected_by_real_numeric_actor_seam():
    class Recurrent(StatelessActor):
        def predict(self, observation, *, deterministic):
            return np.zeros(16), np.zeros(3)
    gate, _ = selector(Recurrent())
    with pytest.raises(RuntimeError, match="invalid 16D action"):
        gate.select(np.zeros(99), action_enabled=True)
    assert gate.summary()["control_decisions"] == 0


def test_skip_only_prediction_preserves_pure_controller_torque_PI_z_and_effective_action():
    baseline, optimized = StatelessActor(), StatelessActor()
    gate, _ = selector(optimized)
    numeric = numeric_actor()
    memories = [(np.array([.1, -.2, .15, -.05]), 1.2, np.zeros(16)) for _ in range(2)]
    for tick, enabled in enumerate([False]*3+[True]*7+[False]*9+[True]*4):
        observation = np.linspace(-.3, .4, 99, dtype=np.float32)
        observations = []
        for integral, _, previous in memories:
            current = observation.copy(); current[59:63] = integral/4.; current[63:79] = previous
            observations.append(current)
        assert np.array_equal(*observations)
        old, _ = numeric("final_policy", baseline, observations[0])
        new, _, _ = gate.select(observations[1], action_enabled=enabled)
        calculations = [residual.compute_residual18(inputs(enabled), action, memory[0], memory[1])
                        for action, memory in zip((old, new), memories)]
        for key in ("safe_torque_nm", "wheel_integral_after_nm", "wheel_common_reference_z_after_rad_s", "applied_action"):
            assert np.array_equal(calculations[0][key], calculations[1][key]), (tick, key)
        memories = [(calc["wheel_integral_after_nm"].copy(), calc["wheel_common_reference_z_after_rad_s"],
                     calc["applied_action"].copy()) for calc in calculations]
        if not enabled:
            assert not np.any(memories[1][2])
    assert baseline.calls == 23 and optimized.calls == 11


def test_skipping_controller_updates_is_a_real_counterexample_even_when_action_gate_closed():
    initial_integral, initial_z = np.array([.1, -.2, .15, -.05]), 1.2
    first = residual.compute_residual18(inputs(False), np.zeros(16), initial_integral, initial_z)
    assert not np.array_equal(first["wheel_integral_after_nm"], initial_integral)
    assert first["wheel_common_reference_z_after_rad_s"] != initial_z
    correct_next = residual.compute_residual18(inputs(True), np.zeros(16), first["wheel_integral_after_nm"],
                                               first["wheel_common_reference_z_after_rad_s"])
    wrongly_skipped_next = residual.compute_residual18(inputs(True), np.zeros(16), initial_integral, initial_z)
    assert not np.array_equal(correct_next["safe_torque_nm"], wrongly_skipped_next["safe_torque_nm"])
