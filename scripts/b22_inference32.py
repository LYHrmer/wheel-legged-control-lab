"""Skip only unused, stateless B22 predictions; keep the full control tick.

This helper does not load a model, prepare observations, evaluate command gates,
advance a controller, or own physics. A NEW worker must pass the gate from its
exact current controller decision and still execute the ordinary control step.
The unchanged strict loader's one batched probe prediction is never skipped.
"""
from __future__ import annotations

import numpy as np

SCHEMA = "b22-gated-stateless-inference32-v1"
MODEL_SHA = "7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691"
POLICY_STATE_SHA = "59440de0c5beb102cec0d292838fb9b125eca7eb56bbffc06d1eaec46ec7aa48"
METADATA_SHA = "af700d1f2612e441c8f0560cca38c77b6a854526d3e841e544ef9d5321672830"
ACTOR_LAYERS = [{"type": "Linear", "in_features": 99, "out_features": 128}, {"type": "Tanh"},
                {"type": "Linear", "in_features": 128, "out_features": 128}, {"type": "Tanh"}]


def require(value, message):
    if not value:
        raise ValueError(message)


def check_stateless_proof32(manifest, strict_load):
    """Check the fixed B22 loader receipt; this does not replace that loader.

    The caller must bind the actual loaded object to the same trusted manifest,
    source/dependency hashes and fresh strict-load receipt in its new session.
    A receipt alone cannot attest an arbitrary object supplied by other code.
    """
    require(isinstance(manifest, dict) and isinstance(strict_load, dict), "strict B22 load proof required")
    require(manifest.get("files", {}).get("final_model.zip", {}).get("sha256") == MODEL_SHA,
            "gated inference is restricted to the fixed B22 checkpoint")
    expected = manifest.get("reload_verification", {})
    for proof in (expected, strict_load):
        require(proof.get("actor_layers") == ACTOR_LAYERS and proof.get("critic_layers") == ACTOR_LAYERS
                and proof.get("observation_space") == {"dtype": "float32", "high": 5., "low": -5., "shape": [99]}
                and proof.get("action_space") == {"dtype": "float32", "high": 1., "low": -1., "shape": [16]}
                and proof.get("loaded_policy_state_sha256") == POLICY_STATE_SHA
                and proof.get("matches_training_final_policy_state") is True
                and proof.get("probe_rows") == 32 and proof.get("probe_actions_byte_exact") is True
                and proof.get("verified_without_engine_or_reset") is True,
                "strict receipt does not prove the fixed stateless feedforward B22 policy")
    require(strict_load.get("metadata_sha256") == manifest.get("metadata_sha256") == METADATA_SHA
            == expected.get("metadata_sha256"), "strict loader metadata identity differs")


class GatedInference32:
    """Return control input and an honest inference ledger for one decision.

    Gate-off output is synthetic zero, so its unobserved policy action is null.
    State preparation, last *effective* action, PI, z, stop latch and all native
    intervals stay with the existing control loop. This API never skips them.
    """

    def __init__(self, actor, model, *, numeric_actor_action, manifest=None, strict_load=None):
        require(type(actor) is str and actor in ("B", "zero"), "actor must be B or zero")
        require(callable(numeric_actor_action), "the bound numeric actor function is required")
        if actor == "B":
            require(model is not None, "B actor requires the strictly loaded model")
            check_stateless_proof32(manifest, strict_load)
        else:
            require(model is None, "zero actor must not hold a policy")
        self.actor, self.model, self.numeric_actor_action = actor, model, numeric_actor_action
        self._controls = self._predictions = self._skips = self._zero_controls = 0

    def select(self, observation, *, action_enabled):
        require(type(action_enabled) is bool, "current original controller gate must be bool")
        obs = np.asarray(observation)
        require(obs.shape == (99,) and obs.dtype.kind in "fiu", "invalid prepared observation99")
        checked = np.asarray(obs, dtype=np.float64)
        require(bool(np.isfinite(checked).all()) and bool((np.abs(checked) <= 5.).all()),
                "invalid prepared observation99")
        if self.actor == "B" and action_enabled:
            action, predicted = self.numeric_actor_action("final_policy", self.model, observation)
            require(predicted is True, "bound B actor must perform a prediction")
            raw = np.asarray(action)
            require(raw.shape == (16,) and raw.dtype.kind in "fiu", "invalid numeric B22 action")
            checked_action = np.asarray(raw, dtype=np.float64)
            require(bool(np.isfinite(checked_action).all()) and bool((np.abs(checked_action) <= 1.).all()),
                    "invalid numeric B22 action")
            action = np.asarray(raw, dtype=np.float32).copy()
            reason, observed = "PREDICTED", True
        else:
            action, predicted, observed = np.zeros(16, dtype=np.float32), False, False
            reason = "GATE_OFF" if self.actor == "B" else "ZERO_ACTOR"
        self._controls += 1
        self._predictions += int(predicted)
        self._skips += int(reason == "GATE_OFF")
        self._zero_controls += int(reason == "ZERO_ACTOR")
        ledger = {"schema": SCHEMA, "actor": self.actor, "action_gate_enabled": action_enabled,
                  "predict_called": predicted, "policy_action_observed": observed,
                  "reason": reason, "raw_policy_action16": action.tolist() if observed else None,
                  "controller_input_action16": action.tolist(), "policy_output_kind": "SB3_clipped_deterministic_action" if observed else "NOT_OBSERVED",
                  "checkpoint_sha256": MODEL_SHA if self.actor == "B" else None}
        return action, predicted, ledger

    def summary(self):
        """Expected API counts, to compare with the worker's independent counter."""
        probe = int(self.actor == "B")
        return {"schema": SCHEMA, "control_decisions": self._controls,
                "control_predictions": self._predictions, "gate_off_skips": self._skips,
                "zero_actor_decisions": self._zero_controls, "required_loader_probe_predict_calls": probe,
                "expected_total_model_predict_calls": probe+self._predictions,
                "full_closed_loop_equivalence": "NOT_VALIDATED"}
