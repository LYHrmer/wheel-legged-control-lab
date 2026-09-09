"""Hand-computed GAE and corrupted recorded updates; no simulator or learner."""

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_d1_ppo_math as audit


def sample():
    rewards = np.asarray([[1.0, 4], [-2, 5], [3, 6]], dtype=np.float32)
    values = np.asarray([[0.2, 0.1], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    advantages = np.asarray([-0.586, -2.3, 2.5, 11.086752, 9.4816, 6.03], dtype=np.float32)
    raw = np.tile([0.2, -0.3], (6, 1))
    before = np.full(6, -0.5 * (0.2**2 + (-0.3) ** 2) - math.log(2 * math.pi))
    after = np.full(6, -0.5 * (0.1**2 + (-0.4) ** 2) - math.log(2 * math.pi))
    ratio = np.exp(after - before)
    data = {
        "observations": np.zeros((6, 3)),
        "actions_raw": raw,
        "old_log_prob": before.copy(),
        "old_values": values.T.reshape(-1),
        "advantages": advantages,
        "returns": advantages + values.T.reshape(-1),
        "gae_rewards": rewards,
        "gae_values": values,
        "gae_episode_starts": np.array([[1, 1], [0, 0], [1, 0]]),
        "gae_last_values": np.array([0, 0.7]),
        "gae_last_dones": np.array([True, False]),
        "gae_gamma": np.array(0.9),
        "gae_lambda": np.array(0.8),
        "action_mean_before": np.zeros((6, 2)),
        "action_mean_after": np.full((6, 2), 0.1),
        "action_std_before": np.ones((6, 2)),
        "action_std_after": np.ones((6, 2)),
        "log_prob_before": before,
        "log_prob_after": after,
        "ratio_before": np.ones(6),
        "ratio_after": ratio,
    }
    row = {
        "n_audit_samples": 6,
        "clip_range": 0.2,
        "param_sha256_before": "a" * 64,
        "param_sha256_after": "b" * 64,
        "parameters_changed": True,
        "approx_reverse_kl_before": 0.0,
        "approx_reverse_kl_after": float(np.mean(ratio - 1 - (after - before))),
        "ratio_clip_fraction_before": 0.0,
        "ratio_clip_fraction_after": 0.0,
        "audit_index": 0,
        "npz": "sample_000000.npz",
    }
    return data, row


def test_independent_auditor_has_no_learning_or_simulator_imports():
    tree = ast.parse(Path(audit.__file__).read_text())
    modules = {
        name.split(".")[0]
        for node in ast.walk(tree)
        for name in (
            [a.name for a in node.names]
            if isinstance(node, ast.Import)
            else [node.module]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    }
    assert modules == {"__future__", "argparse", "hashlib", "json", "pathlib", "numpy"}


def test_gae_end_mask_env_major_order_and_raw_clip_examples():
    data, row = sample()
    original = {k: v.copy() for k, v in data.items()}
    report = audit.audit_sample(data, row)
    assert report["samples"] == 6 and report["parameters_changed"]
    assert report["max_absolute_errors"]["gae"] < 1e-6
    positive, negative = report["raw_advantage_examples_not_optimizer_loss"]
    assert positive["sample"] == 2 and positive["advantage"] == 2.5
    assert negative["sample"] == 0 and negative["advantage"] == pytest.approx(-0.586)
    for example in (positive, negative):
        assert example["surrogate_min"] == min(example["unclipped"], example["clipped"])
    for k in data:
        np.testing.assert_array_equal(data[k], original[k])


@pytest.mark.parametrize(
    "field", ("advantages", "returns", "old_values", "ratio_after", "log_prob_after")
)
def test_altered_recorded_numbers_are_rejected(field):
    data, row = sample()
    data[field][0] += 0.1
    with pytest.raises(ValueError, match="mismatch"):
        audit.audit_sample(data, row)


@pytest.mark.parametrize("field,value", (("gae_last_dones", True), ("gae_episode_starts", 1)))
def test_termination_cannot_be_changed_without_changing_gae(field, value):
    data, row = sample()
    if field == "gae_last_dones":
        data[field][1] = value
    else:
        data[field][1, 0] = value
    with pytest.raises(ValueError, match="GAE"):
        audit.audit_sample(data, row)


@pytest.mark.parametrize(
    "fault", ("negative_std", "nan", "bad_discount", "bad_mask", "shape", "change_label")
)
def test_invalid_distributions_shapes_and_provenance_are_rejected(fault):
    data, row = sample()
    if fault == "negative_std":
        data["action_std_before"][0, 0] = -1
    elif fault == "nan":
        data["actions_raw"][0, 0] = np.nan
    elif fault == "bad_discount":
        data["gae_gamma"] = np.array(1.1)
    elif fault == "bad_mask":
        data["gae_episode_starts"][0, 0] = 2
    elif fault == "shape":
        data["action_mean_before"] = np.zeros((6, 3))
    else:
        row["parameters_changed"] = False
    with pytest.raises(ValueError):
        audit.audit_sample(data, row)


def test_directory_hash_chain_and_read_only_inputs(tmp_path):
    data, row = sample()
    np.savez_compressed(tmp_path / row["npz"], **data)
    (tmp_path / "updates.jsonl").write_text(json.dumps(row) + "\n")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = audit.audit_directory(tmp_path)
    assert result["updates_checked"] == 1
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    second = {**row, "audit_index": 1, "npz": "sample_000001.npz"}
    np.savez_compressed(tmp_path / second["npz"], **data)
    (tmp_path / "updates.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(second) + "\n")
    with pytest.raises(ValueError, match="hash chain"):
        audit.audit_directory(tmp_path)


def test_cli_does_not_replace_existing_output(tmp_path, monkeypatch):
    out = tmp_path / "already.json"
    out.write_text("keep")
    monkeypatch.setattr("sys.argv", ["audit", str(tmp_path), "--output", str(out)])
    with pytest.raises(SystemExit) as error:
        audit.main()
    assert error.value.code == 2 and out.read_text() == "keep"
