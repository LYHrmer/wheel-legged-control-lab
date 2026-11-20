import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts import terrain_ppo86 as smoke


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_three_fixture_seeds_update_mean_and_reload_without_cross_segment_gae(tmp_path):
    group = tmp_path / "technical"
    summary = smoke.run_fixture(group)
    assert summary["status"] == "COMPLETE" and summary["passed"]
    assert summary["d1_optimizer_steps"] == summary["d1_controls"] == 0
    assert summary["synthetic_optimizer_steps"] == 3 and summary["synthetic_transitions"] == 384
    assert summary["deployment_eligible"] is False
    assert [case["seed"] for case in summary["cases"]] == list(smoke.SEEDS)
    initial = {c["actor_sha256_before"] for c in summary["cases"]}
    assert len(initial) == 1
    for case in summary["cases"]:
        folder = group / f"seed_{case['seed']}"
        assert case["actor_sha256_after"] not in initial
        assert case["resets_completed"] == 2
        assert case["step_calls"] == case["steps_returned"] == case["recorded_transitions"] == 128
        assert case["optimizer_steps"] == 1 and case["gradient_norms"]["actor"] > 0
        assert case["gradient_norms"]["critic"] > 0
        assert case["actor_mean_weight_max_abs_delta"] > 0 and case["fixed_probe_reload_exact"]
        assert case["torch_threads"] == case["torch_interop_threads"] == 1
        assert case["max_abs_requested_minus_allocated_n"] > 0
        probes = _read(folder / "reload_probes.json")
        assert probes["mean"] == probes["reloaded_mean"] and probes["value"] == probes["reloaded_value"]
        assert any(row[0] != 0 for row in probes["mean"])
        assert _read(folder / "diagnostic_parameters.json")["deployment_eligible"] is False
        rows = list(map(json.loads, (folder / "transaction.jsonl").read_text().splitlines()))
        assert [i for i, r in enumerate(rows) if r["truncated"]] == [63, 127]
        assert not any(r["terminated"] for r in rows)
        assert rows[0]["allocated_transfer_n"] == rows[64]["allocated_transfer_n"] == 0
        assert rows[0]["observation"][1] == -rows[64]["observation"][1] == 0.5
        update = _read(folder / "ppo_update.json")
        for segment in range(2):
            carry = 0.0
            for idx in reversed(range(64 * segment, 64 * (segment + 1))):
                row = rows[idx]
                carry = row["reward"] + 0.99 * 0.95 * carry  # all initial value parameters are zero
                assert update["advantages"][idx] == pytest.approx(carry)
                expected_logp = -(row["raw_action"] / 0.25) ** 2 / 2 - math.log(0.25 * math.sqrt(2 * math.pi))
                assert row["old_log_prob"] == pytest.approx(expected_logp)
                assert row["env_action"] == np.clip(row["raw_action"], -1, 1)
                assert row["requested_transfer_n"] == 8 * row["env_action"]
    with pytest.raises(FileExistsError):
        smoke.run_fixture(group)


def test_early_physical_end_cannot_be_refilled_and_raw_sample_survives_clipping(tmp_path, monkeypatch):
    class Ending:
        def reset(self, *, seed):
            return [0.3], {"seed": seed}

        def step(self, action):
            assert action == 1
            return [0.4], -1, True, False, {"requested_transfer_n": action, "allocated_transfer_n": 0.0,
                                             "applied_torque_nm": [0.0] * 16, "termination_reason": "fallen"}

    def fixed_noise(shape, *, dtype, device, generator):
        return torch.full(shape, 5.0, dtype=dtype, device=device)

    monkeypatch.setattr(torch, "randn", fixed_noise)
    folder = tmp_path / "physical-contract"
    result = smoke.run_case(Ending(), smoke.SEEDS[0], folder, task="physical", physical_gate_passed=True)
    assert result["status"] == "PARTIAL" and not result["passed"]
    assert result["resets_completed"] == result["recorded_transitions"] == 1
    assert result["optimizer_steps"] == result["updates_attempted"] == 0
    assert not (folder / "summary.json").exists() and not (folder / "diagnostic_parameters.json").exists()
    row = _read(folder / "transaction.jsonl")
    assert row["raw_action"] == 1.25 and row["env_action"] == 1
    assert row["old_log_prob"] == pytest.approx(-(1.25 / 0.25) ** 2 / 2 - math.log(0.25 * math.sqrt(2 * math.pi)))
    assert len(row["applied_torque_nm"]) == 16


def test_physical_gate_precedes_reset_and_nonfinite_transition_is_retained(tmp_path):
    class Invalid:
        def reset(self, *, seed):
            return [0.0]

        def step(self, action):
            return [math.nan], 0.0, False, False, {"requested_transfer_n": action, "allocated_transfer_n": action}

    folder = tmp_path / "blocked"
    with pytest.raises(ValueError, match="admitted"):
        smoke.run_case(Invalid(), smoke.SEEDS[0], folder, task="physical")
    assert not folder.exists()
    with pytest.raises(ValueError, match="nonfinite"):
        smoke.run_case(Invalid(), smoke.SEEDS[0], folder, task="synthetic")
    result = _read(folder / "partial.json")
    assert result["status"] == "PARTIAL" and result["step_calls"] == result["steps_returned"] == 1
    assert result["recorded_transitions"] == result["optimizer_steps"] == 0
