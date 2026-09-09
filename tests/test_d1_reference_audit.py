"""The independent audit must reject wrong metrics and one-step misalignment."""

import csv
import json
import math
from copy import deepcopy

import numpy as np
import pytest

from scripts.audit_d1_reference_dynamics import (
    aggregate,
    audit,
    compare_metrics,
    recompute_episode,
    sha256,
    verify_hashes,
    verify_probe,
)


@pytest.fixture
def episode():
    protocol = {
        "seconds": 4.0,
        "quality_criteria": {
            "minimum_episode_seconds": 4.0,
            "progress_fraction_min": 0.65,
            "progress_fraction_max": 1.35,
            "tail_window_seconds": 1.0,
            "tail_velocity_rmse_absolute_mps": 0.06,
            "tail_velocity_rmse_target_fraction": 0.25,
            "clearance_rmse_max_m": 0.03,
            "max_abs_yaw_rad": 0.15,
            "mean_torque_saturation_fraction_max": 0.05,
        },
    }
    case = {"velocity_mps": 0.25, "case_id": "analytic_flat", "terrain": {"kind": "flat"}}
    rows = []
    for step in range(1, 401):
        error = 0.02 + (0.01 if step % 2 else -0.01)
        rows.append(
            {
                "step": step,
                "time_s": step * 0.01,
                "velocity_error_mps": error,
                "forward_velocity_mps": 0.25 + error,
                "command_velocity_mps": 0.25,
                "clearance_error_m": 0.01,
                "clearance_m": 0.465,
                "command_clearance_m": 0.455,
                "policy_enabled": 0,
                "policy_gated": 0,
                "roll_error_rad": 0.0,
                "pitch_error_rad": 0.0,
                "measured_roll_rad": 0.0,
                "measured_pitch_rad": 0.0,
                "reward_roll_target_rad": 0.0,
                "reward_pitch_target_rad": 0.0,
                "observation_roll_target_rad": 0.0,
                "observation_pitch_target_rad": 0.0,
                "command_roll_rad": 0.0,
                "command_pitch_rad": 0.0,
                "control_roll_target_rad": 0.0,
                "control_pitch_target_rad": 0.0,
                "height_velocity_reference_mps": 0.0,
                "requested_vertical_feedforward_n": 0.0,
                "applied_vertical_feedforward_n": 0.0,
                "terminated": 0,
                "truncated": int(step == 400),
                "termination_reason": "time_limit" if step == 400 else "ongoing",
                "initial_position_x_m": 0.7,
                "position_x_m": 0.7 + 0.25 * step * 0.01,
                "yaw_rad": 0.01,
                "torque_saturation_fraction": 0.0,
                "reward": 2.0,
            }
        )
    return rows, case, protocol


def test_analytic_bias_variance_quality_and_progress(episode):
    rows, case, protocol = episode
    result = recompute_episode(rows, case, protocol)
    assert result["tail_velocity_bias_mps"] == pytest.approx(0.02)
    assert result["tail_velocity_variance_m2ps2"] == pytest.approx(0.0001)
    assert result["tail_velocity_rmse_mps"] == pytest.approx(math.sqrt(0.0005))
    assert result["decomposition_error"] < 1e-18
    assert result["tail_error_type"] == "bias_dominant"
    assert result["progress_m"] == pytest.approx(1)
    assert result["progress_fraction"] == pytest.approx(1)
    assert result["episode_return"] == 800
    assert result["quality_success"] == 1
    assert result["quality_failure_reasons"] == ""


@pytest.mark.parametrize(
    "failure",
    ["incomplete_episode", "progress", "tail_velocity", "clearance", "yaw", "torque_saturation"],
)
def test_each_quality_gate_is_red_capable(episode, failure):
    rows, case, protocol = episode
    if failure == "incomplete_episode":
        rows[-1]["truncated"] = 0
    elif failure == "progress":
        rows[-1]["position_x_m"] = 0.7 + 0.64
    elif failure == "tail_velocity":
        for row in rows[-100:]:
            row["velocity_error_mps"] = 0.2
            row["forward_velocity_mps"] = 0.45
    elif failure == "clearance":
        for row in rows:
            row["clearance_error_m"] = 0.031
            row["clearance_m"] = 0.486
    elif failure == "yaw":
        rows[0]["yaw_rad"] = 0.151
    else:
        for row in rows:
            row["torque_saturation_fraction"] = 0.051
    result = recompute_episode(rows, case, protocol)
    assert result["quality_success"] == 0
    assert result["quality_failure_reasons"] == failure


@pytest.mark.parametrize("axis", ["roll", "pitch"])
def test_filter_requires_previous_post_step_raw_target(episode, axis):
    rows, _, _ = episode
    alpha = -math.expm1(-0.01 / 0.05)
    state = 0.01
    for index, row in enumerate(rows):
        if index:
            state += alpha * (rows[index - 1][f"reward_{axis}_target_rad"] - state)
        row[f"control_{axis}_target_rad"] = state
        row[f"command_{axis}_rad"] = state
        row[f"reward_{axis}_target_rad"] = 0.03 * math.sin(index / 10)
    assert verify_probe(rows, "attitude_tau_50ms")[f"{axis}_recurrence_max_error_rad"] == 0
    wrong = deepcopy(rows)
    for index in range(1, len(wrong)):
        previous = wrong[index - 1][f"control_{axis}_target_rad"]
        current = wrong[index][f"reward_{axis}_target_rad"]
        wrong[index][f"control_{axis}_target_rad"] = previous + alpha * (current - previous)
        wrong[index][f"command_{axis}_rad"] = wrong[index][f"control_{axis}_target_rad"]
    with pytest.raises(ValueError, match="pre-control reference recurrence"):
        verify_probe(wrong, "attitude_tau_50ms")


def test_feedforward_gain_and_both_clip_limits_are_checked(episode):
    rows, _, _ = episode
    for row, hdot in zip(rows, np.resize([0.0, 1.0, -1.0, 4.0, -4.0], len(rows)), strict=True):
        row["height_velocity_reference_mps"] = hdot
        row["requested_vertical_feedforward_n"] = 180 * hdot
        row["applied_vertical_feedforward_n"] = min(500.0, max(-500.0, 180 * hdot))
    checked = verify_probe(rows, "vertical_feedforward")
    assert checked["feedforward_clipped_steps"] == 160
    rows[3]["applied_vertical_feedforward_n"] = 501
    with pytest.raises(ValueError, match="logged feedforward clip"):
        verify_probe(rows, "vertical_feedforward")
    rows[3]["applied_vertical_feedforward_n"] = 500
    rows[1]["requested_vertical_feedforward_n"] = 181
    with pytest.raises(ValueError, match="feedforward request"):
        verify_probe(rows, "vertical_feedforward")


@pytest.mark.parametrize("field", ["velocity_error_mps", "time_s", "policy_enabled"])
def test_inconsistent_trace_is_rejected(episode, field):
    rows, case, protocol = episode
    rows[17][field] += 0.1
    with pytest.raises(ValueError, match="mismatch"):
        recompute_episode(rows, case, protocol)


def test_nonfinite_values_and_post_done_samples_are_rejected(episode):
    rows, case, protocol = episode
    rows[17]["velocity_error_mps"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        recompute_episode(rows, case, protocol)
    rows[17]["velocity_error_mps"] = 0.01
    rows[17]["truncated"] = 1
    with pytest.raises(ValueError, match="after termination"):
        recompute_episode(rows, case, protocol)


def test_changed_metric_is_rejected(episode):
    rows, case, protocol = episode
    result = recompute_episode(rows, case, protocol)
    compare_metrics(result, result)
    with pytest.raises(ValueError, match="quality_success"):
        compare_metrics(result, {**result, "quality_success": 0})


def test_hash_tampering_and_directory_escape_are_rejected(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text("time_s\n0.01\n")
    expected = {path.name: sha256(path)}
    verify_hashes(tmp_path, expected)
    path.write_text("time_s\n0.02\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_hashes(tmp_path, expected)
    with pytest.raises(ValueError, match="escapes input directory"):
        verify_hashes(tmp_path, {"../other.csv": "irrelevant"})


def test_aggregation_uses_case_mean_not_pooled_rmse():
    records = [
        {
            "variant": "baseline",
            "terrain_kind": "bumps",
            "quality_success": passed,
            "velocity_rmse_mps": value,
            "tail_velocity_rmse_mps": value,
            "clearance_rmse_m": value / 10,
        }
        for value, passed in ((0.1, 1), (0.3, 0))
    ]
    result = aggregate(records)
    assert result[0]["mean_tail_velocity_rmse_mps"] == pytest.approx(0.2)
    assert result[0]["quality_passes"] == 1
    assert result[0]["episodes"] == 2


def test_audit_preserves_originals_and_hashes_derived_outputs(tmp_path, episode):
    rows, case, protocol = episode
    protocol.update(variants=["baseline"], cases=[case], repeats=1, source_sha256={})
    result = recompute_episode(rows, case, protocol)
    result.update(variant="baseline", case_id=case["case_id"], repeat=0)
    for name, content in (
        ("protocol.json", protocol),
        ("summary.json", {"source_unchanged_during_run": True, "episodes": 1, "metrics": [result]}),
    ):
        (tmp_path / name).write_text(json.dumps(content))
    for name, content in (
        ("metrics.csv", [result]),
        ("baseline__analytic_flat__repeat_0.csv", rows),
    ):
        with (tmp_path / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(content[0]))
            writer.writeheader()
            writer.writerows(content)
    original = {path.name: sha256(path) for path in tmp_path.iterdir()}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(original))
    manifest_digest = sha256(manifest)
    report = audit(tmp_path, plots=False)
    assert report["episodes_recomputed"] == 1
    assert report["control_samples_recomputed"] == 400
    assert report["metric_max_absolute_difference"] == 0
    verify_hashes(tmp_path, original)
    assert sha256(manifest) == manifest_digest
    derived = json.loads((tmp_path / "derived_manifest.json").read_text())
    verify_hashes(tmp_path, derived["derived_artifacts_sha256"])
