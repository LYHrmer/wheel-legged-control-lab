"""Independent arithmetic checks; no formal holdout simulation is run here."""

from __future__ import annotations

import copy
import json
import math
import zipfile

import numpy as np
import pytest

from scripts import analyze_d1_action_ablation as analysis


@pytest.fixture
def criteria():
    return {
        "minimum_episode_seconds": 4.0,
        "progress_fraction_min": 0.65,
        "progress_fraction_max": 1.35,
        "tail_window_seconds": 1.0,
        "tail_velocity_rmse_absolute_mps": 0.06,
        "tail_velocity_rmse_target_fraction": 0.25,
        "clearance_rmse_max_m": 0.03,
        "max_abs_yaw_rad": 0.15,
        "mean_torque_saturation_fraction_max": 0.05,
    }


def episode_rows(variant="full_gaussian"):
    enabled = variant != "zero_residual"
    z_active = enabled and variant != "fx_only"
    return [
        {
            "step": step,
            "time_s": step * 0.01,
            "position_x_m": step * 0.002,
            "initial_position_x_m": 0.0,
            "forward_velocity_mps": 0.18,
            "command_velocity_mps": 0.2,
            "velocity_error_mps": -0.02,
            "clearance_m": 0.456,
            "command_clearance_m": 0.455,
            "clearance_error_m": 0.001,
            "terminated": 0,
            "truncated": int(step == 400),
            "termination_reason": "time_limit" if step == 400 else "ongoing",
            "policy_enabled": int(enabled),
            "policy_gated": 0,
            "yaw_rad": 0.0,
            "torque_saturation_fraction": 0.0,
            "reward": 4.0,
            "action_longitudinal": 0.3 if enabled else 0.0,
            "action_vertical": -0.4 if z_active else 0.0,
            "raw_mean_x": 0.3 if enabled else 0.0,
            "std_x": 0.9 if enabled else 0.0,
            "raw_mean_z": -0.4 if z_active else 0.0,
            "std_z": 0.8 if z_active else 0.0,
            "vertical_channel_active": int(z_active),
        }
        for step in range(1, 401)
    ]


@pytest.mark.parametrize("variant", (*analysis.VARIANTS, "zero_residual"))
def test_metrics_recomputed_from_physical_samples(criteria, variant):
    result = analysis.recompute_episode(
        episode_rows(variant), {"velocity_mps": 0.2}, criteria, variant
    )
    assert result["control_steps"] == 400
    assert result["completed"] == 1 and result["quality_success"] == 1
    assert result["episode_return"] == 1600
    assert result["velocity_rmse_mps"] == pytest.approx(0.02)
    assert result["tail_velocity_rmse_mps"] == pytest.approx(0.02)
    assert result["clearance_rmse_m"] == pytest.approx(0.001)
    assert result["progress_fraction"] == 1.0
    assert result["policy_enabled_fraction"] == float(variant != "zero_residual")


@pytest.mark.parametrize(
    "fault", ["wrong_error", "after_done", "gating", "wrong_clip", "negative_std", "nan"]
)
def test_inconsistent_telemetry_is_rejected(criteria, fault):
    rows = episode_rows()
    if fault == "wrong_error":
        rows[0]["velocity_error_mps"] = 0.3
    elif fault == "after_done":
        rows[5]["terminated"] = 1
    elif fault == "gating":
        rows[0]["policy_gated"] = 1
    elif fault == "wrong_clip":
        rows[0]["action_longitudinal"] = 0.1
    elif fault == "negative_std":
        rows[0]["std_x"] = -1
    else:
        rows[0]["reward"] = float("nan")
    with pytest.raises(ValueError):
        analysis.recompute_episode(rows, {"velocity_mps": 0.2}, criteria, "full_gaussian")


def test_real_height_failure_is_not_hidden_by_survival(criteria):
    rows = episode_rows()
    for row in rows:
        row["clearance_m"], row["clearance_error_m"] = 0.495, 0.04
    result = analysis.recompute_episode(rows, {"velocity_mps": 0.2}, criteria, "full_gaussian")
    assert result["completed"] == 1
    assert result["quality_success"] == 0
    assert result["quality_failure_reasons"] == "clearance"


def samples(variant="full_gaussian", budget=512):
    dimension = 1 if variant == "fx_only" else 2
    raw = np.tile(np.asarray([1.4, -1.3][:dimension], np.float32), (budget, 1))
    mean = np.full_like(raw, 0.25)
    std = np.ones_like(raw)
    lp = np.sum(-0.5 * (raw - mean) ** 2 - 0.5 * math.log(2 * math.pi), axis=1)
    return {
        "timesteps": np.repeat(np.arange(4, budget + 1, 4), 4),
        "raw_action": raw,
        "executed_action": np.clip(raw, -1, 1),
        "raw_mean": mean,
        "std": std,
        "log_prob": lp,
    }


@pytest.mark.parametrize("variant", analysis.VARIANTS)
def test_actual_sample_clipping_is_separate_from_mean_bounds(variant):
    result = analysis.action_statistics(samples(variant), variant, 512)
    assert len(result) == (1 if variant == "fx_only" else 2)
    for row in result:
        assert row["samples"] == 512
        assert row["sample_clip_fraction"] == 1
        assert row["deterministic_upper_fraction"] == 0
        assert row["deterministic_lower_fraction"] == 0
        assert row["raw_log_prob_max_error"] < 1e-5


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_density",
        "wrong_clip",
        "missing_samples",
        "wrong_times",
        "negative_std",
        "bounded_mu",
        "fx_extra_axis",
    ],
)
def test_sample_audit_rejects_density_and_exposure_errors(fault):
    data = samples()
    variant = "full_gaussian"
    if fault == "wrong_density":
        data["log_prob"] = np.sum(
            -0.5 * (data["executed_action"] - data["raw_mean"]) ** 2 - 0.5 * math.log(2 * math.pi),
            axis=1,
        )
    elif fault == "wrong_clip":
        data["executed_action"][0, 0] = 0.5
    elif fault == "missing_samples":
        data["raw_action"] = data["raw_action"][:-1]
    elif fault == "wrong_times":
        data["timesteps"][4] = 4
    elif fault == "negative_std":
        data["std"][0, 0] = -1
    elif fault == "bounded_mu":
        variant = "bounded_mean"
        data["raw_mean"][0, 0] = 1.01
    else:
        variant = "fx_only"
    with pytest.raises(ValueError):
        analysis.action_statistics(data, variant, 512)


def metric_row(variant, seed, budget, case_id, value):
    return {
        "variant": variant,
        "training_seed": seed,
        "budget": budget,
        "case_id": case_id,
        "completed": 1,
        "quality_success": int(value < 5),
        "velocity_rmse_mps": value,
        "tail_velocity_rmse_mps": value / 2,
        "clearance_rmse_m": value / 100,
        "episode_return": 100 - value,
    }


def development_rows():
    rows = []
    for budget in analysis.BUDGETS:
        indices = range(24) if budget == analysis.BUDGETS[-1] else analysis.FIXED_INDICES
        for index in indices:
            rows.append(
                metric_row(
                    "full_gaussian",
                    9000,
                    budget,
                    f"case_{index}",
                    1 if index in analysis.FIXED_INDICES else 10,
                )
            )
    return rows


def test_development_curve_keeps_the_same_six_cases_at_all_budgets():
    fixed, final = analysis.development_curves(
        development_rows(), [f"case_{index}" for index in analysis.FIXED_INDICES]
    )
    assert len(fixed) == 3 and all(row["cases"] == 6 for row in fixed)
    assert all(row["quality_successes"] == 6 for row in fixed)
    assert all(row["mean_velocity_rmse_mps"] == 1 for row in fixed)
    assert len(final) == 1 and final[0]["cases"] == 24
    assert final[0]["quality_successes"] == 6
    assert final[0]["mean_velocity_rmse_mps"] == 7.75


def test_missing_fixed_development_case_is_rejected():
    rows = development_rows()[1:]
    with pytest.raises(ValueError, match="missing"):
        analysis.development_curves(rows, [f"case_{index}" for index in analysis.FIXED_INDICES])


def holdout_rows():
    rows = [metric_row("zero_residual", None, 131072, f"case_{i}", 1) for i in range(24)]
    for seed, base in zip(analysis.SEEDS, (2, 4, 6), strict=True):
        for variant, delta in (("full_gaussian", 0), ("bounded_mean", 2), ("fx_only", -1)):
            rows.extend(
                metric_row(variant, seed, 131072, f"case_{i}", base + delta) for i in range(24)
            )
    return rows


def test_holdout_pairs_use_seed_as_unit_and_baseline_once():
    rows = holdout_rows()
    pairs = analysis.paired_comparisons(rows[::-1])
    assert len(pairs) == 15
    same_seed = [row for row in pairs if row["reference"] == "full_gaussian"]
    assert len(same_seed) == 6
    for row in same_seed:
        assert row["reference_training_seed"] == row["training_seed"]
        expected = 2 if row["variant"] == "bounded_mean" else -1
        assert row["delta_mean_velocity_rmse_mps"] == expected
        assert row["delta_mean_episode_return"] == -expected
    baseline = [
        r for r in analysis.aggregate_cases(rows, "holdout") if r["variant"] == "zero_residual"
    ]
    assert len(baseline) == 1 and baseline[0]["cases"] == 24


@pytest.mark.parametrize("fault", ["duplicate", "missing", "different_case", "seeded_baseline"])
def test_holdout_pairs_reject_pseudoreplication(fault):
    rows = holdout_rows()
    if fault == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    elif fault == "missing":
        rows.pop()
    elif fault == "different_case":
        rows[-1]["case_id"] = "different_case"
    else:
        rows[0]["training_seed"] = 9000
    with pytest.raises(ValueError):
        analysis.paired_comparisons(rows)


def test_holdout_json_summary_is_recomputed_not_trusted():
    rows = holdout_rows()
    # Hand-computed scalar expectations for the synthetic 24 identical cases.
    controllers = []
    for variant, seed, value in [("zero_residual", None, 1)] + [
        (variant, seed, base + delta)
        for seed, base in zip(analysis.SEEDS, (2, 4, 6), strict=True)
        for variant, delta in (("full_gaussian", 0), ("bounded_mean", 2), ("fx_only", -1))
    ]:
        controllers.append(
            {
                "variant": variant,
                "training_seed": seed,
                "cases": 24,
                "completed": 24,
                "quality_successes": 24 if value < 5 else 0,
                "mean_velocity_rmse_mps": value,
                "mean_tail_velocity_rmse_mps": value / 2,
                "mean_clearance_rmse_m": value / 100,
                "mean_episode_return": 100 - value,
            }
        )
    pairs = []
    for seed, base in zip(analysis.SEEDS, (2, 4, 6), strict=True):
        for variant, delta in (("bounded_mean", 2), ("fx_only", -1)):
            pairs.append(
                {
                    "variant": variant,
                    "training_seed": seed,
                    "delta_quality_successes": 24 * (int(base + delta < 5) - int(base < 5)),
                    "delta_mean_velocity_rmse_mps": delta,
                    "delta_mean_tail_velocity_rmse_mps": delta / 2,
                    "delta_mean_clearance_rmse_m": delta / 100,
                    "delta_mean_episode_return": -delta,
                }
            )
    summary = {"controllers": controllers, "paired_seed_differences": pairs}
    assert analysis.audit_holdout_summary(rows, summary) < 1e-12
    summary["paired_seed_differences"][-1]["delta_mean_velocity_rmse_mps"] = -100
    with pytest.raises(ValueError, match="paired summary"):
        analysis.audit_holdout_summary(rows, summary)


def test_rollout_csv_statistics_are_recomputed_from_npz(tmp_path):
    data = samples()
    np.savez_compressed(tmp_path / "training_action_samples.npz", **data)
    row = {"timesteps": 512, "samples": 512, "max_unchanged_log_prob_error": 0.0}
    for axis in (0, 1):
        row.update(
            {
                f"mu_mean_{axis}": 0.25,
                f"mu_abs_max_{axis}": 0.25,
                f"std_mean_{axis}": 1.0,
                f"sample_clip_fraction_{axis}": 1.0,
                f"deterministic_upper_fraction_{axis}": 0.0,
                f"executed_action_mean_{axis}": 1.0 if axis == 0 else -1.0,
            }
        )
    analysis.write_csv(tmp_path / "training_action_diagnostics.csv", [row])
    result = analysis.audit_actions(tmp_path, "full_gaussian", 9000, 512)
    assert len(result) == 2 and result[0]["rollout_summary_max_error"] == 0
    row["sample_clip_fraction_0"] = 0.0
    analysis.write_csv(tmp_path / "training_action_diagnostics.csv", [row])
    with pytest.raises(ValueError, match="rollout sample_clip_fraction"):
        analysis.audit_actions(tmp_path, "full_gaussian", 9000, 512)


def test_manifest_covers_nested_manifest_and_detects_mutation(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/manifest.json").write_text("{}")
    (tmp_path / "sample.csv").write_text("value\n1\n")
    manifest = {
        str(p.relative_to(tmp_path)): analysis.sha256(p) for p in tmp_path.rglob("*") if p.is_file()
    }
    analysis.write_json(tmp_path / "manifest.json", manifest)
    actual = analysis.audit_manifest(tmp_path)
    assert set(actual) == {"manifest.json", "nested/manifest.json", "sample.csv"}
    (tmp_path / "sample.csv").write_text("value\n2\n")
    with pytest.raises(ValueError, match="SHA256"):
        analysis.audit_manifest(tmp_path)


def test_checkpoint_counters_are_read_from_serialized_model(tmp_path):
    data = {
        "seed": 9000,
        "num_timesteps": 512,
        "_n_updates": 4,
        "n_envs": 4,
        **analysis.PPO_SETTINGS,
    }
    path = tmp_path / "synthetic_counter_fixture.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps(data))
    assert analysis.audit_checkpoint(path, 9000, 512)["ppo_epochs_completed"] == 4
    with pytest.raises(ValueError, match="seed"):
        analysis.audit_checkpoint(path, 10000, 512)


def test_analysis_cannot_write_inside_training_inputs(tmp_path):
    root = tmp_path / "training"
    root.mkdir()
    with pytest.raises(ValueError, match="separate"):
        analysis.run([root], tmp_path / "holdout", root / "analysis")
