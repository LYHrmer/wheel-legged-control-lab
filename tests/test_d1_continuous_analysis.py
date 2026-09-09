import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "continuous_analysis_under_test", ROOT / "scripts/analyze_d1_continuous_policy.py"
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


@pytest.fixture
def rows():
    result = []
    for stage in analysis.STAGES:
        for i in range(2):
            row = {
                "stage": stage,
                "stage_time_s": i + 1,
                "truncated": "0",
                "terminated": "0",
                "time_s": len(result) + 1,
                "terrain_section": ("flat", "bumps", "ramp_up", "ramp_down")[len(result) % 4],
                "yaw_rad": i * (0.2 if stage == "turn_left" else -0.2),
                "reference_yaw_rad": i * (0.2 if stage == "turn_left" else -0.2),
                "reward": 2.0,
                **dict.fromkeys(analysis.FIELDS, 0.0),
            }
            result.append(row)
    result[-1]["truncated"] = "1"
    return result


def test_independent_summary_requires_terrain_and_actual_turns(rows):
    assert analysis.recompute(rows)["quality_pass"]
    for row in rows:
        row["yaw_rad"] = 0
    assert analysis.recompute(rows)["completed"]
    assert not analysis.recompute(rows)["quality_pass"]


def test_completion_does_not_relax_height_threshold(rows):
    for row in rows:
        row["clearance_error_m"] = 0.026
        row["reward"] = 2 - (0.026 / 0.08) ** 2
    summary = analysis.recompute(rows)
    assert summary["completed"] and not summary["quality_pass"]
    assert summary["whole_task_rmse"]["clearance_error_m"] == pytest.approx(0.026)


def test_independent_reward_audit_catches_scaled_or_wrong_reward(rows):
    rows[0]["reward"] *= 0.01
    with pytest.raises(AssertionError):
        analysis.recompute(rows)


def test_recompute_uses_steady_window_but_whole_rmse_keeps_transient(rows):
    rows[0]["stage_time_s"] = 0.5
    rows[0]["velocity_error_mps"] = 0.5
    rows[0]["reward"] = 1 + math.exp(-((0.5 / 0.2) ** 2))
    summary = analysis.recompute(rows)
    assert summary["quality_pass"]
    assert summary["stage_summaries"][0]["steady_rmse"]["velocity_error_mps"] == 0
    assert summary["whole_task_rmse"]["velocity_error_mps"] == pytest.approx(0.125)


def test_summary_comparison_rejects_wrong_stage_or_total_metrics(rows):
    calculated = analysis.recompute(rows)
    stored = copy.deepcopy(calculated)
    assert analysis.compare_summary(calculated, stored) == 0
    stored["stage_summaries"][5]["quality_pass"] = False
    with pytest.raises(ValueError, match="stage result"):
        analysis.compare_summary(calculated, stored)
    stored = copy.deepcopy(calculated)
    stored["whole_task_rmse"]["clearance_error_m"] = 0.001
    with pytest.raises(ValueError, match="RMSE differs"):
        analysis.compare_summary(calculated, stored)


def test_pairs_use_same_noise_case_and_keep_baseline_single_copy():
    rows = [
        {
            "case": "seed617_45s",
            "mode": mode,
            "training_seed": seed,
            "budget": 131072,
            "split": "holdout",
            "completed": True,
            "quality_pass": mode == "zero",
            **dict.fromkeys(analysis.FIELDS, value),
        }
        for mode, seed, value in (
            ("zero", 0, 0.01),
            ("policy", 19000, 0.015),
            ("policy", 20000, 0.008),
        )
    ]
    paired = analysis.paired_rows(rows)
    assert len(paired) == 2
    assert paired[0]["delta_clearance_error_m"] == pytest.approx(0.005)
    assert paired[1]["delta_clearance_error_m"] == pytest.approx(-0.002)
    assert paired[0]["baseline_quality_pass"]
    assert not paired[0]["policy_quality_pass"]


def test_position_phase_check_is_exact_not_a_relaxed_visual_tolerance():
    qpos = np.asarray([[1.0, 0.2, 0.455], [1.02, 0.21, 0.453]])
    qvel = np.asarray([[0.25, 0.1, -0.05], [0.2, 0.1, -0.03]])
    xyz = qpos - 0.002 * qvel
    assert analysis.audit_poststep_position(qpos, qvel, xyz, 0.002) == pytest.approx(0.0005)
    with pytest.raises(AssertionError):
        analysis.audit_poststep_position(qpos, qvel, xyz, 0.01)
    corrupted = xyz.copy()
    corrupted[0, 0] += 0.00001
    with pytest.raises(AssertionError):
        analysis.audit_poststep_position(qpos, qvel, corrupted, 0.002)
