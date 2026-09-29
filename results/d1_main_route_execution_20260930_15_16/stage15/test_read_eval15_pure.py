"""Synthetic numeric boundary tests; no policy, model or physics."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

W = Path(__file__).resolve().parent.parent
for directory in (W, W / "course_impl08", W / "rl11", W / "continuation13",
                  W / "continuation15"):
    sys.path.insert(0, str(directory))

from archive13.atomic_archive_13 import ArchiveWriter
from read_eval15 import document, grouped_global_promotion
from readback15 import score_floor_saved
from recipes15 import ORDER, floor_schedule


def test_eval_reader_rejects_tampered_archive_payload(tmp_path):
    path = tmp_path / "schedule.json"
    ArchiveWriter().commit_json(path, {"seed": 151090})
    assert document(path) == {"seed": 151090}
    path.write_text('{"seed":88701}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        document(path)


def test_legal_early_floor_prefix_is_numeric_failure_not_corrupt_archive():
    schedule = floor_schedule()
    n = 100
    q = np.zeros(23)
    q[2], q[3] = .455, 1.0
    states = {
        "qpos": np.tile(q, (n+1, 1)),
        "qvel": np.zeros((n+1, 22)),
        "ctrl": np.zeros((n+1, 16)),
        "qacc_warmstart": np.zeros((n+1, 22)),
        "observation": np.zeros((n+1, 99)),
        "time": np.arange(n+1) * .01,
    }
    rows = [{
        "tick": tick, "actor": "global_continue",
        "policy_predict_called": True,
        "input_observation99": np.zeros(99),
        "policy_input_action": np.zeros(16),
        "info": {"metrics": {"body_com_vx_mps": 0.0,
                              "body_yaw_rate_rps": 0.0,
                              "clearance_m": .455},
                 "terminal_reason": "fall_or_low_clearance" if tick == n-1 else None},
    } for tick in range(n)]
    native = [{
        "native_index": i,
        "start_time_s": i*.002,
        "end_time_s": (i+1)*.002,
        "after": {"qpos": q.copy()},
        "nonwheel_contact_count": 0,
    } for i in range(5*n)]
    saved = {
        "case_id": schedule.case_id, "actor": "global_continue",
        "seed": schedule.seed, "terrain": "flat", "control_cap": 600,
        "raw_command_sha256": schedule.command_sha256,
        "raw_commands": [asdict(row) for row in schedule.raw_commands],
        "checkpoint_sha256": "a"*64,
    }
    receipt = {
        "case_id": schedule.case_id, "actor": "global_continue",
        "seed": schedule.seed, "control_cap": 600,
        "raw_command_sha256": schedule.command_sha256,
        "checkpoint_sha256": "a"*64, "scoring_actor": "final_policy",
        "completed_controls": n, "termination_recorded": True,
        "terminated": True, "truncated": False,
        "stop_reason": "fall_or_low_clearance", "failure": None,
        "policy_prediction_attempts": n, "policy_predictions": n,
        "native_segment": {"native_attempted": 5*n,
                           "native_returned": 5*n, "record_valid": True,
                           "force_sampling_performed": True,
                           "full_contact_qualification_recorded": True,
                           "archive_failure": None},
    }
    result = score_floor_saved(
        receipt, saved, states, rows, native,
        {"world_collision_geoms": [{"name": "floor", "type": "plane",
                                    "position_m": [0., 0., 0.]}]},
        warning_count=0, geometry_invalid_count=0,
    )
    assert result["numeric_gate_passed"] is False
    assert result["completed_controls"] == n
    assert result["last_drive_observed_ticks"] == 0


def test_grouped_promotion_needs_six_cost_windows_and_strict_highspeed_gain():
    scores = []
    canonical = []
    for case_id in ORDER:
        cap = 1800 if case_id == "ramp_0p45_complete" else 1600
        for actor, multiplier in (("zero", 1.), ("global_continue", 1.),
                                  ("grouped_continue", .8)):
            scores.append({
                "case_id": case_id, "experiment_actor": actor,
                "task_passed": True,
                "rl_terms": {"sse_total": 100.*multiplier,
                             "torque_cost_sum": 100., "drive_complete": True},
                "speed": {"rms_com_vx_error_vs_target_mps": .04*multiplier},
            })
            canonical.append({
                "case_id": case_id, "experiment_actor": actor,
                "completed_controls": cap,
                "com_vx_mps": np.full(cap, .1*multiplier),
                "applied_servo_vx_mps": np.zeros(cap),
                "body_yaw_rate_rps": np.zeros(cap),
                "applied_servo_yaw_rps": np.zeros(cap),
            })
    floors = {name: {"numeric_gate_passed": True}
              for name in ("global_continue", "grouped_continue")}
    scored = {"scores": scores, "all_three_actors_present": True}
    good = grouped_global_promotion(scored, canonical, floors)
    assert good["passed"] is True
    assert good["common_task_count"] == 6
    assert len(good["torque_cost_by_task"]) == 6
    high = next(row for row in scores if row["case_id"] == "flat_1p6"
                and row["experiment_actor"] == "global_continue")
    grouped = next(row for row in scores if row["case_id"] == "flat_1p6"
                   and row["experiment_actor"] == "grouped_continue")
    high["speed"]["rms_com_vx_error_vs_target_mps"] = 0.0
    grouped["speed"]["rms_com_vx_error_vs_target_mps"] = 0.0
    assert grouped_global_promotion(scored, canonical, floors)["passed"] is False
    high["speed"]["rms_com_vx_error_vs_target_mps"] = .04
    grouped["speed"]["rms_com_vx_error_vs_target_mps"] = .032
    grouped["rl_terms"]["drive_complete"] = False
    assert grouped_global_promotion(scored, canonical, floors)["passed"] is False
