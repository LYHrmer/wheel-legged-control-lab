"""Pure full-scorer links using an explicitly historical zero-action prefix.

The fixture contains two old endpoints, one old control and five old native
returns. Its scalar-zero bookkeeping is a format adapter, not an RL rollout.
No environment, model, geometry module, or MuJoCo is imported or instantiated.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from scripts.d1_rolling_residual_scoring import score_rolling_episode
from scripts.d1_rolling_residual_task import RollingEpisodeSpec


def _fixture():
    path = Path(__file__).parent / "fixtures" / "d1_rolling_historical_zero_prefix.json"
    fixture = json.loads(path.read_text())
    assert fixture["provenance"]["prefix_is_partial"] is True
    assert fixture["provenance"]["kind"] == (
        "historical-zero8-prefix-pure-schema-fixture-not-a-new-rolling-rollout"
    )
    payload = fixture["payload"]
    return payload, RollingEpisodeSpec(**payload["spec"])


def test_historical_prefix_is_record_valid_but_cannot_qualify_incomplete_trial():
    payload, spec = _fixture()
    original = deepcopy(payload)
    score = score_rolling_episode(payload, spec)
    assert score["record_valid"] is True, score["failure_reasons"]
    assert score["complete"] is False and score["task_passed"] is False
    assert score["final_stable"] is False and score["speed_gate_passed"] is False
    assert "trial_incomplete" in score["failure_reasons"]
    assert score["metrics"]["native_count"] == 5
    assert score["metrics"]["final_window_wheel_positive_load_fractions"] is None
    assert score["metrics"]["speed_window_mean_body_vx_mps"] is None
    assert payload == original


def test_full_scorer_rejects_bound_collision_geom_identity_mismatch():
    payload, spec = _fixture()
    payload["endpoints"][0]["collision_bounds"][0]["geom_id"] = -1
    score = score_rolling_episode(payload, spec)
    assert score["record_valid"] is False and score["task_passed"] is False
    assert any("collision bound identity" in reason for reason in score["failure_reasons"])


def test_full_scorer_rejects_invalid_frame_and_forged_signed_normal():
    original, spec = _fixture()
    for damage in ("frame", "normal"):
        payload = deepcopy(original)
        contact = payload["native"][4]["contacts"]["contacts"][0]
        if damage == "frame":
            contact["frame_geom1_to_geom2"][0] = [0.0, 0.0, 0.0]
        else:
            contact["normal_terrain_to_robot_world"] = [0.0, 0.0, -1.0]
        score = score_rolling_episode(payload, spec)
        assert score["record_valid"] is False and score["task_passed"] is False
        assert any(("frame" if damage == "frame" else "geom-order sign") in reason
                   for reason in score["failure_reasons"])


def test_full_scorer_rejects_native_torque_disconnected_from_control_trace():
    payload, spec = _fixture()
    payload["native"][0]["ctrl_nm"][0] += 0.01
    score = score_rolling_episode(payload, spec)
    assert score["record_valid"] is False and score["task_passed"] is False
    assert any("native held ctrl" in reason for reason in score["failure_reasons"])
