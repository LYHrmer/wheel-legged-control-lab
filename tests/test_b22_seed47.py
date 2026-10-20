"""Pure checks of the 47 seed-replication pair."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import b22_reward46 as r
import b22_seed47 as s47


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def parent():
    return load_module("p47_parent_test", ROOT / "scripts/run_b22_seed47.py")


@pytest.fixture(scope="module")
def worker():
    return load_module("w47_worker_test", ROOT / "scripts/worker_b22_seed47.py")


@pytest.fixture(scope="module")
def reader():
    return load_module("r47_reader_test", ROOT / "scripts/read_b22_seed47.py")


def test_pair_definition_is_single_variable():
    assert set(s47.ARMS) == {"ffyaw_s21", "ffsse_s21"}
    assert {a["seed"] for a in s47.ARMS.values()} == {201021}
    assert {a["curriculum"] for a in s47.ARMS.values()} == {"yaw_emphasis"}
    assert s47.ARMS["ffyaw_s21"]["beta"] == 0 and s47.ARMS["ffsse_s21"]["beta"] == 8
    assert {a["alpha"] for a in s47.ARMS.values()} == {8}
    assert s47.FEEDFORWARD == 3.9


def test_plan_has_bootstrap_and_the_seed_pair(parent):
    plan = parent.describe42()
    stages = plan["stages"]
    assert [s["name"] for s in stages] == ["bootstrap", "train_ffyaw_s21", "train_ffsse_s21"]
    limits = plan["limits"]
    assert limits["controls"] == sum(s["controls"] for s in stages) == 65536
    assert limits["normal_native"] == sum(s["normal_native"] for s in stages) == 327680
    assert limits["compiler_native"] == sum(s["compiler_native"] for s in stages) == 4
    assert plan["yaw_feedforward"] == 3.9
    assert plan["curricula"] == {"ffyaw_s21": "yaw_emphasis", "ffsse_s21": "yaw_emphasis"}


def test_every_stage_session_passes_the_workers_numeric_gate(parent, worker, monkeypatch):
    monkeypatch.setattr(parent, "identity", lambda path: {"bytes": 1, "sha256": "0" * 64})
    monkeypatch.setattr(parent, "worker_api", lambda: worker)
    for stage in parent.stages42():
        session = parent.make_session42(
            stage,
            {"inputs": {}},
            parent_pid=1,
            nonce="0" * 48,
            started=0.0,
            wall_clock_utc="2026-10-20T00:00:00+00:00",
        )
        worker.validate_numeric_session42(session)
        if stage["mode"] == "train":
            arm = stage["reward_arm"]
            assert session["seed"] == session["ppo_seed"] == 201021
            assert session["beta"] == s47.ARMS[arm]["beta"]
            assert session["alpha"] == 8 and session["curriculum"] == "yaw_emphasis"


def test_the_ppo_seed_reaches_training_through_the_spec(worker):
    source = (ROOT / "scripts/worker_b22_seed47.py").read_text()
    # train22 reads T['ppo_seed_by_stage'][stage-1]; a hardcoded 201011 would silently disable the replication
    assert "spec['training']['ppo_seed_by_stage']=[201001,session['ppo_seed']]" in source
    assert "[201001,201011]" not in source


def test_reader_arm_table_matches_the_shared_definition(worker, reader):
    assert reader.CONTRACT == worker.CONTRACT == "C47_B22_seed_replication_v1"
    for arm, spec in s47.ARMS.items():
        assert reader.ARMS[arm] == {k: spec[k] for k in ("alpha", "beta", "seed")}


def _row(arm, qx, qy):
    command = {"forward_velocity_mps": 0.35, "lateral_velocity_mps": 0.0, "yaw_rate_rps": 0.0}
    metrics = {"body_com_vx_mps": 0.25, "body_yaw_rate_rps": 0.12}
    base = {
        "tracking_vx": 2 * math.exp(-qx),
        "tracking_vy": 1.0,
        "tracking_yaw": math.exp(-qy),
        "height": -0.01,
        "attitude": -0.02,
        "actual_torque": -0.03,
        "action_change": -0.04,
        "leg_speed": -0.05,
        "termination": 0.0,
    }
    spec = s47.ARMS[arm]
    expected = r.apply_tracking_log_sse46(base, qx, qy, alpha=spec["alpha"], beta=spec["beta"])
    row = {
        "info": {
            "training_reward46": {
                "schema": reader_schema(),
                "alpha": spec["alpha"],
                "beta": spec["beta"],
                "reward_arm": arm,
                "normalized_errors": [qx, qy],
            },
            "original_reward_terms42": base,
            "reward_terms": dict(expected),
            "reward_schema": reader_schema(),
            "consumed_command": command,
            "metrics": metrics,
        },
        "reward": sum(expected.values()),
    }
    return row, base, expected


def reader_schema():
    return "d1-b22-training-tracking-log-sse-beta46-v1"


@pytest.mark.parametrize("arm", ["ffyaw_s21", "ffsse_s21"])
def test_reader_recomputes_each_arms_beta(reader, arm):
    qx, qy = r.normalized_errors46(0.25, 0.12, 0.35, 0.0)
    row, base, expected = _row(arm, qx, qy)
    assert reader.reward_expected42(row, base, arm) == expected
    other = "ffsse_s21" if arm == "ffyaw_s21" else "ffyaw_s21"
    with pytest.raises(ValueError):
        reader.reward_expected42(row, base, other)  # stamp beta/arm must match the reader table


def test_parent_pairing_requires_equal_proofs_within_the_seed(tmp_path, monkeypatch, parent):
    monkeypatch.setattr(parent, "OUTPUT", tmp_path)
    proof = {
        "schema": "b22-tracking42-pairing-v1",
        "initial_state": {"qpos": "x"},
        "initial_learning_state": {"p": "x"},
        "first1024_numeric": {"k": "x"},
        "first1024_gaussian": {"k": "x"},
        "seed": 201021,
        "course_arm": "B",
    }
    for arm in s47.ARMS:
        (tmp_path / f"train_{arm}_readback.json").write_text(
            json.dumps({"status": "COMPLETE", "training_valid": True, "pairing42": dict(proof)})
        )
    monkeypatch.setattr(parent, "identity", lambda path: {"sha256": "0" * 64})
    assert parent.verify_training_pair42()["passed"] is True
    changed = json.loads(json.dumps(proof))
    changed["initial_state"] = {"qpos": "different"}
    (tmp_path / "train_ffsse_s21_readback.json").write_text(
        json.dumps({"status": "COMPLETE", "training_valid": True, "pairing42": changed})
    )
    with pytest.raises(ValueError):
        parent.verify_training_pair42()
