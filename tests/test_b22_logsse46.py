"""Pure checks of the 46 arm definition, session gate and independent reward recomputation."""

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import b22_logsse46 as l
import b22_reward46 as r

# COMMAND/METRICS below give qx = ((0.25-0.35)/0.25)^2 = 0.16 and qy = ((0.12-0)/0.4)^2 = 0.09
BASE = {
    "tracking_vx": 2 * math.exp(-0.16),
    "tracking_vy": 1.0,
    "tracking_yaw": math.exp(-0.09),
    "height": -0.01,
    "attitude": -0.02,
    "actual_torque": -0.03,
    "action_change": -0.04,
    "leg_speed": -0.05,
    "termination": 0.0,
}
COMMAND = {"forward_velocity_mps": 0.35, "lateral_velocity_mps": 0.0, "yaw_rate_rps": 0.0}
METRICS = {"body_com_vx_mps": 0.25, "body_yaw_rate_rps": 0.12}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def parent():
    return load_module("p46_parent_test", ROOT / "scripts/run_b22_logsse46.py")


@pytest.fixture(scope="module")
def worker():
    return load_module("w46_worker_test", ROOT / "scripts/worker_b22_logsse46.py")


def test_arm_definition_is_single_variable():
    assert l.ARMS == {"ffsse": {"alpha": 8, "beta": 8, "curriculum": "yaw_emphasis"}}
    assert l.FEEDFORWARD == 3.9 and l.CURRICULUM == "yaw_emphasis"
    assert r.BETAS46 == (0, 8) and r.CAP46 == 4.0


def test_plan_has_bootstrap_and_one_training_stage(parent):
    plan = parent.describe42()
    stages = plan["stages"]
    assert [s["name"] for s in stages] == ["bootstrap", "train_ffsse"]
    assert plan["yaw_feedforward"] == 3.9
    assert plan["curricula"] == {"ffsse": "yaw_emphasis"}
    limits = plan["limits"]
    assert limits["controls"] == sum(s["controls"] for s in stages) == 32768
    assert limits["normal_native"] == sum(s["normal_native"] for s in stages) == 163840
    assert limits["compiler_native"] == sum(s["compiler_native"] for s in stages) == 2
    assert plan["retry_permitted"] is False and plan["automatic_model_replacement"] is False
    assert "46" in plan["execution_contract_id"]


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
            wall_clock_utc="2026-10-19T00:00:00+00:00",
        )
        worker.validate_numeric_session42(session)
        if stage["mode"] == "train":
            assert session["beta"] == 8 and session["alpha"] == 8
            assert session["curriculum"] == "yaw_emphasis" and session["yaw_feedforward"] == 3.9
        else:
            assert session["beta"] is None and session["alpha"] is None


def test_training_wrapper_applies_the_registered_shape(worker):
    class Base:
        def _reward(self, transition, metrics, terminated):
            return dict(BASE)

        def step(self, action):
            fake = SimpleNamespace(decision=SimpleNamespace(motion_command=SimpleNamespace(**COMMAND)))
            self._reward(fake, METRICS, False)  # the real base computes the reward before step returns
            qx, qy = r.normalized_errors46(
                METRICS["body_com_vx_mps"],
                METRICS["body_yaw_rate_rps"],
                COMMAND["forward_velocity_mps"],
                COMMAND["yaw_rate_rps"],
            )
            info = {
                "reward_schema": worker.REWARD_SCHEMA,
                "consumed_command": COMMAND,
                "metrics": METRICS,
                "reward_terms": r.apply_tracking_log_sse46(BASE, qx, qy, alpha=8, beta=8),
            }
            return None, 0.0, False, False, info

    env_type = worker.reward_env42(Base, {"alpha": 8, "beta": 8, "reward_arm": "ffsse"})
    env = object.__new__(env_type)
    transition = SimpleNamespace(
        decision=SimpleNamespace(motion_command=SimpleNamespace(**COMMAND))
    )
    terms = env_type._reward(env, transition, METRICS, False)
    qx, qy = r.normalized_errors46(0.25, 0.12, 0.35, 0.0)
    assert terms == r.apply_tracking_log_sse46(BASE, qx, qy, alpha=8, beta=8)
    _, _, _, _, info = env.step(None)
    assert info["training_reward46"]["beta"] == 8
    assert info["training_reward46"]["normalized_errors"] == [qx, qy]


def test_reader_recomputes_and_rejects_tampering(worker):
    reader = load_module("r46_reader_test", ROOT / "scripts/read_b22_logsse46.py")
    qx, qy = r.normalized_errors46(0.25, 0.12, 0.35, 0.0)
    expected = r.apply_tracking_log_sse46(BASE, qx, qy, alpha=8, beta=8)
    row = {
        "info": {
            "training_reward46": {
                "schema": reader.REWARD_SCHEMA,
                "alpha": 8,
                "beta": 8,
                "reward_arm": "ffsse",
                "normalized_errors": [qx, qy],
            },
            "original_reward_terms42": dict(BASE),
            "reward_terms": dict(expected),
            "reward_schema": reader.REWARD_SCHEMA,
            "consumed_command": COMMAND,
            "metrics": METRICS,
        },
        "reward": sum(expected.values()),
    }
    assert reader.reward_expected42(row, BASE, "ffsse") == expected
    tampered = json.loads(json.dumps(row))
    tampered["info"]["reward_terms"]["tracking_vx"] += 1e-3
    with pytest.raises(ValueError):
        reader.reward_expected42(tampered, BASE, "ffsse")
    wrong_q = json.loads(json.dumps(row))
    wrong_q["info"]["training_reward46"]["normalized_errors"][0] += 1e-6
    with pytest.raises(ValueError):
        reader.reward_expected42(wrong_q, BASE, "ffsse")
    wrong_beta = json.loads(json.dumps(row))
    wrong_beta["info"]["training_reward46"]["beta"] = 0
    with pytest.raises(ValueError):
        reader.reward_expected42(wrong_beta, BASE, "ffsse")


def test_worker_and_reader_agree_on_schema_and_arms(worker, parent):
    reader = load_module("r46_reader_arms", ROOT / "scripts/read_b22_logsse46.py")
    assert worker.CONTRACT == reader.CONTRACT == parent.CONTRACT
    assert worker.REWARD_SCHEMA == reader.REWARD_SCHEMA == r.REWARD_SCHEMA46
    assert set(reader.ARMS) == {"ffsse"}
    for bad in (("train", "ffyaw"), ("train", "control"), ("eval", None), ("train", None)):
        with pytest.raises(ValueError):
            worker.session_defaults42(*bad)
