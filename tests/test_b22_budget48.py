"""Pure checks of the 48 budget experiment and its counted source derivations."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import b22_budget48 as b


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def parent():
    return load_module("p48_parent_test", ROOT / "scripts/run_b22_budget48.py")


@pytest.fixture(scope="module")
def worker():
    return load_module("w48_worker_test", ROOT / "scripts/worker_b22_budget48.py")


def test_arm_definition():
    assert set(b.ARMS) == {"ffyaw_b2_s31", "ffyaw_b2_s41"}
    assert {a["controls"] for a in b.ARMS.values()} == {65536}
    assert {a["rollouts"] for a in b.ARMS.values()} == {64}
    assert {a["seed"] for a in b.ARMS.values()} == {201031, 201041}
    assert {a["beta"] for a in b.ARMS.values()} == {0}
    assert {a["curriculum"] for a in b.ARMS.values()} == {"yaw_emphasis"}
    assert b.FEEDFORWARD == 3.9 and b.ALPHA == 8


def test_learning_derivation_is_counted_and_complete():
    source = (ROOT / "runtime/d1_b22/research/continuation21/learning21.py").read_text()
    derived, proof = b.derive_learning(source)
    assert derived != source and len(proof["replacements"]) == 10
    assert "BudgetSpec(65536, 1024, 256, 4)" in derived
    assert (
        'metadata["training_receipt"]["actual"]["num_timesteps"] == 32768' in derived
    )  # parent history
    assert "BudgetSpec(32768" not in derived
    assert 'actual["num_timesteps"] != 32768' not in derived
    assert 'actual["train_calls"] != 32' not in derived
    assert "require(len(clip_audit.train_records) == 64" in derived
    with pytest.raises(RuntimeError):
        b.derive_learning(
            source.replace("N = 32768", "N = 32768").replace(
                "BudgetSpec(32768, 1024, 256, 4)", "BudgetSpec(1, 1024, 256, 4)"
            )
        )


def test_reader_derivation_is_counted_and_complete():
    source = (ROOT / "runtime/d1_b22/research/continuation22/read_training22.py").read_text()
    derived, proof = b.derive_reader(source)
    assert derived != source and len(proof["replacements"]) == 9
    assert "N = 65536" in derived and "N = 32768" not in derived
    assert '== actual["transitions"] == 65536' in derived
    assert '== actual["train_calls"] == 64' in derived
    assert 'gaussian_blocks"]) == 64' in derived
    assert "64+truncated_episodes" in derived


def test_plan_budgets(parent):
    plan = parent.describe42()
    stages = plan["stages"]
    assert [s["name"] for s in stages] == ["bootstrap", "train_ffyaw_b2_s31", "train_ffyaw_b2_s41"]
    assert [s["controls"] for s in stages[1:]] == [65536, 65536]
    limits = plan["limits"]
    assert limits["controls"] == sum(s["controls"] for s in stages) == 131072
    assert limits["normal_native"] == sum(s["normal_native"] for s in stages) == 655360
    assert limits["compiler_native"] == sum(s["compiler_native"] for s in stages) == 4
    assert plan["yaw_feedforward"] == 3.9


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
            assert session["seed"] == session["ppo_seed"] == b.ARMS[arm]["seed"]
            assert session["controls"] == session["control_limit"] == 65536
            assert session["segments"] == [65536]
            assert session["training_control_limit"] == 65536
            assert session["beta"] == 0


def test_worker_derivations_reach_the_training_path(worker):
    source = (ROOT / "scripts/worker_b22_budget48.py").read_text()
    assert "spec['training']['controls_per_arm_per_stage']=controls" in source
    assert "controls = training_controls48(session)" in source
    assert "train48.derive_learning(source)" in source
    assert "spec['training']['rollouts_per_arm_per_stage']" in source
    reader_source = (ROOT / "scripts/read_b22_budget48.py").read_text()
    assert "train48.derive_reader(source)" in reader_source


def test_reader_arm_table_matches_the_shared_definition():
    reader = load_module("r48_reader_test", ROOT / "scripts/read_b22_budget48.py")
    for arm, spec in b.ARMS.items():
        assert reader.ARMS[arm] == {
            "alpha": spec["alpha"],
            "beta": spec["beta"],
            "seed": spec["seed"],
            "controls": spec["controls"],
        }
    assert reader.CONTRACT == "C48_B22_budget_65536_v1"


def test_bootstrap_example_resolves_to_the_arm_budget(worker):
    example = {
        "mode": "train",
        "reward_arm": "ffyaw_b2_s31",
        "controls": 0,
        "alpha": 8,
        "beta": 0,
        "curriculum": "yaw_emphasis",
    }
    assert worker.training_controls48(example) == 65536
    real = dict(example, controls=65536)
    assert worker.training_controls48(real) == 65536
    with pytest.raises(ValueError):
        worker.training_controls48({"reward_arm": "unknown", "controls": 0})


def test_shared_module_exposes_every_name_the_pipelines_use():
    """Every ``train48.<attr>`` the parent/worker/reader reference must exist on the shared module."""
    import ast

    used = set()
    for name in ("run_b22_budget48.py", "worker_b22_budget48.py", "read_b22_budget48.py"):
        tree = ast.parse((ROOT / "scripts" / name).read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "train48"
            ):
                used.add(node.attr)
    assert {
        "apply_curriculum",
        "derive_learning",
        "derive_reader",
        "keep_record_consistent",
    } <= used
    missing = sorted(name for name in used if not hasattr(b, name))
    assert not missing, missing
    for name in ("ALPHA", "ARMS", "CONTROLS", "CURRICULUM", "ROLLOUTS", "YAW_EMPHASIS_SLOTS"):
        assert hasattr(b, name), name
