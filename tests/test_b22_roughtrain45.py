"""Pure checks of the 45 rough-emphasis curriculum against the real frozen recipes module."""

import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import b22_roughtrain45 as r
import b22_yawtrain44 as y

MANIFEST = ROOT / "runtime/d1_b22/runtime_manifest_v27.json"


@pytest.fixture(scope="module")
def recipes():
    """Execute the frozen recipes22 source with the sealed spec, the way the 42 worker does."""
    registered_root = "/home/lyh/wheel-legged-control-lab"
    entries = json.loads(MANIFEST.read_text())["runtime_environment"]["PYTHONPATH"].split(
        os.pathsep
    )
    rebased = [
        str(ROOT) + p[len(registered_root) :] if p.startswith(registered_root) else p
        for p in entries
    ]
    saved = list(sys.path)
    sys.path[:0] = [
        str(ROOT / "runtime/d1_b22/research/continuation22"),
        *(p for p in rebased if p and Path(p).is_dir() and str(ROOT) in p),
    ]
    try:
        spec = json.loads(
            (ROOT / "results/d1_main_route_execution_20260930_22/spec22.json").read_text()
        )
        module = types.ModuleType("recipes22_for_test45")
        source = (ROOT / "runtime/d1_b22/research/continuation22/recipes22.py").read_text()
        source = source.replace(
            "SPEC = json.loads((Path(__file__).parent / 'spec22.json').read_text())", "SPEC = _spec"
        )
        assert "SPEC = _spec" in source, "frozen recipes22 spec line changed"
        module.__dict__["_spec"] = spec
        module.__file__ = str(ROOT / "runtime/d1_b22/research/continuation22/recipes22.py")
        exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102 -- frozen source
        yield module
    finally:
        sys.path[:] = saved


def choice(recipes, index):
    return json.loads(recipes.select_episode(index, "B").choice_json)


def rough_episodes(recipes):
    return sum(1 for i in range(8, 80) if choice(recipes, i)["terrain"] == "rough")


def test_rough_fraction_arithmetic():
    assert r.rough_episode_fraction(y.ORIGINAL_SLOTS) == pytest.approx(1 / 8)
    assert r.rough_episode_fraction(y.YAW_EMPHASIS_SLOTS) == pytest.approx(1 / 8)
    assert r.rough_episode_fraction(r.ROUGH_EMPHASIS_SLOTS) == pytest.approx(3 / 8)
    assert sum(1 for _, _, yaw in r.ROUGH_EMPHASIS_SLOTS if yaw) == 4  # yaw exposure kept


def test_rough_emphasis_changes_only_slots_0_and_6_relative_to_yaw_emphasis(recipes):
    y.apply_curriculum(recipes, "yaw_emphasis")
    reference = {i: choice(recipes, i) for i in range(80)}
    recipes._SLOTS = y.ORIGINAL_SLOTS
    r.apply_curriculum(recipes, "rough_emphasis")
    new = {i: choice(recipes, i) for i in range(80)}
    assert rough_episodes(recipes) == 3 * ((80 - 8) // 8)
    for i in range(80):
        if i < 8 or (i - 8) % 8 not in (0, 6):
            assert new[i] == reference[i], (
                i
            )  # warmup and every other slot identical to the 44 yaw arm
        else:
            assert new[i]["terrain"] == "rough"
            assert 0.25 <= new[i]["speed_mps"] <= 0.4
            assert bool(new[i]["yaw_segments"]) is ((i - 8) % 8 == 0)


def test_frozen_table_mismatch_and_unknown_curriculum_are_refused(recipes):
    saved = recipes._SLOTS
    recipes._SLOTS = saved[:-1]
    try:
        with pytest.raises(RuntimeError):
            r.apply_curriculum(recipes, "rough_emphasis")
    finally:
        recipes._SLOTS = saved
    with pytest.raises(ValueError):
        r.slots_for("other")


def load_module(name, path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def parent():
    return load_module("p45_parent_test", ROOT / "scripts/run_b22_roughtrain45.py")


def test_plan_has_bootstrap_and_one_training_stage_with_independent_budget(parent):
    plan = parent.describe42()
    stages = plan["stages"]
    assert [s["name"] for s in stages] == ["bootstrap", "train_ffrough"]
    assert plan["yaw_feedforward"] == y.FEEDFORWARD
    assert plan["curricula"] == {"ffrough": "rough_emphasis"}
    limits = plan["limits"]
    assert limits["controls"] == sum(s["controls"] for s in stages) == 32768
    assert limits["normal_native"] == sum(s["normal_native"] for s in stages) == 163840
    assert limits["compiler_native"] == sum(s["compiler_native"] for s in stages) == 2
    assert plan["retry_permitted"] is False and plan["automatic_model_replacement"] is False
    assert "45" in plan["execution_contract_id"]


def test_every_stage_session_passes_the_workers_numeric_gate(parent, monkeypatch):
    worker = load_module("w45_session_gate", ROOT / "scripts/worker_b22_roughtrain45.py")
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
            assert session["curriculum"] == "rough_emphasis" and session["yaw_feedforward"] == 3.9


def test_worker_and_reader_arm_definitions_agree():
    worker = load_module("w45_arms", ROOT / "scripts/worker_b22_roughtrain45.py")
    reader = load_module("r45_arms", ROOT / "scripts/read_b22_roughtrain45.py")
    assert set(r.ARMS) == set(reader.ARMS) == {"ffrough"}
    for bad in (("train", "ffyaw"), ("train", "control"), ("eval", None), ("train", None)):
        with pytest.raises(ValueError):
            worker.session_defaults42(*bad)
    assert reader.CONTRACT == worker.CONTRACT
