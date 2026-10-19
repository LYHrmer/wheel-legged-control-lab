"""Pure checks of the 44 curricula against the real frozen recipes module (no model, no physics)."""

import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import b22_yawtrain44 as y

MANIFEST = ROOT / "runtime/d1_b22/runtime_manifest_v27.json"


@pytest.fixture(scope="module")
def recipes():
    """Execute the frozen recipes22 source with the sealed spec, the way the 42 worker does."""
    paths = json.loads(MANIFEST.read_text())["runtime_environment"]["PYTHONPATH"].split(os.pathsep)
    saved = list(sys.path)
    sys.path[:0] = [str(ROOT / "runtime/d1_b22/research/continuation22"), *(p for p in paths if p)]
    try:
        spec = json.loads(
            (ROOT / "results/d1_main_route_execution_20260930_22/spec22.json").read_text()
        )
        module = types.ModuleType("recipes22_for_test")
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


def yaw_episode_count(recipes):
    return sum(1 for i in range(8, 80) if choice(recipes, i)["yaw_segments"])


def test_tick_fraction_arithmetic():
    assert y.yaw_tick_fraction(y.ORIGINAL_SLOTS) == pytest.approx(0.05)
    assert y.yaw_tick_fraction(y.YAW_EMPHASIS_SLOTS) == pytest.approx(0.20)


def test_original_curriculum_is_a_noop_on_the_real_recipes(recipes):
    before = [recipes.select_episode(i, "B").command_sha256 for i in range(80)]
    y.apply_curriculum(recipes, "original")
    assert [recipes.select_episode(i, "B").command_sha256 for i in range(80)] == before


def test_yaw_emphasis_changes_only_the_three_intended_slots(recipes):
    base = {i: choice(recipes, i) for i in range(80)}
    base_yaw = yaw_episode_count(recipes)
    y.apply_curriculum(recipes, "yaw_emphasis")
    new = {i: choice(recipes, i) for i in range(80)}
    assert yaw_episode_count(recipes) > 2 * base_yaw
    for i in range(80):
        if i < 8 or (i - 8) % 8 in (2, 3, 5, 6, 7):
            assert new[i] == base[i], i  # warmup and untouched slots are identical
        else:
            assert new[i]["terrain"] == base[i]["terrain"]
            assert new[i]["speed_mps"] == pytest.approx(
                base[i]["speed_mps"]
            )  # speed is drawn first
            assert new[i]["yaw_segments"] and not base[i]["yaw_segments"]
            amplitudes = {abs(s[2]) for s in new[i]["yaw_segments"]}
            assert len(amplitudes) == 1 and 0.15 <= amplitudes.pop() <= 0.3


def test_frozen_table_mismatch_is_refused(recipes):
    saved = recipes._SLOTS
    recipes._SLOTS = saved[:-1]
    try:
        with pytest.raises(RuntimeError):
            y.apply_curriculum(recipes, "yaw_emphasis")
    finally:
        recipes._SLOTS = saved


def test_unknown_curriculum_is_refused():
    with pytest.raises(ValueError):
        y.slots_for("other")


def test_derived_verifier_has_exactly_one_edit():
    source = (ROOT / "runtime/d1_b22/research/continuation18/verify_control18.py").read_text()
    derived = y.derived_verifier_source(source, 3.9)
    assert derived != source and "3.9*servo_yaw_rps" in derived
    assert derived.replace("3.9*servo_yaw_rps + 4.0", "servo_yaw_rps + 4.0") == source
    with pytest.raises(RuntimeError):
        y.derived_verifier_source("nothing here", 3.9)


def load_module(name, path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def parent():
    return load_module("p44_parent_test", ROOT / "scripts/run_b22_yawtrain44.py")


def test_plan_has_bootstrap_and_exactly_two_training_stages_with_independent_budget(parent):
    plan = parent.describe42()
    stages = plan["stages"]
    assert [s["name"] for s in stages] == ["bootstrap", "train_ffcontrol", "train_ffyaw"]
    assert [s["mode"] for s in stages] == ["bootstrap", "train", "train"]
    assert {s["alpha"] for s in stages[1:]} == {y.ALPHA}
    assert plan["yaw_feedforward"] == y.FEEDFORWARD
    assert plan["curricula"] == {"ffcontrol": "original", "ffyaw": "yaw_emphasis"}
    limits = plan["limits"]
    assert limits["controls"] == sum(s["controls"] for s in stages) == 65536
    assert limits["normal_native"] == sum(s["normal_native"] for s in stages) == 327680
    assert limits["compiler_native"] == sum(s["compiler_native"] for s in stages) == 4
    assert plan["retry_permitted"] is False and plan["automatic_model_replacement"] is False
    assert "eval" not in [s["mode"] for s in stages]


def test_no_previous_execution_chain_is_bound(parent):
    assert parent.predecessor_failure42()["inputs"] == {}
    assert parent.unused_budget_origin42()["previous_executions"] == 0


def test_worker_session_defaults_carry_the_arm_definition():
    worker = load_module("w44_worker_test", ROOT / "scripts/worker_b22_yawtrain44.py")
    for arm, spec in y.ARMS.items():
        session = worker.session_defaults42("train", arm)
        assert session["alpha"] == spec["alpha"] and session["curriculum"] == spec["curriculum"]
        assert session["yaw_feedforward"] == y.FEEDFORWARD and session["controls"] == 32768
    boot = worker.session_defaults42("bootstrap", None)
    assert boot["curriculum"] is None and boot["yaw_feedforward"] is None and boot["controls"] == 0
    for bad in (("train", "control"), ("train", "deficit8"), ("eval", None), ("train", None)):
        with pytest.raises(ValueError):
            worker.session_defaults42(*bad)


def test_reader_arms_match_the_shared_definition():
    reader = load_module("r44_reader_test", ROOT / "scripts/read_b22_yawtrain44.py")
    assert set(reader.ARMS) == set(y.ARMS) and all(v == y.ALPHA for v in reader.ARMS.values())
    assert (
        reader.CONTRACT == load_module("w44_c", ROOT / "scripts/worker_b22_yawtrain44.py").CONTRACT
    )


def test_record_wrapper_restates_the_yaw_law():
    from types import SimpleNamespace

    import numpy as np

    controller = SimpleNamespace(
        _nominal=SimpleNamespace(yaw_feedback_gain=4.0),
        last_record={"nominal_unclipped_yaw_rps": -1.0},
    )
    controller.compute = lambda *_: "torque"
    y.keep_record_consistent(controller, 3.9)
    command = SimpleNamespace(yaw_rate_rps=0.1)
    state = SimpleNamespace(base_angular_velocity_body=np.array([0.0, 0.0, 0.05]))
    assert controller.compute(command, state, None, None) == "torque"
    expected = 3.9 * 0.1 + 4.0 * (0.1 - 0.05)
    assert controller.last_record["nominal_unclipped_yaw_rps"] == pytest.approx(expected)
    assert controller.last_record["nominal_unlimited_yaw_rps"] == pytest.approx(expected)
    assert controller.last_record["command_feedforward_gain"] == 3.9
