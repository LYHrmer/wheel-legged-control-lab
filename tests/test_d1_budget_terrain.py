"""Frozen final-test instances for a budget study, not new terrain families."""

import importlib.util
import sys
from dataclasses import astuple
from pathlib import Path

import mujoco
import numpy as np
import pytest

from wheel_legged_control.d1.locomotion_terrain import (
    LOCOMOTION_TERRAIN_SUITES,
    D1LocomotionTerrainConfig,
    add_locomotion_terrain,
    locomotion_ground_reference,
    locomotion_terrain_configs,
)

FINAL_ROWS = (
    ("s_bend", 1.20, 0.55, 0.0080, 0.74, 1.04, 1.70, 1.90, 0.0080),
    ("diagonal", -1.20, -0.55, 0.0075, 0.84, 1.14, 2.30, 2.50, 0.0075),
)


@pytest.fixture(scope="module")
def experiment():
    path = Path(__file__).resolve().parents[1] / "scripts/run_d1_locomotion_experiment.py"
    spec = importlib.util.spec_from_file_location("budget_terrain_subject", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("split", ("train", "development"))
def test_budget_study_keeps_training_and_development_instances(split):
    assert locomotion_terrain_configs(split, "budget_compare_v1") == (
        locomotion_terrain_configs(split, "action_compare_v1")
    )


def test_final_instances_match_preregistered_values_and_are_not_old_instances():
    new = locomotion_terrain_configs("holdout", "budget_compare_v1")
    assert tuple(astuple(config)[:-1] for config in new) == FINAL_ROWS
    old = {
        config.to_json()
        for suite in ("v1", "action_compare_v1")
        for split in ("train", "development", "holdout")
        for config in locomotion_terrain_configs(split, suite)
    }
    assert all(config.to_json() not in old for config in new)
    # This deliberately reuses known layout families; only instances are fresh.
    assert [config.layout for config in new] == ["s_bend", "diagonal"]


@pytest.mark.parametrize(
    "suite,split,expected",
    [
        ("v1", "development", (17, 29)),
        ("v1", "holdout", (617, 629)),
        ("action_compare_v1", "development", (1017, 1029)),
        ("action_compare_v1", "holdout", (1617, 1629)),
        ("budget_compare_v1", "development", (1017, 1029)),
        ("budget_compare_v1", "flat", (1017, 1029)),
        ("budget_compare_v1", "holdout", (4617, 4629)),
    ],
)
def test_evaluation_seed_routing_keeps_old_cases(experiment, suite, split, expected):
    assert experiment.evaluation_seeds(split, suite) == expected


def test_new_suite_is_explicit_and_default_remains_v1(experiment, tmp_path):
    defaults = experiment.parser().parse_args(["train", "--output", str(tmp_path)])
    assert defaults.terrain_suite == "v1"
    assert "budget_compare_v1" in LOCOMOTION_TERRAIN_SUITES
    selected = experiment.parser().parse_args(
        ["evaluate", "--output", str(tmp_path), "--terrain-suite", "budget_compare_v1"]
    )
    assert selected.terrain_suite == "budget_compare_v1"


@pytest.mark.parametrize("row", FINAL_ROWS)
def test_new_final_instances_compile_and_keep_flat_spawn_without_running_episode(row):
    config = D1LocomotionTerrainConfig(*row)
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><geom name="floor" type="plane" size="0 0 .1"/>'
        "</worldbody></mujoco>"
    )
    add_locomotion_terrain(spec, config)
    model = spec.compile()
    assert model.hfield_data.shape == (241 * 121,)
    assert np.isfinite(model.hfield_data).all()
    ground = locomotion_ground_reference(model, -3.8, 0.0)
    assert ground.height_m == pytest.approx(0.0, abs=1e-7)
    assert ground.pitch_rad == pytest.approx(0.0, abs=1e-7)
    assert ground.roll_rad == pytest.approx(0.0, abs=1e-7)
