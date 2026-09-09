"""Plot plumbing uses synthetic logs; no synthetic result is published as D1 data."""

from __future__ import annotations

import builtins
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "visualize_d1_terrain_curriculum.py"


@pytest.fixture(scope="module")
def visualizer() -> ModuleType:
    specification = importlib.util.spec_from_file_location("terrain_visualization_cli", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def fake_run(tmp_path: Path) -> Path:
    run = tmp_path / "synthetic_run"
    run.mkdir()
    cases = []
    for index, terrain in enumerate(
        (
            {"kind": "flat"},
            {"kind": "bumps", "amplitude_m": 0.005},
            {"kind": "bumps", "amplitude_m": 0.010},
            {"kind": "ramp", "slope_deg": 2.0},
            {"kind": "ramp", "slope_deg": 4.0},
        )
    ):
        cases.append(
            {
                "case_id": f"case_{index}",
                "terrain": terrain,
                "environment_seed": 100 + index,
                "velocity_mps": 0.25,
                "height_m": 0.455,
            }
        )
    protocol = {
        "schema_version": 1,
        "training_conditions": ["flat", "curriculum"],
        "training_seeds": [0, 1],
        "cases": cases,
        "residual_scale_n": [11.25, 20.0],
        "episode_seconds": 0.04,
        "control_dt_s": 0.01,
    }
    (run / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    metrics = []
    for condition, seed in (
        ("zero_residual", None),
        ("flat_rl", 0),
        ("flat_rl", 1),
        ("curriculum_rl", 0),
        ("curriculum_rl", 1),
    ):
        folder = run / "evaluation" / (condition if seed is None else f"{condition}_seed_{seed}")
        folder.mkdir(parents=True)
        for case_index, case in enumerate(cases):
            error = {"zero_residual": 0.1, "flat_rl": 0.08, "curriculum_rl": 0.12}[condition]
            error += case_index * 0.001 + (seed or 0) * 0.005
            rows = [
                {
                    "time_s": step * 0.01,
                    "forward_velocity_mps": 0.25 + error,
                    "command_velocity_mps": 0.25,
                    "velocity_error_mps": error,
                    "clearance_m": 0.456,
                    "command_clearance_m": 0.455,
                    "action_longitudinal": 0.0 if seed is None else 0.1,
                    "action_vertical": 0.0 if seed is None else -0.05,
                }
                for step in range(1, 5)
            ]
            telemetry = folder / f"{case['case_id']}.csv"
            _write_csv(telemetry, rows)
            metrics.append(
                {
                    "condition": condition,
                    "training_seed": seed,
                    "case_id": case["case_id"],
                    "terrain_kind": case["terrain"]["kind"],
                    "environment_seed": case["environment_seed"],
                    "telemetry_csv": str(telemetry.relative_to(run)),
                    "velocity_rmse_mps": error,
                    "control_steps": 4,
                    "completed": 1,
                }
            )
    _write_csv(run / "metrics.csv", metrics)
    return run


def test_pairing_uses_one_baseline_per_case_and_preserves_training_seeds(visualizer, fake_run):
    protocol, indexed, hashes = visualizer.load_experiment(fake_run)
    differences = visualizer.paired_differences(protocol, indexed)
    assert len(indexed) == 25
    assert len(differences) == 20
    assert len(hashes) == 27
    assert sum(key[0] == "zero_residual" for key in indexed) == 5
    chosen = next(
        row
        for row in differences
        if row["condition"] == "flat_rl"
        and row["training_seed"] == 0
        and row["case_id"] == "case_0"
    )
    assert chosen["rmse_difference_mps"] == pytest.approx(-0.02)


@pytest.mark.parametrize(
    "mutation", ("missing", "duplicate", "seed_mismatch", "shared_log", "bad_rmse")
)
def test_missing_duplicate_and_inconsistent_pairs_are_rejected(visualizer, fake_run, mutation):
    rows = _read_csv(fake_run / "metrics.csv")
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0].copy())
    elif mutation == "seed_mismatch":
        rows[0]["environment_seed"] = "9999"
    elif mutation == "shared_log":
        rows[1]["telemetry_csv"] = rows[0]["telemetry_csv"]
    else:
        rows[0]["velocity_rmse_mps"] = "0.001"
    _write_csv(fake_run / "metrics.csv", rows)
    with pytest.raises(ValueError):
        visualizer.load_experiment(fake_run)


def test_nonfinite_or_inconsistent_telemetry_is_rejected(visualizer, fake_run):
    metrics = _read_csv(fake_run / "metrics.csv")
    path = fake_run / metrics[0]["telemetry_csv"]
    rows = _read_csv(path)
    rows[0]["command_velocity_mps"] = "nan"
    _write_csv(path, rows)
    with pytest.raises(ValueError, match="nonfinite"):
        visualizer.load_experiment(fake_run)
    rows[0]["command_velocity_mps"] = "0.3"
    _write_csv(path, rows)
    with pytest.raises(ValueError, match="velocity error"):
        visualizer.load_experiment(fake_run)


def test_representative_selection_uses_protocol_order_not_scores(visualizer, fake_run):
    protocol = json.loads((fake_run / "protocol.json").read_text())
    # Deliberately assign worse scores to the first cases; these are not an input
    # to selection, including if someone adds them to a future protocol schema.
    for case, score in zip(protocol["cases"], (100.0, 10.0, 0.001, 5.0, 0.002), strict=True):
        case["velocity_rmse_mps"] = score
    selected = visualizer.representative_cases(protocol)
    assert {name: case["case_id"] for name, case in selected.items()} == {
        "flat": "case_0",
        "bumps": "case_1",
        "uphill": "case_3",
    }


def test_baseline_must_really_have_zero_residual_actions(visualizer, fake_run):
    path = fake_run / _read_csv(fake_run / "metrics.csv")[0]["telemetry_csv"]
    rows = _read_csv(path)
    rows[0]["action_longitudinal"] = "0.5"
    _write_csv(path, rows)
    with pytest.raises(ValueError, match="nonzero residual"):
        visualizer.load_experiment(fake_run)


def test_all_four_plots_are_real_files_and_default_has_no_rl_or_opengl_import(
    visualizer, fake_run, tmp_path, monkeypatch
):
    output = tmp_path / "figures"
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "stable_baselines3", "mujoco"):
            raise AssertionError(f"default plot should not import {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    assert visualizer.run_visualization(fake_run, output) == output
    expected_pngs = {
        "paired_case_differences.png",
        "terrain_seed_differences.png",
        "representative_bumps.png",
        "representative_uphill.png",
    }
    assert {path.name for path in output.glob("*.png")} == expected_pngs
    for name in expected_pngs:
        with Image.open(output / name) as figure:
            assert figure.width >= 1000
            assert np.asarray(figure).std() > 1
    selection = json.loads((output / "selection.json").read_text())
    assert selection["training_seed"] == 0
    assert selection["scene_replays"] == []
    summary = json.loads((output / "terrain_seed_summary.json").read_text())
    assert len(summary["rows"]) == 12
    assert {row["case_count"] for row in summary["rows"]} == {1, 2}
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["input_sha256"].items():
        assert hashlib.sha256((fake_run / name).read_bytes()).hexdigest() == digest


def test_existing_output_and_invalid_run_do_not_create_or_overwrite(visualizer, fake_run, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        visualizer.run_visualization(fake_run, output)
    rows = _read_csv(fake_run / "metrics.csv")
    _write_csv(fake_run / "metrics.csv", rows[:-1])
    destination = tmp_path / "not_created"
    with pytest.raises(ValueError, match="incomplete pairing"):
        visualizer.run_visualization(fake_run, destination)
    assert not destination.exists()


def test_scene_profile_reads_collision_reference_without_changing_geometry(visualizer):
    coordinates = []

    class ProfilePlant:
        base_position = np.asarray((0.4, 0.2, 0.455))

        def training_ground_reference(self, x, y):
            coordinates.append((x, y))
            # Deliberately unlike the analytic terrain configuration: the plot
            # must use collision samples, not draw a prettier analytic surface.
            return SimpleNamespace(height_m=0.03 + 0.04 * x)

    plant = ProfilePlant()
    before = plant.base_position.copy()
    relative_x, height_mm, description = visualizer._scene_profile(
        plant, {"kind": "bumps", "amplitude_m": 0.005, "wavelength_m": 0.7}
    )
    np.testing.assert_allclose(height_mm, 40.0 * relative_x, atol=1e-12)
    np.testing.assert_array_equal(plant.base_position, before)
    assert len(coordinates) == 402
    assert {coordinate[1] for coordinate in coordinates} == {0.2}
    assert "5.0 mm" in description and "0.70 m" in description


def test_scene_profile_scales_are_shared_and_based_on_geometry_only(visualizer):
    selected = {
        "flat": {"terrain": {"kind": "flat"}},
        "bumps": {"terrain": {"kind": "bumps", "amplitude_m": 0.005}},
        "uphill": {"terrain": {"kind": "ramp", "slope_deg": 3.0}},
    }
    assert visualizer._scene_profile_limit_mm(selected) == 60.0
    selected["uphill"]["velocity_rmse_mps"] = 1000
    assert visualizer._scene_profile_limit_mm(selected) == 60.0


@pytest.fixture
def fake_v2_run(fake_run):
    protocol = json.loads((fake_run / "protocol.json").read_text())
    protocol.update(
        schema_version=2,
        profile="tracking-v2",
        observation_schema="d1-terrain-tracking-oracle-v2",
        reward_schema="d1-terrain-tracking-v2",
        control_schema="d1-lqr-vmc-local-tangent-v2",
        training_conditions=["flat", "mixed", "curriculum"],
    )
    (fake_run / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    rows = _read_csv(fake_run / "metrics.csv")
    added = []
    for row in rows:
        if row["condition"] != "flat_rl":
            continue
        mixed = {
            **row,
            "condition": "mixed_rl",
            "telemetry_csv": row["telemetry_csv"].replace("flat_rl", "mixed_rl"),
        }
        path = fake_run / mixed["telemetry_csv"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((fake_run / row["telemetry_csv"]).read_bytes())
        added.append(mixed)
    _write_csv(fake_run / "metrics.csv", [*rows, *added])
    return fake_run


def test_tracking_v2_pairs_and_plots_all_three_declared_conditions(
    visualizer, fake_v2_run, tmp_path
):
    protocol, indexed, _ = visualizer.load_experiment(fake_v2_run)
    assert visualizer.protocol_conditions(protocol) == ("flat_rl", "mixed_rl", "curriculum_rl")
    assert len(indexed) == 35
    assert len(visualizer.paired_differences(protocol, indexed)) == 30
    output = tmp_path / "v2_figures"
    visualizer.run_visualization(fake_v2_run, output)
    summary = json.loads((output / "terrain_seed_summary.json").read_text())
    assert len(summary["rows"]) == 18
    assert {row["condition"] for row in summary["rows"]} == {"flat_rl", "mixed_rl", "curriculum_rl"}
    with Image.open(output / "paired_case_differences.png") as figure:
        assert figure.width == 18 * 160
    selection = json.loads((output / "selection.json").read_text())
    assert selection["profile"] == "tracking-v2"
    assert selection["training_seed"] == 0
    assert selection["case_ids"] == {"flat": "case_0", "bumps": "case_1", "uphill": "case_3"}


@pytest.mark.parametrize("remove_all", (False, True))
def test_tracking_v2_requires_every_declared_mixed_episode(visualizer, fake_v2_run, remove_all):
    rows = _read_csv(fake_v2_run / "metrics.csv")
    if remove_all:
        rows = [row for row in rows if row["condition"] != "mixed_rl"]
    else:
        rows.pop()
    _write_csv(fake_v2_run / "metrics.csv", rows)
    with pytest.raises(ValueError, match="incomplete pairing"):
        visualizer.load_experiment(fake_v2_run)


@pytest.mark.parametrize("name", ("observation_schema", "reward_schema", "control_schema"))
def test_tracking_v2_rejects_incompatible_schema_tags(visualizer, fake_v2_run, name):
    path = fake_v2_run / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol[name] = "wrong-schema"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="incompatible"):
        visualizer.load_experiment(fake_v2_run)


def test_replay_environment_uses_protocol_version_not_the_latest_available_env(
    visualizer, fake_v2_run
):
    from wheel_legged_control.d1.terrain_env import D1TerrainResidualEnv
    from wheel_legged_control.d1.terrain_tracking_env import D1TerrainTrackingEnv

    protocol = json.loads((fake_v2_run / "protocol.json").read_text())
    assert visualizer.replay_environment(protocol) is D1TerrainTrackingEnv
    assert visualizer.replay_environment({"schema_version": 1}) is D1TerrainResidualEnv


def test_single_declared_v2_condition_does_not_require_phantom_pairs(
    visualizer, fake_v2_run, tmp_path
):
    path = fake_v2_run / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol["training_conditions"] = ["mixed"]
    path.write_text(json.dumps(protocol))
    rows = _read_csv(fake_v2_run / "metrics.csv")
    _write_csv(
        fake_v2_run / "metrics.csv",
        [row for row in rows if row["condition"] in ("zero_residual", "mixed_rl")],
    )
    visualizer.run_visualization(fake_v2_run, tmp_path / "single_condition")
