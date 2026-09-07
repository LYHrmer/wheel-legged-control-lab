"""CLI and artifact integrity, not a single-seed control-performance benchmark."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from wheel_legged_control.actuator_bench import ActuatorParameters
from wheel_legged_control.actuator_identification import load_actuator_log, predict_actuator_log

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_actuator_identification.py"


@pytest.fixture(scope="module")
def cli_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location("actuator_experiment_cli", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def experiment_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("actuator_cli") / "experiment"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / ".local-deps"), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output),
            "--bench",
            "wheel",
            "--scenario",
            "matched",
            "--duration",
            "0.5",
            "--max-delay-steps",
            "0",
            "--max-nfev",
            "1",
            "--noise-scale",
            "0",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Results:" in completed.stdout
    return output


def test_cli_produces_a_completed_explicitly_offline_experiment(experiment_output: Path) -> None:
    experiment = json.loads((experiment_output / "experiment.json").read_text(encoding="utf-8"))
    summary = json.loads((experiment_output / "summary.json").read_text(encoding="utf-8"))
    assert experiment["schema_version"] == 1
    assert experiment["data_origin"] == "synthetic_only"
    assert experiment["arguments"]["duration"] == 0.5
    assert experiment["arguments"]["max_nfev"] == 1
    assert "offline prediction only" in experiment["protocol"]["scope"]
    assert summary["status"] == "completed"
    assert [case["case"] for case in summary["cases"]] == ["wheel_matched"]
    assert "not closed-loop" in summary["claims"]
    assert len(summary["metrics"]) == 8  # Four trajectories, two prediction models.
    # The one-evaluation budget is intentionally only a pipeline smoke test.
    # No convergence, parameter accuracy, or control improvement is required.


def test_manifest_covers_every_artifact_with_matching_sha256(experiment_output: Path) -> None:
    manifest = json.loads((experiment_output / "manifest.json").read_text(encoding="utf-8"))
    files = {
        str(path.relative_to(experiment_output))
        for path in experiment_output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert set(manifest["sha256"]) == files
    assert {"experiment.json", "summary.json", "metrics.csv"}.issubset(files)
    for relative, expected in manifest["sha256"].items():
        assert hashlib.sha256((experiment_output / relative).read_bytes()).hexdigest() == expected


def test_all_four_split_logs_and_full_predictions_are_saved(experiment_output: Path) -> None:
    destination = experiment_output / "wheel_matched"
    expected_splits = ("calibration", "calibration", "validation", "holdout")
    for index, split in enumerate(expected_splits):
        log = load_actuator_log(destination / f"{split}_{index}.csv")
        assert log.split == split
        assert len(log.commands) == 250
        assert len(log.positions) == len(log.velocities) == 251
        assert log.config.dt == 0.002
        prediction = np.genfromtxt(
            destination / f"{split}_{index}_predictions.csv", delimiter=",", names=True
        )
        assert prediction.shape == (251,)
        assert prediction.dtype.names == (
            "time_s",
            "measured_position_rad",
            "measured_velocity_rad_s",
            "nominal_position_rad",
            "nominal_velocity_rad_s",
            "fitted_position_rad",
            "fitted_velocity_rad_s",
        )
        np.testing.assert_array_equal(prediction["measured_position_rad"], log.positions)
        np.testing.assert_array_equal(prediction["measured_velocity_rad_s"], log.velocities)
        np.testing.assert_allclose(prediction["time_s"], np.arange(251) * 0.002, atol=1e-15)
        for field in prediction.dtype.names:
            assert np.isfinite(prediction[field]).all()
    with (experiment_output / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    assert len(metrics) == 8
    assert {row["split"] for row in metrics} == {"calibration", "validation", "holdout"}
    assert {row["model"] for row in metrics} == {"nominal", "fitted"}


def test_nominal_predictions_use_declared_defaults_not_generator_truth(
    experiment_output: Path,
) -> None:
    experiment = json.loads((experiment_output / "experiment.json").read_text(encoding="utf-8"))
    assert experiment["nominal_parameters"] == asdict(ActuatorParameters())
    assert experiment["nominal_parameters"] != experiment["generator_truth_not_available_to_fitter"]
    destination = experiment_output / "wheel_matched"
    log = load_actuator_log(destination / "holdout_3.csv")
    prediction = np.genfromtxt(destination / "holdout_3_predictions.csv", delimiter=",", names=True)
    actual = np.column_stack(
        (prediction["nominal_position_rad"], prediction["nominal_velocity_rad_s"])
    )
    expected = predict_actuator_log(log, ActuatorParameters())
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert not np.allclose(actual[:, 1], log.velocities)


def test_held_out_plot_and_fit_record_are_present(experiment_output: Path) -> None:
    destination = experiment_output / "wheel_matched"
    image = (destination / "prediction.png").read_bytes()
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(image) > 1000
    fit = json.loads((destination / "fit.json").read_text(encoding="utf-8"))
    assert len(fit["candidates"]) == 1
    assert fit["candidates"][0]["parameters"]["delay_steps"] == 0
    assert fit["candidates"][0]["nfev"] == 1
    assert isinstance(fit["candidates"][0]["optimizer_success"], bool)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--duration", "0.49"),
        ("--duration", "nan"),
        ("--duration", "inf"),
        ("--noise-scale", "-0.1"),
        ("--noise-scale", "nan"),
        ("--noise-scale", "inf"),
        ("--seed", "-1"),
        ("--seed", "1.5"),
        ("--max-delay-steps", "-1"),
        ("--max-delay-steps", "0.5"),
        ("--max-nfev", "0"),
        ("--max-nfev", "-1"),
        ("--max-nfev", "nan"),
    ],
)
def test_invalid_arguments_do_not_create_an_experiment_directory(
    cli_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    value: str,
) -> None:
    output = tmp_path / "must_not_exist"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(output), option, value])
    with pytest.raises(SystemExit) as stopped:
        cli_module.main()
    assert stopped.value.code == 2
    assert not output.exists()


def test_cli_refuses_to_overwrite_an_existing_experiment(
    cli_module: ModuleType, experiment_output: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = {
        path.relative_to(experiment_output): path.read_bytes()
        for path in experiment_output.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(experiment_output)])
    with pytest.raises(SystemExit) as stopped:
        cli_module.main()
    assert stopped.value.code == 2
    after = {
        path.relative_to(experiment_output): path.read_bytes()
        for path in experiment_output.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_help_exits_successfully_without_requiring_output(
    cli_module: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--help"])
    with pytest.raises(SystemExit) as stopped:
        cli_module.main()
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "--output" in help_text
    assert "--max-nfev" in help_text
    assert "--scenario" in help_text
