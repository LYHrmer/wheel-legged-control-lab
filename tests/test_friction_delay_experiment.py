"""Real tiny two-family run of the friction-delay experiment; no faked artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from scripts import evaluate_friction_delay as experiment

TINY = {
    "frictions": [0.0],
    "actual_delays": [0],
    "seeds": [53],
    "families": ["standard", "low_speed"],
    "calibration_duration": 0.5,
    "duration": 1.0,
    "max_delay_steps": 0,
    "max_nfev": 1,
    "noise_scale": 1.0,
}
CELLS = ("friction0_delay0_seed53_standard", "friction0_delay0_seed53_low_speed")


def tiny_args(output, **overrides) -> argparse.Namespace:
    return argparse.Namespace(output=output, **{**TINY, **overrides})


def read_gz(path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """Run the mandatory tiny grid once, spying on the real fitter's inputs."""
    output = tmp_path_factory.mktemp("runs") / "tiny"
    real_fitter = experiment.fit_encoder_parameters
    seen = []

    class VelocityForbiddenLog(experiment.EncoderLog):
        @property
        def velocities(self):
            raise AssertionError("position-only fitting requested a measured velocity")

    def spy(logs, **kwargs):
        seen.append(([log.split for log in logs], [log.name for log in logs], dict(kwargs)))
        assert all(not hasattr(log, "velocities") for log in logs)
        guarded = [
            VelocityForbiddenLog(
                log.config, log.commands, log.positions,
                known_initial_rest=log.known_initial_rest, name=log.name, split=log.split,
            )
            for log in logs
        ]
        return real_fitter(guarded, **kwargs)

    experiment.fit_encoder_parameters = spy
    try:
        result = experiment.run(tiny_args(output))
    finally:
        experiment.fit_encoder_parameters = real_fitter
    return result, seen


def test_public_interface_and_observed_counts(tiny_run):
    output, _ = tiny_run
    assert callable(experiment.main) and callable(experiment.run)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["observed_counts"] == {
        "cells": 2,
        "fits": 2,
        "candidates": 2,
        "prediction_metric_rows": 16,
        "control_cases": 12,
    }
    assert summary["observed_labels"]["families"] == ["low_speed", "standard"]
    assert sorted(summary["observed_labels"]["splits"]) == ["calibration", "holdout", "validation"]
    assert summary["observed_labels"]["controllers"] == ["fitted_ff", "nominal_ff"]
    assert any("position-only" in note for note in summary["limitations"])
    assert any("paired references" in note for note in summary["limitations"])


def test_only_calibration_logs_reach_the_real_fitter(tiny_run):
    _, seen = tiny_run
    assert len(seen) == 2
    for splits, names, kwargs in seen:
        assert splits == ["calibration", "calibration"]
        assert names == ["calibration_0", "calibration_1"]
        assert kwargs == {"max_delay_steps": 0, "max_nfev": 1}


def test_logs_are_position_only_with_t_commands_and_t_plus_one_positions(tiny_run):
    output, _ = tiny_run
    steps = round(TINY["calibration_duration"] / 0.002)
    for cell in CELLS:
        for index, split in enumerate(experiment.SPLITS):
            path = output / cell / f"{split}_{index}.csv"
            text = path.read_text()
            assert "velocity" not in text and "torque_nm" in text
            assert "applied_torque" not in text and "stribeck" not in text
            header = json.loads(text.splitlines()[0][2:])
            assert header["known_initial_rest"] is True and header["split"] == split
            assert not {"armature", "damping", "coulomb_friction", "delay_steps"} & set(header)
            log = experiment.load_encoder_log(path)
            assert len(log.commands) == steps and len(log.positions) == steps + 1


def test_validation_and_holdout_are_byte_identical_across_families(tiny_run):
    output, _ = tiny_run
    for index, split in enumerate(experiment.SPLITS):
        first = (output / CELLS[0] / f"{split}_{index}.csv").read_bytes()
        second = (output / CELLS[1] / f"{split}_{index}.csv").read_bytes()
        if split == "calibration":
            assert first != second, "calibration families must actually differ"
        else:
            assert first == second, f"{split} must be paired across families"


def test_prediction_csv_timing_and_metric_recomputation(tiny_run):
    output, _ = tiny_run
    metrics = read_gz(output / "prediction_metrics.csv.gz")
    assert len(metrics) == 16
    steps = round(TINY["calibration_duration"] / 0.002)
    for cell in CELLS:
        for index, split in enumerate(experiment.SPLITS):
            rows = read_gz(output / cell / f"{split}_{index}_predictions.csv.gz")
            assert len(rows) == steps + 1
            np.testing.assert_allclose(
                [float(row["time_s"]) for row in rows], np.arange(steps + 1) * .002,
                rtol=0, atol=1e-15,
            )
            measured = np.array([float(row["measured_position_rad"]) for row in rows])
            log = experiment.load_encoder_log(output / cell / f"{split}_{index}.csv")
            np.testing.assert_array_equal(measured, log.positions)
            for model in ("nominal", "selected"):
                predicted = np.array([float(row[f"{model}_position_rad"]) for row in rows])
                assert predicted[0] == measured[0]
                expected = float(np.sqrt(np.mean((predicted[1:] - measured[1:]) ** 2)))
                metric = next(
                    row
                    for row in metrics
                    if row["cell"] == cell
                    and row["trajectory"] == f"{split}_{index}"
                    and row["model"] == model
                )
                assert int(metric["samples"]) == steps
                assert float(metric["position_rmse_rad"]) == pytest.approx(expected, rel=1e-9)


def test_every_candidate_calibration_prediction_is_archived_and_loss_recomputable(tiny_run):
    output, _ = tiny_run
    candidates = read_gz(output / "candidates.csv.gz")
    assert len(candidates) == 2
    for candidate in candidates:
        assert candidate["message"] and candidate["optimizer_success"] in ("True", "False")
        assert int(candidate["nfev"]) >= 1
        delay = int(candidate["candidate_delay_steps"])
        rows = read_gz(
            output / candidate["cell"] / f"candidate_delay{delay}_calibration_predictions.csv.gz"
        )
        parts = []
        for trajectory in ("calibration_0", "calibration_1"):
            trace = [row for row in rows if row["trajectory"] == trajectory]
            assert len(trace) == round(TINY["calibration_duration"] / 0.002) + 1
            measured = np.array([float(row["measured_position_rad"]) for row in trace])
            predicted = np.array([float(row["predicted_position_rad"]) for row in trace])
            parts.append(((predicted[1:] - measured[1:]) / 0.05) ** 2)
        raw_loss = float(np.mean(np.concatenate(parts)))
        assert float(candidate["recomputed_calibration_loss"]) == pytest.approx(raw_loss, rel=1e-9)
        assert float(candidate["reported_calibration_loss"]) == pytest.approx(raw_loss, rel=1e-6)


def test_fit_records_report_status_bounds_and_delay_labels(tiny_run):
    output, _ = tiny_run
    fits = read_gz(output / "fits.csv.gz")
    assert len(fits) == 2
    assert {row["family"] for row in fits} == {"standard", "low_speed"}
    for row in fits:
        record = json.loads((output / row["cell"] / "fit.json").read_text())
        assert len(record["candidates"]) == 1
        assert record["actual_delay_steps"] == 0
        assert record["actual_delay_out_of_search_range"] is False
        assert record["delay_compensated"] is False
        assert record["delay_error_steps"] == int(row["delay_error_steps"])
        assert isinstance(record["continuous_parameters_at_bounds"], list)
        assert record["jacobian_rank"] >= 0 and record["jacobian_singular_values"]
        # nfev=1 cannot converge; this must be recorded, never claimed as good.
        assert record["selected_optimizer_success"] is False
        assert row["selected_optimizer_success"] == "False"


def test_control_rows_pair_noise_and_reference_and_recompute_rmse(tiny_run):
    output, _ = tiny_run
    metrics = read_gz(output / "control_metrics.csv.gz")
    assert len(metrics) == 12
    assert {row["controller"] for row in metrics} == {"nominal_ff", "fitted_ff"}
    steps = round(TINY["duration"] / 0.002)
    for cell in CELLS:
        for profile in experiment.PROFILES:
            traces = {
                model: read_gz(output / cell / f"{profile}_{model}_rows.csv.gz")
                for model in ("nominal_ff", "fitted_ff")
            }
            for model, rows in traces.items():
                assert len(rows) == steps
                np.testing.assert_allclose(
                    [float(row["time_s"]) for row in rows], np.arange(steps) * .002,
                    rtol=0, atol=1e-15,
                )
                reference = np.array([float(row["reference_velocity_rad_s"]) for row in rows])
                actual = np.array([float(row["actual_velocity_rad_s"]) for row in rows])
                measured = np.array([float(row["measured_velocity_rad_s"]) for row in rows])
                following = np.array([float(row["next_velocity_rad_s"]) for row in rows])
                np.testing.assert_array_equal(actual[1:], following[:-1])
                np.testing.assert_array_equal(
                    [float(row["error_rad_s"]) for row in rows], reference - measured,
                )
                error = np.array([float(row["tracking_error_rad_s"]) for row in rows])
                assert np.allclose(error, reference - actual, rtol=0, atol=1e-12)
                metric = next(
                    row
                    for row in metrics
                    if row["cell"] == cell
                    and row["profile"] == profile
                    and row["controller"] == model
                )
                assert int(metric["steps"]) == steps
                assert float(metric["velocity_rmse_rad_s"]) == pytest.approx(
                    float(np.sqrt(np.mean(error**2))), rel=1e-9
                )
            noise = [
                np.array(
                    [
                        float(row["measured_velocity_rad_s"]) - float(row["actual_velocity_rad_s"])
                        for row in rows
                    ]
                )
                for rows in traces.values()
            ]
            expected_noise = np.random.default_rng(np.random.SeedSequence((53, 314))).normal(
                size=steps
            ) * .002
            for values in noise:
                np.testing.assert_allclose(values, expected_noise, rtol=0, atol=4e-15)
            assert np.allclose(noise[0], noise[1], rtol=0, atol=4e-15)
            references = [
                np.array([float(row["reference_velocity_rad_s"]) for row in rows])
                for rows in traces.values()
            ]
            assert np.allclose(references[0], references[1], rtol=0, atol=0)


def test_nominal_reference_rows_are_duplicated_across_families(tiny_run):
    output, _ = tiny_run
    for profile in experiment.PROFILES:
        traces = [
            (output / cell / f"{profile}_nominal_ff_rows.csv.gz").read_bytes() for cell in CELLS
        ]
        assert traces[0] == traces[1]


def test_source_archive_hashes_consistency_and_manifest_coverage(tiny_run):
    output, _ = tiny_run
    source = json.loads((output / "source.json").read_text())
    assert set(source["sha256"]) == {
        "scripts/evaluate_friction_delay.py", "scripts/evaluate_wheel_control.py",
        "scripts/run_actuator_identification.py", "src/wheel_legged_control/__init__.py",
        "src/wheel_legged_control/controllers.py", "src/wheel_legged_control/model.py",
        "src/wheel_legged_control/rewards.py", "src/wheel_legged_control/encoder_identification.py",
        "src/wheel_legged_control/actuator_identification.py", "src/wheel_legged_control/actuator_bench.py",
        "src/wheel_legged_control/wheel_control.py", "src/wheel_legged_control/provenance.py",
        "pyproject.toml",
    }
    assert "scripts/evaluate_friction_delay.py" in source["sha256"]
    with tarfile.open(output / "source.tar.gz", "r:gz") as archive:
        members = {member.name for member in archive.getmembers()}
        assert members == set(experiment.SOURCE_FILES)
        for name in experiment.SOURCE_FILES:
            payload = archive.extractfile(name).read()
            assert hashlib.sha256(payload).hexdigest() == source["sha256"][name]
    consistency = json.loads((output / "source_consistency.json").read_text())
    assert consistency["unchanged"] is True
    assert consistency["sha256"] == source["sha256"]
    manifest = json.loads((output / "manifest.json").read_text())["sha256"]
    files = {
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert set(manifest) == files
    for name, digest in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["expected_counts"]["fits"] == 2
    assert protocol["bench_config"] == {"kind": "wheel", "dt": 0.002, "torque_limit_nm": 2.0}


def test_default_grid_declares_the_frozen_54_fit_protocol():
    assert experiment.DEFAULT_FRICTIONS == (0.0, 0.04, 0.08)
    assert experiment.DEFAULT_ACTUAL_DELAYS == (0, 2, 4)
    assert experiment.DEFAULT_SEEDS == (53, 67, 79)
    assert experiment.FAMILIES == ("standard", "low_speed")
    cells = (
        len(experiment.DEFAULT_FRICTIONS)
        * len(experiment.DEFAULT_ACTUAL_DELAYS)
        * len(experiment.DEFAULT_SEEDS)
        * len(experiment.FAMILIES)
    )
    assert cells == 54
    assert cells * 5 == 270
    assert cells * len(experiment.SPLITS) * 2 == 432
    assert cells * len(experiment.PROFILES) * 2 == 324


def test_out_of_range_actual_delay_is_labelled_not_truncated(tmp_path):
    output = experiment.run(
        tiny_args(tmp_path / "far", actual_delays=[6], families=["standard"])
    )
    record = json.loads((output / "friction0_delay6_seed53_standard" / "fit.json").read_text())
    assert record["actual_delay_steps"] == 6
    assert record["actual_delay_out_of_search_range"] is True
    assert record["parameters"]["delay_steps"] == 0
    assert record["delay_error_steps"] == -6


def test_existing_output_is_never_overwritten(tiny_run, tmp_path):
    output, _ = tiny_run
    with pytest.raises(ValueError, match="already exists"):
        experiment.run(tiny_args(output))
    taken = tmp_path / "taken"
    taken.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        experiment.run(tiny_args(taken))
    assert list(taken.iterdir()) == []


def test_direct_script_cli_works_without_pytest_root_injection(tmp_path):
    environment = os.environ.copy()
    # Do not rely on tests/conftest.py adding the repository's parent directory.
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(experiment.ROOT / "src"), *[
            str(path) for path in sys.path if Path(path).name == ".local-deps"
        ]]
    )
    result = subprocess.run(
        [sys.executable, str(experiment.ROOT / "scripts/evaluate_friction_delay.py"), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--frictions" in result.stdout and "--actual-delays" in result.stdout


def test_changed_source_never_produces_a_completed_experiment(tmp_path, monkeypatch):
    original_read = Path.read_bytes
    source = experiment.ROOT / "pyproject.toml"
    source_reads = 0

    def changed_read(path):
        nonlocal source_reads
        payload = original_read(path)
        if path == source:
            source_reads += 1
            if source_reads >= 2:
                return payload + b"\n# source changed while the experiment ran\n"
        return payload

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    output = tmp_path / "source_changed"
    with pytest.raises(RuntimeError, match="source changed"):
        experiment.run(tiny_args(output, families=["standard"]))
    consistency = json.loads((output / "source_consistency.json").read_text())
    assert consistency["unchanged"] is False
    assert not (output / "summary.json").exists()
    assert not (output / "manifest.json").exists()


def test_distinct_friction_values_never_collide_in_output_names(tmp_path):
    output = experiment.run(
        tiny_args(tmp_path / "distinct", frictions=[0.04, 0.040000001], families=["standard"])
    )
    fits = read_gz(output / "fits.csv.gz")
    assert len(fits) == 2
    assert len({row["cell"] for row in fits}) == 2
    assert {float(row["stribeck_friction_nm"]) for row in fits} == {0.04, 0.040000001}


def test_all_five_delay_candidates_have_independently_recomputable_losses(tmp_path):
    output = experiment.run(tiny_args(
        tmp_path / "five_delays", families=["low_speed"], actual_delays=[2], max_delay_steps=4,
    ))
    candidates = read_gz(output / "candidates.csv.gz")
    assert len(candidates) == 5
    assert {int(row["candidate_delay_steps"]) for row in candidates} == set(range(5))
    for candidate in candidates:
        delay = int(candidate["candidate_delay_steps"])
        trace = read_gz(output / candidate["cell"] / f"candidate_delay{delay}_calibration_predictions.csv.gz")
        errors = []
        for trajectory in ("calibration_0", "calibration_1"):
            rows = [row for row in trace if row["trajectory"] == trajectory]
            assert len(rows) == 251
            np.testing.assert_allclose(
                [float(row["time_s"]) for row in rows], np.arange(251) * .002,
                rtol=0, atol=1e-15,
            )
            log = experiment.load_encoder_log(output / candidate["cell"] / f"{trajectory}.csv")
            np.testing.assert_array_equal(
                [float(row["measured_position_rad"]) for row in rows], log.positions,
            )
            errors.extend([
                ((float(row["predicted_position_rad"]) - float(row["measured_position_rad"])) / .05) ** 2
                for row in rows[1:]
            ])
        expected_loss = math.fsum(errors) / len(errors)
        assert float(candidate["reported_calibration_loss"]) == pytest.approx(expected_loss, rel=1e-11)
        assert candidate["optimizer_success"] == "False"


def test_excitation_sequences_match_the_frozen_new_protocol():
    config = experiment.BenchConfig()
    time = np.arange(1500) * .002
    expected = {
        ("low_speed", 0): .12 * np.sin(2*np.pi*.35*time),
        ("low_speed", 1): .10*np.sin(2*np.pi*.55*time+.7) + .025*np.sin(2*np.pi*1.8*time),
    }
    common_validation = .35*np.sin(2*np.pi*(.6*time + 2.2*time**2/6) + .2)
    levels = np.array([0, .22, -.48, .07, .38, -.16, 0])
    common_holdout = levels[np.minimum((time / 3 * 7).astype(int), 6)]
    for family in ("standard", "low_speed"):
        expected[(family, 2)] = common_validation
        expected[(family, 3)] = common_holdout
    for (family, index), commands in expected.items():
        np.testing.assert_allclose(
            experiment.excitation(config, 3., index, family), commands, rtol=0, atol=1e-15,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"frictions": []},
        {"frictions": [0.0, 0.0]},
        {"frictions": [-0.01]},
        {"frictions": [float("nan")]},
        {"frictions": [float("inf")]},
        {"frictions": [True]},
        {"actual_delays": []},
        {"actual_delays": [2, 2]},
        {"actual_delays": [-1]},
        {"actual_delays": [False]},
        {"seeds": []},
        {"seeds": [53, 53]},
        {"seeds": [-3]},
        {"families": []},
        {"families": ["standard", "standard"]},
        {"families": ["unknown"]},
        {"calibration_duration": 0.4},
        {"calibration_duration": float("nan")},
        {"duration": 0.9},
        {"duration": float("inf")},
        {"max_delay_steps": -1},
        {"max_delay_steps": True},
        {"max_nfev": 0},
        {"max_nfev": False},
        {"noise_scale": -1.0},
        {"noise_scale": float("nan")},
    ],
)
def test_invalid_grids_and_budgets_are_rejected_before_creating_output(tmp_path, overrides):
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        experiment.run(tiny_args(output, **overrides))
    assert not output.exists()
