"""Independent audit of ``scripts/analyze_friction_delay.py`` over a real tiny study.

The fixture is a genuine two-family run of the frozen producer, never faked artifacts.
Rejection tests copy that study, tamper with one artifact and then recompute
``manifest.json`` so the analyzer's cheap digest check cannot shortcut the numeric
audit: a ``ValueError`` must come from recomputing losses, metrics and pairings.
Expected values here are hand-written; no production helper is imported to produce them.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_friction_delay as analysis
from scripts import evaluate_friction_delay as producer

ROOT = Path(__file__).resolve().parents[1]

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
PROFILES = ("reversal", "stress", "tracking")
TRAJECTORIES = ("calibration_0", "calibration_1", "validation_2", "holdout_3")
DT = 0.002
POSITION_SCALE_RAD = 0.05
CALIBRATION_STEPS = 250
CONTROL_STEPS = 500
SCHEMA = "friction-delay-analysis-v1"
EXPECTED_COUNTS = {
    "fits": 2,
    "candidates": 2,
    "prediction_metric_rows": 16,
    "control_cases": 12,
}
MATRICES = ("fits", "candidates", "prediction_metrics", "control_metrics")


def read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_gz(path: Path, rows: list[dict]) -> None:
    """Deterministic zero-mtime gzip, matching how the producer writes its tables."""
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        stream = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        stream.detach()


def snapshot(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def refresh_manifest(study: Path) -> None:
    """Re-seal a tampered study so only the numeric audit can still reject it."""
    digests = {
        name: digest
        for name, digest in snapshot(study).items()
        if Path(name).name != "manifest.json"
    }
    (study / "manifest.json").write_text(
        json.dumps({"sha256": digests}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def copied(study: Path, tmp_path: Path, name: str = "tampered") -> Path:
    destination = tmp_path / name
    shutil.copytree(study, destination)
    return destination


def dumped(report: dict) -> dict:
    return json.loads(json.dumps(report, default=str))


def subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / ".local-deps")])
    return environment


ISOLATION_PROGRAM = """
import json
import sys
from pathlib import Path

FORBIDDEN = (
    "mujoco",
    "scipy",
    "torch",
    "wheel_legged_control",
    "scripts.evaluate_friction_delay",
    "scripts.evaluate_wheel_control",
    "scripts.run_actuator_identification",
)


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if any(name == item or name.startswith(item + ".") for item in FORBIDDEN):
            raise ImportError("forbidden production dependency: " + name)
        return None

    def find_module(self, name, path=None):
        return self.find_spec(name, path)


sys.meta_path.insert(0, Blocker())
import numpy  # noqa: F401  stdlib + numpy only
from scripts import analyze_friction_delay as analysis

report = analysis.analyze(Path(sys.argv[1]))
print(json.dumps({"schema": report["schema"], "counts": report["verification"]["counts"]}))
"""


@pytest.fixture(scope="module")
def study(tmp_path_factory) -> Path:
    """One real tiny run of the frozen producer: 2 fits, 16 prediction rows, 12 control cases."""
    output = tmp_path_factory.mktemp("studies") / "tiny"
    return producer.run(argparse.Namespace(output=output, **TINY))


def test_analysis_reports_schema_verified_counts_and_complete_record_arrays(study):
    report = analysis.analyze(study)
    assert report["schema"] == SCHEMA
    verification = report["verification"]
    assert verification["counts"] == EXPECTED_COUNTS
    recorded = json.loads((study / "source.json").read_text())["sha256"]
    assert verification["source_members"] == len(recorded) == 13
    assert "scripts/evaluate_friction_delay.py" in recorded
    assert verification["default_formal_protocol"] is False
    assert len(report["fits"]) == 2
    assert len(report["candidates"]) == 2
    assert len(report["predictions"]) == 16
    assert len(report["controls"]) == 12
    assert {row["cell"] for row in report["fits"]} == set(CELLS)
    assert {row["family"] for row in report["fits"]} == {"standard", "low_speed"}
    assert {row["trajectory"] for row in report["predictions"]} == set(TRAJECTORIES)
    assert {row["model"] for row in report["predictions"]} == {"nominal", "selected"}
    assert {row["profile"] for row in report["controls"]} == set(PROFILES)
    assert {row["controller"] for row in report["controls"]} == {"nominal_ff", "fitted_ff"}


def test_groups_partition_the_grid_by_friction_delay_and_family(study):
    report = analysis.analyze(study)
    groups = report["groups"]
    assert isinstance(groups, list) and len(groups) == 2
    assert {
        (float(group["stribeck_friction_nm"]), int(group["actual_delay_steps"]), group["family"])
        for group in groups
    } == {(0.0, 0, "standard"), (0.0, 0, "low_speed")}
    for group in groups:
        assert group["fits"] == 1
        assert group["selected_delay_steps"] == [0]
        assert group["delay_wrong_count"] == 0
        assert set(group["control_by_profile"]) == set(PROFILES)
    assert sum(group["fits"] for group in groups) == EXPECTED_COUNTS["fits"]


def test_candidate_calibration_loss_is_independently_recomputable(study):
    """Hand loss: first sample dropped, scaled by 0.05 rad, averaged over both logs."""
    report = analysis.analyze(study)
    for entry in report["candidates"]:
        delay = int(entry["candidate_delay_steps"])
        rows = read_gz(
            study / entry["cell"] / f"candidate_delay{delay}_calibration_predictions.csv.gz"
        )
        squares = []
        for trajectory in ("calibration_0", "calibration_1"):
            trace = [row for row in rows if row["trajectory"] == trajectory]
            assert len(trace) == CALIBRATION_STEPS + 1
            np.testing.assert_allclose(
                [float(row["time_s"]) for row in trace],
                np.arange(CALIBRATION_STEPS + 1) * DT,
                rtol=0,
                atol=1e-15,
            )
            squares.extend(
                (
                    (float(row["predicted_position_rad"]) - float(row["measured_position_rad"]))
                    / POSITION_SCALE_RAD
                )
                ** 2
                for row in trace[1:]
            )
        assert len(squares) == 2 * CALIBRATION_STEPS
        expected = math.fsum(squares) / len(squares)
        assert entry["recomputed_calibration_loss"] == pytest.approx(expected, rel=1e-9)


def test_prediction_metrics_are_independently_recomputable(study):
    """RMSE, MAE and peak from the archived traces, first sample excluded."""
    report = analysis.analyze(study)
    for cell in CELLS:
        for trajectory in TRAJECTORIES:
            rows = read_gz(study / cell / f"{trajectory}_predictions.csv.gz")
            assert len(rows) == CALIBRATION_STEPS + 1
            measured = np.array([float(row["measured_position_rad"]) for row in rows])
            for model in ("nominal", "selected"):
                predicted = np.array([float(row[f"{model}_position_rad"]) for row in rows])
                error = predicted[1:] - measured[1:]
                entry = next(
                    row
                    for row in report["predictions"]
                    if row["cell"] == cell
                    and row["trajectory"] == trajectory
                    and row["model"] == model
                )
                assert int(entry["samples"]) == CALIBRATION_STEPS
                assert float(entry["position_rmse_rad"]) == pytest.approx(
                    float(np.sqrt(np.mean(error**2))), rel=1e-9
                )
                assert float(entry["position_mae_rad"]) == pytest.approx(
                    float(np.mean(np.abs(error))), rel=1e-9
                )
                assert float(entry["peak_position_error_rad"]) == pytest.approx(
                    float(np.max(np.abs(error))), rel=1e-9
                )


def test_analyzer_never_calls_producer_metric_helpers(study, monkeypatch):
    from scripts import evaluate_wheel_control as control_producer

    def forbidden(*args, **kwargs):
        raise AssertionError("the analyzer must recompute metrics, never call the producer")

    for module, name in (
        (producer, "calibration_loss"),
        (producer, "predict_encoder_log"),
        (producer, "fit_encoder_parameters"),
        (producer, "acquire_logs"),
        (control_producer, "compute_metrics"),
        (control_producer, "run_closed_loop"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    report = analysis.analyze(study)
    assert report["verification"]["counts"] == EXPECTED_COUNTS
    modules = {
        value.__name__ for value in vars(analysis).values() if isinstance(value, types.ModuleType)
    }
    assert not {"mujoco", "scipy", "torch"} & modules
    assert not [name for name in modules if name.startswith("wheel_legged_control")]
    assert not [name for name in modules if name.startswith("scripts.evaluate")]


@pytest.mark.parametrize(
    "table,column",
    [
        ("candidates", "reported_calibration_loss"),
        ("candidates", "recomputed_calibration_loss"),
        ("fits", "selected_calibration_loss"),
    ],
)
def test_misreported_losses_are_rejected(study, tmp_path, table, column):
    copy = copied(study, tmp_path)
    rows = read_gz(copy / f"{table}.csv.gz")
    rows[0][column] = repr(float(rows[0][column]) * 1.05 + 1e-9)
    write_gz(copy / f"{table}.csv.gz", rows)
    refresh_manifest(copy)
    with pytest.raises(ValueError):
        analysis.analyze(copy)


@pytest.mark.parametrize(
    "column,delta",
    [
        ("predicted_position_rad", 0.01),
        ("measured_position_rad", 0.01),
        ("time_s", 0.004),
    ],
)
def test_polluted_candidate_traces_are_rejected(study, tmp_path, column, delta):
    copy = copied(study, tmp_path)
    path = copy / CELLS[0] / "candidate_delay0_calibration_predictions.csv.gz"
    rows = read_gz(path)
    rows[7][column] = repr(float(rows[7][column]) + delta)
    write_gz(path, rows)
    refresh_manifest(copy)
    with pytest.raises(ValueError):
        analysis.analyze(copy)


@pytest.mark.parametrize(
    "column",
    ["position_rmse_rad", "position_mae_rad", "peak_position_error_rad", "samples"],
)
def test_misreported_prediction_metrics_are_rejected(study, tmp_path, column):
    copy = copied(study, tmp_path)
    rows = read_gz(copy / "prediction_metrics.csv.gz")
    if column == "samples":
        rows[3][column] = str(int(rows[3][column]) - 1)
    else:
        rows[3][column] = repr(float(rows[3][column]) * 1.5 + 1e-9)
    write_gz(copy / "prediction_metrics.csv.gz", rows)
    refresh_manifest(copy)
    with pytest.raises(ValueError):
        analysis.analyze(copy)


def test_polluted_prediction_trace_timestamps_are_rejected(study, tmp_path):
    copy = copied(study, tmp_path)
    path = copy / CELLS[1] / "validation_2_predictions.csv.gz"
    rows = read_gz(path)
    rows[11]["time_s"] = repr(float(rows[11]["time_s"]) + DT)
    write_gz(path, rows)
    refresh_manifest(copy)
    with pytest.raises(ValueError):
        analysis.analyze(copy)


def test_broken_control_current_versus_next_timing_is_rejected(study, tmp_path):
    """``actual[1:] == next[:-1]`` is the only link between consecutive rows."""
    copy = copied(study, tmp_path)
    path = copy / CELLS[0] / "tracking_fitted_ff_rows.csv.gz"
    rows = read_gz(path)
    assert len(rows) == CONTROL_STEPS
    following = [row["next_velocity_rad_s"] for row in rows]
    for row, value in zip(rows, following[1:] + following[:1]):
        row["next_velocity_rad_s"] = value
    write_gz(path, rows)
    refresh_manifest(copy)
    with pytest.raises(ValueError):
        analysis.analyze(copy)


def test_broken_control_noise_pairing_is_rejected(study, tmp_path):
    """Offset the measurement noise while keeping ``error = reference - measured`` exact."""
    copy = copied(study, tmp_path)
    path = copy / CELLS[0] / "stress_fitted_ff_rows.csv.gz"
    rows = read_gz(path)
    for row in rows:
        measured = float(row["measured_velocity_rad_s"]) + 1e-3
        row["measured_velocity_rad_s"] = repr(measured)
        row["error_rad_s"] = repr(float(row["reference_velocity_rad_s"]) - measured)
    write_gz(path, rows)
    refresh_manifest(copy)
    with pytest.raises(ValueError):
        analysis.analyze(copy)


def test_misreported_control_metrics_are_rejected(study, tmp_path):
    copy = copied(study, tmp_path)
    rows = read_gz(copy / "control_metrics.csv.gz")
    rows[0]["velocity_rmse_rad_s"] = repr(float(rows[0]["velocity_rmse_rad_s"]) * 0.5)
    write_gz(copy / "control_metrics.csv.gz", rows)
    refresh_manifest(copy)
    with pytest.raises(ValueError):
        analysis.analyze(copy)


def test_nominal_rows_that_stop_being_paired_across_families_are_rejected(study, tmp_path):
    """Changing only gzip metadata preserves all numbers but breaks byte-identical pairing."""
    copy = copied(study, tmp_path)
    victim = copy / CELLS[0] / "tracking_nominal_ff_rows.csv.gz"
    original = victim.read_bytes()
    changed = bytearray(original)
    changed[4:8] = (1).to_bytes(4, "little")
    victim.write_bytes(changed)
    assert gzip.decompress(original) == gzip.decompress(changed)
    refresh_manifest(copy)
    with pytest.raises(ValueError, match="across families"):
        analysis.analyze(copy)


def test_unsealed_manifest_digest_mismatch_is_rejected(study, tmp_path):
    copy = copied(study, tmp_path)
    rows = read_gz(copy / "fits.csv.gz")
    rows[0]["jacobian_rank"] = "99"
    write_gz(copy / "fits.csv.gz", rows)
    with pytest.raises(ValueError):
        analysis.analyze(copy)


def test_source_hash_changes_are_rejected(study, tmp_path):
    copy = copied(study, tmp_path)
    source = json.loads((copy / "source.json").read_text())
    source["sha256"]["pyproject.toml"] = "0" * 64
    (copy / "source.json").write_text(
        json.dumps(source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    refresh_manifest(copy)
    with pytest.raises(ValueError):
        analysis.analyze(copy)


@pytest.mark.parametrize("field", ["jacobian_rank", "continuous_parameters_at_bounds"])
def test_fit_diagnostic_consistency_is_checked_after_resealing(study, tmp_path, field):
    copy = copied(study, tmp_path)
    path = copy / CELLS[0] / "fit.json"
    fit = json.loads(path.read_text())
    fit[field] = 99 if field == "jacobian_rank" else ["damping"]
    path.write_text(json.dumps(fit), encoding="utf-8")
    refresh_manifest(copy)
    with pytest.raises(ValueError, match="rank|bound"):
        analysis.analyze(copy)


@pytest.mark.parametrize("field", ["observed_labels", "observed_status_counts", "observed_counts"])
def test_summary_cannot_silently_relabel_the_matrix(study, tmp_path, field):
    copy = copied(study, tmp_path)
    path = copy / "summary.json"
    summary = json.loads(path.read_text())
    summary[field] = {}
    path.write_text(json.dumps(summary), encoding="utf-8")
    refresh_manifest(copy)
    with pytest.raises(ValueError, match="labels|counts"):
        analysis.analyze(copy)


@pytest.mark.parametrize("contents", ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": 1e999}'])
def test_strict_json_rejects_ambiguous_or_nonfinite_values(tmp_path, contents):
    path = tmp_path / "bad.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        analysis.read_json(path)


def test_post_experiment_readme_is_explicitly_outside_original_manifest(study, tmp_path):
    copy = copied(study, tmp_path)
    (copy / "README.md").write_text("Written after the experiment.\n", encoding="utf-8")
    report = analysis.analyze(copy)
    assert report["verification"]["post_experiment_docs_not_in_original_manifest"] == ["README.md"]


@pytest.mark.parametrize("matrix", MATRICES)
@pytest.mark.parametrize("mode", ["missing", "duplicate"])
def test_missing_or_duplicated_matrix_rows_are_rejected(study, tmp_path, matrix, mode):
    copy = copied(study, tmp_path)
    rows = read_gz(copy / f"{matrix}.csv.gz")
    rows = rows[:-1] if mode == "missing" else rows + [dict(rows[0])]
    write_gz(copy / f"{matrix}.csv.gz", rows)
    refresh_manifest(copy)
    with pytest.raises(ValueError):
        analysis.analyze(copy)


def test_incomplete_studies_are_rejected(study, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        analysis.analyze(empty)
    with pytest.raises(ValueError):
        analysis.analyze(tmp_path / "absent")
    copy = copied(study, tmp_path, "stripped")
    (copy / "control_metrics.csv.gz").unlink()
    refresh_manifest(copy)
    with pytest.raises(ValueError):
        analysis.analyze(copy)
    partial = copied(study, tmp_path, "partial")
    shutil.rmtree(partial / CELLS[1])
    refresh_manifest(partial)
    with pytest.raises(ValueError):
        analysis.analyze(partial)


def test_run_writes_analysis_json_into_a_new_directory(study, tmp_path):
    output = tmp_path / "analysis"
    report = analysis.run(study, output)
    assert sorted(path.name for path in output.iterdir()) == [
        "analysis.json",
        "analysis_manifest.json",
        "analysis_source.py",
    ]
    inventory = json.loads((output / "analysis_manifest.json").read_text())["sha256"]
    assert inventory == {
        name: snapshot(output)[name] for name in ("analysis.json", "analysis_source.py")
    }
    written = json.loads((output / "analysis.json").read_text())
    assert written == dumped(report)
    assert written["schema"] == SCHEMA
    assert written["verification"]["counts"] == EXPECTED_COUNTS
    assert len(written["groups"]) == 2


def test_run_refuses_unsafe_outputs_and_never_touches_the_study(study, tmp_path):
    before = snapshot(study)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError):
        analysis.run(study, existing)
    assert list(existing.iterdir()) == []
    for unsafe in (study, study / "inside", study / CELLS[0] / "nested", study.parent):
        with pytest.raises(ValueError):
            analysis.run(study, unsafe)
    assert not (study / "inside").exists()
    assert snapshot(study) == before


def test_dangling_output_symlink_is_not_followed(study, tmp_path):
    destination = tmp_path / "absent_destination"
    symlink = tmp_path / "output_link"
    symlink.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        analysis.run(study, symlink)
    assert symlink.is_symlink() and not destination.exists()


def test_cli_help_and_end_to_end_run_without_repository_path_injection(study, tmp_path):
    script = str(ROOT / "scripts/analyze_friction_delay.py")
    environment = subprocess_environment()
    helped = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert helped.returncode == 0, helped.stdout + helped.stderr
    assert "--output" in helped.stdout
    output = tmp_path / "cli_analysis"
    result = subprocess.run(
        [sys.executable, script, str(study), "--output", str(output)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip(), "the CLI must report what it verified on stdout"
    written = json.loads((output / "analysis.json").read_text())
    assert written["schema"] == SCHEMA
    assert written["verification"]["counts"] == EXPECTED_COUNTS
    again = subprocess.run(
        [sys.executable, script, str(study), "--output", str(output)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert again.returncode != 0
    assert json.loads((output / "analysis.json").read_text()) == written


def test_analysis_runs_with_production_and_native_dependencies_forbidden(study, tmp_path):
    environment = os.environ.copy()
    # Repository root plus vendored dependencies only: no src/, no installed package.
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / ".local-deps")])
    result = subprocess.run(
        [sys.executable, "-c", ISOLATION_PROGRAM, str(study)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.splitlines()[-1])
    assert payload == {"schema": SCHEMA, "counts": EXPECTED_COUNTS}
