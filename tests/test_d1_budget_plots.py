"""Independent arithmetic fixtures, not formal experiment evidence."""

import ast
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/plot_d1_budget_study.py"


@pytest.fixture(scope="module")
def plotter():
    spec = importlib.util.spec_from_file_location("independent_budget_plot_subject", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_summary(completed=4, value=1.0):
    metrics = (
        "velocity_rmse_mps",
        "yaw_rmse_rps",
        "height_rmse_m",
        "attitude_rmse_rad",
        "mean_mechanical_power_w",
        "action_rms",
    )
    return {
        "cases": 4,
        "completed": completed,
        "failed": 4 - completed,
        "quality_passes": min(2, completed),
        "completed_only_means": {key: value * (i + 1) for i, key in enumerate(metrics)}
        if completed
        else None,
    }


@pytest.fixture
def arithmetic_report():
    seeds, budgets = [11, 22, 33], [512, 1024, 1536, 4096]
    result = {
        "schema": "d1-budget-analysis-v1",
        "kind": "formal",
        "training_seeds": seeds,
        "checkpoint_budgets": budgets,
        "comparisons": {},
    }
    for split in ("development", "holdout"):
        result["comparisons"][split] = {
            "zero_physical_baseline": independent_summary(value=0.5),
            "per_seed": [
                {
                    "training_seed": seed,
                    "budgets": [
                        {
                            "budget": budget,
                            "modes": {
                                "shared2": independent_summary(value=1 + bindex),
                                "independent8": independent_summary(value=2 + bindex),
                            },
                        }
                        for bindex, budget in enumerate(budgets)
                    ],
                }
                for seed in seeds
            ],
        }
    return result


def test_independent_partial_survivor_is_nan_and_curves_do_not_mutate_input(
    plotter, arithmetic_report
):
    partial = arithmetic_report["comparisons"]["development"]["per_seed"][0]["budgets"][1]
    partial["modes"]["shared2"] = independent_summary(completed=3, value=0.0001)
    original = copy.deepcopy(arithmetic_report)
    curves, counts = plotter.prepare_curves(arithmetic_report)
    series = next(
        row
        for row in curves["development"]["series"]
        if row["mode"] == "shared2" and row["training_seed"] == 11
    )
    assert series["budgets"] == [512, 1024, 1536, 4096]
    for key in plotter.METRICS:
        assert np.isfinite(series["values"][key][0])
        assert np.isnan(series["values"][key][1])
        assert np.isfinite(series["values"][key][2:]).all()
    assert len(counts) == 2 * (3 * 4 * 2 + 1)
    assert len([row for row in counts if row["mode"] == "zero"]) == 2
    assert arithmetic_report == original


def test_independent_all_failed_policy_and_partial_zero_omit_all_means(plotter, arithmetic_report):
    arithmetic_report["comparisons"]["holdout"]["zero_physical_baseline"] = independent_summary(3)
    arithmetic_report["comparisons"]["holdout"]["per_seed"][0]["budgets"][0]["modes"]["shared2"] = (
        independent_summary(0)
    )
    curves, _ = plotter.prepare_curves(arithmetic_report)
    assert all(math.isnan(value) for value in curves["holdout"]["zero"].values())
    assert all(
        math.isnan(values[0]) for values in curves["holdout"]["series"][0]["values"].values()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_seed",
        "missing_seed",
        "extra_seed",
        "duplicate_budget",
        "missing_budget",
        "extra_budget",
        "missing_mode",
        "extra_mode",
        "bool_budget",
        "float_budget",
        "reverse_budget",
        "bool_seed",
        "unknown_kind",
        "unknown_schema",
        "extra_split",
        "missing_split",
    ],
)
def test_independent_bad_matrix_never_gets_plotted(plotter, arithmetic_report, mutation):
    data = arithmetic_report
    seed_rows = data["comparisons"]["development"]["per_seed"]
    budget_rows = seed_rows[0]["budgets"]
    if mutation == "duplicate_seed":
        seed_rows[1]["training_seed"] = seed_rows[0]["training_seed"]
    elif mutation == "missing_seed":
        seed_rows.pop()
    elif mutation == "extra_seed":
        seed_rows.append(copy.deepcopy(seed_rows[0]))
    elif mutation == "duplicate_budget":
        budget_rows[1]["budget"] = budget_rows[0]["budget"]
    elif mutation == "missing_budget":
        budget_rows.pop()
    elif mutation == "extra_budget":
        budget_rows.append(copy.deepcopy(budget_rows[0]))
    elif mutation == "missing_mode":
        budget_rows[0]["modes"].pop("shared2")
    elif mutation == "extra_mode":
        budget_rows[0]["modes"]["bogus"] = independent_summary()
    elif mutation == "bool_budget":
        data["checkpoint_budgets"][0] = True
    elif mutation == "float_budget":
        data["checkpoint_budgets"][0] = 512.0
    elif mutation == "reverse_budget":
        data["checkpoint_budgets"].reverse()
    elif mutation == "bool_seed":
        data["training_seeds"][0] = True
    elif mutation == "unknown_kind":
        data["kind"] = "best"
    elif mutation == "unknown_schema":
        data["schema"] = "unknown"
    elif mutation == "extra_split":
        data["comparisons"]["new"] = {}
    elif mutation == "missing_split":
        data["comparisons"].pop("holdout")
    with pytest.raises(ValueError):
        plotter.prepare_curves(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("completed", True),
        ("completed", 4.0),
        ("completed", 5),
        ("failed", 1),
        ("cases", 3),
        ("quality_passes", 5),
        ("quality_passes", False),
    ],
)
def test_independent_invalid_counts_rejected(plotter, arithmetic_report, field, value):
    arithmetic_report["comparisons"]["development"]["zero_physical_baseline"][field] = value
    with pytest.raises(ValueError):
        plotter.prepare_curves(arithmetic_report)


@pytest.mark.parametrize("value", [True, "0.1", float("nan"), float("inf"), -1.0])
def test_independent_bad_metric_is_rejected_even_on_hidden_failed_point(
    plotter, arithmetic_report, value
):
    row = independent_summary(completed=3)
    row["completed_only_means"]["velocity_rmse_mps"] = value
    arithmetic_report["comparisons"]["development"]["per_seed"][0]["budgets"][0]["modes"][
        "shared2"
    ] = row
    with pytest.raises(ValueError):
        plotter.prepare_curves(arithmetic_report)


def test_independent_real_pngs_input_immutability_and_no_training_imports(
    plotter, arithmetic_report, tmp_path
):
    source = tmp_path / "input"
    source.mkdir()
    input_path = source / "analysis.json"
    input_bytes = json.dumps(arithmetic_report).encode()
    input_path.write_bytes(input_bytes)
    output = tmp_path / "figures"
    plotter.plot_analysis(input_path, output)
    assert input_path.read_bytes() == input_bytes
    for split in ("development", "holdout"):
        decoded = mpimg.imread(output / f"{split}.png")
        assert decoded.ndim == 3 and decoded.shape[0] > 200 and decoded.shape[1] > 400
        assert float(np.std(decoded)) > 0.01
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema"] == "d1-budget-plots-v1"
    assert manifest["input_schema"] == "d1-budget-analysis-v1"
    assert set(manifest["outputs"]) == {"development.png", "holdout.png", "counts.csv"}
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    with (output / "counts.csv").open(newline="") as stream:
        actual_counts = list(csv.DictReader(stream))
    _, expected_counts = plotter.prepare_curves(arithmetic_report)
    assert actual_counts == [
        {key: str(value) for key, value in row.items()} for row in expected_counts
    ]
    encoded_manifest = json.dumps(manifest)
    assert hashlib.sha256(input_bytes).hexdigest() in encoded_manifest
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() in encoded_manifest
    for target in (output, source / "figures", tmp_path):
        with pytest.raises((ValueError, FileExistsError)):
            plotter.plot_analysis(input_path, target)
    imports = []
    for node in ast.walk(ast.parse(SCRIPT.read_text())):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(
        name.split(".")[0]
        in {"torch", "mujoco", "gymnasium", "stable_baselines3", "wheel_legged_control", "scripts"}
        for name in imports
    )


def test_independent_figure_contains_nan_gap_and_smoke_watermark(plotter, arithmetic_report):
    partial = arithmetic_report["comparisons"]["development"]["per_seed"][0]["budgets"][1]
    partial["modes"]["shared2"] = independent_summary(3)
    curves, _ = plotter.prepare_curves(arithmetic_report)
    figure = plotter.make_figure("development", curves["development"], "smoke")
    try:
        assert len(figure.axes) == 6
        for axis in figure.axes:
            assert axis.get_xscale() == "log"
            assert list(axis.get_xticks()) == arithmetic_report["checkpoint_budgets"]
            lines = [line for line in axis.get_lines() if len(line.get_xdata()) == 4]
            assert len(lines) == 6
            assert any(np.isnan(np.asarray(line.get_ydata(), dtype=float)[1]) for line in lines)
        texts = " ".join(text.get_text() for text in figure.texts).lower()
        assert "smoke" in texts and "interface only" in texts
        assert "seed" in texts and ("gap" in texts or "failed" in texts)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_independent_cli_from_unrelated_cwd_and_duplicate_json_keys(
    plotter, arithmetic_report, tmp_path
):
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    source = source_dir / "analysis.json"
    source.write_text(json.dumps(arithmetic_report))
    output = tmp_path / "cli-plots"
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--analysis", str(source), "--output", str(output)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert (output / "development.png").is_file()
    invalid = source_dir / "duplicate.json"
    valid = json.dumps(arithmetic_report)
    invalid.write_text(valid[:-1] + ', "schema": "d1-budget-analysis-v1"}')
    refused_output = tmp_path / "duplicate-must-not-create"
    with pytest.raises(ValueError):
        plotter.plot_analysis(invalid, refused_output)
    assert not refused_output.exists()


def test_independent_source_and_output_symlinks_are_refused(plotter, arithmetic_report, tmp_path):
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    source = source_dir / "analysis.json"
    source.write_text(json.dumps(arithmetic_report))
    alias = tmp_path / "alias.json"
    alias.symlink_to(source)
    dangling = tmp_path / "dangling-output"
    dangling.symlink_to(tmp_path / "missing")
    for input_path, output in [(alias, tmp_path / "figures"), (source, dangling)]:
        with pytest.raises((ValueError, FileExistsError)):
            plotter.plot_analysis(input_path, output)
    assert not (tmp_path / "figures").exists()
    assert dangling.is_symlink()


@pytest.mark.parametrize("change", ["unhashable_kind", "missing_zero", "huge_metric"])
def test_independent_malformed_inputs_raise_contract_error(plotter, arithmetic_report, change):
    if change == "unhashable_kind":
        arithmetic_report["kind"] = []
    elif change == "missing_zero":
        arithmetic_report["comparisons"]["development"].pop("zero_physical_baseline")
    else:
        arithmetic_report["comparisons"]["development"]["zero_physical_baseline"][
            "completed_only_means"
        ]["action_rms"] = 10**400
    with pytest.raises(ValueError):
        plotter.prepare_curves(arithmetic_report)


def test_independent_nonfinite_json_is_rejected_before_writing(
    plotter, arithmetic_report, tmp_path
):
    folder = tmp_path / "input"
    folder.mkdir()
    source = folder / "bad.json"
    arithmetic_report["extra"] = float("nan")
    source.write_text(json.dumps(arithmetic_report))
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        plotter.plot_analysis(source, output)
    assert not output.exists()
