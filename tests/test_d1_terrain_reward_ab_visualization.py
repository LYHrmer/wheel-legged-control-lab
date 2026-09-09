"""Reward A/B plotting preserves logger timing, both seeds and physical units."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def visualizer():
    spec = importlib.util.spec_from_file_location(
        "reward_ab_visualization", ROOT / "scripts" / "visualize_d1_terrain_reward_ab.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocol(scale=1.0):
    return {
        "training_conditions": ["mixed"],
        "training_seeds": [7000, 8000],
        "training_reward_scale": scale,
        "n_envs": 2,
        "ppo": {"n_steps": 128, "n_epochs": 4},
        "actual_budget_per_condition_seed": 512,
        "cases": [{"case_id": "flat"}, {"case_id": "ramp"}],
    }


def logs():
    return [
        {"train/n_updates": "", "time/total_timesteps": "256", "train/explained_variance": ""},
        {"train/n_updates": "4", "time/total_timesteps": "512", "train/explained_variance": "-0.1"},
        {"train/n_updates": "8", "time/total_timesteps": "", "train/explained_variance": "0.2"},
    ]


def metrics():
    return [
        {
            "condition": "zero_residual" if seed == "" else "mixed_rl",
            "training_seed": seed,
            "case_id": case,
            "velocity_rmse_mps": "0.1",
            "quality_success": "1",
        }
        for seed in ("", "7000", "8000")
        for case in ("flat", "ramp")
    ]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_final_flush_is_retained_and_previous_batch_log_is_shifted(visualizer):
    curve = visualizer.update_curve(logs(), protocol())
    assert [row["update_end_timesteps"] for row in curve] == [256, 512]
    assert [row["logger_total_timesteps"] for row in curve] == [512, None]
    assert [row["explained_variance"] for row in curve] == [-0.1, 0.2]


@pytest.mark.parametrize("kind", ["missing_final", "duplicate", "nonfinite"])
def test_invalid_training_curve_is_rejected(visualizer, kind):
    rows = logs()
    if kind == "missing_final":
        rows.pop()
    elif kind == "duplicate":
        rows.append(rows[-1])
    else:
        rows[-1]["train/explained_variance"] = "nan"
    with pytest.raises(ValueError):
        visualizer.update_curve(rows, protocol())


def test_duplicate_or_missing_metric_pairs_are_rejected(visualizer):
    assert visualizer.paired_metrics(metrics(), protocol())["7000"]["velocity_rmse_mean_mps"] == 0.1
    rows = metrics()
    rows[-1] = rows[-2]
    with pytest.raises(ValueError):
        visualizer.paired_metrics(rows, protocol())


def test_plot_writes_separate_manifest_and_preserves_input_manifests(visualizer, tmp_path):
    paths = []
    for label, scale in (("raw", 1.0), ("scaled", 0.01)):
        root = tmp_path / label
        root.mkdir()
        (root / "protocol.json").write_text(json.dumps(protocol(scale)))
        (root / "manifest.json").write_text("historical manifest must remain untouched\n")
        write_csv(root / "metrics.csv", metrics())
        for seed in (7000, 8000):
            destination = root / "training" / f"mixed_seed_{seed}" / "learning_metrics"
            destination.mkdir(parents=True)
            write_csv(destination / "progress.csv", logs())
        paths.append(root)
    before = [(root / "manifest.json").read_bytes() for root in paths]
    output = tmp_path / "figure"
    manifest = visualizer.run(*paths, output)
    assert [(root / "manifest.json").read_bytes() for root in paths] == before
    data = json.loads((output / "plotted_data.json").read_text())
    assert set(data["scaled"]["curves"]) == {"7000", "8000"}
    for name, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        visualizer.run(*paths, output)
    changed = protocol(0.01)
    changed["n_envs"] = 4
    (paths[1] / "protocol.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        visualizer.run(*paths, tmp_path / "bad_protocol")
    assert not (tmp_path / "bad_protocol").exists()
