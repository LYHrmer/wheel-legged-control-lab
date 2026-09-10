"""Plot accounting uses paired ratios, not a ratio of seed averages."""

import hashlib
import json

import numpy as np
import pytest

from scripts import evaluate_friction_delay as experiment
from scripts import plot_friction_delay as plots


def row(arm, value, *, seed="53", **changes):
    return {
        "cell": "plant_" + seed,
        "seed": seed,
        "family": "standard",
        "stribeck_friction_nm": "0.04",
        "actual_delay_steps": "2",
        "profile": "tracking",
        "controller": arm,
        "rmse": value,
        **changes,
    }


def ratios(rows):
    return plots.paired_ratios(
        rows,
        left="fitted",
        right="nominal",
        arm_field="controller",
        metric="rmse",
    )


def test_pair_first_then_aggregate():
    result = ratios(
        [
            row("fitted", 4),
            row("nominal", 2),
            row("fitted", 3, seed="67"),
            row("nominal", 6, seed="67"),
        ]
    )
    assert [item["ratio"] for item in result] == [2, 0.5]
    assert sum(item["ratio"] for item in result) / 2 == 1.25


@pytest.mark.parametrize(
    "rows",
    [
        [row("fitted", 1)],
        [row("fitted", 1), row("fitted", 1), row("nominal", 1)],
        [row("other", 1), row("nominal", 1)],
        [row("fitted", 1), row("nominal", 0)],
        [row("fitted", float("nan")), row("nominal", 1)],
        [row("fitted", -1), row("nominal", 1)],
        [row("fitted", 1), row("nominal", 1, family="low_speed")],
    ],
)
def test_bad_pairs_are_not_silently_pooled(rows):
    with pytest.raises(ValueError):
        ratios(rows)


def test_equal_seed_values_cannot_create_negative_roundoff_error_bars():
    class Axis:
        def errorbar(self, x, mean, **kwargs):
            assert np.all(kwargs["yerr"] >= 0)
            np.testing.assert_allclose(mean, [12.3], rtol=0, atol=2e-15)

        def set_xticks(self, value):
            assert value == [0]

        def grid(self, **kwargs):
            pass

    plots.seed_range(
        Axis(),
        [{"actual_delay_steps": 0, "value": 12.3} for _ in range(3)],
        [0],
        "value",
        "equal seeds",
        "blue",
    )


def test_render_real_two_family_tiny_metrics(tmp_path):
    study = experiment.main(
        [
            "--output",
            str(tmp_path / "study"),
            "--frictions",
            "0",
            "--actual-delays",
            "0",
            "--seeds",
            "53",
            "--calibration-duration",
            ".5",
            "--duration",
            "1",
            "--max-delay-steps",
            "0",
            "--max-nfev",
            "1",
        ]
    )
    output = plots.run(study, tmp_path / "plots")
    record = json.loads((output / "manifest.json").read_text())
    assert set(record["sha256"]) == {"identification.png", "control.png"}
    for name, digest in record["sha256"].items():
        data = (output / name).read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert hashlib.sha256(data).hexdigest() == digest
    for name, digest in record["input_sha256"].items():
        assert hashlib.sha256((study / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        plots.run(study, output)
