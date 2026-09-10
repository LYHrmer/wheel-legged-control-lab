"""Plot paired friction/delay results after their independent arithmetic audit.

Points/lines average repeated measurement-noise seeds. Error bars show the
observed minimum and maximum, never a confidence interval or independent robots.
This reads recorded metrics; it neither refits a model nor selects a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path):
    with gzip.open(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def paired_ratios(rows, *, left, right, arm_field, metric):
    """Pair before aggregating; duplicate or missing arms fail explicitly."""
    groups = {}
    for row in rows:
        arm = row[arm_field]
        if arm not in (left, right):
            raise ValueError(f"unexpected comparison arm: {arm}")
        key = (row["cell"], row.get("profile", row.get("trajectory")))
        pair = groups.setdefault(key, {})
        if arm in pair:
            raise ValueError(f"duplicate comparison arm: {key}/{arm}")
        pair[arm] = row
    result = []
    for key, pair in groups.items():
        if set(pair) != {left, right}:
            raise ValueError(f"incomplete comparison pair: {key}")
        first, second = pair[left], pair[right]
        for field in ("stribeck_friction_nm", "actual_delay_steps", "seed", "family"):
            if first[field] != second[field]:
                raise ValueError(f"unpaired {field}: {key}")
        a, b = float(first[metric]), float(second[metric])
        if not np.isfinite((a, b)).all() or b <= 0 or a < 0:
            raise ValueError("ratio requires finite nonnegative numerator and positive denominator")
        result.append({**first, "ratio": a / b})
    return result


def seed_range(ax, rows, delays, field, label, color, style="-"):
    means, lows, highs = [], [], []
    for delay in delays:
        values = np.asarray(
            [float(row[field]) for row in rows if int(row["actual_delay_steps"]) == delay]
        )
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("plot group has missing or nonfinite measurements")
        means.append(float(values.mean()))
        lows.append(float(values.min()))
        highs.append(float(values.max()))
    means = np.asarray(means)
    ax.errorbar(
        delays,
        means,
        # A floating mean of [12.3, 12.3, 12.3] can exceed its max by one ulp.
        # Keep the recorded values/mean; bar lengths cannot be negative.
        yerr=np.maximum(0.0, [means - lows, np.asarray(highs) - means]),
        marker="o",
        linestyle=style,
        color=color,
        label=label,
        capsize=3,
        linewidth=1.3,
    )
    ax.set_xticks(delays)
    ax.grid(alpha=0.2)


def run(study, output):
    study, output = Path(study), Path(output)
    if output.exists():
        raise FileExistsError("plot output must be a new directory")
    inputs = ["protocol.json", "fits.csv.gz", "prediction_metrics.csv.gz", "control_metrics.csv.gz"]
    digests = {name: hashlib.sha256((study / name).read_bytes()).hexdigest() for name in inputs}
    protocol = json.loads((study / "protocol.json").read_text())
    fits = read_rows(study / "fits.csv.gz")
    predictions = paired_ratios(
        [
            row
            for row in read_rows(study / "prediction_metrics.csv.gz")
            if row["split"] == "holdout"
        ],
        left="selected",
        right="nominal",
        arm_field="model",
        metric="position_rmse_rad",
    )
    control_rows = read_rows(study / "control_metrics.csv.gz")
    if any(row["completed"] != "True" for row in control_rows):
        raise ValueError("partial control episodes must not be plotted as full-horizon ratios")
    control = paired_ratios(
        control_rows,
        left="fitted_ff",
        right="nominal_ff",
        arm_field="controller",
        metric="velocity_rmse_rad_s",
    )
    nominal = {
        (row["cell"], row["profile"]): row
        for row in control_rows
        if row["controller"] == "nominal_ff"
    }
    for row in control:
        row["saturation_delta_pp"] = 100 * (
            float(row["saturation_fraction"])
            - float(nominal[(row["cell"], row["profile"])]["saturation_fraction"])
        )
    grid = protocol["grids"]
    families = grid["calibration_families"]
    frictions = grid["extra_hidden_stribeck_friction_nm"]
    delays = grid["hidden_actual_delay_steps"]
    palette = plt.get_cmap("tab10")
    output.mkdir(parents=True)
    fig, axes = plt.subplots(2, len(families), figsize=(6 * len(families), 7), squeeze=False)
    for column, family in enumerate(families):
        for index, friction in enumerate(frictions):
            select = lambda rows, family=family, friction=friction: [
                row
                for row in rows
                if row["family"] == family and float(row["stribeck_friction_nm"]) == friction
            ]
            label = f"extra friction {friction:g} N m"
            seed_range(
                axes[0, column], select(fits), delays, "fitted_delay_steps", label, palette(index)
            )
            seed_range(axes[1, column], select(predictions), delays, "ratio", label, palette(index))
        axes[0, column].plot(delays, delays, "k:", label="correct delay")
        axes[0, column].set(title=family, ylabel="Selected actuator delay [2 ms steps]")
        axes[0, column].legend(fontsize=8)
        axes[1, column].axhline(1, color="black", linestyle=":")
        axes[1, column].set(
            xlabel="Actual actuator delay [2 ms steps]",
            ylabel="Holdout position RMSE: fitted / nominal",
            yscale="log",
        )
    fig.suptitle("Synthetic position-only identification; bars = noise-seed min/max")
    fig.tight_layout()
    fig.savefig(output / "identification.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), squeeze=False)
    for column, profile in enumerate(("tracking", "reversal", "stress")):
        for family_index, family in enumerate(families):
            for index, friction in enumerate(frictions):
                selected = [
                    row
                    for row in control
                    if row["profile"] == profile
                    and row["family"] == family
                    and float(row["stribeck_friction_nm"]) == friction
                ]
                for line, field in enumerate(("ratio", "saturation_delta_pp")):
                    seed_range(
                        axes[line, column],
                        selected,
                        delays,
                        field,
                        f"{family}, {friction:g} N m",
                        palette(index),
                        "-" if family_index == 0 else "--",
                    )
        axes[0, column].axhline(1, color="black", linestyle=":")
        axes[0, column].set(title=profile, ylabel="Speed RMSE: fitted FF / nominal FF")
        axes[1, column].axhline(0, color="black", linestyle=":")
        axes[1, column].set(
            xlabel="Actual actuator delay [2 ms steps]",
            ylabel="Saturation change [percentage points]",
        )
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Same PI and independent synthetic velocity feedback; bars = noise-seed min/max")
    fig.tight_layout()
    fig.savefig(output / "control.png", dpi=130)
    plt.close(fig)
    if any(
        hashlib.sha256((study / name).read_bytes()).hexdigest() != digest
        for name, digest in digests.items()
    ):
        raise RuntimeError("plot input changed during rendering")
    with (output / "manifest.json").open("x") as stream:
        json.dump(
            {
                "study": str(study),
                "input_sha256": digests,
                "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(output.glob("*.png"))
                },
                "scope": "plots of paired recorded metrics; arithmetic verification is a separate step",
            },
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.study, args.output)


if __name__ == "__main__":
    main()
