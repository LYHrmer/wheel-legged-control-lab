"""Plot both reward-scale seeds without mixing training and physical reward units."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def update_curve(rows: list[dict[str, str]], protocol: dict) -> list[dict]:
    """Place rollout-value EV at its update, including the final logger flush.

    SB3 normally logs the previous update after collecting the next rollout.
    n_updates counts optimization epochs, so it supplies the actual batch
    coordinate when the final flush has no time/total_timesteps field.
    """

    rollout_size = protocol["ppo"]["n_steps"] * protocol["n_envs"]
    epochs = protocol["ppo"]["n_epochs"]
    curve = []
    for row in rows:
        if not row.get("train/n_updates"):
            continue
        updates = int(float(row["train/n_updates"]))
        if updates <= 0 or updates % epochs:
            raise ValueError("n_updates must identify complete PPO epoch groups")
        steps = updates // epochs * rollout_size
        value = float(row["train/explained_variance"])
        if not math.isfinite(value) or (curve and steps <= curve[-1]["update_end_timesteps"]):
            raise ValueError("training log must have finite EV and strictly increasing updates")
        logged = row.get("time/total_timesteps")
        curve.append(
            {
                "update_end_timesteps": steps,
                "explained_variance": value,
                "logger_total_timesteps": int(float(logged)) if logged else None,
            }
        )
    if (
        not curve
        or curve[-1]["update_end_timesteps"] != protocol["actual_budget_per_condition_seed"]
    ):
        raise ValueError("training curve is missing the final update")
    return curve


def paired_metrics(rows: list[dict[str, str]], protocol: dict) -> dict:
    expected_cases = {case["case_id"] for case in protocol["cases"]}
    if len(rows) != len(expected_cases) * (len(protocol["training_seeds"]) + 1):
        raise ValueError("unexpected evaluation row count")
    result = {}
    for seed in (None, *protocol["training_seeds"]):
        condition, seed_text = ("zero_residual", "") if seed is None else ("mixed_rl", str(seed))
        selected = [
            row
            for row in rows
            if row["condition"] == condition and row["training_seed"] == seed_text
        ]
        if (
            len(selected) != len(expected_cases)
            or {row["case_id"] for row in selected} != expected_cases
        ):
            raise ValueError("missing or duplicate paired evaluation cases")
        errors = [float(row["velocity_rmse_mps"]) for row in selected]
        if not all(math.isfinite(value) for value in errors):
            raise ValueError("evaluation errors must be finite")
        result["baseline" if seed is None else str(seed)] = {
            "velocity_rmse_mean_mps": sum(errors) / len(errors),
            "quality_pass": sum(int(row["quality_success"]) for row in selected),
            "case_count": len(selected),
        }
    return result


def run(raw_run: Path, scaled_run: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    inputs = []
    datasets = {}
    protocols = {}
    for label, root in (("raw", raw_run), ("scaled", scaled_run)):
        protocol_path, metrics_path = root / "protocol.json", root / "metrics.csv"
        protocol = json.loads(protocol_path.read_text())
        protocols[label] = protocol
        if protocol["training_conditions"] != ["mixed"] or len(protocol["training_seeds"]) != 2:
            raise ValueError("this figure requires the two-seed mixed-only reward A/B")
        inputs.extend((protocol_path, metrics_path))
        curves = {}
        for seed in protocol["training_seeds"]:
            path = root / "training" / f"mixed_seed_{seed}" / "learning_metrics" / "progress.csv"
            curves[str(seed)] = update_curve(read_rows(path), protocol)
            inputs.append(path)
        datasets[label] = {
            "curves": curves,
            "metrics": paired_metrics(read_rows(metrics_path), protocol),
        }
    differences = {
        key
        for key in protocols["raw"].keys() | protocols["scaled"].keys()
        if protocols["raw"].get(key) != protocols["scaled"].get(key)
    }
    if differences != {"training_reward_scale"}:
        raise ValueError("protocols must differ only in training_reward_scale")
    if (
        protocols["raw"]["training_reward_scale"] != 1.0
        or protocols["scaled"]["training_reward_scale"] != 0.01
    ):
        raise ValueError("expected reward scales 1 and 0.01")
    if datasets["raw"]["metrics"]["baseline"] != datasets["scaled"]["metrics"]["baseline"]:
        raise ValueError("paired baseline metrics differ")

    figure, axes = plt.subplots(1, 3, figsize=(13.6, 4.1), layout="constrained")
    seeds = protocols["raw"]["training_seeds"]
    colors = {"raw": "#64748b", "scaled": "#007f86"}
    for index, seed in enumerate(seeds):
        axis = axes[index]
        for label in ("raw", "scaled"):
            curve = datasets[label]["curves"][str(seed)]
            axis.plot(
                [row["update_end_timesteps"] / 1000 for row in curve],
                [row["explained_variance"] for row in curve],
                color=colors[label],
                label="reward x1" if label == "raw" else "reward x0.01",
                lw=1.4,
            )
        axis.axhline(0, color="0.75", lw=0.7)
        axis.set(
            title=f"Critic EV / seed {seed}",
            xlabel="Update end / 1,000 transitions",
            ylabel="Explained variance",
        )
        axis.legend(frameon=False, fontsize=8)
    for seed, color, marker in zip(seeds, ("#007f86", "#a15f25"), ("o", "s"), strict=True):
        axes[2].plot(
            [0, 1],
            [
                datasets[label]["metrics"][str(seed)]["velocity_rmse_mean_mps"]
                for label in ("raw", "scaled")
            ],
            marker=marker,
            color=color,
            label=f"seed {seed}",
            lw=1.4,
        )
    baseline = datasets["raw"]["metrics"]["baseline"]["velocity_rmse_mean_mps"]
    axes[2].axhline(baseline, color="0.4", ls="--", lw=1, label="zero residual")
    axes[2].set(
        xticks=[0, 1],
        xticklabels=["reward x1", "reward x0.01"],
        xlim=(-0.2, 1.2),
        title="Physical tracking / lower is better",
        ylabel="Mean episode velocity RMSE [m/s]",
    )
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
    output.mkdir(parents=True, exist_ok=False)
    figure.savefig(output / "reward_units_vs_tracking.png", dpi=180)
    plt.close(figure)
    (output / "plotted_data.json").write_text(json.dumps(datasets, indent=2) + "\n")
    manifest = {
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.iterdir())
        },
        "ev_semantics": "rollout-time values, indexed by corresponding update; no post-update reevaluation",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run", type=Path, required=True)
    parser.add_argument("--scaled-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.raw_run, args.scaled_run, args.output)
    print(f"Figure: {args.output / 'reward_units_vs_tracking.png'}")


if __name__ == "__main__":
    main()
