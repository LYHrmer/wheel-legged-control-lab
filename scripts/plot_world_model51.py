"""Plot every initialization from a saved offline-pilot report, without fitting."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report["status"] != "COMPLETE" or len(report["seeds"]) != 3:
        raise ValueError("this figure requires three complete, unselected seeds")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), layout="constrained")
    colors = ("#0072B2", "#D55E00", "#009E73")
    terrains = ("flat", "bumps", "rough", "ramp")
    for i, seed in enumerate(report["seeds"]):
        offset = (i - 1) * 0.24
        label = str(seed["seed"])
        ratios = [seed["gate"]["ratios"][h] for h in ("1", "8")]
        bars = axes[0].bar(np.arange(2) + offset, ratios, 0.23, color=colors[i], label=label)
        axes[0].bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
        terrain_ratios = [seed["scores"]["test"]["8"]["by_terrain"][t]
                          / report["baseline"]["test"]["8"]["by_terrain"][t] for t in terrains]
        axes[1].bar(np.arange(4) + offset, terrain_ratios, 0.23, color=colors[i])
    axes[0].set_xticks([0, 1], ["1 step (10 ms)", "8 steps (80 ms)"])
    axes[0].set_title("Overall velocity prediction error")
    axes[0].plot([0.6, 1.4], [0.9, 0.9], "--", color="#333333", label="8-step gate: 0.90")
    axes[1].set_xticks(np.arange(4), terrains)
    axes[1].set_title("8-step error by terrain")
    axes[1].axhline(1.05, color="#333333", linestyle="--", label="Terrain gate: 1.05")
    for ax in axes:
        ax.axhline(1, color="#777777", linewidth=1)
        ax.set_ylabel("Residual / ridge NRMSE (lower is better)")
        ax.set_ylim(0, 1.85)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Offline dynamics pilot: 3/3 complete, 0/3 pass", fontsize=15)
    fig.supxlabel("Recorded torque inputs; same-course chronological holdout; no policy updates", fontsize=10)
    with args.output.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
