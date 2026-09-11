"""Read-only plotting of D1 checkpoint-budget learning curves.

Consumes a precomputed, independent audit report (``d1-budget-analysis-v1``) and
renders one figure per evaluation split.  This script never recomputes physics,
never runs training or evaluation, never imports project packages, and never
modifies its input.

Adapted from a Claude Opus implementation; validation and tests checked locally.

Plotting rules:
  * A plotted point is the arithmetic mean over all 4 evaluation cases ONLY when
    ``completed == cases == 4``.  Otherwise every metric of that point is NaN so
    the line visibly breaks.  Partial-survivor means are never plotted.
  * The zero-action reference line is drawn only when its own 4 cases completed;
    otherwise it is omitted and the omission is stated on the figure.
  * One curve == one continuous training seed.  Checkpoint budgets along a run
    are dependent observations, not extra replicates.  No CIs, no interpolation,
    no best-checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend must be selected before pyplot import

import matplotlib.pyplot as plt
import numpy as np

SCHEMA = "d1-budget-analysis-v1"
# kind -> exact required number of independent training seeds
REQUIRED_SEED_COUNT = {"formal": 3, "smoke": 1}
SPLITS = ("development", "holdout")
MODES = ("shared2", "independent8")
CASES = 4

METRICS = (
    "velocity_rmse_mps",
    "yaw_rmse_rps",
    "height_rmse_m",
    "attitude_rmse_rad",
    "mean_mechanical_power_w",
    "action_rms",
)

METRIC_LABELS = {
    "velocity_rmse_mps": "speed tracking RMSE [m/s]",
    "yaw_rmse_rps": "yaw-rate RMSE [rad/s]",
    "height_rmse_m": "height RMSE [m]",
    "attitude_rmse_rad": "attitude RMSE [rad]",
    "mean_mechanical_power_w": "mean mechanical power [W]",
    "action_rms": "physical action RMS [normalized 8-D]",
}

METRIC_TITLES = {
    "velocity_rmse_mps": "speed",
    "yaw_rmse_rps": "yaw rate",
    "height_rmse_m": "height",
    "attitude_rmse_rad": "attitude",
    "mean_mechanical_power_w": "mechanical power (activity)",
    "action_rms": "physical action RMS",
}

MODE_COLORS = {"shared2": "#1f77b4", "independent8": "#ff7f0e"}
SEED_LINESTYLES = ("-", "--", ":")
ZERO_COLOR = "#7f7f7f"

COUNT_FIELDS = (
    "split",
    "mode",
    "training_seed",
    "budget",
    "cases",
    "completed",
    "failed",
    "quality_passes",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, where: str) -> dict:
    _require(isinstance(value, dict), f"{where}: expected object, got {type(value).__name__}")
    return value


def _sequence(value: Any, where: str) -> list:
    _require(isinstance(value, list), f"{where}: expected array, got {type(value).__name__}")
    return value


def _strict_int(value: Any, where: str) -> int:
    # bool is an int subclass in Python; it is never a valid count/seed/budget.
    _require(type(value) is int, f"{where}: expected integer, got {type(value).__name__}")
    return value


def _finite_nonnegative(value: Any, where: str) -> float:
    _require(type(value) in (int, float), f"{where}: expected number, got {type(value).__name__}")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{where}: number is outside plotting range") from exc
    if not math.isfinite(number):
        raise ValueError(f"{where}: expected finite number, got {value!r}")
    if number < 0.0:
        raise ValueError(f"{where}: expected non-negative number, got {number!r}")
    return number


def _exact_keys(value: dict, expected: Iterable[str], where: str) -> None:
    expected_set = set(expected)
    actual = set(value)
    missing = sorted(expected_set - actual)
    extra = sorted(actual - expected_set)
    if missing:
        raise ValueError(f"{where}: missing key(s) {missing}")
    if extra:
        raise ValueError(f"{where}: unexpected key(s) {extra}")


def _required_keys(value: dict, expected: Iterable[str], where: str) -> None:
    missing = sorted(k for k in expected if k not in value)
    if missing:
        raise ValueError(f"{where}: missing key(s) {missing}")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key: {key!r}")
        seen.add(key)
    return dict(pairs)


def _reject_nonfinite_json(value: str):
    raise ValueError(f"non-finite JSON value: {value}")


def load_report(text: str) -> dict:
    """Parse report JSON, rejecting duplicate keys and non-finite constants."""
    report = json.loads(
        text, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_nonfinite_json
    )
    return _mapping(report, "report")


def _validate_summary(raw: Any, where: str) -> dict:
    """Validate one upstream SUMMARY block; extra sibling fields are allowed."""
    summary = _mapping(raw, where)
    _required_keys(
        summary,
        ("cases", "completed", "failed", "quality_passes", "completed_only_means"),
        where,
    )

    cases = _strict_int(summary["cases"], f"{where}.cases")
    if cases != CASES:
        raise ValueError(f"{where}.cases: expected {CASES}, got {cases}")

    completed = _strict_int(summary["completed"], f"{where}.completed")
    if not 0 <= completed <= CASES:
        raise ValueError(f"{where}.completed: expected 0..{CASES}, got {completed}")

    failed = _strict_int(summary["failed"], f"{where}.failed")
    if failed != CASES - completed:
        raise ValueError(
            f"{where}.failed: expected {CASES - completed} (cases - completed), got {failed}"
        )

    quality_passes = _strict_int(summary["quality_passes"], f"{where}.quality_passes")
    if not 0 <= quality_passes <= completed:
        raise ValueError(f"{where}.quality_passes: expected 0..{completed}, got {quality_passes}")

    raw_means = summary["completed_only_means"]
    means: dict[str, float] | None
    if completed == 0:
        if raw_means is not None:
            raise ValueError(f"{where}.completed_only_means: expected null when completed == 0")
        means = None
    else:
        block = _mapping(raw_means, f"{where}.completed_only_means")
        _exact_keys(block, METRICS, f"{where}.completed_only_means")
        # Means must be finite and non-negative even when partial (never plotted).
        means = {
            metric: _finite_nonnegative(block[metric], f"{where}.completed_only_means.{metric}")
            for metric in METRICS
        }

    return {
        "cases": cases,
        "completed": completed,
        "failed": failed,
        "quality_passes": quality_passes,
        "means": means,
    }


def _validate_header(report: dict) -> tuple[str, list[int], list[int]]:
    _required_keys(
        report,
        ("schema", "kind", "training_seeds", "checkpoint_budgets", "comparisons"),
        "report",
    )

    schema = report["schema"]
    if schema != SCHEMA:
        raise ValueError(f"report.schema: expected {SCHEMA!r}, got {schema!r}")

    kind = report["kind"]
    if type(kind) is not str or kind not in REQUIRED_SEED_COUNT:
        raise ValueError(
            f"report.kind: expected one of {sorted(REQUIRED_SEED_COUNT)}, got {kind!r}"
        )

    seeds_raw = _sequence(report["training_seeds"], "report.training_seeds")
    expected_seeds = REQUIRED_SEED_COUNT[kind]
    if len(seeds_raw) != expected_seeds:
        raise ValueError(
            f"report.training_seeds: kind {kind!r} requires exactly "
            f"{expected_seeds} seed(s), got {len(seeds_raw)}"
        )
    seeds: list[int] = []
    for index, value in enumerate(seeds_raw):
        seed = _strict_int(value, f"report.training_seeds[{index}]")
        if seed < 0:
            raise ValueError(f"report.training_seeds[{index}]: expected non-negative, got {seed}")
        if seed in seeds:
            raise ValueError(f"report.training_seeds: duplicate seed {seed}")
        seeds.append(seed)

    budgets_raw = _sequence(report["checkpoint_budgets"], "report.checkpoint_budgets")
    if not budgets_raw:
        raise ValueError("report.checkpoint_budgets: must be non-empty")
    budgets: list[int] = []
    for index, value in enumerate(budgets_raw):
        budget = _strict_int(value, f"report.checkpoint_budgets[{index}]")
        if budget <= 0:
            raise ValueError(f"report.checkpoint_budgets[{index}]: expected positive, got {budget}")
        # strictly increasing also rejects duplicates and reversed order
        if budgets and budget <= budgets[-1]:
            raise ValueError(
                "report.checkpoint_budgets: must be strictly increasing, got "
                f"{budgets[-1]} then {budget}"
            )
        budgets.append(budget)

    return kind, seeds, budgets


def _validate_matrix(
    split_block: dict, split: str, seeds: list[int], budgets: list[int]
) -> dict[tuple[int, int, str], dict]:
    """Validate the exact seed x budget x mode matrix of one split."""
    where = f"report.comparisons.{split}"
    _required_keys(split_block, ("zero_physical_baseline", "per_seed"), where)

    per_seed = _sequence(split_block["per_seed"], f"{where}.per_seed")
    if len(per_seed) != len(seeds):
        raise ValueError(
            f"{where}.per_seed: expected {len(seeds)} entries "
            f"(one per declared seed), got {len(per_seed)}"
        )

    cells: dict[tuple[int, int, str], dict] = {}
    seen_seeds: list[int] = []
    for seed_index, raw_entry in enumerate(per_seed):
        seed_where = f"{where}.per_seed[{seed_index}]"
        entry = _mapping(raw_entry, seed_where)
        _required_keys(entry, ("training_seed", "budgets"), seed_where)
        seed = _strict_int(entry["training_seed"], f"{seed_where}.training_seed")
        if seed not in seeds:
            raise ValueError(f"{seed_where}.training_seed: {seed} is not a declared training seed")
        if seed in seen_seeds:
            raise ValueError(f"{seed_where}.training_seed: duplicate seed {seed}")
        seen_seeds.append(seed)

        seed_budgets = _sequence(entry["budgets"], f"{seed_where}.budgets")
        if len(seed_budgets) != len(budgets):
            raise ValueError(
                f"{seed_where}.budgets: expected {len(budgets)} entries "
                f"(one per declared budget), got {len(seed_budgets)}"
            )
        seen_budgets: list[int] = []
        for budget_index, raw_budget in enumerate(seed_budgets):
            budget_where = f"{seed_where}.budgets[{budget_index}]"
            budget_entry = _mapping(raw_budget, budget_where)
            _required_keys(budget_entry, ("budget", "modes"), budget_where)
            budget = _strict_int(budget_entry["budget"], f"{budget_where}.budget")
            if budget not in budgets:
                raise ValueError(
                    f"{budget_where}.budget: {budget} is not a declared checkpoint budget"
                )
            if budget in seen_budgets:
                raise ValueError(f"{budget_where}.budget: duplicate budget {budget}")
            seen_budgets.append(budget)

            modes = _mapping(budget_entry["modes"], f"{budget_where}.modes")
            _exact_keys(modes, MODES, f"{budget_where}.modes")
            for mode in MODES:
                cells[(seed, budget, mode)] = _validate_summary(
                    modes[mode], f"{budget_where}.modes.{mode}"
                )

    missing_seeds = sorted(set(seeds) - set(seen_seeds))
    if missing_seeds:
        raise ValueError(f"{where}.per_seed: missing seed(s) {missing_seeds}")
    return cells


def _point_values(summary: dict) -> dict[str, float]:
    """Full-case mean per metric, or NaN for every metric when incomplete."""
    if summary["completed"] == summary["cases"] == CASES and summary["means"]:
        return {metric: float(summary["means"][metric]) for metric in METRICS}
    return {metric: float("nan") for metric in METRICS}


def prepare_curves(report: Any) -> tuple[dict, list[dict]]:
    """Validate the consumed contract and build plot-ready curves plus counts."""
    report = _mapping(report, "report")
    kind, seeds, budgets = _validate_header(report)

    comparisons = _mapping(report["comparisons"], "report.comparisons")
    _exact_keys(comparisons, SPLITS, "report.comparisons")

    curves: dict[str, dict] = {}
    counts: list[dict] = []

    for split in SPLITS:
        split_block = _mapping(comparisons[split], f"report.comparisons.{split}")
        _required_keys(
            split_block, ("zero_physical_baseline", "per_seed"), f"report.comparisons.{split}"
        )
        zero = _validate_summary(
            split_block["zero_physical_baseline"],
            f"report.comparisons.{split}.zero_physical_baseline",
        )
        cells = _validate_matrix(split_block, split, seeds, budgets)

        # Exactly one zero-summary row per split (never replicated per seed/budget).
        counts.append(
            {
                "split": split,
                "mode": "zero",
                "training_seed": "",
                "budget": "",
                "cases": zero["cases"],
                "completed": zero["completed"],
                "failed": zero["failed"],
                "quality_passes": zero["quality_passes"],
            }
        )

        series: list[dict] = []
        for seed in seeds:
            for mode in MODES:
                values = {metric: [] for metric in METRICS}
                for budget in budgets:
                    point = _point_values(cells[(seed, budget, mode)])
                    for metric in METRICS:
                        values[metric].append(point[metric])
                series.append(
                    {
                        "mode": mode,
                        "training_seed": seed,
                        "budgets": list(budgets),
                        "values": values,
                    }
                )

        # Every matrix entry is retained in counts, including total failures.
        for seed in seeds:
            for budget in budgets:
                for mode in MODES:
                    summary = cells[(seed, budget, mode)]
                    counts.append(
                        {
                            "split": split,
                            "mode": mode,
                            "training_seed": seed,
                            "budget": budget,
                            "cases": summary["cases"],
                            "completed": summary["completed"],
                            "failed": summary["failed"],
                            "quality_passes": summary["quality_passes"],
                        }
                    )

        curves[split] = {"zero": _point_values(zero), "series": series}

    curves["kind"] = kind
    curves["training_seeds"] = list(seeds)
    curves["checkpoint_budgets"] = list(budgets)
    return curves, counts


def _footer_text(kind: str, zero_available: bool) -> str:
    seed_count = REQUIRED_SEED_COUNT[kind]
    lines = [
        (
            "Gaps = any failed case; points require all 4 cases completed. "
            "No partial-success means or filled gaps."
        ),
        (
            f"Independent unit = training seed ({seed_count} in this {kind} run). "
            "Budgets along one run are dependent; no CIs or best-checkpoint selection."
        ),
        (
            "Mechanical power measures activity, not battery consumption. "
            "Action RMS uses the normalized 8-D physical action, not the policy Gaussian."
        ),
    ]
    if zero_available:
        lines.append("Gray dashed line = zero-physical-action reference (all 4 cases completed).")
    else:
        lines.append(
            "Zero-physical-action reference line omitted: its own 4 cases did not all complete (reference missing)."
        )
    return "\n".join(lines)


def make_figure(split: str, prepared: dict, kind: str):
    """Build the 2x3 figure for one split. Never writes files."""
    if split not in SPLITS:
        raise ValueError(f"split: expected one of {list(SPLITS)}, got {split!r}")
    if kind not in REQUIRED_SEED_COUNT:
        raise ValueError(f"kind: expected one of {sorted(REQUIRED_SEED_COUNT)}, got {kind!r}")
    prepared = _mapping(prepared, "prepared")
    _required_keys(prepared, ("zero", "series"), "prepared")

    series = prepared["series"]
    if not series:
        raise ValueError("prepared.series: must be non-empty")
    budgets = list(series[0]["budgets"])
    zero = prepared["zero"]
    zero_available = all(math.isfinite(float(zero[metric])) for metric in METRICS)

    seed_order: list[int] = []
    for entry in series:
        if entry["training_seed"] not in seed_order:
            seed_order.append(entry["training_seed"])

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.0))
    flat_axes = axes.ravel()

    for index, metric in enumerate(METRICS):
        ax = flat_axes[index]
        for entry in series:
            mode = entry["mode"]
            seed = entry["training_seed"]
            style = SEED_LINESTYLES[seed_order.index(seed) % len(SEED_LINESTYLES)]
            ax.plot(
                np.asarray(entry["budgets"], dtype=float),
                np.asarray(entry["values"][metric], dtype=float),
                color=MODE_COLORS[mode],
                linestyle=style,
                marker="o",
                markersize=5.0,
                linewidth=1.8,
                label=f"{mode} - seed {seed}" if index == 0 else None,
            )
        if zero_available:
            ax.axhline(
                float(zero[metric]),
                color=ZERO_COLOR,
                linestyle="--",
                linewidth=1.4,
                label="zero physical action" if index == 0 else None,
            )
        else:
            ax.text(
                0.02,
                0.96,
                "zero reference missing",
                transform=ax.transAxes,
                fontsize=8,
                color=ZERO_COLOR,
                va="top",
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(budget) for budget in budgets])
        ax.minorticks_off()
        ax.set_xlabel("checkpoint budget (training steps, log2)")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(METRIC_TITLES[metric], fontsize=11)
        ax.grid(True, which="major", alpha=0.3)

    fig.suptitle(
        f"D1 checkpoint-budget study - {split} split ({kind}; "
        f"{len(seed_order)} training seed(s), {len(budgets)} budget(s))",
        fontsize=14,
    )

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.145),
        ncol=min(4, max(1, len(labels))),
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.015,
        _footer_text(kind, zero_available),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
    )

    if kind == "smoke":
        fig.text(
            0.5,
            0.5,
            "SMOKE - interface only",
            ha="center",
            va="center",
            fontsize=48,
            color="red",
            alpha=0.18,
            rotation=25,
            zorder=10,
        )

    fig.tight_layout(rect=(0.02, 0.19, 0.98, 0.94))
    return fig


def _reject_traversal(path: Path, label: str) -> None:
    if ".." in path.parts:
        raise ValueError(f"{label}: '..' path traversal is not allowed: {path}")


def _reject_symlink_components(path: Path, label: str, include_self: bool) -> None:
    parts = list(path.parents)[::-1]
    for parent in parts:
        if parent.is_symlink():
            raise ValueError(f"{label}: symlinked path component is not allowed: {parent}")
    if include_self and os.path.islink(path):
        raise ValueError(f"{label}: symlink is not allowed: {path}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _open_exclusive(path: Path, binary: bool):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    if binary:
        return os.fdopen(fd, "wb")
    return os.fdopen(fd, "w", newline="", encoding="utf-8")


def plot_analysis(analysis_path: str | os.PathLike[str], output: str | os.PathLike[str]) -> dict:
    """Validate, then render figures/counts/manifest into a new output directory."""
    analysis = Path(analysis_path)
    out_dir = Path(output)

    _reject_traversal(analysis, "analysis")
    _reject_traversal(out_dir, "output")

    abs_analysis = Path(os.path.abspath(analysis))
    abs_output = Path(os.path.abspath(out_dir))

    _reject_symlink_components(abs_analysis, "analysis", include_self=True)
    _reject_symlink_components(abs_output, "output", include_self=False)

    if not abs_analysis.exists():
        raise OSError(f"analysis: input file does not exist: {abs_analysis}")
    if not abs_analysis.is_file():
        raise ValueError(f"analysis: expected a regular JSON file: {abs_analysis}")

    # lexists also catches dangling symlinks and empty directories
    if os.path.lexists(abs_output):
        raise ValueError(f"output: refusing to reuse existing path: {abs_output}")

    input_tree = abs_analysis.parent
    if abs_output == input_tree or input_tree in abs_output.parents:
        raise ValueError(f"output: must not live inside the input's directory tree ({input_tree})")
    if abs_output in abs_analysis.parents:
        raise ValueError(f"output: must not contain the input file: {abs_output}")

    raw_bytes = abs_analysis.read_bytes()
    report = load_report(raw_bytes.decode("utf-8"))

    # Full validation happens before any output is created.
    curves, counts = prepare_curves(report)
    kind = curves["kind"]

    figures = {split: make_figure(split, curves[split], kind) for split in SPLITS}

    abs_output.mkdir(parents=True)
    written: dict[str, str] = {}
    try:
        for split, figure in figures.items():
            name = f"{split}.png"
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=120)
            plt.close(figure)
            with _open_exclusive(abs_output / name, binary=True) as handle:
                handle.write(buffer.getvalue())
            written[name] = _sha256_bytes(buffer.getvalue())

        with _open_exclusive(abs_output / "counts.csv", binary=False) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(COUNT_FIELDS))
            writer.writeheader()
            writer.writerows(counts)
        written["counts.csv"] = _sha256_file(abs_output / "counts.csv")
    finally:
        for figure in figures.values():
            plt.close(figure)

    script_path = Path(os.path.abspath(__file__))
    manifest = {
        "schema": "d1-budget-plots-v1",
        "input_schema": SCHEMA,
        "kind": kind,
        "scope": (
            "Read-only replot of a precomputed independent D1 budget audit. This plotting "
            "script does not run physics, training or evaluation and does not modify its input. Points are "
            "plotted only when all 4 evaluation cases completed, otherwise the curve breaks. "
            "The zero-physical-action reference is drawn only when its own 4 cases completed. "
            "No confidence intervals, no interpolation, no best-checkpoint selection; the "
            "independent unit is the training seed."
        ),
        "analysis_path": str(abs_analysis),
        "analysis_sha256": _sha256_bytes(raw_bytes),
        "script_path": str(script_path),
        "script_sha256": _sha256_file(script_path),
        "splits": list(SPLITS),
        "modes": list(MODES),
        "metrics": list(METRICS),
        "cases_per_point": CASES,
        "training_seeds": list(curves["training_seeds"]),
        "checkpoint_budgets": list(curves["checkpoint_budgets"]),
        "zero_reference_available": {
            split: all(math.isfinite(curves[split]["zero"][metric]) for metric in METRICS)
            for split in SPLITS
        },
        "outputs": {name: written[name] for name in sorted(written)},
    }

    with _open_exclusive(abs_output / "manifest.json", binary=False) as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
        handle.write("\n")

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="plot_d1_budget_study",
        description=(
            "Plot D1 checkpoint-budget learning curves from a precomputed "
            "d1-budget-analysis-v1 report (read-only)."
        ),
    )
    parser.add_argument("--analysis", required=True, help="path to analysis.json")
    parser.add_argument("--output", required=True, help="new output directory to create")
    args = parser.parse_args(argv)

    try:
        manifest = plot_analysis(args.analysis, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
        return 2  # pragma: no cover - parser.error exits

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
