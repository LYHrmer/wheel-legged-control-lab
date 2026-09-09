"""Recompute the frozen reference probe from CSV, without its metric helper.

Only derived_* files and plots/ are written. Source-hash drift is reported
separately from raw-artifact integrity; it cannot rewrite historical evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEIGHT_KD_NS_PER_M = 180.0
FEEDFORWARD_LIMIT_N = 500.0
VARIANTS = (
    "baseline",
    "vertical_feedforward",
    "attitude_tau_20ms",
    "attitude_tau_50ms",
    "attitude_tau_100ms",
)
COLORS = ("#313a46", "#007e83", "#6699cc", "#a978b4", "#cf633e")
LABELS = ("Baseline", "Vertical FF", "LP 20 ms", "LP 50 ms", "LP 100 ms")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def column(rows: list[dict], name: str) -> np.ndarray:
    values = np.asarray([float(row[name]) for row in rows])
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite {name}")
    return values


def verify_hashes(directory: Path, expected: dict[str, str]) -> None:
    for name, digest in expected.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"artifact path escapes input directory: {name}")
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"artifact hash mismatch: {name}")


def close_max(actual: np.ndarray, expected: np.ndarray, label: str) -> float:
    difference = float(np.max(np.abs(actual - expected)))
    if not math.isfinite(difference) or difference > 1e-10:
        raise ValueError(f"{label} mismatch: max absolute error {difference:g}")
    return difference


def recompute_episode(rows: list[dict], case: dict, protocol: dict) -> dict[str, Any]:
    """Independent arithmetic and the protocol's gates, not runner imports."""
    if len(rows) < 2:
        raise ValueError("an episode needs at least two control samples")
    time = column(rows, "time_s")
    dt = float(time[0])
    if dt <= 0:
        raise ValueError("control sample interval must be positive")
    close_max(time, np.arange(1, len(rows) + 1) * dt, "control time")
    close_max(column(rows, "step"), np.arange(1, len(rows) + 1), "control step")
    velocity = column(rows, "velocity_error_mps")
    clearance = column(rows, "clearance_error_m")
    close_max(
        velocity,
        column(rows, "forward_velocity_mps") - column(rows, "command_velocity_mps"),
        "velocity error",
    )
    close_max(
        clearance,
        column(rows, "clearance_m") - column(rows, "command_clearance_m"),
        "clearance error",
    )
    close_max(column(rows, "policy_enabled"), np.zeros(len(rows)), "zero policy")
    close_max(column(rows, "policy_gated"), np.zeros(len(rows)), "no policy gating")
    for axis in ("roll", "pitch"):
        close_max(
            column(rows, f"{axis}_error_rad"),
            column(rows, f"measured_{axis}_rad") - column(rows, f"reward_{axis}_target_rad"),
            f"{axis} error",
        )
        close_max(
            column(rows, f"reward_{axis}_target_rad"),
            column(rows, f"observation_{axis}_target_rad"),
            f"raw {axis} reward/observation target",
        )
    terminal = column(rows, "terminated")
    truncated = column(rows, "truncated")
    if np.any(terminal[:-1]) or np.any(truncated[:-1]):
        raise ValueError("episode contains samples after termination/truncation")
    if not np.isin(terminal, (0, 1)).all() or not np.isin(truncated, (0, 1)).all():
        raise ValueError("terminal flags must be binary")
    criteria = protocol["quality_criteria"]
    tail_count = round(criteria["tail_window_seconds"] / dt)
    if tail_count < 1:
        raise ValueError("tail window contains no samples")
    tail = velocity[-tail_count:]
    bias = float(np.mean(tail))
    variance = float(np.mean((tail - bias) ** 2))
    displacement = math.fsum(column(rows, "command_velocity_mps")) * dt
    initial_x = column(rows, "initial_position_x_m")
    close_max(initial_x, np.full(len(rows), initial_x[0]), "initial position")
    progress = float(rows[-1]["position_x_m"]) - initial_x[0]
    fraction = float(progress / displacement) if abs(displacement) > 1e-12 else None
    rms = lambda value: math.sqrt(float(np.mean(value**2)))
    result = {
        "control_steps": len(rows),
        "simulated_duration_s": len(rows) * dt,
        "completed": int(truncated[-1] == 1 and terminal[-1] == 0),
        "terminated": int(terminal[-1]),
        "termination_reason": rows[-1]["termination_reason"],
        "velocity_rmse_mps": rms(velocity),
        "clearance_rmse_m": rms(clearance),
        "max_abs_yaw_rad": float(np.max(np.abs(column(rows, "yaw_rad")))),
        "final_position_x_m": float(rows[-1]["position_x_m"]),
        "progress_m": float(progress),
        "commanded_displacement_m": displacement,
        "progress_fraction": fraction,
        "mean_torque_saturation_fraction": float(column(rows, "torque_saturation_fraction").mean()),
        "policy_enabled_fraction": 0.0,
        "policy_gated_fraction": 0.0,
        "episode_return": math.fsum(column(rows, "reward")),
        "tail_velocity_rmse_mps": rms(tail),
        "tail_velocity_mean_mps": float(column(rows, "forward_velocity_mps")[-tail_count:].mean()),
        "tail_command_velocity_mean_mps": float(
            column(rows, "command_velocity_mps")[-tail_count:].mean()
        ),
        "tail_observed_seconds": len(tail) * dt,
        "tail_velocity_bias_mps": bias,
        "tail_velocity_std_mps": math.sqrt(variance),
        "tail_velocity_variance_m2ps2": variance,
        "decomposition_error": abs(float(np.mean(tail**2)) - bias**2 - variance),
        "tail_error_type": "bias_dominant" if bias**2 >= variance else "oscillation_dominant",
    }
    if protocol["seconds"] < criteria["minimum_episode_seconds"] or case["velocity_mps"] <= 0:
        raise ValueError("quality is undefined for this duration/command")
    checks = {
        "incomplete_episode": bool(result["completed"])
        and len(rows) == round(protocol["seconds"] / dt),
        "progress": fraction is not None
        and criteria["progress_fraction_min"] <= fraction <= criteria["progress_fraction_max"],
        "tail_velocity": result["tail_velocity_rmse_mps"]
        <= max(
            criteria["tail_velocity_rmse_absolute_mps"],
            criteria["tail_velocity_rmse_target_fraction"] * case["velocity_mps"],
        ),
        "clearance": result["clearance_rmse_m"] <= criteria["clearance_rmse_max_m"],
        "yaw": result["max_abs_yaw_rad"] <= criteria["max_abs_yaw_rad"],
        "torque_saturation": result["mean_torque_saturation_fraction"]
        <= criteria["mean_torque_saturation_fraction_max"],
    }
    result["quality_success"] = int(all(checks.values()))
    result["quality_failure_reasons"] = "|".join(
        key for key, passed in checks.items() if not passed
    )
    return result


def verify_probe(rows: list[dict], variant: str) -> dict[str, float]:
    """Replay logged recurrence after the first (unobserved reset) sample."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    result = {}
    dt = float(rows[0]["time_s"])
    for axis in ("roll", "pitch"):
        control = column(rows, f"control_{axis}_target_rad")
        raw = column(rows, f"reward_{axis}_target_rad")
        close_max(control, column(rows, f"command_{axis}_rad"), f"{axis} logged command")
        if variant.startswith("attitude_tau_"):
            tau = float(variant.removeprefix("attitude_tau_").removesuffix("ms")) / 1000
            alpha = -math.expm1(-dt / tau)
            expected = control[:-1] + alpha * (raw[:-1] - control[:-1])
        else:
            expected = raw[:-1]
        result[f"{axis}_recurrence_max_error_rad"] = close_max(
            control[1:], expected, f"{axis} pre-control reference recurrence"
        )
    height_velocity = column(rows, "height_velocity_reference_mps")
    request = column(rows, "requested_vertical_feedforward_n")
    applied = column(rows, "applied_vertical_feedforward_n")
    expected = (
        HEIGHT_KD_NS_PER_M * height_velocity
        if variant == "vertical_feedforward"
        else np.zeros(len(rows))
    )
    result["feedforward_request_max_error_n"] = close_max(request, expected, "feedforward request")
    result["feedforward_clip_max_error_n"] = close_max(
        applied,
        np.clip(request, -FEEDFORWARD_LIMIT_N, FEEDFORWARD_LIMIT_N),
        "logged feedforward clip",
    )
    if variant != "vertical_feedforward":
        close_max(height_velocity, np.zeros(len(rows)), "disabled height feedforward")
    result["feedforward_max_abs_request_n"] = float(np.max(np.abs(request)))
    result["feedforward_clipped_steps"] = int(
        np.count_nonzero(np.abs(request) > FEEDFORWARD_LIMIT_N)
    )
    return result


def compare_metrics(recomputed: dict, stored: dict) -> float:
    largest = 0.0
    for key, expected in stored.items():
        if key in ("variant", "case_id", "repeat"):
            continue
        actual = recomputed[key]
        if isinstance(actual, str) or actual is None:
            if actual != expected:
                raise ValueError(f"metric mismatch: {key}: {actual!r} != {expected!r}")
        else:
            difference = abs(actual - float(expected))
            if not math.isfinite(difference) or difference > 1e-9:
                raise ValueError(f"metric mismatch: {key}: {difference:g}")
            largest = max(largest, difference)
    return largest


def aggregate(records: list[dict]) -> list[dict]:
    result = []
    for variant in VARIANTS:
        for kind in ("all", "flat", "bumps", "ramp"):
            group = [
                item
                for item in records
                if item["variant"] == variant and (kind == "all" or item["terrain_kind"] == kind)
            ]
            if not group:
                continue
            result.append(
                {
                    "variant": variant,
                    "terrain_kind": kind,
                    "episodes": len(group),
                    "quality_passes": sum(item["quality_success"] for item in group),
                    **{
                        "mean_" + key: math.fsum(item[key] for item in group) / len(group)
                        for key in (
                            "velocity_rmse_mps",
                            "tail_velocity_rmse_mps",
                            "clearance_rmse_m",
                        )
                    },
                }
            )
    return result


def plot_results(directory: Path, traces: dict, records: list[dict]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_directory = directory / "plots"
    plot_directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    paths = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.7), sharex=True, constrained_layout=True)
    for col, (index, title) in enumerate(
        ((4, "5 mm / 0.25 m/s"), (10, "10 mm / 0.25 m/s"), (11, "10 mm / 0.40 m/s"))
    ):
        case_id = f"tracking_v2_development_{index:02d}_bumps"
        for variant, label, color in zip(VARIANTS, LABELS, COLORS, strict=True):
            rows = traces[(variant, case_id, 0)]
            time = column(rows, "time_s")
            axes[0, col].plot(
                time, column(rows, "velocity_error_mps"), color=color, lw=1.2, label=label
            )
            axes[1, col].plot(time, column(rows, "clearance_error_m") * 1000, color=color, lw=1.2)
        axes[0, col].set_title(f"Development case {index:02d}\n{title}")
        axes[1, col].set_xlabel("Time (s)")
        for ax in axes[:, col]:
            ax.axhline(0, color="#888888", ls=":", lw=0.8)
            ax.axvspan(3, 4, color="#bbbbbb", alpha=0.16)
            ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Forward velocity error (m/s)")
    axes[1, 0].set_ylabel("Clearance error (mm)")
    axes[0, 0].legend(fontsize=8, loc="lower left", ncol=2)
    fig.suptitle("Zero residual, identical development cases; shaded region is the final 1 s")
    paths.append(plot_directory / "bump_tracking.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    case_id = "tracking_v2_development_11_bumps"
    filtered = traces[("attitude_tau_100ms", case_id, 0)]
    ff = traces[("vertical_feedforward", case_id, 0)]
    time = column(filtered, "time_s")
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(
        time[:-1],
        np.rad2deg(column(filtered, "reward_pitch_target_rad")[:-1]),
        color="#313a46",
        ls="--",
        label="Raw target at control time",
    )
    axes[0].plot(
        time[:-1],
        np.rad2deg(column(filtered, "control_pitch_target_rad")[1:]),
        color=COLORS[-1],
        label="LP 100 ms applied target",
    )
    axes[0].plot(
        time,
        np.rad2deg(column(filtered, "measured_pitch_rad")),
        color="#6699cc",
        alpha=0.8,
        label="Measured pitch (post-step)",
    )
    axes[0].set_ylabel("Pitch (deg)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].set_title("Development case 11: each panel uses its own variant's trajectory")
    ff_time = column(ff, "time_s")
    axes[1].plot(
        ff_time - float(ff_time[0]),
        column(ff, "height_velocity_reference_mps"),
        color=COLORS[1],
        label="Height-reference velocity (pre-step)",
    )
    axes[1].plot(
        ff_time,
        column(ff, "body_vertical_velocity_mps"),
        color="#313a46",
        alpha=0.7,
        label="World-z COM velocity (post-step)",
    )
    axes[1].set_ylabel("Vertical velocity (m/s)")
    axes[1].legend(fontsize=8)
    axes[2].plot(
        ff_time - float(ff_time[0]), column(ff, "requested_vertical_feedforward_n"), color=COLORS[1]
    )
    axes[2].set_ylabel("Requested vertical FF (N)")
    axes[2].set_xlabel("Physical time (s)")
    for ax in axes:
        ax.axhline(0, color="#888888", ls=":", lw=0.8)
        ax.grid(alpha=0.2)
    paths.append(plot_directory / "reference_mechanism_case11.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    base = {item["case_id"]: item for item in records if item["variant"] == "baseline"}
    for variant, label, color in zip(VARIANTS[1:], LABELS[1:], COLORS[1:], strict=True):
        selected = [item for item in records if item["variant"] == variant]
        for ax, field, scale in zip(
            axes, ("tail_velocity_rmse_mps", "clearance_rmse_m"), (1, 1000), strict=True
        ):
            ax.plot(
                range(len(selected)),
                [(item[field] - base[item["case_id"]][field]) * scale for item in selected],
                ".-",
                ms=4,
                lw=1,
                color=color,
                label=label,
            )
    for ax in axes:
        ax.axhline(0, color="#313a46", lw=1)
        ax.axvspan(3.5, 15.5, color="#bbbbbb", alpha=0.15)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("Development case index (shaded: bumps)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Tail velocity RMSE change (m/s)")
    axes[1].set_ylabel("Clearance RMSE change (mm)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Paired change relative to baseline; lower is better, all 24 cases shown")
    paths.append(plot_directory / "paired_changes.png")
    fig.savefig(paths[-1], dpi=180)
    plt.close(fig)
    return paths


def audit(directory: Path, *, plots: bool = True) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest_digest = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    verify_hashes(directory, manifest)
    protocol = json.loads((directory / "protocol.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    metric_rows = read_csv(directory / "metrics.csv")

    def key(item):
        return item["variant"], item["case_id"], int(item["repeat"])

    stored = {key(row): row for row in metric_rows}
    summarized = {key(row): row for row in summary["metrics"]}
    expected_count = len(protocol["variants"]) * len(protocol["cases"]) * protocol["repeats"]
    if not (
        len(stored)
        == len(metric_rows)
        == len(summarized)
        == len(summary["metrics"])
        == summary["episodes"]
        == expected_count
    ):
        raise ValueError("duplicate or missing episode metric records")
    records, traces = [], {}
    largest_metric_difference = 0.0
    for variant in protocol["variants"]:
        for case in protocol["cases"]:
            for repeat in range(protocol["repeats"]):
                identity = variant, case["case_id"], repeat
                name = f"{variant}__{case['case_id']}__repeat_{repeat}.csv"
                if name not in manifest:
                    raise ValueError(f"unhashed trace: {name}")
                rows = read_csv(directory / name)
                result = recompute_episode(rows, case, protocol)
                for original in (stored[identity], summarized[identity]):
                    largest_metric_difference = max(
                        largest_metric_difference, compare_metrics(result, original)
                    )
                result.update(verify_probe(rows, variant))
                records.append(
                    {
                        "variant": variant,
                        "case_id": case["case_id"],
                        "repeat": repeat,
                        "terrain_kind": case["terrain"]["kind"],
                        **result,
                    }
                )
                traces[identity] = rows
    current_source_mismatches = [
        name
        for name, digest in protocol["source_sha256"].items()
        if not (ROOT / name).is_file() or sha256(ROOT / name) != digest
    ]
    report = {
        "raw_artifact_hashes_verified": len(manifest),
        "original_manifest_sha256": manifest_digest,
        "episodes_recomputed": len(records),
        "control_samples_recomputed": sum(item["control_steps"] for item in records),
        "metric_max_absolute_difference": largest_metric_difference,
        "decomposition_max_absolute_error_m2ps2": max(
            item["decomposition_error"] for item in records
        ),
        "probe_checks_max_absolute_errors": {
            name: max(item[name] for item in records)
            for name in (
                "roll_recurrence_max_error_rad",
                "pitch_recurrence_max_error_rad",
                "feedforward_request_max_error_n",
                "feedforward_clip_max_error_n",
            )
        },
        "feedforward_max_abs_request_n": max(
            item["feedforward_max_abs_request_n"] for item in records
        ),
        "feedforward_clipped_steps": sum(item["feedforward_clipped_steps"] for item in records),
        "source_hashes_checked_now": len(protocol["source_sha256"]),
        "current_source_mismatches": current_source_mismatches,
        "original_runner_reported_source_unchanged_during_run": summary[
            "source_unchanged_during_run"
        ],
        "limits": [
            "Development cases, one deterministic repeat each; no confidence intervals or held-out generalization claim.",
            "Current source hashes and the runner's historical start/end check do not independently prove every intermediate runtime instruction.",
            "Filter recurrence is checked from sample 2 using the previous post-step raw target; first reset target is not independently logged.",
            "No pre-step world-x COM velocity is logged: hdot=-tan(pitch)*vx_world cannot be completely recomputed from the CSV.",
            "Ground height is sampled at the body origin, but feedforward uses COM velocity. Exact origin-height differentiation needs v_origin=v_COM-omega_world cross (R*r_origin_COM). The existing probe preserves this legacy approximation.",
            "Applied feedforward is a logger-side clip calculation, not an independent measured actuator-force channel.",
            "body_vertical_velocity_mps is post-step WORLD-z inertial-COM velocity because base_velocity() defaults to local=False.",
            "The 180 Ns/m derivative gain and +/-500 N clip were checked against hashed source, not inferred from telemetry.",
        ],
    }
    outputs = []
    metrics_path = directory / "derived_metrics.csv"
    with metrics_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    outputs.append(metrics_path)
    for name, value in (
        ("derived_audit.json", report),
        (
            "derived_summary.json",
            {
                "aggregation_unit": "case; unweighted mean of episode RMSEs, not pooled-sample RMSE",
                "groups": aggregate(records),
            },
        ),
    ):
        path = directory / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        outputs.append(path)
    if plots:
        outputs.extend(plot_results(directory, traces, records))
    verify_hashes(directory, manifest)
    if sha256(manifest_path) != manifest_digest:
        raise ValueError("original manifest changed during audit")
    (directory / "derived_manifest.json").write_text(
        json.dumps(
            {
                "original_manifest_sha256": manifest_digest,
                "audit_script_sha256": sha256(Path(__file__)),
                "derived_artifacts_sha256": {
                    str(path.relative_to(directory)): sha256(path) for path in outputs
                },
            },
            indent=2,
        )
        + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/d1_reference_dynamics")
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.input), indent=2))


if __name__ == "__main__":
    main()
