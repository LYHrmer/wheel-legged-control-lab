"""Position-only calibration and paired single-wheel feedforward experiments.

PACE-inspired, not a reproduction of its position-PD/Isaac/CMA-ES pipeline.
Only the new fitter uses position alone. Closed-loop speed feedback retains the
existing synthetic velocity sensor. No D1 hardware or full-robot claim is made.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import tarfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from numbers import Integral
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# This executable reuses repository experiment helpers, not installed entrypoints.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_wheel_control import PROFILES, compute_metrics, run_closed_loop
from scripts.run_actuator_identification import collect
from wheel_legged_control.actuator_bench import ActuatorParameters, BenchConfig
from wheel_legged_control.actuator_identification import (
    PARAMETER_LOWER,
    PARAMETER_NAMES,
    PARAMETER_UPPER,
    POSITION_SCALE_RAD,
    VELOCITY_SCALE_RAD_S,
    fit_actuator_parameters,
    load_actuator_log,
    predict_actuator_log,
    save_actuator_log,
)
from wheel_legged_control.encoder_identification import (
    EncoderLog,
    encoder_prediction_metrics,
    fit_encoder_parameters,
    load_encoder_log,
    predict_encoder_log,
    save_encoder_log,
)
from wheel_legged_control.provenance import capture_git_provenance
from wheel_legged_control.wheel_control import WheelControlConfig, WheelVelocityController

SOURCE_FILES = (
    "scripts/evaluate_encoder_identification.py",
    "scripts/evaluate_wheel_control.py",
    "scripts/run_actuator_identification.py",
    "src/wheel_legged_control/__init__.py",
    "src/wheel_legged_control/controllers.py",
    "src/wheel_legged_control/model.py",
    "src/wheel_legged_control/rewards.py",
    "src/wheel_legged_control/encoder_identification.py",
    "src/wheel_legged_control/actuator_identification.py",
    "src/wheel_legged_control/actuator_bench.py",
    "src/wheel_legged_control/wheel_control.py",
    "src/wheel_legged_control/provenance.py",
    "pyproject.toml",
)
ARMS = ("pi", "nominal_ff", "position_ff", "position_velocity_ff")
COLORS = ("#777777", "#ba6738", "#24789a", "#8467a7")
TRUTH = ActuatorParameters(0.017, 0.105, 0.043, 2)
SPLITS = ("calibration", "calibration", "validation", "holdout")


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    """Keep complete per-step records; only compress their transport encoding."""
    if not rows:
        raise ValueError("cannot write an empty table")
    with gzip.open(path, "xt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_args(args: argparse.Namespace) -> None:
    for name, minimum in (("duration", 1.0), ("calibration_duration", 0.5), ("noise_scale", 0.0)):
        value = getattr(args, name)
        if isinstance(value, bool) or not np.isfinite(value) or value < minimum:
            raise ValueError(f"{name} must be finite and >= {minimum}")
    for name, minimum in (("max_delay_steps", 0), ("max_nfev", 1)):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if (
        not args.seeds
        or any(isinstance(s, bool) or not isinstance(s, Integral) or s < 0 for s in args.seeds)
        or len(set(args.seeds)) != len(args.seeds)
    ):
        raise ValueError("seeds must be distinct non-negative integers")
    if args.scenario not in ("matched", "mismatch", "both"):
        raise ValueError("unknown scenario")
    if Path(args.output).exists():
        raise FileExistsError("output exists; choose a new directory")


def source_snapshot(output: Path) -> dict:
    payloads = {name: (ROOT / name).read_bytes() for name in SOURCE_FILES}
    source = {
        **capture_git_provenance(ROOT),
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
    }
    with tarfile.open(output / "source.tar.gz", "x:gz") as archive:
        for name, data in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    write_json(output / "source.json", source)
    return source


def save_paired_logs(destination: Path, args: argparse.Namespace, scenario: str, seed: int):
    """Both fitting arms receive identical q and u; both know the bench starts at rest."""
    generated = collect(
        BenchConfig(kind="wheel"),
        TRUTH,
        scenario=scenario,
        duration=args.calibration_duration,
        noise_scale=args.noise_scale,
        seed=seed,
    )
    for index, log in enumerate(generated):
        # Align the initial-state prior. Do not give just one arm noisy v[0].
        velocity = log.velocities.copy()
        velocity[0] = 0.0
        log = replace(log, velocities=velocity)
        name = f"{SPLITS[index]}_{index}"
        save_actuator_log(destination / f"{name}_position_velocity.csv", log)
        save_encoder_log(
            destination / f"{name}_position.csv",
            EncoderLog(
                log.config,
                log.commands,
                log.positions,
                known_initial_rest=True,
                name=log.name,
                split=log.split,
            ),
        )
    # The fitting phase receives fresh objects read through the public file boundary.
    position = [
        load_encoder_log(destination / f"{split}_{i}_position.csv")
        for i, split in enumerate(SPLITS)
    ]
    position_velocity = [
        load_actuator_log(destination / f"{split}_{i}_position_velocity.csv")
        for i, split in enumerate(SPLITS)
    ]
    return position, position_velocity


def fit_status(fit, max_delay_steps: int) -> dict:
    candidate = next(
        row
        for row in fit.candidates
        if row["parameters"]["delay_steps"] == fit.parameters.delay_steps
    )
    values = np.array([getattr(fit.parameters, name) for name in PARAMETER_NAMES])
    at_bound = np.isclose(values, PARAMETER_LOWER, atol=1e-6, rtol=0) | np.isclose(
        values, PARAMETER_UPPER, atol=1e-6, rtol=0
    )
    singular = fit.jacobian_singular_values
    return {
        "optimizer_success": candidate["optimizer_success"],
        "continuous_parameters_at_bounds": [n for n, hit in zip(PARAMETER_NAMES, at_bound) if hit],
        "delay_at_search_boundary": fit.parameters.delay_steps in (0, max_delay_steps),
        "normalized_jacobian_condition": (
            float(singular[0] / singular[-1]) if singular and singular[-1] > 0 else None
        ),
        "diagnostic_scope": "local continuous sensitivity at selected delay, not global identifiability",
    }


def plot_prediction(destination: Path, log: EncoderLog, predictions: dict) -> None:
    time = np.arange(len(log.positions)) * log.config.dt
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    axes[0].plot(time, log.positions, color="0.6", lw=2, label="measured position")
    for (name, values), color in zip(predictions.items(), COLORS[1:]):
        axes[0].plot(time, values, color=color, lw=1, label=name)
        axes[1].plot(time[1:], values[1:] - log.positions[1:], color=color, lw=1, label=name)
    axes[0].set(ylabel="Position [rad]", title=f"{destination.name}: unseen torque trajectory")
    axes[1].set(ylabel="Prediction error [rad]", xlabel="Time [s]")
    axes[0].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(destination / "holdout_prediction.png", dpi=130)
    plt.close(figure)


def plot_control(destination: Path, profiles: dict) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    for i, profile in enumerate(PROFILES):
        records = profiles[profile]
        first = records["pi"]
        axes[i, 0].plot(
            [r["time_s"] for r in first],
            [r["reference_velocity_rad_s"] for r in first],
            color="black",
            ls="--",
            lw=1,
            label="reference",
        )
        for name, color in zip(ARMS, COLORS):
            rows = records[name]
            time = [r["time_s"] for r in rows]
            axes[i, 0].plot(
                time, [r["actual_velocity_rad_s"] for r in rows], color=color, lw=1, label=name
            )
            axes[i, 1].plot(time, [r["tracking_error_rad_s"] for r in rows], color=color, lw=1)
        axes[i, 0].set(title=profile, ylabel="Speed [rad/s]")
        axes[i, 1].set(title=f"{profile} error", ylabel="Reference - speed [rad/s]")
    axes[0, 0].legend(fontsize=7)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.set_xlabel("Time [s]")
    figure.suptitle(f"{destination.name}: fixed-axis simulation; unchanged PI gains")
    figure.savefig(destination / "closed_loop.png", dpi=130)
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = source_snapshot(output)
    config = BenchConfig(kind="wheel")
    control_config = WheelControlConfig(dt=config.dt, torque_limit_nm=config.torque_limit_nm)
    scenarios = ("matched", "mismatch") if args.scenario == "both" else (args.scenario,)
    protocol = {
        "schema": "position-only-bench-comparison-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {**vars(args), "output": str(output)},
        "data_origin": "synthetic_only",
        "paper": "https://arxiv.org/html/2509.06342v2#S2.SS2",
        "paper_adaptation": "known torque replay, smooth friction, bounded least squares; no position PD/CMA-ES",
        "bench": asdict(config),
        "controller_all_arms": asdict(control_config),
        "generator_truth_not_available_to_fitter": asdict(TRUTH),
        "mismatch_extra_stribeck_friction_nm": 0.08,
        "parameter_bounds": dict(zip(PARAMETER_NAMES, zip(PARAMETER_LOWER, PARAMETER_UPPER))),
        "residual_scales": {
            "position_rad": POSITION_SCALE_RAD,
            "velocity_rad_s": VELOCITY_SCALE_RAD_S,
        },
        "initialization_both_fits": "first measured position, known zero velocity, zero pending commands",
        "position_noise_std_rad": args.noise_scale * 0.0002,
        "reference_velocity_noise_std_rad_s": args.noise_scale * 0.002,
        "fit_inputs": {
            "position": ["command", "position"],
            "position_velocity": ["command", "position", "synthetic measured velocity"],
        },
        "fit_selection": "two calibration trajectories only; all integer delays and optimizer statuses retained",
        "heldout_prediction": "whole validation chirp and holdout steps/reversals excluded from fitting",
        "closed_loop_feedback": "synthetic velocity measurement; not a position-only controller",
        "closed_loop_pairing": "same plant, initial state, PI gains, references and noise at each time",
        "delay_compensation": "none; estimated delay is not inverted by feedforward",
        "metric_timing": "current reference minus current true velocity; not next-state velocity",
        "expected_prediction_rows": len(scenarios) * len(args.seeds) * 4 * 3,
        "expected_control_cases": len(scenarios) * len(args.seeds) * len(PROFILES) * len(ARMS),
        "scenarios": list(scenarios),
        "profiles": list(PROFILES),
        "arms": list(ARMS),
        "versions": {name: version(name) for name in ("numpy", "scipy", "mujoco", "matplotlib")},
        "scope": "single known load and hidden parameter set with repeated noise; no population or D1 claim",
    }
    write_json(output / "protocol.json", protocol)
    start = perf_counter()
    predictions_metrics, control_metrics, fits = [], [], []
    for scenario in scenarios:
        for seed in args.seeds:
            destination = output / f"{scenario}_seed{seed}"
            destination.mkdir()
            position, position_velocity = save_paired_logs(destination, args, scenario, seed)
            chosen = {}
            for name, logs, fitter in (
                ("position", position, fit_encoder_parameters),
                ("position_velocity", position_velocity, fit_actuator_parameters),
            ):
                fit_start = perf_counter()
                fit = fitter(logs[:2], max_delay_steps=args.max_delay_steps, max_nfev=args.max_nfev)
                record = {
                    "scenario": scenario,
                    "seed": seed,
                    "observation": name,
                    **asdict(fit),
                    "fit_seconds": perf_counter() - fit_start,
                    "fit_status": fit_status(fit, args.max_delay_steps),
                }
                write_json(destination / f"fit_{name}.json", record)
                fits.append(record)
                chosen[name] = fit.parameters
                print(
                    f"{destination.name}/{name}: delay={fit.parameters.delay_steps}, "
                    f"seconds={record['fit_seconds']:.2f}, "
                    f"converged={record['fit_status']['optimizer_success']}",
                    flush=True,
                )
            for qlog, qvlog in zip(position, position_velocity):
                predictions = {
                    "nominal": predict_encoder_log(qlog, ActuatorParameters()),
                    "position": predict_encoder_log(qlog, chosen["position"]),
                    "position_velocity": predict_actuator_log(qvlog, chosen["position_velocity"])[
                        :, 0
                    ],
                }
                for name, values in predictions.items():
                    predictions_metrics.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "trajectory": qlog.name,
                            "split": qlog.split,
                            "model": name,
                            **encoder_prediction_metrics(qlog, values),
                        }
                    )
                rows = [
                    {
                        "time_s": i * config.dt,
                        "measured_position_rad": qlog.positions[i],
                        **{
                            f"{name}_position_rad": values[i]
                            for name, values in predictions.items()
                        },
                    }
                    for i in range(len(qlog.positions))
                ]
                write_csv_gz(destination / f"{qlog.name}_predictions.csv.gz", rows)
                if qlog.split == "holdout":
                    plot_prediction(destination, qlog, predictions)
            all_records = {}
            for profile in PROFILES:
                records = {}
                for name, parameters in zip(
                    ARMS,
                    (
                        None,
                        ActuatorParameters(),
                        chosen["position"],
                        chosen["position_velocity"],
                    ),
                ):
                    rows = run_closed_loop(
                        config,
                        TRUTH,
                        WheelVelocityController(control_config, parameters),
                        profile=profile,
                        duration=args.duration,
                        seed=seed,
                        noise_scale=args.noise_scale,
                        stribeck_friction_nm=0.08 if scenario == "mismatch" else 0.0,
                    )
                    write_csv_gz(destination / f"{profile}_{name}.csv.gz", rows)
                    metric = {
                        "scenario": scenario,
                        "seed": seed,
                        "profile": profile,
                        "controller": name,
                        **compute_metrics(rows, config.dt),
                    }
                    metric["completed"] = (
                        len(rows) == round(args.duration / config.dt)
                        and not metric["speed_guard_triggered"]
                    )
                    control_metrics.append(metric)
                    records[name] = rows
                all_records[profile] = records
            plot_control(destination, all_records)
    changed = [
        name
        for name, digest in source["sha256"].items()
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
    ]
    write_json(output / "source_consistency.json", {"unchanged": not changed, "changed": changed})
    if changed:
        raise RuntimeError("experiment source changed during execution; do not treat run as frozen")
    if (
        len(predictions_metrics) != protocol["expected_prediction_rows"]
        or len(control_metrics) != protocol["expected_control_cases"]
    ):
        raise RuntimeError("experiment matrix is incomplete")
    write_csv_gz(output / "prediction_metrics.csv.gz", predictions_metrics)
    write_csv_gz(output / "control_metrics.csv.gz", control_metrics)
    write_json(
        output / "summary.json",
        {
            "status": "completed"
            if all(r["completed"] for r in control_metrics)
            else "contains_incomplete_runs",
            "fits": fits,
            "prediction_metrics": predictions_metrics,
            "control_metrics": control_metrics,
            "elapsed_seconds": perf_counter() - start,
            "claims": protocol["scope"],
        },
    )
    write_json(
        output / "manifest.json",
        {
            "sha256": {
                str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.rglob("*"))
                if path.is_file()
            }
        },
    )
    print(f"Results: {output}", flush=True)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    parser.add_argument("--scenario", choices=("matched", "mismatch", "both"), default="both")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--calibration-duration", type=float, default=3.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--max-delay-steps", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    run(args)


if __name__ == "__main__":
    main()
