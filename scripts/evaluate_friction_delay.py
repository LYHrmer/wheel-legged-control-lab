"""Frozen friction/delay/excitation grid for the encoder-only (position-only) fitter.

Hidden-plant low-speed Stribeck friction and true command delay are varied while the
existing ``fit_encoder_parameters`` stays untouched, to observe whether unmodelled
friction is absorbed as an apparent pure delay. Identification is position-only;
closed-loop control keeps its own independent synthetic velocity sensor. This is
PACE-inspired direct-known-torque replay on a synthetic single-axis bench, not the
original PD/Isaac/CMA-ES pipeline and not D1 hardware identification.
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
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter

import numpy as np

# The script can be invoked directly, without pytest's repository-path injection.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_wheel_control import (
    PROFILES,
    compute_metrics,
    run_closed_loop,
)
from scripts.run_actuator_identification import excitation as standard_excitation
from wheel_legged_control.actuator_bench import ActuatorBench, ActuatorParameters, BenchConfig
from wheel_legged_control.actuator_identification import (
    PARAMETER_LOWER,
    PARAMETER_NAMES,
    PARAMETER_UPPER,
    POSITION_SCALE_RAD,
)
from wheel_legged_control.encoder_identification import (
    EncoderLog,
    fit_encoder_parameters,
    load_encoder_log,
    predict_encoder_log,
    save_encoder_log,
)
from wheel_legged_control.provenance import capture_git_provenance
from wheel_legged_control.wheel_control import WheelControlConfig, WheelVelocityController

SCHEMA_VERSION = 1
SPLITS = ("calibration", "calibration", "validation", "holdout")
FAMILIES = ("standard", "low_speed")
DEFAULT_FRICTIONS = (0.0, 0.04, 0.08)
DEFAULT_ACTUAL_DELAYS = (0, 2, 4)
DEFAULT_SEEDS = (53, 67, 79)
POSITION_NOISE_SD_RAD = 0.0002
HOLDOUT_LEVELS = (0.0, 0.22, -0.48, 0.07, 0.38, -0.16, 0.0)
CONTINUOUS_TRUTH = {"armature": 0.017, "damping": 0.105, "coulomb_friction": 0.043}
MIN_CALIBRATION_DURATION_S = 0.5
MIN_CONTROL_DURATION_S = 1.0
SOURCE_FILES = (
    "scripts/evaluate_friction_delay.py",
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


def excitation(config: BenchConfig, duration: float, index: int, family: str) -> np.ndarray:
    """Command sequence for one trajectory; only calibration depends on ``family``.

    Validation and holdout are deliberately family-independent so the two families
    are compared on byte-identical held-out commands for a given plant and seed.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown calibration family: {family}")
    time = np.arange(round(duration / config.dt)) * config.dt
    if index in (0, 1):
        if family == "standard":
            return standard_excitation(config, duration, index)
        if index == 0:
            return 0.12 * np.sin(2 * np.pi * 0.35 * time)
        return 0.10 * np.sin(2 * np.pi * 0.55 * time + 0.7) + 0.025 * np.sin(2 * np.pi * 1.8 * time)
    if index == 2:
        return 0.35 * np.sin(2 * np.pi * (0.6 * time + 2.2 * time**2 / (2 * duration)) + 0.2)
    if index == 3:
        levels = np.asarray(HOLDOUT_LEVELS)
        return levels[np.minimum((time / duration * len(levels)).astype(int), len(levels) - 1)]
    raise ValueError(f"unknown trajectory index: {index}")


def acquire_logs(
    config: BenchConfig,
    truth: ActuatorParameters,
    *,
    family: str,
    duration: float,
    seed: int,
    stribeck_friction_nm: float,
    noise_scale: float = 1.0,
) -> list[EncoderLog]:
    """Real ``ActuatorBench`` rollouts, position channel only.

    Truth, plant object, velocity and applied torque are visible to this generator
    and to nothing downstream: only the saved position CSVs reach the fitter.
    """
    logs = []
    for index, split in enumerate(SPLITS):
        commands = excitation(config, duration, index, family)
        bench = ActuatorBench(config, truth, stribeck_friction_nm=stribeck_friction_nm)
        bench.reset()
        positions = np.concatenate(
            ([bench.state[0]], [bench.step(command)[0] for command in commands])
        )
        # No calibration-family term: held-out noise is paired across families.
        rng = np.random.default_rng(np.random.SeedSequence((seed, index)))
        positions = positions + rng.normal(size=positions.shape) * noise_scale * POSITION_NOISE_SD_RAD
        logs.append(
            EncoderLog(
                config,
                commands,
                positions,
                known_initial_rest=True,
                name=f"{split}_{index}",
                split=split,
            )
        )
    return logs


def calibration_loss(logs, predictions) -> float:
    """Mean over both calibration trajectories of the scaled squared position error."""
    parts = [
        ((prediction[1:] - log.positions[1:]) / POSITION_SCALE_RAD) ** 2
        for log, prediction in zip(logs, predictions)
    ]
    return float(np.mean(np.concatenate(parts)))


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    """Exclusive, deterministic gzip (zero mtime) so identical runs hash identically."""
    with path.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        stream = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        stream.detach()


def read_csv_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _positive_grid(values, name: str, *, integer: bool) -> tuple:
    values = list(values)
    if not values:
        raise ValueError(f"{name} must be a nonempty grid")
    checked = []
    for value in values:
        kind = Integral if integer else Real
        if isinstance(value, bool) or not isinstance(value, kind) or not np.isfinite(value):
            raise ValueError(
                f"{name} must contain finite non-boolean "
                f"{'integer' if integer else 'real'} values"
            )
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        checked.append(int(value) if integer else float(value))
    if len(set(checked)) != len(checked):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(checked)


def _validated(args: argparse.Namespace) -> argparse.Namespace:
    """Full programmatic validation; ``main`` reuses it before the directory exists."""
    frictions = _positive_grid(args.frictions, "frictions", integer=False)
    actual_delays = _positive_grid(args.actual_delays, "actual-delays", integer=True)
    seeds = _positive_grid(args.seeds, "seeds", integer=True)
    families = tuple(args.families)
    if not families or len(set(families)) != len(families):
        raise ValueError("families must be a nonempty grid without duplicates")
    if any(family not in FAMILIES for family in families):
        raise ValueError(f"families must be drawn from {FAMILIES}")
    for name, minimum in (
        ("calibration_duration", MIN_CALIBRATION_DURATION_S),
        ("duration", MIN_CONTROL_DURATION_S),
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite real scalar")
        if float(value) < minimum:
            raise ValueError(f"{name} must be >= {minimum} seconds")
    for name, minimum in (("max_delay_steps", 0), ("max_nfev", 1)):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    noise_scale = args.noise_scale
    if (
        isinstance(noise_scale, bool)
        or not isinstance(noise_scale, Real)
        or not np.isfinite(noise_scale)
        or noise_scale < 0
    ):
        raise ValueError("noise_scale must be a finite non-negative real scalar")
    return argparse.Namespace(
        output=Path(args.output),
        frictions=frictions,
        actual_delays=actual_delays,
        families=families,
        seeds=seeds,
        calibration_duration=float(args.calibration_duration),
        duration=float(args.duration),
        max_delay_steps=int(args.max_delay_steps),
        max_nfev=int(args.max_nfev),
        noise_scale=float(noise_scale),
    )


def _archive_sources(output: Path) -> dict[str, str]:
    payloads = {name: (ROOT / name).read_bytes() for name in SOURCE_FILES}
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    write_json(
        output / "source.json",
        {
            "root_relative_files": list(SOURCE_FILES),
            "sha256": digests,
            "git": capture_git_provenance(ROOT),
            "note": "dependency bytes actually imported by this run, archived in source.tar.gz",
        },
    )
    with tarfile.open(output / "source.tar.gz", "x:gz") as archive:
        for name, data in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    return digests


def _fit_records(fit, args, actual_delay: int) -> dict:
    continuous = np.array([getattr(fit.parameters, name) for name in PARAMETER_NAMES])
    at_bounds = np.isclose(continuous, PARAMETER_LOWER, atol=1e-6) | np.isclose(
        continuous, PARAMETER_UPPER, atol=1e-6
    )
    selected = next(
        row
        for row in fit.candidates
        if row["parameters"]["delay_steps"] == fit.parameters.delay_steps
    )
    return {
        "parameters": asdict(fit.parameters),
        "candidates": list(fit.candidates),
        "jacobian_singular_values": list(fit.jacobian_singular_values),
        "jacobian_rank": fit.jacobian_rank,
        "selected_optimizer_success": selected["optimizer_success"],
        "selected_nfev": selected["nfev"],
        "selected_message": selected["message"],
        "selected_calibration_loss": selected["calibration_loss"],
        "continuous_parameters_at_bounds": [
            name for name, hit in zip(PARAMETER_NAMES, at_bounds) if hit
        ],
        "delay_at_search_boundary": fit.parameters.delay_steps in (0, args.max_delay_steps),
        "actual_delay_steps": actual_delay,
        "actual_delay_out_of_search_range": actual_delay > args.max_delay_steps,
        "delay_error_steps": fit.parameters.delay_steps - actual_delay,
        "selection_rule": "minimum calibration loss only; holdout never used to select",
        "delay_compensated": False,
    }


def run(args: argparse.Namespace) -> Path:
    args = _validated(args)
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output already exists; choose a new experiment directory")
    config = BenchConfig(kind="wheel", dt=0.002, torque_limit_nm=2.0)
    control_config = WheelControlConfig(dt=config.dt, torque_limit_nm=config.torque_limit_nm)
    output.mkdir(parents=True, exist_ok=False)
    cells = [
        (friction, delay, seed, family)
        for friction in args.frictions
        for delay in args.actual_delays
        for seed in args.seeds
        for family in args.families
    ]
    write_json(
        output / "protocol.json",
        {
            "schema_version": SCHEMA_VERSION,
            "data_origin": "synthetic_only",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": {**vars(args), "output": str(output), "families": list(args.families)},
            "versions": {name: version(name) for name in ("numpy", "scipy", "mujoco")},
            "bench_config": asdict(config),
            "controller_config_all_conditions": asdict(control_config),
            "nominal_parameters": asdict(ActuatorParameters()),
            "continuous_truth_not_available_to_fitter_or_controller": CONTINUOUS_TRUTH,
            "grids": {
                "extra_hidden_stribeck_friction_nm": list(args.frictions),
                "hidden_actual_delay_steps": list(args.actual_delays),
                "calibration_families": list(args.families),
                "noise_seeds": list(args.seeds),
            },
            "position_noise_std_rad": args.noise_scale * POSITION_NOISE_SD_RAD,
            "expected_counts": {
                "fits": len(cells),
                "candidates": len(cells) * (args.max_delay_steps + 1),
                "prediction_metric_rows": len(cells) * len(SPLITS) * 2,
                "control_cases": len(cells) * len(PROFILES) * 2,
            },
            "protocol": {
                "identification": "position-only encoder logs; no velocity or applied torque",
                "selection": "two calibration logs only, minimum calibration loss",
                "held_out_pairing": "validation/holdout commands and noise identical across families",
                "seeds": "repeated measurement noise for one object, not independent robots",
                "control": "fitted parameters change controller feedforward only, never the plant",
                "delay_compensation": "none; no pure-delay inversion",
                "nominal_rows": "duplicated across families as paired references, not new samples",
            },
        },
    )
    source_before = _archive_sources(output)
    start = perf_counter()
    prediction_metrics, control_metrics, candidate_rows, fit_rows = [], [], [], []
    for friction, actual_delay, seed, family in cells:
        # repr preserves distinct finite floats; six-digit ':g' can collide.
        friction_name = repr(friction) if friction else "0"
        name = f"friction{friction_name}_delay{actual_delay}_seed{seed}_{family}"
        destination = output / name
        destination.mkdir()
        truth = ActuatorParameters(**CONTINUOUS_TRUTH, delay_steps=actual_delay)
        for log in acquire_logs(
            config,
            truth,
            family=family,
            duration=args.calibration_duration,
            seed=seed,
            stribeck_friction_nm=friction,
            noise_scale=args.noise_scale,
        ):
            save_encoder_log(destination / f"{log.name}.csv", log)
        # Cross the file boundary: only the two calibration CSVs reach the fitter.
        logs = [
            load_encoder_log(destination / f"{split}_{index}.csv")
            for index, split in enumerate(SPLITS)
        ]
        calibration = [log for log in logs if log.split == "calibration"]
        fit = fit_encoder_parameters(
            calibration, max_delay_steps=args.max_delay_steps, max_nfev=args.max_nfev
        )
        record = _fit_records(fit, args, actual_delay)
        write_json(destination / "fit.json", record)
        fit_rows.append(
            {
                "cell": name,
                "stribeck_friction_nm": friction,
                "actual_delay_steps": actual_delay,
                "seed": seed,
                "family": family,
                "fitted_delay_steps": fit.parameters.delay_steps,
                "delay_error_steps": record["delay_error_steps"],
                "actual_delay_out_of_search_range": record["actual_delay_out_of_search_range"],
                "delay_at_search_boundary": record["delay_at_search_boundary"],
                "selected_optimizer_success": record["selected_optimizer_success"],
                "selected_calibration_loss": record["selected_calibration_loss"],
                "jacobian_rank": fit.jacobian_rank,
                **{f"fitted_{key}": value for key, value in asdict(fit.parameters).items()},
            }
        )
        for candidate in fit.candidates:
            parameters = ActuatorParameters(**candidate["parameters"])
            predictions = [predict_encoder_log(log, parameters) for log in calibration]
            delay = candidate["parameters"]["delay_steps"]
            write_csv_gz(
                destination / f"candidate_delay{delay}_calibration_predictions.csv.gz",
                [
                    {
                        "trajectory": log.name,
                        "time_s": index * config.dt,
                        "measured_position_rad": log.positions[index],
                        "predicted_position_rad": prediction[index],
                    }
                    for log, prediction in zip(calibration, predictions)
                    for index in range(len(log.positions))
                ],
            )
            candidate_rows.append(
                {
                    "cell": name,
                    "stribeck_friction_nm": friction,
                    "actual_delay_steps": actual_delay,
                    "seed": seed,
                    "family": family,
                    "candidate_delay_steps": delay,
                    "selected": delay == fit.parameters.delay_steps,
                    "reported_calibration_loss": candidate["calibration_loss"],
                    "recomputed_calibration_loss": calibration_loss(calibration, predictions),
                    "optimizer_success": candidate["optimizer_success"],
                    "nfev": candidate["nfev"],
                    "message": candidate["message"],
                    **{f"candidate_{key}": value for key, value in candidate["parameters"].items()},
                }
            )
        for log in logs:
            predictions = {
                "nominal": predict_encoder_log(log, ActuatorParameters()),
                "selected": predict_encoder_log(log, fit.parameters),
            }
            write_csv_gz(
                destination / f"{log.name}_predictions.csv.gz",
                [
                    {
                        "time_s": index * config.dt,
                        "measured_position_rad": log.positions[index],
                        "nominal_position_rad": predictions["nominal"][index],
                        "selected_position_rad": predictions["selected"][index],
                    }
                    for index in range(len(log.positions))
                ],
            )
            for model, prediction in predictions.items():
                error = prediction[1:] - log.positions[1:]
                prediction_metrics.append(
                    {
                        "cell": name,
                        "stribeck_friction_nm": friction,
                        "actual_delay_steps": actual_delay,
                        "seed": seed,
                        "family": family,
                        "split": log.split,
                        "trajectory": log.name,
                        "model": model,
                        "samples": len(error),
                        "position_rmse_rad": float(np.sqrt(np.mean(error**2))),
                        "position_mae_rad": float(np.mean(np.abs(error))),
                        "peak_position_error_rad": float(np.max(np.abs(error))),
                    }
                )
        for profile in PROFILES:
            for model, parameters in (
                ("nominal_ff", ActuatorParameters()),
                ("fitted_ff", fit.parameters),
            ):
                controller = WheelVelocityController(
                    control_config, feedforward_parameters=parameters
                )
                rows = run_closed_loop(
                    config,
                    truth,
                    controller,
                    profile=profile,
                    duration=args.duration,
                    seed=seed,
                    noise_scale=args.noise_scale,
                    stribeck_friction_nm=friction,
                )
                write_csv_gz(destination / f"{profile}_{model}_rows.csv.gz", rows)
                metric = {
                    "cell": name,
                    "stribeck_friction_nm": friction,
                    "actual_delay_steps": actual_delay,
                    "seed": seed,
                    "family": family,
                    "profile": profile,
                    "controller": model,
                    **compute_metrics(rows, config.dt),
                }
                metric["completed"] = (
                    len(rows) == round(args.duration / config.dt)
                    and not metric["speed_guard_triggered"]
                )
                control_metrics.append(metric)
        print(
            f"{name}: fitted delay={fit.parameters.delay_steps} (actual {actual_delay}), "
            f"converged={record['selected_optimizer_success']}",
            flush=True,
        )
    write_csv_gz(output / "prediction_metrics.csv.gz", prediction_metrics)
    write_csv_gz(output / "control_metrics.csv.gz", control_metrics)
    write_csv_gz(output / "candidates.csv.gz", candidate_rows)
    write_csv_gz(output / "fits.csv.gz", fit_rows)
    source_after = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES
    }
    unchanged = source_before == source_after
    write_json(output / "source_consistency.json", {
        "unchanged": unchanged,
        "sha256": source_after,
        "pre_run_sha256": source_before,
    })
    if not unchanged:
        raise RuntimeError("experiment source changed during execution; no completed summary")
    write_json(
        output / "summary.json",
        {
            "status": "completed"
            if all(row["completed"] for row in control_metrics)
            else "contains_incomplete_runs",
            "observed_counts": {
                "cells": len(cells),
                "fits": len(fit_rows),
                "candidates": len(candidate_rows),
                "prediction_metric_rows": len(prediction_metrics),
                "control_cases": len(control_metrics),
            },
            "observed_labels": {
                "families": sorted({row["family"] for row in fit_rows}),
                "splits": sorted({row["split"] for row in prediction_metrics}),
                "profiles": sorted({row["profile"] for row in control_metrics}),
                "controllers": sorted({row["controller"] for row in control_metrics}),
                "prediction_models": sorted({row["model"] for row in prediction_metrics}),
                "fitted_delay_steps": sorted({row["fitted_delay_steps"] for row in fit_rows}),
            },
            "observed_status_counts": {
                "converged_selected_fits": sum(
                    1 for row in fit_rows if row["selected_optimizer_success"]
                ),
                "nonconverged_selected_fits": sum(
                    1 for row in fit_rows if not row["selected_optimizer_success"]
                ),
                "delay_at_search_boundary": sum(
                    1 for row in fit_rows if row["delay_at_search_boundary"]
                ),
                "actual_delay_out_of_search_range": sum(
                    1 for row in fit_rows if row["actual_delay_out_of_search_range"]
                ),
                "incomplete_control_runs": sum(
                    1 for row in control_metrics if not row["completed"]
                ),
                "speed_guard_triggered": sum(
                    1 for row in control_metrics if row["speed_guard_triggered"]
                ),
            },
            "elapsed_seconds": perf_counter() - start,
            "limitations": [
                "synthetic single-axis bench only; no D1 hardware or whole-robot claim",
                "seeds repeat measurement noise for one object, not independent robots",
                "nominal_ff rows are duplicated across families as paired references",
                "counts above are observed records, not inferred from optimizer convergence",
                "identification is position-only; control feedback uses an independent sensor",
                "no delay compensation or pure-delay inversion is performed anywhere",
            ],
        },
    )
    write_json(
        output / "manifest.json",
        {
            "sha256": {
                str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "manifest.json"
            }
        },
    )
    print(f"Results: {output}")
    return output


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory, never reused")
    parser.add_argument("--frictions", type=float, nargs="+", default=list(DEFAULT_FRICTIONS))
    parser.add_argument("--actual-delays", type=int, nargs="+", default=list(DEFAULT_ACTUAL_DELAYS))
    parser.add_argument("--families", nargs="+", default=list(FAMILIES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--calibration-duration", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--max-delay-steps", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
