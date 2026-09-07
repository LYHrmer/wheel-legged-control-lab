"""CPU-only synthetic identification: collect, fit, and test unseen commands.

No controller/policy is trained or changed by this experiment. Every output
directory is exclusive. See docs/actuator_identification.md for the timing and
the distinction between parameter recovery and model-mismatch prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wheel_legged_control.actuator_bench import ActuatorBench, ActuatorParameters, BenchConfig
from wheel_legged_control.actuator_identification import (
    PARAMETER_LOWER,
    PARAMETER_NAMES,
    PARAMETER_UPPER,
    POSITION_SCALE_RAD,
    VELOCITY_SCALE_RAD_S,
    ActuatorLog,
    fit_actuator_parameters,
    load_actuator_log,
    predict_actuator_log,
    prediction_metrics,
    save_actuator_log,
)
from wheel_legged_control.provenance import capture_git_provenance


def excitation(config: BenchConfig, duration: float, index: int) -> np.ndarray:
    """Whole-trajectory split, not random rows from the same motion."""
    time = np.arange(round(duration / config.dt)) * config.dt
    if index == 0:
        return 0.40 * np.sin(2 * np.pi * (0.4 * time + 1.8 * time**2 / (2 * duration)))
    if index == 1:
        return 0.30 * np.sin(2 * np.pi * 0.7 * time + 1.2) + 0.18 * np.sin(2 * np.pi * 2.7 * time)
    if index == 2:
        return 0.45 * np.sin(2 * np.pi * (0.9 * time + 3.0 * time**2 / (2 * duration)) + 0.7)
    levels = np.asarray((0.0, 0.50, -0.32, 0.12, -0.55, 0.36, 0.0))
    return levels[np.minimum((time / duration * len(levels)).astype(int), len(levels) - 1)]


def collect(
    config: BenchConfig,
    truth: ActuatorParameters,
    *,
    scenario: str,
    duration: float,
    noise_scale: float,
    seed: int,
) -> list[ActuatorLog]:
    """Generator-only access to hidden parameters; these never enter log metadata."""
    logs = []
    for index, split in enumerate(("calibration", "calibration", "validation", "holdout")):
        commands = excitation(config, duration, index)
        bench = ActuatorBench(
            config,
            truth,
            stribeck_friction_nm=0.08 if scenario == "mismatch" else 0.0,
        )
        bench.reset()
        states = np.vstack((bench.state, [bench.step(command) for command in commands]))
        rng = np.random.default_rng(np.random.SeedSequence((seed, index)))
        states += rng.normal(size=states.shape) * (noise_scale * np.asarray((0.0002, 0.002)))
        logs.append(
            ActuatorLog(
                config,
                commands,
                states[:, 0],
                states[:, 1],
                name=f"{split}_{index}",
                split=split,
            )
        )
    return logs


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_case(output: Path, log: ActuatorLog, predictions: dict, candidates: tuple) -> None:
    time = np.arange(len(log.positions)) * log.config.dt
    figure, axes = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True)
    axes[0, 0].step(time[:-1], log.commands, where="post", color="0.25")
    axes[0, 0].set(ylabel="Command torque [N m]", title="Held-out steps and reversals")
    axes[0, 1].plot(time, log.velocities, color="0.65", lw=2, label="Synthetic measurement")
    for model, color in (("nominal", "#c65b32"), ("fitted", "#24789a")):
        prediction = predictions[model]
        axes[0, 1].plot(time, prediction[:, 1], color=color, lw=1.1, label=model)
        axes[1, 0].plot(time[1:], prediction[1:, 1] - log.velocities[1:], color=color, label=model)
    axes[0, 1].set(ylabel="Velocity [rad/s]", title="Free-running prediction")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set(ylabel="Velocity error [rad/s]", xlabel="Time [s]")
    axes[1, 0].axhline(0, color="0.5", lw=0.5)
    axes[1, 1].plot(
        [row["parameters"]["delay_steps"] * log.config.dt * 1000 for row in candidates],
        [row["calibration_loss"] for row in candidates],
        "o-",
        color="#24789a",
    )
    axes[1, 1].set(
        xlabel="Candidate command delay [ms]",
        ylabel="Calibration loss",
        title="Only calibration logs select the model",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle(f"{output.name}: synthetic bench, not D1 hardware identification")
    figure.savefig(output / "prediction.png", dpi=150)
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_files = (
        "scripts/run_actuator_identification.py",
        "src/wheel_legged_control/actuator_identification.py",
        "src/wheel_legged_control/actuator_bench.py",
        "pyproject.toml",
    )
    truth = ActuatorParameters(armature=0.017, damping=0.105, coulomb_friction=0.043, delay_steps=2)
    config = {
        "schema_version": 1,
        "data_origin": "synthetic_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": capture_git_provenance(root),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files
        },
        "versions": {name: version(name) for name in ("numpy", "scipy", "mujoco")},
        "arguments": {**vars(args), "output": str(output)},
        "nominal_parameters": asdict(ActuatorParameters()),
        "generator_truth_not_available_to_fitter": asdict(truth),
        "mismatch_stribeck_friction_nm": 0.08,
        "position_noise_std_rad": args.noise_scale * 0.0002,
        "velocity_noise_std_rad_s": args.noise_scale * 0.002,
        "parameter_bounds": dict(zip(PARAMETER_NAMES, zip(PARAMETER_LOWER, PARAMETER_UPPER))),
        "loss_scales": {"position_rad": POSITION_SCALE_RAD, "velocity_rad_s": VELOCITY_SCALE_RAD_S},
        "protocol": {
            "selection": "two calibration motions only; fixed budget and residual scales",
            "validation": "different chirp; reported, not used to select parameters",
            "holdout": "steps and reversals; never used to select parameters",
            "prediction": "initial measured state, zero command history, then free-running",
            "scope": "offline prediction only; no IDQP/RL benefit or hardware claim",
        },
    }
    write_json(output / "experiment.json", config)
    all_metrics, cases = [], []
    benches = ("wheel", "pendulum") if args.bench == "both" else (args.bench,)
    scenarios = ("matched", "mismatch") if args.scenario == "both" else (args.scenario,)
    start = perf_counter()
    for kind in benches:
        for scenario in scenarios:
            case = f"{kind}_{scenario}"
            destination = output / case
            destination.mkdir()
            print(f"Collecting and fitting {case} ...", flush=True)
            for log in collect(
                BenchConfig(kind=kind),
                truth,
                scenario=scenario,
                duration=args.duration,
                noise_scale=args.noise_scale,
                seed=args.seed,
            ):
                save_actuator_log(destination / f"{log.name}.csv", log)
            # Deliberately cross the file interface. No plant instance or truth
            # object is passed to the fitter, only the two calibration CSVs.
            logs = [
                load_actuator_log(destination / f"{split}_{index}.csv")
                for index, split in enumerate(
                    ("calibration", "calibration", "validation", "holdout")
                )
            ]
            fit_start = perf_counter()
            fit = fit_actuator_parameters(
                logs[:2],
                max_delay_steps=args.max_delay_steps,
                max_nfev=args.max_nfev,
            )
            fit_seconds = perf_counter() - fit_start
            write_json(destination / "fit.json", {**asdict(fit), "fit_seconds": fit_seconds})
            for log in logs:
                predictions = {
                    "nominal": predict_actuator_log(log, ActuatorParameters()),
                    "fitted": predict_actuator_log(log, fit.parameters),
                }
                low_speed = np.abs(log.velocities[1:]) < 0.3
                for model, prediction in predictions.items():
                    errors = prediction[1:, 1] - log.velocities[1:]
                    all_metrics.append(
                        {
                            "case": case,
                            "split": log.split,
                            "trajectory": log.name,
                            "model": model,
                            **prediction_metrics(log, prediction),
                            "low_speed_velocity_rmse_rad_s": (
                                float(np.sqrt(np.mean(errors[low_speed] ** 2)))
                                if low_speed.any()
                                else None
                            ),
                            "low_speed_sample_count": int(low_speed.sum()),
                        }
                    )
                rows = [
                    {
                        "time_s": index * log.config.dt,
                        "measured_position_rad": log.positions[index],
                        "measured_velocity_rad_s": log.velocities[index],
                        "nominal_position_rad": predictions["nominal"][index, 0],
                        "nominal_velocity_rad_s": predictions["nominal"][index, 1],
                        "fitted_position_rad": predictions["fitted"][index, 0],
                        "fitted_velocity_rad_s": predictions["fitted"][index, 1],
                    }
                    for index in range(len(log.positions))
                ]
                write_csv(destination / f"{log.name}_predictions.csv", rows)
                if log.split == "holdout":
                    plot_case(destination, log, predictions, fit.candidates)
            cases.append(
                {
                    "case": case,
                    "parameters": asdict(fit.parameters),
                    "fit_seconds": fit_seconds,
                    "jacobian_rank": fit.jacobian_rank,
                }
            )
            print(
                f"  fitted delay={fit.parameters.delay_steps} steps, {fit_seconds:.2f} s",
                flush=True,
            )
    write_csv(output / "metrics.csv", all_metrics)
    summary = {
        "status": "completed",
        "cases": cases,
        "metrics": all_metrics,
        "elapsed_seconds": perf_counter() - start,
        "claims": "synthetic held-out prediction, not closed-loop or real-robot validation",
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "manifest.json",
        {
            "sha256": {
                str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )
    print(f"Results: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new directory; never overwritten"
    )
    parser.add_argument("--bench", choices=("wheel", "pendulum", "both"), default="both")
    parser.add_argument("--scenario", choices=("matched", "mismatch", "both"), default="both")
    parser.add_argument(
        "--duration", type=float, default=3.0, help="seconds per trajectory, >= 0.5"
    )
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-delay-steps", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=30)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration < 0.5:
        parser.error("duration must be finite and >= 0.5 seconds")
    if not np.isfinite(args.noise_scale) or args.noise_scale < 0:
        parser.error("noise-scale must be finite and non-negative")
    if args.seed < 0 or args.max_delay_steps < 0 or args.max_nfev < 1:
        parser.error("seed/delay must be non-negative; max-nfev must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new experiment directory")
    run(args)


if __name__ == "__main__":
    main()
