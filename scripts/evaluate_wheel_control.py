"""Paired single-wheel PI / nominal feedforward / identified feedforward experiment.

Synthetic fixed-axis plant only: not a rolling wheel, D1 controller or hardware
validation. Calibration torque logs and closed-loop reference motions are separate.
Every run uses a fresh plant and the same measurement noise sequence within a case.
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

from wheel_legged_control.actuator_bench import (
    WHEEL_INERTIA_KG_M2,
    ActuatorBench,
    ActuatorParameters,
    BenchConfig,
)
from wheel_legged_control.actuator_identification import (
    PARAMETER_LOWER,
    PARAMETER_NAMES,
    PARAMETER_UPPER,
    ActuatorLog,
    fit_actuator_parameters,
    load_actuator_log,
    save_actuator_log,
)
from wheel_legged_control.provenance import capture_git_provenance
from wheel_legged_control.wheel_control import WheelControlConfig, WheelVelocityController

PROFILES = ("tracking", "reversal", "stress")
SPEED_GUARD_RAD_S = 30.0
SOURCE_FILES = (
    "scripts/evaluate_wheel_control.py",
    "src/wheel_legged_control/wheel_control.py",
    "src/wheel_legged_control/actuator_bench.py",
    "src/wheel_legged_control/actuator_identification.py",
    "src/wheel_legged_control/provenance.py",
    "pyproject.toml",
)


def reference(profile: str, time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Current smooth command and its analytic derivative; no plant-state preview."""
    time_s = np.asarray(time_s, dtype=float)
    if time_s.ndim != 1 or not np.isfinite(time_s).all() or np.any(time_s < 0):
        raise ValueError("time_s must be a finite non-negative vector")
    if profile == "tracking":
        frequency, amplitude = 0.45, 3.0
    elif profile == "stress":
        frequency, amplitude = 1.0, 18.0
    elif profile == "reversal":
        phase = 2 * np.pi * 0.35 * time_s
        inner = 3.0 * np.sin(phase)
        velocity = 2.5 * np.tanh(inner)
        acceleration = 2.5 * 3.0 * 2 * np.pi * 0.35 * np.cos(phase) / np.cosh(inner) ** 2
        return velocity, acceleration
    else:
        raise ValueError(f"unknown profile: {profile}")
    phase = 2 * np.pi * frequency * time_s
    return amplitude * np.sin(phase), amplitude * 2 * np.pi * frequency * np.cos(phase)


def calibration_logs(
    config: BenchConfig,
    truth: ActuatorParameters,
    *,
    duration: float,
    seed: int,
    noise_scale: float,
    stribeck_friction_nm: float,
) -> list[ActuatorLog]:
    """Generator-only truth access; fitter will subsequently read only saved CSVs."""
    time_s = np.arange(round(duration / config.dt)) * config.dt
    commands = (
        0.40 * np.sin(2 * np.pi * (0.4 * time_s + 1.8 * time_s**2 / (2 * duration))),
        0.30 * np.sin(2 * np.pi * 0.7 * time_s + 1.2) + 0.18 * np.sin(2 * np.pi * 2.7 * time_s),
    )
    logs = []
    for index, command in enumerate(commands):
        plant = ActuatorBench(config, truth, stribeck_friction_nm=stribeck_friction_nm)
        plant.reset()
        states = np.vstack((plant.state, [plant.step(value) for value in command]))
        rng = np.random.default_rng(np.random.SeedSequence((seed, index)))
        states += rng.normal(size=states.shape) * noise_scale * np.array((0.0002, 0.002))
        logs.append(
            ActuatorLog(config, command, states[:, 0], states[:, 1], name=f"calibration_{index}")
        )
    return logs


def run_closed_loop(
    config: BenchConfig,
    truth: ActuatorParameters,
    controller: WheelVelocityController,
    *,
    profile: str,
    duration: float,
    seed: int,
    noise_scale: float,
    stribeck_friction_nm: float = 0.0,
) -> list[dict]:
    """One row per physical step: state_k -> command_k -> state_(k+1).

    Truth and applied torque stay in the evaluator, not in controller.compute().
    A speed guard ends unsafe synthetic runs; partial metrics are labelled later.
    """
    if config.kind != "wheel":
        raise ValueError("closed-loop experiment requires a wheel bench")
    if not np.isfinite(duration) or duration < config.dt * 2:
        raise ValueError("duration must contain at least two physical steps")
    if not np.isfinite(noise_scale) or noise_scale < 0 or seed < 0:
        raise ValueError("noise_scale and seed must be non-negative")
    if (
        controller.config.dt != config.dt
        or controller.config.torque_limit_nm != config.torque_limit_nm
    ):
        raise ValueError("controller and plant must share dt and torque limit")
    plant = ActuatorBench(config, truth, stribeck_friction_nm=stribeck_friction_nm)
    plant.reset()
    controller.reset()
    time_s = np.arange(round(duration / config.dt)) * config.dt
    target_velocity, target_acceleration = reference(profile, time_s)
    # The same draw at each physical time, independent of controller identity.
    rng = np.random.default_rng(np.random.SeedSequence((seed, 314)))
    noise = rng.normal(size=len(time_s)) * noise_scale * 0.002
    rows = []
    for index, time in enumerate(time_s):
        position, velocity = plant.state
        measured_velocity = velocity + noise[index]
        control = controller.compute(
            target_velocity[index], target_acceleration[index], measured_velocity
        )
        next_position, next_velocity = plant.step(control.command_torque_nm)
        if not np.isfinite((next_position, next_velocity)).all():
            raise FloatingPointError("non-finite plant state; run has no completed summary")
        speed_guard_triggered = bool(abs(next_velocity) > SPEED_GUARD_RAD_S)
        rows.append(
            {
                "time_s": time,
                "reference_velocity_rad_s": target_velocity[index],
                "reference_acceleration_rad_s2": target_acceleration[index],
                "position_rad": position,
                "actual_velocity_rad_s": velocity,
                "measured_velocity_rad_s": measured_velocity,
                "tracking_error_rad_s": target_velocity[index] - velocity,
                **asdict(control),
                "applied_torque_nm": plant.applied_torque_nm,
                "next_position_rad": next_position,
                "next_velocity_rad_s": next_velocity,
                "speed_guard_triggered": speed_guard_triggered,
            }
        )
        if speed_guard_triggered:
            break
    return rows


def compute_metrics(rows: list[dict], dt: float) -> dict:
    """Time-aligned reference_k minus true velocity_k, including startup samples."""
    if not rows or not np.isfinite(dt) or dt <= 0:
        raise ValueError("metrics require non-empty rows and finite positive dt")
    error = np.array([row["tracking_error_rad_s"] for row in rows])
    command = np.array([row["command_torque_nm"] for row in rows])
    applied = np.array([row["applied_torque_nm"] for row in rows])
    return {
        "steps": len(rows),
        "duration_s": len(rows) * dt,
        "velocity_rmse_rad_s": float(np.sqrt(np.mean(error**2))),
        "velocity_mae_rad_s": float(np.mean(np.abs(error))),
        "peak_velocity_error_rad_s": float(np.max(np.abs(error))),
        "integrated_absolute_error_rad": float(np.sum(np.abs(error)) * dt),
        "command_torque_rms_nm": float(np.sqrt(np.mean(command**2))),
        "applied_torque_rms_nm": float(np.sqrt(np.mean(applied**2))),
        "peak_requested_torque_nm": float(max(abs(row["requested_torque_nm"]) for row in rows)),
        "saturation_fraction": float(np.mean([row["saturated"] for row in rows])),
        "integration_frozen_fraction": float(np.mean([row["integration_frozen"] for row in rows])),
        "speed_guard_triggered": any(row["speed_guard_triggered"] for row in rows),
    }


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_profile(destination: Path, profile: str, records: dict[str, list[dict]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True)
    first = next(iter(records.values()))
    axes[0, 0].plot(
        [row["time_s"] for row in first],
        [row["reference_velocity_rad_s"] for row in first],
        color="0.2",
        ls="--",
        label="reference",
        lw=1.2,
    )
    for (name, rows), color in zip(records.items(), ("#777777", "#c65b32", "#24789a")):
        time_s = [row["time_s"] for row in rows]
        for axis, field in zip(
            axes.flat,
            (
                "actual_velocity_rad_s",
                "tracking_error_rad_s",
                "command_torque_nm",
                "integral_torque_nm",
            ),
        ):
            axis.plot(time_s, [row[field] for row in rows], color=color, lw=1.1, label=name)
    axes[0, 0].set(ylabel="Velocity [rad/s]", title="State and reference at the same time")
    axes[0, 1].set(ylabel="Reference - velocity [rad/s]", title="True-state tracking error")
    axes[1, 0].set(ylabel="Command torque [N m]", xlabel="Time [s]", title="Before actuator delay")
    axes[1, 1].set(
        ylabel="Integral torque [N m]", xlabel="Time [s]", title="Conditional integration"
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(f"{destination.name} / {profile}: synthetic fixed-axis wheel")
    figure.savefig(destination / f"{profile}.png", dpi=150)
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    bench_config = BenchConfig(kind="wheel")
    control_config = WheelControlConfig(
        dt=bench_config.dt, torque_limit_nm=bench_config.torque_limit_nm
    )
    truth = ActuatorParameters(0.017, 0.105, 0.043, 2)
    experiment = {
        "schema_version": 1,
        "data_origin": "synthetic_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {**vars(args), "output": str(output)},
        "source": capture_git_provenance(root),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES
        },
        "versions": {name: version(name) for name in ("numpy", "scipy", "mujoco", "matplotlib")},
        "bench_config": asdict(bench_config),
        "controller_config_all_conditions": asdict(control_config),
        "known_load_inertia_kg_m2": WHEEL_INERTIA_KG_M2,
        "nominal_parameters": asdict(ActuatorParameters()),
        "generator_truth_not_available_to_fitter_or_controller": asdict(truth),
        "mismatch_extra_stribeck_friction_nm": 0.08,
        "profiles": {
            "tracking": "3*sin(2*pi*0.45*t) rad/s",
            "reversal": "2.5*tanh(3*sin(2*pi*0.35*t)) rad/s",
            "stress": "18*sin(2*pi*1*t) rad/s; deliberately exceeds available torque",
        },
        "protocol": {
            "fit": "two calibration torque logs only; no closed-loop metrics in model selection",
            "selection": "fixed PI gains, three prespecified references; no tuning on closed-loop results",
            "pairing": "same truth, initial state, dt, torque limit, gains and measurement noise stream",
            "calibration_noise_std": {
                "position_rad": args.noise_scale * 0.0002,
                "velocity_rad_s": args.noise_scale * 0.002,
            },
            "feedback_noise_std_rad_s": args.noise_scale * 0.002,
            "sampling": "state_k / reference_k -> controller -> plant step -> next_state; log every dt",
            "metrics": "reference_k - true_velocity_k, not next_velocity; includes startup",
            "delay_compensation": "none; estimated delay is recorded but not inverted or previewed",
            "safety": {"speed_guard_rad_s": SPEED_GUARD_RAD_S, "partial_runs_reported": True},
            "scope": "one synthetic parameter set, one noise seed; no population CI or D1 claim",
        },
    }
    write_json(output / "experiment.json", experiment)
    start = perf_counter()
    metrics, cases = [], []
    scenarios = ("matched", "mismatch") if args.scenario == "both" else (args.scenario,)
    for scenario in scenarios:
        destination = output / scenario
        destination.mkdir()
        extra_friction = 0.08 if scenario == "mismatch" else 0.0
        print(f"Calibrating {scenario} ...", flush=True)
        for log in calibration_logs(
            bench_config,
            truth,
            duration=args.calibration_duration,
            seed=args.seed,
            noise_scale=args.noise_scale,
            stribeck_friction_nm=extra_friction,
        ):
            save_actuator_log(destination / f"{log.name}.csv", log)
        # Cross a file boundary; hidden generator parameters are not log fields.
        logs = [load_actuator_log(destination / f"calibration_{index}.csv") for index in range(2)]
        fit = fit_actuator_parameters(
            logs, max_delay_steps=args.max_delay_steps, max_nfev=args.max_nfev
        )
        selected = next(
            row
            for row in fit.candidates
            if row["parameters"]["delay_steps"] == fit.parameters.delay_steps
        )
        continuous = np.array([getattr(fit.parameters, name) for name in PARAMETER_NAMES])
        boundary = np.isclose(continuous, PARAMETER_LOWER, atol=1e-6) | np.isclose(
            continuous, PARAMETER_UPPER, atol=1e-6
        )
        fit_status = {
            "optimizer_success": selected["optimizer_success"],
            "continuous_parameters_at_bounds": [
                name for name, hit in zip(PARAMETER_NAMES, boundary) if hit
            ],
            "delay_at_search_boundary": fit.parameters.delay_steps in (0, args.max_delay_steps),
            "delay_compensated": False,
            "meaning": "local calibration fit status, not a guarantee of control improvement",
        }
        write_json(destination / "fit.json", {**asdict(fit), "fit_status": fit_status})
        if not fit_status["optimizer_success"]:
            print(
                "  Warning: selected fit exhausted its budget / did not converge; still reported.",
                flush=True,
            )
        cases.append(
            {"scenario": scenario, "parameters": asdict(fit.parameters), "fit_status": fit_status}
        )
        for profile in PROFILES:
            records = {}
            for name, parameters in (
                ("pi", None),
                ("nominal_ff", ActuatorParameters()),
                ("identified_ff", fit.parameters),
            ):
                controller = WheelVelocityController(
                    control_config, feedforward_parameters=parameters
                )
                rows = run_closed_loop(
                    bench_config,
                    truth,
                    controller,
                    profile=profile,
                    duration=args.duration,
                    seed=args.seed,
                    noise_scale=args.noise_scale,
                    stribeck_friction_nm=extra_friction,
                )
                records[name] = rows
                write_csv(destination / f"{profile}_{name}.csv", rows)
                metric = {
                    "scenario": scenario,
                    "profile": profile,
                    "controller": name,
                    **compute_metrics(rows, bench_config.dt),
                }
                metric["completed"] = (
                    len(rows) == round(args.duration / bench_config.dt)
                    and not metric["speed_guard_triggered"]
                )
                metrics.append(metric)
                print(
                    f"  {profile}/{name}: RMSE={metric['velocity_rmse_rad_s']:.5f}, saturation={metric['saturation_fraction']:.1%}, completed={metric['completed']}",
                    flush=True,
                )
            plot_profile(destination, profile, records)
    write_csv(output / "metrics.csv", metrics)
    write_json(
        output / "summary.json",
        {
            "status": "completed"
            if all(row["completed"] for row in metrics)
            else "contains_incomplete_runs",
            "cases": cases,
            "metrics": metrics,
            "elapsed_seconds": perf_counter() - start,
            "claims": "paired synthetic closed-loop comparison only; no whole-robot or hardware validation",
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
    print(f"Results: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new directory; never overwritten"
    )
    parser.add_argument("--scenario", choices=("matched", "mismatch", "both"), default="both")
    parser.add_argument(
        "--duration", type=float, default=6.0, help="seconds per closed-loop motion, >= 1"
    )
    parser.add_argument(
        "--calibration-duration", type=float, default=3.0, help="seconds per calibration, >= 0.5"
    )
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-delay-steps", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=30)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration < 1:
        parser.error("duration must be finite and >= 1 second")
    if not np.isfinite(args.calibration_duration) or args.calibration_duration < 0.5:
        parser.error("calibration-duration must be finite and >= 0.5 seconds")
    if not np.isfinite(args.noise_scale) or args.noise_scale < 0:
        parser.error("noise-scale must be finite and non-negative")
    if args.seed < 0 or args.max_delay_steps < 0 or args.max_nfev < 1:
        parser.error("seed/delay must be non-negative; max-nfev must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new experiment directory")
    run(args)


if __name__ == "__main__":
    main()
