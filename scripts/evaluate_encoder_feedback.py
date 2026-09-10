"""Causal encoder velocity feedback on a fixed-axis synthetic wheel.

The position-only calibration is shared by all feedback arms. Only the evaluator
can see true velocity; encoder observers receive one delivered position and its
sampling timestamp. These model-free observers do not compensate transport delay.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import tarfile
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_encoder_identification import (
    SOURCE_FILES as IDENTIFICATION_SOURCES,
)
from scripts.evaluate_encoder_identification import (
    fit_status,
    write_csv_gz,
    write_json,
)
from scripts.evaluate_wheel_control import PROFILES, SPEED_GUARD_RAD_S, compute_metrics, reference
from scripts.run_actuator_identification import collect
from wheel_legged_control.actuator_bench import ActuatorBench, ActuatorParameters, BenchConfig
from wheel_legged_control.encoder_identification import (
    EncoderLog,
    fit_encoder_parameters,
    load_encoder_log,
    save_encoder_log,
)
from wheel_legged_control.encoder_velocity import EncoderVelocityConfig, EncoderVelocityObserver
from wheel_legged_control.provenance import capture_git_provenance
from wheel_legged_control.wheel_control import WheelControlConfig, WheelVelocityController

SOURCE_FILES = (
    *IDENTIFICATION_SOURCES,
    "scripts/evaluate_encoder_feedback.py",
    "src/wheel_legged_control/encoder_velocity.py",
)
FEEDBACKS = ("synthetic_velocity", "difference_lowpass", "alpha_beta")
CONDITIONS = {
    "noisy": {"position_noise_std_rad": 0.0002, "quantization_rad": 0.0, "delay_steps": 0},
    "quantized": {
        "position_noise_std_rad": 0.0002,
        "quantization_rad": 2 * np.pi / 4096,
        "delay_steps": 0,
    },
    "delayed": {"position_noise_std_rad": 0.0002, "quantization_rad": 0.0, "delay_steps": 5},
}
TRUTH = ActuatorParameters(0.017, 0.105, 0.043, 2)
CALIBRATION_SEED = 89


def _positive_duration(value, minimum, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < minimum
    ):
        raise ValueError(f"{name} must be finite and >= {minimum}")


def validate_args(args):
    _positive_duration(args.duration, 1.0, "duration")
    _positive_duration(args.calibration_duration, 0.5, "calibration_duration")
    for name, minimum in (("max_delay_steps", 0), ("max_nfev", 1)):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if (
        not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or any(isinstance(s, bool) or not isinstance(s, Integral) or s < 0 for s in args.seeds)
    ):
        raise ValueError("seeds must be distinct non-negative integers")
    for name, available in (("conditions", CONDITIONS), ("scenarios", ("matched", "mismatch"))):
        selected = getattr(args, name)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(s not in available for s in selected)
        ):
            raise ValueError(f"invalid {name}")
    if Path(args.output).exists():
        raise FileExistsError("output exists; choose a new directory")


def source_snapshot(output):
    payloads = {name: (ROOT / name).read_bytes() for name in SOURCE_FILES}
    source = {
        **capture_git_provenance(ROOT),
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
    }
    with tarfile.open(output / "source.tar.gz", "x:gz") as archive:
        for name, data in payloads.items():
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o644
            archive.addfile(member, io.BytesIO(data))
    write_json(output / "source.json", source)
    return source


def run_feedback_loop(
    config,
    truth,
    controller,
    *,
    feedback,
    condition,
    profile,
    duration,
    seed,
    stribeck_friction_nm=0.0,
):
    """Sample at k, deliver max(0,k-delay), compute, then integrate k -> k+1.

    Holding sample zero during startup uses the explicit known-rest prior. Once a
    sample is delivered, an observer updates exactly once at its acquisition time.
    Measurement age is logged; an old estimate is never labelled as current truth.
    """
    if feedback not in FEEDBACKS or condition not in CONDITIONS:
        raise ValueError("unknown feedback or sensor condition")
    if (
        config.kind != "wheel"
        or controller.config.dt != config.dt
        or controller.config.torque_limit_nm != config.torque_limit_nm
    ):
        raise ValueError("wheel and controller must share timestep and torque limit")
    _positive_duration(duration, 2 * config.dt, "duration")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    sensor = CONDITIONS[condition]
    plant = ActuatorBench(config, truth, stribeck_friction_nm=stribeck_friction_nm)
    controller.reset()
    observer = (
        None
        if feedback == "synthetic_velocity"
        else EncoderVelocityObserver(EncoderVelocityConfig(method=feedback, dt=config.dt))
    )
    count = round(duration / config.dt)
    times = np.arange(count) * config.dt
    targets, accelerations = reference(profile, times)
    qnoise = np.random.default_rng(np.random.SeedSequence((seed, 401))).normal(size=count)
    vnoise = np.random.default_rng(np.random.SeedSequence((seed, 402))).normal(size=count)
    positions, measured_velocities, true_velocities = [], [], []
    rows, last_delivered = [], -1
    for index, time_s in enumerate(times):
        position, velocity = plant.state
        measured_q = position + qnoise[index] * sensor["position_noise_std_rad"]
        quantum = sensor["quantization_rad"]
        if quantum:
            measured_q = float(np.rint(measured_q / quantum) * quantum)
        positions.append(measured_q)
        # The reference channel is a different sensor, not equally noisy encoder differentiation.
        measured_velocities.append(0.0 if index == 0 else velocity + vnoise[index] * 0.002)
        true_velocities.append(velocity)
        delivered = max(0, index - sensor["delay_steps"])
        sample_time = delivered * config.dt
        updated = delivered != last_delivered
        if observer is None:
            estimated_position = positions[delivered]
            estimated_velocity = measured_velocities[delivered]
        else:
            if last_delivered < 0:
                estimate = observer.reset(
                    sample_time, positions[delivered], known_initial_rest=True
                )
            elif updated:
                estimate = observer.update(sample_time, positions[delivered])
            else:
                estimate = observer.state
            estimated_position = estimate.position_rad
            estimated_velocity = estimate.velocity_rad_s
        last_delivered = delivered
        control = controller.compute(targets[index], accelerations[index], estimated_velocity)
        next_position, next_velocity = plant.step(control.command_torque_nm)
        if not np.isfinite((next_position, next_velocity)).all():
            raise FloatingPointError("non-finite plant state; cannot issue completed summary")
        guard = bool(abs(next_velocity) > SPEED_GUARD_RAD_S)
        rows.append(
            {
                "time_s": time_s,
                "sample_time_s": sample_time,
                "sample_age_s": time_s - sample_time,
                "delivered_sample_index": delivered,
                "measurement_updated": updated,
                "position_measurement_rad": measured_q,
                "delivered_position_rad": positions[delivered],
                "estimated_position_rad": estimated_position,
                "actual_velocity_at_sample_rad_s": true_velocities[delivered],
                "reference_velocity_rad_s": targets[index],
                "reference_acceleration_rad_s2": accelerations[index],
                "position_rad": position,
                "actual_velocity_rad_s": velocity,
                "measured_velocity_rad_s": estimated_velocity,
                "estimation_error_at_sample_rad_s": estimated_velocity - true_velocities[delivered],
                "estimation_error_current_rad_s": estimated_velocity - velocity,
                "tracking_error_rad_s": targets[index] - velocity,
                **asdict(control),
                "applied_torque_nm": plant.applied_torque_nm,
                "next_position_rad": next_position,
                "next_velocity_rad_s": next_velocity,
                "speed_guard_triggered": guard,
            }
        )
        if guard:
            break
    return rows


def feedback_metrics(rows, dt, duration):
    metrics = compute_metrics(rows, dt)
    for timing in ("at_sample", "current"):
        error = np.array([r[f"estimation_error_{timing}_rad_s"] for r in rows])
        metrics[f"estimation_rmse_{timing}_rad_s"] = float(np.sqrt(np.mean(error**2)))
    metrics["completed"] = (
        len(rows) == round(duration / dt) and not metrics["speed_guard_triggered"]
    )
    return metrics


def plot_case(destination, profile, records):
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    first = records[FEEDBACKS[0]]
    axes[0].plot(
        [r["time_s"] for r in first],
        [r["reference_velocity_rad_s"] for r in first],
        "k--",
        label="reference",
        lw=1,
    )
    for feedback, color in zip(FEEDBACKS, ("#777777", "#b86735", "#237e99")):
        rows = records[feedback]
        for axis, field in zip(
            axes, ("actual_velocity_rad_s", "estimation_error_current_rad_s", "command_torque_nm")
        ):
            axis.plot(
                [r["time_s"] for r in rows],
                [r[field] for r in rows],
                color=color,
                lw=1,
                label=feedback,
            )
    for axis, label in zip(
        axes, ("Speed [rad/s]", "Estimate - current speed [rad/s]", "Command [N m]")
    ):
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Control time [s]")
    figure.suptitle(f"{destination.name}/{profile}: fixed-axis encoder feedback")
    figure.savefig(destination / f"{profile}.png", dpi=120)
    plt.close(figure)


def run(args):
    validate_args(args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = source_snapshot(output)
    config = BenchConfig()
    control_config = WheelControlConfig(dt=config.dt, torque_limit_nm=config.torque_limit_nm)
    protocol = {
        "schema": "encoder-feedback-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {**vars(args), "output": str(output)},
        "data_origin": "synthetic_only",
        "bench": asdict(config),
        "controller_all_arms": asdict(control_config),
        "generator_truth_not_available_to_observer": asdict(TRUTH),
        "extra_stribeck_nm": 0.08,
        "calibration_seed": CALIBRATION_SEED,
        "calibration": "two position-only logs per scenario; one shared fit across feedbacks/conditions/seeds",
        "feedbacks": list(FEEDBACKS),
        "profiles": list(PROFILES),
        "sensor_conditions": {c: CONDITIONS[c] for c in args.conditions},
        "observer_configs": {f: asdict(EncoderVelocityConfig(method=f)) for f in FEEDBACKS[1:]},
        "synthetic_velocity_noise_std_rad_s": 0.002,
        "reference_sensor_scope": "independent synthetic velocity sensor; not equivalent noise/quantization to encoder differentiation",
        "measurement_delay": "same transport delay for all feedbacks; sample zero held at startup; known-rest v0=0",
        "timing": "sample k, deliver max(0,k-d), update only new timestamp, controller, physical step",
        "compensation": "none; model-free estimation at sample time, no extrapolation of old state",
        "known_prior": "initial rest, zero pending torque commands, continuous encoder angle",
        "scope": "single hidden parameter set, repeated noise; no contact, D1, hardware or RL claim",
        "expected_control_cases": len(args.scenarios)
        * len(args.conditions)
        * len(args.seeds)
        * len(PROFILES)
        * len(FEEDBACKS),
        "versions": {name: version(name) for name in ("numpy", "scipy", "mujoco", "matplotlib")},
    }
    write_json(output / "protocol.json", protocol)
    start = perf_counter()
    fits, metrics = [], []
    for scenario in args.scenarios:
        calibration = output / f"{scenario}_calibration"
        calibration.mkdir()
        generated = collect(
            config,
            TRUTH,
            scenario=scenario,
            duration=args.calibration_duration,
            noise_scale=1.0,
            seed=CALIBRATION_SEED,
        )
        for log in generated[:2]:
            save_encoder_log(
                calibration / f"{log.name}.csv",
                EncoderLog(
                    log.config,
                    log.commands,
                    log.positions,
                    known_initial_rest=True,
                    name=log.name,
                    split=log.split,
                ),
            )
        logs = [load_encoder_log(calibration / f"calibration_{i}.csv") for i in range(2)]
        fit = fit_encoder_parameters(
            logs, max_delay_steps=args.max_delay_steps, max_nfev=args.max_nfev
        )
        record = {
            "scenario": scenario,
            **asdict(fit),
            "fit_status": fit_status(fit, args.max_delay_steps),
        }
        write_json(calibration / "fit.json", record)
        fits.append(record)
        for condition in args.conditions:
            for seed in args.seeds:
                destination = output / f"{scenario}_{condition}_seed{seed}"
                destination.mkdir()
                for profile in PROFILES:
                    records = {}
                    for feedback in FEEDBACKS:
                        rows = run_feedback_loop(
                            config,
                            TRUTH,
                            WheelVelocityController(control_config, fit.parameters),
                            feedback=feedback,
                            condition=condition,
                            profile=profile,
                            duration=args.duration,
                            seed=seed,
                            stribeck_friction_nm=0.08 if scenario == "mismatch" else 0.0,
                        )
                        write_csv_gz(destination / f"{profile}_{feedback}.csv.gz", rows)
                        records[feedback] = rows
                        metrics.append(
                            {
                                "scenario": scenario,
                                "condition": condition,
                                "seed": seed,
                                "profile": profile,
                                "feedback": feedback,
                                **feedback_metrics(rows, config.dt, args.duration),
                            }
                        )
                    plot_case(destination, profile, records)
                print(f"{destination.name}: 9 paired control cases recorded", flush=True)
    changed = [
        name
        for name, digest in source["sha256"].items()
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
    ]
    write_json(output / "source_consistency.json", {"unchanged": not changed, "changed": changed})
    if changed:
        raise RuntimeError("source changed during experiment")
    if len(metrics) != protocol["expected_control_cases"]:
        raise RuntimeError("incomplete experiment matrix")
    write_csv_gz(output / "metrics.csv.gz", metrics)
    write_json(
        output / "summary.json",
        {
            "status": "completed"
            if all(r["completed"] for r in metrics)
            else "contains_incomplete_runs",
            "fits": fits,
            "metrics": metrics,
            "elapsed_seconds": perf_counter() - start,
            "scope": protocol["scope"],
        },
    )
    write_json(
        output / "manifest.json",
        {
            "sha256": {
                str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(output.rglob("*"))
                if p.is_file()
            }
        },
    )
    print(f"Results: {output}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 103, 107])
    parser.add_argument(
        "--scenarios", nargs="+", choices=("matched", "mismatch"), default=["matched", "mismatch"]
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(CONDITIONS)
    )
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--calibration-duration", type=float, default=3.0)
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
