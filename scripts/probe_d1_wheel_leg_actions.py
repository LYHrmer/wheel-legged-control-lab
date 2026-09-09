"""Deterministic, no-training action-channel probes for the D1 v3 low layer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.model import (
    JOINT_POSITION_HIGH,
    JOINT_POSITION_LOW,
    JOINT_TORQUE_LIMIT,
    JOINT_VELOCITY_LIMIT,
    D1Plant,
)
from wheel_legged_control.d1.sensor_estimation import D1SensorNoise
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource
from wheel_legged_control.d1.state_provider import D1StateProviderConfig, build_d1_state_provider
from wheel_legged_control.d1.wheel_leg_controller import D1WheelLegController

ROOT = Path(__file__).resolve().parents[1]
CASES = ["stand", "start_stop", "turn_left", "turn_right"] + [
    f"action_{i}_{s}" for i in range(8) for s in ("plus", "minus")
]


def command_action(case: str, time_s: float) -> tuple[D1Command, np.ndarray]:
    action = np.zeros(8)
    ramp = float(np.clip((time_s - 1.0) / 0.5, 0.0, 1.0))
    if case.startswith("action_"):
        _, axis, sign = case.split("_")
        action[int(axis)] = (0.5 if sign == "plus" else -0.5) * ramp
        return D1Command(), action
    if case == "start_stop":
        velocity = 0.25 if 1.0 <= time_s < 3.0 else -0.25 if 5.0 <= time_s < 7.0 else 0.0
        return D1Command(forward_velocity_mps=velocity), action
    if case.startswith("turn_"):
        direction = 1 if case == "turn_left" else -1
        return D1Command(
            forward_velocity_mps=0.15 * ramp, yaw_rate_rps=direction * 0.15 * ramp
        ), action
    return D1Command(), action


def run_case(
    case: str,
    *,
    sampling_mode: str = "synchronized",
    wheel_kp: float = 2.2,
    wheel_ki: float = 3.0,
    yaw_gain: float = 4.0,
    state_source: str = "oracle",
    sensor_delay_steps: int = 0,
    seed: int = 17,
) -> tuple[list[dict], dict]:
    kwargs = {} if sampling_mode == "legacy" else {"sampling_mode": sampling_mode}
    plant = D1Plant(**kwargs)
    source = D1MujocoTruthStateSource(plant)
    controller = D1WheelLegController(control_dt=plant.control_dt, yaw_feedback_gain=yaw_gain)
    controller.wheel_kp = wheel_kp
    controller.wheel_ki = wheel_ki
    controller.yaw_feedback_gain = yaw_gain
    state = source.reset()
    provider = None
    if state_source == "sensor":
        config = D1StateProviderConfig(
            kind="imu_encoder_fusion",
            sensor_noise=D1SensorNoise(
                gyro_std_rad_s=0.002,
                accelerometer_std_m_s2=0.03,
                encoder_position_std_rad=0.0005,
                encoder_velocity_std_rad_s=0.005,
            ),
            sensor_delay_steps=sensor_delay_steps,
            initial_position_m=(0.0, 0.0, 0.455),
            initial_rpy_rad=(0.0, 0.0, 0.0),
        )
        provider = build_d1_state_provider(plant, config)
    control_state = provider.reset(seed=seed) if provider is not None else state
    duration = 8.0 if case in ("start_stop", "turn_left", "turn_right") else 4.0
    rows = []
    for step in range(round(duration / plant.control_dt)):
        command, action = command_action(case, step * plant.control_dt)
        ground_height = provider.ground_reference().height_m if provider is not None else 0.0
        control_command = replace(command, base_height_m=command.base_height_m + ground_height)
        before_estimated_yaw_rate = float(control_state.base_angular_velocity_body[2])
        before = perf_counter()
        torque = controller.compute(
            control_command, control_state, action, ground_height_m=ground_height
        )
        compute_ms = (perf_counter() - before) * 1000
        report = controller.last_result
        plant.step(torque)
        state = source.read()
        control_state = provider.advance(step + 1) if provider is not None else state
        offsets = state.foot_offset_world @ state.base_rotation
        row = {
            "step": step + 1,
            "time_s": (step + 1) * plant.control_dt,
            "target_time_s": step * plant.control_dt,
            "measurement_time_s": control_state.measurement_time_s,
            "control_state_time_s": control_state.control_time_s,
            "truth_time_s": state.control_time_s,
            "ground_estimate_before_m": ground_height,
            "control_yaw_rate_before_rps": before_estimated_yaw_rate,
            "estimated_yaw_rate_after_rps": float(control_state.base_angular_velocity_body[2]),
            "case": case,
            "command_vx_mps": command.forward_velocity_mps,
            "command_yaw_rps": command.yaw_rate_rps,
            "effective_yaw_request_rps": report.effective_yaw_request_rps,
            "height_m": float(state.base_position[2]),
            "height_error_m": float(state.base_position[2] - command.base_height_m),
            "vx_mps": float(state.base_linear_velocity_body[0]),
            "yaw_rate_rps": float(state.base_angular_velocity_body[2]),
            "roll_rad": float(state.base_rpy[0]),
            "pitch_rad": float(state.base_rpy[1]),
            "yaw_rad": float(state.base_rpy[2]),
            "compute_ms": compute_ms,
            "fallen": int(state.has_fallen()),
            "torque_limit_fraction": float(np.max(np.abs(torque) / JOINT_TORQUE_LIMIT)),
            "velocity_limit_fraction": float(
                np.max(np.abs(state.joint_velocity) / JOINT_VELOCITY_LIMIT)
            ),
            "joint_position_violation_rad": float(
                max(
                    0,
                    np.max(JOINT_POSITION_LOW - state.joint_position),
                    np.max(state.joint_position - JOINT_POSITION_HIGH),
                )
            ),
        }
        for leg in range(4):
            row[f"wheel_z_body_{leg}_m"] = float(offsets[leg, 2])
            row[f"wheel_qd_{leg}_rad_s"] = float(state.joint_velocity[4 * leg + 3])
            row[f"extension_target_{leg}_m"] = float(report.leg_extension_target_m[leg])
            row[f"wheel_target_{leg}_rad_s"] = float(report.wheel_speed_target_rad_s[leg])
            row[f"wheel_integral_before_{leg}_nm"] = float(
                report.memory_before.wheel_integral_nm[leg]
            )
            row[f"wheel_integral_after_{leg}_nm"] = float(
                report.memory_after.wheel_integral_nm[leg]
            )
            row[f"wheel_raw_torque_{leg}_nm"] = float(report.wheel_nm[4 * leg + 3])
            row[f"wheel_torque_limited_{leg}"] = int(report.torque_limited[4 * leg + 3])
        for axis in range(8):
            row[f"action_{axis}"] = float(action[axis])
        for joint in range(16):
            row[f"q_{joint}_rad"] = float(state.joint_position[joint])
            row[f"target_q_{joint}_rad"] = float(report.joint_target_rad[joint])
            row[f"torque_{joint}_nm"] = float(torque[joint])
        rows.append(row)
        if row["fallen"]:
            break
    tail = rows[-min(100, len(rows)) :]
    rms = lambda values: float(np.sqrt(np.mean(np.square(values))))
    summary = {
        "case": case,
        "completed": len(rows) == round(duration / plant.control_dt) and not rows[-1]["fallen"],
        "executed_steps": len(rows),
        "tail_height_rmse_m": rms([r["height_error_m"] for r in tail]),
        "tail_vx_rmse_mps": rms([r["vx_mps"] - r["command_vx_mps"] for r in tail]),
        "tail_yaw_rmse_rps": rms([r["yaw_rate_rps"] - r["command_yaw_rps"] for r in tail]),
        "max_abs_roll_pitch_rad": max(max(abs(r["roll_rad"]), abs(r["pitch_rad"])) for r in rows),
        "yaw_change_rad": rows[-1]["yaw_rad"] - rows[0]["yaw_rad"],
        "max_velocity_limit_fraction": max(r["velocity_limit_fraction"] for r in rows),
        "max_joint_position_violation_rad": max(r["joint_position_violation_rad"] for r in rows),
        "compute_median_ms": float(np.median([r["compute_ms"] for r in rows])),
        "compute_p99_ms": float(np.percentile([r["compute_ms"] for r in rows], 99)),
        "tail_wheel_z_body_m": [
            float(np.mean([r[f"wheel_z_body_{i}_m"] for r in tail])) for i in range(4)
        ],
        "tail_wheel_velocity_rad_s": [
            float(np.mean([r[f"wheel_qd_{i}_rad_s"] for r in tail])) for i in range(4)
        ],
    }
    summary["stand_quality_pass"] = bool(
        summary["completed"]
        and summary["tail_height_rmse_m"] <= 0.025
        and summary["max_abs_roll_pitch_rad"] <= 0.1
    )
    commanded_yaw = sum(r["command_yaw_rps"] for r in rows) * plant.control_dt
    summary["commanded_yaw_change_rad"] = commanded_yaw
    summary["turn_progress_fraction"] = (
        summary["yaw_change_rad"] / commanded_yaw if abs(commanded_yaw) > 1e-12 else None
    )
    summary["turn_quality_pass"] = (
        bool(
            summary["completed"]
            and summary["turn_progress_fraction"] >= 0.25
            and summary["tail_yaw_rmse_rps"] <= 0.10
        )
        if case.startswith("turn_")
        else None
    )
    summary["turn_70_quality_pass"] = (
        bool(
            summary["completed"]
            and summary["turn_progress_fraction"] >= 0.70
            and summary["tail_yaw_rmse_rps"] <= 0.05
        )
        if case.startswith("turn_")
        else None
    )
    if case == "start_stop":
        summary["velocity_rmse_by_interval_mps"] = {
            f"{start:g}-{end:g}s": rms(
                [
                    r["vx_mps"] - r["command_vx_mps"]
                    for r in rows
                    if start <= r["target_time_s"] < end
                ]
            )
            for start, end in ((0, 1), (1, 3), (3, 5), (5, 7), (7, 8))
        }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument(
        "--sampling-mode", choices=("legacy", "synchronized"), default="synchronized"
    )
    parser.add_argument("--require-standing", action="store_true")
    parser.add_argument("--wheel-kp", type=float, default=2.2)
    parser.add_argument("--wheel-ki", type=float, default=3.0)
    parser.add_argument("--yaw-gain", type=float, default=4.0)
    parser.add_argument("--state-source", choices=("oracle", "sensor"), default="oracle")
    parser.add_argument("--sensor-delay-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--require-turn-70", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    if not np.isfinite(args.wheel_kp) or args.wheel_kp <= 0:
        parser.error("wheel-kp must be finite and positive")
    if not np.isfinite(args.wheel_ki) or args.wheel_ki < 0:
        parser.error("wheel-ki must be finite and nonnegative")
    if not np.isfinite(args.yaw_gain) or args.yaw_gain < 0:
        parser.error("yaw-gain must be finite and nonnegative")
    if (
        args.sensor_delay_steps < 0
        or args.seed < 0
        or (args.state_source == "oracle" and args.sensor_delay_steps)
    ):
        parser.error("seed/delay must be nonnegative and sensor delay requires sensor source")
    args.output.mkdir(parents=True)
    source_files = [
        Path(__file__),
        ROOT / "src/wheel_legged_control/d1/wheel_leg_controller.py",
        ROOT / "src/wheel_legged_control/d1/model.py",
        ROOT / "src/wheel_legged_control/d1/state_estimation.py",
        ROOT / "src/wheel_legged_control/d1/control_context.py",
        ROOT / "src/wheel_legged_control/d1/controllers.py",
        ROOT / "src/wheel_legged_control/d1/terrain.py",
        ROOT / "src/wheel_legged_control/d1/training_terrain.py",
        ROOT / "src/wheel_legged_control/d1/state_provider.py",
        ROOT / "src/wheel_legged_control/d1/sensor_estimation.py",
    ]
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files
    }
    with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
        for path in source_files:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    protocol = {
        "cases": args.cases,
        "sampling_mode": args.sampling_mode,
        "source_sha256": hashes,
        "scope": "deterministic development probes; no RL, no hardware claims",
        "state_source": args.state_source,
        "sensor_delay_steps": args.sensor_delay_steps,
        "measurement_noise_seed": args.seed,
        "sensor_noise_std": {
            "gyro_rad_s": 0.002,
            "accelerometer_m_s2": 0.03,
            "encoder_position_rad": 0.0005,
            "encoder_velocity_rad_s": 0.005,
        }
        if args.state_source == "sensor"
        else None,
        "standing_thresholds": {"height_rmse_m": 0.025, "max_abs_roll_pitch_rad": 0.1},
        "action_amplitude": 0.5,
        "action_order": [
            "extension_FL",
            "extension_FR",
            "extension_RL",
            "extension_RR",
            "wheel_FL",
            "wheel_FR",
            "wheel_RL",
            "wheel_RR",
        ],
        "action_scales": {"extension_m": 0.04, "wheel_speed_rad_s": 4.0},
        "asset_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src/wheel_legged_control/d1/assets").rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        },
        "wheel_kp_nm_per_rad_s": args.wheel_kp,
        "wheel_ki_nm_per_rad": args.wheel_ki,
        "yaw_feedback_gain": args.yaw_gain,
        "yaw_request_limit_rps": 0.6,
        "control_schema": D1WheelLegController.control_schema,
        "turn_70_thresholds": {
            "minimum_commanded_progress_fraction": 0.70,
            "tail_yaw_rmse_max_rps": 0.05,
            "straight_stop_noninferiority_relative_margin": 0.10,
        },
        "turn_thresholds": {
            "minimum_commanded_progress_fraction": 0.25,
            "tail_yaw_rmse_max_rps": 0.10,
        },
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summaries = []
    for case in args.cases:
        rows, summary = run_case(
            case,
            sampling_mode=args.sampling_mode,
            wheel_kp=args.wheel_kp,
            wheel_ki=args.wheel_ki,
            yaw_gain=args.yaw_gain,
            state_source=args.state_source,
            sensor_delay_steps=args.sensor_delay_steps,
            seed=args.seed,
        )
        with (args.output / f"{case}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    source_unchanged = all(
        hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items()
    )
    response_checks = []
    by_case = {s["case"]: s for s in summaries}
    if "stand" in by_case:
        baseline = by_case["stand"]
        for axis in range(8):
            for sign, direction in (("plus", 1), ("minus", -1)):
                name = f"action_{axis}_{sign}"
                if name not in by_case:
                    continue
                metric = "tail_wheel_z_body_m" if axis < 4 else "tail_wheel_velocity_rad_s"
                leg = axis % 4
                delta = by_case[name][metric][leg] - baseline[metric][leg]
                signed = -direction * delta if axis < 4 else direction * delta
                minimum = 0.001 if axis < 4 else 0.05
                response_checks.append(
                    {
                        "case": name,
                        "delta": delta,
                        "unit": "m" if axis < 4 else "rad/s",
                        "minimum_signed_response": minimum,
                        "passed": bool(signed >= minimum and by_case[name]["completed"]),
                    }
                )
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "source_unchanged": source_unchanged,
                "cases": summaries,
                "action_response_checks": response_checks,
            },
            indent=2,
        )
        + "\n"
    )
    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.iterdir()
        if p.is_file()
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.require_standing and any(not s["stand_quality_pass"] for s in summaries):
        raise SystemExit(1)
    if args.require_turn_70 and any(s["turn_70_quality_pass"] is False for s in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
