"""Development-only sensor replay and actual feedback runs; no held-out claims."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np

from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.env import encode_d1_observation
from wheel_legged_control.d1.hierarchical import D1LQRVMCController
from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.sensor_estimation import (
    D1ProprioceptiveEstimator,
    D1SensorNoise,
    D1SensorStateSource,
)
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource
from wheel_legged_control.d1.terrain_tracking_env import terrain_normal_to_rpy
from wheel_legged_control.d1.training_terrain import TrainingTerrainConfig

NOISE_PROFILES = {
    "ideal": D1SensorNoise(),
    "noise": D1SensorNoise(
        gyro_std_rad_s=0.002,
        accelerometer_std_m_s2=0.05,
        encoder_position_std_rad=0.0001,
        encoder_velocity_std_rad_s=0.005,
    ),
    "bias": D1SensorNoise(
        gyro_std_rad_s=0.002,
        accelerometer_std_m_s2=0.05,
        encoder_position_std_rad=0.0001,
        encoder_velocity_std_rad_s=0.005,
        gyro_bias_rad_s=(0.002, -0.003, 0.005),
        accelerometer_bias_m_s2=(0.05, -0.03, 0.08),
    ),
}
TERRAINS = {
    "flat": TrainingTerrainConfig(),
    "ramp_up_4": TrainingTerrainConfig(kind="ramp", slope_deg=4.0),
    "ramp_down_4": TrainingTerrainConfig(kind="ramp", slope_deg=-4.0),
    "bumps_5mm": TrainingTerrainConfig(kind="bumps", amplitude_m=0.005, wavelength_m=0.8),
}
INITIAL_POSITION = (0.0, 0.0, 0.480)


def run_case(
    *,
    mode: str,
    noise_name: str,
    terrain_name: str,
    seed: int,
    duration_s: float = 4.0,
    delay_steps: int = 0,
) -> tuple[dict, list[dict], dict[str, np.ndarray]]:
    """Keep controller gains and spawn prior fixed; switch only its state input.

    Offline: truth state/reference controls the robot; measurements feed an
    unused observer. Closed loop: the observer supplies all control feedback,
    terrain reference and policy-compatible observation. Residual action is zero.
    """
    if mode not in {"offline", "closed_loop"}:
        raise ValueError("mode must be offline or closed_loop")
    if not np.isfinite(duration_s) or not 0.01 <= duration_s <= 10.0:
        raise ValueError("duration_s must be between 0.01 and 10 for this development suite")
    plant = D1Plant(training_terrain=TERRAINS[terrain_name])
    plant.reset(base_position=np.asarray(INITIAL_POSITION))
    controller = D1LQRVMCController(plant)
    truth_source = D1MujocoTruthStateSource(plant)
    source = D1SensorStateSource(
        plant,
        noise=NOISE_PROFILES[noise_name],
        initial_position=INITIAL_POSITION,
        delay_steps=delay_steps,
    )
    truth, estimate = truth_source.reset(), source.reset(seed=seed)
    packets = [source.latest_measurement]
    estimates = [estimate]
    rows = []
    started = time.perf_counter()
    termination = "time_limit"
    for step in range(round(duration_s / plant.control_dt)):
        state = estimate if mode == "closed_loop" else truth
        if mode == "closed_loop":
            ground = source.ground_reference(state)
        else:
            ground = plant.training_ground_reference(*truth.base_position[:2])
        roll, pitch = terrain_normal_to_rpy(
            -np.tan(ground.pitch_rad), np.tan(ground.roll_rad), state.base_rpy[2]
        )
        fraction = float(np.clip((step * plant.control_dt - 0.3) / 0.3, 0, 1))
        velocity_command = 0.35 * fraction**2 * (3 - 2 * fraction)
        command = D1Command(
            forward_velocity_mps=velocity_command,
            base_height_m=ground.height_m + 0.455,
            roll_rad=roll,
            pitch_rad=pitch,
        )
        # This is the same estimation-only input a residual policy will receive.
        observation = np.concatenate(
            (
                encode_d1_observation(
                    state=state,
                    command=command,
                    baseline_longitudinal_force_n=controller.last_longitudinal_force_n,
                    previous_applied_action=np.zeros(2),
                ),
                [pitch, roll],
            )
        )
        assert observation.shape == (44,) and np.isfinite(observation).all()
        plant.step(controller.compute(command, state, residual_force_n=np.zeros(2)))
        truth, estimate = truth_source.read(), source.read()
        packets.append(source.latest_measurement)
        estimates.append(estimate)
        actual_ground = plant.training_ground_reference(*truth.base_position[:2])
        estimated_ground = source.ground_reference(estimate)
        error_rpy = (estimate.base_rpy - truth.base_rpy + np.pi) % (2 * np.pi) - np.pi
        row = {
            "step": step + 1,
            "time_s": (step + 1) * plant.control_dt,
            "command_velocity_mps": velocity_command,
            "truth_x_m": float(truth.base_position[0]),
            "estimate_x_m": float(estimate.base_position[0]),
            "truth_height_m": float(truth.base_position[2]),
            "estimate_height_m": float(estimate.base_position[2]),
            "truth_velocity_body_x_mps": float(truth.base_linear_velocity_body[0]),
            "estimate_velocity_body_x_mps": float(estimate.base_linear_velocity_body[0]),
            "truth_velocity_world_z_mps": float(truth.base_linear_velocity_world[2]),
            "estimate_velocity_world_z_mps": float(estimate.base_linear_velocity_world[2]),
            "truth_pitch_rad": float(truth.base_rpy[1]),
            "estimate_pitch_rad": float(estimate.base_rpy[1]),
            "roll_error_rad": float(error_rpy[0]),
            "pitch_error_rad": float(error_rpy[1]),
            "yaw_error_rad": float(error_rpy[2]),
            "truth_ground_height_m": actual_ground.height_m,
            "estimate_ground_height_m": estimated_ground.height_m,
            "truth_ground_pitch_rad": actual_ground.pitch_rad,
            "estimate_ground_pitch_rad": estimated_ground.pitch_rad,
            "truth_clearance_m": float(truth.base_position[2] - actual_ground.height_m),
            "estimate_clearance_m": float(estimate.base_position[2] - estimated_ground.height_m),
            "contact_switches": int(np.count_nonzero(estimate.wheel_contact)),
            "support_plane_observable": int(
                source.estimator.diagnostics["support_plane_observable"]
            ),
            "accelerometer_used": int(source.estimator.diagnostics["attitude_accelerometer_used"]),
            "measurement_time_s": estimate.measurement_time_s,
            "measurement_age_s": estimate.age_s,
        }
        rows.append(row)
        if row["truth_clearance_m"] < 0.22 or np.max(np.abs(truth.base_rpy[:2])) > 0.85:
            termination = "fallen"
            break
        if np.max(np.abs(truth.base_position[:2])) > 5.5:
            termination = "out_of_bounds"
            break
    elapsed = time.perf_counter() - started
    # A fresh observer with no plant object must exactly reproduce every state.
    replay = D1ProprioceptiveEstimator()
    replay_state = replay.reset(packets[0], initial_position=INITIAL_POSITION)
    replayed = [replay_state]
    for index, current_packet in enumerate(packets[1:], start=1):
        delayed_packet = packets[max(0, index - delay_steps)]
        if index > delay_steps:
            replay_state = replay.update(delayed_packet)
        replayed.append(
            replace(
                replay_state,
                sequence=current_packet.sequence,
                control_time_s=current_packet.time_s,
                measurement_time_s=delayed_packet.time_s,
            )
        )
    replay_exact = all(
        np.array_equal(getattr(actual, field.name), getattr(expected, field.name))
        for actual, expected in zip(replayed, estimates, strict=True)
        for field in fields(actual)
    )

    def rmse(values):
        return float(np.sqrt(np.mean(np.square(values))))

    summary = {
        "mode": mode,
        "noise": noise_name,
        "noise_config": asdict(NOISE_PROFILES[noise_name]),
        "terrain": terrain_name,
        "terrain_config": asdict(TERRAINS[terrain_name]),
        "seed": seed,
        "delay_steps": delay_steps,
        "steps": len(rows),
        "duration_s": len(rows) * plant.control_dt,
        "termination_reason": termination,
        "position_x_final_error_m": rows[-1]["estimate_x_m"] - rows[-1]["truth_x_m"],
        "height_final_error_m": rows[-1]["estimate_height_m"] - rows[-1]["truth_height_m"],
        "pitch_rmse_rad": rmse([r["pitch_error_rad"] for r in rows]),
        "yaw_final_error_rad": rows[-1]["yaw_error_rad"],
        "velocity_estimation_rmse_mps": rmse(
            [r["estimate_velocity_body_x_mps"] - r["truth_velocity_body_x_mps"] for r in rows]
        ),
        "clearance_estimation_rmse_m": rmse(
            [r["estimate_clearance_m"] - r["truth_clearance_m"] for r in rows]
        ),
        "ground_pitch_rmse_rad": rmse(
            [r["estimate_ground_pitch_rad"] - r["truth_ground_pitch_rad"] for r in rows]
        ),
        "tracking_rmse_mps": rmse(
            [r["truth_velocity_body_x_mps"] - r["command_velocity_mps"] for r in rows]
        ),
        "forward_displacement_m": rows[-1]["truth_x_m"],
        "replay_bitwise_equal": replay_exact,
        "support_plane_observable_fraction": float(
            np.mean([r["support_plane_observable"] for r in rows])
        ),
        "wall_time_s": elapsed,
    }
    packet_arrays = {
        field.name: np.asarray([getattr(packet, field.name) for packet in packets])
        for field in fields(packets[0])
    }
    return summary, rows, packet_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/d1_sensor_estimation/development_v1")
    )
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    parser.add_argument("--delays", nargs="+", type=int, default=[0, 2])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {args.output}")
    args.output.mkdir(parents=True)
    summaries = []
    for mode in ("offline", "closed_loop"):
        for terrain in TERRAINS:
            for noise in NOISE_PROFILES:
                for seed in args.seeds:
                    for delay_steps in args.delays:
                        summary, rows, packets = run_case(
                            mode=mode,
                            noise_name=noise,
                            terrain_name=terrain,
                            seed=seed,
                            duration_s=args.duration,
                            delay_steps=delay_steps,
                        )
                        case = f"{mode}_{terrain}_{noise}_delay{delay_steps}_seed{seed}"
                        with (args.output / f"{case}.csv").open("w") as handle:
                            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                            writer.writeheader()
                            writer.writerows(rows)
                        np.savez_compressed(args.output / f"{case}_measurements.npz", **packets)
                        summaries.append(summary)
                        print(
                            case,
                            summary["termination_reason"],
                            f"velocity RMSE {summary['velocity_estimation_rmse_mps']:.4f}",
                            flush=True,
                        )
    source_paths = [Path(__file__), Path("src/wheel_legged_control/d1/sensor_estimation.py")]
    result = {
        "schema": D1ProprioceptiveEstimator.schema,
        "purpose": "development; not held-out evaluation",
        "initial_position_prior_m": INITIAL_POSITION,
        "initial_rpy_prior_rad": [0, 0, 0],
        "initial_velocity_prior_mps": [0, 0, 0],
        "contact_sensor_assumption": "four ideal summed normal-load >1N binary switches; no normal or force given to observer",
        "noise_parameters_status": "illustrative per-sample perturbations, not identified hardware sensor specs",
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths
        },
        "cases": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
