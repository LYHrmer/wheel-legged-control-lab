"""Paired D1 terrain-preview development trials, one CPU environment at a time."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np

from scripts.evaluate_d1_sensor_estimation import NOISE_PROFILES
from wheel_legged_control.d1.control_primitives import terrain_normal_to_rpy
from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.hierarchical import D1LQRVMCController
from wheel_legged_control.d1.linear_model import identify_sagittal_model
from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.sensor_estimation import D1SensorStateSource
from wheel_legged_control.d1.terrain_preview import PreviewChannel
from wheel_legged_control.d1.training_terrain import TrainingTerrainConfig

TERRAINS = {"flat": 0.0, "up": 4.0, "down": -4.0}
SEEDS = (8417, 8429, 8443)
ARMS = ("off", "on", "inverted")
DT = 0.01
STEPS = 600
SPEED_MPS = 0.35
RAMP_WIDTH_M = 0.12
ENTRY_DISTANCE_M = 0.4
SCORE_DISTANCE_M = 0.4
INITIAL_POSITION = (0.0, 0.0, 0.455)
VISIBLE_SLOPE_RAD = math.radians(0.5)
ACTION_THRESHOLD_RAD = 1e-5
CONFIRMATION_TICKS = 3
ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def protocol() -> dict:
    sources = (
        "docs/experiments84/terrain_preview84.json",
        "scripts/terrain_preview84_trials.py",
        "scripts/evaluate_d1_sensor_estimation.py",
        "src/wheel_legged_control/d1/terrain_preview.py",
        "src/wheel_legged_control/d1/model.py",
        "src/wheel_legged_control/d1/hierarchical.py",
        "src/wheel_legged_control/d1/sensor_estimation.py",
        "src/wheel_legged_control/d1/linear_model.py",
        "src/wheel_legged_control/d1/training_terrain.py",
    )
    return {
        "schema": "wheel-terrain-preview84-trials-v1",
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "terrain_deg": TERRAINS,
        "steps_per_case": STEPS,
        "dt_s": DT,
        "speed_mps": SPEED_MPS,
        "settle_s": 0.3,
        "speed_ramp_s": 0.3,
        "initial_position_m": INITIAL_POSITION,
        "proprioception": {
            "model": "synchronized",
            "delay_steps": 2,
            **asdict(NOISE_PROFILES["noise"]),
        },
        "entry_distance_ahead_of_front_axle_m": ENTRY_DISTANCE_M,
        "grade_blend_width_m": RAMP_WIDTH_M,
        "scoring_distance_from_ramp_toe_m": SCORE_DISTANCE_M,
        "minimum_visible_grade_deg": 0.5,
        "minimum_correction_rad": ACTION_THRESHOLD_RAD,
        "consecutive_correction_ticks": CONFIRMATION_TICKS,
        "lead_detection": "first 100 Hz sample of loaded front-wheel/floor contact at or beyond the toe; action time is pre-step",
        "pitch_score": "RMSE in degrees relative to the operating pitch plus the actual surface pitch at the post-step body center, scored from first front contact until the front axle crosses toe + 0.4m",
        "speed_effort_score": "entire recorded episode; scored pairs require all 600 steps",
        "gate": "all six on/off slope pairs cross and improve, median relative pitch improvement >= 10%, correct lower median pitch error than inverted for each slope, measured signed grade and a three-tick signed correction before slope contact, all nine on/off pairs finish with speed RMSE and torque RMS <= 105% of off, no additional limit violation",
        "saturation": "applied torque >= per-axis bound minus 1e-9; violation is requested torque beyond bound by >1e-9",
        "task_controls_cap": 16200,
        "native_task_steps_cap": 81000,
        "one_cold_identification_control_steps": 511,
        "one_cold_identification_native_steps": 2555,
        "training_updates": 0,
        "sources_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources
        },
    }


def ramp_heights(x: np.ndarray, toe_m: float, slope_deg: float) -> np.ndarray:
    """Integrate a smoothstep from zero to the target grade over the entry blend."""
    d = x - toe_m
    t = np.clip(d / RAMP_WIDTH_M, 0.0, 1.0)
    return math.tan(math.radians(slope_deg)) * (
        RAMP_WIDTH_M * (t**3 - 0.5 * t**4) + np.maximum(d - RAMP_WIDTH_M, 0.0)
    )


def install_ramp(plant: D1Plant, toe_m: float, slope_deg: float) -> np.ndarray:
    model, geom = plant.model, plant.floor_geom_id
    field = int(model.geom_dataid[geom])
    rows, cols = int(model.hfield_nrow[field]), int(model.hfield_ncol[field])
    half_x, _, height_scale, _ = model.hfield_size[field]
    heights = ramp_heights(np.linspace(-half_x, half_x, cols), toe_m, slope_deg)
    normalized = (heights - model.geom_pos[geom, 2]) / height_scale
    if np.any((normalized < 0) | (normalized > 1)):
        raise ValueError("terrain exceeds preallocated collision field")
    address = int(model.hfield_adr[field])
    field_values = model.hfield_data[address : address + rows * cols].reshape(rows, cols)
    field_values[:] = normalized
    return field_values.copy()


def front_axle_x(plant: D1Plant) -> float:
    return float(np.mean(plant.measurement_data.xpos[list(plant.wheel_body_ids_by_leg[:2]), 0]))


def front_contact_x(plant: D1Plant) -> float | None:
    """Furthest loaded front-wheel/floor contact from the synchronized physics sample."""
    model, data = plant.model, plant.measurement_data
    front = plant.wheel_body_ids_by_leg[:2]
    x_coordinates = []
    wrench = np.empty(6)
    for i, contact in enumerate(data.contact):
        a, b = int(contact.geom1), int(contact.geom2)
        if plant.floor_geom_id not in (a, b) or contact.efc_address < 0:
            continue
        other = b if a == plant.floor_geom_id else a
        if int(model.geom_bodyid[other]) not in front:
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        if wrench[0] > 0.0:
            x_coordinates.append(float(contact.pos[0]))
    return max(x_coordinates, default=None)


def commanded_speed(seconds: float) -> float:
    t = min(1.0, max(0.0, (seconds - 0.3) / 0.3))
    return SPEED_MPS * t * t * (3 - 2 * t)


def collect_commands(plant, controller, source, state, channel, arm):
    """Derive one applied command using only measurements and controller memory."""
    roll, pitch, yaw = state.base_rpy
    ground = source.ground_reference(state)
    support_roll, support_pitch = terrain_normal_to_rpy(
        -math.tan(ground.pitch_rad), math.tan(ground.roll_rad), yaw
    )
    velocity = commanded_speed(state.control_time_s)
    base = D1Command(
        forward_velocity_mps=velocity,
        base_height_m=ground.height_m + 0.455,
        roll_rad=support_roll,
        pitch_rad=support_pitch,
    )
    # The public preview_baseline reserves its exact command for compute. This
    # read-only tracking calculation leaves the final, changed command free.
    tracked, reference = controller._preview_tracking(base, state)
    baseline_force = -float((controller.gain @ (tracked - reference)).item())
    front_relative = (
        state.foot_position[:2].mean(axis=0) - state.base_position
    ) @ state.base_rotation[:, 0]
    reading = channel.observe(
        plant,
        state.control_time_s,
        measured_body_pitch_rad=float(pitch),
        measured_body_roll_rad=float(roll),
        travelled_m=float(state.base_position[0] - INITIAL_POSITION[0]),
        forward_track_x_m=float(front_relative),
        speed_mps=velocity,
    )
    gain = float(controller.gain[0, 1])
    delta = channel.correction(
        reading,
        arm,
        support_pitch_rad=support_pitch,
        baseline_force_n=baseline_force,
        pitch_gain_n_per_rad=gain,
        force_limit_n=controller.longitudinal_force_limit_n,
        dt_s=DT,
    )
    command = replace(base, pitch_rad=support_pitch + delta)
    return command, reading, float(delta), baseline_force, support_pitch


def case_metrics(
    rows: list[dict],
    terrain: str,
    seed: int,
    arm: str,
    toe: float,
    operating_pitch: float,
    gain: float,
    termination: str,
) -> dict:
    def first(predicate):
        return next((r for r in rows if predicate(r)), None)

    direction = np.sign(TERRAINS[terrain])
    seen = first(
        lambda r: r["preview_valid"] and direction * r["preview_slope_rad"] >= VISIBLE_SLOPE_RAD
    )
    moved = first(lambda r: abs(r["applied_delta_pitch_rad"]) >= ACTION_THRESHOLD_RAD)
    correct = confirmed = None
    streak = 0
    for index, row in enumerate(rows):
        qualified = (
            row["preview_valid"]
            and direction * row["preview_slope_rad"] >= VISIBLE_SLOPE_RAD
            and -direction * row["applied_delta_pitch_rad"] >= ACTION_THRESHOLD_RAD
        )
        streak = streak + 1 if qualified else 0
        if streak == CONFIRMATION_TICKS:
            correct, confirmed = rows[index + 1 - CONFIRMATION_TICKS], row
            break
    contact = first(lambda r: r["front_contact_x_m"] is not None and r["front_contact_x_m"] >= toe)
    crossed = (
        None
        if contact is None
        else first(
            lambda r: (
                r["step"] >= contact["step"] and r["front_track_x_m"] >= toe + SCORE_DISTANCE_M
            )
        )
    )
    window = [] if crossed is None else rows[contact["step"] - 1 : crossed["step"]]
    pitch = (
        math.degrees(math.sqrt(statistics.fmean(r["pitch_error_rad"] ** 2 for r in window)))
        if window
        else None
    )
    result = {
        "terrain": terrain,
        "seed": seed,
        "arm": arm,
        "steps": len(rows),
        "termination_reason": termination,
        "full_horizon": len(rows) == STEPS and termination == "time_limit",
        "progress_m": rows[-1]["base_x_m"] - INITIAL_POSITION[0],
        "first_seen_s": None if seen is None else seen["control_s"],
        "first_seen_capture_s": None if seen is None else seen["preview_captured_s"],
        "first_correction_s": None if moved is None else moved["control_s"],
        "first_correct_correction_s": None if correct is None else correct["control_s"],
        "correction_confirmed_s": None if confirmed is None else confirmed["control_s"],
        "first_front_contact_s": None if contact is None else contact["time_s"],
        "crossed_window_s": None if crossed is None else crossed["time_s"],
        "pitch_window_steps": len(window),
        "pitch_rmse_deg": pitch,
        "operating_pitch_rad": operating_pitch,
        "pitch_gain_n_per_rad": gain,
        "speed_rmse_mps": math.sqrt(statistics.fmean(r["speed_error_mps"] ** 2 for r in rows)),
        "torque_rms_nm": math.sqrt(statistics.fmean(r["torque_mean_square_nm2"] for r in rows)),
        "total_saturation_fraction": sum(r["saturated_axes"] for r in rows) / (16 * len(rows)),
        "torque_bound_violations": sum(r["torque_bound_violations"] for r in rows),
        "preview_valid_steps": sum(r["preview_valid"] for r in rows),
        "nonzero_force_headroom_steps": sum(r["force_headroom_n"] > 0 for r in rows),
    }
    # Whole control-tick lead filters apparent sub-tick timing advantages.
    result["signed_correction_before_contact"] = bool(
        seen and confirmed and contact and confirmed["control_s"] <= contact["time_s"] - DT + 1e-9
    )
    return result


def evaluate(cases: list[dict]) -> dict:
    by_case = {(c["terrain"], c["seed"], c["arm"]): c for c in cases}
    design = {(t, s, a) for t in TERRAINS for s in SEEDS for a in ARMS}
    if set(by_case) != design or len(cases) != len(design):
        raise ValueError("qualification requires all 27 unique terminal cases")
    pairs, slope_gains = [], []
    for terrain in TERRAINS:
        for seed in SEEDS:
            off, on = (by_case[terrain, seed, a] for a in ("off", "on"))
            checks = {
                "complete": off["full_horizon"] and on["full_horizon"],
                "speed": on["speed_rmse_mps"] <= 1.05 * off["speed_rmse_mps"],
                "effort": on["torque_rms_nm"] <= 1.05 * off["torque_rms_nm"],
                "bounds": on["torque_bound_violations"] <= off["torque_bound_violations"],
            }
            improvement = None
            if terrain != "flat":
                baseline, candidate = off["pitch_rmse_deg"], on["pitch_rmse_deg"]
                if baseline is not None and baseline > 0 and candidate is not None:
                    improvement = (baseline - candidate) / baseline
                    slope_gains.append(improvement)
                checks.update(
                    crossed=all(c["crossed_window_s"] is not None for c in (off, on)),
                    signed_lead=on["signed_correction_before_contact"],
                    pitch=improvement is not None and improvement > 0,
                )
            pairs.append(
                {
                    "terrain": terrain,
                    "seed": seed,
                    "checks": checks,
                    "pitch_relative_improvement": improvement,
                    "passed": all(checks.values()),
                }
            )
    sign_controls = {}
    for terrain in ("up", "down"):
        group = [by_case[terrain, s, a] for a in ("on", "inverted") for s in SEEDS]
        eligible = all(c["full_horizon"] and c["pitch_rmse_deg"] is not None for c in group)
        sign_controls[terrain] = eligible and statistics.median(
            c["pitch_rmse_deg"] for c in group[:3]
        ) < statistics.median(c["pitch_rmse_deg"] for c in group[3:])
    median = statistics.median(slope_gains) if len(slope_gains) == 6 else None
    return {
        "passed": bool(
            all(p["passed"] for p in pairs)
            and all(sign_controls.values())
            and median is not None
            and median >= 0.10
        ),
        "pairs": pairs,
        "correct_better_than_inverted_by_slope": sign_controls,
        "median_relative_pitch_improvement": median,
    }


def record_pose(stream, plant: D1Plant) -> None:
    np.concatenate(([plant.data.time], plant.data.qpos, plant.data.qvel)).astype("<f8").tofile(
        stream
    )
    stream.flush()


def run_case(directory: Path, terrain: str, seed: int, arm: str) -> dict:
    directory.mkdir(exist_ok=False)
    write_json(
        directory / "started.json",
        {"terrain": terrain, "seed": seed, "arm": arm, "status_until_metrics": "PARTIAL"},
    )
    plant = D1Plant(training_terrain=TrainingTerrainConfig(), sampling_mode="synchronized")
    if plant.physics_steps != 5 or plant.control_dt != DT:
        raise ValueError("100 Hz control / five native steps required")
    toe = front_axle_x(plant) + ENTRY_DISTANCE_M
    field = install_ramp(plant, toe, TERRAINS[terrain])
    plant.reset(base_position=np.asarray(INITIAL_POSITION))
    with (directory / "terrain.npz").open("xb") as f:
        np.savez_compressed(
            f, hfield=field, ramp_x_m=toe, ramp_width_m=RAMP_WIDTH_M, slope_deg=TERRAINS[terrain]
        )
    controller = D1LQRVMCController(plant)
    source = D1SensorStateSource(
        plant, noise=NOISE_PROFILES["noise"], initial_position=INITIAL_POSITION, delay_steps=2
    )
    state = source.reset(seed=seed)
    channel = PreviewChannel()
    operating_pitch = float(controller.linear_model.operating_state[1])
    gain = float(controller.gain[0, 1])
    rows = []
    termination = "time_limit"
    write_json(
        directory / "pose_layout.json",
        {
            "dtype": "<f8",
            "fields": ["time_s", "qpos", "qvel"],
            "nq": plant.model.nq,
            "nv": plant.model.nv,
            "initial_frame": True,
        },
    )
    with (
        (directory / "poses.raw").open("xb") as poses,
        (directory / "telemetry.csv").open("x", newline="") as telemetry,
        (directory / "frames.jsonl").open("x") as frames,
    ):
        record_pose(poses, plant)
        writer = None
        last_frame = None
        for step in range(STEPS):
            now = float(plant.data.time)
            if (
                abs(state.control_time_s - now) > 1e-9
                or abs(state.measurement_time_s - max(0.0, now - 2 * DT)) > 1e-9
            ):
                raise ValueError("delayed proprioception clock mismatch")
            command, reading, delta, base_force, support_pitch = collect_commands(
                plant, controller, source, state, channel, arm
            )
            torque = controller.compute(command, state)
            plant.step(torque)
            record_pose(poses, plant)
            pos = plant.base_position
            roll, body_pitch, yaw = plant.base_rpy
            terrain_ref = plant.training_ground_reference(float(pos[0]), float(pos[1]))
            _, slope_pitch = terrain_normal_to_rpy(
                -math.tan(terrain_ref.pitch_rad), math.tan(terrain_ref.roll_rad), yaw
            )
            speed = float(plant.base_velocity(local=True)[0][0])
            applied = plant.data.ctrl[plant.actuator_ids]
            limits = plant.actuator_torque_limit_nm
            row = {
                "step": step + 1,
                "control_s": now,
                "estimator_time_s": state.measurement_time_s,
                "time_s": float(plant.data.time),
                "base_x_m": float(pos[0]),
                "front_track_x_m": front_axle_x(plant),
                "front_contact_x_m": front_contact_x(plant),
                "truth_pitch_rad": float(body_pitch),
                "ground_pitch_rad": slope_pitch,
                "pitch_error_rad": float(body_pitch - operating_pitch - slope_pitch),
                "truth_speed_mps": speed,
                "command_speed_mps": command.forward_velocity_mps,
                "speed_error_mps": speed - command.forward_velocity_mps,
                "support_pitch_rad": support_pitch,
                "command_pitch_rad": command.pitch_rad,
                "applied_delta_pitch_rad": delta,
                "baseline_force_n": base_force,
                "force_headroom_n": max(
                    0.0, controller.longitudinal_force_limit_n - abs(base_force)
                ),
                "applied_longitudinal_force_n": controller.last_longitudinal_force_n,
                "torque_mean_square_nm2": float(np.mean(applied**2)),
                "saturated_axes": int(np.count_nonzero(np.abs(applied) >= limits - 1e-9)),
                "torque_bound_violations": int(np.count_nonzero(np.abs(torque) > limits + 1e-9)),
                "preview_valid": bool(reading.valid),
                "preview_slope_rad": reading.slope_rad,
                "preview_reason": reading.reason,
                "preview_sequence": reading.sequence,
                "preview_captured_s": reading.captured_s,
                "preview_available_s": reading.available_s,
                "preview_query_x_m": reading.query_x_m,
                "left_preview_slope_rad": reading.left_slope_rad,
                "right_preview_slope_rad": reading.right_slope_rad,
            }
            if reading.valid and not (reading.captured_s <= reading.available_s <= now + 1e-9):
                raise ValueError("future preview reached a command")
            if writer is None:
                writer = csv.DictWriter(telemetry, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            telemetry.flush()
            if reading.sequence is not None and reading.sequence != last_frame:
                frames.write(
                    json.dumps({"control_s": now, **reading.as_dict()}, allow_nan=False) + "\n"
                )
                frames.flush()
                last_frame = reading.sequence
            rows.append(row)
            if (
                float(pos[2]) - terrain_ref.height_m < 0.22
                or max(abs(roll), abs(body_pitch)) > 0.85
            ):
                termination = "fallen"
                break
            if max(abs(pos[0]), abs(pos[1])) > 5.5:
                termination = "out_of_bounds"
                break
            state = source.read()
    poses = np.fromfile(directory / "poses.raw", dtype="<f8").reshape(
        len(rows) + 1, 1 + plant.model.nq + plant.model.nv
    )
    with (directory / "states.npz").open("xb") as f:
        np.savez_compressed(
            f,
            time_s=poses[:, 0],
            qpos=poses[:, 1 : 1 + plant.model.nq],
            qvel=poses[:, 1 + plant.model.nq :],
        )
    metrics = case_metrics(rows, terrain, seed, arm, toe, operating_pitch, gain, termination)
    metrics["ramp_x_m"] = toe
    write_json(directory / "metrics.json", metrics)
    return metrics


def execute(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    plan = protocol()
    write_json(output / "protocol.json", plan)
    (output / "cases").mkdir()
    cases = []
    active = None
    try:
        cold = identify_sagittal_model.cache_info()
        if cold.hits or cold.misses or cold.currsize:
            raise ValueError("a fresh process is required for cold identification accounting")
        for terrain in TERRAINS:
            for seed in SEEDS:
                for arm in ARMS:
                    active = f"{terrain}_{seed}_{arm}"
                    metrics = run_case(output / "cases" / active, terrain, seed, arm)
                    cases.append(metrics)
                    info = identify_sagittal_model.cache_info()
                    if (info.hits, info.misses, info.currsize) != (len(cases) - 1, 1, 1):
                        raise RuntimeError("unbudgeted identification call or unexpected cache hit")
        if plan["sources_sha256"] != protocol()["sources_sha256"]:
            raise ValueError("source identity changed during the trial")
        task_steps = sum(c["steps"] for c in cases)
        summary = {
            "schema": plan["schema"],
            "status": "COMPLETE",
            "cases": cases,
            "gate": evaluate(cases),
            "terminal_cases": len(cases),
            "full_horizon_cases": sum(c["full_horizon"] for c in cases),
            "task_controls": task_steps,
            "native_task_steps": 5 * task_steps,
            "total_controls_including_identification": task_steps + 511,
            "total_native_steps_including_identification": 5 * task_steps + 2555,
            "identification_cache": info._asdict(),
            "training_updates": 0,
        }
        write_json(output / "summary.json", summary)
        return summary
    except BaseException as exc:
        write_json(
            output / "partial_summary.json",
            {
                "status": "PARTIAL",
                "active_case": active,
                "completed_cases": cases,
                "exception": repr(exc),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = execute(args.output)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "task_controls": summary["task_controls"],
                "gate_passed": summary["gate"]["passed"],
            }
        )
    )


if __name__ == "__main__":
    main()
