"""Record opt-in, nominal-model D1 inverse-dynamics development rollouts.

Run from an installed checkout, for example:
    python scripts/evaluate_d1_inverse_dynamics.py --output /tmp/d1-idqp-dev

These short, oracle-state runs are development evidence, not a held-out audit or
a promotion gate. They do not change the default controller or existing results.
Output files are steps.csv.gz (gzip-compressed CSV) and summary.json.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.inverse_dynamics import D1InverseDynamicsController
from wheel_legged_control.d1.model import D1_JOINT_NAMES, LEG_PREFIXES, D1Plant
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource
from wheel_legged_control.provenance import capture_git_provenance

CONTROL_DT_S = 0.01
SCENARIOS = ("stand", "drive_brake", "push")
VALIDATION_SEEDS = (21, 22, 23)
VALIDATION_STEPS = 600
VECTOR_FIELDS = {
    "input_base_position_m": ("x", "y", "z"),
    "input_base_rpy_rad": ("roll", "pitch", "yaw"),
    "input_origin_velocity_world_mps": ("x", "y", "z"),
    "input_angular_velocity_world_rps": ("x", "y", "z"),
    "input_joint_position_rad": D1_JOINT_NAMES,
    "input_joint_velocity_rps": D1_JOINT_NAMES,
    "input_wheel_contact": LEG_PREFIXES,
    "command_torque_nm": D1_JOINT_NAMES,
    "predicted_generalized_acceleration": tuple(str(i) for i in range(22)),
    "predicted_contact_force_world_n": tuple(
        f"{leg}_{axis}" for leg in LEG_PREFIXES for axis in ("x", "y", "z")
    ),
    "measured_contact_force_world_n": tuple(
        f"{leg}_{axis}" for leg in LEG_PREFIXES for axis in ("x", "y", "z")
    ),
    "measured_contact_active_sample_fraction": LEG_PREFIXES,
    "measured_contact_point_count_mean": LEG_PREFIXES,
    "output_base_position_m": ("x", "y", "z"),
    "output_base_rpy_rad": ("roll", "pitch", "yaw"),
}
SCALAR_FIELDS = (
    "scenario", "seed", "step", "input_time_s", "output_time_s", "applied", "warm_start",
    "command_forward_velocity_mps", "command_height_m", "push_x_n",
    "input_wheel_contacts", "output_wheel_contacts", "undesired_ground_contacts",
    "output_origin_forward_velocity_mps", "velocity_error_mps", "height_error_m",
    "output_origin_horizontal_speed_mps", "attitude_peak_deg", "height_error_peak_m",
    "horizontal_speed_peak_mps", "input_undesired_ground_contacts", "episode_validation_passed",
    "status", "solver_status", "stop_reason", "error", "solve_ms", "compute_wall_ms",
    "dynamics_residual_max", "constraint_violation_max", "torque_fraction_max",
    "allocated_to_physics_force_error_n", "physics_sample_count",
)
CSV_FIELDS = list(SCALAR_FIELDS) + [
    f"{name}_{component}" for name, components in VECTOR_FIELDS.items() for component in components
]


def _vector(row: dict, name: str, values: np.ndarray) -> None:
    for component, value in zip(VECTOR_FIELDS[name], np.asarray(values).ravel(), strict=True):
        row[f"{name}_{component}"] = float(value)


def _episode(
    scenario: str, seed: int, steps: int, *, warm_start: bool = False
) -> tuple[list[dict], dict]:
    plant = D1Plant(control_dt=CONTROL_DT_S, arena="flat")
    initial_rpy = np.r_[np.random.default_rng(seed).uniform(-0.01, 0.01, 2), 0.0]
    xyzw = Rotation.from_euler("xyz", initial_rpy).as_quat()
    plant.reset(base_quaternion=xyzw[[3, 0, 1, 2]])
    states = D1MujocoTruthStateSource(plant)
    state = states.reset(seed=seed)
    controller = D1InverseDynamicsController(control_dt=CONTROL_DT_S, warm_start=warm_start)
    controller.reset()
    rows: list[dict] = []
    stop_reason = "completed"
    for step in range(steps):
        time_s = step * CONTROL_DT_S
        forward = 0.5 if scenario == "drive_brake" and 1.0 <= time_s < 3.0 else 0.0
        push_x = 120.0 if scenario == "push" and 2.0 <= time_s < 2.12 else 0.0
        command = D1Command(forward_velocity_mps=forward)
        row = dict.fromkeys(CSV_FIELDS, "")
        row.update(
            scenario=scenario, seed=seed, step=step, warm_start=warm_start,
            input_time_s=state.control_time_s,
            output_time_s=state.control_time_s, applied=False,
            command_forward_velocity_mps=forward, command_height_m=command.base_height_m,
            push_x_n=push_x, input_wheel_contacts=state.wheel_ground_contacts,
            input_undesired_ground_contacts=state.undesired_ground_contacts,
            attitude_peak_deg=float(np.rad2deg(np.max(np.abs(state.base_rpy[:2])))),
            height_error_peak_m=float(abs(state.base_position[2] - command.base_height_m)),
            horizontal_speed_peak_mps=float(np.linalg.norm(plant.data.qvel[:2])),
        )
        _vector(row, "input_base_position_m", state.base_position)
        _vector(row, "input_base_rpy_rad", state.base_rpy)
        # Evaluation only: the legacy snapshot's linear velocity is at base COM.
        # Freejoint translation velocity is instead at the base_link origin.
        _vector(row, "input_origin_velocity_world_mps", plant.data.qvel[:3])
        _vector(row, "input_angular_velocity_world_rps", state.base_angular_velocity_world)
        _vector(row, "input_joint_position_rad", state.joint_position)
        _vector(row, "input_joint_velocity_rps", state.joint_velocity)
        row.update({
            f"input_wheel_contact_{leg}": bool(active)
            for leg, active in zip(LEG_PREFIXES, state.wheel_contact, strict=True)
        })
        started = perf_counter()
        try:
            torque = controller.compute(command, state)
            row["compute_wall_ms"] = 1e3 * (perf_counter() - started)
            result = controller.last_result
            if result is None:
                raise RuntimeError("controller returned torque without last_result")
            row.update(
                status=result.status, solver_status=result.solver_status,
                solve_ms=result.solve_ms, dynamics_residual_max=result.dynamics_residual_max,
                constraint_violation_max=result.constraint_violation_max,
            )
            _vector(row, "command_torque_nm", torque)
            _vector(row, "predicted_generalized_acceleration", result.generalized_acceleration)
            _vector(row, "predicted_contact_force_world_n", result.contact_force_world_n)
            if result.status != "solved" or not np.isfinite(torque).all():
                raise RuntimeError(f"rejected result status {result.status!r} or non-finite torque")
        except (RuntimeError, ValueError) as error:
            row["compute_wall_ms"] = 1e3 * (perf_counter() - started)
            row.update(status="rejected", error=f"{type(error).__name__}: {error}")
            stop_reason = "controller_rejected"
        if stop_reason == "controller_rejected":
            row["stop_reason"] = stop_reason
            rows.append(row)
            break

        plant.step(
            torque, push_force_world_n=np.asarray((push_x, 0.0, 0.0)),
            measure_contact_wrench=True,
            contact_wrench_reference_world_m=state.base_position,
        )
        measured = plant.last_control_interval_contact_wrench
        fallen = plant.has_fallen()
        row.update(
            applied=True, output_time_s=float(plant.data.time),
            output_wheel_contacts=plant.wheel_ground_contacts,
            undesired_ground_contacts=plant.undesired_ground_contacts,
            physics_sample_count=measured.physics_sample_count,
            torque_fraction_max=float(np.max(np.abs(torque) / plant.actuator_torque_limit_nm)),
            allocated_to_physics_force_error_n=float(np.linalg.norm(
                result.contact_force_world_n.sum(axis=0) - measured.wheel_force_world_n.sum(axis=0)
            )),
        )
        _vector(row, "measured_contact_force_world_n", measured.wheel_force_world_n)
        _vector(
            row, "measured_contact_active_sample_fraction", measured.active_sample_fraction_by_wheel
        )
        _vector(
            row, "measured_contact_point_count_mean", measured.mean_contact_point_count_by_wheel
        )
        _vector(row, "output_base_position_m", plant.base_position)
        _vector(row, "output_base_rpy_rad", plant.base_rpy)
        yaw = plant.base_rpy[2]
        velocity = float(np.asarray((np.cos(yaw), np.sin(yaw), 0.0)) @ plant.data.qvel[:3])
        row.update(
            output_origin_forward_velocity_mps=velocity,
            output_origin_horizontal_speed_mps=float(np.linalg.norm(plant.data.qvel[:2])),
            velocity_error_mps=velocity - forward,
            height_error_m=float(plant.base_position[2] - command.base_height_m),
            attitude_peak_deg=max(
                row["attitude_peak_deg"], float(np.rad2deg(np.max(np.abs(plant.base_rpy[:2]))))
            ),
            height_error_peak_m=max(
                row["height_error_peak_m"], float(abs(plant.base_position[2] - command.base_height_m))
            ),
            horizontal_speed_peak_mps=max(
                row["horizontal_speed_peak_mps"], float(np.linalg.norm(plant.data.qvel[:2]))
            ),
        )
        if fallen:
            stop_reason = "fallen"
            row["stop_reason"] = stop_reason
        rows.append(row)
        if fallen:
            break
        state = states.read()

    applied = [row for row in rows if row["applied"]]

    def rms(name: str, scale: float = 1.0) -> float | None:
        values = np.asarray([row[name] for row in applied], dtype=np.float64)
        return float(scale * np.sqrt(np.mean(values**2))) if len(values) else None

    def maximum(name: str) -> float | None:
        values = [row[name] for row in rows if row[name] != ""]
        return float(max(values)) if values else None

    elapsed = len(applied) * CONTROL_DT_S
    final_window = applied[-round(1.0 / CONTROL_DT_S):]
    summary = {
        "scenario": scenario, "seed": seed, "warm_start": warm_start,
        "initial_rpy_rad": initial_rpy.tolist(),
        "completed": stop_reason == "completed", "stop_reason": stop_reason,
        "elapsed_s": elapsed, "applied_steps": len(applied), "attempted_steps": len(rows),
        "velocity_rms_mps": rms("velocity_error_mps"),
        "final_forward_velocity_mps": (
            applied[-1]["output_origin_forward_velocity_mps"] if applied else None
        ),
        # Initial yaw is zero, so forward displacement is measured along world x.
        "forward_displacement_m": (
            applied[-1]["output_base_position_m_x"] - rows[0]["input_base_position_m_x"]
            if applied else None
        ),
        "final_velocity_window_s": len(final_window) * CONTROL_DT_S,
        "final_one_second_velocity_rms_mps": (
            float(np.sqrt(np.mean([row["velocity_error_mps"] ** 2 for row in final_window])))
            if final_window else None
        ),
        "final_one_second_horizontal_speed_rms_mps": (
            float(np.sqrt(np.mean([
                row["output_origin_horizontal_speed_mps"] ** 2 for row in final_window
            ]))) if final_window else None
        ),
        "attitude_peak_deg": maximum("attitude_peak_deg"),
        "height_error_peak_m": maximum("height_error_peak_m"),
        "max_horizontal_speed_mps": maximum("horizontal_speed_peak_mps"),
        "max_undesired_ground_contacts": max(
            [row["input_undesired_ground_contacts"] for row in rows]
            + [row["undesired_ground_contacts"] for row in applied]
        ),
        "height_rms_mm": rms("height_error_m", 1000.0),
        "roll_rms_deg": rms("output_base_rpy_rad_roll", 180.0 / np.pi),
        "pitch_rms_deg": rms("output_base_rpy_rad_pitch", 180.0 / np.pi),
        "allocated_to_physics_force_rms_n": rms("allocated_to_physics_force_error_n"),
        "max_dynamics_residual": maximum("dynamics_residual_max"),
        "max_constraint_violation": maximum("constraint_violation_max"),
        "max_torque_fraction": maximum("torque_fraction_max"),
        "compute_wall_p99_ms": float(np.percentile([row["compute_wall_ms"] for row in rows], 99)),
        "accepted_solved_steps": sum(row["status"] == "solved" for row in applied),
        "rejected_steps": sum(row["stop_reason"] == "controller_rejected" for row in rows),
        "four_contact_ratio": (
            float(np.mean([row["output_wheel_contacts"] == 4 for row in applied]))
            if applied else None
        ),
        "wheel_contact_active_fraction_mean": {
            leg: float(np.mean([
                row[f"measured_contact_active_sample_fraction_{leg}"] for row in applied
            ])) if applied else None
            for leg in LEG_PREFIXES
        },
    }
    def below(name: str, threshold: float, *, inclusive: bool = False) -> bool:
        value = summary[name]
        return bool(value is not None and np.isfinite(value)
                    and (value <= threshold if inclusive else value < threshold))

    gates = {
        "six_second_protocol": steps == VALIDATION_STEPS,
        "completed_six_seconds": summary["completed"] and len(applied) == VALIDATION_STEPS,
        "no_rejected_steps": summary["rejected_steps"] == 0,
        "attitude_peak_lt_5_deg": below("attitude_peak_deg", 5.0),
        "height_error_peak_lt_0_04_m": below("height_error_peak_m", .04),
        "no_undesired_contacts": summary["max_undesired_ground_contacts"] == 0,
        "constraint_violation_le_1e_4": below("max_constraint_violation", 1e-4, inclusive=True),
        "torque_fraction_le_1": below("max_torque_fraction", 1.0, inclusive=True),
    }
    if scenario == "stand":
        gates["horizontal_speed_peak_lt_0_15_mps"] = below("max_horizontal_speed_mps", .15)
    if scenario in {"drive_brake", "push"}:
        gates["final_second_speed_rms_lt_0_02_mps"] = (
            len(final_window) == 100 and below("final_one_second_horizontal_speed_rms_mps", .02)
        )
    if scenario == "drive_brake":
        displacement = summary["forward_displacement_m"]
        gates["displacement_between_0_70_and_1_10_m"] = (
            displacement is not None and .70 <= displacement <= 1.10
        )
        gates["velocity_rms_lt_0_10_mps"] = below("velocity_rms_mps", .10)
    if scenario == "push":
        gates["velocity_rms_lt_0_07_mps"] = below("velocity_rms_mps", .07)
    summary["validation_gates"] = gates
    summary["validation_passed"] = all(gates.values())
    for row in rows:
        row["episode_validation_passed"] = summary["validation_passed"]
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(VALIDATION_SEEDS))
    parser.add_argument("--seconds", type=float, default=6.0,
                        help="diagnostics may use other durations, but validation requires 6 seconds")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-start", action="store_true", help="experimental solver configuration")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0.0:
        parser.error("--seconds must be positive and finite")
    steps = round(args.seconds / CONTROL_DT_S)
    if not np.isclose(args.seconds, steps * CONTROL_DT_S) or steps < 1:
        parser.error("--seconds must be an integer multiple of 0.01")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be distinct non-negative integers")
    if any(
        (args.output / name).exists() for name in ("steps.csv.gz", "steps.csv", "summary.json")
    ):
        parser.error("output artifacts already exist; choose a new --output directory")
    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    metadata = {
        "development_only": True, "heldout": False, "promotion_evaluation": False,
        "controller_configuration": {"warm_start": args.warm_start},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": capture_git_provenance(Path(__file__).resolve().parent.parent),
        "runtime": {"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "scipy", "mujoco", "osqp")
        }},
        "protocol": {
            "scenarios": list(scenarios), "seeds": args.seeds, "seconds": args.seconds,
            "control_dt_s": CONTROL_DT_S, "arena": "flat", "state_source": "oracle",
            "domain_randomization": False, "initial_roll_pitch_uniform_rad": [-0.01, 0.01],
            "drive_brake": {"velocity_mps": 0.5, "start_s": 1.0, "brake_s": 3.0},
            "push": {"force_world_n": [120.0, 0.0, 0.0], "start_s": 2.0, "end_s": 2.12},
            "physics_contact_samples_per_step": 5,
            "accepted_controller_status": "solved",
            "validation_requires": {"scenarios": list(SCENARIOS),
                                    "seeds": list(VALIDATION_SEEDS), "seconds": 6.0},
        },
        "notes": [
            "Metrics cover applied steps only; incomplete episodes are not comparable to full runs.",
            "Predicted-to-measured force discrepancy includes soft-contact and model mismatch.",
            "Base linear velocity metrics refer to base_link origin, not inertial COM.",
            "Forward displacement is along the initial horizontal heading, world +x.",
            "Final velocity RMS uses up to the last one second of applied steps; see window length.",
            "Braking speed RMS uses horizontal speed magnitude; velocity RMS uses forward error.",
            "Attitude peaks are max absolute roll/pitch, not yaw; heights use absolute world-z error.",
            "Safety peaks and undesired contacts sample control inputs/endpoints, not all substeps.",
            "Four-contact ratio counts control endpoints; per-wheel fractions use physics samples.",
            "No held-out claim or default-controller promotion follows from these runs.",
        ],
        "episodes": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output / "steps.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for scenario in scenarios:
            for seed in args.seeds:
                rows, summary = _episode(scenario, seed, steps, warm_start=args.warm_start)
                writer.writerows(rows)
                stream.flush()
                metadata["episodes"].append(summary)
                print(f"{scenario} seed={seed}: {summary['stop_reason']} at {summary['elapsed_s']:.2f}s; "
                      f"validation_passed={summary['validation_passed']}")
    protocol_complete = (
        args.seconds == 6.0 and steps == VALIDATION_STEPS and set(scenarios) == set(SCENARIOS)
        and set(args.seeds) == set(VALIDATION_SEEDS)
    )
    metadata["protocol_complete"] = protocol_complete
    metadata["validation_gates"] = {
        "complete_nine_episode_protocol": protocol_complete and len(metadata["episodes"]) == 9,
        "all_episodes_passed": all(item["validation_passed"] for item in metadata["episodes"]),
    }
    metadata["validation_passed"] = all(metadata["validation_gates"].values())
    metadata["validation_status"] = (
        "protocol_incomplete" if not protocol_complete
        else "passed" if metadata["validation_passed"] else "failed"
    )
    metadata["artifacts"] = {
        "steps": {
            "filename": "steps.csv.gz",
            "sha256": hashlib.sha256((args.output / "steps.csv.gz").read_bytes()).hexdigest(),
        }
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Development-only results: {args.output}; validation={metadata['validation_status']}")
    if not metadata["validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
