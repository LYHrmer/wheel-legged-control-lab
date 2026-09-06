"""Evaluate low-speed turns and an 8-degree slope with the opt-in D1 IDQP.

Run from an installed checkout, for example:
    python scripts/evaluate_d1_inverse_dynamics_mobility.py --output /tmp/d1-idqp-mobility

Contact-state snapshots generate terrain references. Course geometry is used
only for the initial slope pose and independent platform-arrival checks. These
are nominal-model development runs, not a held-out audit or a default promotion.
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

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.inverse_dynamics import D1InverseDynamicsController
from wheel_legged_control.d1.model import D1_JOINT_NAMES, LEG_PREFIXES, D1Plant
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource, D1StateEstimate
from wheel_legged_control.provenance import capture_git_provenance

CONTROL_DT_S = 0.01
CLEARANCE_M = 0.455
PLANE_FILTER_GAIN = 0.12
# Static wheel/terrain geometry leaves the rear wheel cylinders clear of the
# rough-block edge and the front wheel centers before the ramp. The legacy
# course spawn at x=2.75 overlaps the rough lane; it is unchanged elsewhere.
FLAT_ENTRY_POSITION_M = (2.86, 0.0, 0.455)
SCENARIO_SECONDS = {
    "turn_left": 6.0, "turn_right": 6.0, "ramp_stand": 6.0, "ramp_to_deck": 12.0,
    "ramp_from_flat": 20.0,
}
MIDRAMP_SCENARIOS = ("ramp_stand", "ramp_to_deck")
TRAVERSAL_SCENARIOS = ("ramp_to_deck", "ramp_from_flat")
GATES = {
    "constraint_violation_max": 1e-4,
    "torque_fraction_max": 1.0,
    "attitude_error_peak_rad": float(np.deg2rad(5.0)),
    "clearance_error_peak_m": 0.04,
    "turn_signed_yaw_change_rad": [0.4, 0.8],
    "turn_final_yaw_rate_abs_max_rps": 0.02,
    "turn_final_horizontal_speed_rms_mps": 0.02,
    "ramp_final_horizontal_speed_rms_mps": 0.03,
    "final_window_s": 1.0,
    "minimum_deck_braking_observation_s": 2.0,
    "deck_contact_x_bounds_m": [4.62, 5.38],
    "deck_contact_abs_y_max_m": 0.63,
    "deck_contact_height_tolerance_m": 0.005,
}
VECTOR_FIELDS = {
    "input_qpos": tuple(str(i) for i in range(23)),
    "output_qpos": tuple(str(i) for i in range(23)),
    "input_qvel": tuple(str(i) for i in range(22)),
    "output_qvel": tuple(str(i) for i in range(22)),
    "input_base_position_m": ("x", "y", "z"),
    "output_base_position_m": ("x", "y", "z"),
    "input_base_rpy_rad": ("roll", "pitch", "yaw"),
    "output_base_rpy_rad": ("roll", "pitch", "yaw"),
    "output_origin_velocity_world_mps": ("x", "y", "z"),
    "output_angular_velocity_world_rps": ("x", "y", "z"),
    "input_wheel_contact": LEG_PREFIXES,
    "output_wheel_contact": LEG_PREFIXES,
    "input_contact_point_world_m": tuple(
        f"{leg}_{axis}" for leg in LEG_PREFIXES for axis in ("x", "y", "z")
    ),
    "output_contact_point_world_m": tuple(
        f"{leg}_{axis}" for leg in LEG_PREFIXES for axis in ("x", "y", "z")
    ),
    "input_contact_normal_world": tuple(
        f"{leg}_{axis}" for leg in LEG_PREFIXES for axis in ("x", "y", "z")
    ),
    "output_contact_normal_world": tuple(
        f"{leg}_{axis}" for leg in LEG_PREFIXES for axis in ("x", "y", "z")
    ),
    "command_torque_nm": D1_JOINT_NAMES,
    "predicted_generalized_acceleration": tuple(str(i) for i in range(22)),
    "predicted_contact_force_world_n": tuple(
        f"{leg}_{axis}" for leg in LEG_PREFIXES for axis in ("x", "y", "z")
    ),
}
SCALAR_FIELDS = (
    "scenario", "seed", "step", "input_time_s", "output_time_s", "applied",
    "command_forward_velocity_mps", "command_yaw_rate_rps", "command_height_m",
    "command_vertical_velocity_mps", "command_roll_rad", "command_pitch_rad",
    "support_plane_a", "support_plane_b", "support_plane_c", "support_fit_valid",
    "support_fit_rank", "support_fit_residual_max_m", "support_plane_age_s",
    "support_plane_initialized", "braking_latched", "braking_started_s",
    "status", "solver_status", "stop_reason", "error", "solve_ms", "compute_wall_ms",
    "dynamics_residual_max", "constraint_violation_max", "torque_fraction_max",
    "output_wheel_contacts", "undesired_ground_contacts", "fallen",
    "output_horizontal_speed_mps", "height_tracking_error_m", "clearance_error_m",
    "clearance_fit_valid", "attitude_error_max_rad", "output_yaw_rate_rps",
    "all_wheels_inside_deck", "all_wheels_touching_deck",
)
CSV_FIELDS = list(SCALAR_FIELDS) + [
    f"{name}_{component}" for name, components in VECTOR_FIELDS.items() for component in components
]


def _vector(row: dict, name: str, values: np.ndarray) -> None:
    for component, value in zip(VECTOR_FIELDS[name], np.asarray(values).ravel(), strict=True):
        row[f"{name}_{component}"] = float(value)


def _fit_plane(state: D1StateEstimate) -> tuple[np.ndarray | None, int, float | None]:
    """Fit z=a*x+b*y+c from observed contacts, without terrain model access."""
    points = state.wheel_contact_point[state.wheel_contact]
    if len(points) < 3:
        return None, 0, None
    center = points.mean(axis=0)
    slopes, _, rank, _ = np.linalg.lstsq(
        points[:, :2] - center[:2], points[:, 2] - center[2], rcond=None
    )
    if rank != 2:
        return None, int(rank), None
    plane = np.r_[slopes, center[2] - slopes @ center[:2]]
    residual = points[:, 2] - (points[:, :2] @ plane[:2] + plane[2])
    return plane, int(rank), float(np.max(np.abs(residual)))


class _TerrainReference:
    """Filter observed support planes; hold the last plane through contact loss."""

    def __init__(self) -> None:
        self.plane = np.zeros(3)
        self.last_valid_time: float | None = None

    def command(
        self, state: D1StateEstimate, *, forward_mps: float, yaw_rate_rps: float,
    ) -> tuple[D1Command, dict]:
        fitted, rank, residual = _fit_plane(state)
        if fitted is not None:
            if self.last_valid_time is None:
                self.plane = fitted.copy()
            else:
                self.plane += PLANE_FILTER_GAIN * (fitted - self.plane)
            self.last_valid_time = state.control_time_s
        slope_x, slope_y, offset = self.plane
        yaw = float(state.base_rpy[2])
        forward_slope = slope_x * np.cos(yaw) + slope_y * np.sin(yaw)
        lateral_slope = -slope_x * np.sin(yaw) + slope_y * np.cos(yaw)
        command = D1Command(
            forward_velocity_mps=forward_mps,
            yaw_rate_rps=yaw_rate_rps,
            base_height_m=float(self.plane[:2] @ state.base_position[:2] + offset + CLEARANCE_M),
            base_vertical_velocity_mps=float(forward_mps * forward_slope),
            roll_rad=float(np.arctan(lateral_slope)),
            pitch_rad=float(-np.arctan(forward_slope)),
        )
        diagnostics = {
            "support_plane_a": float(slope_x), "support_plane_b": float(slope_y),
            "support_plane_c": float(offset), "support_fit_valid": fitted is not None,
            "support_fit_rank": rank,
            "support_fit_residual_max_m": residual if residual is not None else "",
            "support_plane_initialized": self.last_valid_time is not None,
            "support_plane_age_s": (
                state.control_time_s - self.last_valid_time
                if self.last_valid_time is not None else ""
            ),
        }
        return command, diagnostics


def _reset_plant(plant: D1Plant, scenario: str, seed: int) -> dict:
    tilt = np.random.default_rng(seed).uniform(-0.01, 0.01, 2)
    perturbation = Rotation.from_euler("xyz", (*tilt, 0.0))
    if scenario in MIDRAMP_SCENARIOS:
        # Initialization only: align the nominal stance with the middle of the ramp.
        ramp_id = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_ramp_up")
        ramp_rotation = plant.data.geom_xmat[ramp_id].reshape(3, 3).copy()
        normal = ramp_rotation[:, 2]
        position = plant.data.geom_xpos[ramp_id] + (
            plant.model.geom_size[ramp_id, 2] + 0.450
        ) * normal
        rotation = Rotation.from_matrix(ramp_rotation) * perturbation
    elif scenario == "ramp_from_flat":
        position = np.asarray(FLAT_ENTRY_POSITION_M)
        rotation = perturbation
    else:
        position = np.asarray((0.0, 0.0, plant.nominal_base_height_m))
        rotation = perturbation
    xyzw = rotation.as_quat()
    plant.reset(base_position=position, base_quaternion=xyzw[[3, 0, 1, 2]])
    return {
        "initial_roll_pitch_perturbation_rad": tilt.tolist(),
        "initial_base_position_m": plant.base_position.tolist(),
        "initial_base_rpy_rad": plant.base_rpy.tolist(),
        "initial_wheel_contacts": plant.wheel_ground_contacts,
    }


def _deck_checks(plant: D1Plant, state: D1StateEstimate) -> tuple[bool, bool]:
    """Independent arrival check; no value from here enters command generation."""
    deck_id = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_ramp_deck")
    deck_z = float(plant.data.geom_xpos[deck_id, 2] + plant.model.geom_size[deck_id, 2])
    points = state.wheel_contact_point
    lower_x, upper_x = GATES["deck_contact_x_bounds_m"]
    inside = bool(
        state.wheel_ground_contacts == 4
        and np.all((points[:, 0] >= lower_x) & (points[:, 0] <= upper_x))
        and np.all(np.abs(points[:, 1]) < GATES["deck_contact_abs_y_max_m"])
        and np.all(np.abs(points[:, 2] - deck_z) <= GATES["deck_contact_height_tolerance_m"])
    )
    touching_bodies: set[int] = set()
    for contact in plant.data.contact:
        if int(contact.efc_address) < 0:
            continue
        if int(contact.geom1) == deck_id:
            other = int(contact.geom2)
        elif int(contact.geom2) == deck_id:
            other = int(contact.geom1)
        else:
            continue
        touching_bodies.add(int(plant.model.geom_bodyid[other]))
    return inside, plant.wheel_body_ids.issubset(touching_bodies)


def _summarize(scenario: str, seed: int, rows: list[dict], initial: dict, stop: str) -> dict:
    applied = [row for row in rows if row["applied"]]
    final = applied[-round(GATES["final_window_s"] / CONTROL_DT_S):]

    def maximum(name: str) -> float | None:
        values = [row[name] for row in applied if row[name] != ""]
        return float(max(values)) if values else None

    def absolute_peak(name: str) -> float | None:
        return float(max(abs(row[name]) for row in applied)) if applied else None

    final_speed = (
        float(np.sqrt(np.mean([row["output_horizontal_speed_mps"] ** 2 for row in final])))
        if final else None
    )
    final_yaw_rate = (
        float(max(abs(row["output_yaw_rate_rps"]) for row in final)) if final else None
    )
    yaw_change = None
    if applied:
        yaws = np.unwrap([
            rows[0]["input_base_rpy_rad_yaw"],
            *[row["output_base_rpy_rad_yaw"] for row in applied],
        ])
        yaw_change = float(yaws[-1] - yaws[0])
    brake_start = next((row["braking_started_s"] for row in rows if row["braking_latched"]), None)
    braking_observation = applied[-1]["output_time_s"] - brake_start if (
        applied and brake_start is not None
    ) else 0.0
    summary = {
        "scenario": scenario, "seed": seed, **initial,
        "completed": stop == "completed", "stop_reason": stop,
        "elapsed_s": len(applied) * CONTROL_DT_S,
        "applied_steps": len(applied), "attempted_steps": len(rows),
        "rejected_steps": sum(row["status"] == "rejected" for row in rows),
        "max_constraint_violation": maximum("constraint_violation_max"),
        "max_dynamics_residual": maximum("dynamics_residual_max"),
        "max_torque_fraction": maximum("torque_fraction_max"),
        "attitude_error_peak_rad": maximum("attitude_error_max_rad"),
        "clearance_error_peak_m": absolute_peak("clearance_error_m"),
        "height_tracking_error_peak_m": absolute_peak("height_tracking_error_m"),
        "final_window_s": len(final) * CONTROL_DT_S,
        "final_horizontal_speed_rms_mps": final_speed,
        "final_yaw_rate_abs_max_rps": final_yaw_rate,
        "yaw_change_rad": yaw_change,
        "braking_started_s": brake_start,
        "braking_observation_s": float(braking_observation),
        "final_one_second_all_wheels_on_deck": (
            len(final) == round(GATES["final_window_s"] / CONTROL_DT_S)
            and all(row["all_wheels_inside_deck"] and row["all_wheels_touching_deck"] for row in final)
        ),
        "support_fit_valid_fraction": (
            float(np.mean([row["support_fit_valid"] for row in applied])) if applied else None
        ),
        "support_plane_age_max_s": maximum("support_plane_age_s"),
        "compute_wall_p50_ms": float(np.median([row["compute_wall_ms"] for row in rows])),
        "compute_wall_p99_ms": float(np.percentile([row["compute_wall_ms"] for row in rows], 99)),
    }
    checks = {
        "completed": stop == "completed",
        "no_rejected_solves": summary["rejected_steps"] == 0,
        "no_fall": bool(applied) and not any(row["fallen"] for row in applied),
        "no_nonwheel_ground_contact": bool(applied) and not any(
            row["undesired_ground_contacts"] for row in applied
        ),
        "constraint_budget": bool(applied) and (
            summary["max_constraint_violation"] <= GATES["constraint_violation_max"]
        ),
        "torque_budget": bool(applied) and summary["max_torque_fraction"] <= 1.0,
        "attitude_error": bool(applied) and (
            summary["attitude_error_peak_rad"] < GATES["attitude_error_peak_rad"]
        ),
        "clearance_error": bool(applied) and (
            summary["clearance_error_peak_m"] < GATES["clearance_error_peak_m"]
        ),
        "complete_final_window": len(final) == round(GATES["final_window_s"] / CONTROL_DT_S),
        "final_speed": bool(final) and final_speed < GATES[
            "turn_final_horizontal_speed_rms_mps" if scenario.startswith("turn_")
            else "ramp_final_horizontal_speed_rms_mps"
        ],
    }
    if scenario.startswith("turn_"):
        sign = 1 if scenario == "turn_left" else -1
        lower_yaw, upper_yaw = GATES["turn_signed_yaw_change_rad"]
        checks["height_tracking_error"] = bool(applied) and (
            summary["height_tracking_error_peak_m"] < GATES["clearance_error_peak_m"]
        )
        checks["signed_yaw_change"] = yaw_change is not None and (
            lower_yaw <= sign * yaw_change <= upper_yaw
        )
        checks["final_yaw_rate"] = bool(final) and (
            final_yaw_rate < GATES["turn_final_yaw_rate_abs_max_rps"]
        )
    elif scenario in MIDRAMP_SCENARIOS:
        checks["initial_four_wheel_contacts"] = initial["initial_wheel_contacts"] == 4
    if scenario in TRAVERSAL_SCENARIOS:
        checks["braking_observed"] = (
            braking_observation >= GATES["minimum_deck_braking_observation_s"]
        )
        checks["four_wheels_on_platform"] = summary["final_one_second_all_wheels_on_deck"]
    summary["gate_checks"] = checks
    summary["passed"] = all(checks.values())
    return summary


def _episode(scenario: str, seed: int, writer: csv.DictWriter, stream) -> dict:
    arena = "course" if scenario.startswith("ramp_") else "flat"
    plant = D1Plant(control_dt=CONTROL_DT_S, arena=arena)
    initial = _reset_plant(plant, scenario, seed)
    states = D1MujocoTruthStateSource(plant)
    state = states.reset(seed=seed)
    controller = D1InverseDynamicsController(control_dt=CONTROL_DT_S, warm_start=True)
    controller.reset()
    reference = _TerrainReference()
    rows: list[dict] = []
    stop = "completed"
    brake_start: float | None = None
    for step in range(round(SCENARIO_SECONDS[scenario] / CONTROL_DT_S)):
        time_s = step * CONTROL_DT_S
        forward, yaw_rate = 0.0, 0.0
        if scenario.startswith("turn_") and 1.0 <= time_s < 4.0:
            forward = 0.3
            yaw_rate = 0.2 if scenario == "turn_left" else -0.2
        if scenario in TRAVERSAL_SCENARIOS:
            if brake_start is None and state.base_position[0] >= 4.9:
                brake_start = time_s
            forward = 0.25 if time_s >= 1.0 and brake_start is None else 0.0
        command, support = reference.command(state, forward_mps=forward, yaw_rate_rps=yaw_rate)
        row = dict.fromkeys(CSV_FIELDS, "")
        row.update(
            scenario=scenario, seed=seed, step=step, input_time_s=state.control_time_s,
            output_time_s=state.control_time_s, applied=False, **support,
            command_forward_velocity_mps=command.forward_velocity_mps,
            command_yaw_rate_rps=command.yaw_rate_rps, command_height_m=command.base_height_m,
            command_vertical_velocity_mps=command.base_vertical_velocity_mps,
            command_roll_rad=command.roll_rad, command_pitch_rad=command.pitch_rad,
            braking_latched=brake_start is not None,
            braking_started_s=brake_start if brake_start is not None else "",
        )
        input_qpos, input_qvel = plant.simulation_state()
        _vector(row, "input_qpos", input_qpos)
        _vector(row, "input_qvel", input_qvel)
        _vector(row, "input_base_position_m", state.base_position)
        _vector(row, "input_base_rpy_rad", state.base_rpy)
        _vector(row, "input_wheel_contact", state.wheel_contact)
        _vector(row, "input_contact_point_world_m", state.wheel_contact_point)
        _vector(row, "input_contact_normal_world", state.wheel_contact_normal)
        started = perf_counter()
        try:
            if scenario in MIDRAMP_SCENARIOS and not support["support_plane_initialized"]:
                raise ValueError("slope initialization needs at least three non-collinear contacts")
            torque = controller.compute(command, state)
            row["compute_wall_ms"] = 1e3 * (perf_counter() - started)
            result = controller.last_result
            if result is None or result.status != "solved" or not np.isfinite(torque).all():
                raise RuntimeError("controller returned no accepted finite result")
            row.update(
                status=result.status, solver_status=result.solver_status, solve_ms=result.solve_ms,
                dynamics_residual_max=result.dynamics_residual_max,
                constraint_violation_max=result.constraint_violation_max,
            )
            _vector(row, "command_torque_nm", torque)
            _vector(row, "predicted_generalized_acceleration", result.generalized_acceleration)
            _vector(row, "predicted_contact_force_world_n", result.contact_force_world_n)
        except (RuntimeError, ValueError) as error:
            row["compute_wall_ms"] = 1e3 * (perf_counter() - started)
            row.update(status="rejected", error=f"{type(error).__name__}: {error}")
            stop = "controller_rejected"
        if stop == "completed":
            plant.step(torque)
            output = states.read()
            output_qpos, output_qvel = plant.simulation_state()
            _vector(row, "output_qpos", output_qpos)
            _vector(row, "output_qvel", output_qvel)
            _vector(row, "output_base_position_m", output.base_position)
            _vector(row, "output_base_rpy_rad", output.base_rpy)
            _vector(row, "output_origin_velocity_world_mps", output_qvel[:3])
            _vector(row, "output_angular_velocity_world_rps", output.base_angular_velocity_world)
            _vector(row, "output_wheel_contact", output.wheel_contact)
            _vector(row, "output_contact_point_world_m", output.wheel_contact_point)
            _vector(row, "output_contact_normal_world", output.wheel_contact_normal)
            output_plane, _, _ = _fit_plane(output)
            clearance_plane = output_plane if output_plane is not None else reference.plane
            ground_z = clearance_plane[:2] @ output.base_position[:2] + clearance_plane[2]
            attitude_error = output.base_rpy[:2] - np.asarray((command.roll_rad, command.pitch_rad))
            if scenario.startswith("turn_"):
                attitude_error = output.base_rpy[:2]
            inside_deck, touching_deck = (
                _deck_checks(plant, output) if arena == "course" else (False, False)
            )
            row.update(
                applied=True, output_time_s=output.control_time_s,
                output_wheel_contacts=output.wheel_ground_contacts,
                undesired_ground_contacts=output.undesired_ground_contacts,
                fallen=plant.has_fallen(),
                torque_fraction_max=float(np.max(np.abs(torque) / plant.actuator_torque_limit_nm)),
                output_horizontal_speed_mps=float(np.linalg.norm(output_qvel[:2])),
                height_tracking_error_m=float(output.base_position[2] - command.base_height_m),
                clearance_error_m=float(output.base_position[2] - ground_z - CLEARANCE_M),
                clearance_fit_valid=output_plane is not None,
                attitude_error_max_rad=float(np.max(np.abs(attitude_error))),
                output_yaw_rate_rps=float(output.base_angular_velocity_world[2]),
                all_wheels_inside_deck=inside_deck, all_wheels_touching_deck=touching_deck,
            )
            if row["fallen"]:
                stop = "fallen"
            elif row["undesired_ground_contacts"]:
                stop = "nonwheel_ground_contact"
            state = output
        if stop != "completed":
            row["stop_reason"] = stop
        rows.append(row)
        writer.writerow(row)
        stream.flush()
        if stop != "completed":
            break
    return _summarize(scenario, seed, rows, initial, stop)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", nargs="+", choices=("all", *SCENARIO_SECONDS), default=["all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[21, 22, 23])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.scenario)) != len(args.scenario) or (
        "all" in args.scenario and len(args.scenario) != 1
    ):
        parser.error("select distinct scenarios, or all by itself")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be distinct non-negative integers")
    if any((args.output / name).exists() for name in ("steps.csv.gz", "summary.json")):
        parser.error("output artifacts already exist; choose a new --output directory")
    scenarios = list(SCENARIO_SECONDS) if args.scenario == ["all"] else args.scenario
    source = capture_git_provenance(Path(__file__).resolve().parent.parent)
    source["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata = {
        "development_only": True, "heldout": False, "promotion_evaluation": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source": source,
        "runtime": {"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "scipy", "mujoco", "osqp")
        }},
        "protocol": {
            "scenarios": scenarios, "seeds": args.seeds,
            "seconds_by_scenario": {name: SCENARIO_SECONDS[name] for name in scenarios},
            "control_dt_s": CONTROL_DT_S, "state_source": "oracle",
            "arena_by_scenario": {
                name: "course" if name.startswith("ramp_") else "flat" for name in scenarios
            },
            "warm_start": True, "domain_randomization": False,
            "initial_roll_pitch_perturbation_uniform_rad": [-0.01, 0.01],
            "midramp_initialization": {
                "scenarios": list(MIDRAMP_SCENARIOS), "normal_clearance_m": 0.450,
                "pose": "ramp center plus normal clearance; ramp attitude times tilt perturbation",
                "require_initial_four_contacts": True,
            },
            "flat_initialization": {
                "turn_position_m": [0.0, 0.0, 0.455],
                "ramp_from_flat_position_m": list(FLAT_ENTRY_POSITION_M),
                "flat_entry_fixture": (
                    "selected by static wheel/terrain clearance, not rollout success; "
                    "legacy x=2.75 course spawn overlaps the rough-block tail"
                ),
                "pose": "level attitude times tilt perturbation; initial contact count unrestricted",
            },
            "turn": {"forward_mps": 0.3, "yaw_rate_magnitude_rps": 0.2,
                     "start_s": 1.0, "brake_s": 4.0},
            "ramp_traversal": {"scenarios": list(TRAVERSAL_SCENARIOS),
                               "forward_mps": 0.25, "start_s": 1.0,
                               "brake_latch_base_x_m": 4.9},
            "terrain_reference": {
                "source": "state.wheel_contact_point; no terrain labels or geometry",
                "plane": "z=a*x+b*y+c, centered least squares, >=3 contacts and XY rank=2",
                "filter_gain": PLANE_FILTER_GAIN,
                "invalid_fit": "hold last valid plane; never revert to the initial prior",
                "flat_initialization": "known z=0 prior only before the first observed support plane",
                "midramp_initialization": "reject until a support plane is observed",
                "vertical_clearance_m": CLEARANCE_M,
                "world_height_m": "a*base_x+b*base_y+c+vertical_clearance_m",
                "vertical_velocity_mps": "forward_mps*(a*cos(yaw)+b*sin(yaw))",
            },
        },
        "gates": GATES,
        "notes": [
            "Course geometry initializes midramp poses and independently checks arrival only.",
            "Flat-entry uses a geometry-checked spawn clear of rough blocks; its height prior is z=0.",
            "Stairs, jumps, descents, and slope turning are not tested.",
            "Turn attitude is absolute roll/pitch; slope attitude error uses commanded roll/pitch.",
            "Clearance error uses an unfiltered output-contact plane, or the held reference if invalid.",
            "Deck checks require four wheel bodies touching the deck and all four mean contact points inside.",
            "Noisy contacts, domain randomization, and hardware deployment are outside this protocol.",
            "qpos/qvel snapshots are saved for replay; controller inputs remain immutable state estimates.",
            "Compute timing excludes terrain-reference fitting, physics stepping, and CSV logging.",
            "Safety and arrival checks sample control endpoints, not every physics substep.",
            "Yaw-rate metrics are world angular-velocity z, not exact Euler-yaw derivatives.",
        ],
        "episodes": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output / "steps.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for scenario in scenarios:
            for seed in args.seeds:
                summary = _episode(scenario, seed, writer, stream)
                metadata["episodes"].append(summary)
                failed = [name for name, passed in summary["gate_checks"].items() if not passed]
                print(f"{scenario} seed={seed}: {summary['stop_reason']}; failed gates={failed}")
    metadata["passed"] = all(episode["passed"] for episode in metadata["episodes"])
    metadata["artifacts"] = {
        "steps": {"filename": "steps.csv.gz",
                  "sha256": hashlib.sha256((args.output / "steps.csv.gz").read_bytes()).hexdigest()},
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Development-only results: {args.output}")
    if not metadata["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
