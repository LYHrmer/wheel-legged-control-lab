"""Single-input-cost D1 baseline and zero-action ramp smoke cases."""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import TypedDict
from zipfile import ZipFile

import numpy as np

from scripts import terrain_preview84_trials as trial
from scripts import terrain_preview85_sensing as sensing
from scripts.terrain_preview85_action import SupportTransferVMC
from scripts.terrain_preview85_common import Episode, baseline_command, physical_sample
from wheel_legged_control.d1.hierarchical import D1_LONGITUDINAL_FORCE_LIMIT_N, D1LQRVMCController
from wheel_legged_control.d1.linear_model import D1SagittalLinearModel
from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.sensor_estimation import D1SensorStateSource
from wheel_legged_control.d1.training_terrain import TrainingTerrainConfig

ARCHIVED_MODEL = trial.ROOT / "docs/assets/project85/recordings.zip"
MODEL_MEMBER = "trials/identified_model.npz"
MODEL_SHA256 = "c2637a4988d1530fdcc5bf81883b3d86d120caf4071d531c48726e5a5b5be66b"
NORMALIZED_INPUT_COST = 1.0 / D1_LONGITUDINAL_FORCE_LIMIT_N**2


def load_model(path: Path = ARCHIVED_MODEL) -> D1SagittalLinearModel:
    """Read the scored round-85 calibration without invoking physical identification."""
    path = Path(path)
    if path.suffix == ".zip":
        with ZipFile(path) as archive:
            data = archive.read(MODEL_MEMBER)
    else:
        data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError("round-85 identified model SHA-256 mismatch")
    with np.load(io.BytesIO(data), allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in ("a", "b", "operating_state", "affine_drift")}
        for array in arrays.values():
            array.setflags(write=False)
        return D1SagittalLinearModel(**arrays, control_dt=float(saved["control_dt"].item()))


class BaselineLQR(D1LQRVMCController):
    """Save the unbounded LQR output on the normal path to the VMC actuator command."""

    def _compose_torque(self, state, command, baseline_force_n, residual_force_n, vertical_feedforward_force_n):
        self.last_unclipped_baseline_force_n = float(baseline_force_n)
        return super()._compose_torque(
            state, command, baseline_force_n, residual_force_n, vertical_feedforward_force_n,
        )


def controller_for(plant, model: D1SagittalLinearModel, *, normalized_input_cost: bool,
                   side_allocator: bool = False) -> BaselineLQR:
    if model is None or model.control_dt != trial.DT or plant.control_dt != trial.DT:
        raise ValueError("an archived 100 Hz model and matching plant are required")
    if type(normalized_input_cost) is not bool or type(side_allocator) is not bool:
        raise ValueError("input-cost and side-allocator choices must be explicit booleans")
    r = np.array([[NORMALIZED_INPUT_COST]]) if normalized_input_cost else None
    controller = BaselineLQR(plant, linear_model=model, r=r)
    low_type = SupportTransferVMC
    if side_allocator:
        from scripts.terrain_axle86 import SideTransferVMC

        low_type = SideTransferVMC
    controller.low_level = low_type(plant)
    controller.low_level.wheel_velocity_gain = 0.0
    return controller


def new_episode(terrain: str, seed: int, model: D1SagittalLinearModel, *,
                normalized_input_cost: bool, side_allocator: bool = False, damping_model=None) -> Episode:
    """Preserve the original terrain, controller timing and delayed IMU/encoder path."""
    if terrain not in ("up", "down"):
        raise ValueError("expected an up or down ramp")
    plant = D1Plant(training_terrain=TrainingTerrainConfig(), sampling_mode="synchronized")
    if plant.physics_steps != 5 or plant.control_dt != trial.DT:
        raise ValueError("expected five native steps per 100 Hz tick")
    toe = trial.front_axle_x(plant) + trial.ENTRY_DISTANCE_M
    trial.install_ramp(plant, toe, trial.TERRAINS[terrain])
    plant.reset(base_position=np.asarray(trial.INITIAL_POSITION))
    controller = controller_for(plant, model, normalized_input_cost=normalized_input_cost,
                                side_allocator=side_allocator)
    source = D1SensorStateSource(
        plant, noise=trial.NOISE_PROFILES["noise"], initial_position=trial.INITIAL_POSITION,
        delay_steps=2,
    )
    ep = Episode(plant, controller, source, source.reset(seed=seed), toe)
    if damping_model is not None:
        from scripts.terrain_damping86 import attach_damping

        attach_damping(ep, damping_model)
    return ep


def range_clock() -> sensing.PairedRangeClock:
    clock = sensing.PairedRangeClock()
    clock.mounts = {"82": clock.mounts["82"]}
    return clock


class BaselineMetrics(TypedDict):
    terrain: str
    seed: int
    normalized_input_cost: bool
    task_steps: int
    termination_reason: str
    full_horizon: bool
    first_front_contact_s: float | None
    crossed_window_s: float | None
    zero_window_complete: bool
    zero_window_pitch_rmse_deg: float | None
    max_progress_after_toe_m: float | None
    sensor_passed: bool
    true_chord_passed: bool
    first_sensor_window_corroborated: bool
    first_confirmation_s: float | None
    fx_limited_ticks: int
    torque_bound_violations: int
    speed_rmse_mps: float
    torque_rms_nm: float
    passed: bool


def score_rows(rows: list[dict], terrain: str, seed: int, toe: float, *,
               normalized_input_cost: bool) -> tuple[BaselineMetrics, dict]:
    """Score the original contact, crossing and three-tick packet predicates from raw ticks."""
    if not rows or terrain not in ("up", "down"):
        raise ValueError("a nonempty up/down trajectory is required")
    termination = rows[-1]["termination_reason"] or "time_limit"
    full = termination == "time_limit" and len(rows) == trial.STEPS
    contact = next((r["time_s"] for r in rows if r["front_contact_x_m"] is not None
                    and r["front_contact_x_m"] >= toe), None)
    after = [r for r in rows if contact is not None and r["time_s"] >= contact]
    crossed = next((r["time_s"] for r in after
                    if r["front_track_x_m"] >= toe + trial.SCORE_DISTANCE_M), None)
    window = [r["pitch_error_rad"] for r in after if crossed is not None and r["time_s"] <= crossed]
    direction = 1 if terrain == "up" else -1
    measured = sensing.summarize_mount(rows, "82", direction, contact)
    chords = sensing.summarize_mount(rows, "82", direction, contact, mode="matched_chords")
    first = sensing.first_window_ground_evidence(rows, "82", direction, measured)
    violations = sum(r["torque_bound_violations"] for r in rows)
    metrics = BaselineMetrics(
        terrain=terrain, seed=seed, normalized_input_cost=normalized_input_cost,
        task_steps=len(rows), termination_reason=termination, full_horizon=full,
        first_front_contact_s=contact, crossed_window_s=crossed, zero_window_complete=crossed is not None,
        zero_window_pitch_rmse_deg=math.degrees(math.sqrt(np.mean(np.square(window)))) if window else None,
        max_progress_after_toe_m=max((r["front_track_x_m"] - toe for r in after), default=None),
        sensor_passed=measured["passed"], true_chord_passed=chords["passed"],
        first_sensor_window_corroborated=bool(first and first["all_three_confirmed"]),
        first_confirmation_s=measured["first_confirmation_s"],
        fx_limited_ticks=sum(abs(r["applied_longitudinal_force_n"]) >= D1_LONGITUDINAL_FORCE_LIMIT_N - 1e-9
                             for r in rows),
        torque_bound_violations=violations,
        speed_rmse_mps=math.sqrt(np.mean([r["speed_error_mps"]**2 for r in rows])),
        torque_rms_nm=math.sqrt(np.mean([r["torque_mean_square_nm2"] for r in rows])),
        passed=full and crossed is not None and measured["passed"] and violations == 0,
    )
    return metrics, {"measured": measured, "matched_ground_chords": chords,
                     "first_registered_window": first}


def run_zero_case(directory: Path, terrain: str, seed: int, model: D1SagittalLinearModel, *,
                  normalized_input_cost: bool, damping_model=None) -> BaselineMetrics:
    """Run exactly one zero-residual case, retaining every tick up to a fall or the time limit."""
    directory.mkdir(parents=True, exist_ok=False)
    trial.write_json(directory / "started.json", {"terrain": terrain, "seed": seed,
                     "normalized_input_cost": normalized_input_cost, "damping_enabled": damping_model is not None,
                     "status_until_metrics": "PARTIAL"})
    ep = new_episode(terrain, seed, model, normalized_input_cost=normalized_input_cost,
                     damping_model=damping_model)
    if damping_model is not None:
        from scripts.terrain_damping86 import record_applied

    trial.write_json(directory / "configuration.json", {
        "model_sha256": MODEL_SHA256, "input_cost": float(ep.controller.r.item()),
        "state_cost": ep.controller.q.tolist(), "gain": ep.controller.gain.tolist(),
        "operating_state": model.operating_state.tolist(), "toe_x_m": ep.toe,
        "fx_limit_n": ep.controller.longitudinal_force_limit_n,
        "joint_torque_limits_nm": ep.plant.actuator_torque_limit_nm.tolist(),
        "proprioceptive_delay_s": 2 * trial.DT, "range_latency_s": sensing.preview.LATENCY_S,
        "nearest_beam_deg": 82, "task_step_limit": trial.STEPS, "damping_enabled": damping_model is not None,
    })
    clock = range_clock()
    rows = []
    with (directory / "captures.jsonl").open("x") as captures, (directory / "ticks.jsonl").open("x") as ticks:
        for _ in range(trial.STEPS):
            state = ep.state
            command = baseline_command(ep)
            readings = clock.observe(ep, command, captures)
            # All simulated truth below is recorded separately from the controller inputs.
            pre_roll, pre_pitch, _ = ep.plant.base_rpy
            pre_speed = float(ep.plant.base_velocity(local=True)[0][0])
            torque = ep.controller.compute(command, state)
            ep.plant.step(torque)
            physical = physical_sample(ep, torque)
            applied = ep.plant.data.ctrl[ep.plant.actuator_ids]
            if damping_model is not None:
                record_applied(ep, applied, physical["time_s"])
            speed = float(ep.plant.base_velocity(local=True)[0][0])
            memory = ep.controller.control_memory
            estimate_roll, estimate_pitch, estimate_yaw = state.base_rpy
            row = {
                **physical, "control_s": float(state.control_time_s), "mounts": readings,
                "estimator_time_s": float(state.measurement_time_s),
                "estimator_age_s": float(state.control_time_s - state.measurement_time_s),
                "estimated_roll_rad": float(estimate_roll), "estimated_pitch_rad": float(estimate_pitch),
                "estimated_yaw_rad": float(estimate_yaw), "estimated_base_x_m": float(state.base_position[0]),
                "estimated_base_z_m": float(state.base_position[2]),
                "estimated_speed_mps": float(state.base_linear_velocity_body[0]),
                "estimated_pitch_rate_rps": float(state.base_angular_velocity_body[1]),
                "command_roll_rad": command.roll_rad, "command_pitch_rad": command.pitch_rad,
                "command_height_m": command.base_height_m, "command_speed_mps": command.forward_velocity_mps,
                "pre_truth_roll_rad": float(pre_roll), "pre_truth_pitch_rad": float(pre_pitch),
                "pre_truth_speed_mps": pre_speed, "truth_roll_rad": float(ep.plant.base_rpy[0]),
                "truth_speed_mps": speed, "speed_error_mps": speed - command.forward_velocity_mps,
                "lqr_distance_m": memory.distance_m, "lqr_reference_distance_m": memory.distance_reference_m,
                "baseline_force_n": ep.controller.last_unclipped_baseline_force_n,
                "applied_longitudinal_force_n": ep.controller.last_longitudinal_force_n,
                "requested_torque_nm": torque.tolist(), "applied_torque_nm": applied.tolist(),
                "torque_mean_square_nm2": float(np.mean(applied**2)),
            }
            sensing.write_row(ticks, row)
            rows.append(row)
            if physical["termination_reason"]:
                break
            ep.state = ep.source.read()
    metrics, evidence = score_rows(rows, terrain, seed, ep.toe, normalized_input_cost=normalized_input_cost)
    trial.write_json(directory / "sensor_windows.json", evidence)
    trial.write_json(directory / "metrics.json", metrics)
    return metrics
