"""Shared 100 Hz D1 baseline for the registered sensing and axle-response trials."""

import math
from dataclasses import dataclass

import numpy as np

from scripts import terrain_preview84_trials as trial
from wheel_legged_control.d1.control_primitives import terrain_normal_to_rpy
from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.hierarchical import D1LQRVMCController
from wheel_legged_control.d1.linear_model import identify_sagittal_model
from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.sensor_estimation import D1SensorStateSource
from wheel_legged_control.d1.state_estimation import D1StateEstimate
from wheel_legged_control.d1.training_terrain import TrainingTerrainConfig


@dataclass
class Episode:
    plant: D1Plant
    controller: D1LQRVMCController
    source: D1SensorStateSource
    state: D1StateEstimate
    toe: float


def new_episode(terrain, seed, controller_class=D1LQRVMCController):
    if terrain not in ("up", "down"):
        raise ValueError("round 85 requires an up or down ramp")
    plant = D1Plant(training_terrain=TrainingTerrainConfig(), sampling_mode="synchronized")
    if plant.physics_steps != 5 or plant.control_dt != trial.DT:
        raise ValueError("expected five native steps per 100 Hz tick")
    toe = trial.front_axle_x(plant) + trial.ENTRY_DISTANCE_M
    trial.install_ramp(plant, toe, trial.TERRAINS[terrain])
    plant.reset(base_position=np.asarray(trial.INITIAL_POSITION))
    model = identify_sagittal_model(trial.DT, "legacy")
    controller = controller_class(plant, linear_model=model)
    source = D1SensorStateSource(
        plant, noise=trial.NOISE_PROFILES["noise"], initial_position=trial.INITIAL_POSITION,
        delay_steps=2,
    )
    return Episode(plant, controller, source, source.reset(seed=seed), toe)


def baseline_command(ep):
    """Build the round-84 zero-residual body command using only delayed observations."""
    state = ep.state
    if (abs(state.control_time_s - ep.plant.data.time) > 1e-9 or
            abs(state.measurement_time_s - max(0.0, state.control_time_s - 2 * trial.DT)) > 1e-9):
        raise ValueError("proprioception timestamps disagree with the control clock")
    ground = ep.source.ground_reference(state)
    roll, pitch = terrain_normal_to_rpy(
        -math.tan(ground.pitch_rad), math.tan(ground.roll_rad), state.base_rpy[2]
    )
    return D1Command(
        forward_velocity_mps=trial.commanded_speed(state.control_time_s),
        base_height_m=ground.height_m + trial.INITIAL_POSITION[2],
        roll_rad=roll, pitch_rad=pitch,
    )


def physical_sample(ep, torque=None):
    """Post-step physical truth used exclusively to evaluate the applied control."""
    plant = ep.plant
    pos = plant.base_position
    roll, pitch, yaw = plant.base_rpy
    ground = plant.training_ground_reference(float(pos[0]), float(pos[1]))
    _, slope_pitch = terrain_normal_to_rpy(-math.tan(ground.pitch_rad), math.tan(ground.roll_rad), yaw)
    limits = plant.actuator_torque_limit_nm
    applied = plant.data.ctrl[plant.actuator_ids]
    requested = applied if torque is None else np.asarray(torque)
    if not np.isfinite([*pos, roll, pitch, yaw, *requested, *applied]).all():
        raise FloatingPointError("nonfinite physical state or actuator torque")
    termination = None
    if pos[2] - ground.height_m < 0.22 or max(abs(roll), abs(pitch)) > 0.85:
        termination = "fallen"
    elif max(abs(pos[0]), abs(pos[1])) > 5.5:
        termination = "out_of_bounds"
    return {
        "step": round(plant.data.time / trial.DT),
        "time_s": float(plant.data.time),
        "qpos": plant.data.qpos.tolist(),
        "qvel": plant.data.qvel.tolist(),
        "base_x_m": float(pos[0]),
        "front_track_x_m": trial.front_axle_x(plant),
        "front_contact_x_m": trial.front_contact_x(plant),
        "truth_pitch_rad": float(pitch),
        "ground_pitch_rad": float(slope_pitch),
        "pitch_error_rad": float(pitch - ep.controller.linear_model.operating_state[1] - slope_pitch),
        "saturated_axes": int(np.count_nonzero(np.abs(applied) >= limits - 1e-9)),
        "torque_bound_violations": int(np.count_nonzero(
            np.maximum(np.abs(requested), np.abs(applied)) > limits + 1e-9
        )),
        "termination_reason": termination,
    }
