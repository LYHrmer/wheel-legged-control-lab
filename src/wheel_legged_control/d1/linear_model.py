"""Contact-aware reduced linear model identified from the full D1 simulation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np

from .controllers import D1Command, D1VMCController
from .model import D1Plant
from .state_estimation import D1MujocoTruthStateSource

SAGITTAL_STATE_SIZE = 4
SAGITTAL_ACTION_SIZE = 1
SAGITTAL_STATE_LABELS = ("x", "pitch", "x_velocity", "pitch_rate")


@dataclass(frozen=True)
class D1SagittalLinearModel:
    """Discrete model around the VMC-supported four-wheel contact pose."""

    a: np.ndarray
    b: np.ndarray
    operating_state: np.ndarray
    affine_drift: np.ndarray
    control_dt: float


def sagittal_state(plant: D1Plant) -> np.ndarray:
    full = plant.reduced_state()
    return full[np.asarray((0, 1, 3, 4), dtype=np.int32)]


def _pitch_perturbed_quaternion(quaternion: np.ndarray, pitch_delta: float) -> np.ndarray:
    delta = np.asarray(
        (np.cos(pitch_delta / 2.0), 0.0, np.sin(pitch_delta / 2.0), 0.0),
        dtype=np.float64,
    )
    result = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(result, delta, quaternion)
    return result


@lru_cache(maxsize=4)
def identify_sagittal_model(control_dt: float = 0.01) -> D1SagittalLinearModel:
    """Numerically identify the D1/VMC one-step dynamics at four-wheel contact."""

    plant = D1Plant(control_dt=control_dt)
    low_level = D1VMCController(plant)
    state_source = D1MujocoTruthStateSource(plant)
    low_level.wheel_velocity_gain = 0.0
    hold = D1Command(base_height_m=plant.nominal_base_height_m)

    state = state_source.reset()
    settle_steps = round(5.0 / control_dt)
    for _ in range(settle_steps):
        plant.step(low_level.compute(hold, state))
        state = state_source.read()
    equilibrium_qpos, equilibrium_qvel = plant.simulation_state()
    equilibrium_qvel[:] = 0.0
    plant.set_simulation_state(equilibrium_qpos, equilibrium_qvel)
    operating_state = sagittal_state(plant)

    def transition(state_delta: np.ndarray, force_n: float) -> np.ndarray:
        qpos = equilibrium_qpos.copy()
        qvel = equilibrium_qvel.copy()
        qpos[0] += state_delta[0]
        qpos[3:7] = _pitch_perturbed_quaternion(qpos[3:7], state_delta[1])
        qvel[0] += state_delta[2]
        qvel[4] += state_delta[3]
        plant.set_simulation_state(qpos, qvel)
        state = state_source.reset()
        torque = low_level.compute(
            hold,
            state,
            longitudinal_force_n=float(force_n),
        )
        plant.step(torque)
        return sagittal_state(plant)

    state_epsilon = np.asarray((1e-4, 1e-4, 1e-3, 1e-3), dtype=np.float64)
    input_epsilon = 0.2
    a = np.empty((SAGITTAL_STATE_SIZE, SAGITTAL_STATE_SIZE), dtype=np.float64)
    for index, epsilon in enumerate(state_epsilon):
        offset = np.zeros(SAGITTAL_STATE_SIZE, dtype=np.float64)
        offset[index] = epsilon
        a[:, index] = (transition(offset, 0.0) - transition(-offset, 0.0)) / (2.0 * epsilon)
    b = (
        transition(np.zeros(SAGITTAL_STATE_SIZE), input_epsilon)
        - transition(np.zeros(SAGITTAL_STATE_SIZE), -input_epsilon)
    ).reshape(-1, 1) / (2.0 * input_epsilon)
    affine_drift = transition(np.zeros(SAGITTAL_STATE_SIZE), 0.0) - operating_state

    for array in (a, b, operating_state, affine_drift):
        array.setflags(write=False)
    return D1SagittalLinearModel(a, b, operating_state, affine_drift, control_dt)
