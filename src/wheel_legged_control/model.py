"""MuJoCo plant and local linear model for the planar wheel-legged robot."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

import mujoco
import numpy as np
from scipy.signal import cont2discrete

STATE_SIZE = 6
ACTION_SIZE = 2


@dataclass(frozen=True)
class PlantLimits:
    """Physical and safety limits used by controllers and environments."""

    wheel_force_n: float = 80.0
    leg_force_min_n: float = 0.0
    leg_force_max_n: float = 260.0
    leg_extension_m: float = 0.12
    fall_pitch_rad: float = 0.70


class WheelLeggedPlant:
    """Small MuJoCo wrapper with a six-state control-oriented interface.

    State order is ``[x, pitch, leg_extension, x_dot, pitch_dot, leg_dot]``.
    Control order is ``[equivalent_wheel_force, symmetric_leg_force]``.
    """

    def __init__(self) -> None:
        model_path = files("wheel_legged_control.assets").joinpath("wheel_legged_planar.xml")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.limits = PlantLimits()
        self.control_dt = 0.02
        self.physics_steps = round(self.control_dt / self.model.opt.timestep)

        self._torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        self._nominal_body_mass = self.model.body_mass.copy()
        self._nominal_body_inertia = self.model.body_inertia.copy()
        self._nominal_damping = self.model.dof_damping.copy()
        self._equilibrium_control = self._find_equilibrium_control()

    @property
    def equilibrium_control(self) -> np.ndarray:
        return self._equilibrium_control.copy()

    @property
    def actuator_low(self) -> np.ndarray:
        return self.model.actuator_ctrlrange[:, 0].copy()

    @property
    def actuator_high(self) -> np.ndarray:
        return self.model.actuator_ctrlrange[:, 1].copy()

    def _find_equilibrium_control(self, leg_extension: float = 0.0) -> np.ndarray:
        data = mujoco.MjData(self.model)
        data.qpos[:] = (0.0, 0.0, leg_extension)
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)
        # Both actuators have unit transmission, so the bias forces at their DOFs
        # are the controls that hold the nominal pose.
        return np.asarray(
            (data.qfrc_bias[0], data.qfrc_bias[2]), dtype=np.float64
        )

    def equilibrium_at(self, leg_extension: float) -> np.ndarray:
        return self._find_equilibrium_control(float(leg_extension))

    def state(self) -> np.ndarray:
        return np.concatenate((self.data.qpos[:3], self.data.qvel[:3])).astype(
            np.float64, copy=True
        )

    def set_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (STATE_SIZE,):
            raise ValueError(f"state must have shape ({STATE_SIZE},), got {state.shape}")
        self.data.qpos[:3] = state[:3]
        self.data.qvel[:3] = state[3:]
        self.data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        if state is None:
            state = np.zeros(STATE_SIZE, dtype=np.float64)
        self.set_state(state)
        return self.state()

    def set_domain(self, mass_scale: float = 1.0, damping_scale: float = 1.0) -> None:
        """Apply episode-level model variation, restoring nominal values first."""

        if mass_scale <= 0.0 or damping_scale <= 0.0:
            raise ValueError("domain scales must be positive")
        self.model.body_mass[:] = self._nominal_body_mass
        self.model.body_inertia[:] = self._nominal_body_inertia
        self.model.dof_damping[:] = self._nominal_damping
        self.model.body_mass[self._torso_id] *= mass_scale
        self.model.body_inertia[self._torso_id] *= mass_scale
        self.model.dof_damping[:] *= damping_scale
        mujoco.mj_setConst(self.model, self.data)

    def step(self, control: np.ndarray, push_force_n: float = 0.0) -> np.ndarray:
        control = np.asarray(control, dtype=np.float64)
        if control.shape != (ACTION_SIZE,):
            raise ValueError(f"control must have shape ({ACTION_SIZE},), got {control.shape}")
        self.data.ctrl[:] = np.clip(control, self.actuator_low, self.actuator_high)
        for _ in range(self.physics_steps):
            self.data.xfrc_applied[:] = 0.0
            self.data.xfrc_applied[self._torso_id, 0] = push_force_n
            mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[:] = 0.0
        return self.state()

    def acceleration(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Return generalized acceleration without advancing the simulation."""

        data = mujoco.MjData(self.model)
        data.qpos[:3] = np.asarray(state[:3], dtype=np.float64)
        data.qvel[:3] = np.asarray(state[3:], dtype=np.float64)
        data.ctrl[:] = np.asarray(control, dtype=np.float64)
        mujoco.mj_forward(self.model, data)
        return np.asarray(data.qacc[:3], dtype=np.float64).copy()

    def linearize(self) -> tuple[np.ndarray, np.ndarray]:
        """Numerically linearize and discretize the plant at the upright pose."""

        x0 = np.zeros(STATE_SIZE, dtype=np.float64)
        u0 = self.equilibrium_control

        def derivative(state: np.ndarray, control: np.ndarray) -> np.ndarray:
            return np.concatenate((state[3:], self.acceleration(state, control)))

        a_cont = np.empty((STATE_SIZE, STATE_SIZE), dtype=np.float64)
        b_cont = np.empty((STATE_SIZE, ACTION_SIZE), dtype=np.float64)
        state_eps = np.array([1e-5, 1e-6, 1e-6, 1e-5, 1e-6, 1e-6])
        action_eps = np.array([1e-3, 1e-3])

        for index, epsilon in enumerate(state_eps):
            offset = np.zeros(STATE_SIZE)
            offset[index] = epsilon
            a_cont[:, index] = (
                derivative(x0 + offset, u0) - derivative(x0 - offset, u0)
            ) / (2.0 * epsilon)

        for index, epsilon in enumerate(action_eps):
            offset = np.zeros(ACTION_SIZE)
            offset[index] = epsilon
            b_cont[:, index] = (
                derivative(x0, u0 + offset) - derivative(x0, u0 - offset)
            ) / (2.0 * epsilon)

        c_cont = np.zeros((1, STATE_SIZE))
        d_cont = np.zeros((1, ACTION_SIZE))
        a_disc, b_disc, _, _, _ = cont2discrete(
            (a_cont, b_cont, c_cont, d_cont), self.control_dt
        )
        return np.asarray(a_disc), np.asarray(b_disc)
