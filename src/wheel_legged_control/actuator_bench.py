"""Contact-free, single-axis MuJoCo benches for actuator identification lessons.

All mass and inertia values below are teaching assumptions, not identified D1
hardware properties. ``wheel`` is a free rotor; ``pendulum`` is a hanging load,
not a full D1 leg. Both accept torque directly, without an internal position PD
loop. Position is an unwrapped hinge angle in radians.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral, Real

import mujoco
import numpy as np

WHEEL_RADIUS_M = 0.087
WHEEL_WIDTH_M = 0.04
WHEEL_MASS_KG = 0.5
WHEEL_INERTIA_KG_M2 = 0.5 * WHEEL_MASS_KG * WHEEL_RADIUS_M**2
PENDULUM_LENGTH_M = 0.25
PENDULUM_RADIUS_M = 0.01
PENDULUM_MASS_KG = 1.0
PENDULUM_COM_INERTIA_KG_M2 = (
    PENDULUM_MASS_KG * (3.0 * PENDULUM_RADIUS_M**2 + PENDULUM_LENGTH_M**2) / 12.0
)
PENDULUM_PIVOT_INERTIA_KG_M2 = (
    PENDULUM_COM_INERTIA_KG_M2 + PENDULUM_MASS_KG * (PENDULUM_LENGTH_M / 2.0) ** 2
)
GRAVITY_M_S2 = 9.81
FRICTION_SMOOTHING_RAD_S = 0.05
STRIBECK_VELOCITY_RAD_S = 0.3


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


@dataclass(frozen=True)
class ActuatorParameters:
    """Joint-side inertia [kg m²], damping [N m s/rad], friction [N m], delay.

    ``armature`` adds to the known load inertia. ``coulomb_friction`` is the
    magnitude of a smooth, odd friction law, not a static sticking threshold.
    Delay is in physics steps: zero uses this step's command, one the previous
    step's command. Commands preceding a reset are zero.
    """

    armature: float = 0.01
    damping: float = 0.08
    coulomb_friction: float = 0.025
    delay_steps: int = 0

    def __post_init__(self) -> None:
        for name in ("armature", "damping", "coulomb_friction"):
            value = _finite_scalar(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if (
            isinstance(self.delay_steps, bool)
            or not isinstance(self.delay_steps, Integral)
            or self.delay_steps < 0
        ):
            raise ValueError("delay_steps must be a non-negative integer")
        object.__setattr__(self, "delay_steps", int(self.delay_steps))


@dataclass(frozen=True)
class BenchConfig:
    """Known bench setup: load type, physics timestep [s], torque limit [N m]."""

    kind: str = "wheel"
    dt: float = 0.002
    torque_limit_nm: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in ("wheel", "pendulum"):
            raise ValueError("kind must be 'wheel' or 'pendulum'")
        for name in ("dt", "torque_limit_nm"):
            value = _finite_scalar(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)


class ActuatorBench:
    """A fixed-base rotor or pendulum advanced exactly once per ``step``.

    The optional Stribeck term is extra low-speed friction for hidden-plant
    mismatch experiments. It is intentionally outside ``ActuatorParameters``.
    Friction is evaluated at the beginning of each physics step and held during
    that step; viscous damping uses MuJoCo's implicit-fast integration.
    """

    def __init__(
        self,
        config: BenchConfig,
        parameters: ActuatorParameters,
        *,
        stribeck_friction_nm: float = 0.0,
    ) -> None:
        if not isinstance(config, BenchConfig) or not isinstance(parameters, ActuatorParameters):
            raise TypeError("config and parameters must be BenchConfig and ActuatorParameters")
        self.config = config
        self.parameters = parameters
        self._stribeck = _finite_scalar(stribeck_friction_nm, "stribeck_friction_nm")
        if self._stribeck < 0.0:
            raise ValueError("stribeck_friction_nm must be non-negative")
        self._model = mujoco.MjModel.from_xml_string(self._xml())
        self._data = mujoco.MjData(self._model)
        self._commands: deque[float] = deque()
        self._applied_torque_nm = 0.0
        self.reset()

    def _xml(self) -> str:
        if self.config.kind == "wheel":
            mass = WHEEL_MASS_KG
            com = "0 0 0"
            transverse = mass * (3.0 * WHEEL_RADIUS_M**2 + WHEEL_WIDTH_M**2) / 12.0
            inertia = f"{transverse} {WHEEL_INERTIA_KG_M2} {transverse}"
            geom = (
                f'<geom type="cylinder" size="{WHEEL_RADIUS_M} {WHEEL_WIDTH_M / 2.0}" '
                'quat="0.7071067811865476 0.7071067811865476 0 0"/>'
            )
            gravity = 0.0
        else:
            mass = PENDULUM_MASS_KG
            com = f"0 0 {-PENDULUM_LENGTH_M / 2.0}"
            axial = 0.5 * mass * PENDULUM_RADIUS_M**2
            transverse = PENDULUM_COM_INERTIA_KG_M2
            inertia = f"{transverse} {transverse} {axial}"
            geom = (
                f'<geom type="cylinder" size="{PENDULUM_RADIUS_M}" '
                f'fromto="0 0 0 0 0 {-PENDULUM_LENGTH_M}"/>'
            )
            gravity = GRAVITY_M_S2
        limit = self.config.torque_limit_nm
        return f"""<mujoco model="actuator_bench">
          <compiler angle="radian"/>
          <option timestep="{self.config.dt}" gravity="0 0 {-gravity}"
                  integrator="implicitfast"/>
          <default><geom contype="0" conaffinity="0"/></default>
          <worldbody><body name="load">
            <inertial pos="{com}" mass="{mass}" diaginertia="{inertia}"/>
            <joint name="axis" type="hinge" axis="0 1 0" limited="false"
                   armature="{self.parameters.armature}"
                   damping="{self.parameters.damping}" frictionloss="0"/>
            {geom}
          </body></worldbody>
          <actuator><motor joint="axis" gear="1" ctrllimited="true"
                           ctrlrange="{-limit} {limit}"/></actuator>
        </mujoco>"""

    def reset(self, position: float = 0.0, velocity: float = 0.0) -> None:
        position = _finite_scalar(position, "position")
        velocity = _finite_scalar(velocity, "velocity")
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[0] = position
        self._data.qvel[0] = velocity
        self._commands = deque([0.0] * self.parameters.delay_steps)
        self._applied_torque_nm = 0.0
        mujoco.mj_forward(self._model, self._data)

    @property
    def state(self) -> np.ndarray:
        """Return a fresh ``[position_rad, velocity_rad_s]`` array."""
        return np.asarray([self._data.qpos[0], self._data.qvel[0]], dtype=np.float64)

    @property
    def applied_torque_nm(self) -> float:
        """Clipped and delayed motor torque, excluding friction and gravity.

        This is a synthetic-data diagnostic, not an assumed measured signal
        available to identification from command/encoder logs.
        """
        return self._applied_torque_nm

    def step(self, command_torque_nm: float) -> np.ndarray:
        command = _finite_scalar(command_torque_nm, "command_torque_nm")
        limit = self.config.torque_limit_nm
        self._commands.append(float(np.clip(command, -limit, limit)))
        self._applied_torque_nm = self._commands.popleft()
        velocity = self._data.qvel[0]
        friction = self.parameters.coulomb_friction + self._stribeck * np.exp(
            -((velocity / STRIBECK_VELOCITY_RAD_S) ** 2)
        )
        self._data.qfrc_applied[0] = -friction * np.tanh(velocity / FRICTION_SMOOTHING_RAD_S)
        self._data.ctrl[0] = self._applied_torque_nm
        mujoco.mj_step(self._model, self._data)
        return self.state
