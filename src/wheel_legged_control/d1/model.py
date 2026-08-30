"""MuJoCo plant for the public 16-DOF D1 wheel-legged model."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

import mujoco
import numpy as np

from .terrain import SUPPORTED_D1_ARENAS, add_d1_terrain

LEG_PREFIXES = ("FL", "FR", "RL", "RR")
D1_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in LEG_PREFIXES
    for joint in ("hip", "thigh", "calf", "foot")
)
D1_LEG_JOINT_NAMES = tuple(name for name in D1_JOINT_NAMES if "_foot_" not in name)
D1_WHEEL_JOINT_NAMES = tuple(name for name in D1_JOINT_NAMES if "_foot_" in name)

NOMINAL_JOINT_POSITION = np.asarray(
    [value for _ in LEG_PREFIXES for value in (0.0, 0.8, -1.5, 0.0)],
    dtype=np.float64,
)
JOINT_TORQUE_LIMIT = np.asarray(
    [value for _ in LEG_PREFIXES for value in (80.0, 80.0, 80.0, 12.0)],
    dtype=np.float64,
)
JOINT_VELOCITY_LIMIT = np.asarray(
    [value for _ in LEG_PREFIXES for value in (20.0, 20.0, 20.0, 30.0)],
    dtype=np.float64,
)
JOINT_POSITION_LOW = np.asarray(
    [value for _ in LEG_PREFIXES for value in (-0.785398, -1.8326, -2.775, -np.inf)],
    dtype=np.float64,
)
JOINT_POSITION_HIGH = np.asarray(
    [value for _ in LEG_PREFIXES for value in (0.785398, 3.40339, -0.855, np.inf)],
    dtype=np.float64,
)


@dataclass(frozen=True)
class D1ModelSummary:
    """Dimensions and mass used as a reproducibility guard."""

    nq: int
    nv: int
    nu: int
    body_count: int
    joint_count: int
    geom_count: int
    total_mass_kg: float


def _asset_urdf_path() -> str:
    path = files("wheel_legged_control.d1.assets").joinpath("urdf", "robot.urdf")
    return str(path)


def build_d1_model(
    *,
    timestep: float = 0.002,
    ground_friction: float = 0.9,
    arena: str = "flat",
) -> mujoco.MjModel:
    """Build a floating-base D1 world from the redistributable URDF.

    MuJoCo's URDF importer intentionally creates a fixed-base, actuator-free
    model.  This function is the single construction seam that adds the free
    base, direct-torque motors, contact parameters, and ground plane required
    for dynamics experiments.
    """

    if timestep <= 0.0:
        raise ValueError("timestep must be positive")
    if ground_friction <= 0.0:
        raise ValueError("ground_friction must be positive")
    if arena not in SUPPORTED_D1_ARENAS:
        raise ValueError(f"arena must be one of {SUPPORTED_D1_ARENAS}")

    spec = mujoco.MjSpec.from_file(_asset_urdf_path())
    spec.modelname = "d1_full_body_control_lab"
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations = 20
    spec.option.ls_iterations = 5

    base = spec.body("base_link")
    # The public training stacks spawn higher and let the robot fall onto the
    # wheels.  A contact-consistent pose is better for deterministic tests.
    base.pos = (0.0, 0.0, 0.455)
    base.add_freejoint(name="floating_base_joint")

    torque_by_name = dict(zip(D1_JOINT_NAMES, JOINT_TORQUE_LIMIT, strict=True))
    velocity_by_name = dict(zip(D1_JOINT_NAMES, JOINT_VELOCITY_LIMIT, strict=True))
    for joint_name in D1_JOINT_NAMES:
        joint = spec.joint(joint_name)
        joint.damping = (0.1, 0.0, 0.0)
        joint.armature = 0.01
        joint.frictionloss = 0.2
        limit = float(torque_by_name[joint_name])
        spec.add_actuator(
            name=joint_name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=joint_name,
            gear=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ctrllimited=1,
            ctrlrange=(-limit, limit),
            forcelimited=1,
            forcerange=(-limit, limit),
            velrange=(-velocity_by_name[joint_name], velocity_by_name[joint_name]),
        )

    for geom in spec.geoms:
        if geom.contype:
            geom.condim = 3
            geom.margin = 0.001
            geom.friction = (ground_friction, 0.005, 0.0001)
            geom.group = 3
            geom.rgba = (0.5, 0.5, 0.5, 0.0)
        else:
            body_name = geom.parent.name
            if body_name.endswith("_foot"):
                geom.rgba = (0.06, 0.08, 0.10, 1.0)
            elif body_name == "base_link":
                geom.rgba = (0.82, 0.86, 0.90, 1.0)
            elif body_name.endswith("_thigh"):
                geom.rgba = (0.17, 0.45, 0.72, 1.0)
            else:
                geom.rgba = (0.72, 0.76, 0.80, 1.0)

    add_d1_terrain(spec, arena, ground_friction)
    spec.worldbody.add_light(
        name="key_light",
        pos=(0.0, -1.5, 3.0),
        dir=(0.0, 0.2, -1.0),
        castshadow=1,
        diffuse=(0.8, 0.8, 0.8),
        specular=(0.2, 0.2, 0.2),
    )
    spec.visual.headlight.ambient = (0.35, 0.35, 0.35)
    spec.visual.headlight.diffuse = (0.75, 0.75, 0.75)
    spec.visual.headlight.specular = (0.15, 0.15, 0.15)
    spec.visual.global_.offwidth = 960
    spec.visual.global_.offheight = 540
    return spec.compile()


class D1Plant:
    """Full-body D1 dynamics with a narrow torque-control interface."""

    nominal_base_height_m = 0.455
    wheel_radius_m = 0.087

    def __init__(
        self,
        *,
        control_dt: float = 0.01,
        ground_friction: float = 0.9,
        arena: str = "flat",
    ) -> None:
        self.model = build_d1_model(ground_friction=ground_friction, arena=arena)
        self.data = mujoco.MjData(self.model)
        ratio = control_dt / self.model.opt.timestep
        if control_dt <= 0.0 or not np.isclose(ratio, round(ratio)):
            raise ValueError("control_dt must be a positive integer multiple of the physics step")
        self.control_dt = float(control_dt)
        self.physics_steps = round(ratio)

        self.joint_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in D1_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids].copy()
        self.dof_addresses = self.model.jnt_dofadr[self.joint_ids].copy()
        self.actuator_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in D1_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.terrain_geom_ids = frozenset(
            geom_id
            for geom_id in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
            .startswith(("floor", "terrain_"))
        )
        self.arena = arena
        self.wheel_body_ids = frozenset(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot")
            for leg in LEG_PREFIXES
        )

        self._nominal_body_mass = self.model.body_mass.copy()
        self._nominal_body_inertia = self.model.body_inertia.copy()
        self._nominal_damping = self.model.dof_damping.copy()
        self._nominal_friction = self.model.geom_friction.copy()
        self._actuator_strength_scale = 1.0
        self.reset()

    @property
    def summary(self) -> D1ModelSummary:
        return D1ModelSummary(
            nq=self.model.nq,
            nv=self.model.nv,
            nu=self.model.nu,
            body_count=self.model.nbody,
            joint_count=self.model.njnt,
            geom_count=self.model.ngeom,
            total_mass_kg=float(self.model.body_mass.sum()),
        )

    @property
    def nominal_total_mass_kg(self) -> float:
        """Return URDF mass before per-episode domain randomization."""

        return float(self._nominal_body_mass.sum())

    @property
    def joint_position(self) -> np.ndarray:
        return self.data.qpos[self.qpos_addresses].copy()

    @property
    def joint_velocity(self) -> np.ndarray:
        return self.data.qvel[self.dof_addresses].copy()

    @property
    def base_position(self) -> np.ndarray:
        return self.data.xpos[self.base_body_id].copy()

    @property
    def base_quaternion(self) -> np.ndarray:
        return self.data.xquat[self.base_body_id].copy()

    @property
    def base_rpy(self) -> np.ndarray:
        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        pitch = np.arcsin(-np.clip(rotation[2, 0], -1.0, 1.0))
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.asarray((roll, pitch, yaw), dtype=np.float64)

    @property
    def projected_gravity_body(self) -> np.ndarray:
        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        return rotation.T @ np.asarray((0.0, 0.0, -1.0), dtype=np.float64)

    def base_velocity(self, *, local: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return base-origin linear and angular velocity.

        MuJoCo's local object velocity follows the body's inertial frame.  The
        controller interface instead defines ``local=True`` in the visible
        ``base_link`` frame, so the world-frame result is projected explicitly.
        """

        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_body_id,
            velocity,
            0,
        )
        linear = velocity[3:].copy()
        angular = velocity[:3].copy()
        if local:
            rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
            linear = rotation.T @ linear
            angular = rotation.T @ angular
        return linear, angular

    def reduced_state(self) -> np.ndarray:
        """Return ``[x, pitch, z, vx, pitch_rate, vz]`` in world coordinates."""

        linear_velocity, angular_velocity = self.base_velocity(local=False)
        return np.asarray(
            (
                self.base_position[0],
                self.base_rpy[1],
                self.base_position[2],
                linear_velocity[0],
                angular_velocity[1],
                linear_velocity[2],
            ),
            dtype=np.float64,
        )

    def simulation_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Copy generalized position and velocity for deterministic replay."""

        return self.data.qpos.copy(), self.data.qvel.copy()

    def set_simulation_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        qpos_array = np.asarray(qpos, dtype=np.float64)
        qvel_array = np.asarray(qvel, dtype=np.float64)
        if qpos_array.shape != (self.model.nq,) or qvel_array.shape != (self.model.nv,):
            raise ValueError("full simulation state has the wrong shape")
        self.data.qpos[:] = qpos_array
        self.data.qvel[:] = qvel_array
        self.data.qacc_warmstart[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @property
    def wheel_ground_contacts(self) -> int:
        contacting_bodies: set[int] = set()
        for contact in self.data.contact:
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 not in self.terrain_geom_ids and geom2 not in self.terrain_geom_ids:
                continue
            other = geom2 if geom1 in self.terrain_geom_ids else geom1
            if int(self.model.geom_bodyid[other]) in self.wheel_body_ids:
                contacting_bodies.add(int(self.model.geom_bodyid[other]))
        return len(contacting_bodies)

    @property
    def undesired_ground_contacts(self) -> int:
        contacting_bodies: set[int] = set()
        for contact in self.data.contact:
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 not in self.terrain_geom_ids and geom2 not in self.terrain_geom_ids:
                continue
            other = geom2 if geom1 in self.terrain_geom_ids else geom1
            body_id = int(self.model.geom_bodyid[other])
            if body_id not in self.wheel_body_ids and body_id != 0:
                contacting_bodies.add(body_id)
        return len(contacting_bodies)

    def reset(
        self,
        *,
        base_position: np.ndarray | None = None,
        base_quaternion: np.ndarray | None = None,
        joint_position: np.ndarray | None = None,
        joint_velocity: np.ndarray | None = None,
    ) -> None:
        mujoco.mj_resetData(self.model, self.data)
        position = (
            np.asarray((0.0, 0.0, self.nominal_base_height_m), dtype=np.float64)
            if base_position is None
            else np.asarray(base_position, dtype=np.float64)
        )
        quaternion = (
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
            if base_quaternion is None
            else np.asarray(base_quaternion, dtype=np.float64)
        )
        joints = (
            NOMINAL_JOINT_POSITION
            if joint_position is None
            else np.asarray(joint_position, dtype=np.float64)
        )
        velocities = (
            np.zeros(len(D1_JOINT_NAMES), dtype=np.float64)
            if joint_velocity is None
            else np.asarray(joint_velocity, dtype=np.float64)
        )
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("base_position and base_quaternion must have shapes (3,) and (4,)")
        if joints.shape != (len(D1_JOINT_NAMES),) or velocities.shape != joints.shape:
            raise ValueError("joint state must contain one value per D1 joint")

        self.data.qpos[:3] = position
        self.data.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
        self.data.qpos[self.qpos_addresses] = joints
        self.data.qvel[:] = 0.0
        self.data.qvel[self.dof_addresses] = velocities
        self.data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_domain(
        self,
        *,
        base_mass_scale: float = 1.0,
        damping_scale: float = 1.0,
        friction_scale: float = 1.0,
        actuator_strength_scale: float = 1.0,
    ) -> None:
        scales = (base_mass_scale, damping_scale, friction_scale, actuator_strength_scale)
        if any(scale <= 0.0 for scale in scales):
            raise ValueError("all domain scales must be positive")
        self.model.body_mass[:] = self._nominal_body_mass
        self.model.body_inertia[:] = self._nominal_body_inertia
        self.model.dof_damping[:] = self._nominal_damping
        self.model.geom_friction[:] = self._nominal_friction
        self.model.body_mass[self.base_body_id] *= base_mass_scale
        self.model.body_inertia[self.base_body_id] *= base_mass_scale
        self.model.dof_damping[:] *= damping_scale
        self.model.geom_friction[:, 0] *= friction_scale
        self._actuator_strength_scale = float(actuator_strength_scale)
        mujoco.mj_setConst(self.model, self.data)

    def step(
        self,
        torque_nm: np.ndarray,
        *,
        push_force_world_n: np.ndarray | None = None,
        push_torque_world_nm: np.ndarray | None = None,
    ) -> None:
        torque = np.asarray(torque_nm, dtype=np.float64)
        if torque.shape != (len(D1_JOINT_NAMES),):
            raise ValueError(f"torque must have shape ({len(D1_JOINT_NAMES)},)")
        applied = np.clip(
            torque,
            -JOINT_TORQUE_LIMIT * self._actuator_strength_scale,
            JOINT_TORQUE_LIMIT * self._actuator_strength_scale,
        )
        force = (
            np.zeros(3, dtype=np.float64)
            if push_force_world_n is None
            else np.asarray(push_force_world_n, dtype=np.float64)
        )
        moment = (
            np.zeros(3, dtype=np.float64)
            if push_torque_world_nm is None
            else np.asarray(push_torque_world_nm, dtype=np.float64)
        )
        if force.shape != (3,) or moment.shape != (3,):
            raise ValueError("push force and torque must have shape (3,)")

        self.data.ctrl[self.actuator_ids] = applied
        for _ in range(self.physics_steps):
            self.data.xfrc_applied[:] = 0.0
            self.data.xfrc_applied[self.base_body_id, :3] = force
            self.data.xfrc_applied[self.base_body_id, 3:] = moment
            mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[:] = 0.0

    def has_fallen(self) -> bool:
        roll, pitch, _ = self.base_rpy
        return bool(
            not np.isfinite(self.data.qpos).all()
            or self.base_position[2] < 0.22
            or abs(roll) > 0.85
            or abs(pitch) > 0.85
        )
