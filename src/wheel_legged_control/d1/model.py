"""MuJoCo plant for the public 16-DOF D1 wheel-legged model."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

import mujoco
import numpy as np

from .actuator_channel import ActuatorChannel, ActuatorTrace
from .locomotion_terrain import (
    D1LocomotionTerrainConfig,
    add_locomotion_terrain,
    locomotion_ground_reference,
)
from .terrain import SUPPORTED_D1_ARENAS, add_d1_terrain
from .training_terrain import (
    TrainingGroundReference,
    TrainingTerrainConfig,
    add_training_heightfield,
    training_ground_reference,
    update_training_heightfield,
)

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


@dataclass(frozen=True)
class D1MeasuredContactWrench:
    """Wheel-ground reaction sampled from the latest MuJoCo physics step.

    Forces act from the terrain on the robot and are expressed in world axes.
    ``wrench_world`` is ordered ``[Fx, Fy, Fz, Mx, My, Mz]`` and its moment is
    taken about ``reference_position_world_m``.
    """

    reference_position_world_m: np.ndarray
    wheel_force_world_n: np.ndarray
    mean_contact_point_count_by_wheel: np.ndarray
    active_sample_fraction_by_wheel: np.ndarray
    wrench_world: np.ndarray
    physics_sample_count: int = 1

    def __post_init__(self) -> None:
        arrays = (
            ("reference_position_world_m", (3,), np.float64),
            ("wheel_force_world_n", (len(LEG_PREFIXES), 3), np.float64),
            ("mean_contact_point_count_by_wheel", (len(LEG_PREFIXES),), np.float64),
            ("active_sample_fraction_by_wheel", (len(LEG_PREFIXES),), np.float64),
            ("wrench_world", (6,), np.float64),
        )
        for name, shape, dtype in arrays:
            value = np.asarray(getattr(self, name), dtype=dtype).copy()
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.physics_sample_count < 1:
            raise ValueError("physics_sample_count must be positive")
        if np.any(self.mean_contact_point_count_by_wheel < 0.0):
            raise ValueError("mean contact-point counts must be non-negative")
        if np.any(self.active_sample_fraction_by_wheel < 0.0) or np.any(
            self.active_sample_fraction_by_wheel > 1.0
        ):
            raise ValueError("active sample fractions must lie in [0, 1]")


def _asset_urdf_path() -> str:
    path = files("wheel_legged_control.d1.assets").joinpath("urdf", "robot.urdf")
    return str(path)


def build_d1_model(
    *,
    timestep: float = 0.002,
    ground_friction: float = 0.9,
    arena: str = "flat",
    training_terrain: TrainingTerrainConfig | None = None,
    locomotion_terrain: D1LocomotionTerrainConfig | None = None,
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
    if training_terrain is not None:
        if not isinstance(training_terrain, TrainingTerrainConfig):
            raise ValueError("training_terrain must be a TrainingTerrainConfig")
        if arena != "flat":
            raise ValueError("training_terrain requires arena='flat'")
    if locomotion_terrain is not None:
        if not isinstance(locomotion_terrain, D1LocomotionTerrainConfig):
            raise TypeError("locomotion_terrain must be D1LocomotionTerrainConfig")
        if arena != "flat" or training_terrain is not None:
            raise ValueError("locomotion_terrain requires arena='flat' and no training_terrain")

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
    if training_terrain is not None:
        add_training_heightfield(spec)
    if locomotion_terrain is not None:
        add_locomotion_terrain(spec, locomotion_terrain)
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
    model = spec.compile()
    if training_terrain is not None:
        update_training_heightfield(model, training_terrain)
    return model


class D1Plant:
    """Full-body D1 dynamics with a narrow torque-control interface.

    ``legacy_mixed`` preserves archived experiments. New control loops use
    ``synchronized``: post-integration observations are evaluated in a separate
    MjData, so observing the plant cannot perturb the integrator warm start.
    """

    nominal_base_height_m = 0.455
    wheel_radius_m = 0.087

    def __init__(
        self,
        *,
        control_dt: float = 0.01,
        ground_friction: float = 0.9,
        arena: str = "flat",
        training_terrain: TrainingTerrainConfig | None = None,
        sampling_mode: str = "legacy_mixed",
        actuator_channel: ActuatorChannel | None = None,
        locomotion_terrain: D1LocomotionTerrainConfig | None = None,
    ) -> None:
        if sampling_mode not in ("legacy_mixed", "synchronized"):
            raise ValueError("sampling_mode must be 'legacy_mixed' or 'synchronized'")
        self.sampling_mode = sampling_mode
        self.model = build_d1_model(
            ground_friction=ground_friction, arena=arena, training_terrain=training_terrain,
            locomotion_terrain=locomotion_terrain,
        )
        self.training_terrain = training_terrain
        self.locomotion_terrain = locomotion_terrain
        self.data = mujoco.MjData(self.model)
        self._measurement_data = (
            mujoco.MjData(self.model) if sampling_mode == "synchronized" else self.data
        )
        ratio = control_dt / self.model.opt.timestep
        if control_dt <= 0.0 or not np.isclose(ratio, round(ratio)):
            raise ValueError("control_dt must be a positive integer multiple of the physics step")
        self.control_dt = float(control_dt)
        self.physics_steps = round(ratio)
        if actuator_channel is not None:
            if not isinstance(actuator_channel, ActuatorChannel):
                raise TypeError("actuator_channel must be an ActuatorChannel")
            if actuator_channel.config.n_axes != 16 or not np.isclose(
                actuator_channel.config.physics_dt_s, self.model.opt.timestep, rtol=0, atol=1e-12
            ):
                raise ValueError("actuator channel must match 16 axes and the physics timestep")
        self.actuator_channel = actuator_channel
        self._last_actuator_traces: tuple[ActuatorTrace, ...] = ()

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
        self.wheel_body_ids_by_leg = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot")
            for leg in LEG_PREFIXES
        )
        self.wheel_body_ids = frozenset(self.wheel_body_ids_by_leg)
        self._wheel_index_by_body_id = {
            body_id: index for index, body_id in enumerate(self.wheel_body_ids_by_leg)
        }

        self._nominal_body_mass = self.model.body_mass.copy()
        self._nominal_body_inertia = self.model.body_inertia.copy()
        self._nominal_damping = self.model.dof_damping.copy()
        self._nominal_friction = self.model.geom_friction.copy()
        self._actuator_strength_scale = 1.0
        self.reset()

    def set_training_terrain(self, config: TrainingTerrainConfig) -> None:
        """Change elevations without recompiling; call reset before stepping.

        Only plants constructed with ``training_terrain`` support this operation.
        Existing renderer contexts require a heightfield upload after changes.
        """

        if self.training_terrain is None:
            raise ValueError("plant was not constructed with a training terrain")
        update_training_heightfield(self.model, config)
        self.training_terrain = config

    def training_ground_reference(self, x: float, y: float) -> TrainingGroundReference:
        """Return oracle height/pitch/roll from the finite training collision map."""

        return training_ground_reference(self.model, x, y)

    def locomotion_ground_reference(self, x: float, y: float) -> TrainingGroundReference:
        """Read the current fixed 2-D collision road for oracle evaluation only."""

        return locomotion_ground_reference(self.model, x, y)

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
    def measurement_data(self) -> mujoco.MjData:
        """Latest measurement-phase data; only simulator sensor Adapters use it.

        This is not an immutable public state. Controllers consume frozen
        D1StateEstimate objects. In synchronized mode all fields describe the
        current integrated qpos/qvel under the last interval's held input.
        Reads do not refresh, integrate, or change the physics solver state.
        """

        return self._measurement_data

    def refresh_measurements(self) -> None:
        """Refresh after deliberate raw state edits, without advancing time.

        reset/step/set_simulation_state call this automatically. mj_forward
        runs only on the sampling copy and preserves the last applied forces;
        it reports the instantaneous acceleration, not an interval average.
        """

        if self.sampling_mode == "synchronized":
            mujoco.mj_copyData(self._measurement_data, self.model, self.data)
            mujoco.mj_forward(self.model, self._measurement_data)

    @property
    def joint_position(self) -> np.ndarray:
        return self.measurement_data.qpos[self.qpos_addresses].copy()

    @property
    def joint_velocity(self) -> np.ndarray:
        return self.measurement_data.qvel[self.dof_addresses].copy()

    @property
    def base_position(self) -> np.ndarray:
        return self.measurement_data.xpos[self.base_body_id].copy()

    @property
    def base_quaternion(self) -> np.ndarray:
        return self.measurement_data.xquat[self.base_body_id].copy()

    @property
    def base_rpy(self) -> np.ndarray:
        rotation = self.measurement_data.xmat[self.base_body_id].reshape(3, 3)
        pitch = np.arcsin(-np.clip(rotation[2, 0], -1.0, 1.0))
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.asarray((roll, pitch, yaw), dtype=np.float64)

    @property
    def projected_gravity_body(self) -> np.ndarray:
        rotation = self.measurement_data.xmat[self.base_body_id].reshape(3, 3)
        return rotation.T @ np.asarray((0.0, 0.0, -1.0), dtype=np.float64)

    def base_velocity(self, *, local: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return base inertial-COM linear velocity and body angular velocity.

        MuJoCo's local object velocity follows the body's inertial frame.  The
        controller interface instead defines ``local=True`` in the visible
        ``base_link`` frame, so the world-frame result is projected explicitly.
        """

        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.measurement_data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_body_id,
            velocity,
            0,
        )
        linear = velocity[3:].copy()
        angular = velocity[:3].copy()
        if local:
            rotation = self.measurement_data.xmat[self.base_body_id].reshape(3, 3)
            linear = rotation.T @ linear
            angular = rotation.T @ angular
        return linear, angular

    def base_origin_velocity(self, *, local: bool = False) -> np.ndarray:
        """Velocity of the visible origin, not its displaced inertial COM."""

        linear, angular = self.base_velocity(local=False)
        rotation = self.measurement_data.xmat[self.base_body_id].reshape(3, 3)
        offset_world = rotation @ self.model.body_ipos[self.base_body_id]
        origin = linear - np.cross(angular, offset_world)
        return rotation.T @ origin if local else origin

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
        self.refresh_measurements()

    @property
    def wheel_ground_contacts(self) -> int:
        contacting_bodies: set[int] = set()
        for contact in self.measurement_data.contact:
            if int(contact.efc_address) < 0:
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 not in self.terrain_geom_ids and geom2 not in self.terrain_geom_ids:
                continue
            other = geom2 if geom1 in self.terrain_geom_ids else geom1
            if int(self.model.geom_bodyid[other]) in self.wheel_body_ids:
                contacting_bodies.add(int(self.model.geom_bodyid[other]))
        return len(contacting_bodies)

    def measure_wheel_contact_wrench(
        self,
        reference_position_world_m: np.ndarray | None = None,
    ) -> D1MeasuredContactWrench:
        """Measure the published wheel-ground wrench without controller output.

        MuJoCo reports each contact force in a transposed contact frame and with
        the force acting on ``geom2``.  This method handles both geom orderings,
        rotates the result into world axes, and sums moments at the actual
        contact points. Synchronized mode evaluates it at the integrated state;
        legacy mode retains the final physics substep's pre-integration phase.
        This is not the control-period average stored separately below.
        """

        reference = (
            self.base_position
            if reference_position_world_m is None
            else np.asarray(reference_position_world_m, dtype=np.float64)
        )
        if reference.shape != (3,) or not np.isfinite(reference).all():
            raise ValueError("reference_position_world_m must be a finite shape-(3,) vector")

        return self._measure_wheel_contact_wrench(self.measurement_data, reference)

    def _measure_wheel_contact_wrench(
        self, data: mujoco.MjData, reference: np.ndarray
    ) -> D1MeasuredContactWrench:

        wheel_forces = np.zeros((len(LEG_PREFIXES), 3), dtype=np.float64)
        contact_counts = np.zeros(len(LEG_PREFIXES), dtype=np.int32)
        total_force = np.zeros(3, dtype=np.float64)
        total_moment = np.zeros(3, dtype=np.float64)
        local_wrench = np.empty(6, dtype=np.float64)
        for contact_id, contact in enumerate(data.contact):
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            terrain1 = geom1 in self.terrain_geom_ids
            terrain2 = geom2 in self.terrain_geom_ids
            if terrain1 == terrain2 or int(contact.efc_address) < 0:
                continue
            robot_geom = geom2 if terrain1 else geom1
            robot_body = int(self.model.geom_bodyid[robot_geom])
            wheel_index = self._wheel_index_by_body_id.get(robot_body)
            if wheel_index is None:
                continue

            mujoco.mj_contactForce(self.model, data, contact_id, local_wrench)
            contact_frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            force_world = contact_frame.T @ local_wrench[:3]
            torque_world = contact_frame.T @ local_wrench[3:]
            if robot_geom == geom1:
                force_world = -force_world
                torque_world = -torque_world

            wheel_forces[wheel_index] += force_world
            contact_counts[wheel_index] += 1
            total_force += force_world
            total_moment += np.cross(np.asarray(contact.pos) - reference, force_world)
            total_moment += torque_world

        return D1MeasuredContactWrench(
            reference_position_world_m=reference,
            wheel_force_world_n=wheel_forces,
            mean_contact_point_count_by_wheel=contact_counts,
            active_sample_fraction_by_wheel=contact_counts > 0,
            wrench_world=np.concatenate((total_force, total_moment)),
        )

    @property
    def last_control_interval_contact_wrench(self) -> D1MeasuredContactWrench:
        """Return the latest sampled or physics-substep-averaged wheel wrench."""

        return self._last_control_interval_contact_wrench

    @property
    def last_control_interval_actuator_traces(self) -> tuple[ActuatorTrace, ...]:
        """One immutable torque receipt per 2 ms substep when a channel is used.

        Empty for the historical direct-torque path. Each trace's applied_nm
        is the value actually sent to the model, not the slower policy action.
        """

        return self._last_actuator_traces

    @property
    def undesired_ground_contacts(self) -> int:
        contacting_bodies: set[int] = set()
        for contact in self.measurement_data.contact:
            if int(contact.efc_address) < 0:
                continue
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
        if self.actuator_channel is not None:
            self.actuator_channel.reset()
        self._last_actuator_traces = ()
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
        self.refresh_measurements()
        self._last_control_interval_contact_wrench = self.measure_wheel_contact_wrench()

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
        torque_limit = JOINT_TORQUE_LIMIT * self._actuator_strength_scale
        self.model.actuator_ctrlrange[:, 0] = -torque_limit
        self.model.actuator_ctrlrange[:, 1] = torque_limit
        self.model.actuator_forcerange[:, 0] = -torque_limit
        self.model.actuator_forcerange[:, 1] = torque_limit
        mujoco.mj_setConst(self.model, self.data)
        self.refresh_measurements()

    @property
    def actuator_torque_limit_nm(self) -> np.ndarray:
        """Current per-joint torque limits after domain randomization."""

        limits = JOINT_TORQUE_LIMIT * self._actuator_strength_scale
        limits.setflags(write=False)
        return limits

    def step(
        self,
        torque_nm: np.ndarray,
        *,
        push_force_world_n: np.ndarray | None = None,
        push_torque_world_nm: np.ndarray | None = None,
        measure_contact_wrench: bool = False,
        contact_wrench_reference_world_m: np.ndarray | None = None,
    ) -> None:
        torque = np.asarray(torque_nm, dtype=np.float64)
        if torque.shape != (len(D1_JOINT_NAMES),) or not np.isfinite(torque).all():
            raise ValueError(f"torque must be finite with shape ({len(D1_JOINT_NAMES)},)")
        channel = self.actuator_channel
        if channel is not None:
            # Do not silently clip after the trace: that would make applied_nm
            # lie about the torque entering MuJoCo. Reconfigure between episodes
            # when domain randomization changes the actuator limits.
            limits = channel.config.torque_limit_nm
            if limits is None or not np.array_equal(
                np.broadcast_to(np.asarray(limits), (16,)), self.actuator_torque_limit_nm
            ):
                raise ValueError("actuator channel torque limits must equal current plant limits")
        applied = np.clip(
            torque,
            -self.actuator_torque_limit_nm,
            self.actuator_torque_limit_nm,
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
        if (
            force.shape != (3,) or moment.shape != (3,)
            or not np.isfinite(force).all() or not np.isfinite(moment).all()
        ):
            raise ValueError("push force and torque must be finite with shape (3,)")
        reference = (
            self.base_position
            if contact_wrench_reference_world_m is None
            else np.asarray(contact_wrench_reference_world_m, dtype=np.float64)
        )
        if reference.shape != (3,) or not np.isfinite(reference).all():
            raise ValueError(
                "contact_wrench_reference_world_m must be a finite shape-(3,) vector"
            )

        self.data.ctrl[self.actuator_ids] = applied
        contact_samples: list[D1MeasuredContactWrench] = []
        actuator_traces: list[ActuatorTrace] = []
        for _ in range(self.physics_steps):
            if channel is not None:
                trace = channel.step(torque, self.data.qvel[self.dof_addresses])
                self.data.ctrl[self.actuator_ids] = trace.applied_nm
                actuator_traces.append(trace)
            self.data.xfrc_applied[:] = 0.0
            self.data.xfrc_applied[self.base_body_id, :3] = force
            self.data.xfrc_applied[self.base_body_id, 3:] = moment
            mujoco.mj_step(self.model, self.data)
            if measure_contact_wrench:
                contact_samples.append(self._measure_wheel_contact_wrench(self.data, reference))
        # Capture with the interval's actual held push/torque still present.
        # Clearing before forward would make the IMU omit that applied wrench.
        self.refresh_measurements()
        self.data.xfrc_applied[:] = 0.0
        self._last_actuator_traces = tuple(actuator_traces)
        if not contact_samples:
            contact_samples.append(self._measure_wheel_contact_wrench(self.data, reference))
        sample_count = len(contact_samples)
        self._last_control_interval_contact_wrench = D1MeasuredContactWrench(
            reference_position_world_m=reference,
            wheel_force_world_n=np.mean(
                [sample.wheel_force_world_n for sample in contact_samples], axis=0
            ),
            mean_contact_point_count_by_wheel=np.mean(
                [sample.mean_contact_point_count_by_wheel for sample in contact_samples], axis=0
            ),
            active_sample_fraction_by_wheel=np.mean(
                [sample.active_sample_fraction_by_wheel for sample in contact_samples], axis=0
            ),
            wrench_world=np.mean([sample.wrench_world for sample in contact_samples], axis=0),
            physics_sample_count=sample_count,
        )

    def has_fallen(self) -> bool:
        roll, pitch, _ = self.base_rpy
        return bool(
            not np.isfinite(self.data.qpos).all()
            or self.base_position[2] < 0.22
            or abs(roll) > 0.85
            or abs(pitch) > 0.85
        )
