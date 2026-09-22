"""Fixed native plane plus one real 15 mm collision box, independently versioned.

Builder assembly and complete cache initialization mirror the frozen D1 source;
reset, step, measurements, protections and dynamics are inherited unchanged.
No post-compile model substitution and no global factory monkeypatch are used.
"""
from __future__ import annotations

import mujoco
import numpy as np

from wheel_legged_control.d1.locomotion_terrain import (
    LOCOMOTION_MAP_HALF_SIZE_M,
    LOCOMOTION_SPAWN_XY_M,
)
from wheel_legged_control.d1.model import (
    D1_JOINT_NAMES,
    JOINT_TORQUE_LIMIT,
    JOINT_VELOCITY_LIMIT,
    LEG_PREFIXES,
    ActuatorChannel,
    ActuatorTrace,
    D1Plant,
    _asset_urdf_path,
)
from wheel_legged_control.d1.terrain import _add_box, add_d1_terrain
from wheel_legged_control.d1.training_terrain import TrainingGroundReference

SINGLE_STEP_PLANT_SCHEMA = "d1-native-single-15mm-box-plant-v1"
BOX_NAME = "terrain_single_15mm_box"
BOX_HALF_SIZE_M = (0.18, 0.62, 0.0075)
BOX_CENTER_M = (LOCOMOTION_SPAWN_XY_M[0]+0.70, LOCOMOTION_SPAWN_XY_M[1], 0.0075)


def build_single_step_model(*, obstacle_enabled: bool):
    if type(obstacle_enabled) is not bool:
        raise TypeError("obstacle_enabled must be bool")
    timestep, ground_friction = 0.002, 0.9
    spec = mujoco.MjSpec.from_file(_asset_urdf_path())
    spec.modelname = SINGLE_STEP_PLANT_SCHEMA
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations = 20
    spec.option.ls_iterations = 5
    base = spec.body("base_link")
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
    add_d1_terrain(spec, "flat", ground_friction)
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
    if obstacle_enabled:
        _add_box(spec, name=BOX_NAME, position=BOX_CENTER_M, half_size=BOX_HALF_SIZE_M,
                 friction=ground_friction, rgba=(0.45, 0.35, 0.25, 1.0))
    model = spec.compile()
    return model


class D1SingleStepPlant(D1Plant):
    """Fresh model/data and every cache rebound before inherited reset."""

    def __init__(self, *, obstacle_enabled: bool, control_dt=0.01, actuator_channel=None):
        sampling_mode, arena = "synchronized", "flat"
        training_terrain = locomotion_terrain = None
        self.obstacle_enabled = obstacle_enabled
        if sampling_mode not in ("legacy_mixed", "synchronized"):
            raise ValueError("sampling_mode must be 'legacy_mixed' or 'synchronized'")
        self.sampling_mode = sampling_mode
        self.model = build_single_step_model(obstacle_enabled=obstacle_enabled)
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
        self.box_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, BOX_NAME)
        if any(v < 0 for v in (*self.joint_ids, *self.actuator_ids, *self.wheel_body_ids_by_leg,
                               self.base_body_id, self.floor_geom_id)):
            raise RuntimeError("compiled named binding missing")
        self.validate_single_step()

    def validate_single_step(self):
        model = self.model
        floor = self.floor_geom_id
        expected = {floor} | ({self.box_geom_id} if self.obstacle_enabled else set())
        world_collisions = {i for i in range(model.ngeom) if model.geom_bodyid[i] == 0
                            and (model.geom_contype[i] or model.geom_conaffinity[i])}
        if model.nhfield or expected != set(self.terrain_geom_ids) or world_collisions != expected:
            raise RuntimeError("unexpected collision terrain")
        if (model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE
                or not np.array_equal(model.geom_pos[floor], (0., 0., 0.))
                or not np.array_equal(model.geom_quat[floor], (1., 0., 0., 0.))):
            raise RuntimeError("native floor changed")
        if self.obstacle_enabled:
            g = self.box_geom_id
            if (g < 0 or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX
                    or not np.array_equal(model.geom_pos[g], BOX_CENTER_M)
                    or not np.array_equal(model.geom_size[g], BOX_HALF_SIZE_M)
                    or not np.array_equal(model.geom_quat[g], (1., 0., 0., 0.))):
                raise RuntimeError("fixed box geometry changed")
        elif self.box_geom_id != -1:
            raise RuntimeError("box present in disabled condition")
        if self.control_dt != .01 or self.physics_steps != 5 or model.opt.timestep != .002:
            raise RuntimeError("fixed task timing changed")

    @property
    def collision_terrain_metadata(self):
        self.validate_single_step()
        return {"schema": SINGLE_STEP_PLANT_SCHEMA, "obstacle_enabled": self.obstacle_enabled,
                "collision_geometry": "native_plane_plus_box" if self.obstacle_enabled else "native_plane",
                "floor_geom_id": int(self.floor_geom_id), "box_geom_id": int(self.box_geom_id),
                "box_name": BOX_NAME, "box_center_m": list(BOX_CENTER_M),
                "box_half_size_m": list(BOX_HALF_SIZE_M), "heightfield_present": False,
                "terrain_geom_ids": sorted(self.terrain_geom_ids),
                "query_boundary": "closed axis-aligned rectangle; discontinuous vertical sides",
                "world_height_reference_m": .455, "height_semantics": "world_z minus true local ground",
                "control_dt_s": self.control_dt, "native_dt_s": float(self.model.opt.timestep),
                "mujoco_version": mujoco.mj_versionString()}

    def locomotion_ground_reference(self, x, y):
        coordinates = np.asarray((x, y))
        if coordinates.shape != (2,) or coordinates.dtype.kind not in "fiu" or not np.isfinite(coordinates).all():
            raise ValueError("ground query requires finite real coordinates")
        if np.any(np.abs(coordinates) > LOCOMOTION_MAP_HALF_SIZE_M):
            raise ValueError("ground query outside finite reference domain")
        self.validate_single_step()
        center = np.asarray(BOX_CENTER_M[:2]); half = np.asarray(BOX_HALF_SIZE_M[:2])
        inside = self.obstacle_enabled and bool(np.all(coordinates >= center-half) and np.all(coordinates <= center+half))
        return TrainingGroundReference(height_m=.015 if inside else 0., pitch_rad=0., roll_rad=0.)
