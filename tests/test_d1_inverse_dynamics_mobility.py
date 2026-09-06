"""Physical initialization and summary checks for the development mobility runner."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import mujoco
import numpy as np
import pytest

from wheel_legged_control.d1.model import D1Plant


@pytest.fixture(scope="module")
def mobility_runner():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_d1_inverse_dynamics_mobility.py"
    spec = spec_from_file_location("d1_mobility_runner", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", (21, 22, 23))
def test_flat_entry_starts_clear_of_rough_blocks_and_before_ramp(mobility_runner, seed):
    plant = D1Plant(arena="course")
    mobility_runner._reset_plant(plant, "ramp_from_flat", seed)
    model, data = plant.model, plant.data

    # A slightly airborne reset is allowed, but any active terrain contact must
    # be on the floor, not the preceding obstacle lane.
    active_terrain = {
        int(geom)
        for contact in data.contact if int(contact.efc_address) >= 0
        for geom in (contact.geom1, contact.geom2) if int(geom) in plant.terrain_geom_ids
    }
    assert active_terrain.issubset({plant.floor_geom_id})

    def box_extent_x(geom):
        rotation = data.geom_xmat[geom].reshape(3, 3)
        return float(np.abs(rotation[0]) @ model.geom_size[geom])

    rough = [
        geom for geom in plant.terrain_geom_ids
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "")
        .startswith("terrain_rough_")
    ]
    rough_end = max(data.geom_xpos[geom, 0] + box_extent_x(geom) for geom in rough)
    rear_geoms = [
        geom for geom in range(model.ngeom)
        if int(model.geom_bodyid[geom]) in plant.wheel_body_ids_by_leg[2:]
        and model.geom_contype[geom]
    ]
    assert len(rear_geoms) == 2
    for geom in rear_geoms:
        assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER
        rotation = data.geom_xmat[geom].reshape(3, 3)
        radius, half_width = model.geom_size[geom, :2]
        half_extent = radius * np.linalg.norm(rotation[0, :2]) + half_width * abs(rotation[0, 2])
        assert data.geom_xpos[geom, 0] - half_extent > rough_end

    ramp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_ramp_up")
    ramp_entry = data.geom_xpos[ramp, 0] - box_extent_x(ramp)
    assert all(data.xpos[body, 0] < ramp_entry for body in plant.wheel_body_ids_by_leg[:2])


@pytest.mark.parametrize("applied_steps,complete_window", ((10, False), (100, True)))
def test_platform_window_summary_requires_one_full_second(
    mobility_runner, applied_steps, complete_window,
):
    rows = []
    for step in range(applied_steps):
        row = dict.fromkeys(mobility_runner.CSV_FIELDS, "")
        row.update(
            scenario="ramp_to_deck", seed=21, step=step,
            input_time_s=step / 100, output_time_s=(step + 1) / 100,
            applied=True, status="solved", compute_wall_ms=1.0,
            input_base_rpy_rad_yaw=0.0, output_base_rpy_rad_yaw=0.0,
            output_yaw_rate_rps=0.0, output_horizontal_speed_mps=0.0,
            braking_latched=True, braking_started_s=0.0,
            all_wheels_inside_deck=True, all_wheels_touching_deck=True,
            constraint_violation_max=0.0, dynamics_residual_max=0.0,
            torque_fraction_max=0.5, attitude_error_max_rad=0.0,
            clearance_error_m=0.0, height_tracking_error_m=0.0,
            support_fit_valid=True, support_plane_age_s=0.0,
            fallen=False, undesired_ground_contacts=0,
        )
        rows.append(row)
    rejected = dict.fromkeys(mobility_runner.CSV_FIELDS, "")
    rejected.update(
        scenario="ramp_to_deck", seed=21, step=applied_steps,
        input_time_s=applied_steps / 100, output_time_s=applied_steps / 100,
        applied=False, status="rejected", compute_wall_ms=1.0,
        braking_latched=True, braking_started_s=0.0,
        stop_reason="controller_rejected",
    )
    rows.append(rejected)

    summary = mobility_runner._summarize(
        "ramp_to_deck", 21, rows, {"initial_wheel_contacts": 4}, "controller_rejected",
    )

    assert summary["final_one_second_all_wheels_on_deck"] is complete_window
    assert summary["passed"] is False
