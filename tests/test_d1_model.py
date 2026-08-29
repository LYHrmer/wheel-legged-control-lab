import numpy as np

from wheel_legged_control.d1.controllers import D1Command, D1VMCController
from wheel_legged_control.d1.hierarchical import (
    D1_LONGITUDINAL_FORCE_LIMIT_N,
    D1LQRVMCController,
    D1MPCVMCController,
)
from wheel_legged_control.d1.model import D1Plant


def test_d1_model_has_expected_full_body_dimensions() -> None:
    plant = D1Plant()
    summary = plant.summary
    assert (summary.nq, summary.nv, summary.nu) == (23, 22, 16)
    assert summary.body_count == 18
    assert summary.joint_count == 17
    assert summary.geom_count >= 43
    assert np.isclose(summary.total_mass_kg, 48.14686526)


def test_vmc_holds_four_wheel_contact_pose() -> None:
    plant = D1Plant()
    controller = D1VMCController(plant)
    for _ in range(100):
        plant.step(controller.compute(D1Command()))
    assert not plant.has_fallen()
    assert plant.wheel_ground_contacts == 4
    assert plant.undesired_ground_contacts == 0
    assert plant.base_position[2] > 0.40
    assert abs(plant.base_rpy[1]) < np.deg2rad(2.0)


def test_identified_lqr_stabilizes_and_drives_forward() -> None:
    plant = D1Plant()
    controller = D1LQRVMCController(plant)
    assert np.max(np.abs(controller.closed_loop_eigenvalues)) < 1.0
    for _ in range(150):
        plant.step(controller.compute(D1Command(forward_velocity_mps=0.35)))
    assert not plant.has_fallen()
    assert plant.base_position[0] > 0.15
    assert plant.base_velocity(local=False)[0][0] > 0.10


def test_mpc_respects_outer_force_constraint() -> None:
    plant = D1Plant()
    controller = D1MPCVMCController(plant, horizon=10)
    plant.step(controller.compute(D1Command(forward_velocity_mps=0.5)))
    assert abs(controller.last_longitudinal_force_n) <= D1_LONGITUDINAL_FORCE_LIMIT_N
    assert np.isfinite(plant.data.qpos).all()
