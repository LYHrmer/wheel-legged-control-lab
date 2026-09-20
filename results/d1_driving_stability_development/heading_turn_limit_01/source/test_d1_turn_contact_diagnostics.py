"""Synthetic force frames and full Jacobian velocities; no physics integration."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import d1_turn_contact_diagnostics as module


@pytest.mark.parametrize("robot_is_geom1", (False, True))
def test_force_ordering_moment_and_velocity_decomposition(monkeypatch, robot_is_geom1):
    contact = SimpleNamespace(geom1=1 if robot_is_geom1 else 0,
                              geom2=0 if robot_is_geom1 else 1,
                              efc_address=0, pos=np.array([1., 0., 0.]),
                              frame=np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]))
    qvel = np.zeros(22)
    qvel[[0, 1, 6, 9]] = [1., .25, 2., 3.]
    plant = SimpleNamespace(model=SimpleNamespace(nv=22, geom_bodyid=np.array([0, 1])),
                            measurement_data=SimpleNamespace(qvel=qvel, contact=[contact], time=2.5),
                            base_position=np.zeros(3), dof_addresses=np.arange(6, 22),
                            terrain_geom_ids={0}, _wheel_index_by_body_id={1: 0})

    def force(model, data, index, output):
        output[:] = [10., 2., 3., 1., 2., 3.]

    def jac(model, data, jp, jr, point, body):
        jp[:] = 0
        if body == 1:
            jp[0, 0], jp[1, 6], jp[2, 9] = 1, 1, 1
        else:
            jp[0, 1] = 1

    monkeypatch.setattr(module.mujoco, "mj_contactForce", force)
    monkeypatch.setattr(module.mujoco, "mj_jac", jac)
    original = qvel.copy()
    result = module.sample_turn_contacts(plant)
    sign = -1 if robot_is_geom1 else 1
    np.testing.assert_array_equal(result["total_wrench_world_6"], sign*np.array([2, 3, 10, 2, -7, 4]))
    assert result["contact_count_by_wheel"] == [1, 0, 0, 0]
    c = result["contacts"][0]
    assert c["robot_point_velocity_world_mps"] == [1., 2., 3.]
    assert c["terrain_point_velocity_world_mps"] == [.25, 0., 0.]
    assert c["tangential_relative_velocity_world_mps"] == [.75, 2., 0.]
    assert c["leg_dof_velocity_world_mps"] == [0., 2., 0.]
    assert c["wheel_dof_velocity_world_mps"] == [0., 0., 3.]
    assert c["free_base_velocity_world_mps"] == [1., 0., 0.]
    assert c["decomposition_residual_norm_mps"] == 0
    np.testing.assert_array_equal(qvel, original)


def test_no_contact_does_not_create_zero_slip_sample():
    plant = SimpleNamespace(model=SimpleNamespace(nv=22),
                            measurement_data=SimpleNamespace(qvel=np.zeros(22), contact=[], time=0.),
                            base_position=np.zeros(3), dof_addresses=np.arange(6, 22),
                            terrain_geom_ids={0}, _wheel_index_by_body_id={1: 0})
    result = module.sample_turn_contacts(plant)
    assert result["contacts"] == []
    assert result["contact_count_by_wheel"] == [0]*4
    assert result["total_wrench_world_6"] == [0.]*6


def test_inactive_or_nonwheel_contacts_are_omitted():
    contacts = [SimpleNamespace(geom1=0, geom2=1, efc_address=-1),
                SimpleNamespace(geom1=0, geom2=2, efc_address=0)]
    plant = SimpleNamespace(model=SimpleNamespace(nv=22, geom_bodyid=np.array([0, 1, 2])),
                            measurement_data=SimpleNamespace(qvel=np.zeros(22), contact=contacts, time=0.),
                            base_position=np.zeros(3), dof_addresses=np.arange(6, 22),
                            terrain_geom_ids={0}, _wheel_index_by_body_id={1: 0})
    assert module.sample_turn_contacts(plant)["contacts"] == []
