from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wheel_legged_control.d1.contact_allocation import (
    D1_CANDIDATE_CONSTRAINT_VIOLATION_TOL,
    D1AllocationStatus,
    D1ConstrainedContactAllocator,
    D1ContactAllocationRequest,
    D1LegacyContactAllocator,
    D1WrenchTrackingStatus,
)
from wheel_legged_control.d1.controllers import D1Command, D1VMCController
from wheel_legged_control.d1.env import D1ResidualEnv
from wheel_legged_control.d1.hierarchical import D1LQRVMCController, D1MPCVMCController
from wheel_legged_control.d1.linear_model import identify_sagittal_model
from wheel_legged_control.d1.model import JOINT_TORQUE_LIMIT, D1Plant
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource


def _ideal_four_contact_state():
    plant = D1Plant()
    state = D1MujocoTruthStateSource(plant).reset()
    offsets = np.asarray(
        (
            (0.40, 0.25, -0.45),
            (0.40, -0.25, -0.45),
            (-0.40, 0.25, -0.45),
            (-0.40, -0.25, -0.45),
        )
    )
    jacobian = np.zeros((4, 3, 4))
    jacobian[:, 2, 2] = 0.20
    jacobian[:, 0, 3] = -plant.wheel_radius_m
    return plant, replace(
        state,
        foot_position=state.base_position + offsets + np.asarray((0.0, 0.0, 0.087)),
        wheel_contact=np.ones(4, dtype=np.bool_),
        wheel_contact_point=state.base_position + offsets,
        wheel_contact_normal=np.tile((0.0, 0.0, 1.0), (4, 1)),
        wheel_contact_jacobian=jacobian,
    )


def test_allocator_reports_no_contact_without_inventing_support_force() -> None:
    plant = D1Plant()
    state = D1MujocoTruthStateSource(plant).reset()
    committed_torque = np.linspace(-4.0, 4.0, 16)
    allocator = D1ConstrainedContactAllocator(
        friction_coefficient=0.7,
        normal_force_limit_n=plant.nominal_total_mass_kg * 9.81 * 0.65,
        torque_limit_nm=JOINT_TORQUE_LIMIT,
    )

    result = allocator.solve(
        D1ContactAllocationRequest(
            state=state,
            desired_force_world_n=np.asarray((0.0, 0.0, plant.nominal_total_mass_kg * 9.81)),
            desired_moment_world_nm=np.zeros(3),
            committed_torque_nm=committed_torque,
        )
    )

    assert result.status is D1AllocationStatus.NO_CONTACT
    assert result.wrench_tracking_status is D1WrenchTrackingStatus.LIMITED
    np.testing.assert_array_equal(result.contact_force_world_n, np.zeros((4, 3)))
    np.testing.assert_array_equal(result.contact_torque_nm, np.zeros(16))
    np.testing.assert_array_equal(result.command_torque_nm, committed_torque)
    np.testing.assert_array_equal(result.achieved_wrench_world, np.zeros(6))
    assert result.max_constraint_violation == 0.0


def test_no_contact_reports_committed_torque_limit_violation() -> None:
    plant = D1Plant()
    state = D1MujocoTruthStateSource(plant).reset()
    committed_torque = np.zeros(16)
    committed_torque[0] = 100.0
    allocator = D1ConstrainedContactAllocator(
        friction_coefficient=0.7,
        normal_force_limit_n=300.0,
        torque_limit_nm=JOINT_TORQUE_LIMIT,
    )

    result = allocator.solve(
        D1ContactAllocationRequest(
            state=state,
            desired_force_world_n=np.zeros(3),
            desired_moment_world_nm=np.zeros(3),
            committed_torque_nm=committed_torque,
        )
    )

    assert result.status is D1AllocationStatus.NO_CONTACT
    assert result.max_constraint_violation == pytest.approx(0.25)


def test_allocator_balances_a_symmetric_four_contact_wrench() -> None:
    plant, state = _ideal_four_contact_state()
    weight_n = plant.nominal_total_mass_kg * 9.81
    allocator = D1ConstrainedContactAllocator(
        friction_coefficient=0.7,
        normal_force_limit_n=weight_n * 0.65,
        torque_limit_nm=JOINT_TORQUE_LIMIT,
    )

    result = allocator.solve(
        D1ContactAllocationRequest(
            state=state,
            desired_force_world_n=np.asarray((0.0, 0.0, weight_n)),
            desired_moment_world_nm=np.zeros(3),
            committed_torque_nm=np.zeros(16),
        )
    )

    assert result.status is D1AllocationStatus.CONVERGED
    assert result.wrench_tracking_status is D1WrenchTrackingStatus.TRACKED
    np.testing.assert_allclose(result.contact_force_world_n[:, :2], 0.0, atol=1e-10)
    np.testing.assert_allclose(
        result.contact_force_world_n[:, 2],
        weight_n / 4.0,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        result.achieved_wrench_world,
        np.asarray((0.0, 0.0, weight_n, 0.0, 0.0, 0.0)),
        atol=1e-8,
    )
    np.testing.assert_array_equal(
        result.wrench_reference_position_world_m,
        state.base_position,
    )
    assert np.max(np.abs(result.command_torque_nm)) < np.max(JOINT_TORQUE_LIMIT)


def test_allocator_warm_start_is_observable_and_reset_by_contact_changes() -> None:
    _, state = _ideal_four_contact_state()
    allocator = D1ConstrainedContactAllocator(
        friction_coefficient=0.7,
        normal_force_limit_n=300.0,
        torque_limit_nm=JOINT_TORQUE_LIMIT,
    )
    request = D1ContactAllocationRequest(
        state=state,
        desired_force_world_n=np.asarray((20.0, 0.0, 200.0)),
        desired_moment_world_nm=np.zeros(3),
        committed_torque_nm=np.zeros(16),
    )

    first = allocator.solve(request)
    second = allocator.solve(request)
    changed_state = replace(
        state,
        wheel_contact=np.asarray((True, True, True, False)),
    )
    changed = allocator.solve(replace(request, state=changed_state))
    allocator.reset()
    after_reset = allocator.solve(request)

    assert not first.warm_started
    assert second.warm_started
    assert not changed.warm_started
    assert not after_reset.warm_started


def test_allocator_tracks_longitudinal_force_inside_the_friction_pyramid() -> None:
    _, state = _ideal_four_contact_state()
    allocator = D1ConstrainedContactAllocator(
        friction_coefficient=0.5,
        normal_force_limit_n=300.0,
        torque_limit_nm=JOINT_TORQUE_LIMIT,
    )

    result = allocator.solve(
        D1ContactAllocationRequest(
            state=state,
            desired_force_world_n=np.asarray((80.0, 0.0, 200.0)),
            desired_moment_world_nm=np.asarray((0.0, -36.0, 0.0)),
            committed_torque_nm=np.zeros(16),
        )
    )

    assert result.status is D1AllocationStatus.CONVERGED
    assert result.wrench_tracking_status is D1WrenchTrackingStatus.TRACKED
    np.testing.assert_allclose(
        result.contact_force_world_n,
        np.tile((20.0, 0.0, 50.0), (4, 1)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.achieved_wrench_world,
        result.desired_wrench_world,
        atol=1e-6,
    )
    assert np.all(
        np.abs(result.contact_force_world_n[:, 0])
        <= 0.5 * result.contact_force_world_n[:, 2] + 1e-9
    )


def test_allocator_friction_pyramid_is_inside_the_coulomb_cone() -> None:
    _, state = _ideal_four_contact_state()
    allocator = D1ConstrainedContactAllocator(
        friction_coefficient=0.5,
        normal_force_limit_n=300.0,
        torque_limit_nm=np.full(16, 1e6),
    )

    result = allocator.solve(
        D1ContactAllocationRequest(
            state=state,
            desired_force_world_n=np.asarray((100.0, 100.0, 200.0)),
            desired_moment_world_nm=np.zeros(3),
            committed_torque_nm=np.zeros(16),
        )
    )

    tangential_norm = np.linalg.norm(result.contact_force_world_n[:, :2], axis=1)
    assert np.all(
        tangential_norm <= 0.5 * result.contact_force_world_n[:, 2] + 1e-8
    )


def test_allocator_is_equivariant_to_a_global_yaw_rotation() -> None:
    _, state = _ideal_four_contact_state()
    yaw = np.pi / 2.0
    rotation = np.asarray(
        (
            (np.cos(yaw), -np.sin(yaw), 0.0),
            (np.sin(yaw), np.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    rotated_state = replace(
        state,
        base_position=rotation @ state.base_position,
        base_rotation=rotation @ state.base_rotation,
        foot_position=state.foot_position @ rotation.T,
        foot_jacobian=np.einsum("ab,ibj->iaj", rotation, state.foot_jacobian),
        wheel_contact_point=state.wheel_contact_point @ rotation.T,
        wheel_contact_normal=state.wheel_contact_normal @ rotation.T,
        wheel_contact_jacobian=np.einsum(
            "ab,ibj->iaj", rotation, state.wheel_contact_jacobian
        ),
    )
    force = np.asarray((40.0, -20.0, 320.0))
    moment = np.asarray((6.0, -12.0, 4.0))

    def solve(test_state, desired_force, desired_moment):
        allocator = D1ConstrainedContactAllocator(
            friction_coefficient=0.7,
            normal_force_limit_n=300.0,
            torque_limit_nm=JOINT_TORQUE_LIMIT,
        )
        return allocator.solve(
            D1ContactAllocationRequest(
                state=test_state,
                desired_force_world_n=desired_force,
                desired_moment_world_nm=desired_moment,
                committed_torque_nm=np.zeros(16),
            )
        )

    original = solve(state, force, moment)
    rotated = solve(rotated_state, rotation @ force, rotation @ moment)

    np.testing.assert_allclose(
        rotated.contact_force_world_n,
        original.contact_force_world_n @ rotation.T,
        atol=1e-5,
    )
    np.testing.assert_allclose(rotated.contact_torque_nm, original.contact_torque_nm, atol=1e-5)


def test_vmc_with_constrained_allocator_does_not_invent_airborne_contact_force() -> None:
    plant = D1Plant()
    state = D1MujocoTruthStateSource(plant).reset()
    allocator = D1ConstrainedContactAllocator(
        friction_coefficient=0.7,
        normal_force_limit_n=plant.nominal_total_mass_kg * 9.81 * 0.65,
        torque_limit_nm=JOINT_TORQUE_LIMIT,
    )
    controller = D1VMCController(plant, contact_allocator=allocator)

    torque = controller.compute(
        D1Command(),
        state,
        longitudinal_force_n=100.0,
    )

    assert controller.last_breakdown.allocation_status is D1AllocationStatus.NO_CONTACT
    np.testing.assert_array_equal(
        controller.last_breakdown.contact_force_world_n,
        np.zeros((4, 3)),
    )
    np.testing.assert_array_equal(controller.last_breakdown.support_force_n, np.zeros(4))
    np.testing.assert_array_equal(torque, np.zeros(16))


def test_allocator_rejects_a_non_unit_active_contact_normal() -> None:
    _, state = _ideal_four_contact_state()
    invalid_normal = state.wheel_contact_normal.copy()
    invalid_normal[0] = (0.0, 0.0, 2.0)
    invalid_state = replace(state, wheel_contact_normal=invalid_normal)
    allocator = D1ConstrainedContactAllocator(
        friction_coefficient=0.7,
        normal_force_limit_n=300.0,
        torque_limit_nm=JOINT_TORQUE_LIMIT,
    )

    with pytest.raises(ValueError, match="unit"):
        allocator.solve(
            D1ContactAllocationRequest(
                state=invalid_state,
                desired_force_world_n=np.asarray((0.0, 0.0, 200.0)),
                desired_moment_world_nm=np.zeros(3),
                committed_torque_nm=np.zeros(16),
            )
        )


def test_allocator_rejects_a_nonpositive_wrench_characteristic_length() -> None:
    with pytest.raises(ValueError, match="characteristic_length"):
        D1ConstrainedContactAllocator(
            friction_coefficient=0.7,
            normal_force_limit_n=300.0,
            torque_limit_nm=JOINT_TORQUE_LIMIT,
            wrench_characteristic_length_m=0.0,
        )


@pytest.mark.parametrize("allocator_kind", ["legacy", "constrained"])
def test_allocator_rejects_nonpositive_torque_limits(allocator_kind: str) -> None:
    limits = JOINT_TORQUE_LIMIT.copy()
    limits[3] = 0.0

    with pytest.raises(ValueError, match="torque_limit_nm must be positive"):
        if allocator_kind == "legacy":
            D1LegacyContactAllocator(
                normal_force_limit_n=300.0,
                wheel_radius_m=0.087,
                torque_limit_nm=limits,
            )
        else:
            D1ConstrainedContactAllocator(
                friction_coefficient=0.7,
                normal_force_limit_n=300.0,
                torque_limit_nm=limits,
            )


def test_allocator_rejects_a_contact_normal_pointing_away_from_the_robot() -> None:
    _, state = _ideal_four_contact_state()
    invalid_normal = state.wheel_contact_normal.copy()
    invalid_normal[0] *= -1.0
    invalid_state = replace(state, wheel_contact_normal=invalid_normal)
    allocator = D1ConstrainedContactAllocator(
        friction_coefficient=0.7,
        normal_force_limit_n=300.0,
        torque_limit_nm=JOINT_TORQUE_LIMIT,
    )

    with pytest.raises(ValueError, match="toward the robot"):
        allocator.solve(
            D1ContactAllocationRequest(
                state=invalid_state,
                desired_force_world_n=np.asarray((0.0, 0.0, 200.0)),
                desired_moment_world_nm=np.zeros(3),
                committed_torque_nm=np.zeros(16),
            )
        )


def test_allocator_marks_an_unreachable_wrench_limited_without_breaking_constraints() -> None:
    _, state = _ideal_four_contact_state()
    active = state.wheel_contact.copy()
    active[0] = False
    state = replace(state, wheel_contact=active)
    allocator = D1ConstrainedContactAllocator(
        friction_coefficient=0.4,
        normal_force_limit_n=300.0,
        torque_limit_nm=JOINT_TORQUE_LIMIT,
    )

    result = allocator.solve(
        D1ContactAllocationRequest(
            state=state,
            desired_force_world_n=np.asarray((1000.0, 0.0, 2000.0)),
            desired_moment_world_nm=np.zeros(3),
            committed_torque_nm=np.zeros(16),
        )
    )

    assert result.status in {
        D1AllocationStatus.CONVERGED,
        D1AllocationStatus.FEASIBLE_NONCONVERGED,
    }
    assert result.wrench_tracking_status is D1WrenchTrackingStatus.LIMITED
    np.testing.assert_array_equal(result.contact_force_world_n[0], np.zeros(3))
    normal_force = result.contact_force_world_n[:, 2]
    assert np.all(normal_force >= -1e-8)
    assert np.all(normal_force <= 300.0 + 1e-8)
    force_tolerance_n = 300.0 * D1_CANDIDATE_CONSTRAINT_VIOLATION_TOL
    assert np.all(
        np.abs(result.contact_force_world_n[:, 0])
        <= 0.4 * normal_force + force_tolerance_n
    )
    assert np.all(np.abs(result.command_torque_nm) <= JOINT_TORQUE_LIMIT + 1e-7)
    assert result.max_constraint_violation <= 1e-7


def test_allocator_respects_torque_already_committed_to_joint_control() -> None:
    _, state = _ideal_four_contact_state()
    committed = np.zeros(16)
    committed[np.asarray((2, 6, 10, 14))] = -79.0
    allocator = D1ConstrainedContactAllocator(
        friction_coefficient=0.7,
        normal_force_limit_n=300.0,
        torque_limit_nm=JOINT_TORQUE_LIMIT,
    )

    result = allocator.solve(
        D1ContactAllocationRequest(
            state=state,
            desired_force_world_n=np.asarray((0.0, 0.0, 200.0)),
            desired_moment_world_nm=np.zeros(3),
            committed_torque_nm=committed,
        )
    )

    assert result.status is D1AllocationStatus.CONVERGED
    assert result.wrench_tracking_status is D1WrenchTrackingStatus.LIMITED
    np.testing.assert_allclose(result.contact_force_world_n[:, 2], 5.0, atol=1e-5)
    np.testing.assert_allclose(
        result.command_torque_nm[np.asarray((2, 6, 10, 14))],
        -80.0,
        atol=1e-6,
    )
    assert result.achieved_wrench_world[2] == pytest.approx(20.0, abs=1e-5)
    assert result.max_constraint_violation <= 1e-7


def test_allocator_uses_an_explicit_deterministic_fallback_when_constraints_conflict() -> None:
    _, state = _ideal_four_contact_state()
    state = replace(state, wheel_contact_jacobian=np.zeros((4, 3, 4)))
    committed = np.zeros(16)
    committed[0] = 100.0
    allocator = D1ConstrainedContactAllocator(
        friction_coefficient=0.7,
        normal_force_limit_n=300.0,
        torque_limit_nm=JOINT_TORQUE_LIMIT,
    )
    request = D1ContactAllocationRequest(
        state=state,
        desired_force_world_n=np.asarray((0.0, 0.0, 200.0)),
        desired_moment_world_nm=np.zeros(3),
        committed_torque_nm=committed,
    )

    first = allocator.solve(request)
    allocator.reset()
    second = allocator.solve(request)

    assert first.status is D1AllocationStatus.FALLBACK
    assert first.status_reason == "solver_rejected"
    np.testing.assert_array_equal(first.contact_force_world_n, second.contact_force_world_n)
    np.testing.assert_array_equal(first.command_torque_nm, second.command_torque_nm)
    assert np.all(np.abs(first.command_torque_nm) <= JOINT_TORQUE_LIMIT)
    assert first.max_constraint_violation == pytest.approx(0.25)


def test_fallback_wrench_matches_the_force_that_fits_remaining_torque(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, state = _ideal_four_contact_state()
    committed = np.zeros(16)
    committed[np.asarray((2, 6, 10, 14))] = -79.0
    monkeypatch.setattr(
        "wheel_legged_control.d1.contact_allocation.minimize",
        lambda *args, **kwargs: SimpleNamespace(
            x=np.full(12, np.nan),
            success=False,
            nit=1,
            status=4,
            message="Inequality constraints incompatible",
        ),
    )
    allocator = D1ConstrainedContactAllocator(
        friction_coefficient=0.7,
        normal_force_limit_n=300.0,
        torque_limit_nm=JOINT_TORQUE_LIMIT,
    )

    result = allocator.solve(
        D1ContactAllocationRequest(
            state=state,
            desired_force_world_n=np.asarray((0.0, 0.0, 200.0)),
            desired_moment_world_nm=np.zeros(3),
            committed_torque_nm=committed,
        )
    )

    assert result.status is D1AllocationStatus.FALLBACK
    assert np.all(np.abs(result.command_torque_nm) <= JOINT_TORQUE_LIMIT + 1e-9)
    np.testing.assert_allclose(result.contact_force_world_n[:, 2], 5.0, atol=1e-8)
    assert result.achieved_wrench_world[2] == pytest.approx(20.0, abs=1e-8)
    assert result.max_constraint_violation <= 1e-9


def test_feasible_nonconverged_candidate_is_used_and_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, state = _ideal_four_contact_state()
    candidate = np.tile((0.0, 0.0, 50.0), 4)
    monkeypatch.setattr(
        "wheel_legged_control.d1.contact_allocation.minimize",
        lambda *args, **kwargs: SimpleNamespace(
            x=candidate,
            success=False,
            fun=0.0,
            nit=7,
            status=8,
            message="Positive directional derivative for linesearch",
        ),
    )
    allocator = D1ConstrainedContactAllocator(
        friction_coefficient=0.7,
        normal_force_limit_n=300.0,
        torque_limit_nm=JOINT_TORQUE_LIMIT,
    )

    result = allocator.solve(
        D1ContactAllocationRequest(
            state=state,
            desired_force_world_n=np.asarray((0.0, 0.0, 200.0)),
            desired_moment_world_nm=np.zeros(3),
            committed_torque_nm=np.zeros(16),
        )
    )

    assert result.status is D1AllocationStatus.FEASIBLE_NONCONVERGED
    assert result.status_reason == "slsqp_status_8"
    assert result.wrench_tracking_status is D1WrenchTrackingStatus.TRACKED
    np.testing.assert_allclose(result.contact_force_world_n[:, 2], 50.0)
    np.testing.assert_allclose(result.achieved_wrench_world, result.desired_wrench_world)


def test_constrained_vmc_holds_the_nominal_four_wheel_pose() -> None:
    plant = D1Plant()
    source = D1MujocoTruthStateSource(plant)
    controller = D1VMCController(
        plant,
        contact_allocator=D1ConstrainedContactAllocator(
            friction_coefficient=0.7,
            normal_force_limit_n=plant.nominal_total_mass_kg * 9.81 * 0.65,
            torque_limit_nm=JOINT_TORQUE_LIMIT,
        ),
    )
    state = source.reset()
    statuses = []

    for _ in range(100):
        plant.step(controller.compute(D1Command(), state))
        statuses.append(controller.last_breakdown.allocation_status)
        state = source.read()

    assert not plant.has_fallen()
    assert plant.wheel_ground_contacts == 4
    assert plant.undesired_ground_contacts == 0
    assert plant.base_position[2] > 0.40
    assert abs(plant.base_rpy[1]) < np.deg2rad(2.0)
    assert D1AllocationStatus.FALLBACK not in statuses
    assert controller.last_breakdown.allocation_constraint_violation <= 1e-7


def test_lqr_can_drive_through_the_constrained_contact_allocator() -> None:
    plant = D1Plant()
    source = D1MujocoTruthStateSource(plant)
    allocator = D1ConstrainedContactAllocator(
        friction_coefficient=0.7,
        normal_force_limit_n=plant.nominal_total_mass_kg * 9.81 * 0.65,
        torque_limit_nm=JOINT_TORQUE_LIMIT,
    )
    controller = D1LQRVMCController(plant, contact_allocator=allocator)
    state = source.reset()

    for _ in range(150):
        plant.step(controller.compute(D1Command(forward_velocity_mps=0.35), state))
        state = source.read()

    assert not plant.has_fallen()
    assert plant.base_position[0] > 0.15
    assert plant.base_velocity(local=False)[0][0] > 0.10
    assert controller.low_level.last_breakdown.allocation_status in {
        D1AllocationStatus.CONVERGED,
        D1AllocationStatus.FEASIBLE_NONCONVERGED,
    }


def test_constrained_lqr_completes_the_nominal_command_schedule() -> None:
    env = D1ResidualEnv(
        baseline="lqr",
        contact_allocation="constrained",
        randomize=False,
        episode_seconds=4.1,
    )
    observation, _ = env.reset(seed=21, options={"scenario": "nominal"})
    terminated = truncated = False

    while not (terminated or truncated):
        observation, _, terminated, truncated, _ = env.step(np.zeros(2))

    assert np.isfinite(observation).all()
    assert not terminated
    env.close()


def test_hierarchical_controller_identifies_the_selected_allocation_path() -> None:
    legacy_model = identify_sagittal_model(0.01, "legacy")
    constrained_model = identify_sagittal_model(0.01, "constrained")
    assert not np.allclose(legacy_model.b, constrained_model.b, atol=1e-7, rtol=1e-4)

    plant = D1Plant()
    allocator = D1ConstrainedContactAllocator(
        friction_coefficient=0.7,
        normal_force_limit_n=plant.nominal_total_mass_kg * 9.81 * 0.65,
        torque_limit_nm=JOINT_TORQUE_LIMIT,
    )
    controller = D1LQRVMCController(plant, contact_allocator=allocator)

    assert controller.linear_model is constrained_model


def test_hierarchical_cost_matches_the_selected_allocation_path() -> None:
    plant = D1Plant()
    legacy_allocator = D1LegacyContactAllocator(
        normal_force_limit_n=plant.nominal_total_mass_kg * 9.81 * 0.65,
        wheel_radius_m=plant.wheel_radius_m,
        torque_limit_nm=JOINT_TORQUE_LIMIT,
    )
    allocator = D1ConstrainedContactAllocator(
        friction_coefficient=0.7,
        normal_force_limit_n=plant.nominal_total_mass_kg * 9.81 * 0.65,
        torque_limit_nm=JOINT_TORQUE_LIMIT,
    )

    legacy_lqr = D1LQRVMCController(plant)
    explicit_legacy_lqr = D1LQRVMCController(
        plant,
        contact_allocator=legacy_allocator,
    )
    constrained_lqr = D1LQRVMCController(plant, contact_allocator=allocator)
    constrained_mpc = D1MPCVMCController(plant, contact_allocator=allocator)
    explicit_r = np.asarray(((4e-6,),))
    overridden = D1LQRVMCController(plant, r=explicit_r, contact_allocator=allocator)
    overridden_mpc = D1MPCVMCController(plant, r=explicit_r, contact_allocator=allocator)

    assert legacy_lqr.allocation_mode == "legacy"
    assert explicit_legacy_lqr.allocation_mode == "legacy"
    assert constrained_lqr.allocation_mode == "constrained"
    np.testing.assert_allclose(legacy_lqr.r, np.asarray(((1e-8,),)))
    np.testing.assert_array_equal(explicit_legacy_lqr.r, legacy_lqr.r)
    np.testing.assert_allclose(constrained_lqr.r, np.asarray(((1e-6,),)))
    np.testing.assert_allclose(constrained_mpc.r, constrained_lqr.r)
    np.testing.assert_array_equal(overridden.r, explicit_r)
    np.testing.assert_array_equal(overridden_mpc.r, explicit_r)
    assert np.max(np.abs(constrained_lqr.closed_loop_eigenvalues)) < 1.0


def test_env_updates_allocator_limits_with_actuator_randomization() -> None:
    env = D1ResidualEnv(
        contact_allocation="constrained",
        randomize=False,
    )

    env.reset(
        seed=21,
        options={
            "scenario": "randomized",
            "randomize": False,
            "actuator_strength_scale": 0.85,
        },
    )

    allocator = env.controller.low_level.contact_allocator
    assert allocator is not None
    np.testing.assert_allclose(allocator.torque_limit_nm, 0.85 * JOINT_TORQUE_LIMIT)
    env.close()


def test_legacy_allocator_reproduces_the_existing_vmc_torque() -> None:
    plant = D1Plant()
    state = D1MujocoTruthStateSource(plant).reset()
    command = D1Command(
        forward_velocity_mps=0.25,
        yaw_rate_rps=0.15,
        base_height_m=0.46,
        roll_rad=0.02,
        pitch_rad=-0.03,
    )
    original = D1VMCController(plant)
    adapted = D1VMCController(
        plant,
        contact_allocator=D1LegacyContactAllocator(
            normal_force_limit_n=plant.nominal_total_mass_kg * 9.81 * 0.65,
            wheel_radius_m=plant.wheel_radius_m,
            torque_limit_nm=JOINT_TORQUE_LIMIT,
        ),
    )

    expected = original.compute(
        command,
        state,
        longitudinal_force_n=70.0,
        vertical_force_offset_n=25.0,
    )
    actual = adapted.compute(
        command,
        state,
        longitudinal_force_n=70.0,
        vertical_force_offset_n=25.0,
    )

    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert adapted.last_breakdown.allocation_status is D1AllocationStatus.LEGACY
