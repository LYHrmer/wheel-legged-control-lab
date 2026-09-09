from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from wheel_legged_control.d1.control_context import (
    D1AppliedAction,
    D1ControlContext,
    D1ControllerMemory,
    D1ControlProposal,
    D1ForceBaseline,
    D1JointTargetBaseline,
)
from wheel_legged_control.d1.state_estimation import D1StateEstimate


def state_at(tick=0, measurement_time_s=None):
    time_s = tick * 0.01
    return D1StateEstimate(
        sequence=tick,
        control_time_s=time_s,
        measurement_time_s=time_s if measurement_time_s is None else measurement_time_s,
        base_position=np.asarray((0, 0, 0.455)),
        base_rotation=np.eye(3),
        base_linear_velocity_body=np.zeros(3),
        base_angular_velocity_body=np.zeros(3),
        base_linear_velocity_world=np.zeros(3),
        base_angular_velocity_world=np.zeros(3),
        joint_position=np.zeros(16),
        joint_velocity=np.zeros(16),
        foot_position=np.zeros((4, 3)),
        foot_jacobian=np.zeros((4, 3, 4)),
        wheel_contact=np.zeros(4, dtype=bool),
        wheel_contact_point=np.zeros((4, 3)),
        wheel_contact_normal=np.zeros((4, 3)),
        wheel_contact_jacobian=np.zeros((4, 3, 4)),
        undesired_ground_contacts=0,
    )


@pytest.mark.parametrize("size", (1, 2, 8, 12))
def test_context_keeps_current_proposal_and_previous_execution_distinct(size):
    state = state_at(2, measurement_time_s=0.0)
    baseline = D1ForceBaseline(12.0, 400.0, 3.0)
    proposal = D1ControlProposal(2, 0.02, baseline, D1ControllerMemory(0.1, 0.2, 0.3))
    previous = D1AppliedAction(1, 0.01, 0.02, np.full(size, 0.25))
    context = D1ControlContext(state, proposal, f"test-action-{size}", size, previous)
    assert context.state is state  # Reuse immutable publication, no duplicate estimator state.
    assert context.proposal.baseline is baseline
    assert context.state.age_s == 0.02
    assert context.proposal.control_time_s == previous.end_time_s
    np.testing.assert_array_equal(context.previous_normalized_action, np.full(size, 0.25))
    with pytest.raises(ValueError):
        context.previous_normalized_action.setflags(write=True)


def test_force_and_joint_target_baselines_do_not_fabricate_equivalent_fields():
    targets = np.arange(16, dtype=float)
    baseline = D1JointTargetBaseline(targets, np.zeros(4))
    targets[:] = 0
    np.testing.assert_array_equal(baseline.nominal_joint_target_rad, np.arange(16))
    assert not hasattr(baseline, "longitudinal_force_n")
    with pytest.raises(ValueError):
        baseline.nominal_joint_target_rad.setflags(write=True)
    context = D1ControlContext(state_at(), D1ControlProposal(0, 0, baseline), "joint8", 8)
    assert context.proposal.memory.distance_m is None
    np.testing.assert_array_equal(context.previous_normalized_action, np.zeros(8))
    with pytest.raises(FrozenInstanceError):
        context.action_size = 2


def test_stale_proposals_cannot_be_relabelled_as_current_context():
    proposal = D1ControlProposal(0, 0.0, D1ForceBaseline(1, 2))
    previous = D1AppliedAction(0, 0, 0.01, np.zeros(2))
    with pytest.raises(ValueError, match="current state"):
        D1ControlContext(state_at(1), proposal, "force2", 2, previous)
    with pytest.raises(ValueError, match="current state"):
        D1ControlContext(state_at(1), replace(proposal, tick=1), "force2", 2, previous)


def test_action_receipt_rejects_missing_stale_or_wrong_shape_history():
    proposal = D1ControlProposal(2, 0.02, D1ForceBaseline(1, 2))
    with pytest.raises(ValueError, match="requires"):
        D1ControlContext(state_at(2), proposal, "force2", 2)
    with pytest.raises(ValueError, match="end at"):
        D1ControlContext(
            state_at(2), proposal, "force2", 2, D1AppliedAction(0, 0, 0.01, np.zeros(2))
        )
    with pytest.raises(ValueError, match="size"):
        D1ControlContext(
            state_at(2), proposal, "force2", 2, D1AppliedAction(1, 0.01, 0.02, np.zeros(8))
        )
    with pytest.raises(ValueError, match="reset"):
        D1ControlContext(
            state_at(),
            replace(proposal, tick=0, control_time_s=0),
            "force2",
            2,
            D1AppliedAction(0, 0, 0.01, np.zeros(2)),
        )


@pytest.mark.parametrize("action", ([np.nan], [np.inf], [1.01], [], [[0, 0]]))
def test_action_receipt_requires_actual_finite_clipped_vector(action):
    with pytest.raises(ValueError):
        D1AppliedAction(0, 0, 0.01, action)


def test_controller_memory_absence_is_not_an_invented_zero_integral():
    empty = D1ControllerMemory()
    assert empty.distance_m is None and empty.yaw_integral_nm is None
    zero = D1ControllerMemory(distance_m=0.0, distance_reference_m=0.0, yaw_integral_nm=0.0)
    assert zero != empty
    with pytest.raises(ValueError, match="together"):
        D1ControllerMemory(distance_m=0.0)
    with pytest.raises(ValueError, match="finite"):
        D1ControllerMemory(distance_m=np.nan, distance_reference_m=0.0)
