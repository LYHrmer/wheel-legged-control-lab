import numpy as np

from wheel_legged_control.rewards import calculate_reward


def test_perfect_tracking_has_unit_reward() -> None:
    reward = calculate_reward(np.zeros(6), 0.0, 0.0, np.zeros(2))
    assert reward.total == 1.0
    assert reward.as_dict()["total"] == reward.total


def test_residual_and_termination_penalties_are_explicit() -> None:
    nominal = calculate_reward(np.zeros(6), 0.0, 0.0, np.ones(2))
    fallen = calculate_reward(np.zeros(6), 0.0, 0.0, np.ones(2), terminated=True)
    assert nominal.residual_penalty == -0.08
    assert fallen.termination_penalty == -10.0
    assert fallen.total == nominal.total - 10.0
