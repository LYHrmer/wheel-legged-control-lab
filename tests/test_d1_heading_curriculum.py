"""Protect the new task's absolute curriculum clock and real 85-value output."""

import numpy as np
import pytest

from scripts.d1_course_curriculum import D1CourseCurriculumEnv
from scripts.d1_heading_curriculum import CURRICULUM_STEPS, D1HeadingCurriculumEnv
from scripts.d1_heading_tracking_env import D1HeadingTrackingEnv
from scripts.run_d1_course_curriculum import forward_command


@pytest.mark.parametrize("budget", [32, 256, 16384, 65536])
def test_curriculum_clock_is_independent_of_learner_budget(monkeypatch, budget):
    def initialize(self, **kwargs):
        self._condition = kwargs["condition"]
        self._total_steps = kwargs["total_steps"]

    monkeypatch.setattr(D1CourseCurriculumEnv, "__init__", initialize)
    env = D1HeadingCurriculumEnv(total_steps=budget, seed=49001)
    assert env.training_budget_steps == budget
    assert env.total_steps == CURRICULUM_STEPS == 16384
    for transitions, level in [(0, 0), (4095, 0), (4096, 1), (8192, 2),
                               (12288, 3), (16384, 3), (65536, 3)]:
        env.total_transitions = transitions
        assert env._select_level(0) == env._select_level(3) == level


@pytest.mark.parametrize("budget", [True, 1.5, "65536", 0, -1])
def test_invalid_training_budget_rejected_before_env_construction(budget):
    with pytest.raises((TypeError, ValueError)):
        D1HeadingCurriculumEnv(total_steps=budget, seed=49001)


def test_real_heading_env_keeps_level_until_reset_and_exposes_85_values():
    env = D1HeadingCurriculumEnv(total_steps=65536, seed=49001,
                                episode_seconds=0.04, command_source=forward_command)
    try:
        obs, _ = env.reset()
        assert isinstance(env.unwrapped, D1HeadingTrackingEnv)
        assert obs.shape == (85,) and obs.dtype == np.float32
        assert env.current_level == 0
        original = env.unwrapped
        # Scheduling fixture only: does not represent thousands of physical steps.
        env.total_transitions = 4095
        new_obs, reward, terminated, truncated, _ = env.step(np.zeros(8, np.float32))
        assert new_obs.shape == (85,) and np.isfinite(new_obs).all()
        assert np.isfinite(reward) and not terminated and not truncated
        assert env.total_transitions == 4096
        assert env.current_level == 0 and env.unwrapped is original
        env.reset()
        assert env.current_level == 1 and env.unwrapped is not original
        assert env.unwrapped.episode_metadata["task_schema"] == "d1-heading-reference-task-v1"
    finally:
        env.close()
