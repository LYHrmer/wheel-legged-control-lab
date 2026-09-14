"""Heading-task curriculum with a fixed 16,384-transition difficulty schedule."""
from __future__ import annotations

from scripts.d1_course_curriculum import D1CourseCurriculumEnv

CURRICULUM_STEPS = 16384


class D1HeadingCurriculumEnv(D1CourseCurriculumEnv):
    """Use the unchanged reset-time curriculum to train the new 85-value task.

    ``training_budget_steps`` describes the learner's separate final budget.
    The inherited ``total_steps`` remains 16,384 even for a 65,536-step run;
    an active episode keeps its selected level until the next reset.
    """

    def __init__(self, *, total_steps, seed, episode_seconds=32.0, command_source=None):
        if isinstance(total_steps, bool) or not isinstance(total_steps, int):
            raise TypeError("total_steps must be a non-boolean integer")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self.training_budget_steps = total_steps
        super().__init__(condition="curriculum", total_steps=CURRICULUM_STEPS,
                         seed=seed, episode_seconds=episode_seconds,
                         command_source=command_source)

    def _build_env(self, terrain):
        from scripts.d1_heading_tracking_env import D1HeadingTrackingEnv

        return D1HeadingTrackingEnv(
            terrain=terrain, episode_seconds=self._episode_seconds,
            command_source=self._command_source,
        )
