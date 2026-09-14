"""Check real terrain scaling, episode scheduling and unchanged physical steps."""

import json
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from scripts import d1_course_curriculum as course
from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.locomotion_commands import D1CommandSchedule, D1CommandSegment
from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv
from wheel_legged_control.d1.locomotion_terrain import (
    D1LocomotionTerrainConfig,
    locomotion_terrain_configs,
)


class CountedEnv(gym.Env):
    """Small deterministic lifecycle fixture; real MuJoCo is checked below."""

    def __init__(self, *, terrain, episode_seconds, **kwargs):
        self.terrain = terrain
        self.observation_space = gym.spaces.Box(-5, 5, (82,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (8,), np.float32)
        self.plant = SimpleNamespace(data=SimpleNamespace(qpos=np.array([-3.8, 0, 0.455])))
        self.last_transition = None
        self.limit = round(episode_seconds / 0.01)
        self.early_failure = False
        self.raise_step = False
        self.close_calls = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.reset_seed, self.reset_options = seed, options
        self.last_transition = None
        return np.zeros(82, np.float32), {"fixture_seed": seed}

    def step(self, action):
        if self.raise_step:
            raise RuntimeError("injected physical step failure")
        self.steps += 1
        x = -3.8 - 0.1 * self.steps
        self.last_transition = SimpleNamespace(
            truth=SimpleNamespace(
                base_position=np.array([x, 0, 0.455]),
                base_rpy=np.array([0.02 * self.steps, -0.01, 0]),
            )
        )
        terminated = self.early_failure
        truncated = self.steps >= self.limit and not terminated
        return (
            np.full(82, self.steps, np.float32),
            self.steps * 0.25,
            terminated,
            truncated,
            {
                "metrics": {"clearance_m": 0.455 - self.steps * 0.001},
                "terrain_exposure": {"nonflat_steps": self.steps, "path_m": self.steps * 0.1},
                "terminal_reason": "fall_or_body_contact"
                if terminated
                else "time_limit"
                if truncated
                else None,
                "unchanged": "fixture marker",
            },
        )

    def close(self):
        self.close_calls += 1


@pytest.fixture
def fake_base(monkeypatch):
    monkeypatch.setattr(course, "D1LocomotionEnv", CountedEnv)


def wrapper(condition="curriculum", *, total_steps=8, seed=19, episode_seconds=0.04, **kwargs):
    return course.D1CourseCurriculumEnv(
        condition=condition,
        total_steps=total_steps,
        seed=seed,
        episode_seconds=episode_seconds,
        **kwargs,
    )


@pytest.mark.parametrize("level", range(4))
@pytest.mark.parametrize("index", range(4))
def test_level_zero_is_really_flat_and_every_training_index_scales(level, index):
    actual = course.scaled_training_terrain(level, index)
    if level == 0:
        assert actual == D1LocomotionTerrainConfig()
        return
    original = locomotion_terrain_configs("train")[index]
    changed = {"slope_deg", "cross_slope_deg", "ripple_amplitude_m", "step_height_m"}
    for name, value in asdict(original).items():
        if name in changed:
            assert getattr(actual, name) == pytest.approx(value * level / 3)
        else:
            assert getattr(actual, name) == value


@pytest.mark.parametrize("bad", [True, np.bool_(True), 1.5, "1", -1, 4])
def test_bad_level_or_index_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        course.scaled_training_terrain(bad, 0)
    with pytest.raises((TypeError, ValueError)):
        course.scaled_training_terrain(1, bad)


def test_budget_boundary_waits_for_reset_and_actual_early_failures_count(fake_base):
    env = wrapper()
    try:
        env.reset()
        old = env.unwrapped
        assert env.current_level == 0
        for _ in range(3):
            env.step(np.zeros(8))
        assert env.total_transitions == 3
        assert env.current_level == 0 and env.unwrapped is old
        env.reset()
        assert env.current_level == 1 and env.unwrapped is not old
        assert old.close_calls == 1
        assert env.episode_records[0]["terminal_reason"] == "manual_reset"
        env.unwrapped.early_failure = True
        _, _, terminated, truncated, _ = env.step(np.zeros(8))
        assert terminated and not truncated
        assert env.total_transitions == 4
        env.reset()
        assert env.current_level == 2
        assert env.episode_records[-1]["transitions"] == 1
    finally:
        env.close()


def test_seed_and_geometry_streams_pair_across_conditions(fake_base):
    sequences = {}
    for condition in course.CONDITIONS:
        env = wrapper(condition, episode_seconds=0.01, total_steps=8)
        try:
            records = []
            for _ in range(8):
                env.reset()
                records.append((env.current_terrain_index, env.current_episode_seed))
                env.step(np.zeros(8))
            sequences[condition] = records
        finally:
            env.close()
    assert sequences["flat"] == sequences["mixed"] == sequences["curriculum"]
    repeat = wrapper("mixed", episode_seconds=0.01)
    try:
        observed = []
        for _ in range(8):
            repeat.reset()
            observed.append((repeat.current_terrain_index, repeat.current_episode_seed))
            repeat.step(np.zeros(8))
        assert observed == sequences["mixed"]
    finally:
        repeat.close()


def snapshot(env):
    return (
        env.unwrapped,
        env.current_level,
        env.current_terrain_index,
        env.current_episode_seed,
        env.total_transitions,
        deepcopy(env.episode_records),
        deepcopy(env._level_rng.bit_generator.state),
        deepcopy(env._terrain_rng.bit_generator.state),
        deepcopy(env._episode_seed_rng.bit_generator.state),
    )


def schedule(seconds):
    return D1CommandSchedule((D1CommandSegment(seconds, D1MotionCommand(), 0),))


def test_invalid_reset_preserves_active_episode_rng_and_physics(fake_base):
    env = wrapper(command_source=lambda _: D1MotionCommand())
    try:
        env.reset()
        env.step(np.zeros(8))
        previous = snapshot(env)
        for kwargs in (
            {"seed": 4},
            {"options": {"wrong": 1}},
            {"options": []},
            {"options": {"schedule": "invalid"}},
            {"options": {"schedule": schedule(0.05)}},
            {"options": {"schedule": schedule(0.04)}},
        ):
            with pytest.raises((TypeError, ValueError)):
                env.reset(**kwargs)
            assert snapshot(env) == previous
        env.step(np.zeros(8))
        assert env.total_transitions == 2
    finally:
        env.close()


def test_valid_schedule_passes_through_and_initial_seed_can_be_set(fake_base):
    env = wrapper()
    try:
        explicit = schedule(0.04)
        env.reset(seed=21, options={"schedule": explicit})
        assert env.seed_value == 21
        assert env.unwrapped.reset_options["schedule"] is explicit
    finally:
        env.close()


def test_records_conserve_steps_keep_signed_progress_and_do_not_call_it_success(fake_base):
    env = wrapper(episode_seconds=0.02)
    env.reset()
    for _ in range(2):
        _, reward, _, _, info = env.step(np.zeros(8))
        assert info["unchanged"] == "fixture marker"
    complete = env.episode_records[0]
    assert complete["completed"] and complete["terminal_reason"] == "time_limit"
    assert complete["reward_sum"] == 0.75 and reward == 0.5
    assert complete["forward_progress_m"] == pytest.approx(-0.2)
    assert complete["minimum_clearance_m"] == pytest.approx(0.453)
    assert complete["maximum_abs_roll_pitch_rad"] == 0.04
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(8))
    env.reset()
    env.reset()  # Zero-transition reset is not fabricated exposure.
    assert len(env.episode_records) == 1
    env.step(np.zeros(8))
    base = env.unwrapped
    env.close()
    env.close()
    assert base.close_calls == 1
    assert len(env.episode_records) == 2
    assert env.episode_records[-1]["terminal_reason"] == "budget_cut"
    assert not env.episode_records[-1]["completed"]
    assert sum(r["transitions"] for r in env.episode_records) == env.total_transitions == 3
    assert all("success" not in r for r in env.episode_records)
    json.dumps(env.episode_records, allow_nan=False)
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_invalid_actions_are_rejected_before_step_and_physics_errors_require_reset(fake_base):
    env = wrapper()
    try:
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros(8))
        env.reset()
        for action in (np.zeros(2), np.full(8, np.nan)):
            with pytest.raises(ValueError):
                env.step(action)
            assert env.total_transitions == 0 and env.unwrapped.steps == 0
        env.unwrapped.raise_step = True
        with pytest.raises(RuntimeError, match="injected"):
            env.step(np.zeros(8))
        assert env.total_transitions == 0
        env.unwrapped.raise_step = False
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros(8))
        env.reset()
        env.step(np.zeros(8))
    finally:
        env.close()


@pytest.mark.parametrize("seconds", [True, np.bool_(True), "1", np.nan, np.inf, -1, 0.015])
def test_invalid_episode_duration_is_rejected_before_constructing_env(seconds, fake_base):
    with pytest.raises((TypeError, ValueError)):
        wrapper(episode_seconds=seconds)


def test_real_mujoco_wrapper_matches_direct_environment_step_for_step():
    env = wrapper("mixed", episode_seconds=0.04, total_steps=8)
    direct = None
    try:
        observation, _ = env.reset()
        direct = D1LocomotionEnv(
            baseline="wheel_leg",
            action_mode="independent8",
            episode_seconds=0.04,
            terrain=env.terrain_config,
        )
        expected, _ = direct.reset(seed=env.current_episode_seed)
        np.testing.assert_array_equal(observation, expected)
        assert observation.shape == (82,) and env.action_space.shape == (8,)
        actions = np.random.default_rng(7).uniform(-0.1, 0.1, (4, 8))
        for action in actions:
            actual = env.step(action)
            wanted = direct.step(action)
            np.testing.assert_array_equal(actual[0], wanted[0])
            assert actual[1:4] == wanted[1:4]
            assert actual[4]["reward_terms"] == wanted[4]["reward_terms"]
            assert actual[4]["terrain_exposure"] == wanted[4]["terrain_exposure"]
            assert np.isfinite(actual[0]).all() and np.isfinite(actual[1])
        assert env.total_transitions == 4
        assert env.episode_records[0]["transitions"] == 4
        json.dumps(env.episode_records, allow_nan=False)
    finally:
        env.close()
        if direct is not None:
            direct.close()
