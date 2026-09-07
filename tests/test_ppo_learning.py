import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wheel_legged_control.ppo_learning import clipped_surrogate, generalized_advantage_estimate


def test_gae_matches_hand_calculation() -> None:
    advantages, returns = generalized_advantage_estimate(
        [1.0, 2.0, 3.0],
        [0.5, 0.6, 0.7],
        [0.6, 0.7, 0.8],
        [False, False, False],
        [False, False, True],
        gamma=0.9,
        gae_lambda=0.8,
    )
    # delta=[1.04, 2.03, 3.02]; reverse recurrence uses gamma*lambda=0.72.
    np.testing.assert_allclose(advantages, [4.067168, 4.2044, 3.02])
    np.testing.assert_allclose(returns, [4.567168, 4.8044, 3.72])


@pytest.mark.parametrize(
    ("terminated", "truncated", "expected"),
    [(True, False, 0.5), (False, True, 9.5), (False, False, 9.5), (True, True, 0.5)],
)
def test_bootstrap_at_terminal_timeout_and_rollout_tail(
    terminated: bool, truncated: bool, expected: float
) -> None:
    advantages, returns = generalized_advantage_estimate(
        [1.0], [0.5], [10.0], [terminated], [truncated], gamma=0.9
    )
    np.testing.assert_allclose(advantages, [expected])
    np.testing.assert_allclose(returns, [expected + 0.5])


@pytest.mark.parametrize("boundary", ["terminated", "truncated"])
def test_gae_does_not_leak_across_reset_episodes(boundary: str) -> None:
    masks = {"terminated": [False, False], "truncated": [False, False]}
    masks[boundary][0] = True
    advantages, _ = generalized_advantage_estimate(
        [1.0, 1000.0], [0.0, 0.0], [2.0, 0.0], **masks, gamma=0.9, gae_lambda=1.0
    )
    assert advantages[0] == pytest.approx(1.0 if boundary == "terminated" else 2.8)
    assert advantages[1] == pytest.approx(1000.0)


def test_vectorized_gae_matches_each_environment() -> None:
    rewards = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    values = rewards / 10.0
    next_values = values + 0.1
    terminated = np.array([[False, True], [False, False], [True, False]])
    truncated = np.array([[False, False], [True, False], [False, False]])
    batch = generalized_advantage_estimate(rewards, values, next_values, terminated, truncated)
    for env_index in range(2):
        single = generalized_advantage_estimate(
            rewards[:, env_index],
            values[:, env_index],
            next_values[:, env_index],
            terminated[:, env_index],
            truncated[:, env_index],
        )
        for actual, expected in zip(batch, single, strict=True):
            np.testing.assert_allclose(actual[:, env_index], expected)


def test_lambda_zero_is_one_step_td() -> None:
    advantages, _ = generalized_advantage_estimate(
        [1.0, 2.0],
        [0.5, 0.6],
        [0.6, 0.7],
        [False, False],
        [False, True],
        gamma=0.9,
        gae_lambda=0.0,
    )
    np.testing.assert_allclose(advantages, [1.04, 2.03])


def test_zero_discount_ignores_future_values_and_carry() -> None:
    advantages, returns = generalized_advantage_estimate(
        [1.0, 2.0],
        [0.5, 0.6],
        [100.0, 200.0],
        [False, False],
        [False, False],
        gamma=0.0,
    )
    np.testing.assert_allclose(advantages, [0.5, 1.4])
    np.testing.assert_allclose(returns, [1.0, 2.0])


def test_lambda_one_matches_discounted_return_for_completed_episode() -> None:
    _, returns = generalized_advantage_estimate(
        [1.0, 2.0, 3.0],
        [0.5, 0.6, 0.7],
        [0.6, 0.7, 100.0],
        [False, False, True],
        [False, False, False],
        gamma=0.9,
        gae_lambda=1.0,
    )
    np.testing.assert_allclose(returns, [5.23, 4.7, 3.0])


def test_gae_rejects_arithmetic_overflow() -> None:
    largest = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="non-finite"):
        generalized_advantage_estimate([largest], [0.0], [largest], [False], [False], gamma=1.0)


@pytest.mark.parametrize(
    ("name", "invalid"),
    [
        ("rewards", []),
        ("rewards", [np.nan]),
        ("values", [np.inf]),
        ("next_values", [np.nan]),
        ("values", [[0.0]]),
        ("next_values", [0.0, 1.0]),
        ("terminated", [1]),
        ("truncated", [np.nan]),
        ("terminated", [[False]]),
        ("gamma", -0.1),
        ("gamma", np.inf),
        ("gamma", True),
        ("gae_lambda", 1.1),
        ("gae_lambda", np.nan),
    ],
)
def test_gae_rejects_invalid_inputs(name: str, invalid: object) -> None:
    inputs = {
        "rewards": [1.0],
        "values": [0.0],
        "next_values": [0.0],
        "terminated": [False],
        "truncated": [False],
        "gamma": 0.99,
        "gae_lambda": 0.95,
    }
    inputs[name] = invalid
    with pytest.raises((TypeError, ValueError)):
        generalized_advantage_estimate(**inputs)


def test_ppo_clip_branches_for_both_advantage_signs() -> None:
    ratios = np.array([1.5, 0.5, 1.5, 0.5])
    result = clipped_surrogate(np.zeros(4), np.log(ratios), [2.0, 2.0, -2.0, -2.0])
    np.testing.assert_allclose(result.ratio, ratios)
    np.testing.assert_allclose(result.unclipped, [3.0, 1.0, -3.0, -1.0])
    np.testing.assert_allclose(result.clipped, [2.4, 1.6, -2.4, -1.6])
    np.testing.assert_allclose(result.minimum, [2.4, 1.0, -3.0, -1.6])
    assert result.loss == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("advantage", "ratio", "objective_derivative"),
    [(2.0, 1.5, 0.0), (2.0, 0.5, 1.0), (-2.0, 1.5, -3.0), (-2.0, 0.5, 0.0)],
)
def test_clipping_does_not_zero_every_outside_gradient(
    advantage: float, ratio: float, objective_derivative: float
) -> None:
    # Finite difference with respect to new joint-action log probability.
    epsilon = 1e-6
    plus = clipped_surrogate([0.0], [np.log(ratio) + epsilon], [advantage])
    minus = clipped_surrogate([0.0], [np.log(ratio) - epsilon], [advantage])
    derivative = float((plus.minimum[0] - minus.minimum[0]) / (2.0 * epsilon))
    assert derivative == pytest.approx(objective_derivative, abs=1e-8)


def test_ppo_accepts_continuous_log_densities_above_zero() -> None:
    result = clipped_surrogate([2.0], [2.0], [1.0])
    assert result.loss == -1.0


def test_ppo_vectorized_samples_are_not_normalized_or_mutated() -> None:
    old = np.zeros((2, 2))
    new = np.log([[1.5, 0.5], [1.5, 0.5]])
    advantages = np.array([[2.0, 2.0], [-2.0, -2.0]])
    originals = [array.copy() for array in (old, new, advantages)]
    result = clipped_surrogate(old, new, advantages)
    np.testing.assert_allclose(result.minimum, [[2.4, 1.0], [-3.0, -1.6]])
    assert result.loss == pytest.approx(0.3)
    for actual, original in zip((old, new, advantages), originals, strict=True):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize(
    ("old", "new", "advantages", "clip"),
    [
        ([], [], [], 0.2),
        ([0.0], [np.nan], [1.0], 0.2),
        ([np.inf], [0.0], [1.0], 0.2),
        ([0.0], [0.0], [np.nan], 0.2),
        ([0.0], [[0.0]], [1.0], 0.2),
        ([0.0], [0.0], [1.0, 2.0], 0.2),
        ([0.0], [0.0], [1.0], -0.1),
        ([0.0], [0.0], [1.0], 1.0),
        ([0.0], [0.0], [1.0], np.nan),
        ([0.0], [1000.0], [1.0], 0.2),
        ([0.0], [-1000.0], [1.0], 0.2),
    ],
)
def test_ppo_rejects_invalid_inputs(
    old: object, new: object, advantages: object, clip: float
) -> None:
    with pytest.raises(ValueError):
        clipped_surrogate(old, new, advantages, clip)


def test_walkthrough_is_read_only_by_default_and_refuses_overwrite(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository / "src"), str(repository / ".local-deps")]
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, str(repository / "examples" / "ppo_walkthrough.py")]
    plain = subprocess.run(
        command, cwd=tmp_path, env=environment, capture_output=True, text=True, check=False
    )
    assert plain.returncode == 0, plain.stderr
    assert "GAE(timeout)" in plain.stdout
    assert not list(tmp_path.iterdir())

    output = tmp_path / "tables"
    saved = subprocess.run(
        [*command, "--output", str(output)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert saved.returncode == 0, saved.stderr
    report_file = output / "walkthrough.json"
    original = report_file.read_bytes()
    report = json.loads(original)
    assert report["synthetic_only"] is True
    assert report["actor_loss"] == pytest.approx(0.3)
    assert report["equivalent_gamma_at_10ms"] == pytest.approx(np.sqrt(0.99))
    assert {path.name for path in output.iterdir()} == {
        "walkthrough.json",
        "trajectory.csv",
        "clipping.csv",
        "discount_timing.csv",
    }
    repeated = subprocess.run(
        [*command, "--output", str(output)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert repeated.returncode != 0
    assert "refusing to overwrite" in repeated.stderr
    assert report_file.read_bytes() == original
