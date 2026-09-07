import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PPO update lab needs the optional rl dependency")

from wheel_legged_control.ppo_learning import clipped_surrogate, generalized_advantage_estimate
from wheel_legged_control.ppo_update import (
    FrozenBatch,
    UpdateConfig,
    clipping_fixture,
    one_sgd_update,
    ppo_terms,
    synthetic_rollout,
)


def test_real_update_matches_numpy_and_changes_both_networks() -> None:
    actor, critic, batch, _ = synthetic_rollout()
    snapshot = {field.name: getattr(batch, field.name).clone() for field in fields(batch)}
    config = UpdateConfig()
    trace = one_sgd_update(actor, critic, batch, config)
    assert trace["optimizer_steps"] == 1
    np.testing.assert_array_equal(trace["before"]["ratio"], np.ones(8))
    for stage in ("before", "after"):
        values = trace[stage]
        expected = clipped_surrogate(batch.old_log_prob, values["log_prob"], batch.advantages)
        assert values["actor_loss"] == pytest.approx(expected.loss, abs=1e-14)
        np.testing.assert_allclose(values["minimum"], expected.minimum)
        mse = np.mean((np.asarray(values["value"]) - batch.returns.numpy()) ** 2)
        assert values["value_loss"] == pytest.approx(mse, abs=1e-14)
        assert values["total_loss"] == pytest.approx(values["actor_loss"] + 0.5 * mse)
    for group in ("actor", "critic"):
        assert trace["gradient_norms"][group] > 0
        assert any(
            row["delta"] != 0 for row in trace["parameters"] if row["parameter"].startswith(group)
        )
    assert len(trace["parameters"]) == 7
    for row in trace["parameters"]:
        assert row["after"] == pytest.approx(row["before"] - 0.01 * row["gradient"], abs=1e-14)
    for name, expected in snapshot.items():
        torch.testing.assert_close(getattr(batch, name), expected, rtol=0, atol=0)


def test_batch_owns_storage_and_targets_never_receive_gradients() -> None:
    actor, critic, source, _ = synthetic_rollout()
    original = {
        field.name: getattr(source, field.name).clone().requires_grad_() for field in fields(source)
    }
    batch = FrozenBatch(**original)
    for name, tensor in original.items():
        assert getattr(batch, name).data_ptr() != tensor.data_ptr()
        assert not getattr(batch, name).requires_grad
        # Even a caller explicitly turning on batch gradients must not open a target path.
        getattr(batch, name).requires_grad_()
    one_sgd_update(actor, critic, batch)
    for name, tensor in original.items():
        assert tensor.grad is None
        assert getattr(batch, name).grad is None
    with torch.no_grad():
        original["observations"].fill_(123)
    assert not torch.equal(batch.observations, original["observations"])


def test_actor_and_critic_gradient_paths_are_separate() -> None:
    actor, critic, batch, _ = synthetic_rollout()
    terms = ppo_terms(actor, critic, batch, UpdateConfig())
    assert all(
        gradient is None
        for gradient in torch.autograd.grad(
            terms["actor_loss"], tuple(critic.parameters()), allow_unused=True, retain_graph=True
        )
    )
    assert all(
        gradient is None
        for gradient in torch.autograd.grad(
            terms["value_loss"], tuple(actor.parameters()), allow_unused=True, retain_graph=True
        )
    )


def test_all_seven_parameter_gradients_match_finite_difference() -> None:
    actor, critic, batch, _ = synthetic_rollout()
    config = UpdateConfig(ent_coef=0.03)
    parameters = [*actor.parameters(), *critic.parameters()]
    gradients = torch.autograd.grad(
        ppo_terms(actor, critic, batch, config)["total_loss"], parameters
    )
    epsilon = 1e-6
    for parameter, gradient in zip(parameters, gradients, strict=True):
        for index in range(parameter.numel()):
            with torch.no_grad():
                flat = parameter.view(-1)
                original = float(flat[index])
                flat[index] = original + epsilon
                plus = float(ppo_terms(actor, critic, batch, config)["total_loss"])
                flat[index] = original - epsilon
                minus = float(ppo_terms(actor, critic, batch, config)["total_loss"])
                flat[index] = original
            assert float(gradient.flatten()[index]) == pytest.approx(
                (plus - minus) / (2 * epsilon), abs=2e-9
            )


def test_entropy_coefficient_has_correct_sign_and_does_not_affect_critic() -> None:
    actor, critic, batch, _ = synthetic_rollout()
    base = one_sgd_update(actor, critic, batch)
    actor, critic, batch, _ = synthetic_rollout()
    entropy = one_sgd_update(actor, critic, batch, UpdateConfig(ent_coef=0.1))
    for baseline, changed in zip(base["parameters"], entropy["parameters"], strict=True):
        difference = -0.1 if baseline["parameter"] == "actor.log_std" else 0
        assert changed["gradient"] - baseline["gradient"] == pytest.approx(difference, abs=1e-14)


def test_zero_value_coefficient_keeps_critic_fixed() -> None:
    actor, critic, batch, _ = synthetic_rollout()
    trace = one_sgd_update(actor, critic, batch, UpdateConfig(vf_coef=0))
    for row in trace["parameters"]:
        if row["parameter"].startswith("critic"):
            assert row["gradient"] == 0
            assert row["delta"] == 0
    assert trace["gradient_norms"]["actor"] > 0


def test_shared_actor_critic_parameters_are_rejected() -> None:
    actor, critic, batch, _ = synthetic_rollout()
    critic.bias = actor.bias
    with pytest.raises(ValueError, match="share parameters"):
        one_sgd_update(actor, critic, batch)


def test_rollout_is_on_policy_and_gae_uses_final_observation() -> None:
    actor, critic, batch, rollout = synthetic_rollout()
    with torch.no_grad():
        expected_log_prob = actor(batch.observations).log_prob(batch.actions).sum(-1)
        expected_next = critic(torch.tensor(rollout["next_observations"], dtype=torch.float64))
    torch.testing.assert_close(batch.old_log_prob, expected_log_prob, rtol=0, atol=0)
    np.testing.assert_allclose(rollout["next_values"], expected_next)
    assert rollout["truncated"] == [False] * 7 + [True]
    assert not any(rollout["terminated"])
    advantage, returns = generalized_advantage_estimate(
        rollout["rewards"],
        rollout["values"],
        rollout["next_values"],
        rollout["terminated"],
        rollout["truncated"],
        gamma=0.9,
        gae_lambda=0.8,
    )
    np.testing.assert_allclose(batch.advantages, advantage)
    np.testing.assert_allclose(batch.returns, returns)
    assert advantage[-1] == pytest.approx(
        rollout["rewards"][-1] + 0.9 * rollout["next_values"][-1] - rollout["values"][-1]
    )
    next_v = np.asarray(rollout["next_observations"])[:, 0]
    action = batch.actions[:, 0].numpy()
    np.testing.assert_allclose(next_v, 0.85 * batch.observations[:, 0].numpy() + 0.25 * action)
    np.testing.assert_allclose(rollout["rewards"], 1 - (next_v - 0.8) ** 2 - 0.05 * action**2)


def test_clipping_fixture_exercises_both_signs_and_gradients() -> None:
    fixture = clipping_fixture()
    np.testing.assert_allclose([row["minimum"] for row in fixture], [2.4, 1, -3, -1.6])
    np.testing.assert_allclose(
        [row["objective_derivative_wrt_log_prob"] for row in fixture], [0, 1, -3, 0]
    )
    actor, critic, original, _ = synthetic_rollout()
    batch = FrozenBatch(
        original.observations[:4],
        original.actions[:4],
        original.old_log_prob[:4]
        - torch.log(torch.tensor([1.5, 0.5, 1.5, 0.5], dtype=torch.float64)),
        [2, 2, -2, -2],
        original.returns[:4],
    )
    terms = ppo_terms(actor, critic, batch, UpdateConfig())
    assert float(terms["actor_loss"].detach()) == pytest.approx(0.3)
    np.testing.assert_allclose(terms["minimum"].detach(), [2.4, 1, -3, -1.6])


def test_joint_log_density_sums_action_dimensions() -> None:
    class TwoActionActor(torch.nn.Module):
        def forward(self, observations):
            return torch.distributions.Normal(observations, torch.ones_like(observations))

    _, critic, source, _ = synthetic_rollout()
    actor = TwoActionActor()
    actions = torch.cat((source.actions, -source.actions), dim=1)
    expected = actor(source.observations).log_prob(actions).sum(-1)
    batch = FrozenBatch(source.observations, actions, expected, source.advantages, source.returns)
    terms = ppo_terms(actor, critic, batch, UpdateConfig())
    torch.testing.assert_close(terms["log_prob"], expected)
    torch.testing.assert_close(terms["ratio"], torch.ones(8, dtype=torch.float64))


def test_repeated_seed_is_identical_and_does_not_change_global_rng() -> None:
    rng_before = torch.random.get_rng_state().clone()
    actor, critic, batch, _ = synthetic_rollout(17)
    first = one_sgd_update(actor, critic, batch)
    actor, critic, second_batch, _ = synthetic_rollout(17)
    second = one_sgd_update(actor, critic, second_batch)
    assert first == second
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
    _, _, different, _ = synthetic_rollout(18)
    assert not torch.equal(batch.actions, different.actions)


@pytest.mark.parametrize("name", ["learning_rate", "clip_range", "vf_coef", "ent_coef"])
@pytest.mark.parametrize("value", [-0.1, np.nan, np.inf, True])
def test_config_rejects_invalid_values(name, value) -> None:
    with pytest.raises((ValueError, TypeError)):
        UpdateConfig(**{name: value})


@pytest.mark.parametrize("kwargs", [{"learning_rate": 0}, {"clip_range": 1}])
def test_config_rejects_endpoint_values(kwargs) -> None:
    with pytest.raises(ValueError):
        UpdateConfig(**kwargs)


@pytest.mark.parametrize("seed", [-1, True, 0.5, 2**63])
def test_rollout_rejects_invalid_seeds(seed) -> None:
    with pytest.raises(ValueError):
        synthetic_rollout(seed)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("observations", []),
        ("observations", [[np.nan, 1.0]]),
        ("actions", [1.0] * 8),
        ("actions", [[np.inf]] * 8),
        ("old_log_prob", [0.0] * 7),
        ("advantages", [[1.0]] * 8),
        ("returns", [1j] * 8),
        ("returns", [True] * 8),
    ],
)
def test_batch_rejects_invalid_shapes_and_values(name, value) -> None:
    _, _, source, _ = synthetic_rollout()
    inputs = {field.name: getattr(source, field.name) for field in fields(source)}
    inputs[name] = value
    with pytest.raises(ValueError):
        FrozenBatch(**inputs)


def test_loss_rejects_broadcasting_and_ratio_underflow() -> None:
    actor, critic, source, _ = synthetic_rollout()
    bad_actions = FrozenBatch(
        source.observations,
        torch.cat((source.actions, source.actions), dim=1),
        source.old_log_prob,
        source.advantages,
        source.returns,
    )
    with pytest.raises(ValueError, match="action shape"):
        ppo_terms(actor, critic, bad_actions, UpdateConfig())
    underflow = FrozenBatch(
        source.observations,
        source.actions,
        source.old_log_prob + 1000,
        source.advantages,
        source.returns,
    )
    with pytest.raises(ValueError, match="finite float64"):
        ppo_terms(actor, critic, underflow, UpdateConfig())


def test_invalid_post_update_policy_restores_parameters() -> None:
    actor, critic, batch, _ = synthetic_rollout()
    before = [p.detach().clone() for p in [*actor.parameters(), *critic.parameters()]]
    with pytest.raises(ValueError, match="restored"):
        one_sgd_update(actor, critic, batch, UpdateConfig(learning_rate=1e6))
    for parameter, expected in zip(
        [*actor.parameters(), *critic.parameters()], before, strict=True
    ):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


def test_update_rejects_non_float64_parameters() -> None:
    actor, critic, batch, _ = synthetic_rollout()
    with pytest.raises(ValueError, match="CPU float64"):
        one_sgd_update(actor.float(), critic, batch)


def test_cli_is_read_only_by_default_and_exports_recomputable_trace(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository / "src"), str(repository / ".local-deps")]
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, str(repository / "examples" / "ppo_update_walkthrough.py")]

    def run(*args):
        return subprocess.run(
            [*command, *args],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    plain = run()
    assert plain.returncode == 0, plain.stderr
    assert "exactly one SGD optimizer.step()" in plain.stdout
    assert not list(tmp_path.iterdir())
    output = tmp_path / "trace"
    saved = run("--output", str(output))
    assert saved.returncode == 0, saved.stderr
    original = (output / "update.json").read_bytes()
    report = json.loads(original)
    assert report["synthetic_only"] is True
    assert report["update"]["optimizer_steps"] == 1
    with (output / "samples.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for stage in ("before", "after"):
        terms = clipped_surrogate(
            [float(row["old_log_prob"]) for row in rows],
            [float(row[f"log_prob_{stage}"]) for row in rows],
            [float(row["advantage"]) for row in rows],
        )
        assert terms.loss == pytest.approx(report["update"][stage]["actor_loss"], abs=1e-14)
    with (output / "parameters.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            assert float(row["after"]) == pytest.approx(
                float(row["before"]) - 0.01 * float(row["gradient"]), abs=1e-14
            )
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["files"]) == 4
    for filename, digest in manifest["files"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == digest
    for filename, digest in report["provenance"]["source_sha256"].items():
        assert hashlib.sha256((repository / filename).read_bytes()).hexdigest() == digest
    repeated = run("--output", str(output))
    assert repeated.returncode != 0
    assert "refusing to overwrite" in repeated.stderr
    assert (output / "update.json").read_bytes() == original
    invalid = run("--learning-rate", "nan", "--output", str(tmp_path / "invalid"))
    assert invalid.returncode != 0
    assert not (tmp_path / "invalid").exists()
